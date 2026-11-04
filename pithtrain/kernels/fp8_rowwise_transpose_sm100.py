"""
The reference the SM90 kernel of fp8_quant_grad is bit-identical to.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8_quant_grad_sm100"]


@triton.jit
def fp8cast_rowwise_transpose_kernel(x_ptr, out_tok_ptr, scale_tok_ptr, out_t_ptr, scale_t_ptr, M, N, stride_x_row, stride_x_col, stride_out_tok_row, scale_tok_cols, stride_out_t_row, stride_scale_t_row, BLOCK_N: tl.constexpr, NATIVE_E8M0: tl.constexpr):
    BLOCK_ROWS: tl.constexpr = 128
    BLOCKS_PER_ROW: tl.constexpr = BLOCK_N // BLOCK_ROWS

    pid_row, pid_col = tl.program_id(0), tl.program_id(1)
    row_start, col_start = pid_row * BLOCK_ROWS, pid_col * BLOCK_N
    row_offs, col_offs = row_start + tl.arange(0, BLOCK_ROWS), col_start + tl.arange(0, BLOCK_N)

    # amax runs on the input dtype; only the multiply is widened to fp32
    x_ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    tile_mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    x = tl.load(x_ptrs, mask=tile_mask, other=0.0)
    x_abs = tl.abs(x)
    # one row of the reshaped tile per 1x128 block, so a reduction along it is the block amax
    block_amax = tl.max(tl.reshape(x_abs, BLOCK_ROWS * BLOCKS_PER_ROW, BLOCK_ROWS), axis=1)
    tok_scale, tok_rcp = compute_fp8_scale(block_amax, NATIVE_E8M0)
    # widened to fp32 only after the amax, which keeps the original instruction order (and speed)
    x_f32 = x.to(tl.float32)
    x_blocks = tl.reshape(x_f32, BLOCK_ROWS * BLOCKS_PER_ROW, BLOCK_ROWS)
    tok_fp8 = tl.reshape((x_blocks * tok_rcp[:, None]).to(tl.float8e4nv), BLOCK_ROWS, BLOCK_N)

    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=tile_mask)

    tok_scale_col_offs = pid_col * BLOCKS_PER_ROW + tl.arange(0, BLOCKS_PER_ROW)
    tok_scale_ptrs = scale_tok_ptr + row_offs[:, None] * scale_tok_cols + tok_scale_col_offs[None, :]
    tok_scale_mask = (row_offs[:, None] < M) & (tok_scale_col_offs[None, :] < scale_tok_cols)
    tl.store(tok_scale_ptrs, tl.reshape(tok_scale, BLOCK_ROWS, BLOCKS_PER_ROW), mask=tok_scale_mask)

    # a column of x is a row of x.T, so its 128x1 block of the tile is a 1x128 block of x.T
    amax_t = tl.max(x_abs, axis=0)
    scale_t, rcp_t = compute_fp8_scale(amax_t, NATIVE_E8M0)
    fp8_t = tl.trans((x_f32 * rcp_t[None, :]).to(tl.float8e4nv))

    out_t_row_offs, out_t_col_offs = col_start + tl.arange(0, BLOCK_N), row_start + tl.arange(0, BLOCK_ROWS)
    out_t_ptrs = out_t_ptr + out_t_row_offs[:, None] * stride_out_t_row + out_t_col_offs[None, :]
    out_t_mask = (out_t_row_offs[:, None] < N) & (out_t_col_offs[None, :] < M)
    tl.store(out_t_ptrs, fp8_t, mask=out_t_mask)

    scale_t_ptrs = scale_t_ptr + col_offs * stride_scale_t_row + pid_row
    tl.store(scale_t_ptrs, scale_t, mask=col_offs < N)


def fp8_quant_grad_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    BLOCK = 128

    scale_tok_cols, scale_t_cols = (N + BLOCK - 1) // BLOCK, (M + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((N, scale_t_cols), dtype=torch.float32, device=x.device)

    BLOCK_N = 128
    grid = ((M + BLOCK - 1) // BLOCK, (N + BLOCK_N - 1) // BLOCK_N)

    fp8cast_rowwise_transpose_kernel[grid](x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_t_ptr=out_t, scale_t_ptr=scale_t, M=M, N=N, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_tok_row=N, scale_tok_cols=scale_tok_cols, stride_out_t_row=M, stride_scale_t_row=scale_t_cols, BLOCK_N=BLOCK_N, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2)

    return out_tok, scale_tok, out_t, scale_t
