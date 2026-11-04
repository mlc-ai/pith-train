"""
The reference the SM90 kernel of fp8_quant_act is bit-identical to.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8_quant_act_sm100"]


@triton.jit
def fp8cast_rowwise_blockwise_transpose_kernel(x_ptr, out_tok_ptr, scale_tok_ptr, out_blk_t_ptr, scale_blk_t_ptr, M, K, stride_x_row, stride_x_col, stride_out_tok_row, stride_scale_tok_row, stride_out_blk_t_row, scale_blk_t_cols, NATIVE_E8M0: tl.constexpr):
    BLOCK: tl.constexpr = 128

    pid_row, pid_col = tl.program_id(0), tl.program_id(1)
    row_start, col_start = pid_row * BLOCK, pid_col * BLOCK
    row_offs, col_offs = row_start + tl.arange(0, BLOCK), col_start + tl.arange(0, BLOCK)

    # amax runs on the input dtype; only the multiply is widened to fp32
    x_ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    tile_mask = (row_offs[:, None] < M) & (col_offs[None, :] < K)
    x = tl.load(x_ptrs, mask=tile_mask, other=0.0)
    x_abs = tl.abs(x)

    # the tile is one 1x128 block wide, so a row's amax is its block's
    row_amax = tl.max(x_abs, axis=1)
    tok_scale, tok_rcp = compute_fp8_scale(row_amax, NATIVE_E8M0)
    x_f32 = x.to(tl.float32)
    tok_fp8 = (x_f32 * tok_rcp[:, None]).to(tl.float8e4nv)

    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=tile_mask)
    tok_scale_ptrs = scale_tok_ptr + row_offs * stride_scale_tok_row + pid_col
    tl.store(tok_scale_ptrs, tok_scale, mask=row_offs < M)

    # the 128x128 amax from the row amaxes, without a second pass over the tile
    blk_amax = tl.max(row_amax, axis=0)
    blk_scale, blk_rcp = compute_fp8_scale(blk_amax, NATIVE_E8M0)
    blk_fp8_t = tl.trans((x_f32 * blk_rcp).to(tl.float8e4nv))

    out_t_row_offs, out_t_col_offs = col_start + tl.arange(0, BLOCK), row_start + tl.arange(0, BLOCK)
    out_blk_t_ptrs = out_blk_t_ptr + out_t_row_offs[:, None] * stride_out_blk_t_row + out_t_col_offs[None, :]
    out_t_mask = (out_t_row_offs[:, None] < K) & (out_t_col_offs[None, :] < M)
    tl.store(out_blk_t_ptrs, blk_fp8_t, mask=out_t_mask)

    tl.store(scale_blk_t_ptr + pid_col * scale_blk_t_cols + pid_row, blk_scale)


def fp8_quant_act_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    BLOCK = 128

    scale_tok_cols, scale_blk_t_rows, scale_blk_t_cols = (K + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK, (M + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_blk_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_blk_t = torch.empty((scale_blk_t_rows, scale_blk_t_cols), dtype=torch.float32, device=x.device)

    grid = ((M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)

    fp8cast_rowwise_blockwise_transpose_kernel[grid](x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_blk_t_ptr=out_blk_t, scale_blk_t_ptr=scale_blk_t, M=M, K=K, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_tok_row=K, stride_scale_tok_row=scale_tok_cols, stride_out_blk_t_row=M, scale_blk_t_cols=scale_blk_t_cols, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2)

    return out_tok, scale_tok, out_blk_t, scale_blk_t
