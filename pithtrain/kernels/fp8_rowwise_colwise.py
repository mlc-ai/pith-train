"""
fp8_rowwise_kmajor_sm100 also launches fp8cast_rowwise_colwise_kernel, with WRITE_KMAJOR for the
K-major colwise output.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8cast_rowwise_colwise", "fp8cast_rowwise_colwise_kernel", "quantize_row_blocks"]


@triton.jit
def quantize_row_blocks(x_abs, x_f32, BLOCK_ROWS: tl.constexpr, BLOCK_N: tl.constexpr, NATIVE_E8M0: tl.constexpr):
    """
    1x128 quantization of a (BLOCK_ROWS, BLOCK_N) tile: fp8 (BLOCK_ROWS, BLOCK_N) and scales
    (BLOCK_ROWS * BLOCK_N / 128,), the scales of a row consecutive.
    """
    BLOCKS_PER_ROW: tl.constexpr = BLOCK_N // 128
    # one row of the reshaped tile per 1x128 block, so a reduction along it is the block amax
    block_amax = tl.max(tl.reshape(x_abs, BLOCK_ROWS * BLOCKS_PER_ROW, 128), axis=1)
    tok_scale, tok_rcp = compute_fp8_scale(block_amax, NATIVE_E8M0)
    x_blocks = tl.reshape(x_f32, BLOCK_ROWS * BLOCKS_PER_ROW, 128)
    tok_fp8 = tl.reshape((x_blocks * tok_rcp[:, None]).to(tl.float8e4nv), BLOCK_ROWS, BLOCK_N)
    return tok_fp8, tok_scale


@triton.jit
def fp8cast_rowwise_colwise_kernel(x_ptr, out_tok_ptr, scale_tok_ptr, out_ch_ptr, scale_ch_ptr, M, N, stride_x_row, stride_x_col, stride_out_tok_row, scale_tok_cols, stride_out_ch_row, scale_ch_cols, block_to_group_ptr, cumsum_ptr, BLOCK_N: tl.constexpr, NATIVE_E8M0: tl.constexpr, WRITE_KMAJOR: tl.constexpr = False):
    """
    1x128 rowwise and 128x1 colwise quantization, M % 128 == 0.

    With WRITE_KMAJOR the colwise output is written K-major (each group's (K_g, N) block stored as
    (N, K_g), concatenated) with scales (N, M / 128); this needs block_to_group_ptr and cumsum_ptr
    (grouped_mm_offs). Otherwise colwise is (M, N) with scales (M / 128, N).
    """
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
    x_f32 = x.to(tl.float32)

    tok_fp8, tok_scale = quantize_row_blocks(x_abs, x_f32, BLOCK_ROWS, BLOCK_N, NATIVE_E8M0)
    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=tile_mask)

    tok_scale_col_offs = pid_col * BLOCKS_PER_ROW + tl.arange(0, BLOCKS_PER_ROW)
    tok_scale_ptrs = scale_tok_ptr + row_offs[:, None] * scale_tok_cols + tok_scale_col_offs[None, :]
    tok_scale_mask = (row_offs[:, None] < M) & (tok_scale_col_offs[None, :] < scale_tok_cols)
    tl.store(tok_scale_ptrs, tl.reshape(tok_scale, BLOCK_ROWS, BLOCKS_PER_ROW), mask=tok_scale_mask)

    ch_amax = tl.max(x_abs, axis=0)
    ch_scale, ch_rcp = compute_fp8_scale(ch_amax, NATIVE_E8M0)
    ch_fp8 = (x_f32 * ch_rcp[None, :]).to(tl.float8e4nv)
    col_mask = col_offs < N

    if WRITE_KMAJOR:
        # groups are 128-aligned, so each pid_row belongs to exactly one group
        group_idx = tl.load(block_to_group_ptr + pid_row).to(tl.int64)
        group_end, group_start = tl.load(cumsum_ptr + group_idx).to(tl.int64), tl.load(cumsum_ptr + group_idx - 1, mask=group_idx > 0, other=0).to(tl.int64)
        num_group_rows, group_row_start = group_end - group_start, row_start - group_start

        group_base = group_start * N
        tile_row_offs = tl.arange(0, BLOCK_ROWS).to(tl.int64)
        kmajor_offs = group_base + col_offs[None, :].to(tl.int64) * num_group_rows + (group_row_start + tile_row_offs)[:, None]
        tl.store(out_ch_ptr + kmajor_offs, ch_fp8, mask=tile_mask)

        ch_scale_ptrs = scale_ch_ptr + col_offs * scale_ch_cols + pid_row
        tl.store(ch_scale_ptrs, ch_scale, mask=col_mask)
    else:
        out_ch_ptrs = out_ch_ptr + row_offs[:, None] * stride_out_ch_row + col_offs[None, :]
        tl.store(out_ch_ptrs, ch_fp8, mask=tile_mask)

        ch_scale_ptrs = scale_ch_ptr + pid_row * scale_ch_cols + col_offs
        tl.store(ch_scale_ptrs, ch_scale, mask=col_mask)


def fp8cast_rowwise_colwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise and colwise FP8 quantization of x from one read. x is bf16 or fp32 (M, N) with
    M % 128 == 0.

    Returns (tok_fp8, tok_scale, ch_fp8, ch_scale): tok_fp8 float8_e4m3fn (M, N) with 1x128 row
    blocks and tok_scale float32 (M, ceil(N / 128)); ch_fp8 float8_e4m3fn (M, N) with 128x1 column
    blocks and ch_scale float32 (M / 128, N). Scales are row-major.
    """
    assert x.ndim == 2 and x.shape[0] % 128 == 0, f"Expected 2D tensor with M%128==0, got shape {x.shape}"
    M, N = x.shape
    BLOCK = 128

    num_row_blocks, scale_tok_cols = M // BLOCK, (N + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_ch = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_ch = torch.empty((num_row_blocks, N), dtype=torch.float32, device=x.device)

    BLOCK_N = 128
    grid = (num_row_blocks, (N + BLOCK_N - 1) // BLOCK_N)

    # block_to_group_ptr and cumsum_ptr are read only with WRITE_KMAJOR; any tensor fills them
    fp8cast_rowwise_colwise_kernel[grid](x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_ch_ptr=out_ch, scale_ch_ptr=scale_ch, M=M, N=N, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_tok_row=N, scale_tok_cols=scale_tok_cols, stride_out_ch_row=N, scale_ch_cols=N, block_to_group_ptr=out_tok, cumsum_ptr=out_tok, BLOCK_N=BLOCK_N, NATIVE_E8M0=NATIVE_E8M0, WRITE_KMAJOR=False, num_warps=4, num_stages=2)

    return out_tok, scale_tok, out_ch, scale_ch
