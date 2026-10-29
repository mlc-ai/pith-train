"""
Triton 128x128 blockwise FP8 of a weight and its transpose (2-D, and batched 3-D for grouped
experts): fp8_quant_weight / fp8_quant_weight_grouped on SM90, where they are called inside
torch.compile regions, which cannot trace the CuTe-DSL launchers.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8cast_blockwise_transpose", "fp8cast_blockwise_transpose_grouped"]


@triton.jit
def fp8cast_blockwise_transpose_kernel(x_ptr, out_ptr, scale_ptr, out_t_ptr, scale_t_ptr, M, K, stride_x_row, stride_x_col, stride_out_row, scale_cols, stride_out_t_row, scale_t_cols, NATIVE_E8M0: tl.constexpr):
    BLOCK: tl.constexpr = 128

    pid_row, pid_col = tl.program_id(0), tl.program_id(1)
    row_start, col_start = pid_row * BLOCK, pid_col * BLOCK
    row_offs, col_offs = row_start + tl.arange(0, BLOCK), col_start + tl.arange(0, BLOCK)

    x_ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    tile_mask = (row_offs[:, None] < M) & (col_offs[None, :] < K)
    x_bf16 = tl.load(x_ptrs, mask=tile_mask, other=0.0)

    x_abs = tl.abs(x_bf16)
    row_amax = tl.max(x_abs, axis=1)
    block_amax = tl.max(row_amax, axis=0)

    scale, rcp = compute_fp8_scale(block_amax, NATIVE_E8M0)

    x_f32 = x_bf16.to(tl.float32)
    fp8_tile = (x_f32 * rcp).to(tl.float8e4nv)

    out_ptrs = out_ptr + row_offs[:, None] * stride_out_row + col_offs[None, :]
    tl.store(out_ptrs, fp8_tile, mask=tile_mask)

    # tl.trans miscompiles fp8 tiles: store the same tile to swapped-index addresses instead.
    out_t_ptrs = out_t_ptr + col_offs[None, :] * stride_out_t_row + row_offs[:, None]
    tl.store(out_t_ptrs, fp8_tile, mask=tile_mask)

    scale_offset = pid_row * scale_cols + pid_col
    tl.store(scale_ptr + scale_offset, scale)

    scale_t_offset = pid_col * scale_t_cols + pid_row
    tl.store(scale_t_ptr + scale_t_offset, scale)


def fp8cast_blockwise_transpose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    BLOCK = 128
    scale_rows, scale_cols = (M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK

    out = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((scale_rows, scale_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((scale_cols, scale_rows), dtype=torch.float32, device=x.device)

    grid = (scale_rows, scale_cols)

    fp8cast_blockwise_transpose_kernel[grid](x_ptr=x, out_ptr=out, scale_ptr=scale, out_t_ptr=out_t, scale_t_ptr=scale_t, M=M, K=K, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_row=K, scale_cols=scale_cols, stride_out_t_row=M, scale_t_cols=scale_rows, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2)

    return out, scale, out_t, scale_t


@triton.jit
def fp8cast_blockwise_transpose_batched_kernel(x_ptr, out_ptr, scale_ptr, out_t_ptr, scale_t_ptr, G, N, K, stride_x_g, stride_x_row, stride_x_col, stride_out_g, stride_out_row, scale_cols, stride_scale_g, stride_out_t_g, stride_out_t_row, scale_t_cols, stride_scale_t_g, NATIVE_E8M0: tl.constexpr):
    BLOCK: tl.constexpr = 128

    pid_row, pid_col, pid_g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row_start, col_start = pid_row * BLOCK, pid_col * BLOCK
    row_offs, col_offs = row_start + tl.arange(0, BLOCK), col_start + tl.arange(0, BLOCK)

    x_base = x_ptr + pid_g * stride_x_g
    x_ptrs = x_base + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    tile_mask = (row_offs[:, None] < N) & (col_offs[None, :] < K)
    x_bf16 = tl.load(x_ptrs, mask=tile_mask, other=0.0)

    x_abs = tl.abs(x_bf16)
    row_amax = tl.max(x_abs, axis=1)
    block_amax = tl.max(row_amax, axis=0)

    scale, rcp = compute_fp8_scale(block_amax, NATIVE_E8M0)

    x_f32 = x_bf16.to(tl.float32)
    fp8_tile = (x_f32 * rcp).to(tl.float8e4nv)

    out_base = out_ptr + pid_g * stride_out_g
    out_ptrs = out_base + row_offs[:, None] * stride_out_row + col_offs[None, :]
    tl.store(out_ptrs, fp8_tile, mask=tile_mask)

    # tl.trans miscompiles fp8 tiles: store the same tile to swapped-index addresses instead.
    out_t_base = out_t_ptr + pid_g * stride_out_t_g
    out_t_ptrs = out_t_base + col_offs[None, :] * stride_out_t_row + row_offs[:, None]
    tl.store(out_t_ptrs, fp8_tile, mask=tile_mask)

    scale_offset = pid_g * stride_scale_g + pid_row * scale_cols + pid_col
    tl.store(scale_ptr + scale_offset, scale)

    scale_t_offset = pid_g * stride_scale_t_g + pid_col * scale_t_cols + pid_row
    tl.store(scale_t_ptr + scale_t_offset, scale)


def fp8cast_blockwise_transpose_grouped(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    G, N, K = x.shape
    BLOCK = 128
    scale_rows, scale_cols = (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK

    out = torch.empty((G, N, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((G, scale_rows, scale_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((G, scale_cols, scale_rows), dtype=torch.float32, device=x.device)

    grid = (scale_rows, scale_cols, G)

    fp8cast_blockwise_transpose_batched_kernel[grid](x_ptr=x, out_ptr=out, scale_ptr=scale, out_t_ptr=out_t, scale_t_ptr=scale_t, G=G, N=N, K=K, stride_x_g=x.stride(0), stride_x_row=x.stride(1), stride_x_col=x.stride(2), stride_out_g=N * K, stride_out_row=K, scale_cols=scale_cols, stride_scale_g=scale_rows * scale_cols, stride_out_t_g=K * N, stride_out_t_row=N, scale_t_cols=scale_rows, stride_scale_t_g=scale_cols * scale_rows, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2)

    return out, scale, out_t, scale_t
