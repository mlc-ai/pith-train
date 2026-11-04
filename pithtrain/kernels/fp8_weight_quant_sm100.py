"""
Triton kernels of the fp8_weight_quant feature: the 128x128 blockwise FP8 of a weight and its
transpose (2-D, and batched 3-D for grouped experts), and the FP8 parameter all-gather kernels.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8_pack_transpose_sm100", "fp8_scales_sm100", "fp8_shard_amax_sm100", "fp8_shard_cast_sm100", "fp8_quant_weight_grouped_sm100", "fp8_quant_weight_sm100"]


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


def fp8_quant_weight_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def fp8_quant_weight_grouped_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


BLOCK = 128


@triton.jit
def fp8_shard_tile_kernel(x_ptr, amax_ptr, out_ptr, row_offset, local_rows, K, tile_row_start, amax_group_offset, amax_stride_g, amax_cols, CAST: tl.constexpr, NATIVE_E8M0: tl.constexpr):
    """
    One 128x128 tile (global rows [tile_row_idx * 128, tile_row_idx * 128 + 128) of local expert
    pid(2)) of a row shard holding global rows [row_offset, row_offset + local_rows). CAST False
    writes the amax of the shard's rows in the tile to the grid; CAST True casts them with the
    reduced amax grid.
    """
    BLOCK: tl.constexpr = 128
    pid_row, pid_col, pid_g = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    tile_row_idx = tile_row_start + pid_row
    row_offs = tile_row_idx * BLOCK + tl.arange(0, BLOCK) - row_offset
    col_offs = pid_col * BLOCK + tl.arange(0, BLOCK)
    tile_mask = (row_offs[:, None] >= 0) & (row_offs[:, None] < local_rows) & (col_offs[None, :] < K)
    x_offs = pid_g.to(tl.int64) * local_rows * K + row_offs[:, None] * K + col_offs[None, :]
    # Round to bf16 first, as FSDP's mixed-precision cast does.
    x_bf16 = tl.load(x_ptr + x_offs, mask=tile_mask, other=0.0).to(tl.bfloat16)

    amax_offset = (amax_group_offset + pid_g) * amax_stride_g + tile_row_idx * amax_cols + pid_col
    if CAST:
        amax = tl.load(amax_ptr + amax_offset)
        _, rcp = compute_fp8_scale(amax, NATIVE_E8M0)
        x_fp8 = (x_bf16.to(tl.float32) * rcp).to(tl.float8e4nv)
        tl.store(out_ptr + x_offs, x_fp8, mask=tile_mask)
    else:
        amax = tl.max(tl.max(tl.abs(x_bf16), axis=1), axis=0)
        tl.store(amax_ptr + amax_offset, amax.to(tl.float32))


@triton.jit
def fp8_scale_kernel(amax_ptr, scale_ptr, scale_t_ptr, num_rows, num_cols, NATIVE_E8M0: tl.constexpr):
    elem_offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    group_idx = tl.program_id(1)
    elem_mask = elem_offs < num_rows * num_cols
    amax = tl.load(amax_ptr + group_idx * num_rows * num_cols + elem_offs, mask=elem_mask, other=0.0)
    scale, _ = compute_fp8_scale(amax, NATIVE_E8M0)
    tl.store(scale_ptr + group_idx * num_rows * num_cols + elem_offs, scale, mask=elem_mask)
    row_idx, col_idx = elem_offs // num_cols, elem_offs % num_cols
    tl.store(scale_t_ptr + group_idx * num_rows * num_cols + col_idx * num_rows + row_idx, scale, mask=elem_mask)


@triton.jit
def fp8_pack_transpose_kernel(x_ptr, out_ptr, out_t_ptr, N, K):
    BLOCK: tl.constexpr = 128
    pid_row, pid_col, pid_g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row_offs, col_offs = pid_row * BLOCK + tl.arange(0, BLOCK), pid_col * BLOCK + tl.arange(0, BLOCK)
    tile_mask = (row_offs[:, None] < N) & (col_offs[None, :] < K)
    group_base = pid_g.to(tl.int64) * N * K
    tile = tl.load(x_ptr + group_base + row_offs[:, None] * K + col_offs[None, :], mask=tile_mask)
    tl.store(out_ptr + group_base + row_offs[:, None] * K + col_offs[None, :], tile, mask=tile_mask)
    # Swapped-index store, since tl.trans miscompiles fp8 tiles.
    tl.store(out_t_ptr + group_base + col_offs[None, :] * N + row_offs[:, None], tile, mask=tile_mask)


def shard_tiles(shard: torch.Tensor, amax: torch.Tensor, out: Optional[torch.Tensor], row_offset: int, group_offset: int) -> None:
    num_local_groups, local_rows, K = shard.shape
    if shard.numel() == 0:
        return
    tile_row_start, tile_row_end = row_offset // BLOCK, (row_offset + local_rows + BLOCK - 1) // BLOCK
    grid = (tile_row_end - tile_row_start, triton.cdiv(K, BLOCK), num_local_groups)
    fp8_shard_tile_kernel[grid](shard, amax, out if out is not None else shard, row_offset, local_rows, K, tile_row_start, group_offset, amax.stride(0), amax.shape[2], CAST=out is not None, NATIVE_E8M0=NATIVE_E8M0, num_warps=8)


def fp8_shard_amax_sm100(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:
    shard_tiles(shard, amax, None, row_offset, group_offset)


def fp8_shard_cast_sm100(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:
    shard_tiles(shard, amax, out, row_offset, group_offset)


def fp8_scales_sm100(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    G, num_rows, num_cols = amax.shape
    grid = (triton.cdiv(num_rows * num_cols, 1024), G)
    fp8_scale_kernel[grid](amax, scale, scale_t, num_rows, num_cols, NATIVE_E8M0=NATIVE_E8M0)


def fp8_pack_transpose_sm100(x: torch.Tensor, out: torch.Tensor) -> None:
    G, N, K = x.shape
    x, out = x.view(torch.uint8), out.view(torch.uint8)
    grid = (triton.cdiv(N, BLOCK), triton.cdiv(K, BLOCK), G)
    fp8_pack_transpose_kernel[grid](x, out, out[x.numel() :], N, K, num_warps=4)
