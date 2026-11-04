"""
Per-group column sums over rows sorted by group, deterministic and atomic-free.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import SM_COUNT

__all__ = ["block_segment_sum", "segment_sum"]


@triton.jit
def segment_sum_kernel(x_ptr, offs_padded_ptr, out_ptr, H: tl.constexpr, HEAD_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr):
    group_idx, h_block_idx = tl.program_id(0), tl.program_id(1)
    row_start, row_end = tl.load(offs_padded_ptr + group_idx), tl.load(offs_padded_ptr + group_idx + 1)

    h_offs = h_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H

    acc_sum = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    for tile_start in tl.range(row_start, row_end, BLOCK_M):
        row_offs = tile_start + tl.arange(0, BLOCK_M)
        row_mask = row_offs < row_end
        x_ptrs = x_ptr + row_offs[:, None] * H + h_offs[None, :]
        x = tl.load(x_ptrs, mask=row_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        acc_sum += tl.sum(x, axis=0)

    tl.store(out_ptr + group_idx * H + h_offs, acc_sum, mask=h_mask)


@triton.jit
def segment_tile(offs_padded_ptr, G, BLOCK_M: tl.constexpr, BLOCK_G: tl.constexpr):
    """
    Groups are cut into BLOCK_M-row tiles from their own start, numbered consecutively across
    groups. Returns (tiles before each group, tiles per group, group starts, group ends) as BLOCK_G
    vectors; past G a lane has no tiles and zero bounds, and its tiles-before is the total count.
    """
    group_offs = tl.arange(0, BLOCK_G)
    group_starts, group_ends = tl.load(offs_padded_ptr + group_offs, mask=group_offs < G, other=0), tl.load(offs_padded_ptr + group_offs + 1, mask=group_offs < G, other=0)
    num_tiles = (group_ends - group_starts + BLOCK_M - 1) // BLOCK_M
    first_tile = tl.cumsum(num_tiles, 0) - num_tiles
    return first_tile, num_tiles, group_starts, group_ends


@triton.jit
def segment_tile_sum_kernel(x_ptr, offs_padded_ptr, partials_ptr, G, H: tl.constexpr, HEAD_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_G: tl.constexpr):
    """
    The tile is the one the per-group loop of segment_sum_kernel would take, so its sum has
    the same bits.
    """
    tile_idx, h_block_idx = tl.program_id(0), tl.program_id(1)
    first_tile, num_tiles, group_starts, group_ends = segment_tile(offs_padded_ptr, G, BLOCK_M, BLOCK_G)
    group_offs = tl.arange(0, BLOCK_G)
    group_idx = tl.sum(((first_tile + num_tiles) <= tile_idx).to(tl.int32), axis=0)
    if group_idx >= G:
        return
    group_mask = group_offs == group_idx
    row_start = tl.sum(tl.where(group_mask, group_starts, 0), axis=0) + (tile_idx - tl.sum(tl.where(group_mask, first_tile, 0), axis=0)) * BLOCK_M
    row_end = tl.sum(tl.where(group_mask, group_ends, 0), axis=0)

    h_offs = h_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H
    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < row_end
    x_ptrs = x_ptr + row_offs[:, None].to(tl.int64) * H + h_offs[None, :]
    x = tl.load(x_ptrs, mask=row_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
    tl.store(partials_ptr + tile_idx.to(tl.int64) * H + h_offs, tl.sum(x, axis=0), mask=h_mask)


@triton.jit
def segment_tile_reduce_kernel(partials_ptr, offs_padded_ptr, out_ptr, G, H: tl.constexpr, HEAD_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_G: tl.constexpr):
    group_idx, h_block_idx = tl.program_id(0), tl.program_id(1)
    first_tile, num_tiles, _, _ = segment_tile(offs_padded_ptr, G, BLOCK_M, BLOCK_G)
    group_mask = tl.arange(0, BLOCK_G) == group_idx
    tile_start, group_num_tiles = tl.sum(tl.where(group_mask, first_tile, 0), axis=0), tl.sum(tl.where(group_mask, num_tiles, 0), axis=0)

    h_offs = h_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H
    acc_sum = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    for i in tl.range(0, group_num_tiles):
        acc_sum += tl.load(partials_ptr + (tile_start + i).to(tl.int64) * H + h_offs, mask=h_mask, other=0.0)
    tl.store(out_ptr + group_idx * H + h_offs, acc_sum, mask=h_mask)


def segment_sum(x: torch.Tensor, grouped_mm_offs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Per-group column sums of rows already sorted by group (the bias gradient when x is the gradient
    of y = z + bias[group_ids]): a new (G, H) tensor of out_dtype, row g the sum of rows
    [offs[g - 1], offs[g]) of x, with the start 0 for g = 0.

    - x: (M, H) contiguous, any float dtype; not modified.
    - grouped_mm_offs: integer (G,) non-decreasing cumulative group ends; ends past M are clamped to
      M and rows past the last end are ignored. Groups need not be multiples of 128; an empty group
      sums to zero.
    - out_dtype: the result dtype; the sums are fp32, rounded once.

    Deterministic summation order: each group is cut into 128-row tiles from its own start, and the
    tiles' fp32 column sums are added in row order starting from zero.
    """
    assert x.is_contiguous(), "x must be contiguous"
    M, H = x.shape
    G = grouped_mm_offs.shape[0]

    # a leading zero lets every group read [start, end) without a branch
    offs_padded = torch.empty(G + 1, dtype=torch.int32, device=x.device)
    offs_padded[0] = 0
    offs_padded[1:].copy_(grouped_mm_offs.to(torch.int32).clamp(max=M))

    HEAD_BLOCK, BLOCK_M = 128, 128
    out = torch.empty((G, H), dtype=torch.float32, device=x.device)
    num_h_blocks = triton.cdiv(H, HEAD_BLOCK)
    if G * num_h_blocks >= 2 * SM_COUNT:
        # enough groups to fill the GPU with one block per (group, h-tile)
        meta = dict(H=H, HEAD_BLOCK=HEAD_BLOCK, BLOCK_M=BLOCK_M)
        segment_sum_kernel[(G, num_h_blocks)](x, offs_padded, out, **meta)
    else:
        BLOCK_G = max(16, triton.next_power_of_2(G))
        # each group has at most one partial tile: a bound on the tile count without a sync
        max_tiles = triton.cdiv(M, BLOCK_M) + G
        partials = torch.empty((max_tiles, H), dtype=torch.float32, device=x.device)
        meta = dict(H=H, HEAD_BLOCK=HEAD_BLOCK, BLOCK_M=BLOCK_M, BLOCK_G=BLOCK_G)
        segment_tile_sum_kernel[(max_tiles, num_h_blocks)](x, offs_padded, partials, G, **meta)
        segment_tile_reduce_kernel[(G, num_h_blocks)](partials, offs_padded, out, G, **meta)
    return out.to(out_dtype) if out_dtype is not torch.float32 else out


@triton.jit
def block_segment_sum_kernel(partials_ptr, offs_ptr, out_ptr, H, BLOCK_H: tl.constexpr):
    group_idx = tl.program_id(0)
    h_offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    block_end, block_start = tl.load(offs_ptr + group_idx) // 128, tl.load(offs_ptr + group_idx - 1, mask=group_idx > 0, other=0) // 128
    acc_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for block_idx in tl.range(block_start, block_end):
        acc_sum += tl.load(partials_ptr + block_idx.to(tl.int64) * H + h_offs, mask=h_mask, other=0.0)
    tl.store(out_ptr + group_idx * H + h_offs, acc_sum.to(out_ptr.dtype.element_ty), mask=h_mask)


def block_segment_sum(partials: torch.Tensor, grouped_mm_offs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    (G, H) per-expert sums, in dtype, of the (M / 128, H) fp32 partials of 128-row blocks, added in
    row order. Experts are the 128-aligned row ranges given by grouped_mm_offs (int32 (G,)).
    """
    G, H = grouped_mm_offs.shape[0], partials.shape[1]
    out = torch.empty((G, H), dtype=dtype, device=partials.device)
    block_segment_sum_kernel[(G, triton.cdiv(H, 512))](partials, grouped_mm_offs, out, H, BLOCK_H=512)
    return out
