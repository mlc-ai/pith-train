from typing import Optional

import torch
import triton
import triton.language as tl

from pithtrain.kernels.host_transfer import get_pinned_buffer

__all__ = ["GEMM_ALLOC_ALIGNMENT", "scatter_by_expert"]

GEMM_ALLOC_ALIGNMENT = 1024

# The padded sizes are copied to the host on a side stream so the copy overlaps the scatter kernel.
shared_copy_stream: Optional[torch.cuda.Stream] = None
shared_copy_event: Optional[torch.cuda.Event] = None


@triton.jit
def compute_group_offsets_kernel(
    expert_idx_ptr,  # [num_tokens] int64
    group_counts_ptr,  # [num_groups] int64, pre-zeroed
    grouped_mm_offs_ptr,  # [num_groups] int32
    ks_ptr,  # [num_groups] int32, padded sizes
    completion_counter_ptr,  # [1] int64, pre-zeroed
    num_tokens,
    num_ctas,
    PADDING_ALIGNMENT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    for token_start in range(pid * BLOCK, num_tokens, num_ctas * BLOCK):
        token_offs = token_start + tl.arange(0, BLOCK)
        token_mask = token_offs < num_tokens
        expert_idx = tl.load(expert_idx_ptr + token_offs, mask=token_mask, other=0)
        tl.atomic_add(group_counts_ptr + expert_idx, 1, mask=token_mask)

    # the last CTA to finish its counts does the pad + prefix sum
    is_last = tl.atomic_add(completion_counter_ptr, 1) == num_ctas - 1
    if is_last:
        group_end: tl.int64 = 0
        for group_idx in tl.static_range(NUM_GROUPS):
            count = tl.load(group_counts_ptr + group_idx)
            padded_count = ((count + PADDING_ALIGNMENT - 1) // PADDING_ALIGNMENT) * PADDING_ALIGNMENT
            group_end, group_offs = group_end + padded_count, group_idx + tl.arange(0, 1)
            tl.store(grouped_mm_offs_ptr + group_offs, group_end.to(tl.int32))
            tl.store(ks_ptr + group_offs, padded_count.to(tl.int32))


@triton.jit
def split_rows(num_rows, num_parts, part_idx):
    rows_per_part = (num_rows + num_parts - 1) // num_parts
    row_start = part_idx * rows_per_part
    row_end = tl.minimum(row_start + rows_per_part, num_rows)
    return row_start, row_end


@triton.jit
def zero_rows(output_tokens_ptr, first_row, row_start, row_end, dst_row_stride, hidden_size, zero_dtype: tl.constexpr, BLOCK_H: tl.constexpr, NUM_H_BLOCKS: tl.constexpr):
    for row_idx in range(row_start, row_end):
        row_base = (first_row + row_idx) * dst_row_stride
        for i in tl.static_range(NUM_H_BLOCKS):
            h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = h_offs < hidden_size
            tl.store(output_tokens_ptr + row_base + h_offs, tl.zeros([BLOCK_H], dtype=zero_dtype), mask=h_mask)


@triton.jit
def scatter_for_grouped_gemm_kernel(
    input_tokens_ptr,  # [num_tokens, hidden_size]
    expert_idx_ptr,  # [num_tokens] int64
    grouped_mm_offs_ptr,  # [num_groups] int32, padded cumulative offsets
    group_cursors_ptr,  # [num_groups] int64, pre-zeroed (atomic row positions)
    group_counts_ptr,  # [num_groups] int64, from the prep kernel
    output_tokens_ptr,  # [max_rows, hidden_size]
    reverse_shuffle_idxs_ptr,  # [num_tokens] int64
    hidden_size,
    src_row_stride,
    dst_row_stride,
    num_tokens,
    num_groups,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    GEMM_ALLOC_ALIGNMENT: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    zero_dtype = input_tokens_ptr.dtype.element_ty

    expert_idx = tl.load(expert_idx_ptr + token_idx)
    pos_in_group = tl.atomic_add(group_cursors_ptr + expert_idx, 1)
    prev_group_idx = tl.maximum(expert_idx - 1, 0)
    group_start = tl.load(grouped_mm_offs_ptr + prev_group_idx, mask=expert_idx > 0, other=0)
    out_row_idx = pos_in_group + group_start.to(tl.int64)
    tl.store(reverse_shuffle_idxs_ptr + token_idx, out_row_idx)

    src_row_base, dst_row_base = token_idx * src_row_stride, out_row_idx * dst_row_stride
    for i in tl.static_range(NUM_H_BLOCKS):
        h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h_offs < hidden_size
        token_vals = tl.load(input_tokens_ptr + src_row_base + h_offs, mask=h_mask)
        tl.store(output_tokens_ptr + dst_row_base + h_offs, token_vals, mask=h_mask)

    # this group's padding rows, split among the group's programs
    num_group_tokens = tl.load(group_counts_ptr + expert_idx)
    group_end = tl.load(grouped_mm_offs_ptr + expert_idx).to(tl.int64)
    padded_count = group_end - group_start.to(tl.int64)
    num_pad_rows = padded_count - num_group_tokens

    if num_pad_rows > 0:
        pad_start, pad_end = split_rows(num_pad_rows, num_group_tokens, pos_in_group)
        zero_rows(output_tokens_ptr, group_start.to(tl.int64) + num_group_tokens, pad_start, pad_end, dst_row_stride, hidden_size, zero_dtype, BLOCK_H, NUM_H_BLOCKS)

    # rows [last_group_end, M_rounded) of the allocation, split among all programs
    last_group_end = tl.load(grouped_mm_offs_ptr + num_groups - 1).to(tl.int64)
    M_rounded = ((last_group_end + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT)
    num_tail_rows = M_rounded - last_group_end

    if num_tail_rows > 0:
        tail_start, tail_end = split_rows(num_tail_rows, num_tokens, token_idx)
        zero_rows(output_tokens_ptr, last_group_end, tail_start, tail_end, dst_row_stride, hidden_size, zero_dtype, BLOCK_H, NUM_H_BLOCKS)


def get_copy_stream_and_event(device):
    global shared_copy_stream, shared_copy_event
    if shared_copy_stream is None:
        shared_copy_stream = torch.cuda.Stream(device=device)
        shared_copy_event = torch.cuda.Event()
    return shared_copy_stream, shared_copy_event


def scatter_by_expert(input_tokens: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, padding_alignment: int = 128, group_counts: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Reorder input_tokens by expert_idxs into expert-major groups, each padded with zero rows to
    a multiple of padding_alignment. Returns (output_tokens, reverse_shuffle_idxs,
    grouped_mm_offs, ks_tensor, ks).

    - input_tokens: (num_tokens, hidden) with unit inner stride (any row stride, any dtype).
      expert_idxs: int64 (num_tokens,) contiguous, values in [0, num_groups); a length other than
      num_tokens raises ValueError. Inputs are not modified.
    - group_counts: the true rows per group on the host, if known; ks then follows without the
      device-to-host copy (and host sync), and must match expert_idxs.
    - output_tokens: (sum(ks), hidden) row-major in input_tokens' dtype, a prefix view of an
      allocation whose rows past sum(ks), up to a multiple of 1024, are zero too. Group g occupies
      rows [offs[g - 1], offs[g]), its tokens first, then zero rows.
    - reverse_shuffle_idxs: int64 (num_tokens,), the output row of each input token.
    - grouped_mm_offs: int32 (G,) cumulative padded group ends. ks_tensor: int32 (G,) padded group
      sizes (zero for an empty group). ks: the same as a host list.
    - num_tokens == 0 returns empty tensors, grouped_mm_offs and ks_tensor of shape (0,), and
      ks == [].

    Rows are copied exactly. Placement within a group is atomic_add order, not deterministic.
    """
    num_tokens, hidden_size = input_tokens.shape[0], input_tokens.shape[1]
    device = input_tokens.device

    if num_tokens == 0:
        output_tokens = torch.empty((0, hidden_size), device=device, dtype=input_tokens.dtype)
        reverse_shuffle_idxs = torch.empty((0,), device=device, dtype=torch.int64)
        grouped_mm_offs = torch.empty((0,), device=device, dtype=torch.int32)
        ks_tensor = torch.empty((0,), device=device, dtype=torch.int32)
        return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, []

    # upper bound on sum(ks) without a device sync
    max_rows = num_tokens + num_groups * (padding_alignment - 1)
    max_rows = ((max_rows + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT)

    # the scatter kernel zeroes the padding and tail rows
    output_tokens = torch.empty((max_rows, hidden_size), device=device, dtype=input_tokens.dtype)
    reverse_shuffle_idxs = torch.empty((num_tokens,), device=device, dtype=torch.int64)
    grouped_mm_offs = torch.empty((num_groups,), device=device, dtype=torch.int32)
    ks_tensor = torch.empty((num_groups,), device=device, dtype=torch.int32)
    all_zeros = torch.zeros((num_groups * 2 + 1,), device=device, dtype=torch.int64)
    device_group_counts, group_cursors, completion_counter = all_zeros[:num_groups], all_zeros[num_groups : num_groups * 2], all_zeros[num_groups * 2 :]

    # The prep kernel masks its expert_idxs loads with num_tokens: a shorter expert_idxs would be
    # read out of bounds and the garbage would index group_counts.
    if expert_idxs.shape[0] != num_tokens:
        raise ValueError(f"{expert_idxs.shape[0]=} must equal the input row count {num_tokens=}")

    BLOCK = 1024
    num_ctas = min(triton.cdiv(num_tokens, BLOCK), 32)
    compute_group_offsets_kernel[(num_ctas,)](expert_idxs, device_group_counts, grouped_mm_offs, ks_tensor, completion_counter, num_tokens, num_ctas, PADDING_ALIGNMENT=padding_alignment, NUM_GROUPS=num_groups, BLOCK=BLOCK)

    # ks_tensor is final after the prep kernel
    copy_stream, copy_event = get_copy_stream_and_event(device)
    copy_event.record()

    BLOCK_H = 256
    NUM_H_BLOCKS = triton.cdiv(hidden_size, BLOCK_H)
    scatter_for_grouped_gemm_kernel[(num_tokens,)](input_tokens, expert_idxs, grouped_mm_offs, group_cursors, device_group_counts, output_tokens, reverse_shuffle_idxs, hidden_size, input_tokens.stride(0), output_tokens.stride(0), num_tokens, num_groups, BLOCK_H=BLOCK_H, NUM_H_BLOCKS=NUM_H_BLOCKS, GEMM_ALLOC_ALIGNMENT=GEMM_ALLOC_ALIGNMENT)

    if group_counts is not None:
        ks = [(count + padding_alignment - 1) // padding_alignment * padding_alignment for count in group_counts]
    else:
        ks_host = get_pinned_buffer("ks", num_groups, torch.int32)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(copy_event)
            ks_host.copy_(ks_tensor, non_blocking=True)
        copy_stream.synchronize()
        ks = ks_host.tolist()
    # trimmed to sum(ks) exactly: DeepGEMM's k_grouped_fp8_gemm_tn_contiguous asserts it
    actual_M = sum(ks)
    output_tokens = output_tokens[:actual_M]

    return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks
