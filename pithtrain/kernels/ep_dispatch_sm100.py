"""
Triton kernels of the ep_dispatch feature (SM100 path, and the SM90 fallback). The sender side is
three launches: a per-CTA bincount (tl.histogram, no atomics), a single-CTA reduce + prefix sum that
also builds send_meta, and a dedup scatter + counting sort that places rows with atomic_add, so the
order within a bucket is launch-dependent.
"""

import torch
import triton
import triton.language as tl

__all__ = ["ep_dispatch_recv_idx_sm100", "ep_dispatch_send_idx_sm100"]


@triton.jit
def dedup_bincount_kernel(
    topk_ids_ptr,
    per_cta_expert_hist_ptr,  # [num_ctas * NE_PADDED] int64, output
    per_cta_rank_hist_ptr,  # [num_ctas * EP_PADDED] int64, output
    M,
    topk_row_stride,
    TOPK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NE_PADDED: tl.constexpr,  # next_power_of_2(NUM_EXPERTS + 1)
    EP_PADDED: tl.constexpr,  # next_power_of_2(EP_SIZE + 1)
    BLOCK: tl.constexpr,
):
    """
    Per-CTA histograms of slots per expert and of distinct tokens per EP rank.
    """
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    acc_expert_hist, acc_rank_hist = tl.zeros([NE_PADDED], dtype=tl.int64), tl.zeros([EP_PADDED], dtype=tl.int64)

    for tok_start in range(pid * BLOCK, M, num_ctas * BLOCK):
        tok_offs = tok_start + tl.arange(0, BLOCK)
        tok_mask = tok_offs < M

        # per-token visited-rank bitmask: int32 limits ep_size to 32
        visited_ranks = tl.zeros([BLOCK], dtype=tl.int32)

        for j in tl.static_range(TOPK):
            expert_idx = tl.load(topk_ids_ptr + tok_offs * topk_row_stride + j, mask=tok_mask, other=0)

            # masked lanes count into the sentinel bin NUM_EXPERTS
            safe_expert_idx = tl.where(tok_mask, expert_idx, NUM_EXPERTS).to(tl.int32)
            acc_expert_hist += tl.histogram(safe_expert_idx, NE_PADDED).to(tl.int64)

            rank_idx = expert_idx // EXPERTS_PER_RANK
            rank_bit = (1 << rank_idx).to(tl.int32)
            is_visited = (visited_ranks & rank_bit) != 0
            is_first_visit = tok_mask & ~is_visited
            visited_ranks = visited_ranks | rank_bit

            safe_rank_idx = tl.where(is_first_visit, rank_idx, EP_SIZE).to(tl.int32)
            acc_rank_hist += tl.histogram(safe_rank_idx, EP_PADDED).to(tl.int64)

    expert_offs = tl.arange(0, NE_PADDED)
    tl.store(per_cta_expert_hist_ptr + pid * NE_PADDED + expert_offs, acc_expert_hist)
    rank_offs = tl.arange(0, EP_PADDED)
    tl.store(per_cta_rank_hist_ptr + pid * EP_PADDED + rank_offs, acc_rank_hist)


@triton.jit
def reduce_and_prefix_sum_kernel(
    per_cta_expert_hist_ptr,  # [num_ctas * NE_PADDED] int64
    per_cta_rank_hist_ptr,  # [num_ctas * EP_PADDED] int64
    dedup_tokens_per_rank_ptr,  # [EP_SIZE] int64
    tokens_per_ep_rank_ptr,  # [EP_SIZE] int64
    expert_starts_ptr,  # [NUM_EXPERTS] int64
    rank_starts_ptr,  # [EP_SIZE] int64
    dedup_counters_ptr,  # [EP_SIZE] int64, output, zeroed here for kernel 3
    sort_counters_ptr,  # [NUM_EXPERTS] int64, output, zeroed here for kernel 3
    send_meta_ptr,  # [EP_SIZE * (EXPERTS_PER_RANK + 1)] int64
    NUM_CTAS: tl.constexpr,
    NE_PADDED: tl.constexpr,
    EP_PADDED: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """
    Single CTA: reduce the per-CTA histograms into the counts, their exclusive starts and send_meta,
    and zero kernel 3's atomic counters.
    """
    expert_prefix, rank_prefix = tl.zeros([], dtype=tl.int64), tl.zeros([], dtype=tl.int64)
    for rank_idx in tl.static_range(EP_SIZE):
        rank_slot_count = tl.zeros([], dtype=tl.int64)
        for local_expert_idx in tl.static_range(EXPERTS_PER_RANK):
            expert_count = tl.zeros([], dtype=tl.int64)
            for cta_idx in range(NUM_CTAS):
                expert_count += tl.load(per_cta_expert_hist_ptr + cta_idx * NE_PADDED + rank_idx * EXPERTS_PER_RANK + local_expert_idx)
            tl.store(expert_starts_ptr + rank_idx * EXPERTS_PER_RANK + local_expert_idx, expert_prefix)
            expert_prefix += expert_count
            rank_slot_count += expert_count
            tl.store(send_meta_ptr + rank_idx * (EXPERTS_PER_RANK + 1) + local_expert_idx, expert_count)

        tl.store(tokens_per_ep_rank_ptr + rank_idx, rank_slot_count)

        rank_dedup_count = tl.zeros([], dtype=tl.int64)
        for cta_idx in range(NUM_CTAS):
            rank_dedup_count += tl.load(per_cta_rank_hist_ptr + cta_idx * EP_PADDED + rank_idx)
        tl.store(dedup_tokens_per_rank_ptr + rank_idx, rank_dedup_count)
        tl.store(rank_starts_ptr + rank_idx, rank_prefix)
        rank_prefix += rank_dedup_count
        tl.store(send_meta_ptr + rank_idx * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK, rank_dedup_count)

        tl.store(dedup_counters_ptr + rank_idx, tl.zeros([], dtype=tl.int64))

    for expert_idx in tl.static_range(NUM_EXPERTS):
        tl.store(sort_counters_ptr + expert_idx, tl.zeros([], dtype=tl.int64))


@triton.jit
def dedup_scatter_expand_kernel(
    topk_ids_ptr,
    expert_starts_ptr,  # [num_experts] int64 (from kernel 2)
    rank_starts_ptr,  # [ep_size] int64 (from kernel 2)
    dedup_counters_ptr,  # [ep_size] int64, pre-zeroed (atomics)
    sort_counters_ptr,  # [num_experts] int64, pre-zeroed (atomics)
    dispatch_token_idxs_ptr,  # [total_dedup] int64, output
    dedup_local_pos_ptr,  # [M * ep_size] int64, scratch
    idxs_ptr,  # [M * TOPK] int64, output
    expand_idx_ptr,  # [M * TOPK] int64, output
    M,
    topk_row_stride,
    TOPK: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Pass 1 places each token once per visited rank in dispatch_token_idxs and records its position
    in dedup_local_pos; pass 2 counting-sorts the slots by expert into idxs and looks up expand_idx
    from dedup_local_pos.
    """
    pid = tl.program_id(0)
    tok_start = pid * BLOCK_M
    tok_offs = tok_start + tl.arange(0, BLOCK_M)
    tok_mask = tok_offs < M

    visited_ranks = tl.zeros([BLOCK_M], dtype=tl.int32)

    for j in tl.static_range(TOPK):
        expert_idx = tl.load(topk_ids_ptr + tok_offs * topk_row_stride + j, mask=tok_mask, other=0)
        rank_idx = expert_idx // EXPERTS_PER_RANK

        rank_bit = (1 << rank_idx).to(tl.int32)
        is_visited = (visited_ranks & rank_bit) != 0
        is_first_visit = tok_mask & ~is_visited
        visited_ranks = visited_ranks | rank_bit

        dedup_pos = tl.atomic_add(dedup_counters_ptr + rank_idx, 1, mask=is_first_visit)
        rank_start = tl.load(rank_starts_ptr + rank_idx, mask=is_first_visit, other=0)
        tl.store(dispatch_token_idxs_ptr + rank_start + dedup_pos, tok_offs, mask=is_first_visit)
        tl.store(dedup_local_pos_ptr + tok_offs * EP_SIZE + rank_idx, dedup_pos, mask=is_first_visit)

    for j in tl.static_range(TOPK):
        expert_idx = tl.load(topk_ids_ptr + tok_offs * topk_row_stride + j, mask=tok_mask, other=0)
        rank_idx = expert_idx // EXPERTS_PER_RANK

        expert_start = tl.load(expert_starts_ptr + expert_idx, mask=tok_mask, other=0)
        sort_pos = tl.atomic_add(sort_counters_ptr + expert_idx, 1, mask=tok_mask)
        sorted_idx = expert_start + sort_pos

        flat_idx = tok_offs * TOPK + j
        tl.store(idxs_ptr + sorted_idx, flat_idx, mask=tok_mask)

        local_pos = tl.load(dedup_local_pos_ptr + tok_offs * EP_SIZE + rank_idx, mask=tok_mask, other=0)
        tl.store(expand_idx_ptr + sorted_idx, local_pos, mask=tok_mask)


# Scratch of ep_dispatch_send_idx_sm100, one int64 allocation per shape and stream. Only
# temporaries consumed in stream order before the next call on the same stream live here (and
# send_meta, documented as valid until then); outputs that outlive the call are fresh.
SCRATCH: dict[tuple, tuple[torch.Tensor, ...]] = {}
SCRATCH_ENTRIES = 8


def dispatch_scratch(device: torch.device, M: int, num_ctas: int, ne_padded: int, ep_padded: int, num_experts: int, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, ...]:
    key = (device, torch.cuda.current_stream(device).cuda_stream, M, num_experts, ep_size)
    scratch_views = SCRATCH.get(key)
    if scratch_views is None:
        sizes = (
            num_ctas * ne_padded,  # per_cta_expert_hist
            num_ctas * ep_padded,  # per_cta_rank_hist
            num_experts,  # expert_starts
            ep_size,  # rank_starts
            ep_size,  # dedup_counters
            num_experts,  # sort_counters
            ep_size * (experts_per_rank + 1),  # send_meta
            M * ep_size,  # dedup_local_pos
        )
        scratch_buffer = torch.empty(sum(sizes), dtype=torch.int64, device=device)
        scratch_views = tuple(scratch_buffer.split(sizes))
        if len(SCRATCH) >= SCRATCH_ENTRIES:
            SCRATCH.clear()
        SCRATCH[key] = scratch_views
    return scratch_views


def ep_dispatch_send_idx_sm100(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, num_topk = topk_ids.shape
    device = topk_ids.device
    assert ep_size <= 32, f"ep_size={ep_size} exceeds int32 bitmask limit of 32"

    if M == 0:
        send_meta = torch.zeros(ep_size * (experts_per_rank + 1), dtype=torch.int64, device=device)
        if meta_out is None:
            meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=device)
        meta_out[: 2 * ep_size].zero_()
        return (meta_out[:ep_size], meta_out[ep_size : 2 * ep_size], torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device), send_meta)

    # at most 32 CTAs: kernel 2 reduces the per-CTA histograms in one CTA with NUM_CTAS constexpr
    BLOCK = 1024
    num_ctas = min(triton.cdiv(M, BLOCK), 32)
    # tl.histogram needs a power-of-2 bin count; +1 for the sentinel bin of masked-out lanes
    ne_padded, ep_padded = triton.next_power_of_2(num_experts + 1), triton.next_power_of_2(ep_size + 1)

    (per_cta_expert_hist, per_cta_rank_hist, expert_starts, rank_starts, dedup_counters, sort_counters, send_meta, dedup_local_pos) = dispatch_scratch(device, M, num_ctas, ne_padded, ep_padded, num_experts, ep_size, experts_per_rank)

    dedup_bincount_kernel[(num_ctas,)](topk_ids, per_cta_expert_hist, per_cta_rank_hist, M, topk_ids.stride(0), TOPK=num_topk, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, NUM_EXPERTS=num_experts, NE_PADDED=ne_padded, EP_PADDED=ep_padded, BLOCK=BLOCK)

    if meta_out is None:
        meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=device)
    tokens_per_ep_rank, dedup_tokens_per_gpu = meta_out[:ep_size], meta_out[ep_size : 2 * ep_size]

    reduce_and_prefix_sum_kernel[(1,)](per_cta_expert_hist, per_cta_rank_hist, dedup_tokens_per_gpu, tokens_per_ep_rank, expert_starts, rank_starts, dedup_counters, sort_counters, send_meta, NUM_CTAS=num_ctas, NE_PADDED=ne_padded, EP_PADDED=ep_padded, NUM_EXPERTS=num_experts, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank)

    dispatch_token_idxs = torch.empty(M * ep_size, dtype=torch.int64, device=device)
    idxs = torch.empty(M * num_topk, dtype=torch.int64, device=device)
    expand_idx = torch.empty(M * num_topk, dtype=torch.int64, device=device)

    BLOCK_M = 128
    grid = (triton.cdiv(M, BLOCK_M),)
    dedup_scatter_expand_kernel[grid](topk_ids, expert_starts, rank_starts, dedup_counters, sort_counters, dispatch_token_idxs, dedup_local_pos, idxs, expand_idx, M, topk_ids.stride(0), TOPK=num_topk, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, BLOCK_M=BLOCK_M)

    return (tokens_per_ep_rank, dedup_tokens_per_gpu, dispatch_token_idxs, idxs, expand_idx, send_meta)


@triton.jit
def expert_and_expand_idx_kernel(
    recv_meta_ptr,  # [EP_SIZE * (EXPERTS_PER_RANK + 1)] int64, input
    received_expand_idx_ptr,  # [total] int64, input
    expert_idxs_ptr,  # [total] int64, output
    expand_idx_ptr,  # [total] int64, output
    total,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Per received slot: its local expert, found by an unrolled pass over the (sender, expert)
    segments, and its expand_idx plus the dedup counts of the preceding senders.

    total is a kernel argument, not a device-side sum, so the store mask and the caller's allocation
    are the same number (a mismatch would write out of bounds).
    """
    pid = tl.program_id(0)
    slot_offs = pid * BLOCK + tl.arange(0, BLOCK)
    slot_mask = slot_offs < total

    expert_idx = tl.zeros([BLOCK], dtype=tl.int64)
    slot_dedup_start = tl.zeros([BLOCK], dtype=tl.int64)
    seg_start = tl.zeros([], dtype=tl.int64)
    dedup_start = tl.zeros([], dtype=tl.int64)
    for sender_idx in tl.static_range(EP_SIZE):
        for local_expert_idx in tl.static_range(EXPERTS_PER_RANK):
            seg_count = tl.load(recv_meta_ptr + sender_idx * (EXPERTS_PER_RANK + 1) + local_expert_idx)
            is_in_seg = (slot_offs >= seg_start) & (slot_offs < seg_start + seg_count)
            expert_idx, slot_dedup_start = tl.where(is_in_seg, local_expert_idx, expert_idx), tl.where(is_in_seg, dedup_start, slot_dedup_start)
            seg_start += seg_count
        dedup_start += tl.load(recv_meta_ptr + sender_idx * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK)

    received_idx = tl.load(received_expand_idx_ptr + slot_offs, mask=slot_mask)
    tl.store(expert_idxs_ptr + slot_offs, expert_idx, mask=slot_mask)
    tl.store(expand_idx_ptr + slot_offs, received_idx + slot_dedup_start, mask=slot_mask)


def ep_dispatch_recv_idx_sm100(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    # the kernel indexes both inputs with unit stride
    recv_meta, received_expand_idx = recv_meta.contiguous(), received_expand_idx.contiguous()
    total = received_expand_idx.shape[0]
    expert_idxs, expand_idx = torch.empty_like(received_expand_idx), torch.empty_like(received_expand_idx)
    if total == 0:
        return expert_idxs, expand_idx
    BLOCK = 1024
    expert_and_expand_idx_kernel[(triton.cdiv(total, BLOCK),)](recv_meta, received_expand_idx, expert_idxs, expand_idx, total, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, BLOCK=BLOCK)
    return expert_idxs, expand_idx
