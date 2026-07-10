"""
Fused Triton kernels for expert-parallel dispatch with token deduplication.

Replaces ~22 small PyTorch kernel launches (scatter, argsort, nonzero,
searchsorted, etc.) in prepare_dispatch with three Triton kernels:

  Kernel 1 (_dedup_bincount_kernel):
    Atomic-free parallel bincount using per-CTA private histograms
    via tl.histogram (warp-level reduction, no global atomics).
    Each CTA writes its histogram to a separate global memory slice.

  Kernel 2 (_reduce_and_prefix_sum_kernel):
    Single-CTA kernel that fuses cross-CTA histogram reduction,
    grouped sums, exclusive prefix sums, counter zeroing, and
    send_meta interleaved layout construction.

  Kernel 3 (_dedup_scatter_expand_kernel):
    Pass 1 - dedup scatter: build dispatch_token_idxs + dedup_local_pos lookup table
    Pass 2 - counting sort + expand_idx: produce idxs and expand_idx via table lookup

Key algorithmic improvements over the PyTorch version:
  - Counting sort O(n) replaces argsort O(n log n)
  - O(1) lookup table replaces searchsorted O(n log N)
  - Atomic scatter with pre-allocated output replaces nonzero (no dynamic alloc / sync)
  - tl.histogram replaces global atomic bincount (zero contention)
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from pithtrain.models.interface import AllToAllSplits, RoutingInfo
from pithtrain.operators.token_scatter import get_pinned_buffer, padded_index_gather


@triton.jit
def _dedup_bincount_kernel(
    topk_ids_ptr,
    per_cta_expert_hist_ptr,  # [num_ctas * NE_PADDED] int64, output
    per_cta_gpu_hist_ptr,  # [num_ctas * EP_PADDED] int64, output
    m,
    stride_topk_m,  # topk_ids.stride(0)
    K: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NE_PADDED: tl.constexpr,  # next_power_of_2(NUM_EXPERTS + 1)
    EP_PADDED: tl.constexpr,  # next_power_of_2(EP_SIZE + 1)
    BLOCK: tl.constexpr,
):
    """Atomic-free parallel bincount over topk_ids.

    Reads topk_ids (m, K) and produces two per-CTA histogram slices:
      per_cta_expert_hist - token count per expert (for counting sort buckets).
      per_cta_gpu_hist    - unique-token count per EP rank (dedup via bitmask).
    """
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    # Per-CTA private histograms (register-backed, no atomics needed)
    local_expert_hist = tl.zeros([NE_PADDED], dtype=tl.int64)
    local_gpu_hist = tl.zeros([EP_PADDED], dtype=tl.int64)

    for start in range(pid * BLOCK, m, num_ctas * BLOCK):
        tok_offs = start + tl.arange(0, BLOCK)
        mask = tok_offs < m

        # Per-token GPU visit bitmask (int32 supports up to 32 EP ranks)
        visited = tl.zeros([BLOCK], dtype=tl.int32)

        for j in tl.static_range(K):
            expert_id = tl.load(
                topk_ids_ptr + tok_offs * stride_topk_m + j,
                mask=mask,
                other=0,
            )

            # Expert histogram: sentinel value for masked lanes
            safe_expert_id = tl.where(mask, expert_id, NUM_EXPERTS).to(tl.int32)
            local_expert_hist += tl.histogram(safe_expert_id, NE_PADDED).to(tl.int64)

            # GPU visit tracking
            gpu_id = expert_id // EXPERTS_PER_RANK
            gpu_bit = (1 << gpu_id).to(tl.int32)
            already_visited = (visited & gpu_bit) != 0
            first_visit = mask & ~already_visited
            visited = visited | gpu_bit

            # Dedup GPU histogram: sentinel value for non-first-visit lanes
            safe_gpu_id = tl.where(first_visit, gpu_id, EP_SIZE).to(tl.int32)
            local_gpu_hist += tl.histogram(safe_gpu_id, EP_PADDED).to(tl.int64)

    # Flush to per-CTA global memory slices (regular stores, zero atomics)
    expert_offs = tl.arange(0, NE_PADDED)
    tl.store(
        per_cta_expert_hist_ptr + pid * NE_PADDED + expert_offs,
        local_expert_hist,
    )
    gpu_offs = tl.arange(0, EP_PADDED)
    tl.store(
        per_cta_gpu_hist_ptr + pid * EP_PADDED + gpu_offs,
        local_gpu_hist,
    )


@triton.jit
def _reduce_and_prefix_sum_kernel(
    # Inputs (from kernel 1)
    per_cta_expert_hist_ptr,  # [num_ctas * NE_PADDED] int64
    per_cta_gpu_hist_ptr,  # [num_ctas * EP_PADDED] int64
    # Outputs: aggregated histograms
    dedup_tokens_per_gpu_ptr,  # [EP_SIZE] int64
    tokens_per_ep_rank_ptr,  # [EP_SIZE] int64
    # Outputs: kernel 3 inputs (prefix sums + zeroed counters)
    expert_starts_ptr,  # [NUM_EXPERTS] int64
    gpu_starts_ptr,  # [EP_SIZE] int64
    dedup_counters_ptr,  # [EP_SIZE] int64 (zeroed)
    sort_counters_ptr,  # [NUM_EXPERTS] int64 (zeroed)
    # Output: interleaved send_meta
    send_meta_ptr,  # [EP_SIZE * (EXPERTS_PER_RANK + 1)] int64
    # Constants
    NUM_CTAS: tl.constexpr,
    NE_PADDED: tl.constexpr,
    EP_PADDED: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """Reduce per-CTA histograms and derive all metadata for kernel 3.

    Reads per-CTA expert and GPU histograms from kernel 1, and produces:
      dedup_tokens_per_gpu - unique tokens per EP rank (cross-CTA sum).
      tokens_per_ep_rank   - total tokens per EP rank (grouped expert sum).
      expert_starts        - exclusive prefix sum over tokens_per_expert.
      gpu_starts           - exclusive prefix sum over dedup_tokens_per_gpu.
      dedup_counters       - zeroed atomic counters for kernel 3 dedup scatter.
      sort_counters        - zeroed atomic counters for kernel 3 counting sort.
      send_meta            - interleaved (ep_size, experts_per_rank + 1) layout
                             packing tokens_per_expert and dedup counts for a
                             single metadata all-to-all.
    """
    expert_prefix = tl.zeros([], dtype=tl.int64)
    gpu_prefix = tl.zeros([], dtype=tl.int64)
    for g in tl.static_range(EP_SIZE):
        # Reduce + prefix-sum experts for this EP rank (fused: no intermediate buffer)
        rank_total = tl.zeros([], dtype=tl.int64)
        for e in tl.static_range(EXPERTS_PER_RANK):
            expert_val = tl.zeros([], dtype=tl.int64)
            for c in range(NUM_CTAS):
                expert_val += tl.load(
                    per_cta_expert_hist_ptr + c * NE_PADDED + g * EXPERTS_PER_RANK + e
                )
            tl.store(expert_starts_ptr + g * EXPERTS_PER_RANK + e, expert_prefix)
            expert_prefix += expert_val
            rank_total += expert_val
            tl.store(send_meta_ptr + g * (EXPERTS_PER_RANK + 1) + e, expert_val)

        tl.store(tokens_per_ep_rank_ptr + g, rank_total)

        # Reduce + prefix-sum dedup GPU histogram
        gpu_val = tl.zeros([], dtype=tl.int64)
        for c in range(NUM_CTAS):
            gpu_val += tl.load(per_cta_gpu_hist_ptr + c * EP_PADDED + g)
        tl.store(dedup_tokens_per_gpu_ptr + g, gpu_val)
        tl.store(gpu_starts_ptr + g, gpu_prefix)
        gpu_prefix += gpu_val
        tl.store(send_meta_ptr + g * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK, gpu_val)

        # Zero counters for kernel 3
        tl.store(dedup_counters_ptr + g, tl.zeros([], dtype=tl.int64))

    # Zero sort counters
    for e in tl.static_range(NUM_EXPERTS):
        tl.store(sort_counters_ptr + e, tl.zeros([], dtype=tl.int64))


@triton.jit
def _dedup_scatter_expand_kernel(
    topk_ids_ptr,
    expert_starts_ptr,  # [num_experts] int64 (from kernel 1)
    gpu_starts_ptr,  # [ep_size] int64 (from kernel 1)
    dedup_counters_ptr,  # [ep_size] int64, pre-zeroed (atomics)
    sort_counters_ptr,  # [num_experts] int64, pre-zeroed (atomics)
    dispatch_token_idxs_ptr,  # [total_dedup] int64, output
    dedup_local_pos_ptr,  # [m * ep_size] int64, scratch
    idxs_ptr,  # [m * k] int64, output
    expand_idx_ptr,  # [m * k] int64, output
    m,
    stride_topk_m,
    K: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Two-pass kernel: dedup scatter then counting sort with expand_idx.

    Reads topk_ids and prefix sums from kernel 2, and produces:
      dispatch_token_idxs - token row indices grouped by EP rank (dedup).
      idxs                - flat indices into topk_ids.view(-1), sorted by expert.
      expand_idx          - per-slot local position within the token's dedup
                            chunk, for reconstructing the full layout after
                            all-to-all.
    Uses dedup_local_pos as scratch for O(1) expand_idx lookup in pass 2.
    """
    pid = tl.program_id(0)
    tok_start = pid * BLOCK_M
    tok_offs = tok_start + tl.arange(0, BLOCK_M)
    mask = tok_offs < m

    # -- Pass 1: Dedup scatter --
    # For each token, determine unique GPUs and atomically assign
    # positions in dispatch_token_idxs. Record local positions in
    # dedup_local_pos for expand_idx lookup in Pass 2.
    visited = tl.zeros([BLOCK_M], dtype=tl.int32)

    for j in tl.static_range(K):
        expert_id = tl.load(
            topk_ids_ptr + tok_offs * stride_topk_m + j,
            mask=mask,
            other=0,
        )
        gpu_id = expert_id // EXPERTS_PER_RANK

        gpu_bit = (1 << gpu_id).to(tl.int32)
        already_visited = (visited & gpu_bit) != 0
        first_visit = mask & ~already_visited
        visited = visited | gpu_bit

        # Atomic position in this GPU's dedup chunk
        dedup_pos = tl.atomic_add(dedup_counters_ptr + gpu_id, 1, mask=first_visit)
        gpu_start = tl.load(gpu_starts_ptr + gpu_id, mask=first_visit, other=0)
        tl.store(
            dispatch_token_idxs_ptr + gpu_start + dedup_pos,
            tok_offs,
            mask=first_visit,
        )
        # Record local position for Pass 2 lookup
        tl.store(
            dedup_local_pos_ptr + tok_offs * EP_SIZE + gpu_id,
            dedup_pos,
            mask=first_visit,
        )

    # -- Pass 2: Counting sort + expand_idx --
    for j in tl.static_range(K):
        expert_id = tl.load(
            topk_ids_ptr + tok_offs * stride_topk_m + j,
            mask=mask,
            other=0,
        )
        gpu_id = expert_id // EXPERTS_PER_RANK

        # Counting sort: place flat_idx at the expert's bucket
        expert_start = tl.load(expert_starts_ptr + expert_id, mask=mask, other=0)
        sort_pos = tl.atomic_add(sort_counters_ptr + expert_id, 1, mask=mask)
        sorted_idx = expert_start + sort_pos

        flat_idx = tok_offs * K + j
        tl.store(idxs_ptr + sorted_idx, flat_idx, mask=mask)

        # Lookup dedup_local_pos for expand_idx
        local_pos = tl.load(
            dedup_local_pos_ptr + tok_offs * EP_SIZE + gpu_id,
            mask=mask,
            other=0,
        )
        tl.store(expand_idx_ptr + sorted_idx, local_pos, mask=mask)


def fused_dedup_prepare_dispatch(
    topk_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
) -> tuple[
    torch.Tensor,  # tokens_per_ep_rank: (ep_size,) int64
    torch.Tensor,  # dedup_tokens_per_gpu: (ep_size,) int64
    torch.Tensor,  # dispatch_token_idxs: (m * ep_size,) int64, over-allocated
    torch.Tensor,  # idxs: (m * k,) int64
    torch.Tensor,  # expand_idx: (m * k,) int64
    torch.Tensor,  # send_meta: (ep_size * (experts_per_rank + 1),) int64
]:
    """
    Fused Triton implementation of dedup dispatch index computation.

    Replaces the PyTorch sequence of scatter/argsort/nonzero/searchsorted with
    three Triton kernels: bincount, reduce+prefix-sum, and counting sort +
    dedup scatter + expand_idx computation.

    Returns:
        tokens_per_ep_rank: (ep_size,) - total tokens routed to each EP rank.
        dedup_tokens_per_gpu: (ep_size,) - unique tokens per EP rank (after dedup).
        dispatch_token_idxs: (m * ep_size,) - over-allocated; first
            sum(dedup_tokens_per_gpu) entries are valid token row indices for
            the dedup dispatch gather.
        idxs: (m * k,) - counting-sorted flat indices into topk_ids.view(-1),
            grouped by expert.
        expand_idx: (m * k,) - for each sorted slot, the local position within
            that token's dedup chunk (used to reconstruct full layout from
            deduplicated tokens after all-to-all).
        send_meta: (ep_size * (experts_per_rank + 1),) - interleaved metadata
            for a single all-to-all that piggybacks dedup counts alongside
            per-expert token counts.  Viewed as (ep_size, experts_per_rank + 1):
              [:, :experts_per_rank]  = tokens_per_expert per rank
              [:, experts_per_rank]   = dedup_tokens_per_gpu per rank
    """
    m, k = topk_ids.shape
    device = topk_ids.device
    assert ep_size <= 32, f"ep_size={ep_size} exceeds int32 bitmask limit of 32"

    if m == 0:
        send_meta = torch.zeros(ep_size * (experts_per_rank + 1), dtype=torch.int64, device=device)
        return (
            torch.zeros(ep_size, dtype=torch.int64, device=device),
            torch.zeros(ep_size, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            send_meta,
        )

    # -- Kernel 1: atomic-free bincount with per-CTA private histograms --
    # Cap CTAs at 32: kernel 2 reduces per-CTA histograms in a single CTA with
    # NUM_CTAS as constexpr (fully unrolled), so more CTAs = more code size and
    # register pressure.  32 x 1024 = 32K tokens already exceeds typical micro-batches.
    BLOCK = 1024
    num_ctas = min(triton.cdiv(m, BLOCK), 32)
    # tl.histogram requires power-of-2 num_bins; +1 for the sentinel bin that
    # absorbs masked-out lanes (sentinel values = NUM_EXPERTS / EP_SIZE).
    ne_padded = triton.next_power_of_2(num_experts + 1)
    ep_padded = triton.next_power_of_2(ep_size + 1)

    per_cta_expert_hist = torch.empty(num_ctas * ne_padded, dtype=torch.int64, device=device)
    per_cta_gpu_hist = torch.empty(num_ctas * ep_padded, dtype=torch.int64, device=device)

    _dedup_bincount_kernel[(num_ctas,)](
        # inputs:
        topk_ids,
        # outputs:
        per_cta_expert_hist,
        per_cta_gpu_hist,
        m,
        topk_ids.stride(0),
        K=k,
        EP_SIZE=ep_size,
        EXPERTS_PER_RANK=experts_per_rank,
        NUM_EXPERTS=num_experts,
        NE_PADDED=ne_padded,
        EP_PADDED=ep_padded,
        BLOCK=BLOCK,
    )

    # -- Kernel 2: reduce histograms, prefix sums, send_meta (single CTA) --
    dedup_tokens_per_gpu = torch.empty(ep_size, dtype=torch.int64, device=device)
    tokens_per_ep_rank = torch.empty(ep_size, dtype=torch.int64, device=device)
    expert_starts = torch.empty(num_experts, dtype=torch.int64, device=device)
    gpu_starts = torch.empty(ep_size, dtype=torch.int64, device=device)
    dedup_counters = torch.empty(ep_size, dtype=torch.int64, device=device)
    sort_counters = torch.empty(num_experts, dtype=torch.int64, device=device)
    send_meta = torch.empty(ep_size * (experts_per_rank + 1), dtype=torch.int64, device=device)

    _reduce_and_prefix_sum_kernel[(1,)](
        # inputs:
        per_cta_expert_hist,
        per_cta_gpu_hist,
        # outputs:
        dedup_tokens_per_gpu,
        tokens_per_ep_rank,
        expert_starts,
        gpu_starts,
        # tensors to be zeroed:
        dedup_counters,
        sort_counters,
        send_meta,
        NUM_CTAS=num_ctas,
        NE_PADDED=ne_padded,
        EP_PADDED=ep_padded,
        NUM_EXPERTS=num_experts,
        EP_SIZE=ep_size,
        EXPERTS_PER_RANK=experts_per_rank,
    )

    dispatch_token_idxs = torch.empty(m * ep_size, dtype=torch.int64, device=device)
    dedup_local_pos = torch.empty(m * ep_size, dtype=torch.int64, device=device)
    idxs = torch.empty(m * k, dtype=torch.int64, device=device)
    expand_idx = torch.empty(m * k, dtype=torch.int64, device=device)

    # -- Kernel 3: dedup scatter + counting sort + expand_idx --
    BLOCK_M = 128
    grid = (triton.cdiv(m, BLOCK_M),)
    _dedup_scatter_expand_kernel[grid](
        # inputs:
        topk_ids,
        expert_starts,
        gpu_starts,
        # auxiliary, zeroed:
        dedup_counters,
        sort_counters,
        # outputs:
        dispatch_token_idxs,
        dedup_local_pos,
        idxs,
        expand_idx,
        m,
        topk_ids.stride(0),
        K=k,
        EP_SIZE=ep_size,
        EXPERTS_PER_RANK=experts_per_rank,
        BLOCK_M=BLOCK_M,
    )

    return (
        tokens_per_ep_rank,
        dedup_tokens_per_gpu,
        dispatch_token_idxs,
        idxs,
        expand_idx,
        send_meta,
    )


# -- Post-all-to-all kernels --


@triton.jit
def _build_expert_idxs_kernel(
    tokens_per_expert_group_ptr,  # [NUM_EXPERTS] int64
    expert_idxs_ptr,  # [total] int64, output
    output_splits_tensor_ptr,  # [EP_SIZE] int64, output
    NUM_EXPERTS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build local expert index for each received token slot.

    Reads tokens_per_expert_group (NUM_EXPERTS,) and produces:
      expert_idxs          - local expert index (expert_id % experts_per_rank)
                             for each token slot, via segmented fill.
      output_splits_tensor - total tokens per EP rank (grouped sum), written
                             by CTA 0.
    """
    pid = tl.program_id(0)

    # Compute total and output_splits_tensor (only from CTA 0)
    total = tl.zeros([], dtype=tl.int64)
    for e in tl.static_range(NUM_EXPERTS):
        total += tl.load(tokens_per_expert_group_ptr + e)

    if pid == 0:
        for g in tl.static_range(EP_SIZE):
            rank_sum = tl.zeros([], dtype=tl.int64)
            for e in tl.static_range(EXPERTS_PER_RANK):
                rank_sum += tl.load(tokens_per_expert_group_ptr + g * EXPERTS_PER_RANK + e)
            tl.store(output_splits_tensor_ptr + g, rank_sum)

    # Parallel fill: each thread determines which expert segment it belongs to.
    # We compute expert starts inline via a running accumulator,
    # and check if each position falls in [start, start + count).
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    expert_idx = tl.zeros([BLOCK], dtype=tl.int64)
    start = tl.zeros([], dtype=tl.int64)
    for e in tl.static_range(NUM_EXPERTS):
        count_e = tl.load(tokens_per_expert_group_ptr + e)
        expert_idx = tl.where(mask & (offs >= start), e, expert_idx)
        start += count_e

    # Map to local expert index within EP rank
    local_idx = expert_idx % EXPERTS_PER_RANK
    tl.store(expert_idxs_ptr + offs, local_idx, mask=mask)


def build_expert_idxs(
    tokens_per_expert_group: torch.Tensor,
    ep_size: int,
    experts_per_rank: int,
    max_total: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused expert_idxs + output_splits_tensor, replacing arange + mod + repeat_interleave + sum.

    expert_idxs is over-allocated to max_total. The caller must slice it to
    [:sum(output_splits)] after obtaining output_splits on CPU.
    """
    num_experts = tokens_per_expert_group.shape[0]
    device = tokens_per_expert_group.device

    expert_idxs = torch.empty(max_total, dtype=torch.int64, device=device)
    output_splits_tensor = torch.empty(ep_size, dtype=torch.int64, device=device)

    if max_total == 0:
        output_splits_tensor.zero_()
        return expert_idxs, output_splits_tensor

    BLOCK = 1024
    grid = (triton.cdiv(max_total, BLOCK),)
    _build_expert_idxs_kernel[grid](
        tokens_per_expert_group,
        expert_idxs,
        output_splits_tensor,
        NUM_EXPERTS=num_experts,
        EP_SIZE=ep_size,
        EXPERTS_PER_RANK=experts_per_rank,
        BLOCK=BLOCK,
    )
    return expert_idxs, output_splits_tensor


@triton.jit
def _adjust_expand_idx_kernel(
    received_expand_idx_ptr,  # [N] int64, input
    out_ptr,  # [N] int64, output
    dedup_tokens_from_each_gpu_ptr,  # [EP_SIZE] int64, possibly strided
    output_splits_tensor_ptr,  # [EP_SIZE] int64
    N,
    DEDUP_STRIDE: tl.constexpr,
    EP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Adjust received expand_idx by adding per-sender dedup offsets.

    Reads received_expand_idx (N,) containing local positions from each sender,
    dedup_tokens_from_each_gpu (EP_SIZE,) and output_splits_tensor (EP_SIZE,).
    For each element, determines its sender rank via output_splits boundaries
    and adds the exclusive cumsum of dedup counts, so positions become global
    indices into the concatenated dedup buffer.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    expand_val = tl.load(received_expand_idx_ptr + offs, mask=mask)

    # For each element, find sender rank via linear scan over cumulative output_splits,
    # and add the corresponding exclusive cumsum of dedup counts.
    # EP_SIZE is tiny (<=32), so linear scan is fine.
    offset = tl.zeros([BLOCK], dtype=tl.int64)
    boundary = tl.zeros([], dtype=tl.int64)
    dedup_start = tl.zeros([], dtype=tl.int64)
    for g in tl.static_range(EP_SIZE):
        split_g = tl.load(output_splits_tensor_ptr + g)
        next_boundary = boundary + split_g
        in_rank = (offs >= boundary) & (offs < next_boundary)
        offset = tl.where(in_rank, dedup_start, offset)
        boundary = next_boundary
        dedup_start += tl.load(dedup_tokens_from_each_gpu_ptr + g * DEDUP_STRIDE)

    tl.store(out_ptr + offs, expand_val + offset, mask=mask)


def adjust_expand_idx(
    received_expand_idx: torch.Tensor,
    dedup_tokens_from_each_gpu: torch.Tensor,
    output_splits_tensor: torch.Tensor,
) -> torch.Tensor:
    """Fused expand_idx adjustment replacing cumsum + repeat_interleave + add."""
    N = received_expand_idx.shape[0]
    ep_size = dedup_tokens_from_each_gpu.shape[0]
    out = torch.empty_like(received_expand_idx)
    if N == 0:
        return out
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _adjust_expand_idx_kernel[grid](
        received_expand_idx,
        out,
        dedup_tokens_from_each_gpu,
        output_splits_tensor,
        N,
        DEDUP_STRIDE=dedup_tokens_from_each_gpu.stride(0),
        EP_SIZE=ep_size,
        BLOCK=BLOCK,
    )
    return out


@torch.compiler.disable
def prepare_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    ep_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, RoutingInfo]:
    """
    Expert-parallel dispatch with token deduplication.

    When ep_size > 1, deduplicates tokens routed to the same EP rank via
    multiple experts to cut all-to-all volume, and computes expand_idx to
    reconstruct the full mapping on the receiver side. When ep_size == 1,
    replicates each token once per selected expert.
    """

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    k = topk_ids.shape[1]
    expert_idxs = topk_ids.view(-1)

    if ep_size == 1:
        dispatch_tokens = (
            hidden_states.unsqueeze(1).expand(-1, k, -1).reshape(-1, hidden_states.shape[-1])
        )
        return dispatch_tokens, RoutingInfo(topk_weight, expert_idxs)

    (
        tokens_per_ep_rank,
        dedup_tokens_per_gpu,
        dispatch_token_idxs,
        idxs,
        expand_idx,
        send_meta,  # (ep_size, experts_per_rank + 1) interleaved: [tpe_0..., dedup_0, ...]
    ) = fused_dedup_prepare_dispatch(topk_ids, num_experts, ep_size, experts_per_rank)

    # -- Metadata all-to-all (piggyback dedup counts alongside per-expert token counts) --
    recv_meta = send_meta.new_empty(send_meta.shape[0])
    torch.compiler.disable(dist.all_to_all_single)(recv_meta, send_meta, group=ep_group)
    recv_meta_2d = recv_meta.view(ep_size, experts_per_rank + 1)
    tokens_per_expert_group = recv_meta_2d[:, :experts_per_rank].reshape(-1)
    dedup_tokens_from_each_gpu = recv_meta_2d[:, experts_per_rank]  # (ep_size,)

    m = topk_ids.shape[0]
    expert_idxs, output_splits_tensor = build_expert_idxs(
        tokens_per_expert_group, ep_size, experts_per_rank, max_total=m * k * ep_size
    )

    # -- Batch D-to-H copies on main stream --
    dedup_input_splits_cpu = get_pinned_buffer(
        "dedup_input_splits", ep_size, dedup_tokens_per_gpu.dtype
    )
    dedup_output_splits_cpu = get_pinned_buffer(
        "dedup_output_splits", ep_size, dedup_tokens_from_each_gpu.dtype
    )
    input_splits_cpu = get_pinned_buffer("input_splits", ep_size, tokens_per_ep_rank.dtype)
    output_splits_cpu = get_pinned_buffer("output_splits", ep_size, output_splits_tensor.dtype)
    dedup_input_splits_cpu.copy_(dedup_tokens_per_gpu, non_blocking=True)
    dedup_output_splits_cpu.copy_(dedup_tokens_from_each_gpu, non_blocking=True)
    input_splits_cpu.copy_(tokens_per_ep_rank, non_blocking=True)
    output_splits_cpu.copy_(output_splits_tensor, non_blocking=True)
    torch.cuda.current_stream(hidden_states.device).synchronize()
    dedup_input_splits = dedup_input_splits_cpu.tolist()
    dedup_output_splits = dedup_output_splits_cpu.tolist()
    input_splits = input_splits_cpu.tolist()
    output_splits = output_splits_cpu.tolist()

    # Trim over-allocated expert_idxs to actual size
    total_output = sum(output_splits)
    expert_idxs = expert_idxs[:total_output]

    # -- expand_idx all-to-all + adjustment --
    received_expand_idx = expand_idx.new_empty(total_output)
    torch.compiler.disable(dist.all_to_all_single)(
        received_expand_idx,
        expand_idx,
        output_splits,
        input_splits,
        ep_group,
    )
    expand_idx = adjust_expand_idx(
        received_expand_idx, dedup_tokens_from_each_gpu, output_splits_tensor
    )

    # Trim + gather dedup tokens for dispatch
    total_dedup = sum(dedup_input_splits)
    dispatch_tokens = padded_index_gather(hidden_states, dispatch_token_idxs[:total_dedup])

    return dispatch_tokens, RoutingInfo(
        topk_weight,
        expert_idxs,
        idxs,
        expand_idx,
        AllToAllSplits(dedup_input_splits, dedup_output_splits),
        AllToAllSplits(input_splits, output_splits),
    )
