"""Stable grouped-row indices, independent of the CUDA scatter implementation."""

import torch


def stable_group_order(
    expert_idxs: torch.Tensor, actual_counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return original rows sorted stably by expert and unpadded group starts.

    ``actual_counts`` is the integer histogram of ``expert_idxs``. Both inputs
    stay on their current device. For sorted position ``p`` with expert ``e``,
    ``p - starts[e]`` is its unique, stable expert-local ordinal.

    Stability is relative to this call's incoming row order. It cannot repair
    nondeterministic token ordering introduced by an earlier EP dispatch.
    """
    order = torch.argsort(expert_idxs, stable=True)
    starts = actual_counts.cumsum(dim=0) - actual_counts
    return order, starts


def stable_dedup_prepare_dispatch(
    topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int
) -> tuple[torch.Tensor, ...]:
    """Build EP metadata in destination-rank, expert, original-slot order.

    Deduplicated tokens use ascending original token indices within each
    destination rank. Expert slots preserve their original token/top-k order.
    The dispatch index buffer has the existing fixed ``tokens * ep_size``
    capacity; only its prefix of ``sum(dedup_counts)`` entries is consumed.

    This uses device integer sorts and O(tokens * top_k * ep_size) temporary
    boolean storage. It trades the fused atomic scatter's throughput for stable
    grouped-GEMM reduction order, without changing routes or token weights.
    The caller already disables compilation for distributed metadata handling.
    """
    rows, top_k = topk_ids.shape
    flat_ids = topk_ids.reshape(-1)
    counts = torch.bincount(flat_ids, minlength=num_experts)
    tokens_per_rank = counts.reshape(ep_size, experts_per_rank).sum(dim=1)
    ranks = topk_ids // experts_per_rank
    destinations = torch.arange(ep_size, device=topk_ids.device)
    visits = (ranks[:, :, None] == destinations[None, None, :]).any(dim=1)
    dedup_counts = visits.sum(dim=0)
    local_positions = visits.to(torch.int64).cumsum(dim=0) - 1
    idxs = torch.argsort(flat_ids, stable=True)
    token_ids = idxs // top_k
    rank_ids = flat_ids[idxs] // experts_per_rank
    expand_idx = local_positions[token_ids, rank_ids]
    capacity = rows * ep_size
    rank_major = visits.transpose(0, 1).reshape(-1)
    keys = torch.arange(capacity, device=topk_ids.device)
    keys = keys + (~rank_major).to(torch.int64) * capacity
    dispatch_token_idxs = torch.argsort(keys) % rows if rows else keys
    send_meta = torch.cat(
        (counts.reshape(ep_size, experts_per_rank), dedup_counts[:, None]), dim=1
    ).reshape(-1)
    return (tokens_per_rank, dedup_counts, dispatch_token_idxs, idxs, expand_idx, send_meta)
