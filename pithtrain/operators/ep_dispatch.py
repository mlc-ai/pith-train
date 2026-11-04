"""
Expert-parallel dispatch: from the router's top-k to the deduplicated tokens to send and the routing
metadata of stages 3-5 (RoutingInfo). prepare_dispatch_issue and PendingDispatch.finish split it in
two, so that independent GPU work can be enqueued while the metadata is copied to the host.

deterministic=True builds the sender metadata with stable integer sorts, so every bucket is in
ascending token order on any architecture (the index kernel uses atomic order off SM90).
deduplicate=False sends already-expanded (tokens, K, hidden) slots one by one, in expert order
with ties in slot order, keeping each slot's backward lane independent (the Qwen3-Omni Talker
joins its expert and shared-branch bf16 derivatives at the source).
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist

from pithtrain.kernels import ep_dispatch_recv_idx, ep_dispatch_send_idx
from pithtrain.models.interface import AllToAllSplits, RoutingInfo
from pithtrain.operators.all_to_all import all_to_all_sync
from pithtrain.operators.stable_scatter import stable_dedup_prepare_dispatch
from pithtrain.operators.token_scatter import padded_index_gather

__all__ = [
    "PendingDispatch",
    "prepare_dispatch",
    "prepare_dispatch_issue",
    "prepare_independent_dispatch",
]

# At most one deferred dispatch is outstanding per process: its metadata lands in the single pinned
# buffer below, which the next issue would overwrite.
outstanding_dispatches: list["PendingDispatch"] = []


@lru_cache(maxsize=None)
def pinned_meta_buffer(numel: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Cached because freeing a pinned tensor records a CUDA event in PyTorch's host allocator.
    """
    return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)


@dataclass(slots=True, eq=False)
class PendingDispatch:
    hidden_states: torch.Tensor
    topk_weight: torch.Tensor
    dispatch_token_idxs: torch.Tensor
    idxs: torch.Tensor
    expand_idx: torch.Tensor
    meta: torch.Tensor
    meta_host: torch.Tensor
    meta_copied: torch.cuda.Event
    ep_size: int
    experts_per_rank: int
    ep_group: dist.ProcessGroup
    is_finished: bool = False

    @torch.compiler.disable
    def finish(self) -> tuple[torch.Tensor, RoutingInfo]:
        assert not self.is_finished, "finish() called twice on the same pending dispatch"
        self.is_finished = True
        assert outstanding_dispatches and outstanding_dispatches[-1] is self
        ep_size, experts_per_rank, ep_group = self.ep_size, self.experts_per_rank, self.ep_group
        self.meta_copied.synchronize()
        host_counts = self.meta_host.tolist()
        outstanding_dispatches.pop()
        stride = experts_per_rank + 1
        input_splits = host_counts[:ep_size]
        dedup_input_splits = host_counts[ep_size : 2 * ep_size]
        recv_counts = host_counts[2 * ep_size :]
        dedup_output_splits = recv_counts[experts_per_rank::stride]
        # Stage 3 groups its GEMMs by expert_counts, so its scatter need not sync to learn them.
        output_splits = [sum(recv_counts[sender * stride : sender * stride + experts_per_rank]) for sender in range(ep_size)]  # fmt: skip
        expert_counts = [
            sum(recv_counts[expert::stride][:ep_size]) for expert in range(experts_per_rank)
        ]

        # One entry per received token slot: the senders decide the row count, not this rank.
        expand_idx = self.expand_idx
        received_expand_idx = expand_idx.new_empty(sum(output_splits))
        all_to_all_sync(received_expand_idx, expand_idx, output_splits, input_splits, ep_group)
        recv_meta = self.meta[2 * ep_size :]
        expert_idxs, expand_idx = ep_dispatch_recv_idx(
            recv_meta, received_expand_idx, ep_size, experts_per_rank
        )

        num_dedup_tokens = sum(dedup_input_splits)
        dedup_token_idxs = self.dispatch_token_idxs[:num_dedup_tokens]
        dispatch_tokens = padded_index_gather(self.hidden_states, dedup_token_idxs)

        return dispatch_tokens, RoutingInfo(
            self.topk_weight,
            expert_idxs,
            self.idxs,
            expand_idx,
            AllToAllSplits(dedup_input_splits, dedup_output_splits),
            AllToAllSplits(input_splits, output_splits),
            expert_counts,
        )


@torch.compiler.disable
def prepare_dispatch_issue(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    ep_group: dist.ProcessGroup,
    *,
    deterministic: bool = False,
) -> PendingDispatch:
    """
    Issue the device side of an expert-parallel dispatch on the current stream without
    blocking the host.
    """
    assert ep_size > 1
    assert not outstanding_dispatches, "a deferred dispatch is still outstanding"
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # One int64 buffer holds everything the host needs, so one device-to-host copy fetches it:
    # [tokens_per_ep_rank (ep_size) | dedup_tokens_per_gpu (ep_size) | recv_meta (ep_size *
    # (experts_per_rank + 1))], the last being, per sender, its token count for each local expert
    # followed by its dedup count. The kernel writes the first two parts in place.
    stride = experts_per_rank + 1
    meta = torch.empty(2 * ep_size + ep_size * stride, dtype=torch.int64, device=topk_ids.device)
    if deterministic:
        (
            tokens_per_ep_rank,
            dedup_tokens_per_gpu,
            dispatch_token_idxs,
            idxs,
            expand_idx,
            send_meta,
        ) = stable_dedup_prepare_dispatch(topk_ids, num_experts, ep_size, experts_per_rank)
        meta[:ep_size].copy_(tokens_per_ep_rank)
        meta[ep_size : 2 * ep_size].copy_(dedup_tokens_per_gpu)
    else:
        _, _, dispatch_token_idxs, idxs, expand_idx, send_meta = ep_dispatch_send_idx(
            topk_ids, num_experts, ep_size, experts_per_rank, meta
        )

    recv_meta = meta[2 * ep_size :]
    all_to_all_sync(recv_meta, send_meta, group=ep_group)

    meta_host = pinned_meta_buffer(meta.shape[0], meta.dtype)
    meta_copied = torch.cuda.Event()
    meta_host.copy_(meta, non_blocking=True)
    meta_copied.record()

    pending = PendingDispatch(
        hidden_states, topk_weight, dispatch_token_idxs, idxs, expand_idx, meta, meta_host,
        meta_copied, ep_size, experts_per_rank, ep_group,
    )  # fmt: skip
    outstanding_dispatches.append(pending)
    return pending


@torch.compiler.disable
def prepare_independent_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    ep_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, RoutingInfo]:
    """
    Send every already-expanded (tokens, K, hidden) slot once, without deduplication, in expert
    order with ties in slot order. idxs is a permutation of the slots, so its backward never adds
    two expert derivatives before the source.
    """
    assert ep_size > 1
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_ids = topk_ids.reshape(-1)
    counts = torch.bincount(flat_ids, minlength=num_experts)
    recv_counts = torch.empty_like(counts)
    all_to_all_sync(recv_counts, counts, group=ep_group)
    tokens_per_ep_rank = counts.view(ep_size, experts_per_rank).sum(dim=1)
    host = torch.cat((tokens_per_ep_rank, recv_counts)).tolist()
    input_splits, recv = host[:ep_size], host[ep_size:]
    output_splits = [sum(recv[s * experts_per_rank : (s + 1) * experts_per_rank]) for s in range(ep_size)]  # fmt: skip
    expert_counts = [sum(recv[expert::experts_per_rank]) for expert in range(experts_per_rank)]
    total = sum(output_splits)
    idxs = torch.argsort(flat_ids, stable=True)
    local_experts = torch.arange(experts_per_rank, device=flat_ids.device).repeat(ep_size)
    expert_idxs = torch.repeat_interleave(local_experts, recv_counts, output_size=total)
    dispatch_tokens = padded_index_gather(hidden_states, idxs)
    splits = AllToAllSplits(input_splits, output_splits)
    return dispatch_tokens, RoutingInfo(
        topk_weight,
        expert_idxs,
        idxs,
        torch.arange(total, device=flat_ids.device),
        splits,
        splits,
        expert_counts,
    )


@torch.compiler.disable
def prepare_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    ep_group: dist.ProcessGroup,
    *,
    deterministic: bool = False,
    deduplicate: bool = True,
) -> tuple[torch.Tensor, RoutingInfo]:
    """
    With ep_size > 1 a token sent to several experts on one EP rank is sent once, and expand_idx
    restores the copies on the receiver. deterministic and deduplicate=False are described in the
    module docstring; deduplicate=False takes hidden_states shaped (tokens, K, hidden).
    """
    k = topk_ids.shape[1]
    if not deduplicate and (hidden_states.ndim != 3 or hidden_states.shape[:2] != topk_ids.shape):
        raise ValueError("Independent dispatch lanes must have shape (tokens, K, hidden)")
    expert_idxs = topk_ids.view(-1)

    if ep_size == 1:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        dispatch_tokens = (
            hidden_states.unsqueeze(1).expand(-1, k, -1).reshape(-1, hidden_states.shape[-1])
            if deduplicate
            else hidden_states
        )
        return dispatch_tokens, RoutingInfo(topk_weight, expert_idxs)

    if not deduplicate:
        return prepare_independent_dispatch(
            hidden_states, topk_ids, topk_weight, num_experts, ep_size, experts_per_rank, ep_group
        )

    pending = prepare_dispatch_issue(
        hidden_states, topk_ids, topk_weight, num_experts, ep_size, experts_per_rank, ep_group,
        deterministic=deterministic,
    )  # fmt: skip
    return pending.finish()
