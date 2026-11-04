"""
Slot maps, row gathers and segment sums around the fused BF16 experts' grouped GEMMs. SM90 only:
callers check bf16_experts_supported first. Each function imports its SM90 implementation (cutlass)
on first call, so importing this module does not load cutlass.
"""

from typing import NamedTuple

import torch

__all__ = ["bf16_expert_gather_rows", "bf16_expert_gather_sum", "bf16_expert_group_slots", "bf16_expert_segment_sum"]


def bf16_expert_group_slots(expert_idxs: torch.Tensor, expand_idx: torch.Tensor | None, num_groups: int, num_tokens: int) -> NamedTuple:
    """
    Index maps of the grouped expert input (no rows are copied), as a record whose fields callers
    read and pass back to the bf16_expert_* kernels.

    - expert_idxs: int64 (S,) contiguous, the local expert of each token slot, values in [0,
      num_groups), 0 < num_groups <= 64 (asserted).
    - expand_idx: int64 (S,) contiguous, the received-token row (< num_tokens) each slot reads, or
      None for the identity (the received tokens are the slots).
    - Grouped rows are expert-major with no padding, stable (slot order) within each expert. Fields:
      cu_seqlens int32 (G + 1,), cumulative slot counts per expert; a_idx int32 (S,), the
      received-token row read by grouped row r; row_of int32 (S,), the grouped row of slot i;
      slot_of int32 (S,), its inverse; mask int64 (num_tokens,), per token the bitmask of its local
      experts, and table, an internal per-(token, expert) grouped-row table (mask and table are None
      without expand_idx).
    - Exact and deterministic; no host sync. S == 0 gives zero cu_seqlens (and mask).
    """
    from pithtrain.kernels.bf16_expert_slots_sm90 import bf16_expert_group_slots_sm90
    return bf16_expert_group_slots_sm90(expert_idxs, expand_idx, num_groups, num_tokens)


def bf16_expert_gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    x[idx] as a fresh bf16 (len(idx), N) exact row copy, for a contiguous bf16 (rows, N) x (N a
    multiple of 8) and int32 row indices idx.
    """
    from pithtrain.kernels.bf16_expert_slots_sm90 import bf16_expert_gather_rows_sm90
    return bf16_expert_gather_rows_sm90(x, idx)


def bf16_expert_gather_sum(grad: torch.Tensor, maps: NamedTuple, num_groups: int, max_slots: int) -> torch.Tensor:
    """
    The received tokens' gradient from the grouped input gradient: a fresh bf16 (num_tokens, H)
    whose row t sums the rows of grad of token t's slots, in fp32 in ascending local-expert order
    starting from zero, rounded once; a token with no slot gets zeros.

    - grad: bf16 (S, H) contiguous in grouped-row order, H a multiple of 8.
    - maps: from bf16_expert_group_slots with an expand_idx (mask and table required).
    - max_slots: an upper bound on any token's local experts (min(top_k, num_groups)).

    Deterministic.
    """
    from pithtrain.kernels.bf16_expert_slots_sm90 import bf16_expert_gather_sum_sm90
    return bf16_expert_gather_sum_sm90(grad, maps, num_groups, max_slots)


def bf16_expert_segment_sum(x: torch.Tensor, cu_host: list[int], cu: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
    """
    Per-segment column sums (the bias gradient of grouped rows): a fresh fp32 (G, N) whose row g
    sums rows r in [cu[g], cu[g + 1]) of x, read as x[idx[r]] when idx is given.

    - x: bf16 (rows, N) contiguous, N a multiple of 8. idx: optional int32 row index into x.
    - cu: int32 (G + 1,) on the device, and cu_host the same values as a host list (it sizes the
      work, so there is no device sync).
    - Order: each segment is cut into 256-row chunks from its start, each chunk summed row by row in
      fp32, then the chunk sums added in order; deterministic. An empty segment sums to zero.
    """
    from pithtrain.kernels.bf16_expert_slots_sm90 import bf16_expert_segment_sum_sm90
    return bf16_expert_segment_sum_sm90(x, cu_host, cu, idx)
