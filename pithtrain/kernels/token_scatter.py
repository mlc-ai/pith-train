from typing import Optional

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["GEMM_ALLOC_ALIGNMENT", "scatter_by_expert"]

GEMM_ALLOC_ALIGNMENT = 1024


def scatter_by_expert(input_tokens: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, padding_alignment: int = 128, group_counts: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Reorder input_tokens by expert_idxs into expert-major groups, each padded with zero rows to
    a multiple of padding_alignment. Returns (output_tokens, reverse_shuffle_idxs,
    grouped_mm_offs, ks_tensor, ks).

    - input_tokens: (num_tokens, hidden) with unit inner stride (any row stride, any dtype).
      expert_idxs: int64 (num_tokens,) contiguous, values in [0, num_groups), length num_tokens.
      Inputs are not modified.
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

    Rows are copied exactly. Placement within a group: token order on SM90 (so every output is
    deterministic); unspecified on SM100 (atomic_add order, not deterministic). Consumers locate a
    token only through reverse_shuffle_idxs. An expert_idxs length other than num_tokens raises
    ValueError on SM90 (also for num_tokens == 0); SM100 does not check it.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.token_scatter_sm90 import scatter_by_expert_sm90
            return scatter_by_expert_sm90(input_tokens, expert_idxs, num_groups, padding_alignment, group_counts)
        case 10:
            from pithtrain.kernels.token_scatter_sm100 import scatter_by_expert_sm100
            return scatter_by_expert_sm100(input_tokens, expert_idxs, num_groups, padding_alignment, group_counts)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")

