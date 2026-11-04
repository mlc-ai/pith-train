from typing import Optional

import torch

from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100
from pithtrain.kernels.token_scatter import scatter_by_expert

__all__ = ["scatter_quant_sm100"]


def scatter_quant_sm100(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor] = None, padding_alignment: int = 128, group_counts: Optional[list[int]] = None) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    assert padding_alignment == 128, "the K-major FP8 layout needs 128-row groups"
    x = gathered if expand_idx is None else gathered[expand_idx]
    output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = scatter_by_expert(x, expert_idxs, num_groups, group_counts=group_counts)
    return fp8cast_rowwise_kmajor_sm100(output_tokens, grouped_mm_offs), reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor
