"""
Picker of group_bias_add: the SM90 implementation in group_bias_add_sm90, the TIRx kernel in
group_bias_add_sm100 on SM100.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["group_bias_add"]


def group_bias_add(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> None:
    """
    In place y[r] += bias[group_indices[r]] per row, without materializing bias[group_indices].

    y is (M, H) of any float dtype, bias (G, H), group_indices integer (M,) with values in [0, G).
    Each sum is computed in fp32 and rounded once to y's dtype, bit-identical to the PyTorch
    fallback y.copy_(y + bias[group_indices]). Deterministic.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.group_bias_add_sm90 import group_bias_add_sm90
            group_bias_add_sm90(y, bias, group_indices)
        case 10:
            from pithtrain.kernels.group_bias_add_sm100 import group_bias_add_sm100
            group_bias_add_sm100(y, bias, group_indices)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
