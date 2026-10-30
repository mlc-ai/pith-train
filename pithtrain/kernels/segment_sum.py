"""
Picker of segment_sum: the SM90 implementation in segment_sum_sm90, the TIRx kernel in
segment_sum_sm100 on SM100.

Per-group column sums over rows sorted by group, deterministic and atomic-free.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["segment_sum"]


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
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.segment_sum_sm90 import segment_sum_sm90
            return segment_sum_sm90(x, grouped_mm_offs, out_dtype)
        case 10:
            from pithtrain.kernels.segment_sum_sm100 import segment_sum_sm100
            return segment_sum_sm100(x, grouped_mm_offs, out_dtype)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
