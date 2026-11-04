from typing import Optional

import torch

from pithtrain.contexts import training
from pithtrain.kernels import fp8_group_indices, scatter_by_expert

__all__ = [
    "PaddedIndexGather",
    "ScatterForGroupedGemm",
    "padded_index_gather",
    "precompute_group_indices",
    "scatter_for_grouped_gemm",
]


class ScatterForGroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input_tokens, expert_idxs, num_groups, padding_alignment=128, group_counts=None
    ):
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = scatter_by_expert(
            input_tokens, expert_idxs, num_groups, padding_alignment, group_counts
        )
        ctx.save_for_backward(reverse_shuffle_idxs)
        return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks

    @staticmethod
    def backward(
        ctx,
        grad_output_tokens,
        grad_reverse_shuffle_idxs,
        grad_grouped_mm_offs,
        grad_ks_tensor,
        grad_ks,
    ):
        (reverse_shuffle_idxs,) = ctx.saved_tensors
        grad_input_tokens = grad_output_tokens[reverse_shuffle_idxs]
        return grad_input_tokens, None, None, None, None


class PaddedIndexGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, index: torch.Tensor, pad_to_multiple: int) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.input_shape = x.shape
        ctx.pad_to_multiple = pad_to_multiple
        actual = index.shape[0]
        padded = (actual + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple
        buf = x.new_empty(padded, *x.shape[1:])
        torch.index_select(x, 0, index, out=buf[:actual])
        return buf[:actual]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (index,) = ctx.saved_tensors
        rows = ctx.input_shape[0]
        padded_rows = (rows + ctx.pad_to_multiple - 1) // ctx.pad_to_multiple * ctx.pad_to_multiple
        grad_buf = torch.zeros(
            padded_rows,
            *ctx.input_shape[1:],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        grad_buf[:rows].scatter_add_(0, index.unsqueeze(-1).expand_as(grad_output), grad_output)
        return grad_buf[:rows], None, None


def padded_index_gather(
    x: torch.Tensor, index: torch.Tensor, pad_to_multiple: int = 1024
) -> torch.Tensor:
    """
    x[index] for a 1-D index, without saving x for backward. The output is a view of a buffer
    padded to a multiple of pad_to_multiple rows, so varying token counts reuse allocator blocks.
    """
    return PaddedIndexGather.apply(x, index, pad_to_multiple)


def scatter_for_grouped_gemm(
    input_tokens: torch.Tensor,
    expert_idxs: torch.Tensor,
    num_groups: int,
    padding_alignment: int = 128,
    group_counts: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """
    Scatter input_tokens (m, hidden) into expert groups by expert_idxs (m,) int64, each group's
    rows zero-padded to a multiple of padding_alignment. Returns (output_tokens,
    reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor):

    - output_tokens: (grouped_mm_offs[-1], hidden), the tokens grouped by expert.
    - reverse_shuffle_idxs: (m,) int64, the row of output_tokens each input token went to.
    - grouped_mm_offs: (num_groups,) int32, cumulative padded group ends.
    - ks, ks_tensor: the padded group sizes, as a host list and a (num_groups,) int32 tensor.

    group_counts, the rows per group on the host if the caller has them, saves the blocking
    device-to-host copy of the group sizes.
    """
    output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = (
        ScatterForGroupedGemm.apply(
            input_tokens, expert_idxs, num_groups, padding_alignment, group_counts
        )
    )
    return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor


def precompute_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> Optional[torch.Tensor]:
    """
    Group index of each of the M rows, computed once for all grouped FP8 projections. None wherever
    the FP8 grouped GEMM takes no such hint.
    """
    if training.fp8:
        return fp8_group_indices(grouped_mm_offs, M)
    return None
