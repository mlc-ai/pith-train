"""
Expert-grouped token scatter for the grouped GEMMs, and an index gather that does not save its
input. The fused scatter places rows within a group in atomic order; deterministic=True (or the
Talker's canonical=True) instead keeps every group's rows in input order, so bf16 grouped-wgrad
sums do not depend on scheduling. That ordered placement is computed with stable integer sorts.
"""

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


def ordered_scatter(
    input_tokens: torch.Tensor,
    expert_idxs: torch.Tensor,
    num_groups: int,
    padding_alignment: int,
    group_counts: Optional[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    scatter_by_expert's contract with every group's rows in input order: a stable sort by expert
    gives each row its ordinal in its group. Rows are copied exactly; padding and the allocation
    tail up to a multiple of 1024 rows are zero.
    """
    num_tokens, hidden_size = input_tokens.shape[0], input_tokens.shape[1]
    device = input_tokens.device
    if expert_idxs.shape[0] != num_tokens:
        raise ValueError(f"{expert_idxs.shape[0]=} must equal the input row count {num_tokens=}")
    if num_tokens == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        output_tokens = torch.empty((0, hidden_size), device=device, dtype=input_tokens.dtype)
        return output_tokens, torch.empty((0,), device=device, dtype=torch.int64), empty, empty, []
    counts = torch.bincount(expert_idxs, minlength=num_groups)
    padded = (counts + padding_alignment - 1) // padding_alignment * padding_alignment
    ks_tensor = padded.to(torch.int32)
    grouped_mm_offs = padded.cumsum(0).to(torch.int32)
    if group_counts is not None:
        ks = [(c + padding_alignment - 1) // padding_alignment * padding_alignment for c in group_counts]  # fmt: skip
    else:
        ks = ks_tensor.tolist()
    total = sum(ks)
    order = torch.argsort(expert_idxs, stable=True)
    sorted_ids = expert_idxs[order]
    ordinal = torch.arange(num_tokens, device=device) - (counts.cumsum(0) - counts)[sorted_ids]
    positions = (padded.cumsum(0) - padded)[sorted_ids] + ordinal
    reverse_shuffle_idxs = torch.empty_like(order)
    reverse_shuffle_idxs[order] = positions
    buffer = input_tokens.new_zeros(((total + 1023) // 1024 * 1024, hidden_size))
    buffer.index_copy_(0, reverse_shuffle_idxs, input_tokens)
    return buffer[:total], reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks


class ScatterForGroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tokens,
        expert_idxs,
        num_groups,
        padding_alignment=128,
        group_counts=None,
        ordered=False,
    ):
        scatter = ordered_scatter if ordered else scatter_by_expert
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = scatter(
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
        return (grad_input_tokens,) + (None,) * (len(ctx.needs_input_grad) - 1)


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
    *,
    deterministic: bool = False,
    canonical: bool = False,
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
    device-to-host copy of the group sizes. deterministic=True keeps each group's rows in input
    order (it does not make an upstream EP dispatch or the GEMMs deterministic); canonical=True is
    the same placement, used by the Talker's independent expert slots.
    """
    output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = (
        ScatterForGroupedGemm.apply(
            input_tokens,
            expert_idxs,
            num_groups,
            padding_alignment,
            group_counts,
            deterministic or canonical,
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
