"""
x + bias[group_ids] for rows sorted by expert, with an atomic-free backward: the bias gradient is
a per-group row sum (segment_sum) instead of autograd's index_put_ with bf16 atomic
adds, which dominates GPU time for gpt-oss's expert biases.
"""

import torch

from pithtrain.kernels import segment_sum

__all__ = ["indexed_bias_add"]


class IndexedBiasAddFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, group_ids, grouped_mm_offs):
        ctx.bias_dtype = bias.dtype
        ctx.bias_shape = bias.shape
        ctx.save_for_backward(grouped_mm_offs)
        return x + bias[group_ids]

    @staticmethod
    def backward(ctx, grad_out):
        (offs,) = ctx.saved_tensors
        if grad_out.shape[0] == 0 or offs.numel() == 0:
            # An EP rank that received no tokens, or has no groups.
            bias_grad = grad_out.new_zeros(ctx.bias_shape, dtype=ctx.bias_dtype)
            return grad_out, bias_grad, None, None
        grad_out_contig = grad_out.contiguous()
        bias_grad = segment_sum(grad_out_contig, offs, ctx.bias_dtype)
        return grad_out, bias_grad, None, None


def indexed_bias_add(
    x: torch.Tensor,
    bias: torch.Tensor,
    group_ids: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> torch.Tensor:
    """
    x (M, H) + bias[group_ids], bias (G, H). group_ids must be non-decreasing and grouped_mm_offs
    the matching cumulative end offsets of the groups, the last equal to M.
    """
    return IndexedBiasAddFunc.apply(x, bias, group_ids, grouped_mm_offs)
