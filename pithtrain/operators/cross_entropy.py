# Adapted from NVIDIA TransformerEngine (Apache 2.0)
# https://github.com/NVIDIA/TransformerEngine
#   transformer_engine/common/triton/cross_entropy.py  (Triton kernels)
#   transformer_engine/pytorch/triton/cross_entropy.py  (autograd wrapper)
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modifications: fused online-softmax + cross-entropy into a single kernel,
# removed tensor-parallelism / label-smoothing / distributed support.

import torch

from pithtrain.kernels import cross_entropy_fwd_bwd

__all__ = ["cross_entropy"]


class CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, ignore_index, reduction):
        loss, inp = cross_entropy_fwd_bwd(inp, target, ignore_index, reduction)
        ctx.save_for_backward(inp.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        inp.mul_(grad_output.to(inp.dtype))
        return inp, None, None, None


def cross_entropy(
    inp: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Mean (default) or summed cross-entropy of logits inp (N, V) against target (N,), as a float32 scalar; targets equal
    to ignore_index are left out. Computes in fp32 and, when inp is row-contiguous, overwrites it in
    place with its gradient, in inp's dtype. Use sum reduction before normalizing accumulated
    gradients to avoid rounding a low-precision mean before multiplying by the token count.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    return CrossEntropy.apply(inp, target, ignore_index, reduction)
