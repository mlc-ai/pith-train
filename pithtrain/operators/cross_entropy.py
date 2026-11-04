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
        if reduction == "sum" and torch.are_deterministic_algorithms_enabled():
            # Strict training uses PyTorch's fp32 log-softmax arithmetic. The fused path is
            # mathematically equivalent, but a bf16 rounding boundary can change the sign of a
            # near-zero downstream gradient. The destructive API and the saved gradient's dtype stay
            # the same; this opt-in path needs transient fp32 vocabulary memory.
            if inp.stride(-1) != 1 or inp.stride(-2) != inp.shape[-1]:
                inp = inp.contiguous()
            with torch.enable_grad():
                logits = inp.detach().float().requires_grad_()
                loss = torch.nn.functional.cross_entropy(
                    logits, target, ignore_index=ignore_index, reduction="sum"
                )
                gradient = torch.autograd.grad(loss, logits)[0]
            inp.copy_(gradient)
            ctx.save_for_backward(inp.detach())
            return loss.detach()
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
    Cross-entropy of logits inp (N, V) against target (N,), as a float32 scalar; targets equal
    to ignore_index are left out. reduction "mean" (default) divides by the number of counted
    targets; "sum" does not, for token-weighted training that normalizes the accumulated fp32
    gradients once after the data-parallel reduction. Computes in fp32 and, when inp is
    row-contiguous, overwrites it in place with its gradient, in inp's dtype. With PyTorch
    deterministic algorithms enabled, "sum" uses PyTorch's fp32 cross-entropy arithmetic instead of
    the fused kernel, with transient fp32 vocabulary memory.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    return CrossEntropy.apply(inp, target, ignore_index, reduction)
