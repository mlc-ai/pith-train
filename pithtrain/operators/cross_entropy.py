# Adapted from NVIDIA TransformerEngine (Apache 2.0)
# https://github.com/NVIDIA/TransformerEngine
#   transformer_engine/common/triton/cross_entropy.py  (Triton kernels)
#   transformer_engine/pytorch/triton/cross_entropy.py  (autograd wrapper)
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modifications: fused online-softmax + cross-entropy into a single kernel,
# removed tensor-parallelism / label-smoothing / distributed support.

import torch
import torch.nn.functional as F

from pithtrain.kernels import cross_entropy_fwd_bwd

__all__ = ["cross_entropy"]


def fp32_cross_entropy_sum_(
    inp: torch.Tensor, target: torch.Tensor, ignore_index: int, chunk_bytes: int | None = None
) -> torch.Tensor:
    """
    Summed cross-entropy of the row-contiguous logits inp (N, V) in PyTorch's fp32 arithmetic,
    overwriting inp with its gradient rounded once to inp's dtype; returns the fp32 loss. chunk_bytes
    bounds each fp32 (rows, V) temporary by taking the rows in chunks, which changes the order of
    the loss's sum but no row's gradient; None takes all rows at once.
    """
    num_rows, num_cols = inp.shape
    chunk_rows = num_rows if chunk_bytes is None else chunk_bytes // (4 * max(num_cols, 1))
    chunk_rows = max(chunk_rows, 1)
    loss = torch.zeros((), dtype=torch.float32, device=inp.device)
    for start in range(0, num_rows, chunk_rows):
        rows = slice(start, start + chunk_rows)
        with torch.enable_grad():
            logits = inp[rows].detach().float().requires_grad_()
            chunk_loss = F.cross_entropy(
                logits, target[rows], ignore_index=ignore_index, reduction="sum"
            )
            gradient = torch.autograd.grad(chunk_loss, logits)[0]
        inp[rows].copy_(gradient)
        loss += chunk_loss.detach()
    return loss


class CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, ignore_index, reduction):
        if reduction == "sum":
            # The kernels divide the gradient by the target count before rounding it, and a later
            # multiply by the count would round it again. Strict training (deterministic
            # algorithms) takes all rows at once; otherwise each fp32 temporary is at most 256 MiB.
            if inp.stride(-1) != 1 or inp.stride(-2) != inp.shape[-1]:
                inp = inp.contiguous()
            deterministic = torch.are_deterministic_algorithms_enabled()
            chunk_bytes = None if deterministic else 1 << 28
            loss = fp32_cross_entropy_sum_(inp, target, ignore_index, chunk_bytes)
        else:
            loss, inp = cross_entropy_fwd_bwd(inp, target, ignore_index)
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
    row-contiguous, overwrites it in place with its gradient, in inp's dtype. "mean" runs the fused
    kernel; "sum" runs PyTorch's fp32 cross-entropy, all rows at once under deterministic
    algorithms and otherwise in row chunks of bounded fp32 memory.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    return CrossEntropy.apply(inp, target, ignore_index, reduction)
