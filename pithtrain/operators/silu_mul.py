"""
silu(gate) is not saved for backward; silu_mul_bwd recomputes it from gate to save memory.
"""

from __future__ import annotations

import torch

from pithtrain.kernels import silu_mul_bwd, silu_mul_fwd

__all__ = ["silu_mul"]


class SiLUMulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        out = silu_mul_fwd(gate, up)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        return silu_mul_bwd(grad_output, gate, up)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Element-wise silu(gate) * up of contiguous tensors of the same shape and dtype, in that dtype.
    """
    return SiLUMulFunc.apply(gate, up)
