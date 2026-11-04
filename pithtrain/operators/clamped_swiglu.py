"""
Fused clamped SwiGLU of gpt-oss, (up_c + 1) * gate_c * sigmoid(alpha * gate_c) with gate_c =
min(gate, limit) and up_c = clamp(up, -limit, limit), where gate and up are the even and odd
columns of gate_up. Only gate_up is saved for backward; the clamped branches get zero
gradient at saturation.
"""

from __future__ import annotations

import torch

from pithtrain.kernels import clamped_swiglu_bwd, clamped_swiglu_fwd

__all__ = ["clamped_swiglu"]


class ClampedSwiGLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
        out = clamped_swiglu_fwd(gate_up, alpha, limit)
        ctx.save_for_backward(gate_up)
        ctx.alpha = float(alpha)
        ctx.limit = float(limit)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gate_up,) = ctx.saved_tensors
        return clamped_swiglu_bwd(grad_output, gate_up, ctx.alpha, ctx.limit), None, None


def clamped_swiglu(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Clamped SwiGLU of contiguous gate_up (M, 2N), returning (M, N) in gate_up's dtype.
    """
    return ClampedSwiGLUFunc.apply(gate_up, alpha, limit)
