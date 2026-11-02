"""
Picker of clamped_swiglu_fwd / clamped_swiglu_bwd: the Triton kernels in clamped_swiglu_sm90, the
TIRx kernels in clamped_swiglu_sm100 on SM100 (the same bits).
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["clamped_swiglu_bwd", "clamped_swiglu_fwd"]


def clamped_swiglu_fwd(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Clamped SwiGLU of an interleaved gate_up: a new contiguous (M, N) tensor in gate_up's dtype,
    (up_clamped + 1) * gate_clamped * sigmoid(alpha * gate_clamped) with gate = gate_up[:, 0::2], up
    = gate_up[:, 1::2], gate_clamped = min(gate, limit), up_clamped = clamp(up, -limit, limit).

    gate_up is contiguous (M, 2N), bf16 (any float dtype). alpha and limit are Python floats
    (gpt-oss: 1.702, 7.0). Computed in fp32 and rounded once to the input dtype. Deterministic.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.clamped_swiglu_sm90 import clamped_swiglu_fwd_sm90
            return clamped_swiglu_fwd_sm90(gate_up, alpha, limit)
        case 10:
            from pithtrain.kernels.clamped_swiglu_sm100 import clamped_swiglu_fwd_sm100
            return clamped_swiglu_fwd_sm100(gate_up, alpha, limit)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def clamped_swiglu_bwd(grad_output: torch.Tensor, gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Gradient of clamped_swiglu_fwd with respect to the interleaved gate_up: a new (M, 2N) tensor
    like gate_up, gate gradients at even and up gradients at odd columns.

    grad_output is contiguous (M, N) in gate_up's dtype; gate_up is the saved contiguous (M, 2N)
    forward input. The gate gradient is zero where gate > limit, the up gradient zero where |up| >
    limit (the boundary itself passes the gradient, as torch.clamp does). fp32 arithmetic, rounded
    once to the dtype. Deterministic.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.clamped_swiglu_sm90 import clamped_swiglu_bwd_sm90
            return clamped_swiglu_bwd_sm90(grad_output, gate_up, alpha, limit)
        case 10:
            from pithtrain.kernels.clamped_swiglu_sm100 import clamped_swiglu_bwd_sm100
            return clamped_swiglu_bwd_sm100(grad_output, gate_up, alpha, limit)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
