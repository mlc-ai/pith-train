"""
Picker of silu_mul_fwd / silu_mul_bwd: the Triton kernels in silu_mul_sm90, the TIRx kernels in
silu_mul_sm100 on SM100 (the same bits).
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["silu_mul_bwd", "silu_mul_fwd"]


def silu_mul_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    silu(gate) * up as a new contiguous tensor like gate. gate, up: contiguous, same shape and float
    dtype (bf16 in practice); not modified. Computed in fp32 and rounded once to the dtype; close
    to, not bit-identical with, the PyTorch fp32 formula (Triton's sigmoid). Deterministic;
    fp8_silu_mul_fwd reproduces it bit for bit.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.silu_mul_sm90 import silu_mul_fwd_sm90
            return silu_mul_fwd_sm90(gate, up)
        case 10:
            from pithtrain.kernels.silu_mul_sm100 import silu_mul_fwd_sm100
            return silu_mul_fwd_sm100(gate, up)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def silu_mul_bwd(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients of silu_mul_fwd as new (grad_gate, grad_up). grad_output, gate, up: contiguous, same
    shape and dtype; not modified. Computed in fp32, each rounded once to the dtype. Deterministic;
    fp8_silu_mul_bwd reproduces both bit for bit.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.silu_mul_sm90 import silu_mul_bwd_sm90
            return silu_mul_bwd_sm90(grad_output, gate, up)
        case 10:
            from pithtrain.kernels.silu_mul_sm100 import silu_mul_bwd_sm100
            return silu_mul_bwd_sm100(grad_output, gate, up)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
