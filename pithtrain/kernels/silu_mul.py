"""
Picker of silu_mul, the fused silu(gate) * up of a SwiGLU MLP and its gradients. In eager SM90
runs the CuTe-DSL kernels (silu_mul_sm90) and SM100 the TIRx kernels (silu_mul_sm100); inside a
torch.compile region, which can trace neither launch, both emit the same fp32 formulas as plain
PyTorch for Inductor to fuse with the neighbouring ops (a cheaper launch than an opaque op, and no
graph break). All compute in fp32 and round once to the dtype; the eager kernels of both
architectures give the same bf16 bytes, the compiled path may differ from them by an output ulp.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["silu_mul_bwd", "silu_mul_fwd"]


def silu_mul_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    silu(gate) * up as a new contiguous tensor like gate. gate, up: contiguous, same shape and float
    dtype (bf16 in practice); not modified. Computed in fp32 and rounded once to the dtype; close
    to, not bit-identical with, the PyTorch fp32 formula (Triton's sigmoid). Deterministic;
    fp8_silu_mul_fwd reproduces it bit for bit outside torch.compile (inside a compiled region both
    architectures run Inductor's code for the same formula, within an output ulp). On SM90 fewer
    than 2^31 elements. Callable inside torch.compile regions; runs on the current stream.
    """
    if torch.compiler.is_compiling():
        # traced once; eager calls get these checks from the kernels
        assert gate.shape == up.shape and gate.dtype == up.dtype, f"mismatch: {gate.shape} {gate.dtype} vs {up.shape} {up.dtype}"
        assert gate.is_contiguous() and up.is_contiguous(), "gate and up must be contiguous"
        gate_f32 = gate.float()
        return (gate_f32 * torch.sigmoid(gate_f32) * up.float()).to(gate.dtype)
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
    shape and dtype; not modified. Computed in fp32 (silu(gate) recomputed from gate), each rounded
    once to the dtype. Deterministic; fp8_silu_mul_bwd reproduces both bit for bit outside
    torch.compile (inside a compiled region both architectures run Inductor's code for the same
    formulas, within an output ulp). On SM90 fewer than 2^31 elements. Callable inside torch.compile
    regions; runs on the current stream.
    """
    if torch.compiler.is_compiling():
        # traced once; eager calls get these checks from the kernels
        assert grad_output.shape == gate.shape == up.shape, f"shape mismatch: {grad_output.shape}, {gate.shape}, {up.shape}"
        assert grad_output.dtype == gate.dtype == up.dtype, f"dtype mismatch: {grad_output.dtype}, {gate.dtype}, {up.dtype}"
        assert grad_output.is_contiguous() and gate.is_contiguous() and up.is_contiguous(), "grad_output, gate and up must be contiguous"
        grad_f32, gate_f32, up_f32 = grad_output.float(), gate.float(), up.float()
        gate_sigmoid = torch.sigmoid(gate_f32)
        gate_silu = gate_f32 * gate_sigmoid
        grad_gate, grad_up = grad_f32 * up_f32 * gate_sigmoid * (1.0 + gate_f32 - gate_silu), grad_f32 * gate_silu
        return grad_gate.to(gate.dtype), grad_up.to(up.dtype)
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.silu_mul_sm90 import silu_mul_bwd_sm90
            return silu_mul_bwd_sm90(grad_output, gate, up)
        case 10:
            from pithtrain.kernels.silu_mul_sm100 import silu_mul_bwd_sm100
            return silu_mul_bwd_sm100(grad_output, gate, up)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
