import torch

from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100, kmajor_empty_outputs
from pithtrain.kernels.silu_mul import silu_mul_bwd, silu_mul_fwd

__all__ = ["fp8_silu_mul_gu_bwd_sm100", "fp8_silu_mul_bwd_sm100", "fp8_silu_mul_fwd_sm100", "fp8_silu_mul_gu_fwd_sm100"]


def fp8_silu_mul_fwd_sm100(g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = g.shape
    if M == 0 or N == 0:
        return kmajor_empty_outputs(M, N, g.device)
    return fp8cast_rowwise_kmajor_sm100(silu_mul_fwd(g, u), grouped_mm_offs)


def fp8_silu_mul_bwd_sm100(dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    M, N = g.shape
    if M == 0 or N == 0:
        return kmajor_empty_outputs(M, N, g.device), kmajor_empty_outputs(M, N, g.device)
    grad_gate, grad_up = silu_mul_bwd(dh, g, u)
    return (fp8cast_rowwise_kmajor_sm100(grad_gate, grouped_mm_offs), fp8cast_rowwise_kmajor_sm100(grad_up, grouped_mm_offs))


def fp8_silu_mul_gu_fwd_sm100(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return kmajor_empty_outputs(0, N, gu.device)
    gate, up = gu[:, :N].contiguous(), gu[:, N:].contiguous()
    return fp8cast_rowwise_kmajor_sm100(silu_mul_fwd(gate, up), grouped_mm_offs)


def fp8_silu_mul_gu_bwd_sm100(dh: torch.Tensor, gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return kmajor_empty_outputs(0, 2 * N, gu.device)
    gate, up = gu[:, :N].contiguous(), gu[:, N:].contiguous()
    grad_gate, grad_up = silu_mul_bwd(dh, gate, up)
    return fp8cast_rowwise_kmajor_sm100(torch.cat([grad_gate, grad_up], 1), grouped_mm_offs)
