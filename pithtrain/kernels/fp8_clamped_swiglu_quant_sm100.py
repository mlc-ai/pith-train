"""
Unfused SM100 implementation of fp8_clamped_swiglu_quant: the biased gate_up, the Triton
clamped_swiglu kernels, then fp8cast_rowwise_kmajor_sm100 (and block_segment_sum for the bias
gradient). Scales are row-major; kmajor_channels is ignored.
"""

import torch

from pithtrain.kernels.clamped_swiglu import clamped_swiglu_bwd, clamped_swiglu_fwd
from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100, kmajor_empty_outputs
from pithtrain.kernels.segment_sum import block_segment_sum

__all__ = ["biased_gate_up", "clamped_swiglu_bwd_biased", "fp8_clamped_swiglu_bwd_sm100", "clamped_swiglu_quant_fwd_sm100", "fp8_clamped_swiglu_kmajor_sm100", "fp8_clamped_swiglu_rowwise_sm100"]

BLOCK = 128


def find_row_groups(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor:
    """
    The group of each of the M rows, int64 (M,); rows past the last group end take the last group.
    """
    row_idx = torch.arange(M, device=grouped_mm_offs.device, dtype=torch.int64)
    group_ends = grouped_mm_offs.to(torch.int64)
    return torch.searchsorted(group_ends, row_idx, right=True).clamp_(max=group_ends.shape[0] - 1)


def biased_gate_up(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor) -> torch.Tensor:
    return gate_up + bias[find_row_groups(grouped_mm_offs, gate_up.shape[0])]


def clamped_swiglu_bwd_biased(dh: torch.Tensor, gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    bf16 d(gate_up) of clamped_swiglu_bwd at gate_up + bias, dh cast to its dtype as autograd does.
    """
    x = biased_gate_up(gate_up, bias, grouped_mm_offs)
    return clamped_swiglu_bwd(dh.to(x.dtype), x, alpha, limit)


def clamped_swiglu_quant_fwd_sm100(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0:
        return kmajor_empty_outputs(M, N, gate_up.device)
    activation = clamped_swiglu_fwd(biased_gate_up(gate_up, bias, grouped_mm_offs), alpha, limit)
    return fp8cast_rowwise_kmajor_sm100(activation, grouped_mm_offs)


def fp8_clamped_swiglu_rowwise_sm100(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    return clamped_swiglu_quant_fwd_sm100(gate_up, bias, grouped_mm_offs, alpha, limit)[:2]


def fp8_clamped_swiglu_kmajor_sm100(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    return clamped_swiglu_quant_fwd_sm100(gate_up, bias, grouped_mm_offs, alpha, limit)[2:]


def fp8_clamped_swiglu_bwd_sm100(dh: torch.Tensor, gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0:
        return kmajor_empty_outputs(M, 2 * N, gate_up.device), bias.new_zeros(bias.shape)
    grad_gate_up = clamped_swiglu_bwd_biased(dh, gate_up, bias, grouped_mm_offs, alpha, limit)
    block_sums = grad_gate_up.float().view(M // BLOCK, BLOCK, 2 * N).sum(1)
    return fp8cast_rowwise_kmajor_sm100(grad_gate_up, grouped_mm_offs), block_segment_sum(block_sums, grouped_mm_offs, bias.dtype)
