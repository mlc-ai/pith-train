from typing import Optional

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["scatter_quant"]


def scatter_quant(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor] = None, padding_alignment: int = 128, group_counts: Optional[list[int]] = None) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """
    The MoE stage-3 FP8 input: x = gathered[expand_idx] scattered by expert as scatter_by_expert
    does, then quantized as fp8_quant_grouped does on SM90. Returns ((x_fp8, x_scale, x_ch_fp8,
    x_ch_scale), reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor).

    - gathered: bf16 (m_src, N) received (deduplicated) tokens. expert_idxs: int64 (m,), the
      local expert (< num_groups) of each token slot. expand_idx: int64 (m,), the row of gathered
      of each slot, or None for the identity (then m_src must equal m, else ValueError). Inputs
      are not modified.
    - padding_alignment must be 128. group_counts: the true per-expert slot counts on the host, if
      known; it must match expert_idxs and saves the device-to-host copy (and host sync) of the
      padded group sizes.
    - The 4-tuple is fp8_quant_grouped's SM90 K-major layout over the M = sum(ks) grouped rows
      (each group padded with zero rows to a multiple of 128). reverse_shuffle_idxs: int64 (m,), the
      grouped row of each slot. grouped_mm_offs: int32 (G,) cumulative padded group ends. ks: host
      list of the padded group sizes; ks_tensor: the same as int32 (G,) on the device. With m == 0
      every output is empty, grouped_mm_offs and ks_tensor of shape (0,).

    Row placement within a group: stable (slot order, fully deterministic) where the fused SM90
    kernels cover the input; atomic_add order otherwise (m or N zero, more than 512 groups, N not a
    multiple of 8, non-contiguous or misaligned inputs, non-int64 indices) and on SM100. The rowwise
    FP8 of each slot, grouped_mm_offs and ks do not depend on placement; the per-channel FP8 does
    (128-row blocks share a scale). Consumers locate a slot only through reverse_shuffle_idxs.

    Used where fp8_fused_experts_supported(). On SM100 it returns the same K-major layout with
    row-major scales, which SM100's k_grouped_fp8_gemm does not read. Scale strides are unspecified.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_scatter_quant_sm90 import scatter_quant_sm90
            return scatter_quant_sm90(gathered, expert_idxs, num_groups, expand_idx, padding_alignment, group_counts)
        case 10:
            from pithtrain.kernels.fp8_scatter_quant_sm100 import scatter_quant_sm100
            return scatter_quant_sm100(gathered, expert_idxs, num_groups, expand_idx, padding_alignment, group_counts)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
