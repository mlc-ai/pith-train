"""
Picker of the pinned dense chunked gated delta rule (chunk size 64, no initial or final state) at
Qwen3.5's geometry: on SM90 a CuTe-DSL forward and FLA's backward kernels launched with the configs
FLA's autotuner picks on H200, TIRx kernels on SM100. The callers' custom ops keep both opaque to
torch.compile.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["chunk_gdr_bwd_pinned", "chunk_gdr_fwd_pinned", "gated_delta_rule_pinned_supported"]


def pinned_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether a call has the geometry the kernels take: B == 1, HV == 32, HV a multiple of H, K == V
    == 128, bf16 q / k / v / beta and an fp32 gate. On SM90 FLA's autotune keys hold B (cumsum), H,
    HV, K and V, and the backward's single K block needs K <= 128.
    """
    B, _, H, K = k.shape
    HV, V = v.shape[2:]
    return B == 1 and HV == 32 and HV % H == 0 and K == 128 and V == 128 and q.shape == k.shape and q.dtype == k.dtype == v.dtype == beta.dtype == torch.bfloat16 and g.dtype == torch.float32


def gated_delta_rule_pinned_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether chunk_gdr_fwd_pinned / chunk_gdr_bwd_pinned apply to this call: the geometry of
    pinned_shapes, on SM100, or on SM90 with the recorded FLA and Triton versions. Otherwise the
    caller uses FLA's autotuned entry points. Constant per geometry: torch.compile guards on the
    shape and dtype checks and treats the version check as a constant.
    """
    if not pinned_shapes(q, k, v, g, beta):
        return False
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_pinned_sm90 import pinned_launch_supported
            return pinned_launch_supported()
        case 10:
            return True
        case _:
            return False


def chunk_gdr_fwd_pinned(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dense chunked gated delta rule forward (chunk size 64, scale K ** -0.5, no initial or final
    state): fresh (o, g_cum, A).

    q, k: bf16 (B, T, H, K) contiguous, unrepeated key heads. v: bf16 (B, T, HV, V). g: fp32 (B, T,
    HV), the raw log gate. beta: bf16 (B, T, HV). HV is a multiple of H; value head j reads key head
    j // (HV / H). Inputs are not modified.

    o: bf16 (B, T, HV, V). g_cum: fp32 (B, T, HV) and A: bf16 (B, T, HV, 64) are what the backward
    reads, inputs of chunk_gdr_bwd_pinned only: their shapes and dtypes are fixed, their contents the
    implementation's (on SM90 FLA's chunk-local cumulative gate and solved WY matrix). On SM90 equal
    to FLA 0.5.2's chunk_gated_delta_rule_fwd on q / k repeat-interleaved to HV heads up to bf16
    rounding (bit-identical when T is not a multiple of 64). Deterministic. Only where
    gated_delta_rule_pinned_supported is True.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_pinned_sm90 import chunk_gdr_fwd_pinned_sm90
            return chunk_gdr_fwd_pinned_sm90(q, k, v, g, beta)
        case 10:
            from pithtrain.kernels.gated_delta_rule_pinned_sm100 import chunk_gdr_fwd_pinned_sm100
            return chunk_gdr_fwd_pinned_sm100(q, k, v, g, beta)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def chunk_gdr_bwd_pinned(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward of chunk_gdr_fwd_pinned: fresh (dq, dk, dv, dg, dbeta).

    q, k, v, beta: the forward inputs; g_cum, A: the forward outputs; do: bf16 (B, T, HV, V), the
    gradient of o. Inputs are not modified.

    dq, dk: (B, T, HV, K) in q's / k's dtype, per value head (the caller sums each group of HV / H
    heads, in repeat_interleave order, for the key-head gradients). dv like v; dg: fp32 (B, T, HV),
    with respect to the raw gate g; dbeta like beta. On SM90 bit-identical to FLA 0.5.2's
    chunk_gated_delta_rule_bwd on repeat-interleaved q / k. Deterministic. Only where
    gated_delta_rule_pinned_supported is True.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_pinned_sm90 import chunk_gdr_bwd_pinned_sm90
            return chunk_gdr_bwd_pinned_sm90(q, k, v, g_cum, beta, A, do)
        case 10:
            from pithtrain.kernels.gated_delta_rule_pinned_sm100 import chunk_gdr_bwd_pinned_sm100
            return chunk_gdr_bwd_pinned_sm100(q, k, v, g_cum, beta, A, do)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
