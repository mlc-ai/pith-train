"""
Picker of the pinned dense chunked gated delta rule (chunk size 64, no initial or final state) at
Qwen3.5's geometry: on SM90 CuTe-DSL kernels in both directions at every T (a ragged variant of
each when T is not a multiple of 64); TIRx kernels on SM100. The callers' custom
ops keep both opaque to torch.compile.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["chunk_gdr_bwd_pinned", "chunk_gdr_fwd_pinned", "gated_delta_rule_pinned_supported"]

# The SM100 backward recovers dbeta by dividing a beta-scaled product by beta + 1e-10, which loses
# the gradient at beta == 0 and scales it down below ~1e-9. Both SM100 directions see beta raised to
# this floor (a power of two, exact in bf16), where that division is accurate to ~1e-4; the
# returned dbeta is the gradient there, passed straight through to the caller's beta.
SM100_BETA_FLOOR = 2.0**-20


def consistent_geometry(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether the five tensors are what the kernels index without checking: ranks 4 / 4 / 4 / 3 / 3,
    all contiguous on one CUDA device, v, g and beta sharing q / k's batch and length with g and
    beta (B, T, HV), and T > 0. Never raises on a malformed call.
    """
    tensors = (q, k, v, g, beta)
    if tuple(t.dim() for t in tensors) != (4, 4, 4, 3, 3):
        return False
    B, T = k.shape[:2]
    HV = v.shape[2]
    return T > 0 and tuple(v.shape[:2]) == (B, T) and tuple(g.shape) == (B, T, HV) and tuple(beta.shape) == (B, T, HV) and len({t.device for t in tensors}) == 1 and k.device.type == "cuda" and all(t.is_contiguous() for t in tensors)


def pinned_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether a call has the geometry the kernels take: B == 1, HV == 32, HV a multiple of H, K == V
    == 128, bf16 q / k / v / beta and an fp32 gate, with consistent_geometry.
    """
    if not consistent_geometry(q, k, v, g, beta):
        return False
    B, _, H, K = k.shape
    HV, V = v.shape[2:]
    return B == 1 and HV == 32 and HV % H == 0 and K == 128 and V == 128 and q.shape == k.shape and q.dtype == k.dtype == v.dtype == beta.dtype == torch.bfloat16 and g.dtype == torch.float32


def folded_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether a call has the geometry of Qwen3.5's heads with the batch folded into the head axis:
    B == 1, HV a multiple of 32 and H == HV / 2 (b samples of H 16 / HV 32 fold to H 16 b / HV 32
    b), K == V == 128, bf16 q / k / v / beta and an fp32 gate, with consistent_geometry. The SM90
    kernels index every head by HV and H, so this is pinned_shapes' geometry repeated b times.
    """
    if not consistent_geometry(q, k, v, g, beta):
        return False
    B, _, H, K = k.shape
    HV, V = v.shape[2:]
    return B == 1 and HV % 32 == 0 and HV == 2 * H and K == 128 and V == 128 and q.shape == k.shape and q.dtype == k.dtype == v.dtype == beta.dtype == torch.bfloat16 and g.dtype == torch.float32


def gated_delta_rule_pinned_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether chunk_gdr_fwd_pinned / chunk_gdr_bwd_pinned apply to this call. On SM90 the geometry of
    pinned_shapes, and also the folded geometry of folded_shapes (a batch folded into the heads).
    On SM100 the geometry the TIRx kernels were evolved and verified at, one sample's heads:
    pinned_shapes with H 16 or 32 key heads over the 32 value heads (not a folded batch, whose
    samples a caller runs one at a time). Otherwise False. Constant per geometry: torch.compile
    guards on the shape, device and dtype checks.
    """
    match ARCH_MAJOR:
        case 9:
            return pinned_shapes(q, k, v, g, beta) or folded_shapes(q, k, v, g, beta)
        case 10:
            return pinned_shapes(q, k, v, g, beta) and k.shape[2] in (16, 32)
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
    reads, inputs of chunk_gdr_bwd_pinned only: their shapes and dtypes are fixed, their contents
    the implementation's (on SM90 the chunk-local cumulative gate and solved WY matrix of FLA
    0.5.2's chunk algorithm). On SM90 at every T (a T that is not a multiple of 64 runs as if
    zero-padded to one, FLA's ragged semantics). o matches the fp32 recurrence to calc_diff < 1e-4
    and a largest absolute error within 2e-2 of the reference's largest magnitude. On SM100 a beta
    below 2^-20 runs as 2^-20 (SM100_BETA_FLOOR), in both directions. Deterministic. Only where
    gated_delta_rule_pinned_supported is True.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_pinned_fwd_sm90 import chunk_gdr_fwd_pinned_sm90
            return chunk_gdr_fwd_pinned_sm90(q, k, v, g, beta)
        case 10:
            from pithtrain.kernels.gated_delta_rule_pinned_sm100 import chunk_gdr_fwd_pinned_sm100
            return chunk_gdr_fwd_pinned_sm100(q, k, v, g, beta.clamp(min=SM100_BETA_FLOOR))
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def chunk_gdr_bwd_pinned(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward of chunk_gdr_fwd_pinned: fresh (dq, dk, dv, dg, dbeta).

    q, k, v, beta: the forward inputs; g_cum, A: the forward outputs; do: bf16 (B, T, HV, V), the
    gradient of o. Inputs are not modified.

    dq, dk: (B, T, HV, K) in q's / k's dtype, per value head (the caller sums each group of HV / H
    heads, in repeat_interleave order, for the key-head gradients). dv like v; dg: fp32 (B, T, HV),
    with respect to the raw gate g; dbeta like beta. On SM90 CuTe kernels at every T, on FLA
    0.5.2's chunk algorithm and rounding points. The gradients match the fp32 recurrence's autograd
    to calc_diff < 1e-3 and a largest absolute error within 2e-2 of the reference's largest
    magnitude. On SM100 a beta below 2^-20 runs as 2^-20, as in the forward, and dbeta there is the
    gradient at 2^-20 (the kernel's dbeta loses the gradient at beta == 0 otherwise). Deterministic.
    Only where gated_delta_rule_pinned_supported is True.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_pinned_bwd_sm90 import chunk_gdr_bwd_pinned_sm90
            return chunk_gdr_bwd_pinned_sm90(q, k, v, g_cum, beta, A, do)
        case 10:
            from pithtrain.kernels.gated_delta_rule_pinned_sm100 import chunk_gdr_bwd_pinned_sm100
            return chunk_gdr_bwd_pinned_sm100(q, k, v, g_cum, beta.clamp(min=SM100_BETA_FLOOR), A, do)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
