"""
Picker of the context-parallel chunked gated delta rule at Qwen3.5's geometry (chunk size 64): the
per-rank pieces of FLA 0.5.2's context-parallel structure on SM90 CuTe-DSL kernels, no Triton
kernel. SM90 only: gated_delta_rule_cp_supported is False elsewhere, and the other entry points
raise there. The collectives between the pieces (an all-gather of the rank maps in each direction)
belong to the caller.

Forward, on rank r of n ranks holding consecutive slices of one sequence:
    g_cum, A, w, u, hm = gdr_cp_fwd_local(k, v, g, beta, transition=r < n - 1)
    ag = all_gather(hm)                       # (n, HV, K, V + K)
    h0 = gdr_cp_merge(ag, r, r, reverse=False) if r > 0 else None
    o = gdr_cp_fwd_output(q, k, w, u, g_cum, h0)
Backward:
    w, u, dv_local, dhm = gdr_cp_bwd_local(q, k, v, g_cum, beta, A, do, transition=r > 0)
    ag = all_gather(dhm)
    h, v_new = gdr_cp_bwd_states(k, w, u, g_cum, h0)
    dht = gdr_cp_merge(ag, r, n - 1 - r, reverse=True) if r < n - 1 else None
    dq, dk, dv, dg, dbeta = gdr_cp_bwd_output(q, k, v, w, v_new, do, dv_local, h, g_cum, beta, A, u, dht)
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["gated_delta_rule_cp_supported", "gdr_cp_bwd_local", "gdr_cp_bwd_output", "gdr_cp_bwd_states", "gdr_cp_fwd_local", "gdr_cp_fwd_output", "gdr_cp_merge"]

SM90_ONLY = "the context-parallel gated delta rule kernels run on SM90 only; gated_delta_rule_cp_supported is False on compute capability {}.x"


def gated_delta_rule_cp_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether the gdr_cp_* entry points apply to a rank's slice (q, k (1, T, H, K), v (1, T, HV, V),
    g, beta (1, T, HV)): on SM90, all contiguous on one CUDA device, T > 0, HV a multiple of 32 and
    H == HV / 2 (Qwen3.5's 16 key / 32 value heads, or a batch of b folded into the heads, H 16 b /
    HV 32 b), K == V == 128, bf16 q / k / v / beta and an fp32 gate. False on every other
    architecture. Constant per geometry: torch.compile guards on the shape, device and dtype checks.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_cp_sm90 import gated_delta_rule_cp_supported_sm90
            return gated_delta_rule_cp_supported_sm90(q, k, v, g, beta)
        case _:
            return False


def gdr_cp_fwd_local(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, transition: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The rank-local forward before the all-gather: fresh (g_cum, A, w, u, hm).

    k: bf16 (1, T, H, K) contiguous, unrepeated key heads; v: bf16 (1, T, HV, V); g: fp32 (1, T,
    HV), the raw log gate; beta: bf16 (1, T, HV). Value head j reads key head j // (HV / H).
    g_cum: fp32 (1, T, HV), the chunk-local cumulative gate in log2 units, and A: bf16 (1, T, HV,
    64), the solved WY matrix, both bit for bit FLA's and what the backward reads; w: bf16 (1, T,
    HV, K) and u: bf16 (1, T, HV, V), FLA's recompute_w_u. hm: fp32 (HV, K, V + K): with transition
    the rank's affine state map from a zero state, he (columns :V, the final state of the
    recurrence) and M (columns V:, the product of the chunk transitions, a TF32 chain) as FLA's
    pre_process_fwd_kernel_merged computes them; without transition (the last rank) allocated and
    unwritten. Deterministic; ragged T (not a multiple of 64) runs as if zero-padded. SM90 only.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_cp_sm90 import gdr_cp_fwd_local_sm90
            return gdr_cp_fwd_local_sm90(k, v, g, beta, transition)
        case _:
            raise NotImplementedError(SM90_ONLY.format(ARCH_MAJOR))


def gdr_cp_merge(ag: torch.Tensor, rank: int, count: int, reverse: bool) -> torch.Tensor:
    """
    A fresh fp32 (HV, K, V) state folded from the gathered rank maps ag, fp32 (W, HV, K, V + K)
    contiguous (hm of every rank, or dhm in the backward): starting at zero, over count ranks s =
    rank - count .. rank - 1 (reverse: s = rank + count .. rank + 1), h = M_s h + he_s with he_s =
    ag[s, :, :, :V] and M_s = ag[s, :, :, V:], a TF32 product with he_s as its accumulator, FLA's
    merge_fwd_bwd_kernel in its context-parallel mode. Slots outside those count ranks are never
    read. count == 0 returns zeros. rank and count are runtime arguments (no compile per rank or
    group size). Deterministic. SM90 only.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_cp_sm90 import gdr_cp_merge_sm90
            return gdr_cp_merge_sm90(ag, rank, count, reverse)
        case _:
            raise NotImplementedError(SM90_ONLY.format(ARCH_MAJOR))


def gdr_cp_fwd_output(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cum: torch.Tensor, h0: torch.Tensor | None) -> torch.Tensor:
    """
    A fresh o, bf16 (1, T, HV, V): the chunked output from the initial state h0, fp32 (HV, K, V)
    contiguous (None: zero), with q bf16 (1, T, H, K) and the rest from gdr_cp_fwd_local; FLA's
    chunk_gated_delta_rule_fwd_h(h0) and chunk_fwd_o to bf16 rounding. Deterministic. SM90 only.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_cp_sm90 import gdr_cp_fwd_output_sm90
            return gdr_cp_fwd_output_sm90(q, k, w, u, g_cum, h0)
        case _:
            raise NotImplementedError(SM90_ONLY.format(ARCH_MAJOR))


def gdr_cp_bwd_local(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, transition: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The rank-local backward before the all-gather: fresh (w, u, dv_local, dhm).

    q, k, v, beta: the forward inputs; g_cum, A: from gdr_cp_fwd_local; do: bf16 (1, T, HV, V), the
    gradient of o. w, u: bf16, FLA's recompute_w_u; dv_local: bf16 (1, T, HV, V), FLA's
    chunk_bwd_dv_local. dhm: fp32 (HV, K, V + K): with transition the rank's backward map, the
    gradient of its initial state from a zero final gradient (columns :V) and the reverse chain of
    the transposed chunk transitions (columns V:), as FLA's pre_process_bwd_kernel_merged computes
    them; without transition (the first rank) allocated and unwritten. Deterministic. SM90 only.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_cp_sm90 import gdr_cp_bwd_local_sm90
            return gdr_cp_bwd_local_sm90(q, k, v, g_cum, beta, A, do, transition)
        case _:
            raise NotImplementedError(SM90_ONLY.format(ARCH_MAJOR))


def gdr_cp_bwd_states(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g_cum: torch.Tensor, h0: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fresh (h, v_new): the chunk-entry states, bf16 (1, ceil(T / 64), HV, V, K) V-first, and v_new,
    bf16 like u, recomputed from the initial state h0, fp32 (HV, K, V) (None: zero); FLA's
    chunk_gated_delta_rule_fwd_h(h0) given the same w and u. Inputs of gdr_cp_bwd_output only.
    Deterministic. SM90 only.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_cp_sm90 import gdr_cp_bwd_states_sm90
            return gdr_cp_bwd_states_sm90(k, w, u, g_cum, h0)
        case _:
            raise NotImplementedError(SM90_ONLY.format(ARCH_MAJOR))


def gdr_cp_bwd_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, v_new: torch.Tensor, do: torch.Tensor, dv_local: torch.Tensor, h: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, u: torch.Tensor, dht: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The rank's input gradients (dq, dk, dv, dg, dbeta) given the gradient of its final state dht,
    fp32 (HV, K, V) contiguous (None: zero): the reverse state-gradient recurrence from dht, then
    the per-chunk gradients, on FLA 0.5.2's rounding points. dq, dk: (1, T, HV, K) per value head
    (the caller sums each group of HV / H heads); dv like v; dg: fp32 (1, T, HV) with respect to
    the raw gate; dbeta like beta. Overwrites u (its buffer holds du) and dv_local (its buffer holds
    dv, which is returned). Deterministic. SM90 only.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.gated_delta_rule_cp_sm90 import gdr_cp_bwd_output_sm90
            return gdr_cp_bwd_output_sm90(q, k, v, w, v_new, do, dv_local, h, g_cum, beta, A, u, dht)
        case _:
            raise NotImplementedError(SM90_ONLY.format(ARCH_MAJOR))
