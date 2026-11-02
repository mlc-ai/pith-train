"""
Picker of one rank's share of the chunked gated delta rule (chunk size 64, scale K ** -0.5) under
context parallelism at Qwen3.5's geometry, the rank holding a contiguous shard of the sequence
split over the CP group. Two families of entry points, one per architecture, each a direction as
stages around one all-gather over the CP group that the caller runs:

- SM90, CuTe-DSL kernels (gated_delta_rule_cp_supported, gdr_cp_*): the per-rank pieces of FLA
  0.5.2's context-parallel structure on unrepeated key heads, no Triton kernel. Forward, on rank
  r of n ranks holding consecutive slices of one sequence:
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
- SM100, TIRx kernels (chunk_gdr_cp_supported, gated_delta_rule_cp_plan, chunk_gdr_cp_*): q and k
  per value head. The stage before the all-gather returns the rank's fp32 (HV, K, V + K) payload
  (FLA 0.5.2's state summary in the forward, its gradient summary in the backward), the stage
  after takes every rank's payload stacked in rank order, (world_size, HV, K, V + K) fp32, as
  torch.distributed.all_gather_into_tensor returns it.

Each family's query is False on the other architecture, and its other entry points raise there.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["chunk_gdr_cp_bwd_post", "chunk_gdr_cp_bwd_pre", "chunk_gdr_cp_fwd_post", "chunk_gdr_cp_fwd_pre", "chunk_gdr_cp_supported", "gated_delta_rule_cp_plan", "gated_delta_rule_cp_supported", "gdr_cp_bwd_local", "gdr_cp_bwd_output", "gdr_cp_bwd_states", "gdr_cp_fwd_local", "gdr_cp_fwd_output", "gdr_cp_merge"]

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

SM100_ONLY = "the TIRx context-parallel gated delta rule kernels run on SM100 only; chunk_gdr_cp_supported is False on compute capability {}.x"


def chunk_gdr_cp_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether the chunk_gdr_cp_* entry points apply to this rank's call: on SM100, at the geometry the
    kernels were evolved and verified at: q, k, v bf16 (1, S, 32, 128) with S >= 1 (q / k given per
    value head, so already repeated to the 32 value heads where fewer key heads serve them), g fp32
    (1, S, 32), beta bf16 (1, S, 32). Otherwise the caller uses FLA's CP path. Depends only on
    shapes, dtypes and the device, so it gives the same answer on every rank of a CP group and for
    a forward and its backward (whose g is the forward's g_cum).
    """
    if ARCH_MAJOR != 10:
        return False
    S = v.shape[1] if v.dim() == 4 else 0
    shape = (1, S, 32, 128)
    return S >= 1 and tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == shape and tuple(g.shape) == tuple(beta.shape) == shape[:3] and q.dtype == k.dtype == v.dtype == beta.dtype == torch.bfloat16 and g.dtype == torch.float32


def gated_delta_rule_cp_plan(cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor, rank: int, is_first_rank: bool, is_last_rank: bool, pre_num_ranks: int, post_num_ranks: int) -> dict:
    """
    The host-side plan of one rank's chunk_gdr_cp_* calls for a CP layout; build it once per layout
    and pass it to every stage. The arguments are FLA's CP context of the rank (what
    fla.ops.cp.build_cp_context returns) and its rank in the CP group:

    cu_seqlens: int32 (n + 1,) on the device, the local sequence boundaries of the rank's shard,
    cu_seqlens_cpu the same values on the host. Only the first local sequence may continue from
    preceding ranks and only the last may continue on following ranks. is_first_rank /
    is_last_rank: whether the first local sequence starts / the last one ends on this rank.
    pre_num_ranks / post_num_ranks: how many preceding / following ranks those sequences span.

    The plan is opaque (its contents are the implementation's); it holds cu_seqlens, which must
    stay alive and unchanged while the plan is used. Only where gated_delta_rule_cp_supported is
    True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import gated_delta_rule_cp_plan_sm100
            return gated_delta_rule_cp_plan_sm100(cu_seqlens, cu_seqlens_cpu, rank, is_first_rank, is_last_rank, pre_num_ranks, post_num_ranks)
        case _:
            raise NotImplementedError(SM100_ONLY.format(ARCH_MAJOR))


def chunk_gdr_cp_fwd_pre(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, plan: dict) -> tuple[torch.Tensor, tuple]:
    """
    The CP forward up to the all-gather: (payload, saved).

    q, k: bf16 (1, S, HV, K) per value head; v: bf16 (1, S, HV, V); g: fp32 (1, S, HV), the raw
    (natural-log, negative) gate; beta: bf16 (1, S, HV) in (0, 1); plan from
    gated_delta_rule_cp_plan. Inputs are not modified (non-contiguous or misaligned ones are
    copied).

    payload: fresh fp32 (HV, K, V + K), FLA 0.5.2's forward state summary of the rank's last local
    sequence (zero on the last rank of that sequence), to all-gather over the CP group. saved:
    opaque tensors for chunk_gdr_cp_fwd_post. Asynchronous on torch's current stream. Only where
    chunk_gdr_cp_supported is True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import chunk_gdr_cp_fwd_pre_sm100
            return chunk_gdr_cp_fwd_pre_sm100(q, k, v, g, beta, plan)
        case _:
            raise NotImplementedError(SM100_ONLY.format(ARCH_MAJOR))


def chunk_gdr_cp_fwd_post(gathered: torch.Tensor, saved: tuple, plan: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The CP forward after the all-gather: fresh (o, g_cum, A, initial_state).

    gathered: fp32 (world_size, HV, K, V + K), every rank's chunk_gdr_cp_fwd_pre payload in rank
    order; saved: that call's saved.

    o: bf16 (1, S, HV, V), the rank's output. g_cum: fp32 (1, S, HV), A: bf16 (1, S, HV, 64) and
    initial_state: fp32 (1, HV, K, V) are inputs of chunk_gdr_cp_bwd_pre only: their shapes and
    dtypes are FLA's, their contents the implementation's (A also carries state checkpoints for the
    backward). Only where chunk_gdr_cp_supported is True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import chunk_gdr_cp_fwd_post_sm100
            return chunk_gdr_cp_fwd_post_sm100(gathered, saved, plan)
        case _:
            raise NotImplementedError(SM100_ONLY.format(ARCH_MAJOR))


def chunk_gdr_cp_bwd_pre(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, initial_state: torch.Tensor, plan: dict) -> tuple[torch.Tensor, tuple]:
    """
    The CP backward up to the all-gather: (payload, saved).

    q, k, v, beta: the forward inputs; g_cum, A, initial_state: chunk_gdr_cp_fwd_post's outputs;
    do: bf16 (1, S, HV, V), the gradient of o; plan: the forward's. Inputs are not modified.

    payload: fresh fp32 (HV, K, V + K), FLA 0.5.2's gradient summary of the rank's first local
    sequence (zero on the first rank of that sequence), to all-gather over the CP group. saved:
    opaque tensors for chunk_gdr_cp_bwd_post. Only where chunk_gdr_cp_supported is True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import chunk_gdr_cp_bwd_pre_sm100
            return chunk_gdr_cp_bwd_pre_sm100(q, k, v, g_cum, beta, A, do, initial_state, plan)
        case _:
            raise NotImplementedError(SM100_ONLY.format(ARCH_MAJOR))


def chunk_gdr_cp_bwd_post(gathered: torch.Tensor, saved: tuple, plan: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The CP backward after the all-gather: fresh (dq, dk, dv, dg, dbeta).

    gathered: fp32 (world_size, HV, K, V + K), every rank's chunk_gdr_cp_bwd_pre payload in rank
    order; saved: that call's saved.

    dq, dk: bf16 (1, S, HV, K) per value head (the caller sums each group for key-head gradients);
    dv: bf16 like v; dg: fp32 (1, S, HV), with respect to the raw gate g; dbeta: bf16 like beta.
    Matches FLA 0.5.2's CP chunk_gated_delta_rule_bwd to its CP verifier's 1e-4 relative L2 (not
    bit for bit). Only where chunk_gdr_cp_supported is True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import chunk_gdr_cp_bwd_post_sm100
            return chunk_gdr_cp_bwd_post_sm100(gathered, saved, plan)
        case _:
            raise NotImplementedError(SM100_ONLY.format(ARCH_MAJOR))
