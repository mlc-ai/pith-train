"""
Picker of one rank's share of the chunked gated delta rule (chunk size 64, scale K ** -0.5) under
context parallelism, at Qwen3.5's geometry: TIRx kernels on SM100. SM90 has no kernel here
(gated_delta_rule_cp_supported is False there and the caller runs FLA's CP path).

The rank holds a contiguous shard of the sequence split over the CP group. Each direction is two
stages around one all-gather over the CP group, which the caller runs: the stage before returns
the rank's fp32 (HV, K, V + K) payload (FLA 0.5.2's state summary in the forward, its gradient
summary in the backward), the stage after takes every rank's payload stacked in rank order,
(world_size, HV, K, V + K) fp32, as torch.distributed.all_gather_into_tensor returns it.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["chunk_gdr_cp_bwd_post", "chunk_gdr_cp_bwd_pre", "chunk_gdr_cp_fwd_post", "chunk_gdr_cp_fwd_pre", "gated_delta_rule_cp_plan", "gated_delta_rule_cp_supported"]


def gated_delta_rule_cp_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
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
            raise NotImplementedError(f"gated_delta_rule_cp has no kernel for compute capability {ARCH_MAJOR}.x")


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
    gated_delta_rule_cp_supported is True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import chunk_gdr_cp_fwd_pre_sm100
            return chunk_gdr_cp_fwd_pre_sm100(q, k, v, g, beta, plan)
        case _:
            raise NotImplementedError(f"gated_delta_rule_cp has no kernel for compute capability {ARCH_MAJOR}.x")


def chunk_gdr_cp_fwd_post(gathered: torch.Tensor, saved: tuple, plan: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The CP forward after the all-gather: fresh (o, g_cum, A, initial_state).

    gathered: fp32 (world_size, HV, K, V + K), every rank's chunk_gdr_cp_fwd_pre payload in rank
    order; saved: that call's saved.

    o: bf16 (1, S, HV, V), the rank's output. g_cum: fp32 (1, S, HV), A: bf16 (1, S, HV, 64) and
    initial_state: fp32 (1, HV, K, V) are inputs of chunk_gdr_cp_bwd_pre only: their shapes and
    dtypes are FLA's, their contents the implementation's (A also carries state checkpoints for the
    backward). Only where gated_delta_rule_cp_supported is True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import chunk_gdr_cp_fwd_post_sm100
            return chunk_gdr_cp_fwd_post_sm100(gathered, saved, plan)
        case _:
            raise NotImplementedError(f"gated_delta_rule_cp has no kernel for compute capability {ARCH_MAJOR}.x")


def chunk_gdr_cp_bwd_pre(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, initial_state: torch.Tensor, plan: dict) -> tuple[torch.Tensor, tuple]:
    """
    The CP backward up to the all-gather: (payload, saved).

    q, k, v, beta: the forward inputs; g_cum, A, initial_state: chunk_gdr_cp_fwd_post's outputs;
    do: bf16 (1, S, HV, V), the gradient of o; plan: the forward's. Inputs are not modified.

    payload: fresh fp32 (HV, K, V + K), FLA 0.5.2's gradient summary of the rank's first local
    sequence (zero on the first rank of that sequence), to all-gather over the CP group. saved:
    opaque tensors for chunk_gdr_cp_bwd_post. Only where gated_delta_rule_cp_supported is True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import chunk_gdr_cp_bwd_pre_sm100
            return chunk_gdr_cp_bwd_pre_sm100(q, k, v, g_cum, beta, A, do, initial_state, plan)
        case _:
            raise NotImplementedError(f"gated_delta_rule_cp has no kernel for compute capability {ARCH_MAJOR}.x")


def chunk_gdr_cp_bwd_post(gathered: torch.Tensor, saved: tuple, plan: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The CP backward after the all-gather: fresh (dq, dk, dv, dg, dbeta).

    gathered: fp32 (world_size, HV, K, V + K), every rank's chunk_gdr_cp_bwd_pre payload in rank
    order; saved: that call's saved.

    dq, dk: bf16 (1, S, HV, K) per value head (the caller sums each group for key-head gradients);
    dv: bf16 like v; dg: fp32 (1, S, HV), with respect to the raw gate g; dbeta: bf16 like beta.
    Matches FLA 0.5.2's CP chunk_gated_delta_rule_bwd to its CP verifier's 1e-4 relative L2 (not
    bit for bit). Only where gated_delta_rule_cp_supported is True.
    """
    match ARCH_MAJOR:
        case 10:
            from pithtrain.kernels.gated_delta_rule_cp_sm100 import chunk_gdr_cp_bwd_post_sm100
            return chunk_gdr_cp_bwd_post_sm100(gathered, saved, plan)
        case _:
            raise NotImplementedError(f"gated_delta_rule_cp has no kernel for compute capability {ARCH_MAJOR}.x")
