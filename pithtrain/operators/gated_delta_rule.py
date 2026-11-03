"""
Chunked gated delta rule of the Qwen3.5 Gated DeltaNet as custom ops, so that the linear-attention
region compiles as one graph.

The recurrence is independent per head, so the batch folds into the head axis first (batch 1 at b
times the heads); where the kernels take only one sample's heads the samples run one at a time.
Without context parallelism it runs the pinned kernels where gated_delta_rule_pinned_supported says
so. Under context parallelism (contiguous layout) it runs the library's CP kernels: where
chunk_gdr_cp_supported says so (SM100 at Qwen3.5's geometry) the TIRx kernels, each direction as
two stages around one all-gather of the ranks' state summaries over the CP group; where
gated_delta_rule_cp_supported says so (SM90 at Qwen3.5's geometry, q and k with the unrepeated key
heads) the CuTe kernels of FLA's context-parallel structure, each direction around one all-gather
of the ranks' state maps. The all-gathers run here. Any other call raises ValueError naming the
geometry the kernels cover: no other implementation is kept.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from torch.distributed import ProcessGroup, all_gather_into_tensor, get_rank, get_world_size
from torch.distributed.distributed_c10d import _resolve_process_group

from pithtrain.kernels import (
    chunk_gdr_bwd_pinned,
    chunk_gdr_cp_bwd_post,
    chunk_gdr_cp_bwd_pre,
    chunk_gdr_cp_fwd_post,
    chunk_gdr_cp_fwd_pre,
    chunk_gdr_cp_supported,
    chunk_gdr_fwd_pinned,
    gated_delta_rule_cp_plan,
    gated_delta_rule_cp_supported,
    gated_delta_rule_pinned_supported,
    gdr_cp_bwd_local,
    gdr_cp_bwd_output,
    gdr_cp_bwd_states,
    gdr_cp_fwd_local,
    gdr_cp_fwd_output,
    gdr_cp_merge,
)

__all__ = ["dense_gated_delta_rule", "gated_delta_rule"]

# mypy: ignore-errors

CHUNK_SIZE = 64  # the kernels' chunk size (FLA's BT)

SUPPORTED_GEOMETRY = (
    "Qwen3.5's Gated DeltaNet geometry: q, k [b, s, 16, 128] (or 32 key heads), v [b, s, 32, 128], "
    "beta [b, s, 32] in bf16 and g [b, s, 32] in fp32, s > 0, on a device the kernels run on "
    "(under context parallelism, 16 key heads)"
)


def unsupported(q, k, v, g, beta, cp: bool) -> ValueError:
    shapes = ", ".join(f"{n} {tuple(t.shape)} {t.dtype}" for n, t in zip(("q", "k", "v", "g", "beta"), (q, k, v, g, beta)))  # fmt: skip
    where = "under context parallelism" if cp else "without context parallelism"
    return ValueError(f"gated_delta_rule has no kernel for {shapes} {where}; the kernels cover {SUPPORTED_GEOMETRY}")  # fmt: skip


@lru_cache(maxsize=None)
def cp_plan(group_name: str, seq_len: int) -> dict:
    """
    The CP kernels' host-side plan for this rank: one sequence of seq_len * world_size tokens spans
    the group in rank order, so the rank's shard is one local sequence that continues from every
    preceding rank and on every following one (the context FLA's build_cp_context derives for that
    layout). Cached because the plan reaches host memory.
    """
    group = _resolve_process_group(group_name)
    rank, world_size = get_rank(group), get_world_size(group)
    cu_seqlens_cpu = torch.tensor([0, seq_len], dtype=torch.int32)
    cu_seqlens = cu_seqlens_cpu.to(torch.cuda.current_device())
    return gated_delta_rule_cp_plan(cu_seqlens, cu_seqlens_cpu, rank, rank == 0, rank == world_size - 1, rank, world_size - 1 - rank)  # fmt: skip


def cp_all_gather(payload: torch.Tensor, group_name: str) -> torch.Tensor:
    """
    Every rank's payload stacked in rank order, (world_size, *payload.shape): the one collective of
    each CP direction.
    """
    group = _resolve_process_group(group_name)
    gathered = payload.new_empty(get_world_size(group), *payload.shape)
    all_gather_into_tensor(gathered, payload, group=group)
    return gathered


@lru_cache(maxsize=None)
def cp_layout(group_name: str) -> Tuple[int, int]:
    """
    (rank, size) of the CP group, read once per group (host values, no device sync).
    """
    group = _resolve_process_group(group_name)
    return get_rank(group), get_world_size(group)


def gather_maps(local: torch.Tensor, group_name: str, async_op: bool):
    """
    Every rank's CuTe state map stacked, (size, HV, K, V + K), and the collective's work handle
    when async_op (its wait makes the current stream wait, no host sync), else None.
    """
    size = cp_layout(group_name)[1]
    gathered = local.new_empty(size, *local.shape)
    work = all_gather_into_tensor(gathered, local, group=_resolve_process_group(group_name), async_op=async_op)  # fmt: skip
    return gathered, work


def cute_cp_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:  # fmt: skip
    """
    The CuTe context-parallel forward (gated_delta_rule_cp_supported): every rank but the last
    computes its state map, every rank all-gathers the maps, every rank but the first folds its
    predecessors' into its initial state h0, which the backward starts from again (zeros on the
    first rank, as FLA saves).
    """
    rank, size = cp_layout(cp_group_name)
    g_cum, A, w, u, hm = gdr_cp_fwd_local(k, v, g, beta, rank < size - 1)
    gathered, _ = gather_maps(hm, cp_group_name, False)
    h0 = gdr_cp_merge(gathered, rank, rank, False)
    o = gdr_cp_fwd_output(q, k, w, u, g_cum, h0 if rank > 0 else None)
    return o, g_cum, A, [h0.unsqueeze(0)]


def cute_cp_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, h0: torch.Tensor, cp_group_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """
    The CuTe context-parallel backward: every rank but the first computes its backward map, all
    ranks all-gather the maps, then every rank but the last folds its successors' maps into the
    gradient of its final state. The all-gather is synchronous: overlapping it with the state
    recompute saved nothing measurable, since it mostly waits for the slowest rank's map. dq and
    dk come back per value head and are summed over each group of value heads.
    """
    rank, size = cp_layout(cp_group_name)
    w, u, dv_local, dhm = gdr_cp_bwd_local(q, k, v, g_cum, beta, A, do, rank > 0)
    gathered, _ = gather_maps(dhm, cp_group_name, False)
    h, v_new = gdr_cp_bwd_states(k, w, u, g_cum, h0[0] if rank > 0 else None)
    dht = gdr_cp_merge(gathered, rank, size - 1 - rank, True) if rank < size - 1 else None
    dq, dk, dv, dg, dbeta = gdr_cp_bwd_output(q, k, v, w, v_new, do, dv_local, h, g_cum, beta, A, u, dht)  # fmt: skip
    T, H, K = k.shape[1:]
    dq, dk = (x.view(1, T, H, -1, K).sum(3) for x in (dq, dk))
    return dq, dk, dv, dg, dbeta


@torch.library.custom_op("pithtrain::gated_delta_rule_fwd", mutates_args=())
def gdr_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:  # fmt: skip
    """
    The CP forward of one rank on the library's CP kernels (callers check chunk_gdr_cp_supported,
    q and k repeated to the value heads, or gated_delta_rule_cp_supported, q and k with the key
    heads): (o, g_cum, A, [initial_state]), the last three for the backward, which recomputes the
    states from the merged initial state rather than paying a second all-gather.
    """
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    if gated_delta_rule_cp_supported(q, k, v, g, beta):
        return cute_cp_fwd(q, k, v, g, beta, cp_group_name)
    plan = cp_plan(cp_group_name, q.shape[1])
    payload, saved = chunk_gdr_cp_fwd_pre(q, k, v, g, beta, plan)
    o, g_out, A, initial_state = chunk_gdr_cp_fwd_post(cp_all_gather(payload, cp_group_name), saved, plan)  # fmt: skip
    return o, g_out, A, [initial_state]


@gdr_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group_name: str):  # fmt: skip
    b, s, hv, vd = v.shape
    o = torch.empty((b, s, hv, vd), dtype=q.dtype, device=q.device)
    A = torch.empty((b, s, hv, CHUNK_SIZE), dtype=q.dtype, device=q.device)
    state = [torch.empty((1, hv, q.shape[-1], vd), dtype=torch.float32, device=q.device)]
    return o, torch.empty_like(g, dtype=torch.float32), A, state


@torch.library.custom_op("pithtrain::gated_delta_rule_bwd", mutates_args=())
def gdr_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, initial_state: List[torch.Tensor], cp_group_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    q, k, v, g_out, beta, A, do = (t.contiguous() for t in (q, k, v, g_out, beta, A, do))
    if gated_delta_rule_cp_supported(q, k, v, g_out, beta):
        return cute_cp_bwd(q, k, v, g_out, beta, A, do, initial_state[0], cp_group_name)
    plan = cp_plan(cp_group_name, q.shape[1])
    payload, saved = chunk_gdr_cp_bwd_pre(q, k, v, g_out, beta, A, do, initial_state[0], plan)
    return chunk_gdr_cp_bwd_post(cp_all_gather(payload, cp_group_name), saved, plan)


@gdr_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, initial_state: List[torch.Tensor], cp_group_name: str):  # fmt: skip
    # Contiguous like the real gradients; plain empty_like would inherit the input strides.
    contiguous = torch.contiguous_format
    return torch.empty_like(q, memory_format=contiguous), torch.empty_like(k, memory_format=contiguous), torch.empty_like(v, memory_format=contiguous), torch.empty_like(g_out, memory_format=contiguous), torch.empty_like(beta, memory_format=contiguous)  # fmt: skip


def gdr_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:  # fmt: skip
    q, k, v, _, beta, cp_group_name = inputs
    _, g_out, A, state = output
    ctx.save_for_backward(q, k, v, g_out, beta, A, *state)
    ctx.cp_group_name = cp_group_name


def gdr_backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor, *_) -> Tuple:
    q, k, v, g_out, beta, A, *state = ctx.saved_tensors
    return *gdr_bwd(q, k, v, g_out, beta, A, do, state, ctx.cp_group_name), None


gdr_fwd.register_autograd(gdr_backward, setup_context=gdr_setup_context)


def select_key_heads(x: torch.Tensor) -> torch.Tensor:
    """
    [b, s, heads, dim], from either that or its [b, s, heads, group, dim] expand over a group.
    """
    if x.dim() == 5:
        x = x[:, :, :, 0]
    return x if x.is_contiguous() else x.contiguous()


@torch.library.custom_op("pithtrain::gated_delta_rule_dense_fwd", mutates_args=())
def dense_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """
    chunk_gated_delta_rule_fwd without CP: (output, chunk-cumulated gate, WY matrix A), the last
    two for the backward. v, g and beta have HV heads; q and k have HV heads, or H heads given as
    their [b, s, H, HV // H, dim] expand.
    """
    return chunk_gdr_fwd_pinned(select_key_heads(q), select_key_heads(k), v, g, beta)


@dense_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
    B, T, HV = g.shape
    A = torch.empty(B, T, HV, CHUNK_SIZE, device=k.device, dtype=k.dtype)
    return torch.empty_like(v), torch.empty_like(g, dtype=torch.float), A


@torch.library.custom_op("pithtrain::gated_delta_rule_dense_bwd", mutates_args=())
def dense_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """
    chunk_gated_delta_rule_bwd, dense: (dq, dk, dv, dg, dbeta), dg with respect to the raw gate.
    With grouped value heads, dq and dk are per value head, shaped like the expands q and k came in
    as; autograd's backward of the expand sums them over each group, rounding as a repeat_interleave
    of q and k would.
    """
    dq, dk, dv, dg, dbeta = chunk_gdr_bwd_pinned(select_key_heads(q), select_key_heads(k), v, g_cum, beta, A, do)  # fmt: skip
    return dq.view(q.shape), dk.view(k.shape), dv, dg, dbeta


@dense_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor):  # fmt: skip
    return tuple(torch.empty_like(t, memory_format=torch.contiguous_format) for t in (q, k, v, g_cum, beta))  # fmt: skip


def dense_setup_context(ctx, inputs, output) -> None:
    q, k, v, _, beta = inputs
    _, g_cum, A = output
    ctx.save_for_backward(q, k, v, g_cum, beta, A)


def dense_backward(ctx, do: torch.Tensor, *_) -> tuple:
    q, k, v, g_cum, beta, A = ctx.saved_tensors
    return dense_bwd(q, k, v, g_cum, beta, A, do.contiguous())


dense_fwd.register_autograd(dense_backward, setup_context=dense_setup_context)


def dense_gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """
    gated_delta_rule without context parallelism, through the pinned kernels; callers check
    gated_delta_rule_pinned_supported first.

    On SM90 the forward and backward run CuTe kernels on FLA 0.5.2's chunk algorithm and rounding
    points at every length (no Triton kernel; a length that is not a multiple of 64 runs as if
    zero-padded to one, FLA's ragged semantics); the forward's saved g_cum and A are bit-identical
    to FLA's. Against the fp32 recurrence, measured against FLA: calc_diff within 1.001x of FLA's
    and the largest single-element error within 1.04x of FLA's in grad_check's cases (T = 4096 /
    8192, 3 seeds), and within the verifier's 1.10x bounds in its cases. On SM100 TIRx kernels at
    Qwen3.5's geometry.
    """
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    B, S, H, K = k.shape
    group_size = v.shape[2] // H
    if group_size > 1:
        # A view, not a copy: the kernels read each key head once per group by index.
        q, k = (t.unsqueeze(3).expand(B, S, H, group_size, K) for t in (q, k))
    o, _, _ = dense_fwd(q, k, v, g, beta)
    return o


def fold_batch(t: torch.Tensor) -> torch.Tensor:
    """
    [b, s, heads, ...] -> [1, s, b * heads, ...]; a view at batch 1, else a copy.
    """
    return t.transpose(0, 1).reshape(1, t.shape[1], -1, *t.shape[3:])


def unfold_batch(o: torch.Tensor, b: int) -> torch.Tensor:
    """
    fold_batch's inverse on the output, [1, s, b * heads, dim] -> [b, s, heads, dim].
    """
    return o if b == 1 else o.reshape(o.shape[1], b, -1, o.shape[-1]).transpose(0, 1)


def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group: Optional[ProcessGroup] = None) -> torch.Tensor:  # fmt: skip
    """
    Chunked gated delta rule of q and k [b, s, heads, dim], v [b, s, value_heads, value_dim], g and
    beta [b, s, value_heads], returning [b, s, value_heads, value_dim]. value_heads is a multiple
    of heads, and each key head serves a group of consecutive value heads. With a cp_group the
    recurrence is sharded over it: s is the rank-local length, in the contiguous layout.

    The recurrence is independent per head, so the batch folds into the head axis before anything
    else: value head j = b * value_heads + h of the fold reads key head j // (value_heads / heads)
    = b * heads + h // (value_heads / heads), the same grouping, and the kernels run batch b at b
    times the heads where they take that geometry (SM90's pinned kernels); where they take only one
    sample's heads (SM100) the samples run one at a time. Raises ValueError (before launching
    anything) for a geometry, dtype or device no kernel covers; see SUPPORTED_GEOMETRY.
    """
    b = q.shape[0]
    if cp_group is None:
        # Contiguous before the check: the pinned kernels take contiguous tensors, and
        # dense_gated_delta_rule copies anyway.
        folded = tuple(fold_batch(t).contiguous() for t in (q, k, v, g, beta))
        if gated_delta_rule_pinned_supported(*folded):
            return unfold_batch(dense_gated_delta_rule(*folded), b)
        if b > 1 and gated_delta_rule_pinned_supported(*(t[:1].contiguous() for t in (q, k, v, g, beta))):  # fmt: skip
            # The kernels take one sample's heads but not the fold (SM100): one sample at a time.
            return torch.cat([dense_gated_delta_rule(*(t[i : i + 1] for t in (q, k, v, g, beta))) for i in range(b)])  # fmt: skip
        raise unsupported(q, k, v, g, beta, cp=False)
    # Under CP the SM90 CuTe kernels take q and k with the key heads, folded.
    folded = [fold_batch(t).contiguous() for t in (q, k, v, g, beta)]
    if gated_delta_rule_cp_supported(*folded):
        o, *_ = gdr_fwd(*folded, cp_group.group_name)
        return unfold_batch(o, b)
    # The SM100 TIRx kernels take q and k per value head: repeated, then folded.
    repeats = v.shape[2] // q.shape[2] if q.dim() == v.dim() == 4 and q.shape[2] > 0 else 1
    qr, kr = (t.repeat_interleave(repeats, dim=2) if repeats > 1 else t for t in (q, k))
    folded = [fold_batch(t) for t in (qr, kr, v, g, beta)]
    if chunk_gdr_cp_supported(*folded):
        o, *_ = gdr_fwd(*folded, cp_group.group_name)
        return unfold_batch(o, b)
    if b > 1 and chunk_gdr_cp_supported(*(t[:1] for t in (qr, kr, v, g, beta))):
        # The CP kernels take one sample's heads but not the fold: one sample at a time, each with
        # its own pair of all-gathers (every rank of the group holds the same batch size).
        return torch.cat([gdr_fwd(*(t[i : i + 1] for t in (qr, kr, v, g, beta)), cp_group.group_name)[0] for i in range(b)])  # fmt: skip
    raise unsupported(q, k, v, g, beta, cp=True)
