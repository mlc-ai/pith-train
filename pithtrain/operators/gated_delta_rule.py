"""
Chunked gated delta rule of the Qwen3.5 Gated DeltaNet as custom ops, so that the linear-attention
region compiles as one graph (FLA's own entry point is excluded from torch.compile).

The recurrence is independent per head, so the batch folds into the head axis first (batch 1 at b
times the heads). Without context parallelism the pinned kernels run where
gated_delta_rule_pinned_supported says so. Under context parallelism (contiguous layout), where
chunk_gdr_cp_supported says so (SM100 at Qwen3.5's geometry) it runs the library's CP kernels, each
direction as two stages around one all-gather of the ranks' state summaries over the CP group, run
here; elsewhere FLA's context-parallel path, which runs the same all-gather itself.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import fla.ops.common.chunk_o
import torch
from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
from fla.ops.utils import prepare_chunk_indices
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
    gated_delta_rule_pinned_supported,
)

__all__ = ["dense_gated_delta_rule", "gated_delta_rule"]

# mypy: ignore-errors

# Turn off FLA's issue 640 guard: the BK=64 backward miscompile it avoids does not occur at
# head_k_dim=128.
fla.ops.common.chunk_o.TRITON_ABOVE_3_4_0 = False

CHUNK_SIZE = 64  # FLA's chunk size (BT)


@lru_cache(maxsize=None)
def cp_state(group_name: Optional[str], seq_len: int) -> Tuple:
    """
    FLA's context-parallel context and chunk indices for this rank. Cached because build_cp_context
    reaches host memory.
    """
    if group_name is None:
        return None, None
    group = _resolve_process_group(group_name)
    device = torch.cuda.current_device()
    # One global sequence spanning the group, which is the shape FLA merges state over.
    cu_seqlens = torch.tensor([0, seq_len * get_world_size(group)], dtype=torch.int32, device=device)  # fmt: skip
    context = build_cp_context(cu_seqlens, group)
    chunk_indices = prepare_chunk_indices(context.cu_seqlens, CHUNK_SIZE, cu_seqlens_cpu=context.cu_seqlens_cpu)  # fmt: skip
    return context, chunk_indices


@lru_cache(maxsize=None)
def cp_plan(group_name: str, seq_len: int) -> dict:
    """
    The CP kernels' host-side plan for this rank, from FLA's context of cp_state.
    """
    context, _ = cp_state(group_name, seq_len)
    rank = get_rank(_resolve_process_group(group_name))
    return gated_delta_rule_cp_plan(context.cu_seqlens, context.cu_seqlens_cpu, rank, context.is_first_rank, context.is_last_rank, context.pre_num_ranks, context.post_num_ranks)  # fmt: skip


def cp_all_gather(payload: torch.Tensor, group_name: str) -> torch.Tensor:
    """
    Every rank's payload stacked in rank order, (world_size, *payload.shape): the one collective of
    each CP direction, as FLA's CP path runs it.
    """
    group = _resolve_process_group(group_name)
    gathered = payload.new_empty(get_world_size(group), *payload.shape)
    all_gather_into_tensor(gathered, payload, group=group)
    return gathered


@torch.library.custom_op("pithtrain::gated_delta_rule_fwd", mutates_args=())
def gdr_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group_name: Optional[str], cp_kernels: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:  # fmt: skip
    """
    cp_kernels (decided once by the caller, so the backward takes the same path): run the
    library's CP kernels instead of FLA's CP path; only with a cp_group_name.
    """
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    if cp_kernels:
        plan = cp_plan(cp_group_name, q.shape[1])
        payload, saved = chunk_gdr_cp_fwd_pre(q, k, v, g, beta, plan)
        o, g_out, A, initial_state = chunk_gdr_cp_fwd_post(cp_all_gather(payload, cp_group_name), saved, plan)  # fmt: skip
        return o, g_out, A, [initial_state]
    cp_context, chunk_indices = cp_state(cp_group_name, q.shape[1])
    g_out, o, A, _, initial_state, _ = chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=q.shape[-1] ** -0.5, initial_state=None, output_final_state=False, cu_seqlens=None if cp_context is None else cp_context.cu_seqlens, cp_context=cp_context, chunk_indices=chunk_indices, state_v_first=False, use_gate_in_kernel=False, A_log=None, dt_bias=None)  # fmt: skip
    # Under CP initial_state is the merged predecessor state; the backward recomputes h from it
    # rather than paying a second all-gather. Without CP it is None.
    return o.to(q.dtype), g_out, A, [] if initial_state is None else [initial_state]


@gdr_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group_name: Optional[str], cp_kernels: bool):  # fmt: skip
    b, s, hv, vd = v.shape
    o = torch.empty((b, s, hv, vd), dtype=q.dtype, device=q.device)
    A = torch.empty((b, s, hv, CHUNK_SIZE), dtype=q.dtype, device=q.device)
    state = [] if cp_group_name is None else [torch.empty((1, hv, q.shape[-1], vd), dtype=torch.float32, device=q.device)]  # fmt: skip
    return o, torch.empty_like(g, dtype=torch.float32), A, state


@torch.library.custom_op("pithtrain::gated_delta_rule_bwd", mutates_args=())
def gdr_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, initial_state: List[torch.Tensor], cp_group_name: Optional[str], cp_kernels: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    q, k, v, g_out, beta, A, do = (t.contiguous() for t in (q, k, v, g_out, beta, A, do))
    if cp_kernels:
        # g_out, A and initial_state are the CP kernels' own, so the forward's path is taken.
        plan = cp_plan(cp_group_name, q.shape[1])
        payload, saved = chunk_gdr_cp_bwd_pre(q, k, v, g_out, beta, A, do, initial_state[0], plan)
        return chunk_gdr_cp_bwd_post(cp_all_gather(payload, cp_group_name), saved, plan)
    cp_context, chunk_indices = cp_state(cp_group_name, q.shape[1])
    dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(q=q, k=k, v=v, g=g_out, beta=beta, A=A, scale=q.shape[-1] ** -0.5, initial_state=initial_state[0] if initial_state else None, do=do, dht=None, cu_seqlens=None if cp_context is None else cp_context.cu_seqlens, cp_context=cp_context, chunk_indices=chunk_indices, state_v_first=False, use_gate_in_kernel=False, g_input=None, A_log=None, dt_bias=None)  # fmt: skip
    return dq.to(q.dtype).contiguous(), dk.to(k.dtype).contiguous(), dv.to(v.dtype).contiguous(), dg.to(g_out.dtype).contiguous(), db.to(beta.dtype).contiguous()  # fmt: skip


@gdr_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, initial_state: List[torch.Tensor], cp_group_name: Optional[str], cp_kernels: bool):  # fmt: skip
    # Contiguous like the real gradients; plain empty_like would inherit the input strides.
    contiguous = torch.contiguous_format
    return torch.empty_like(q, memory_format=contiguous), torch.empty_like(k, memory_format=contiguous), torch.empty_like(v, memory_format=contiguous), torch.empty_like(g_out, memory_format=contiguous), torch.empty_like(beta, memory_format=contiguous)  # fmt: skip


def gdr_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:  # fmt: skip
    q, k, v, _, beta, cp_group_name, cp_kernels = inputs
    _, g_out, A, state = output
    ctx.save_for_backward(q, k, v, g_out, beta, A, *state)
    ctx.cp_group_name, ctx.cp_kernels = cp_group_name, cp_kernels


def gdr_backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor, *_) -> Tuple:
    q, k, v, g_out, beta, A, *state = ctx.saved_tensors
    grads = gdr_bwd(q, k, v, g_out, beta, A, do, state, ctx.cp_group_name, ctx.cp_kernels)
    return *grads, None, None


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
    Qwen3.5's geometry. Calls off the pinned geometry still run FLA (gdr_fwd / gdr_bwd).
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
    = b * heads + h // (value_heads / heads), the same grouping, and the pinned kernels run batch
    b at b times the heads where they take that geometry (SM90); where they take only one sample's
    heads (SM100) the samples run one at a time.
    """
    b = q.shape[0]
    if cp_group is None:
        # Contiguous before the check: the pinned kernels take contiguous tensors, and both paths copy
        # anyway (dense_gated_delta_rule and gdr_fwd).
        folded = tuple(fold_batch(t).contiguous() for t in (q, k, v, g, beta))
        if gated_delta_rule_pinned_supported(*folded):
            return unfold_batch(dense_gated_delta_rule(*folded), b)
        if b > 1 and gated_delta_rule_pinned_supported(*(t[:1].contiguous() for t in (q, k, v, g, beta))):  # fmt: skip
            # The kernels take one sample's heads but not the fold (SM100): one sample at a time.
            return torch.cat([dense_gated_delta_rule(*(t[i : i + 1] for t in (q, k, v, g, beta))) for i in range(b)])  # fmt: skip
        repeats = v.shape[2] // q.shape[2]
        if repeats > 1:
            q = q.repeat_interleave(repeats, dim=2)
            k = k.repeat_interleave(repeats, dim=2)
        o, *_ = gdr_fwd(q, k, v, g, beta, None, False)
        return o
    # Under CP: q and k repeated to the value heads, then folded; the library's CP kernels where
    # chunk_gdr_cp_supported says so, else FLA's context-parallel path.
    repeats = v.shape[2] // q.shape[2]
    if repeats > 1:
        q = q.repeat_interleave(repeats, dim=2)
        k = k.repeat_interleave(repeats, dim=2)
    folded = [fold_batch(t) for t in (q, k, v, g, beta)]
    o, *_ = gdr_fwd(*folded, cp_group.group_name, chunk_gdr_cp_supported(*folded))
    return unfold_batch(o, b)
