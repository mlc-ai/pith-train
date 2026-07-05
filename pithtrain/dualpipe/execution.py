"""
Execution for each stage in the schedule.

Stage Mapping:
    - Stage 1: Attention (LN + Attn + LN + Expert selection)
    - Stage 2: Dispatch (All-to-all dispatch for expert parallelism)
    - Stage 3: MLP (Expert/MLP computation)
    - Stage 4: Combine (All-to-all combine for expert parallelism)
    - Stage 5: Aggregate (Weighted expert output + residual connection)
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx

from pithtrain.dualpipe.utils import WeightGradStore, run_backward
from pithtrain.models.interface import AllToAllSplits, LayerProtocol, ModelProtocol, MoERouting
from pithtrain.operators.all_to_all import direct_all_to_all


@dataclass(init=False, slots=True)
class ExecutionCtx:
    """Shared context for the overlapped forward-backward execution loop."""

    comp_stream: torch.cuda.Stream
    """Main compute stream for forward/backward kernels."""
    comm_stream: torch.cuda.Stream
    """Separate stream for asynchronous all-to-all communication."""
    fwd_event: torch.cuda.Event
    """Event recorded after forward compute; comm_stream waits on it before dispatch."""
    bwd_event: torch.cuda.Event
    """Event recorded after backward compute; comm_stream waits on it before combine."""
    fwd_comm_work: Optional[torch.distributed.Work]
    """Async work handle for the in-flight forward all-to-all (dispatch or combine)."""
    bwd_comm_work: Optional[torch.distributed.Work]
    """Async work handle for the in-flight backward all-to-all."""
    fwd_comm_deferred_free: List[torch.Tensor]
    """Tensors whose storage should be freed after the next fwd_comm_work.wait().

    Callers append tensors here after launching async forward comms (e.g.
    all-to-all in Stage 2 / Stage 4).  The subsequent stage that waits on
    fwd_comm_work drains and frees this list automatically.
    """


# ------------------------------------------------------------
# STAGE1(F/B)
# ------------------------------------------------------------


class Stage1Args(NamedTuple):
    prev_hidden_states: torch.Tensor
    next_hidden_states: torch.Tensor


class Stage1Outs(NamedTuple):
    dispatch_tokens: torch.Tensor
    residual: torch.Tensor
    topk_weight: Optional[torch.Tensor] = None


@dataclass(init=False, slots=True)
class Stage1Record:
    args: Stage1Args
    outs: Stage1Outs


def stage1_f(ctx: ExecutionCtx, layer: LayerProtocol, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]):
    """Stage1 forward."""
    nvtx.range_push("layer%02d.stage1_f" % layer.idx)
    record = Stage1Record()

    prev_hidden_states = hidden_states
    next_hidden_states = hidden_states.detach().requires_grad_()
    record.args = Stage1Args(prev_hidden_states, next_hidden_states)

    dispatch_tokens, residual, routing = layer.forward_stage1(next_hidden_states, rotary_posemb)
    ctx.comp_stream.record_event(ctx.fwd_event)

    topk_weight = routing.topk_weight if routing is not None else None
    record.outs = Stage1Outs(dispatch_tokens, residual, topk_weight)

    nvtx.range_pop()
    return record, dispatch_tokens, residual, routing


def stage1_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage1Record, grad_tensors: tuple):
    """Stage1 backward."""
    nvtx.range_push("layer%02d.stage1_b" % layer.idx)

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    run_backward(record.outs, grad_tensors)

    hidden_states_grad = record.args.next_hidden_states.grad
    record.args.prev_hidden_states.grad = hidden_states_grad

    nvtx.range_pop()
    return hidden_states_grad


# ------------------------------------------------------------
# STAGE2(F/B)
# ------------------------------------------------------------


@dataclass(init=False, slots=True)
class Stage2Record:
    ctx: Optional[tuple]


def stage2_f(ctx: ExecutionCtx, layer: LayerProtocol, dispatch_tokens: torch.Tensor, dispatch_splits: Optional[AllToAllSplits], ep_group: Optional[torch.distributed.ProcessGroup] = None):
    """Stage2 forward: all-to-all dispatch for expert parallelism."""
    nvtx.range_push("layer%02d.stage2_f" % layer.idx)
    record = Stage2Record()

    ctx.comm_stream.wait_event(ctx.fwd_event)

    dispatch_tokens = dispatch_tokens.detach()
    if dispatch_splits is not None:
        with torch.cuda.stream(ctx.comm_stream):
            gathered_tokens = direct_all_to_all(dispatch_tokens, dispatch_splits.output_splits, dispatch_splits.input_splits, ep_group)
        record.ctx = (dispatch_splits, ep_group)
    else:
        gathered_tokens = dispatch_tokens
        record.ctx = None

    ctx.fwd_comm_work = getattr(gathered_tokens, "comm_work", None)
    setattr(gathered_tokens, "comm_work", None)

    nvtx.range_pop()
    return record, gathered_tokens


def stage2_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage2Record, grad_tensors: tuple):
    """Stage2 backward: reverse all-to-all."""
    nvtx.range_push("layer%02d.stage2_b" % layer.idx)

    ctx.comm_stream.wait_event(ctx.bwd_event)

    if record.ctx is not None:
        dispatch_splits, group = record.ctx
        with torch.cuda.stream(ctx.comm_stream):
            dispatch_tokens_grad = direct_all_to_all(grad_tensors[0], dispatch_splits.input_splits, dispatch_splits.output_splits, group)
        ctx.bwd_comm_work = dispatch_tokens_grad.comm_work
        dispatch_tokens_grad.comm_work = None
    else:
        dispatch_tokens_grad = grad_tensors[0]
        ctx.bwd_comm_work = None

    nvtx.range_pop()
    return dispatch_tokens_grad


# ------------------------------------------------------------
# STAGE3(F/B/W)
# ------------------------------------------------------------


class Stage3Args(NamedTuple):
    gathered_tokens: torch.Tensor


class Stage3Outs(NamedTuple):
    moe_outs: torch.Tensor


@dataclass(init=False, slots=True)
class Stage3Record:
    args: Stage3Args
    outs: Stage3Outs


def _drain_deferred_free(ctx: ExecutionCtx) -> None:
    """Free tensor storage that was deferred until after the comm wait."""
    for t in ctx.fwd_comm_deferred_free:
        t.untyped_storage().resize_(0)
    ctx.fwd_comm_deferred_free.clear()


def stage3_f(ctx: ExecutionCtx, layer: LayerProtocol, gathered_tokens: torch.Tensor, expert_idxs: Optional[torch.Tensor], expand_idx: Optional[torch.Tensor] = None):
    """Stage3 forward."""
    nvtx.range_push("layer%02d.stage3_f" % layer.idx)
    record = Stage3Record()

    gathered_tokens = gathered_tokens.detach().requires_grad_()
    record.args = Stage3Args(gathered_tokens)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
    _drain_deferred_free(ctx)

    moe_outs = layer.forward_stage3(gathered_tokens, expert_idxs, expand_idx)
    record.outs = Stage3Outs(moe_outs)
    # Free the args storage - only safe for MoE layers with EP where
    # padded_index_gather is the first consumer and doesn't save the input.
    # When ep_size==1, gathered_tokens shares storage with dispatch_tokens.
    if expert_idxs is not None and ctx.fwd_comm_work is not None:
        gathered_tokens.untyped_storage().resize_(0)

    ctx.comp_stream.record_event(ctx.fwd_event)

    nvtx.range_pop()
    return record, moe_outs


def stage3_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage3Record, grad_tensors: Stage3Outs):
    """Stage3 backward for input."""
    nvtx.range_push("layer%02d.stage3_b" % layer.idx)

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    WeightGradStore.enabled = True
    run_backward(record.outs, grad_tensors)
    WeightGradStore.enabled = False

    ctx.comp_stream.record_event(ctx.bwd_event)

    gathered_tokens_grad = record.args.gathered_tokens.grad

    nvtx.range_pop()
    return gathered_tokens_grad


def stage3_w(ctx: ExecutionCtx, layer: LayerProtocol):
    """Stage3 backward for weight."""
    nvtx.range_push("layer%02d.stage3_w" % layer.idx)

    WeightGradStore.flush()
    WeightGradStore.pop()

    nvtx.range_pop()


# ------------------------------------------------------------
# STAGE4(F/B)
# ------------------------------------------------------------


@dataclass(init=False, slots=True)
class Stage4Record:
    ctx: Optional[tuple]


def stage4_f(ctx: ExecutionCtx, layer: LayerProtocol, moe_outs: torch.Tensor, combine_splits: Optional[AllToAllSplits], ep_group: Optional[torch.distributed.ProcessGroup] = None):
    """Stage4 forward: all-to-all combine for expert parallelism."""
    nvtx.range_push("layer%02d.stage4_f" % layer.idx)
    record = Stage4Record()

    moe_outs = moe_outs.detach()
    ctx.comm_stream.wait_event(ctx.fwd_event)

    if combine_splits is not None:
        with torch.cuda.stream(ctx.comm_stream):
            moe_outs = direct_all_to_all(moe_outs, combine_splits.input_splits, combine_splits.output_splits, ep_group)
        record.ctx = (combine_splits, ep_group)
    else:
        record.ctx = None

    ctx.fwd_comm_work = getattr(moe_outs, "comm_work", None)
    setattr(moe_outs, "comm_work", None)

    nvtx.range_pop()
    return record, moe_outs


def stage4_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage4Record, grad_tensors: tuple):
    """Stage4 backward: reverse all-to-all."""
    nvtx.range_push("layer%02d.stage4_b" % layer.idx)

    ctx.comm_stream.wait_event(ctx.bwd_event)

    if record.ctx is not None:
        combine_splits, group = record.ctx
        with torch.cuda.stream(ctx.comm_stream):
            moe_outs_grad = direct_all_to_all(grad_tensors[0], combine_splits.output_splits, combine_splits.input_splits, group)
        ctx.bwd_comm_work = moe_outs_grad.comm_work
        moe_outs_grad.comm_work = None
    else:
        moe_outs_grad = grad_tensors[0]
        ctx.bwd_comm_work = None

    nvtx.range_pop()
    return moe_outs_grad


# ------------------------------------------------------------
# STAGE5(F/B)
# ------------------------------------------------------------


class Stage5Args(NamedTuple):
    moe_outs: torch.Tensor
    topk_weight: torch.Tensor
    residual: torch.Tensor


class Stage5Outs(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class Stage5Record:
    args: Stage5Args
    outs: Stage5Outs


def stage5_f(ctx: ExecutionCtx, layer: LayerProtocol, moe_outs: torch.Tensor, routing: Optional[MoERouting], residual: torch.Tensor):
    """Stage5 forward."""
    nvtx.range_push("layer%02d.stage5_f" % layer.idx)
    record = Stage5Record()

    moe_outs = moe_outs.detach().requires_grad_()
    topk_weight = routing.topk_weight if routing is not None else None
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    residual = residual.detach().requires_grad_()
    record.args = Stage5Args(moe_outs, topk_weight, residual)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
    _drain_deferred_free(ctx)

    moe_local_idxs = routing.moe_local_idxs if routing is not None else None
    hidden_states = layer.forward_stage5(moe_outs, moe_local_idxs, topk_weight, residual)
    record.outs = Stage5Outs(hidden_states)

    nvtx.range_pop()
    return record, hidden_states


def stage5_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage5Record, grad_tensors: Stage5Outs):
    """Stage5 backward."""
    nvtx.range_push("layer%02d.stage5_b" % layer.idx)

    run_backward(record.outs, grad_tensors)

    ctx.comp_stream.record_event(ctx.bwd_event)

    moe_outs_grad, topk_weight_grad, residual_grad = [t.grad if t is not None else None for t in record.args]

    nvtx.range_pop()
    return moe_outs_grad, topk_weight_grad, residual_grad


# ------------------------------------------------------------
# STAGE5_AND_STAGE1(F/B) - Merged stage 5 + stage 1
# ------------------------------------------------------------


def stage5_and_stage1_f(ctx: ExecutionCtx, prev_layer: LayerProtocol, next_layer: LayerProtocol, moe_outs: torch.Tensor, routing: Optional[MoERouting], residual: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]):
    """
    Merged Stage5 and Stage1 forward.
    Returns (stage5_args, stage1_outs, dispatch_tokens, residual, routing) for the next layer.
    """
    nvtx.range_push("layer%02d_stage5_f_layer%02d_stage1_f" % (prev_layer.idx, next_layer.idx))

    moe_outs = moe_outs.detach().requires_grad_()
    topk_weight = routing.topk_weight if routing is not None else None
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    residual = residual.detach().requires_grad_()
    stage5_args = Stage5Args(moe_outs, topk_weight, residual)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
    _drain_deferred_free(ctx)

    moe_local_idxs = routing.moe_local_idxs if routing is not None else None
    hidden_states = prev_layer.forward_stage5(moe_outs, moe_local_idxs, topk_weight, residual)

    dispatch_tokens, next_residual, next_routing = next_layer.forward_stage1(hidden_states, rotary_posemb)
    ctx.comp_stream.record_event(ctx.fwd_event)

    next_topk_weight = next_routing.topk_weight if next_routing is not None else None
    stage1_outs = Stage1Outs(dispatch_tokens, next_residual, next_topk_weight)

    nvtx.range_pop()
    return stage5_args, stage1_outs, dispatch_tokens, next_residual, next_routing


def stage5_and_stage1_b(ctx: ExecutionCtx, next_layer: LayerProtocol, prev_layer: LayerProtocol, stage1_outs: Stage1Outs, stage5_args: Stage5Args, grad_tensors: tuple):
    """
    Merged Stage5 and Stage1 backward.
    Takes stage1_outs (from next layer) and stage5_args (from prev layer) separately.
    """
    nvtx.range_push("layer%02d_stage5_b_layer%02d_stage1_b" % (prev_layer.idx, next_layer.idx))

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    run_backward(stage1_outs, grad_tensors)

    ctx.comp_stream.record_event(ctx.bwd_event)

    moe_outs_grad, topk_weight_grad, residual_grad = [t.grad if t is not None else None for t in stage5_args]

    nvtx.range_pop()
    return moe_outs_grad, topk_weight_grad, residual_grad


# ------------------------------------------------------------
# PROLOG(F/B)
# ------------------------------------------------------------


class PrologArgs(NamedTuple):
    pass


class PrologOuts(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class PrologRecord:
    args: PrologArgs
    outs: PrologOuts


def prolog_f(module: ModelProtocol, hidden_states: torch.Tensor, record: PrologRecord) -> torch.Tensor:
    """Prolog forward: embed the input tokens, recording into ``record`` for the backward."""
    nvtx.range_push("prolog_f")
    record.args = PrologArgs()
    hidden_states = module.forward_prolog(hidden_states)
    record.outs = PrologOuts(hidden_states)
    nvtx.range_pop()
    return hidden_states


def prolog_b(module: ModelProtocol, record: PrologRecord, grad_tensors: PrologOuts):
    """Prolog backward."""
    nvtx.range_push("prolog_b")

    run_backward(record.outs, grad_tensors)

    nvtx.range_pop()
    return


# ------------------------------------------------------------
# EPILOG(F/B)
# ------------------------------------------------------------


class EpilogArgs(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class EpilogRecord:
    args: EpilogArgs


def epilog_f(module: ModelProtocol, hidden_states: torch.Tensor, record: EpilogRecord) -> torch.Tensor:
    """
    Epilog forward: norm + lm_head, recording its input activation into ``record``.

    The backward is handled by ``loss.backward()`` which traverses the autograd
    graph through norm -> lm_head -> criterion.  The only thing the caller needs
    from the record is ``args.hidden_states.grad`` (populated by autograd).
    """
    nvtx.range_push("epilog_f")
    hidden_states = hidden_states.detach().requires_grad_()
    record.args = EpilogArgs(hidden_states)
    logits = module.forward_epilog(hidden_states)
    nvtx.range_pop()
    return logits


# ------------------------------------------------------------
# INTERMEDIATE TENSORS
# ------------------------------------------------------------


@dataclass(init=False, slots=True)
class IntermediateTensorsLayer:
    stage1: Stage1Record
    stage2: Stage2Record
    stage3: Stage3Record
    stage4: Stage4Record
    stage5: Stage5Record


@dataclass(init=False, slots=True)
class IntermediateTensors:
    prolog: Optional[PrologRecord]
    epilog: Optional[EpilogRecord]
    layers: List[IntermediateTensorsLayer]


def create_intermediate_tensors_layer() -> IntermediateTensorsLayer:
    """Create a pre-allocated IntermediateTensorsLayer with all records."""
    layer = IntermediateTensorsLayer()
    layer.stage1 = Stage1Record()
    layer.stage2 = Stage2Record()
    layer.stage2.ctx = None
    layer.stage3 = Stage3Record()
    layer.stage4 = Stage4Record()
    layer.stage4.ctx = None
    layer.stage5 = Stage5Record()
    return layer


def create_intermediate_tensors(num_layers: int, has_prolog: bool, has_epilog: bool) -> IntermediateTensors:
    """Create a pre-allocated IntermediateTensors structure for reuse across iterations."""
    tensors = IntermediateTensors()
    tensors.prolog = PrologRecord() if has_prolog else None
    tensors.epilog = EpilogRecord() if has_epilog else None
    tensors.layers = [create_intermediate_tensors_layer() for _ in range(num_layers)]
    return tensors
