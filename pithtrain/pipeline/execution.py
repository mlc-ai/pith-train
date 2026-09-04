"""
Execution for each stage in the schedule.

Each decoder layer is split into five stages so the pipeline scheduler can interleave different
micro-batches and overlap the compute of one with the communication of another.

- Stage 1: pre-dispatch compute.
- Stage 2: dispatch all-to-all.
- Stage 3: expert compute.
- Stage 4: combine all-to-all.
- Stage 5: post-combine compute.

WeightGradStore and run_backward are derived from dualpipe/utils.py in DeepSeek's
DualPipe project (https://github.com/deepseek-ai/DualPipe), licensed under the MIT License.
Copyright (c) 2025 DeepSeek. See pithtrain/pipeline/LICENSE for the full license text. The
five-stage decomposition is an original addition.
"""

import queue
from dataclasses import dataclass, field, fields
from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed
from torch.autograd import Variable

from pithtrain.contexts import distributed
from pithtrain.models.interface import AllToAllSplits, LayerProtocol, ModelProtocol, RoutingInfo
from pithtrain.operators.all_to_all import direct_all_to_all

# fmt: off

# ------------------------------------------------------------
# DEFERRED WEIGHT GRADIENTS / MANUAL BACKWARD
# ------------------------------------------------------------


class WeightGradStore:
    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    pairs = [(t, g) for t, g in zip(tensors, grad_tensors) if t is not None]
    if not pairs:
        return
    tensors, grad_tensors = map(tuple, zip(*pairs))
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    with torch.autograd.set_multithreading_enabled(False):
        Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)


@dataclass(slots=True)
class ExecutionCtx:
    """
    Shared context for the overlapped forward-backward execution loop.
    """

    comm_stream: torch.cuda.Stream
    """Separate stream for asynchronous all-to-all communication."""
    comp_stream: torch.cuda.Stream = field(default_factory=torch.cuda.default_stream)
    """Main compute stream for forward/backward kernels."""
    fwd_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    """Event recorded after forward compute; comm_stream waits on it before dispatch."""
    bwd_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    """Event recorded after backward compute; comm_stream waits on it before combine."""
    fwd_comm_work: Optional[torch.distributed.Work] = None
    """Async work handle for the in-flight forward all-to-all (dispatch or combine)."""
    bwd_comm_work: Optional[torch.distributed.Work] = None
    """Async work handle for the in-flight backward all-to-all."""
    fwd_comm_deferred_free: List[torch.Tensor] = field(default_factory=list)
    """
    Tensors whose storage should be freed after the next fwd_comm_work.wait().

    Callers append tensors here after launching async forward comms (e.g. all-to-all in
    Stage 2 / Stage 4). The subsequent stage that waits on fwd_comm_work drains and frees
    this list automatically.
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


@dataclass(slots=True)
class Stage1Record:
    args: Optional[Stage1Args] = None
    outs: Optional[Stage1Outs] = None


def stage1_f(ctx: ExecutionCtx, layer: LayerProtocol, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor], cu_seqlens: Optional[torch.Tensor] = None):
    """
    Stage1 forward.
    """
    nvtx.range_push("layer%02d.stage1_f" % layer.idx)
    record = Stage1Record()

    prev_hidden_states = hidden_states
    next_hidden_states = hidden_states.detach().requires_grad_()
    record.args = Stage1Args(prev_hidden_states, next_hidden_states)

    dispatch_tokens, residual, routing = layer.forward_stage1(next_hidden_states, rotary_posemb, cu_seqlens)
    ctx.comp_stream.record_event(ctx.fwd_event)

    topk_weight = routing.topk_weight if routing is not None else None
    record.outs = Stage1Outs(dispatch_tokens, residual, topk_weight)

    nvtx.range_pop()
    return record, dispatch_tokens, residual, routing


def stage1_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage1Record, grad_tensors: tuple):
    """
    Stage1 backward.
    """
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


@dataclass(slots=True)
class Stage2Record:
    splits: Optional[AllToAllSplits] = None


def stage2_f(ctx: ExecutionCtx, layer: LayerProtocol, dispatch_tokens: torch.Tensor, dispatch_splits: Optional[AllToAllSplits]):
    """
    Stage2 forward: all-to-all dispatch for expert parallelism.
    """
    nvtx.range_push("layer%02d.stage2_f" % layer.idx)
    record = Stage2Record()

    ctx.comm_stream.wait_event(ctx.fwd_event)

    dispatch_tokens = dispatch_tokens.detach()
    record.splits = dispatch_splits
    if dispatch_splits is not None:
        with torch.cuda.stream(ctx.comm_stream):
            gathered_tokens = direct_all_to_all(dispatch_tokens, dispatch_splits.output_splits, dispatch_splits.input_splits, distributed.ep_group)
    else:
        gathered_tokens = dispatch_tokens

    ctx.fwd_comm_work = getattr(gathered_tokens, "comm_work", None)
    setattr(gathered_tokens, "comm_work", None)

    nvtx.range_pop()
    return record, gathered_tokens


def stage2_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage2Record, grad_tensors: tuple):
    """
    Stage2 backward: reverse all-to-all.
    """
    nvtx.range_push("layer%02d.stage2_b" % layer.idx)

    ctx.comm_stream.wait_event(ctx.bwd_event)

    if record.splits is not None:
        with torch.cuda.stream(ctx.comm_stream):
            dispatch_tokens_grad = direct_all_to_all(grad_tensors[0], record.splits.input_splits, record.splits.output_splits, distributed.ep_group)
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


@dataclass(slots=True)
class Stage3Record:
    args: Optional[Stage3Args] = None
    outs: Optional[Stage3Outs] = None


def _drain_deferred_free(ctx: ExecutionCtx) -> None:
    """
    Free tensor storage that was deferred until after the comm wait.
    """
    for t in ctx.fwd_comm_deferred_free:
        t.untyped_storage().resize_(0)
    ctx.fwd_comm_deferred_free.clear()


def stage3_f(ctx: ExecutionCtx, layer: LayerProtocol, gathered_tokens: torch.Tensor, expert_idxs: Optional[torch.Tensor], expand_idx: Optional[torch.Tensor] = None):
    """
    Stage3 forward.
    """
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


def stage3_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage3Record, grad_tensors: tuple):
    """
    Stage3 backward for input.
    """
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


def stage3_w(layer: LayerProtocol):
    """
    Stage3 backward for weight.
    """
    nvtx.range_push("layer%02d.stage3_w" % layer.idx)

    WeightGradStore.flush()
    WeightGradStore.pop()

    nvtx.range_pop()


# ------------------------------------------------------------
# STAGE4(F/B)
# ------------------------------------------------------------


@dataclass(slots=True)
class Stage4Record:
    splits: Optional[AllToAllSplits] = None


def stage4_f(ctx: ExecutionCtx, layer: LayerProtocol, moe_outs: torch.Tensor, combine_splits: Optional[AllToAllSplits]):
    """
    Stage4 forward: all-to-all combine for expert parallelism.
    """
    nvtx.range_push("layer%02d.stage4_f" % layer.idx)
    record = Stage4Record()

    moe_outs = moe_outs.detach()
    ctx.comm_stream.wait_event(ctx.fwd_event)

    record.splits = combine_splits
    if combine_splits is not None:
        with torch.cuda.stream(ctx.comm_stream):
            moe_outs = direct_all_to_all(moe_outs, combine_splits.input_splits, combine_splits.output_splits, distributed.ep_group)

    ctx.fwd_comm_work = getattr(moe_outs, "comm_work", None)
    setattr(moe_outs, "comm_work", None)

    nvtx.range_pop()
    return record, moe_outs


def stage4_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage4Record, grad_tensors: tuple):
    """
    Stage4 backward: reverse all-to-all.
    """
    nvtx.range_push("layer%02d.stage4_b" % layer.idx)

    ctx.comm_stream.wait_event(ctx.bwd_event)

    if record.splits is not None:
        with torch.cuda.stream(ctx.comm_stream):
            moe_outs_grad = direct_all_to_all(grad_tensors[0], record.splits.output_splits, record.splits.input_splits, distributed.ep_group)
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


@dataclass(slots=True)
class Stage5Record:
    args: Optional[Stage5Args] = None
    outs: Optional[Stage5Outs] = None


def stage5_f(ctx: ExecutionCtx, layer: LayerProtocol, moe_outs: torch.Tensor, routing: Optional[RoutingInfo], residual: torch.Tensor):
    """
    Stage5 forward.
    """
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


def stage5_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage5Record, grad_tensors: tuple):
    """
    Stage5 backward.
    """
    nvtx.range_push("layer%02d.stage5_b" % layer.idx)

    run_backward(record.outs, grad_tensors)

    ctx.comp_stream.record_event(ctx.bwd_event)

    moe_outs_grad, topk_weight_grad, residual_grad = [t.grad if t is not None else None for t in record.args]

    nvtx.range_pop()
    return moe_outs_grad, topk_weight_grad, residual_grad


# ------------------------------------------------------------
# STAGE5_STAGE1(F) / STAGE1_STAGE5(B) - the fused pair across a layer boundary
# ------------------------------------------------------------


def stage5_stage1_f(ctx: ExecutionCtx, prev_layer: LayerProtocol, next_layer: LayerProtocol, moe_outs: torch.Tensor, routing: Optional[RoutingInfo], residual: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor], cu_seqlens: Optional[torch.Tensor] = None):
    """
    Fused forward: prev_layer stage5, then next_layer stage1.

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

    dispatch_tokens, next_residual, next_routing = next_layer.forward_stage1(hidden_states, rotary_posemb, cu_seqlens)
    ctx.comp_stream.record_event(ctx.fwd_event)

    next_topk_weight = next_routing.topk_weight if next_routing is not None else None
    stage1_outs = Stage1Outs(dispatch_tokens, next_residual, next_topk_weight)

    nvtx.range_pop()
    return stage5_args, stage1_outs, dispatch_tokens, next_residual, next_routing


def stage1_stage5_b(ctx: ExecutionCtx, next_layer: LayerProtocol, prev_layer: LayerProtocol, stage1_outs: Stage1Outs, stage5_args: Stage5Args, grad_tensors: tuple):
    """
    Fused backward: next_layer stage1, then prev_layer stage5.

    Takes stage1_outs (from the next layer) and stage5_args (from the previous layer) separately.
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


class PrologOuts(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(slots=True)
class PrologRecord:
    outs: Optional[PrologOuts] = None


def prolog_f(module: ModelProtocol, hidden_states: torch.Tensor, record: PrologRecord) -> torch.Tensor:
    """
    Prolog forward: embed the input tokens, recording into record for the backward.
    """
    nvtx.range_push("prolog_f")
    hidden_states = module.forward_prolog(hidden_states)
    record.outs = PrologOuts(hidden_states)
    nvtx.range_pop()
    return hidden_states


def prolog_b(module: ModelProtocol, record: PrologRecord, grad_tensors: tuple):
    """
    Prolog backward.
    """
    nvtx.range_push("prolog_b")

    run_backward(record.outs, grad_tensors)

    nvtx.range_pop()


# ------------------------------------------------------------
# EPILOG(F/B)
# ------------------------------------------------------------


class EpilogArgs(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(slots=True)
class EpilogRecord:
    args: Optional[EpilogArgs] = None


def epilog_f(module: ModelProtocol, hidden_states: torch.Tensor, record: EpilogRecord) -> torch.Tensor:
    """
    Epilog forward: norm + lm_head, recording its input activation into record.

    The backward is handled by loss.backward() which traverses the autograd
    graph through norm -> lm_head -> objective.  The only thing the caller needs
    from the record is args.hidden_states.grad (populated by autograd).
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
class LayerRecord:
    stage1: Stage1Record
    stage2: Stage2Record
    stage3: Stage3Record
    stage4: Stage4Record
    stage5: Stage5Record


@dataclass(init=False, slots=True)
class ChunkRecord:
    prolog: Optional[PrologRecord]
    epilog: Optional[EpilogRecord]
    layers: List[LayerRecord]


def create_layer_record() -> LayerRecord:
    """
    Create a pre-allocated LayerRecord with all records.
    """
    layer = LayerRecord()
    layer.stage1 = Stage1Record()
    layer.stage2 = Stage2Record()
    layer.stage3 = Stage3Record()
    layer.stage4 = Stage4Record()
    layer.stage5 = Stage5Record()
    return layer


def create_chunk_record(num_layers: int, has_prolog: bool, has_epilog: bool) -> ChunkRecord:
    """
    Create a pre-allocated ChunkRecord structure for reuse across iterations.
    """
    record = ChunkRecord()
    record.prolog = PrologRecord() if has_prolog else None
    record.epilog = EpilogRecord() if has_epilog else None
    record.layers = [create_layer_record() for _ in range(num_layers)]
    return record


def clear_layer_records(layer: LayerRecord) -> None:
    """
    Clear tensor references from a layer's records while keeping records pre-allocated.
    """
    for f in fields(layer):
        record = getattr(layer, f.name)
        for rf in fields(record):
            setattr(record, rf.name, None)


# ------------------------------------------------------------
# SEQUENTIAL (NON-OVERLAPPED) LAYER + CHUNK PASSES
# ------------------------------------------------------------


def layer_forward_dispatch(dispatch_tokens: torch.Tensor, dispatch_splits: Optional[AllToAllSplits]):
    """
    All-to-all dispatch.
    """
    if dispatch_splits is None:
        return dispatch_tokens
    return direct_all_to_all(dispatch_tokens, dispatch_splits.output_splits, dispatch_splits.input_splits, distributed.ep_group)


def layer_forward_combine(outs: torch.Tensor, combine_splits: Optional[AllToAllSplits]):
    """
    All-to-all combine.
    """
    if combine_splits is None:
        return outs
    return direct_all_to_all(outs, combine_splits.input_splits, combine_splits.output_splits, distributed.ep_group)


def layer_forward(
    layer: LayerProtocol,
    hidden_states: torch.Tensor,
    rotary_posemb: Tuple[torch.Tensor, torch.Tensor],
    layer_record: LayerRecord,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Forward pass for a DualPipeV decoder layer.

    Records each stage's tensors into layer_record for the pipeline backward.
    """

    # Stage 1.
    nvtx.range_push("layer%02d.stage1_f" % layer.idx)
    record = Stage1Record()
    prev_hidden_states = hidden_states
    next_hidden_states = hidden_states.detach().requires_grad_()
    record.args = Stage1Args(prev_hidden_states, next_hidden_states)

    dispatch_tokens, residual, routing = layer.forward_stage1(next_hidden_states, rotary_posemb, cu_seqlens)

    has_experts = routing is not None

    record.outs = Stage1Outs(dispatch_tokens, residual, routing.topk_weight if has_experts else None)
    layer_record.stage1 = record
    nvtx.range_pop()

    # Stage 2.
    nvtx.range_push("layer%02d.stage2_f" % layer.idx)
    record = Stage2Record()
    record.splits = routing.dispatch_splits if has_experts else None
    gathered_tokens = layer_forward_dispatch(dispatch_tokens.detach(), record.splits)
    fwd_comm_work = getattr(gathered_tokens, "comm_work", None)
    setattr(gathered_tokens, "comm_work", None)
    layer_record.stage2 = record
    nvtx.range_pop()

    # Stage 3.
    nvtx.range_push("layer%02d.stage3_f" % layer.idx)
    record = Stage3Record()
    gathered_tokens = gathered_tokens.detach().requires_grad_()
    record.args = Stage3Args(gathered_tokens)

    if fwd_comm_work is not None:
        fwd_comm_work.wait()
    # Stage 2 all-to-all has completed - dispatch_tokens storage is no longer read.
    # Free it now; run_backward only needs the grad_fn chain, not the values.
    # Guard: only when a2a actually occurred (ep_size > 1); otherwise dispatch_tokens
    # and gathered_tokens share storage.
    if has_experts and fwd_comm_work is not None:
        dispatch_tokens.untyped_storage().resize_(0)

    moe_outs = layer.forward_stage3(gathered_tokens, routing.expert_idxs if has_experts else None, routing.expand_idx if has_experts else None)

    record.outs = Stage3Outs(moe_outs)
    # Free args storage - values no longer needed, only .grad is read after backward.
    # Only safe for MoE layers with EP: padded_index_gather is the first consumer and
    # doesn't save the input.  For dense layers or ep_size==1, gate_proj/up_proj may
    # save gathered_tokens directly, or it shares storage with dispatch_tokens.
    if has_experts and fwd_comm_work is not None:
        gathered_tokens.untyped_storage().resize_(0)
    layer_record.stage3 = record
    nvtx.range_pop()

    # Stage 4.
    nvtx.range_push("layer%02d.stage4_f" % layer.idx)
    record = Stage4Record()
    record.splits = routing.combine_splits if has_experts else None
    moe_outs = layer_forward_combine(moe_outs.detach(), record.splits)
    fwd_comm_work = getattr(moe_outs, "comm_work", None)
    setattr(moe_outs, "comm_work", None)
    layer_record.stage4 = record
    nvtx.range_pop()

    # Stage 5.
    nvtx.range_push("layer%02d.stage5_f" % layer.idx)
    record = Stage5Record()
    moe_outs = moe_outs.detach().requires_grad_()
    topk_weight = routing.topk_weight if has_experts else None
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    residual = residual.detach().requires_grad_()
    record.args = Stage5Args(moe_outs, topk_weight, residual)

    if fwd_comm_work is not None:
        fwd_comm_work.wait()
    # Stage 4 all-to-all has completed - Stage 3 output is no longer read.
    if has_experts and fwd_comm_work is not None:
        layer_record.stage3.outs.moe_outs.untyped_storage().resize_(0)
    moe_local_idxs = routing.moe_local_idxs if has_experts else None
    hidden_states = layer.forward_stage5(moe_outs, moe_local_idxs, topk_weight, residual)

    record.outs = Stage5Outs(hidden_states)
    layer_record.stage5 = record
    nvtx.range_pop()

    return hidden_states


def layer_backward(
    layer: LayerProtocol,
    output_grads: Optional[List[torch.Tensor]],
    layer_record: LayerRecord,
):
    """
    Backward pass for a DualPipeV decoder layer.

    Handles both normal and merged cases using asymmetric None pattern:
    - Merged stage1: stage1.outs is set, stage1.args is None
      -> Run backward on stage1.outs, grads flow to prev layer's stage5.args
      -> Return None to signal prev layer to get grads from stage5.args
    - Merged stage5: stage5.args is set, stage5.outs is None
      -> Get grads from stage5.args.*.grad (already computed by next layer)
    """

    stage1_record = layer_record.stage1
    stage5_record = layer_record.stage5
    stage5_was_merged = stage5_record.args is not None and stage5_record.outs is None
    stage1_is_merged = stage1_record.outs is not None and stage1_record.args is None

    # Stage 5.
    if stage5_was_merged:
        nvtx.range_push("layer%02d.stage5_merged_skip" % layer.idx)
    else:
        nvtx.range_push("layer%02d.stage5_b" % layer.idx)
        run_backward(stage5_record.outs, output_grads)
    moe_outs_grad, topk_weight_grad, residual_grad = [t.grad if t is not None else None for t in stage5_record.args]
    nvtx.range_pop()

    # Stage 4.
    nvtx.range_push("layer%02d.stage4_b" % layer.idx)
    record = layer_record.stage4
    if record.splits is not None:
        moe_outs_grad = direct_all_to_all(moe_outs_grad, record.splits.output_splits, record.splits.input_splits, distributed.ep_group)
        bwd_comm_work = moe_outs_grad.comm_work
        moe_outs_grad.comm_work = None
    else:
        bwd_comm_work = None
    nvtx.range_pop()

    # Stage 3.
    nvtx.range_push("layer%02d.stage3_b" % layer.idx)
    record = layer_record.stage3

    if bwd_comm_work is not None:
        bwd_comm_work.wait()

    run_backward(record.outs, (moe_outs_grad,))
    gathered_tokens_grad = record.args.gathered_tokens.grad
    nvtx.range_pop()

    # Stage 2.
    nvtx.range_push("layer%02d.stage2_b" % layer.idx)
    record = layer_record.stage2
    if record.splits is not None:
        dispatch_tokens_grad = direct_all_to_all(gathered_tokens_grad, record.splits.input_splits, record.splits.output_splits, distributed.ep_group)
        bwd_comm_work = dispatch_tokens_grad.comm_work
        dispatch_tokens_grad.comm_work = None
    else:
        dispatch_tokens_grad = gathered_tokens_grad
        bwd_comm_work = None
    nvtx.range_pop()

    # Stage 1.
    nvtx.range_push("layer%02d.stage1_b" % layer.idx)
    if bwd_comm_work is not None:
        bwd_comm_work.wait()

    run_backward(stage1_record.outs, (dispatch_tokens_grad, residual_grad, topk_weight_grad))
    if stage1_is_merged:
        # Grads landed on the previous layer's stage5.args, which reads them itself.
        input_grads = None
    else:
        input_grads = stage1_record.args.next_hidden_states.grad
        stage1_record.args.prev_hidden_states.grad = input_grads
    nvtx.range_pop()

    clear_layer_records(layer_record)
    return input_grads


def model_forward(
    module: ModelProtocol,
    hidden_states: torch.Tensor,
    chunk_record: ChunkRecord,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sequential (non-overlapped) forward for one pipeline chunk: prolog -> layers -> epilog.

    Records each stage's tensors into chunk_record for the pipeline backward.
    """
    if module.stage_index == 0:
        hidden_states = prolog_f(module, hidden_states, chunk_record.prolog)

    rotary_posemb = module.forward_posemb(hidden_states.shape[1], cu_seqlens)
    for layer, layer_record in zip(module.layers.values(), chunk_record.layers):
        hidden_states = layer_forward(layer, hidden_states, rotary_posemb, layer_record, cu_seqlens)

    if module.stage_index == module.stage_count - 1:
        hidden_states = epilog_f(module, hidden_states, chunk_record.epilog)

    return hidden_states


def model_backward(
    module: ModelProtocol,
    output_grads: Optional[List[torch.Tensor]],
    loss: Optional[torch.Tensor],
    chunk_record: ChunkRecord,
):
    """
    Sequential (non-overlapped) backward for one pipeline chunk: epilog -> layers -> prolog.

    Backprops through the tensors model_forward saved in chunk_record and
    returns the input gradients to hand back to the previous pipeline stage.
    """
    if loss is not None:
        loss.backward()
        loss.detach_()
        output_grads = (chunk_record.epilog.args.hidden_states.grad,)
        chunk_record.epilog.args = None

    input_grads = output_grads
    for layer, layer_record in zip(
        reversed(list(module.layers.values())), reversed(chunk_record.layers)
    ):
        input_grads = (layer_backward(layer, input_grads, layer_record),)

    final_grads = input_grads
    if module.stage_index == 0:
        record = chunk_record.prolog
        run_backward(record.outs, input_grads)
        record.outs = None
        final_grads = (None,)
    return final_grads
