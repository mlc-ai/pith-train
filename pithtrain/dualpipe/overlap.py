"""
Overlapped forward-backward execution loop for DualPipeV.

Interleaves the forward pass of one model replica with the backward pass
of another, decomposing each transformer layer into 5 stages
(Attention / Dispatch / MLP / Combine / Aggregate) for fine-grained
computation-communication overlap.
"""

from dataclasses import fields
from typing import Callable, List, Optional

import torch
import torch.cuda.nvtx as nvtx

from pithtrain.dualpipe.execution import (
    ChunkRecord,
    ExecutionCtx,
    LayerRecord,
    epilog_f,
    layer_backward,
    layer_forward,
    prolog_b,
    prolog_f,
    stage1_b,
    stage1_f,
    stage2_b,
    stage2_f,
    stage3_b,
    stage3_f,
    stage3_w,
    stage4_b,
    stage4_f,
    stage5_and_stage1_b,
    stage5_and_stage1_f,
    stage5_b,
    stage5_f,
)
from pithtrain.models.interface import ModelProtocol


def _clear_layer_records(layer: LayerRecord) -> None:
    """Clear tensor references from a layer's records while keeping records pre-allocated."""
    for field in fields(layer):
        record = getattr(layer, field.name)
        for rf in fields(record):
            setattr(record, rf.name, None)


def overlapped_forward_backward(
    module0: ModelProtocol,
    inputs0: List[torch.Tensor],
    criterion0: Optional[Callable],
    labels0: Optional[List[torch.Tensor]],
    chunk_record0: ChunkRecord,
    cu_seqlens0: Optional[torch.Tensor],
    module1: ModelProtocol,
    loss1: Optional[torch.Tensor],
    outputs1: Optional[List[torch.Tensor]],
    output_grads1: Optional[List[torch.Tensor]],
    chunk_record1: ChunkRecord,
    comm_stream: Optional[torch.cuda.Stream],
    ep_group: Optional[torch.distributed.ProcessGroup] = None,
):
    assert abs(len(module0.layers) - len(module1.layers)) <= 1
    assert len(chunk_record1.layers) == len(module1.layers)
    num_layers = min(len(module0.layers), len(module1.layers))
    module0_layers = [layer for _, layer in module0.layers.items()]
    module1_layers = [layer for _, layer in module1.layers.items()]

    (hidden_states,) = inputs0
    cu_seqlens = cu_seqlens0
    # chunk_record0 is pre-allocated and passed in
    layer_idx0 = 0  # Index into chunk_record0.layers

    ctx = ExecutionCtx()
    ctx.comp_stream = torch.cuda.default_stream()
    ctx.comm_stream = comm_stream
    ctx.fwd_event = torch.cuda.Event()
    ctx.bwd_event = torch.cuda.Event()
    ctx.fwd_comm_work = None
    ctx.bwd_comm_work = None
    ctx.fwd_comm_deferred_free = []

    # Module 1 layer L-1 stage 5 backward
    if loss1 is not None:
        nvtx.range_push("loss1.backward()")
        loss1.backward()
        loss1.detach_()
        nvtx.range_pop()
        output_grads1 = [chunk_record1.epilog.args.hidden_states.grad]
        # Clear tensor refs but keep pre-allocated record
        chunk_record1.epilog.args = None
        loss1 = None
    assert output_grads1 is not None

    record = chunk_record1.layers[-1].stage5
    moe_outs_grad, topk_weight_grad, residual_grad = stage5_b(
        ctx, module1_layers[-1], record, tuple(output_grads1)
    )

    # Module 1 layer L-1 stage 4 backward
    record = chunk_record1.layers[-1].stage4
    moe_outs_grad = stage4_b(ctx, module1_layers[-1], record, (moe_outs_grad,))

    # Module 0 layer 0 stage 1 forward
    if module0.stage_index == 0:
        hidden_states = prolog_f(module0, hidden_states, chunk_record0.prolog)

    rotary_posemb = module0.forward_posemb(hidden_states.shape[1], cu_seqlens)

    record, dispatch_tokens, residual, routing = stage1_f(
        ctx, module0_layers[0], hidden_states, rotary_posemb, cu_seqlens
    )
    chunk_record0.layers[layer_idx0].stage1.args = record.args
    chunk_record0.layers[layer_idx0].stage1.outs = record.outs

    for l in range(num_layers):  # noqa: E741
        if l != 0:
            # Detect merged case using asymmetric None pattern:
            # - Merged: stage1.outs is set, stage1.args is None (at next layer)
            #           stage5.args is set, stage5.outs is None (at prev layer)
            # - Normal: both args and outs are set for both stage1 and stage5
            stage1_record = chunk_record1.layers[-l].stage1
            use_merged = (
                hasattr(stage1_record, "outs")
                and stage1_record.outs is not None
                and not (hasattr(stage1_record, "args") and stage1_record.args is not None)
            )

            if use_merged:
                next_layer, prev_layer = module1_layers[-l], module1_layers[-l - 1]
                stage1_outs = stage1_record.outs
                stage5_args = chunk_record1.layers[-l - 1].stage5.args
                grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)  # noqa: F821
                moe_outs_grad, topk_weight_grad, residual_grad = stage5_and_stage1_b(
                    ctx, next_layer, prev_layer, stage1_outs, stage5_args, grad_tensors
                )
                # Clear tensor refs but keep pre-allocated records
                _clear_layer_records(chunk_record1.layers[-l])
            else:
                # Module 1 layer L-l stage 1 backward
                record = chunk_record1.layers[-l].stage1
                grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)  # noqa: F821
                hidden_states_grad = stage1_b(ctx, module1_layers[-l], record, grad_tensors)
                # Module 1 layer L-l-1 stage 5 backward
                record = chunk_record1.layers[-l - 1].stage5
                moe_outs_grad, topk_weight_grad, residual_grad = stage5_b(
                    ctx, module1_layers[-l - 1], record, (hidden_states_grad,)
                )
                # Clear tensor refs but keep pre-allocated records
                _clear_layer_records(chunk_record1.layers[-l])

            # Module 0 layer l-1 stage 4 forward
            record, moe_outs = stage4_f(
                ctx,
                module0_layers[l - 1],
                moe_outs,  # noqa: F821
                routing.combine_splits if routing is not None else None,
                ep_group,
            )
            chunk_record0.layers[layer_idx0].stage4.ctx = record.ctx
            if routing is not None and ctx.fwd_comm_work is not None:
                ctx.fwd_comm_deferred_free.append(
                    chunk_record0.layers[layer_idx0].stage3.outs.moe_outs
                )  # freed after Stage 5 waits

            # Module 1 layer L-l-1 stage 4 backward
            record = chunk_record1.layers[-l - 1].stage4
            moe_outs_grad = stage4_b(ctx, module1_layers[-l - 1], record, (moe_outs_grad,))

            # merge the stage 5 and stage 1 forward into a single stage if
            # 1. module0 has fewer layers, or that both modules have the same number of layers, so no handle of the extra layer
            # 2. we aren't the first or the last layer
            if len(module0_layers) <= len(module1.layers) or (l - 1 > 0 and l < num_layers - 1):
                # Merge the stage 5 and stage 1 forward into a single stage.
                # Use asymmetric None pattern for detection:
                # - Store stage5.args at prev layer (no outs)
                # - Store stage1.outs at next layer (no args)
                prev_layer, next_layer = module0_layers[l - 1], module0_layers[l]
                stage5_args, stage1_outs, dispatch_tokens, residual, routing = stage5_and_stage1_f(
                    ctx,
                    prev_layer,
                    next_layer,
                    moe_outs,
                    routing,
                    residual,
                    rotary_posemb,
                    cu_seqlens,
                )
                # Store stage5.args at prev layer (no outs -> merged indicator)
                chunk_record0.layers[layer_idx0].stage5.args = stage5_args
                # stage5.outs stays None
                layer_idx0 += 1
                # Store stage1.outs at next layer (no args -> merged indicator)
                # stage1.args stays None
                chunk_record0.layers[layer_idx0].stage1.outs = stage1_outs
            else:
                # Module 0 layer l-1 stage 5 forward
                record, hidden_states = stage5_f(
                    ctx,
                    module0_layers[l - 1],
                    moe_outs,
                    routing,
                    residual,
                )
                chunk_record0.layers[layer_idx0].stage5.args = record.args
                chunk_record0.layers[layer_idx0].stage5.outs = record.outs
                layer_idx0 += 1
                # Module 0 layer l stage 1 forward
                record, dispatch_tokens, residual, routing = stage1_f(
                    ctx, module0_layers[l], hidden_states, rotary_posemb, cu_seqlens
                )
                chunk_record0.layers[layer_idx0].stage1.args = record.args
                chunk_record0.layers[layer_idx0].stage1.outs = record.outs

        # Module 1 layer L-l-1 stage 3 backward
        record = chunk_record1.layers[-l - 1].stage3
        gathered_tokens_grad = stage3_b(ctx, module1_layers[-l - 1], record, (moe_outs_grad,))

        # Module 0 layer l stage 2 forward
        record, gathered_tokens = stage2_f(
            ctx,
            module0_layers[l],
            dispatch_tokens,
            routing.dispatch_splits if routing is not None else None,
            ep_group,
        )
        chunk_record0.layers[layer_idx0].stage2.ctx = record.ctx
        if routing is not None and ctx.fwd_comm_work is not None:
            ctx.fwd_comm_deferred_free.append(dispatch_tokens)  # freed after Stage 3 waits

        # Module 1 layer L-l-1 stage 2 backward
        record = chunk_record1.layers[-l - 1].stage2
        dispatch_tokens_grad = stage2_b(
            ctx, module1_layers[-l - 1], record, (gathered_tokens_grad,)
        )

        # Module 1 layer L-l-1 stage 3 weight backward
        stage3_w(ctx, module1_layers[-l - 1])

        # Module 0 layer l stage 3 forward
        record, moe_outs = stage3_f(
            ctx,
            module0_layers[l],
            gathered_tokens,
            routing.expert_idxs if routing is not None else None,
            routing.expand_idx if routing is not None else None,
        )
        chunk_record0.layers[layer_idx0].stage3.args = record.args
        chunk_record0.layers[layer_idx0].stage3.outs = record.outs

    # Module 0 layer L-1 stage 4 forward
    record, moe_outs = stage4_f(
        ctx,
        module0_layers[num_layers - 1],
        moe_outs,
        routing.combine_splits if routing is not None else None,
        ep_group,
    )
    chunk_record0.layers[layer_idx0].stage4.ctx = record.ctx
    if routing is not None and ctx.fwd_comm_work is not None:
        ctx.fwd_comm_deferred_free.append(
            chunk_record0.layers[layer_idx0].stage3.outs.moe_outs
        )  # freed after Stage 5 waits

    # Module 1 layer 0 stage 1 backward
    record = chunk_record1.layers[-num_layers].stage1

    grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)
    hidden_states_grad = stage1_b(ctx, module1_layers[-num_layers], record, grad_tensors)

    # Clear tensor refs but keep pre-allocated records
    _clear_layer_records(chunk_record1.layers[-num_layers])

    # Module 0 layer L-1 stage 5 forward
    record, hidden_states = stage5_f(
        ctx,
        module0_layers[num_layers - 1],
        moe_outs,
        routing,
        residual,
    )
    chunk_record0.layers[layer_idx0].stage5.args = record.args
    chunk_record0.layers[layer_idx0].stage5.outs = record.outs
    layer_idx0 += 1

    if len(module0.layers) == len(module1.layers) + 1:
        # There is an extra layer in module0 for forward
        hidden_states = layer_forward(
            module0_layers[-1],
            hidden_states,
            rotary_posemb,
            chunk_record0.layers[layer_idx0],
            cu_seqlens,
        )
        layer_idx0 += 1
    elif len(module0.layers) + 1 == len(module1.layers):
        # There is an extra layer in module1 for backward
        hidden_states_grad = layer_backward(
            module1_layers[0],
            (hidden_states_grad,),
            None,
            chunk_record1.layers[-num_layers - 1],
        )
    else:
        assert len(module0.layers) == len(module1.layers)

    final_grads = (hidden_states_grad,)
    if module1.stage_index == 0:
        record = chunk_record1.prolog
        prolog_b(module1, record, (hidden_states_grad,))
        final_grads = (None,)
        # Clear tensor refs but keep pre-allocated record
        record.args = None
        record.outs = None
    if module0.stage_index == module0.stage_count - 1:
        hidden_states = epilog_f(module0, hidden_states, chunk_record0.epilog)

    # Run criterion if needed
    outputs0 = [hidden_states]
    if criterion0 is not None:
        nvtx.range_push("criterion0(*outputs0, *labels0)")
        loss0 = criterion0(*outputs0, *labels0)
        nvtx.range_pop()
    else:
        loss0 = None

    del ctx.fwd_comm_work, ctx.bwd_comm_work, ctx.fwd_event, ctx.bwd_event
    # chunk_record0 was modified in place, no need to return it
    return outputs0, loss0, final_grads
