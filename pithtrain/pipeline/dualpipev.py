"""
DualPipeV: Overlapped forward-backward pipeline parallelism.

The DualPipeV class in this module is derived from the DualPipeV
implementation in DeepSeek's DualPipe project
(https://github.com/deepseek-ai/DualPipe), which is licensed under the
MIT License. Copyright (c) 2025 DeepSeek. See pithtrain/pipeline/LICENSE
and the project-root NOTICE file for the full license text and details
of which portions are derived.

The 8-step scheduling algorithm in DualPipeV.step() and the P2P
communication orchestration methods are closely adapted from the original,
as are the _append_irecv / _append_isend helpers, which were merged
in from dualpipe/comm.py of the same project.
The overlapped_forward_backward() function, FSDP integration, FP8 weight
caching, and the 5-stage decomposition are original additions.

Stage Mapping:
    - Stage 1: Attention (LN + Attn + LN + Expert selection)
    - Stage 2: Dispatch (All-to-all dispatch for expert parallelism)
    - Stage 3: MLP (Expert/MLP computation)
    - Stage 4: Combine (All-to-all combine for expert parallelism)
    - Stage 5: Aggregate (Weighted expert output + residual connection)
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard

from pithtrain.contexts import distributed, training
from pithtrain.models.interface import ModelProtocol
from pithtrain.operators.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.pipeline.execution import (
    ChunkRecord,
    ExecutionCtx,
    WeightGradStore,
    clear_layer_records,
    create_chunk_record,
    epilog_f,
    layer_backward,
    layer_forward,
    model_backward,
    prolog_b,
    prolog_f,
    stage1_b,
    stage1_f,
    stage1_stage5_b,
    stage2_b,
    stage2_f,
    stage3_b,
    stage3_f,
    stage3_w,
    stage4_b,
    stage4_f,
    stage5_b,
    stage5_f,
    stage5_stage1_f,
)


def layer_partition(num_layers: int, stage_count: int, stage_index: int) -> range:
    """
    Return the layer ids that pipeline stage stage_index owns.

    Layers are spread across the stages as evenly as possible, with the two edges
    kept slightly lighter since they also carry embed_tokens and norm plus lm_head.
    Each stage owns a contiguous block, and this returns the block for stage_index.
    """
    base, remainder = divmod(num_layers, stage_count)
    layers = [base] * stage_count
    for _ in range(remainder):
        min_val = min(layers)
        best = next((i for i in range(1, stage_count - 1) if layers[i] == min_val), None)
        if best is None:
            best = layers.index(min_val)
        layers[best] += 1
    begin = sum(layers[:stage_index])
    end = begin + layers[stage_index]
    return range(begin, end)


@dataclass(slots=True, kw_only=True)
class Microbatch:
    """
    One micro-batch of work for DualPipeV.step.

    The caller partitions the global batch into these and hands the same list to every pipeline
    rank, so each derives its own receive-buffer shapes without a broadcast. Only the first rank
    reads the tensors: under the V-shape it holds both the model inputs and the final model
    outputs.

    Attributes:
        model_inputs: Inputs to the model, for instance the token ids, handed to its first
            stage positionally. The batch and sequence dimensions of the first one size the
            activation buffers every pipeline stage receives, so it must lead with those two.
        cu_seqlens: Document boundaries when this micro-batch packs several sequences, or None
            when each row holds a single sequence.
        objective_inputs: Whatever the objective needs for this micro-batch, passed through
            untouched. The engine never inspects it.
    """

    model_inputs: Tuple[torch.Tensor, ...]
    cu_seqlens: Optional[torch.Tensor]
    objective_inputs: Any


class DualPipeV(nn.Module):
    """
    V-shaped bidirectional pipeline parallelism scheduler.

    Derived from the DualPipeV class in DeepSeek's DualPipe project
    (https://github.com/deepseek-ai/DualPipe), which implements the algorithm
    described in the DeepSeek-V3 Technical Report (https://arxiv.org/abs/2412.19437).
    The original V-shape "cut-in-half" procedure was introduced by Sea AI Lab.

    This implementation extends the original with:
      - A 5-stage overlapped forward-backward loop (overlapped_forward_backward)
        that decomposes each transformer layer into Attention / Dispatch / MLP /
        Combine / Aggregate stages for fine-grained computation-communication overlap.
      - FSDP2 integration (hook suppression during the pipeline loop, manual
        post_backward invocation after the loop).
      - FP8 weight caching across micro-batches via FP8WeightCacheControl.
      - Pre-allocated ChunkRecord for zero-allocation pipeline execution.
    """

    def __init__(self, modules: Tuple[nn.Module, nn.Module]) -> None:
        super().__init__()

        device = torch.device(torch.cuda.current_device())
        assert next(modules[0].parameters()).device == device
        self.module = nn.ModuleList(modules)
        self.p2p_shapes: List[List[Tuple[int, ...]]] = []

        pp_rank, pp_size = distributed.pp_rank, distributed.pp_size
        self.prev_pp_rank = pp_rank - 1 if pp_rank > 0 else None
        self.next_pp_rank = pp_rank + 1 if pp_rank < pp_size - 1 else None
        self.is_first_pp_rank = pp_rank == 0
        self.is_last_pp_rank = pp_rank == pp_size - 1

        self.comm_stream = torch.cuda.Stream(device=device)

        self._num_chunks_allocated = 0
        self.chunk_records: Tuple[List[ChunkRecord], List[ChunkRecord]] = ([], [])
        self.cu_seqlens_chunks: Optional[List[torch.Tensor]] = None

        self.forward_only = False
        self._reset_states()

    def _ensure_chunk_records_allocated(self, num_chunks: int) -> None:
        """
        Pre-allocate ChunkRecord structures for reuse across iterations.
        """
        if self._num_chunks_allocated == num_chunks:
            return
        self.chunk_records = tuple(
            [
                create_chunk_record(
                    len(m.layers), m.stage_index == 0, m.stage_index == m.stage_count - 1
                )
                for _ in range(num_chunks)
            ]
            for m in self.module
        )
        self._num_chunks_allocated = num_chunks

    def _reset_states(self) -> None:
        WeightGradStore.clear()

        self.input_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = (
            [],
            [],
        )
        self.output_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        # Note: chunk_records is pre-allocated and reused, not reset here
        self.input_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.output_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = (
            [],
            [],
        )
        self.objective_inputs: Optional[List[Any]] = None
        self.loss_chunks: List[Optional[torch.Tensor]] = []
        self.objective_output_chunks: List[Any] = []
        self.objective: Optional[Callable] = None

        self.current_f_chunk_id: List[int] = [0, 0]
        self.current_b_chunk_id: List[int] = [0, 0]
        self.current_send_f_chunk_id: List[int] = [0, 0]
        self.current_send_b_chunk_id: List[int] = [0, 0]
        self.current_recv_f_chunk_id: List[int] = [0, 0]
        self.current_recv_b_chunk_id: List[int] = [0, 0]
        self.comm_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []

    def setup_step_metadata(self, microbatches: List[Microbatch]) -> int:
        """
        Record the per-step shapes.

        Every pipeline rank is given the same micro-batches, so each derives the activation shape
        of every one locally and no metadata crosses the pipeline. Micro-batches may differ in
        shape from one another. The hidden dimension is read from the local model.

        Returns the number of micro-batches in this step.
        """
        device = distributed.device
        hidden = self.module[0].hidden_size

        # One shape per micro-batch, so a ragged step sizes each receive buffer correctly.
        self.p2p_shapes = []
        for mb in microbatches:
            first_input, *_ = mb.model_inputs
            batch, sequence, *_ = first_input.shape
            self.p2p_shapes.append([(batch, sequence, hidden)])

        # A micro-batch without boundaries is left dense; attention dispatches on None.
        self.cu_seqlens_chunks = None
        if any(mb.cu_seqlens is not None for mb in microbatches):
            chunks = []
            for mb in microbatches:
                cu = mb.cu_seqlens
                chunks.append(None if cu is None else cu.to(device=device, dtype=torch.int32))
            self.cu_seqlens_chunks = chunks

        return len(microbatches)

    def _forward_compute_chunk(self, phase: int) -> None:
        chunk_id = self.current_f_chunk_id[phase]
        self.current_f_chunk_id[phase] += 1
        inputs = self.input_chunks[phase][chunk_id]
        if self.forward_only:
            self.input_chunks[phase][chunk_id] = None

        is_last_stage = self.is_first_pp_rank and phase == 1

        nvtx.range_push(f"forward chunk {chunk_id} (phase{phase})")
        # Set pre-allocated chunk_record on module to avoid FSDP kwarg handling issues
        chunk_record = self.chunk_records[phase][chunk_id]
        self.module[phase].chunk_record = chunk_record
        cu_seqlens = (
            self.cu_seqlens_chunks[chunk_id] if self.cu_seqlens_chunks is not None else None
        )
        outputs = self.module[phase](*inputs, cu_seqlens=cu_seqlens)
        self.module[phase].chunk_record = None
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
        if is_last_stage:
            loss, objective_output = self.objective(tuple(outputs), self.objective_inputs[chunk_id])
            self.loss_chunks.append(loss)
            self.objective_output_chunks.append(objective_output)
        nvtx.range_pop()

        if self.is_last_pp_rank and phase == 0:
            self.input_chunks[1].append([output.detach().requires_grad_() for output in outputs])
        if not is_last_stage:
            self.output_chunks[phase].append(outputs)
        # No need to append - chunk_record is pre-allocated and was modified in place

    def _backward_compute_chunk(self, phase: int, enable_zb: bool = False) -> None:
        if self.forward_only:
            return

        chunk_id = self.current_b_chunk_id[phase]
        self.current_b_chunk_id[phase] += 1

        is_last_stage = self.is_first_pp_rank and phase == 1

        nvtx.range_push(f"backward chunk {chunk_id} (phase{phase})")
        WeightGradStore.enabled = enable_zb
        if is_last_stage:
            loss = self.loss_chunks[chunk_id]
            assert loss is not None, (
                "the objective must return a loss when gradients are enabled; "
                "a None loss is only valid under torch.no_grad()"
            )
            input_grads = model_backward(
                self.module[phase],
                None,
                loss,
                self.chunk_records[phase][chunk_id],
            )
            loss.detach_()
        else:
            outputs = self.output_chunks[phase][chunk_id]
            self.output_chunks[phase][chunk_id] = None
            output_grads = self.output_grad_chunks[phase][chunk_id]
            self.output_grad_chunks[phase][chunk_id] = None
            non_empty = [(t, g) for t, g in zip(outputs, output_grads) if g is not None]
            outputs, output_grads = list(zip(*non_empty))
            input_grads = model_backward(
                self.module[phase],
                output_grads,
                None,
                self.chunk_records[phase][chunk_id],
            )
        # Note: chunk_record is pre-allocated and reused; backward clears tensor refs inside
        WeightGradStore.enabled = False
        if enable_zb:
            WeightGradStore.flush()
        nvtx.range_pop()

        self.input_chunks[phase][chunk_id] = None
        if self.is_last_pp_rank and phase == 1:
            self.output_grad_chunks[0].append(input_grads)
        else:
            self.input_grad_chunks[phase].append(input_grads)

    def _forward_backward_compute_chunk(self, phase0: int, phase1: int) -> None:
        if self.forward_only:
            self._forward_compute_chunk(phase0)
            return

        # pre-forward
        chunk_id0 = self.current_f_chunk_id[phase0]
        self.current_f_chunk_id[phase0] += 1
        module0 = self.module[phase0]
        inputs0 = self.input_chunks[phase0][chunk_id0]
        cu_seqlens0 = (
            self.cu_seqlens_chunks[chunk_id0] if self.cu_seqlens_chunks is not None else None
        )
        is_last_stage0 = self.is_first_pp_rank and phase0 == 1

        if is_last_stage0:
            objective_inputs0 = self.objective_inputs[chunk_id0]
            objective0 = self.objective
        else:
            objective_inputs0 = None
            objective0 = None

        # pre-backward
        chunk_id1 = self.current_b_chunk_id[phase1]
        self.current_b_chunk_id[phase1] += 1
        module1 = self.module[phase1]
        is_last_stage1 = self.is_first_pp_rank and phase1 == 1

        if is_last_stage1:
            loss1 = self.loss_chunks[chunk_id1]
            outputs1 = []
            output_grads1 = []
        else:
            loss1 = None
            outputs1 = self.output_chunks[phase1][chunk_id1]
            self.output_chunks[phase1][chunk_id1] = None
            output_grads1 = self.output_grad_chunks[phase1][chunk_id1]
            self.output_grad_chunks[phase1][chunk_id1] = None
            non_empty = [(t, g) for t, g in zip(outputs1, output_grads1) if g is not None]
            outputs1, output_grads1 = list(zip(*non_empty))

        # forward & backward (chunk_record0 is modified in place)
        nvtx.range_push(
            f"forward chunk {chunk_id0} (phase{phase0}) backward chunk {chunk_id1} (phase{phase1})"
        )
        outputs0, loss0, objective_output0, input_grads1 = overlapped_forward_backward(
            module0,
            inputs0,
            objective0,
            objective_inputs0,
            self.chunk_records[phase0][chunk_id0],
            cu_seqlens0,
            module1,
            loss1,
            outputs1,
            output_grads1,
            self.chunk_records[phase1][chunk_id1],
            self.comm_stream,
        )
        nvtx.range_pop()

        # post-forward
        if self.is_last_pp_rank and phase0 == 0:
            self.input_chunks[1].append([output.detach().requires_grad_() for output in outputs0])
        if not is_last_stage0:
            self.output_chunks[phase0].append(outputs0)
        if is_last_stage0:
            self.loss_chunks.append(loss0)
            self.objective_output_chunks.append(objective_output0)

        # post-backward
        self.input_chunks[phase1][chunk_id1] = None
        if self.is_last_pp_rank and phase1 == 1:
            self.output_grad_chunks[0].append(input_grads1)
        else:
            self.input_grad_chunks[phase1].append(input_grads1)

    def _forward_chunk(self, phase: int, recv: bool = True, send: bool = True) -> None:
        if recv:
            self._recv_forward(phase)
        self._commit_and_wait_comm()

        self._forward_compute_chunk(phase)

        if send:
            self._send_forward(phase)

    def _backward_chunk(
        self, phase: int, enable_zb: bool = False, recv: bool = True, send: bool = True
    ) -> None:
        if recv:
            self._recv_backward(phase)
        self._commit_and_wait_comm()

        self._backward_compute_chunk(phase, enable_zb)

        if send:
            self._send_backward(phase)

    def _forward_backward_chunk(self, phase0: int, phase1: int, recv0: bool = True) -> None:
        if recv0:
            self._recv_forward(phase0)
        self._recv_backward(phase1)
        self._commit_and_wait_comm()

        self._forward_backward_compute_chunk(phase0, phase1)

        self._send_forward(phase0)
        self._send_backward(phase1)

    def _weight_chunk(self) -> None:
        if self.forward_only:
            return

        self._commit_and_wait_comm()

        # Assume FIFO
        nvtx.range_push("weight chunk")
        WeightGradStore.pop()
        nvtx.range_pop()

    def _free_tensors(self) -> None:
        for tensor in self.to_free:
            assert tensor._base is None, (
                f"pipeline stage should not return view tensors {dist.get_rank(), tensor.shape}"
            )
            tensor.data = torch.Tensor()
        self.to_free = []

    def _append_irecv(self, src: int, chunk_id: int) -> List[torch.Tensor]:
        """
        Post a receive for one activation, sized by the shape agreed for that micro-batch.
        """
        tensors = [
            torch.empty(
                shape,
                dtype=training.PARAM_DTYPE,
                device=distributed.device,
                requires_grad=True,
            )
            for shape in self.p2p_shapes[chunk_id]
        ]
        src = dist.distributed_c10d.get_global_rank(distributed.pp_group, src)
        for tensor in tensors:
            self.comm_ops.append(dist.P2POp(dist.irecv, tensor, src))
        return tensors

    def _append_isend(self, tensors: List[torch.Tensor], dst: int) -> None:
        """
        Post a send for one activation.
        """
        dst = dist.distributed_c10d.get_global_rank(distributed.pp_group, dst)
        for tensor in tensors:
            if tensor is not None:
                self.comm_ops.append(dist.P2POp(dist.isend, tensor, dst))

    def _recv_forward(self, phase: int) -> None:
        if (self.is_first_pp_rank and phase == 0) or (self.is_last_pp_rank and phase == 1):
            return

        chunk_id = self.current_recv_f_chunk_id[phase]
        self.current_recv_f_chunk_id[phase] += 1
        tensors = self._append_irecv(
            self.prev_pp_rank if phase == 0 else self.next_pp_rank, chunk_id
        )
        self.input_chunks[phase].append(tensors)

    def _send_forward(self, phase: int) -> None:
        if (self.is_first_pp_rank and phase == 1) or (self.is_last_pp_rank and phase == 0):
            return

        chunk_id = self.current_send_f_chunk_id[phase]
        self.current_send_f_chunk_id[phase] += 1
        tensors = self.output_chunks[phase][chunk_id]

        self._append_isend(tensors, self.next_pp_rank if phase == 0 else self.prev_pp_rank)
        self.to_free.extend(tensors)

    def _recv_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_first_pp_rank and phase == 1) or (self.is_last_pp_rank and phase == 0):
            return

        chunk_id = self.current_recv_b_chunk_id[phase]
        self.current_recv_b_chunk_id[phase] += 1
        tensors = self._append_irecv(
            self.next_pp_rank if phase == 0 else self.prev_pp_rank, chunk_id
        )
        self.output_grad_chunks[phase].append(tensors)

    def _send_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_first_pp_rank and phase == 0) or (self.is_last_pp_rank and phase == 1):
            return

        chunk_id = self.current_send_b_chunk_id[phase]
        self.current_send_b_chunk_id[phase] += 1
        tensors = self.input_grad_chunks[phase][chunk_id]
        self.input_grad_chunks[phase][chunk_id] = None

        self._append_isend(tensors, self.prev_pp_rank if phase == 0 else self.next_pp_rank)

    def _commit_and_wait_comm(self) -> None:
        if not self.comm_ops:
            return
        nvtx.range_push("pipeline send/recv")
        reqs = dist.batch_isend_irecv(self.comm_ops)
        for req in reqs:
            req.wait()
        self.comm_ops = []
        self._free_tensors()
        nvtx.range_pop()

    def step(
        self,
        microbatches: List[Microbatch],
        objective: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Execute a training or inference step.

        The objective runs on the stage holding the final model outputs, before that
        micro-batch's backward; the loss it returns is then backpropagated and the outputs
        released. Nothing else from a micro-batch survives the step.

        Arguments:
            microbatches: The micro-batches to run, in order. Required on every pipeline rank,
                identical on each. See Microbatch.
            objective: Invoked once per micro-batch as objective(model_outputs,
                objective_inputs), returning a (loss, objective_output) pair. The loss is
                scheduled for backward, or None for a collection-only pass under
                torch.no_grad(). Required only on the first pipeline rank.

        Returns:
            One objective_output per micro-batch, in order; empty on ranks that do not hold the
            final model outputs, which under the V-shape is every rank but the first. The loss is
            detached in place after its backward, so return loss.detach() if the value is needed
            once the step is done.
        """
        self.forward_only = not torch.is_grad_enabled()

        # Disable reshard and gradient sync after backward for FSDP
        for module in self.module:
            if isinstance(module, FSDPModule):
                module.set_is_last_backward(False)
                module.set_reshard_after_backward(False)
                module.set_requires_gradient_sync(False)
                # Suppress FSDP's root post-backward callback during the pipeline
                # loop. Each run_backward() would otherwise queue this callback,
                # which iterates ALL FSDP states (~150-250 us CPU overhead per
                # backward stage). The flag resets when run_post_backward() calls
                # _root_post_backward_final_callback() directly after the loop.
                if not self.forward_only:
                    fully_shard.state(module)._state_ctx.post_backward_final_callback_queued = True

        pp_rank, pp_size = distributed.pp_rank, distributed.pp_size

        if self.is_first_pp_rank:
            assert objective is not None, (
                "the first pipeline rank holds the final model outputs and needs an objective"
            )

        self._reset_states()
        FP8WeightCacheControl.step()
        num_chunks = self.setup_step_metadata(microbatches)
        assert num_chunks >= pp_size * 2, (
            f"the V-shape gives each rank two model chunks, so a step needs at least "
            f"{pp_size * 2} micro-batches, got {num_chunks}"
        )
        self._ensure_chunk_records_allocated(num_chunks)

        if self.is_first_pp_rank:
            self.input_chunks = ([tuple(mb.model_inputs) for mb in microbatches], [])
            self.objective_inputs = [mb.objective_inputs for mb in microbatches]
            self.objective = objective

        # Step 1: nF0
        step_1 = (pp_size - pp_rank - 1) * 2
        for i in range(step_1):
            self._forward_chunk(0)

        # Step 2: nF0F1
        step_2 = pp_rank + 1
        self._recv_forward(0)
        for i in range(step_2):
            self._forward_chunk(0, recv=False, send=False)
            self._recv_forward(0)
            self._forward_chunk(1, send=(not self.is_last_pp_rank) or (i < step_2 - 1))
            self._send_forward(0)

        # Step 3: nB1W1F1 (Use zero bubble)
        step_3 = pp_size - pp_rank - 1
        for i in range(step_3):
            self._backward_chunk(1, enable_zb=True)
            self._recv_forward(1)
            self._weight_chunk()
            self._forward_chunk(1, recv=False)

        # Step 4 (Main step): nF0B1F1B0
        step_4 = num_chunks - pp_size * 2 + pp_rank + 1
        for i in range(step_4):
            if i == 0:
                if self.is_last_pp_rank:
                    # NOTE: We don't overlap these two chunks to further reduce bubble size.
                    self._forward_chunk(0, recv=False, send=False)
                    self._send_forward(1)
                    self._backward_chunk(1, send=False)
                    self._send_forward(0)
                    self._send_backward(1)
                else:
                    self._forward_backward_chunk(0, 1, recv0=False)
            else:
                self._forward_backward_chunk(0, 1)
            self._forward_backward_chunk(1, 0)

        # Step 5: nB1F1B0
        step_5 = pp_size - pp_rank - 1
        for i in range(step_5):
            self._backward_chunk(1)
            self._forward_backward_chunk(1, 0)

        # Step 6: nB1B0 (The second half of the chunks use zero bubble)
        step_6 = pp_rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and pp_rank % 2 == 1:
                enable_zb = True
            self._backward_chunk(1, enable_zb=enable_zb)
            if i == step_6 // 2 and pp_rank % 2 == 0:
                enable_zb = True
            self._backward_chunk(0, enable_zb=enable_zb)

        # Step 7: nWB0 (Use zero bubble)
        step_7 = pp_size - pp_rank - 1
        for i in range(step_7):
            self._weight_chunk()
            self._backward_chunk(0, enable_zb=True)

        # Step 8: nW
        step_8 = pp_rank + 1
        for i in range(step_8):
            self._weight_chunk()
        assert WeightGradStore.funcs_queue.empty()

        self._commit_and_wait_comm()

        objective_outputs = self.objective_output_chunks

        self._reset_states()

        # Release FP8 weight caches so the memory is available for optimizer.step().
        FP8WeightCacheControl.clear(*self.module)

        def run_post_backward(fsdp_module: FSDPModule) -> None:
            fsdp_module.set_is_last_backward(True)
            fsdp_module.set_reshard_after_backward(True)
            fsdp_module.set_requires_gradient_sync(True)
            fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
            for state in fsdp_state._state_ctx.all_states:
                if state._fsdp_param_group:
                    # Fold wgrad-delay's trailing param_dtype write into the
                    # reduce_dtype accumulator so foreach_reduce sees uniform
                    # dtype. accumulate first (to_accumulated would otherwise
                    # clobber prior fp32 contributions).
                    for fsdp_param in state._fsdp_param_group.fsdp_params:
                        if hasattr(fsdp_param, "_unsharded_param"):
                            fsdp_param.accumulate_unsharded_grad_if_needed()
                            fsdp_param.to_accumulated_grad_if_needed()
                    state._fsdp_param_group.post_backward()

            # Pipeline backward bypasses .backward(), so FSDP's autograd hooks never fire. Call its
            # post-backward callback manually to sync grad reduction ops back to the default stream.
            fsdp_state._root_post_backward_final_callback()

        for module in self.module:
            if isinstance(module, FSDPModule):
                run_post_backward(module)

        return objective_outputs


# ------------------------------------------------------------
# OVERLAPPED FORWARD-BACKWARD
# ------------------------------------------------------------


def overlapped_forward_backward(
    module0: ModelProtocol,
    inputs0: List[torch.Tensor],
    objective0: Optional[Callable],
    objective_inputs0: Optional[Tuple[torch.Tensor, ...]],
    chunk_record0: ChunkRecord,
    cu_seqlens: Optional[torch.Tensor],
    module1: ModelProtocol,
    loss1: Optional[torch.Tensor],
    outputs1: Optional[List[torch.Tensor]],
    output_grads1: Optional[List[torch.Tensor]],
    chunk_record1: ChunkRecord,
    comm_stream: Optional[torch.cuda.Stream],
):
    """
    Interleave the forward pass of one model replica with the backward pass of another.

    Each transformer layer is decomposed into 5 stages (Attention / Dispatch / MLP / Combine /
    Aggregate) for fine-grained computation-communication overlap.
    """
    assert abs(len(module0.layers) - len(module1.layers)) <= 1
    assert len(chunk_record1.layers) == len(module1.layers)
    num_layers = min(len(module0.layers), len(module1.layers))
    module0_layers = list(module0.layers.values())
    module1_layers = list(module1.layers.values())

    (hidden_states,) = inputs0
    layer_idx0 = 0  # Index into chunk_record0.layers
    dispatch_tokens_grad = moe_outs = None  # carried across loop iterations

    ctx = ExecutionCtx(comm_stream=comm_stream)

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
            # Merged layers set outs without args at the next layer, and args without outs
            # at the previous one; unmerged layers set both.
            stage1_record = chunk_record1.layers[-l].stage1
            use_merged = stage1_record.outs is not None and stage1_record.args is None

            if use_merged:
                next_layer, prev_layer = module1_layers[-l], module1_layers[-l - 1]
                stage1_outs = stage1_record.outs
                stage5_args = chunk_record1.layers[-l - 1].stage5.args
                grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)
                moe_outs_grad, topk_weight_grad, residual_grad = stage1_stage5_b(
                    ctx, next_layer, prev_layer, stage1_outs, stage5_args, grad_tensors
                )
                clear_layer_records(chunk_record1.layers[-l])
            else:
                # Module 1 layer L-l stage 1 backward
                record = chunk_record1.layers[-l].stage1
                grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)
                hidden_states_grad = stage1_b(ctx, module1_layers[-l], record, grad_tensors)
                # Module 1 layer L-l-1 stage 5 backward
                record = chunk_record1.layers[-l - 1].stage5
                moe_outs_grad, topk_weight_grad, residual_grad = stage5_b(
                    ctx, module1_layers[-l - 1], record, (hidden_states_grad,)
                )
                clear_layer_records(chunk_record1.layers[-l])

            # Module 0 layer l-1 stage 4 forward
            record, moe_outs = stage4_f(
                ctx,
                module0_layers[l - 1],
                moe_outs,
                routing.combine_splits if routing is not None else None,
            )
            chunk_record0.layers[layer_idx0].stage4.splits = record.splits
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
                # stage5.args goes on the previous layer with outs left None, stage1.outs on
                # the next layer with args left None -- that is the merged marker.
                prev_layer, next_layer = module0_layers[l - 1], module0_layers[l]
                stage5_args, stage1_outs, dispatch_tokens, residual, routing = stage5_stage1_f(
                    ctx,
                    prev_layer,
                    next_layer,
                    moe_outs,
                    routing,
                    residual,
                    rotary_posemb,
                    cu_seqlens,
                )
                chunk_record0.layers[layer_idx0].stage5.args = stage5_args
                layer_idx0 += 1
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
        )
        chunk_record0.layers[layer_idx0].stage2.splits = record.splits
        if routing is not None and ctx.fwd_comm_work is not None:
            ctx.fwd_comm_deferred_free.append(dispatch_tokens)  # freed after Stage 3 waits

        # Module 1 layer L-l-1 stage 2 backward
        record = chunk_record1.layers[-l - 1].stage2
        dispatch_tokens_grad = stage2_b(
            ctx, module1_layers[-l - 1], record, (gathered_tokens_grad,)
        )

        # Module 1 layer L-l-1 stage 3 weight backward
        stage3_w(module1_layers[-l - 1])

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
    )
    chunk_record0.layers[layer_idx0].stage4.splits = record.splits
    if routing is not None and ctx.fwd_comm_work is not None:
        ctx.fwd_comm_deferred_free.append(
            chunk_record0.layers[layer_idx0].stage3.outs.moe_outs
        )  # freed after Stage 5 waits

    # Module 1 layer 0 stage 1 backward
    record = chunk_record1.layers[-num_layers].stage1

    grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)
    hidden_states_grad = stage1_b(ctx, module1_layers[-num_layers], record, grad_tensors)

    clear_layer_records(chunk_record1.layers[-num_layers])

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
            chunk_record1.layers[-num_layers - 1],
        )
    else:
        assert len(module0.layers) == len(module1.layers)

    final_grads = (hidden_states_grad,)
    if module1.stage_index == 0:
        record = chunk_record1.prolog
        prolog_b(module1, record, (hidden_states_grad,))
        final_grads = (None,)
        record.outs = None  # keep the pre-allocated record, drop the tensor ref
    if module0.stage_index == module0.stage_count - 1:
        hidden_states = epilog_f(module0, hidden_states, chunk_record0.epilog)

    outputs0 = [hidden_states]
    if objective0 is not None:
        nvtx.range_push("objective0(model_outputs0, objective_inputs0)")
        loss0, objective_output0 = objective0(tuple(outputs0), objective_inputs0)
        nvtx.range_pop()
    else:
        loss0, objective_output0 = None, None

    # chunk_record0 was modified in place, no need to return it
    return outputs0, loss0, objective_output0, final_grads
