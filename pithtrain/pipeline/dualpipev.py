"""
DualPipeV: overlapped forward-backward pipeline parallelism.

Each rank holds two model chunks in a V-shape, and each decoder layer runs as five stages:

- Stage 1: attention (norm, attention, norm, expert routing).
- Stage 2: dispatch all-to-all.
- Stage 3: experts or MLP.
- Stage 4: combine all-to-all.
- Stage 5: weighted expert output plus the residual.

The DualPipeV class is derived from DeepSeek's DualPipe project
(https://github.com/deepseek-ai/DualPipe), licensed under the MIT License. Copyright (c) 2025
DeepSeek. The 8-step schedule in DualPipeV.step, the point-to-point methods and the
append_irecv / append_isend helpers (from dualpipe/comm.py) are adapted from it;
overlapped_forward_backward, the FSDP integration, FP8 weight caching and the five-stage split
are not. See pithtrain/pipeline/LICENSE and the NOTICE file at the project root.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard

from pithtrain.contexts import distributed, training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.models.interface import ModelProtocol
from pithtrain.pipeline.execution import (
    ChunkRecord,
    ExecutionCtx,
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
from pithtrain.pipeline.guard_skip import GuardSkip

__all__ = [
    "DualPipeV",
    "Microbatch",
    "layer_partition",
    "layer_sizes",
    "min_chunks",
    "overlapped_forward_backward",
]


def layer_sizes(num_layers: int, stage_count: int) -> list[int]:
    """
    Layers are spread as evenly as possible, with any remainder kept off the two edge stages, which
    also hold embed_tokens and norm plus lm_head.
    """
    base, remainder = divmod(num_layers, stage_count)
    layers = [base] * stage_count
    for _ in range(remainder):
        min_val = min(layers)
        best = next((i for i in range(1, stage_count - 1) if layers[i] == min_val), None)
        if best is None:
            best = layers.index(min_val)
        layers[best] += 1
    return layers


def layer_partition(
    num_layers: int, stage_count: int, stage_index: int, sizes: Sequence[int] | None = None
) -> range:
    """
    The two stages of one rank may differ by at most one layer, as the overlapped schedule requires.
    """
    if sizes is None:
        sizes = layer_sizes(num_layers, stage_count)
    assert len(sizes) == stage_count and sum(sizes) == num_layers, sizes
    assert all(abs(sizes[i] - sizes[stage_count - 1 - i]) <= 1 for i in range(stage_count)), sizes
    begin = sum(sizes[:stage_index])
    return range(begin, begin + sizes[stage_index])


def min_chunks(pp_size: int) -> int:
    """
    Return the smallest number of micro-batches a step can schedule at this pp_size. With fewer,
    low pipeline ranks skip work their neighbours wait on and the step hangs; a single pipeline
    rank posts no point-to-point, so it needs one.
    """
    return 1 if pp_size == 1 else pp_size * 2


@dataclass(slots=True, kw_only=True)
class Microbatch:
    """
    One micro-batch of work for DualPipeV.step. Every pipeline rank gets the same list, so each
    sizes its receive buffers and reads cu_seqlens without a broadcast; only the first rank uses the
    values of model_inputs and objective_inputs.
    """

    model_inputs: Tuple[torch.Tensor, ...]
    """
    Model inputs, such as token ids, passed positionally to the first stage. The first one must
    lead with the batch and sequence dimensions, which size the activations between stages.
    """

    cu_seqlens: Optional[torch.Tensor]
    """
    Document boundaries of a packed sample, or None when each row is one sequence. Under context
    parallelism they cover the whole sample while the tensors hold this rank's share.
    """

    objective_inputs: Any


class DualPipeV(nn.Module):
    """
    V-shaped bidirectional pipeline scheduler over the two model chunks of this rank, from the
    DeepSeek-V3 Technical Report (https://arxiv.org/abs/2412.19437).

    Data parallelism is ZeRO-1: during a step every rank holds the full weights and accumulates
    full-size gradients, and each FSDP group reduce-scatters its gradients once at the end.
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

        self.num_chunks_allocated = 0
        self.chunk_records: Tuple[List[ChunkRecord], List[ChunkRecord]] = ([], [])
        self.cu_seqlens_chunks: Optional[List[torch.Tensor]] = None

        self.forward_only = False
        self.guard_skip = GuardSkip()
        self.reset_states()

    def ensure_chunk_records_allocated(self, num_chunks: int) -> None:
        if self.num_chunks_allocated == num_chunks:
            return
        self.chunk_records = tuple(
            [
                create_chunk_record(
                    len(module.layers),
                    module.stage_index == 0,
                    module.stage_index == module.stage_count - 1,
                )
                for _ in range(num_chunks)
            ]
            for module in self.module
        )
        self.num_chunks_allocated = num_chunks

    def reset_states(self) -> None:
        WeightGradStore.clear()
        training.current_microbatch = None

        self.input_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])  # fmt: skip
        self.output_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
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
        Micro-batches may differ in shape.
        """
        device = distributed.device
        hidden = self.module[0].hidden_size

        self.p2p_shapes = []
        for microbatch in microbatches:
            first_input, *_ = microbatch.model_inputs
            batch, sequence, *_ = first_input.shape
            self.p2p_shapes.append([(batch, sequence, hidden)])

        # A micro-batch without boundaries is left dense; attention dispatches on None.
        self.cu_seqlens_chunks = None
        if any(microbatch.cu_seqlens is not None for microbatch in microbatches):
            chunks = []
            for microbatch in microbatches:
                cu_seqlens = microbatch.cu_seqlens
                chunks.append(None if cu_seqlens is None else cu_seqlens.to(device=device, dtype=torch.int32))  # fmt: skip
            self.cu_seqlens_chunks = chunks

        return len(microbatches)

    def forward_compute_chunk(self, phase: int) -> None:
        chunk_id = self.current_f_chunk_id[phase]
        self.current_f_chunk_id[phase] += 1
        inputs = self.input_chunks[phase][chunk_id]
        if self.forward_only:
            self.input_chunks[phase][chunk_id] = None
            if self.is_last_pp_rank and phase == 1:
                # The local phase handoff has no send to release its output reference.
                self.output_chunks[0][chunk_id] = None

        is_last_stage = self.is_first_pp_rank and phase == 1

        nvtx.range_push(f"forward chunk {chunk_id} (phase{phase})")
        # Passed as a module attribute rather than a forward kwarg, which FSDP would process.
        chunk_record = self.chunk_records[phase][chunk_id]
        self.module[phase].chunk_record = chunk_record
        cu_seqlens = (
            self.cu_seqlens_chunks[chunk_id] if self.cu_seqlens_chunks is not None else None
        )
        training.current_microbatch = chunk_id
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

    def backward_compute_chunk(self, phase: int, enable_zb: bool = False) -> None:
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
            non_empty = [(out, grad) for out, grad in zip(outputs, output_grads) if grad is not None]  # fmt: skip
            outputs, output_grads = list(zip(*non_empty))
            input_grads = model_backward(
                self.module[phase],
                output_grads,
                None,
                self.chunk_records[phase][chunk_id],
            )
        WeightGradStore.enabled = False
        if enable_zb:
            WeightGradStore.flush()
        nvtx.range_pop()

        self.input_chunks[phase][chunk_id] = None
        if self.is_last_pp_rank and phase == 1:
            self.output_grad_chunks[0].append(input_grads)
        else:
            self.input_grad_chunks[phase].append(input_grads)

    def forward_backward_compute_chunk(self, phase0: int, phase1: int) -> None:
        if self.forward_only:
            self.forward_compute_chunk(phase0)
            return

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
            non_empty = [(out, grad) for out, grad in zip(outputs1, output_grads1) if grad is not None]  # fmt: skip
            outputs1, output_grads1 = list(zip(*non_empty))

        nvtx.range_push(
            f"forward chunk {chunk_id0} (phase{phase0}) backward chunk {chunk_id1} (phase{phase1})"
        )
        # Publish the forward's micro-batch: the backward half replays the saved autograd graph
        # without re-entering model code, so nothing there reads it.
        training.current_microbatch = chunk_id0
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

        if self.is_last_pp_rank and phase0 == 0:
            self.input_chunks[1].append([output.detach().requires_grad_() for output in outputs0])
        if not is_last_stage0:
            self.output_chunks[phase0].append(outputs0)
        if is_last_stage0:
            self.loss_chunks.append(loss0)
            self.objective_output_chunks.append(objective_output0)

        self.input_chunks[phase1][chunk_id1] = None
        if self.is_last_pp_rank and phase1 == 1:
            self.output_grad_chunks[0].append(input_grads1)
        else:
            self.input_grad_chunks[phase1].append(input_grads1)

    def forward_chunk(self, phase: int, recv: bool = True, send: bool = True) -> None:
        if recv:
            self.recv_forward(phase)
        self.commit_and_wait_comm()

        self.forward_compute_chunk(phase)

        if send:
            self.send_forward(phase)

    def backward_chunk(
        self, phase: int, enable_zb: bool = False, recv: bool = True, send: bool = True
    ) -> None:
        if recv:
            self.recv_backward(phase)
        self.commit_and_wait_comm()

        self.backward_compute_chunk(phase, enable_zb)

        if send:
            self.send_backward(phase)

    def forward_backward_chunk(self, phase0: int, phase1: int, recv0: bool = True) -> None:
        if recv0:
            self.recv_forward(phase0)
        self.recv_backward(phase1)
        self.commit_and_wait_comm()

        self.forward_backward_compute_chunk(phase0, phase1)

        self.send_forward(phase0)
        self.send_backward(phase1)

    def weight_chunk(self) -> None:
        if self.forward_only:
            return

        self.commit_and_wait_comm()

        nvtx.range_push("weight chunk")
        WeightGradStore.pop()
        nvtx.range_pop()

    def free_sent_tensors(self) -> None:
        for tensor in self.to_free:
            assert tensor._base is None, (
                f"pipeline stage should not return view tensors {dist.get_rank(), tensor.shape}"
            )
            tensor.data = torch.Tensor()
        self.to_free = []

    def append_irecv(self, src: int, chunk_id: int) -> List[torch.Tensor]:
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

    def append_isend(self, tensors: List[torch.Tensor], dst: int) -> None:
        dst = dist.distributed_c10d.get_global_rank(distributed.pp_group, dst)
        for tensor in tensors:
            if tensor is not None:
                self.comm_ops.append(dist.P2POp(dist.isend, tensor, dst))

    def recv_forward(self, phase: int) -> None:
        if (self.is_first_pp_rank and phase == 0) or (self.is_last_pp_rank and phase == 1):
            return

        chunk_id = self.current_recv_f_chunk_id[phase]
        self.current_recv_f_chunk_id[phase] += 1
        tensors = self.append_irecv(
            self.prev_pp_rank if phase == 0 else self.next_pp_rank, chunk_id
        )
        self.input_chunks[phase].append(tensors)

    def send_forward(self, phase: int) -> None:
        if (self.is_first_pp_rank and phase == 1) or (self.is_last_pp_rank and phase == 0):
            return

        chunk_id = self.current_send_f_chunk_id[phase]
        self.current_send_f_chunk_id[phase] += 1
        tensors = self.output_chunks[phase][chunk_id]

        self.append_isend(tensors, self.next_pp_rank if phase == 0 else self.prev_pp_rank)
        self.to_free.extend(tensors)

    def recv_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_first_pp_rank and phase == 1) or (self.is_last_pp_rank and phase == 0):
            return

        chunk_id = self.current_recv_b_chunk_id[phase]
        self.current_recv_b_chunk_id[phase] += 1
        tensors = self.append_irecv(
            self.next_pp_rank if phase == 0 else self.prev_pp_rank, chunk_id
        )
        self.output_grad_chunks[phase].append(tensors)

    def send_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_first_pp_rank and phase == 0) or (self.is_last_pp_rank and phase == 1):
            return

        chunk_id = self.current_send_b_chunk_id[phase]
        self.current_send_b_chunk_id[phase] += 1
        tensors = self.input_grad_chunks[phase][chunk_id]
        self.input_grad_chunks[phase][chunk_id] = None

        self.append_isend(tensors, self.prev_pp_rank if phase == 0 else self.next_pp_rank)

    def commit_and_wait_comm(self) -> None:
        if not self.comm_ops:
            return
        nvtx.range_push("pipeline send/recv")
        reqs = dist.batch_isend_irecv(self.comm_ops)
        for req in reqs:
            req.wait()
        self.comm_ops = []
        self.free_sent_tensors()
        nvtx.range_pop()

    def step(
        self,
        microbatches: List[Microbatch],
        objective: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Run one training step, or a forward-only step under torch.no_grad().

        microbatches must be identical on every pipeline rank, at least min_chunks(pp_size) long.
        objective, required on the first pipeline rank only, is called once per micro-batch as
        objective(model_outputs, objective_inputs) and returns (loss, objective_output); the loss
        is backpropagated, or None in a forward-only step.

        Returns one objective_output per micro-batch on the first pipeline rank and an empty list
        elsewhere. Each loss is detached in place after its backward, so an objective_output that
        needs the loss value must hold loss.detach().
        """
        assert not (training.offloaded and self is training.model), (
            "training state is offloaded; call reload_training_state"
        )
        # A step whose signature already ran with full Dynamo guards skips evaluating them; see
        # guard_skip for why that is safe.
        return self.guard_skip.run(self, microbatches, self.run_step, microbatches, objective)

    def run_step(self, microbatches: List[Microbatch], objective: Optional[Callable]) -> List[Any]:
        self.forward_only = not torch.is_grad_enabled()

        # ZeRO-1 for the step: no resharding, gradient sync or root post-backward callback after
        # each backward; the end of step() runs all three once.
        for module in self.module:
            if isinstance(module, FSDPModule):
                module.set_is_last_backward(False)
                module.set_reshard_after_backward(False)
                module.set_requires_gradient_sync(False)
                if not self.forward_only:
                    state_ctx = fully_shard.state(module)._state_ctx
                    assert hasattr(state_ctx, "post_backward_final_callback_queued")
                    state_ctx.post_backward_final_callback_queued = True

        pp_rank, pp_size = distributed.pp_rank, distributed.pp_size

        if self.is_first_pp_rank:
            assert objective is not None, (
                "the first pipeline rank holds the final model outputs and needs an objective"
            )

        self.reset_states()
        FP8WeightCacheControl.step()
        num_chunks = self.setup_step_metadata(microbatches)
        floor = min_chunks(pp_size)
        assert num_chunks >= floor, f"{pp_size=} requires {floor=} chunks got {num_chunks=}"
        self.ensure_chunk_records_allocated(num_chunks)

        if self.is_first_pp_rank:
            self.input_chunks = ([tuple(microbatch.model_inputs) for microbatch in microbatches], [])  # fmt: skip
            self.objective_inputs = [microbatch.objective_inputs for microbatch in microbatches]
            self.objective = objective

        # Step 1: nF0
        step_1 = (pp_size - pp_rank - 1) * 2
        for i in range(step_1):
            self.forward_chunk(0)

        # Step 2: nF0F1
        step_2 = pp_rank + 1
        self.recv_forward(0)
        for i in range(step_2):
            self.forward_chunk(0, recv=False, send=False)
            self.recv_forward(0)
            self.forward_chunk(1, send=(not self.is_last_pp_rank) or (i < step_2 - 1))
            self.send_forward(0)

        # Step 3: nB1W1F1
        step_3 = pp_size - pp_rank - 1
        for i in range(step_3):
            self.backward_chunk(1, enable_zb=True)
            self.recv_forward(1)
            self.weight_chunk()
            self.forward_chunk(1, recv=False)

        # Step 4 (Main step): nF0B1F1B0
        step_4 = num_chunks - pp_size * 2 + pp_rank + 1
        for i in range(step_4):
            if i == 0:
                if self.is_last_pp_rank:
                    # These two chunks are not overlapped, which shrinks the bubble further.
                    self.forward_chunk(0, recv=False, send=False)
                    self.send_forward(1)
                    self.backward_chunk(1, send=False)
                    self.send_forward(0)
                    self.send_backward(1)
                else:
                    self.forward_backward_chunk(0, 1, recv0=False)
            else:
                self.forward_backward_chunk(0, 1)
            self.forward_backward_chunk(1, 0)

        # Step 5: nB1F1B0
        step_5 = pp_size - pp_rank - 1
        for i in range(step_5):
            self.backward_chunk(1)
            self.forward_backward_chunk(1, 0)

        # Step 6: nB1B0 (The second half of the chunks use zero bubble)
        step_6 = pp_rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and pp_rank % 2 == 1:
                enable_zb = True
            self.backward_chunk(1, enable_zb=enable_zb)
            if i == step_6 // 2 and pp_rank % 2 == 0:
                enable_zb = True
            self.backward_chunk(0, enable_zb=enable_zb)

        # Step 7: nWB0
        step_7 = pp_size - pp_rank - 1
        for i in range(step_7):
            self.weight_chunk()
            self.backward_chunk(0, enable_zb=True)

        # Step 8: nW
        step_8 = pp_rank + 1
        for i in range(step_8):
            self.weight_chunk()
        assert WeightGradStore.funcs_queue.empty()

        self.commit_and_wait_comm()

        objective_outputs = self.objective_output_chunks

        self.reset_states()

        # Release FP8 weight caches so the memory is available for optimizer.step().
        FP8WeightCacheControl.clear(*self.module)

        # End of the ZeRO-1 step: fold the bf16 gradients into the fp32 buffers, reduce-scatter
        # each group once, reshard its weights, and run the root callback skipped above.
        for module in self.module:
            if not isinstance(module, FSDPModule):
                continue
            # Nothing overlaps the end-of-step reduce, so run its copies on the compute stream:
            # buffers then come from the compute stream's allocator pool, which the pipeline has
            # just emptied, instead of a reduce-scatter pool per root that stays reserved (and
            # unusable by the pipeline) through the whole next step.
            comm_ctx = fully_shard.state(module)._comm_ctx
            comm_ctx.reduce_scatter_stream = torch.cuda.current_stream()
            module.set_is_last_backward(True)
            module.set_reshard_after_backward(True)
            module.set_requires_gradient_sync(True)
            fsdp_state = fully_shard.state(module)
            for state in fsdp_state._state_ctx.all_states:
                if state._fsdp_param_group:
                    for fsdp_param in state._fsdp_param_group.fsdp_params:
                        if hasattr(fsdp_param, "_unsharded_param"):
                            fsdp_param.accumulate_unsharded_grad_if_needed()
                            fsdp_param.to_accumulated_grad_if_needed()
                    state._fsdp_param_group.post_backward()
            fsdp_state._root_post_backward_final_callback()

        return objective_outputs


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
    Run the forward of one micro-batch through module0 interleaved with the backward of another
    through module1, stage by stage, so each all-to-all of one overlaps compute of the other.

    The two modules may differ by one layer. Records are filled in chunk_record0 and consumed from
    chunk_record1 in place. Returns (outputs0, loss0, objective_output0, input_grads1).
    """
    assert abs(len(module0.layers) - len(module1.layers)) <= 1
    assert len(chunk_record1.layers) == len(module1.layers)
    num_layers = min(len(module0.layers), len(module1.layers))
    module0_layers = list(module0.layers.values())
    module1_layers = list(module1.layers.values())

    (hidden_states,) = inputs0
    layer_idx0 = 0
    dispatch_tokens_grad = moe_outs = None

    ctx = ExecutionCtx(comm_stream=comm_stream)

    if loss1 is not None:
        nvtx.range_push("loss1.backward()")
        loss1.backward()
        loss1.detach_()
        nvtx.range_pop()
        output_grads1 = [chunk_record1.epilog.args.hidden_states.grad]
        chunk_record1.epilog.args = None
        loss1 = None
    assert output_grads1 is not None

    record = chunk_record1.layers[-1].stage5
    moe_outs_grad, topk_weight_grad, residual_grad = stage5_b(
        ctx, module1_layers[-1], record, tuple(output_grads1)
    )

    record = chunk_record1.layers[-1].stage4
    moe_outs_grad = stage4_b(ctx, module1_layers[-1], record, (moe_outs_grad,))

    if module0.stage_index == 0:
        hidden_states = prolog_f(module0, hidden_states, chunk_record0.prolog)

    rotary_posemb = module0.forward_posemb(hidden_states.shape[1], cu_seqlens)

    record, dispatch_tokens, residual, routing = stage1_f(
        ctx, module0_layers[0], hidden_states, rotary_posemb, cu_seqlens
    )
    chunk_record0.layers[layer_idx0].stage1.args = record.args
    chunk_record0.layers[layer_idx0].stage1.outs = record.outs

    for layer_idx in range(num_layers):
        if layer_idx != 0:
            # Merged boundary: stage1.outs without stage1.args on the next layer, stage5.args
            # without stage5.outs on the previous one; unmerged boundaries set all four.
            stage1_record = chunk_record1.layers[-layer_idx].stage1
            use_merged = stage1_record.outs is not None and stage1_record.args is None

            if use_merged:
                next_layer, prev_layer = module1_layers[-layer_idx], module1_layers[-layer_idx - 1]
                stage1_outs = stage1_record.outs
                stage5_args = chunk_record1.layers[-layer_idx - 1].stage5.args
                grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)
                moe_outs_grad, topk_weight_grad, residual_grad = stage1_stage5_b(
                    ctx, next_layer, prev_layer, stage1_outs, stage5_args, grad_tensors
                )
                clear_layer_records(chunk_record1.layers[-layer_idx])
            else:
                record = chunk_record1.layers[-layer_idx].stage1
                grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)
                hidden_states_grad = stage1_b(ctx, module1_layers[-layer_idx], record, grad_tensors)
                record = chunk_record1.layers[-layer_idx - 1].stage5
                moe_outs_grad, topk_weight_grad, residual_grad = stage5_b(
                    ctx, module1_layers[-layer_idx - 1], record, (hidden_states_grad,)
                )
                clear_layer_records(chunk_record1.layers[-layer_idx])

            record, moe_outs = stage4_f(
                ctx,
                module0_layers[layer_idx - 1],
                moe_outs,
                routing.combine_splits if routing is not None else None,
            )
            chunk_record0.layers[layer_idx0].stage4.splits = record.splits
            if routing is not None and ctx.fwd_comm_work is not None:
                ctx.fwd_comm_deferred_free.append(
                    chunk_record0.layers[layer_idx0].stage3.outs.moe_outs
                )  # freed after Stage 5 waits

            record = chunk_record1.layers[-layer_idx - 1].stage4
            moe_outs_grad = stage4_b(ctx, module1_layers[-layer_idx - 1], record, (moe_outs_grad,))

            # Fuse module0's stage 5 of the previous layer with stage 1 of this one, except at the
            # first and last boundaries when module0 has the extra layer.
            if len(module0_layers) <= len(module1.layers) or (layer_idx - 1 > 0 and layer_idx < num_layers - 1):  # fmt: skip
                prev_layer, next_layer = module0_layers[layer_idx - 1], module0_layers[layer_idx]
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
                record, hidden_states = stage5_f(
                    ctx,
                    module0_layers[layer_idx - 1],
                    moe_outs,
                    routing,
                    residual,
                )
                chunk_record0.layers[layer_idx0].stage5.args = record.args
                chunk_record0.layers[layer_idx0].stage5.outs = record.outs
                layer_idx0 += 1
                record, dispatch_tokens, residual, routing = stage1_f(
                    ctx, module0_layers[layer_idx], hidden_states, rotary_posemb, cu_seqlens
                )
                chunk_record0.layers[layer_idx0].stage1.args = record.args
                chunk_record0.layers[layer_idx0].stage1.outs = record.outs

        record = chunk_record1.layers[-layer_idx - 1].stage3
        gathered_tokens_grad = stage3_b(ctx, module1_layers[-layer_idx - 1], record, (moe_outs_grad,))  # fmt: skip

        record, gathered_tokens = stage2_f(
            ctx,
            module0_layers[layer_idx],
            dispatch_tokens,
            routing.dispatch_splits if routing is not None else None,
        )
        chunk_record0.layers[layer_idx0].stage2.splits = record.splits
        if routing is not None and ctx.fwd_comm_work is not None:
            ctx.fwd_comm_deferred_free.append(dispatch_tokens)  # freed after Stage 3 waits

        record = chunk_record1.layers[-layer_idx - 1].stage2
        dispatch_tokens_grad = stage2_b(
            ctx, module1_layers[-layer_idx - 1], record, (gathered_tokens_grad,)
        )

        stage3_w(module1_layers[-layer_idx - 1])

        record, moe_outs = stage3_f(
            ctx,
            module0_layers[layer_idx],
            gathered_tokens,
            routing.expert_idxs if routing is not None else None,
            routing.expand_idx if routing is not None else None,
            routing.expert_counts if routing is not None else None,
        )
        chunk_record0.layers[layer_idx0].stage3.args = record.args
        chunk_record0.layers[layer_idx0].stage3.outs = record.outs

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

    record = chunk_record1.layers[-num_layers].stage1

    grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)
    hidden_states_grad = stage1_b(ctx, module1_layers[-num_layers], record, grad_tensors)

    clear_layer_records(chunk_record1.layers[-num_layers])

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
        hidden_states = layer_forward(
            module0_layers[-1],
            hidden_states,
            rotary_posemb,
            chunk_record0.layers[layer_idx0],
            cu_seqlens,
        )
        layer_idx0 += 1
    elif len(module0.layers) + 1 == len(module1.layers):
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
        record.outs = None
    if module0.stage_index == module0.stage_count - 1:
        hidden_states = epilog_f(module0, hidden_states, chunk_record0.epilog)

    outputs0 = [hidden_states]
    if objective0 is not None:
        nvtx.range_push("objective0(model_outputs0, objective_inputs0)")
        loss0, objective_output0 = objective0(tuple(outputs0), objective_inputs0)
        nvtx.range_pop()
    else:
        loss0, objective_output0 = None, None

    return outputs0, loss0, objective_output0, final_grads
