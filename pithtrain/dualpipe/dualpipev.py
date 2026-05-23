"""
DualPipeV: Overlapped forward-backward pipeline parallelism.

The ``DualPipeV`` class in this module is derived from the DualPipeV
implementation in DeepSeek's DualPipe project
(https://github.com/deepseek-ai/DualPipe), which is licensed under the
MIT License. Copyright (c) 2025 DeepSeek. See ``pithtrain/dualpipe/LICENSE``
and the project-root ``NOTICE`` file for the full license text and details
of which portions are derived.

The 8-step scheduling algorithm in ``DualPipeV.step()`` and the P2P
communication orchestration methods are closely adapted from the original.
The ``overlapped_forward_backward()`` function (see ``overlap.py``), FSDP
integration, FP8 weight caching, and the 5-stage decomposition are original
additions.

Stage Mapping:
    - Stage 1: Attention (LN + Attn + LN + Expert selection)
    - Stage 2: Dispatch (All-to-all dispatch for expert parallelism)
    - Stage 3: MLP (Expert/MLP computation)
    - Stage 4: Combine (All-to-all combine for expert parallelism)
    - Stage 5: Aggregate (Weighted expert output + residual connection)
"""

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard

from pithtrain.dualpipe import comm
from pithtrain.dualpipe.execution import IntermediateTensors, create_intermediate_tensors
from pithtrain.dualpipe.overlap import overlapped_forward_backward
from pithtrain.dualpipe.utils import FP8WeightCacheControl, WeightGradStore, gather, scatter


class DualPipeV(nn.Module):
    """V-shaped bidirectional pipeline parallelism scheduler.

    Derived from the DualPipeV class in DeepSeek's DualPipe project
    (https://github.com/deepseek-ai/DualPipe), which implements the algorithm
    described in the `DeepSeek-V3 Technical Report <https://arxiv.org/abs/2412.19437>`_.
    The original V-shape "cut-in-half" procedure was introduced by Sea AI Lab.

    This implementation extends the original with:
      - A 5-stage overlapped forward-backward loop (``overlapped_forward_backward``)
        that decomposes each transformer layer into Attention / Dispatch / MLP /
        Combine / Aggregate stages for fine-grained computation-communication overlap.
      - FSDP2 integration (hook suppression during the pipeline loop, manual
        ``post_backward`` invocation after the loop).
      - FP8 weight caching across micro-batches via ``FP8WeightCacheControl``.
      - Pre-allocated ``IntermediateTensors`` for zero-allocation pipeline execution.
    """

    def __init__(
        self,
        modules: Tuple[nn.Module, nn.Module],
        pp_group: dist.ProcessGroup,
        ep_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()

        device = torch.device(torch.cuda.current_device())
        assert next(modules[0].parameters()).device == device
        self.module = nn.ModuleList(modules)
        self.batch_dim = 0
        self.rank = torch.distributed.get_rank()

        self.pp_group = pp_group
        self.ep_group = ep_group
        self.pp_size = self.pp_group.size()
        self.ep_size = self.ep_group.size()
        self.ep_rank = self.ep_group.rank()
        self.pp_rank = self.pp_group.rank()
        self.prev_pp_rank = self.pp_rank - 1 if self.pp_rank > 0 else None
        self.next_pp_rank = self.pp_rank + 1 if self.pp_rank < self.pp_size - 1 else None
        self.is_first_pp_rank = self.pp_rank == 0
        self.is_last_pp_rank = self.pp_rank == self.pp_size - 1

        self.comm_stream = torch.cuda.Stream(device=device)

        # Pre-allocation tracking
        self._num_chunks_allocated = 0
        self.intermediate_tensors_chunks: Tuple[
            List[IntermediateTensors], List[IntermediateTensors]
        ] = ([], [])

    def _ensure_intermediate_tensors_allocated(self, num_chunks: int) -> None:
        """Pre-allocate IntermediateTensors structures for reuse across iterations."""
        if self._num_chunks_allocated == num_chunks:
            return
        self.intermediate_tensors_chunks = (
            [
                create_intermediate_tensors(
                    len(self.module[0].layers),
                    self.module[0].embed_tokens is not None,
                    self.module[0].norm is not None,
                )
                for _ in range(num_chunks)
            ],
            [
                create_intermediate_tensors(
                    len(self.module[1].layers),
                    self.module[1].embed_tokens is not None,
                    self.module[1].norm is not None,
                )
                for _ in range(num_chunks)
            ],
        )
        self._num_chunks_allocated = num_chunks

    def _reset_states(self) -> None:
        WeightGradStore.clear()

        self.input_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = (
            [],
            [],
        )
        self.output_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        # Note: intermediate_tensors_chunks is pre-allocated and reused, not reset here
        self.input_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.output_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = (
            [],
            [],
        )
        self.labels: List[List[torch.Tensor]] = None
        self.loss_chunks: List[torch.Tensor] = []
        self.criterion: Callable = None

        self.current_f_chunk_id: List[int] = [0, 0]
        self.current_b_chunk_id: List[int] = [0, 0]
        self.current_send_f_chunk_id: List[int] = [0, 0]
        self.current_send_b_chunk_id: List[int] = [0, 0]
        self.current_recv_f_chunk_id: List[int] = [0, 0]
        self.current_recv_b_chunk_id: List[int] = [0, 0]
        self.comm_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []

    def _forward_compute_chunk(self, phase: int) -> None:
        chunk_id = self.current_f_chunk_id[phase]
        self.current_f_chunk_id[phase] += 1
        inputs = self.input_chunks[phase][chunk_id]
        if self.forward_only:
            self.input_chunks[phase][chunk_id] = None

        is_last_stage = self.is_first_pp_rank and phase == 1

        nvtx.range_push(f"forward chunk {chunk_id} (phase{phase})")
        # Set pre-allocated intermediate_tensors on module to avoid FSDP kwarg handling issues
        intermediate_tensors = self.intermediate_tensors_chunks[phase][chunk_id]
        self.module[phase]._intermediate_tensors = intermediate_tensors
        outputs = self.module[phase](*inputs)
        self.module[phase]._intermediate_tensors = None
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
        if is_last_stage and self.criterion is not None:
            labels = self.labels[chunk_id]
            loss = self.criterion(*outputs, *labels)
            self.loss_chunks.append(loss)
        nvtx.range_pop()

        if self.is_last_pp_rank and phase == 0:
            self.input_chunks[1].append([output.detach().requires_grad_() for output in outputs])
        if (not is_last_stage) or self.return_outputs:
            self.output_chunks[phase].append(outputs)
        # No need to append - intermediate_tensors is pre-allocated and was modified in place

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
            input_grads = self.module[phase].__class__.backward(
                self.module[phase],
                None,
                loss,
                self.intermediate_tensors_chunks[phase][chunk_id],
            )
            loss.detach_()
        else:
            outputs = self.output_chunks[phase][chunk_id]
            if not self.return_outputs:
                self.output_chunks[phase][chunk_id] = None
            output_grads = self.output_grad_chunks[phase][chunk_id]
            self.output_grad_chunks[phase][chunk_id] = None
            non_empty = [(t, g) for t, g in zip(outputs, output_grads) if g is not None]
            outputs, output_grads = list(zip(*non_empty))
            if len(outputs) > 0:
                input_grads = self.module[phase].__class__.backward(
                    self.module[phase],
                    output_grads,
                    None,
                    self.intermediate_tensors_chunks[phase][chunk_id],
                )
        # Note: intermediate_tensors is pre-allocated and reused; backward clears tensor refs inside
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
        is_last_stage0 = self.is_first_pp_rank and phase0 == 1

        if is_last_stage0 and self.criterion is not None:
            labels0 = self.labels[chunk_id0]
            criterion0 = self.criterion
        else:
            labels0 = []
            criterion0 = None

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
            if not self.return_outputs:
                self.output_chunks[phase1][chunk_id1] = None
            output_grads1 = self.output_grad_chunks[phase1][chunk_id1]
            self.output_grad_chunks[phase1][chunk_id1] = None
            non_empty = [(t, g) for t, g in zip(outputs1, output_grads1) if g is not None]
            outputs1, output_grads1 = list(zip(*non_empty))

        # forward & backward (intermediate_tensors0 is modified in place)
        nvtx.range_push(
            f"forward chunk {chunk_id0} (phase{phase0}) backward chunk {chunk_id1} (phase{phase1})"
        )
        outputs0, loss0, input_grads1 = overlapped_forward_backward(
            module0,
            inputs0,
            criterion0,
            labels0,
            self.intermediate_tensors_chunks[phase0][chunk_id0],
            module1,
            loss1,
            outputs1,
            output_grads1,
            self.intermediate_tensors_chunks[phase1][chunk_id1],
            self.comm_stream,
            self.ep_group,
        )
        nvtx.range_pop()

        # post-forward
        if self.is_last_pp_rank and phase0 == 0:
            self.input_chunks[1].append([output.detach().requires_grad_() for output in outputs0])
        if (not is_last_stage0) or self.return_outputs:
            self.output_chunks[phase0].append(outputs0)
        if is_last_stage0 and self.criterion is not None:
            self.loss_chunks.append(loss0)

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

    def _recv_forward(self, phase: int) -> None:
        if (self.is_first_pp_rank and phase == 0) or (self.is_last_pp_rank and phase == 1):
            return

        self.current_recv_f_chunk_id[phase] += 1
        tensors = comm.append_irecv(
            self.comm_ops,
            self.prev_pp_rank if phase == 0 else self.next_pp_rank,
            self.pp_group,
        )
        self.input_chunks[phase].append(tensors)

    def _send_forward(self, phase: int) -> None:
        if (self.is_first_pp_rank and phase == 1) or (self.is_last_pp_rank and phase == 0):
            return

        chunk_id = self.current_send_f_chunk_id[phase]
        self.current_send_f_chunk_id[phase] += 1
        tensors = self.output_chunks[phase][chunk_id]

        comm.append_isend(
            self.comm_ops,
            tensors,
            self.next_pp_rank if phase == 0 else self.prev_pp_rank,
            self.pp_group,
        )

        if not self.return_outputs:
            self.to_free.extend(tensors)

    def _recv_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_first_pp_rank and phase == 1) or (self.is_last_pp_rank and phase == 0):
            return

        self.current_recv_b_chunk_id[phase] += 1
        tensors = comm.append_irecv(
            self.comm_ops,
            self.next_pp_rank if phase == 0 else self.prev_pp_rank,
            self.pp_group,
        )
        assert None not in tensors
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

        comm.append_isend(
            self.comm_ops,
            tensors,
            self.prev_pp_rank if phase == 0 else self.next_pp_rank,
            self.pp_group,
        )

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
        *inputs: Optional[torch.Tensor],
        num_chunks: int = 0,
        criterion: Optional[Callable] = None,
        labels: List[Optional[torch.Tensor]] = [],
        return_outputs: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
        """
        Execute a training or inference step.

        Arguments:
            *inputs: Module inputs. Required only on the first rank.
            num_chunks: The number of micro-batches.
            criterion: Loss function, invoked as ``criterion(*outputs, *labels)``. Required only on the first rank.
            labels: Labels of the loss function. Required only on the first rank.
            return_outputs: Whether to return outputs on the first rank. Default: ``False``.

        Returns: (loss, outputs)
            loss: Loss for the batch. Returned only on the first rank.
            outputs: Module outputs. Returned only if ``return_outputs=True`` and on the first rank.

        """
        assert comm.TENSOR_SHAPES is not None and comm.TENSOR_DTYPE is not None, (
            "You need to call set_p2p_tensor_shapes and set_p2p_tensor_dtype before executing a step."
        )
        self.forward_only = not torch.is_grad_enabled()
        self.return_outputs = return_outputs

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

        pp_rank = self.pp_rank
        pp_size = self.pp_size
        assert num_chunks > 0 and num_chunks >= pp_size * 2, f"{num_chunks=}, {pp_size=}"

        if not self.forward_only and self.is_first_pp_rank:
            assert criterion is not None

        self._reset_states()
        if FP8WeightCacheControl.enabled:
            FP8WeightCacheControl.step()
        self._ensure_intermediate_tensors_allocated(num_chunks)

        if self.is_first_pp_rank:
            self.input_chunks = (scatter(inputs, num_chunks, self.batch_dim), [])
            self.labels = scatter(labels, num_chunks, self.batch_dim)
            self.criterion = criterion

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

        loss, outputs = None, None
        if self.is_first_pp_rank:
            if criterion is not None:
                loss = torch.stack(self.loss_chunks)
            if return_outputs:
                outputs = gather(self.output_chunks[1], self.batch_dim)
                if len(outputs) == 1:
                    outputs = outputs[0]

        self._reset_states()

        # Release FP8 weight caches so the memory is available for optimizer.step().
        # They will be regenerated on the next forward pass.
        if FP8WeightCacheControl.enabled:
            FP8WeightCacheControl.clear_caches(*self.module)

        # Manually call post backward for FSDP
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

            # it would be much better if pipelining backward invoked .backward so autograd hooks
            # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
            # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
            fsdp_state._root_post_backward_final_callback()

        for module in self.module:
            if isinstance(module, FSDPModule):
                run_post_backward(module)

        return loss, outputs
