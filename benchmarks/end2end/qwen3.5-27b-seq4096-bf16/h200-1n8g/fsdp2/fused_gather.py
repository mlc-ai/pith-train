"""Prototype direct FP32-to-BF16 packing into the collective input with PyTorch 2.15."""

import torch
from torch.distributed.fsdp._fully_shard import _fsdp_param_group
from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGatherResult
from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState


def enable() -> None:
    original_all_gather = _fsdp_param_group.foreach_all_gather

    @torch.no_grad()
    def foreach_all_gather(
        fsdp_params, group, async_op, copy_in_stream, all_gather_stream, device, all_gather_comm
    ):
        if device.type != "cuda" or any(
            param.param_dtype != torch.bfloat16
            or param.orig_dtype != torch.float32
            or param.offload_to_cpu
            or param.sharded_state != ShardedState.SHARDED
            or hasattr(param._sharded_local_tensor, "fsdp_pre_all_gather")
            for param in fsdp_params
        ):
            return original_all_gather(
                fsdp_params,
                group,
                async_op,
                copy_in_stream,
                all_gather_stream,
                device,
                all_gather_comm,
            )

        with torch.cuda.stream(copy_in_stream):
            shards = [param._sharded_param_data for param in fsdp_params]
            sizes = [shard.numel() for shard in shards]
            packed_size = sum(sizes)
            output = all_gather_comm.allocate(
                (packed_size * group.size(),), dtype=torch.bfloat16, device=device
            )
            local, output = torch.ops.fsdp.all_gather_copy_in(
                shards, output, sizes, packed_size, group.rank()
            )
        all_gather_stream.wait_stream(copy_in_stream)
        with torch.cuda.stream(all_gather_stream):
            work = all_gather_comm(output, local, group, async_op)
            event = all_gather_stream.record_event()
        return AllGatherResult(
            output, event, work, [[torch.bfloat16]] * len(shards), [[n] for n in sizes], sizes
        )

    _fsdp_param_group.foreach_all_gather = foreach_all_gather
