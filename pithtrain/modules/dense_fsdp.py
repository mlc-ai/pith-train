"""FSDP2 configuration and sharding for a dense text tower."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import batched, groupby
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from pithtrain.config import SlottedDefault
from pithtrain.contexts import logging, training
from pithtrain.kernels import stage_compile_options

if TYPE_CHECKING:
    from pithtrain.models.qwen35 import Qwen35TextModel

__all__ = ["DenseFSDPCfg", "shard_dense_model"]


@dataclass(init=False, slots=True)
class DenseFSDPCfg(SlottedDefault):
    blocks_per_group: int = 1
    """Number of consecutive decoder blocks per communication bucket."""

    group_by_attention_type: bool = False
    """Start a new bucket at each boundary between DeltaNet and gated attention."""

    edge_sharding: Literal["separate", "root"] = "separate"
    """Shard embedding and LM head independently or put both in the root's bucket."""

    reshard_after_forward: bool | int = True
    """Reshard to the full mesh, retain gathered parameters, or reshard to a smaller group."""

    keep_last_groups: int = 0
    """Keep this many final decoder buckets materialized until backward."""

    keep_gathered_across_microbatches: bool = False
    """Keep the selected final buckets unsharded through intermediate backward passes."""

    skip_embedding_backward_gather: bool = False
    """Skip gathering the embedding weight in backward when it is its own bucket."""

    prefetch: int = 0
    """Explicit lookahead in buckets in both directions. Zero uses FSDP's implicit prefetch."""

    reduce_dtype: Literal["fp32", "bf16"] = "fp32"

    bf16_sharded_gradients: bool = False
    """Accumulate reduced shards in BF16 and upcast once before AdamW."""

    reduce_at_step_end: bool = False
    """Accumulate unsharded gradients across microbatches and reduce them in the last backward."""

    overlap_microbatch_reductions: bool = False
    """Wait for pending reduce-scatters only on the optimizer step's final backward."""

    prefetch_data_during_optimizer: bool = False
    """Prepare the next batch while the current optimizer kernels run."""

    reduce_scatter_buffers: int = 1
    """Maximum number of in-flight reduce-scatter staging buffers."""

    separate_reduce_scatter: bool = False
    """Use a dedicated communicator for reduce-scatter to allow backward all-gather overlap."""

    collective_buffers: Literal["default", "process-group", "symmetric"] = "default"
    """Select the allocator for all-gather and reduce-scatter staging buffers."""

    activation_checkpoint: Literal["none", "selective", "full", "offload"] = "none"

    compile_blocks: bool = False

    compile_mode: Literal[
        "default", "max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead"
    ] = "default"


def checkpoint_policy(ctx, op, *args, **kwargs):
    if op in {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.pithtrain.gated_delta_rule_dense_fwd.default,
        torch.ops.pithtrain.flash_attn4_mha_fwd.default,
    }:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def selective_checkpoint_contexts():
    return create_selective_checkpoint_contexts(checkpoint_policy)


def offload_saved_activations(layer, forward):
    """Keep parameter storages on device and stage large saved activations in pinned memory."""

    def call(*args, **kwargs):
        parameter_storages = {p.untyped_storage().data_ptr() for p in layer.parameters()}

        def pack(tensor):
            if (
                not tensor.is_cuda
                or tensor.numel() * tensor.element_size() < 1024**2
                or tensor.untyped_storage().data_ptr() in parameter_storages
            ):
                return tensor.detach(), None
            host = torch.empty_like(tensor, device="cpu", pin_memory=True)
            host.copy_(tensor.detach(), non_blocking=True)
            return host, tensor.device

        def unpack(saved):
            tensor, device = saved
            return tensor if device is None else tensor.to(device, non_blocking=True)

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return forward(*args, **kwargs)

    return call


def log_bucket_sizes(
    buckets: list[tuple[str, list[nn.Module]]], shard_size: int, reduce_dtype: torch.dtype
) -> None:
    logger = getattr(logging, "stdout", None)
    if logger is None:
        return
    sizes = Counter()
    for name, modules in buckets:
        numel = sum(
            (-(-p.size(0) // shard_size) * shard_size) * (p.numel() // p.size(0))
            for module in modules
            for p in module.parameters()
        )
        sizes[name, numel] += 1
    payloads = (
        f"{count} {name} {numel * training.PARAM_DTYPE.itemsize / 2**20:.1f}/"
        f"{numel * reduce_dtype.itemsize / 2**20:.1f}"
        for (name, numel), count in sizes.items()
    )
    logger.info("Dense FSDP buckets (AG/RS MiB) | %s", ", ".join(payloads))


def shard_dense_model(model: Qwen35TextModel, mesh: DeviceMesh, cfg: DenseFSDPCfg) -> None:
    if cfg.blocks_per_group < 1:
        raise ValueError("dense_fsdp.blocks_per_group must be positive")
    if cfg.keep_last_groups < 0:
        raise ValueError("dense_fsdp.keep_last_groups must be nonnegative")
    if cfg.keep_gathered_across_microbatches and not cfg.keep_last_groups:
        raise ValueError("Keeping buckets across microbatches requires keep_last_groups > 0")
    if cfg.edge_sharding not in ("separate", "root"):
        raise ValueError("dense_fsdp.edge_sharding must be separate or root")
    if cfg.edge_sharding == "root" and cfg.skip_embedding_backward_gather:
        raise ValueError("Skipping the embedding backward gather requires a separate embedding")
    if cfg.prefetch < 0:
        raise ValueError("dense_fsdp.prefetch must be nonnegative")
    if cfg.reduce_dtype not in ("fp32", "bf16"):
        raise ValueError("dense_fsdp.reduce_dtype must be fp32 or bf16")
    if cfg.bf16_sharded_gradients and cfg.reduce_dtype != "bf16":
        raise ValueError("BF16 sharded gradients require BF16 reduction")
    if cfg.activation_checkpoint not in ("none", "selective", "full", "offload"):
        raise ValueError(
            "dense_fsdp.activation_checkpoint must be none, selective, full or offload"
        )

    if cfg.collective_buffers not in ("default", "process-group", "symmetric"):
        raise ValueError(
            "dense_fsdp.collective_buffers must be default, process-group or symmetric"
        )
    if cfg.collective_buffers == "symmetric" and cfg.reduce_scatter_buffers != 1:
        raise ValueError("Symmetric reduce-scatter requires one input buffer")

    reduce_dtype = torch.float32 if cfg.reduce_dtype == "fp32" else torch.bfloat16
    if cfg.bf16_sharded_gradients:
        for param in model.parameters():
            param.grad_dtype = torch.bfloat16
    mp = MixedPrecisionPolicy(
        param_dtype=training.PARAM_DTYPE,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    if cfg.compile_blocks:
        from torch._inductor import list_mode_options

        compile_options = list_mode_options(cfg.compile_mode)
        if cfg.compile_mode in ("default", "reduce-overhead"):
            compile_options |= stage_compile_options()
    for layer in model.layers:
        if cfg.activation_checkpoint in ("selective", "full"):
            ctx = (
                {"context_fn": selective_checkpoint_contexts}
                if cfg.activation_checkpoint == "selective"
                else {}
            )
            layer.forward = partial(
                checkpoint, layer.forward, use_reentrant=False, preserve_rng_state=False, **ctx
            )
        if cfg.compile_blocks:
            layer.forward = torch.compile(
                layer.forward,
                fullgraph=True,
                dynamic=False,
                options=compile_options or None,
            )
        if cfg.activation_checkpoint == "offload":
            layer.forward = offload_saved_activations(layer, layer.forward)

    decoder_groups = [
        list(group)
        for _, run in groupby(
            model.layers, key=lambda layer: cfg.group_by_attention_type and layer.is_linear
        )
        for group in batched(run, cfg.blocks_per_group)
    ]
    buckets = []
    for group in decoder_groups:
        kinds = {layer.layer_type for layer in group}
        label = next(iter(kinds)) if len(kinds) == 1 else "hybrid"
        buckets.append((label, group))
    if cfg.edge_sharding == "separate":
        buckets += [
            ("embedding", [model.embed_tokens]),
            ("head", [model.lm_head]),
            ("root", [model.norm]),
        ]
    else:
        buckets.append(("root", [model.embed_tokens, model.norm, model.lm_head]))
    log_bucket_sizes(buckets, mesh.size(mesh.ndim - 1), reduce_dtype)

    for i, group in enumerate(decoder_groups):
        keep = i >= len(decoder_groups) - cfg.keep_last_groups
        reshard = False if keep else cfg.reshard_after_forward
        fully_shard(
            group[0] if len(group) == 1 else group,
            mesh=mesh,
            reshard_after_forward=reshard,
            mp_policy=mp,
        )
    if cfg.keep_gathered_across_microbatches:
        model.retained_decoder_units = tuple(
            group[0] for group in decoder_groups[-cfg.keep_last_groups :]
        )

    # One FSDP state per communication bucket, even when several blocks share it.
    order = [group[0] for group in decoder_groups]
    if cfg.edge_sharding == "separate":
        fully_shard(model.embed_tokens, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
        fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
        if cfg.skip_embedding_backward_gather:
            model.embed_tokens.set_unshard_in_backward(False)
        order = [model.embed_tokens, *order, model.lm_head]
    fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp)

    if cfg.edge_sharding == "root" and cfg.prefetch:
        model.set_modules_to_forward_prefetch(order[: cfg.prefetch])
        model.set_modules_to_backward_prefetch(order[-cfg.prefetch :][::-1])
    for i, module in enumerate(order):
        module.set_gradient_divide_factor(1.0)
        if cfg.prefetch:
            module.set_modules_to_forward_prefetch(order[i + 1 : i + 1 + cfg.prefetch])
            module.set_modules_to_backward_prefetch(order[max(0, i - cfg.prefetch) : i][::-1])
    model.set_gradient_divide_factor(1.0)
    if cfg.reduce_scatter_buffers != 1:
        model.set_reduce_scatter_max_input_buffers(cfg.reduce_scatter_buffers)
    if cfg.separate_reduce_scatter:
        model.set_separate_reduce_scatter_group()
    if cfg.collective_buffers != "default":
        for unit in (*order, model):
            if cfg.collective_buffers == "symmetric":
                unit.set_force_sum_reduction_for_comms(True)
                unit.set_symm_mem_for_comm()
            else:
                unit.set_allocate_memory_from_process_group_for_comm(True)
