"""FSDP2 configuration and sharding for a dense text tower."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

import torch
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from pithtrain.config import SlottedDefault
from pithtrain.contexts import training
from pithtrain.kernels import stage_compile_options

if TYPE_CHECKING:
    from pithtrain.models.qwen35 import Qwen35TextModel

__all__ = ["DenseFSDPCfg", "shard_dense_model"]


@dataclass(init=False, slots=True)
class DenseFSDPCfg(SlottedDefault):
    reshard_after_forward: bool | int = True
    """Reshard to the full mesh, retain gathered parameters, or reshard to a smaller group."""

    prefetch: int = 0
    """Explicit lookahead in blocks in both directions. Zero uses FSDP's implicit prefetch."""

    reduce_dtype: Literal["fp32", "bf16"] = "fp32"

    activation_checkpoint: Literal["none", "selective", "full"] = "none"

    compile_blocks: bool = False


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


def shard_dense_model(model: Qwen35TextModel, mesh: DeviceMesh, cfg: DenseFSDPCfg) -> None:
    if cfg.prefetch < 0:
        raise ValueError("dense_fsdp.prefetch must be nonnegative")
    if cfg.reduce_dtype not in ("fp32", "bf16"):
        raise ValueError("dense_fsdp.reduce_dtype must be fp32 or bf16")
    if cfg.activation_checkpoint not in ("none", "selective", "full"):
        raise ValueError("dense_fsdp.activation_checkpoint must be none, selective or full")

    mp = MixedPrecisionPolicy(
        param_dtype=training.PARAM_DTYPE,
        reduce_dtype=torch.float32 if cfg.reduce_dtype == "fp32" else torch.bfloat16,
        cast_forward_inputs=False,
    )
    for layer in model.layers:
        if cfg.activation_checkpoint != "none":
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
                options=stage_compile_options() or None,
            )
        fully_shard(layer, mesh=mesh, reshard_after_forward=cfg.reshard_after_forward, mp_policy=mp)

    fully_shard(model.embed_tokens, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
    fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
    fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp)

    order = [model.embed_tokens, *model.layers, model.lm_head]
    for i, module in enumerate(order):
        module.set_gradient_divide_factor(1.0)
        if cfg.prefetch:
            module.set_modules_to_forward_prefetch(order[i + 1 : i + 1 + cfg.prefetch])
            module.set_modules_to_backward_prefetch(order[max(0, i - cfg.prefetch) : i][::-1])
    model.set_gradient_divide_factor(1.0)
