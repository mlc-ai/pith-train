"""Benchmark dense Qwen3.5 with FSDP2 on eight H200 GPUs."""

from __future__ import annotations

import argparse
import os
from functools import partial
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

if TYPE_CHECKING:
    from pithtrain.models.qwen35 import Qwen35TextModel


def checkpoint_layers(model: Qwen35TextModel, *, mlp_tail: int) -> None:
    """Checkpoint stock decoder stages, replaying only the MLP in the tail."""
    saved_ops = [
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.pithtrain.gated_delta_rule_dense_fwd.default,
        torch.ops.pithtrain.flash_attn4_mha_fwd.default,
    ]
    selective_checkpoint = partial(
        checkpoint,
        use_reentrant=False,
        preserve_rng_state=False,
        context_fn=partial(create_selective_checkpoint_contexts, saved_ops),
    )
    for i, layer in enumerate(model.layers):
        layer.forward = layer.forward_stock_stages
        if i < len(model.layers) - mlp_tail:
            layer.forward = partial(selective_checkpoint, layer.forward)
        else:
            layer.forward_stage3 = partial(selective_checkpoint, layer.forward_stage3)


def shard_model(model: Qwen35TextModel, mesh: DeviceMesh) -> None:
    checkpoint_layers(model, mlp_tail=12)

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blocks in batched(model.layers, 2):
        fully_shard(list(blocks), mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        blocks[0].set_gradient_divide_factor(1.0)

    fully_shard(model.embed_tokens, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    model.embed_tokens.set_gradient_divide_factor(1.0)

    fully_shard(model.lm_head, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    model.lm_head.set_gradient_divide_factor(1.0)

    # The root owns the final norm and drives FSDP's implicit prefetch.
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=False)
    model.set_gradient_divide_factor(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("examples/pretrain_lm/qwen3.5-27b"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", type=Path, help="Tokenized corpus directory")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    # PithTrain's kernel imports query the current GPU.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

    cfg = PretrainLMCfg()
    cfg.dataset = args.dataset if args.dataset is not None else Path(".")
    train = cfg.training
    train.model = args.model
    train.hf_import_path = args.weights
    train.benchmark = args.dataset is None
    train.sequence_length = args.seq
    train.global_batch_size = args.batch
    train.micro_batch_size = 2
    train.max_steps = args.steps
    train.lr = 1e-5
    train.optimizer = partial(make_adamw_optimizer, fused=True)
    train.scheduler = make_constant_scheduler
    launch(cfg, shard_dense_model=shard_model)
