"""Benchmark Qwen3.5-27B at 4K tokens on one 8-GPU H200 node: pp 1, dp 8, cp 1, ep 1."""

from functools import partial
from itertools import batched
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from pithtrain.models.qwen35 import Qwen35TextModel
from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch


def customize(model: nn.Module) -> None:
    """
    Recompute: selective activation checkpointing saves the matmul, Gated DeltaNet and FA4 forward
    outputs. The first 52 of 64 layers replay the rest of each layer, the fewest that fit in
    memory; the last 12 replay only the MLP.

    Sharding: FSDP2 over all data-parallel ranks, in groups of two decoder layers plus the
    embedding, the LM head and the root, with BF16 parameters and FP32 gradient reduction. The
    gradient divide factor is 1.0 because the training step normalizes by the global token count.
    """
    model = cast(Qwen35TextModel, model)
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
        if i < len(model.layers) - 12:
            layer.forward = partial(selective_checkpoint, layer.forward)
        else:
            layer.forward_stage3 = partial(selective_checkpoint, layer.forward_stage3)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # Two layers per FSDP group.
    for layers in batched(model.layers, 2):
        fully_shard(list(layers), mp_policy=mp, reshard_after_forward=True)
        layers[0].set_gradient_divide_factor(1.0)
    for module in (model.embed_tokens, model.lm_head):
        fully_shard(module, mp_policy=mp, reshard_after_forward=True)
        module.set_gradient_divide_factor(1.0)
    # The root owns the final norm and drives FSDP's implicit prefetch.
    fully_shard(model, mp_policy=mp, reshard_after_forward=False)
    model.set_gradient_divide_factor(1.0)


cfg = PretrainLMCfg()
cfg.dataset = Path("/synthetic")

distributed = cfg.distributed
distributed.pipeline_parallel_size = 1
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 1

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-27b")
training.optimizer = make_adamw_optimizer
training.scheduler = make_constant_scheduler
training.customize = customize
training.lr = 1e-5
training.max_steps = 12
training.micro_batch_size = 2
training.global_batch_size = 64
training.sequence_length = 4096
training.fp8 = False
training.benchmark = True


if __name__ == "__main__":
    launch(cfg)
