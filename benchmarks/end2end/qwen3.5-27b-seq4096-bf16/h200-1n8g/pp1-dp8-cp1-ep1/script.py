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
    for blocks in batched(model.layers, 2):
        fully_shard(list(blocks), mp_policy=mp, reshard_after_forward=True)
        blocks[0].set_gradient_divide_factor(1.0)
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
