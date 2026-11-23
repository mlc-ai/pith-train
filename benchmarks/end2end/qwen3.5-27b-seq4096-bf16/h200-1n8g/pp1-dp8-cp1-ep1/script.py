"""
Benchmark dense Qwen3.5 with FSDP2 on eight H200 GPUs.

The recipe, not the framework, decides how to shard and recompute the model, as one plain function
on PyTorch's public APIs.
"""

from functools import partial
from itertools import batched
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from pithtrain.contexts import distributed
from pithtrain.models.qwen35 import Qwen35TextModel
from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()
cfg.dataset = Path("/synthetic")

cfg.distributed.pipeline_parallel_size = 1
cfg.distributed.context_parallel_size = 1
cfg.distributed.expert_parallel_size = 1

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-27b")
training.optimizer = partial(make_adamw_optimizer, fused=True)
training.scheduler = make_constant_scheduler
training.lr = 1e-5
training.max_steps = 8
training.micro_batch_size = 2
training.global_batch_size = 64
training.sequence_length = 4096
training.fp8 = False
training.benchmark = True


def prepare_model(model: Qwen35TextModel) -> None:
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
    mesh = distributed.attn_mesh["dp"]
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blocks in batched(model.layers, 2):
        fully_shard(list(blocks), mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        blocks[0].set_gradient_divide_factor(1.0)
    for module in (model.embed_tokens, model.lm_head):
        fully_shard(module, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        module.set_gradient_divide_factor(1.0)
    # The root owns the final norm and drives FSDP's implicit prefetch.
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=False)
    model.set_gradient_divide_factor(1.0)


if __name__ == "__main__":
    launch(cfg, prepare_model=prepare_model)
