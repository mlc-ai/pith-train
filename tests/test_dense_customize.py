"""
Check unsharded dense training on one GPU and the FSDP hook contract on two GPUs.

    torchrun --nproc-per-node=1 tests/test_dense_customize.py
    torchrun --nproc-per-node=2 tests/test_dense_customize.py
"""

from functools import partial
from unittest.mock import patch

import pytest
import torch
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor
from transformers import Qwen3_5TextConfig

from pithtrain.contexts import distributed, training
from pithtrain.modules.dataset import SyntheticDataset
from pithtrain.modules.distributed import setup_distributed
from pithtrain.modules.logging import setup_logging
from pithtrain.modules.training import (
    make_adamw_optimizer,
    make_constant_scheduler,
    setup_training,
    weight_generator,
)
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, train_step


def check_meta(model):
    assert all(param.is_meta for param in model.parameters())


def check_unsharded(cfg):
    cfg.training.customize = check_meta
    setup_training(cfg)
    model = training.model
    assert not isinstance(model, FSDPModule)
    assert model.lm_head.main_grad is None
    assert all(not isinstance(param, DTensor) for param in model.parameters())

    dataset = SyntheticDataset(model.embed_tokens.num_embeddings, cfg.training.sequence_length, 42)
    initial = model.lm_head.weight.detach().clone()
    train_step(cfg, dataset, 0)
    assert not torch.equal(model.lm_head.weight, initial)
    print("[PASS] unsharded dense training")


def check_sharded(cfg):
    cfg.training.customize = check_meta
    with pytest.raises(ValueError, match="gradients will not be reduced across the 2"):
        setup_training(cfg)

    def shard(model, *, mesh=None):
        check_meta(model)
        fully_shard(model, mesh=mesh)

    hsdp_mesh = DeviceMesh("cuda", [[0, 1]], mesh_dim_names=("replica", "shard"))
    for mesh in (distributed.attn_mesh["cp"], hsdp_mesh):
        cfg.training.customize = partial(shard, mesh=mesh)
        with pytest.raises(ValueError, match="HSDP and other mesh layouts are unsupported"):
            setup_training(cfg)

    cfg.training.customize = shard
    setup_training(cfg)
    weight = training.model.lm_head.weight
    assert isinstance(weight, DTensor)
    generator = weight_generator(cfg.training.seed, "lm_head.weight", torch.device("cpu"))
    expected = torch.empty(weight.shape).normal_(std=cfg.training.init_std, generator=generator)
    torch.testing.assert_close(weight.detach().full_tensor().cpu(), expected, rtol=0, atol=0)
    if distributed.rank == 0:
        print("[PASS] dense FSDP mesh validation and seeded weights")


def main():
    config = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=512,
        intermediate_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    cfg = PretrainLMCfg()
    cfg.training.model = "qwen3.5-test"
    cfg.training.sequence_length = 128
    cfg.training.global_batch_size = cfg.training.micro_batch_size = 2
    cfg.training.max_steps = 1
    cfg.training.lr = 1e-4
    cfg.training.optimizer = make_adamw_optimizer
    cfg.training.scheduler = make_constant_scheduler
    setup_distributed(cfg)
    setup_logging(cfg)
    with patch("pithtrain.modules.training.AutoConfig.from_pretrained", return_value=config):
        if distributed.dp_size == 1:
            check_unsharded(cfg)
        else:
            assert distributed.dp_size == 2
            check_sharded(cfg)


if __name__ == "__main__":
    main()
