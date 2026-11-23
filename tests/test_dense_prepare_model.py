"""
Check dense setup without FSDP on one GPU and HSDP gradient clipping on four GPUs.

    torchrun --nproc-per-node=1 tests/test_dense_prepare_model.py
    torchrun --nproc-per-node=4 tests/test_dense_prepare_model.py
"""

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
from pithtrain.modules.optimizer import clip_grad_norm, local_grads
from pithtrain.modules.training import (
    make_adamw_optimizer,
    make_constant_scheduler,
    setup_training,
)
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, train_step


def check_meta(model):
    assert all(param.is_meta for param in model.parameters())


def check_unsharded(cfg):
    setup_training(cfg, prepare_model=check_meta)
    model = training.model
    assert not isinstance(model, FSDPModule)
    assert training.dense_grad_norm_group is None
    assert model.lm_head.main_grad is None
    assert all(not isinstance(param, DTensor) for param in model.parameters())

    dataset = SyntheticDataset(model.embed_tokens.num_embeddings, cfg.training.sequence_length, 42)
    initial = model.lm_head.weight.detach().clone()
    train_step(cfg, dataset, 0)
    assert not torch.equal(model.lm_head.weight, initial)
    print("[PASS] unsharded dense training")


def check_hsdp(cfg):
    replicas = 2
    with pytest.raises(ValueError, match="gradients will not be reduced across the 4"):
        setup_training(cfg, prepare_model=check_meta)

    def prepare_model(model):
        check_meta(model)
        dp_mesh = distributed.attn_mesh["dp"]
        mesh = DeviceMesh(
            dp_mesh.device_type,
            dp_mesh.mesh.reshape(replicas, -1),
            mesh_dim_names=("replica", "shard"),
        )
        fully_shard(model, mesh=mesh)

    setup_training(cfg, prepare_model=prepare_model)
    model = training.model
    group = training.dense_grad_norm_group
    assert group.size() == 2
    for param in model.parameters():
        param.grad = torch.full_like(param, group.rank() + 1)
    # The old norm summed over the world and divided by the replica count.
    power = torch.nn.utils.get_total_norm(local_grads(model)) ** 2.0
    torch.distributed.all_reduce(power)
    expected = ((power / replicas) ** 0.5).clamp(min=1e-6)
    actual = clip_grad_norm(model, max_norm=1.0, group=group)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for grad in local_grads(model):
        torch.testing.assert_close(grad, torch.full_like(grad, (group.rank() + 1) / actual))
    if distributed.rank == 0:
        print(f"[PASS] HSDP clipping matches legacy norm {actual.item().hex()} bitwise")


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
            assert distributed.dp_size == 4
            check_hsdp(cfg)


if __name__ == "__main__":
    main()
