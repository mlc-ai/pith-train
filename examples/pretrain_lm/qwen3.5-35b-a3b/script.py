"""Pretrain Qwen3.5-35B-A3B at 4K context on one 8-GPU H200 node: pp 1, ep 8, micro-batch 1."""

from functools import partial
from pathlib import Path

from pithtrain.modules.training import make_muon_optimizer, make_wsd_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

cfg.dataset = Path("workspace/datasets/dclm-baseline/toktxt/qwen3.5")

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 1
distributed.expert_parallel_size = 8

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-35b-a3b")
training.optimizer = make_muon_optimizer
kwargs = dict(start_lr=1.0e-5, warmup_ratio=0.03, final_lr=1.0e-5)
training.scheduler = partial(make_wsd_scheduler, **kwargs)
training.lr = 3.0e-4
training.max_steps = 4096
training.micro_batch_size = 1
training.global_batch_size = 1024
training.sequence_length = 4096
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8 = False
training.save_interval = 256
training.save_location = Path("workspace/checkpoints/qwen3.5-35b-a3b")

if __name__ == "__main__":
    launch(cfg)
