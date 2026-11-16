"""Benchmark DeepSeek-V2-Lite at 4K tokens on one 8-GPU H200 node: pp 2, cp 1, ep 4, micro-batch 4."""

from pathlib import Path

from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.pipeline_parallel_size = 2
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 4

training = cfg.training
training.model = Path("examples/pretrain_lm/deepseek-v2-lite")
training.optimizer = make_muon_optimizer
training.scheduler = make_constant_scheduler
training.lr = 3.0e-4
training.max_steps = 12
training.micro_batch_size = 4
training.global_batch_size = 512
training.sequence_length = 4096
training.moe_load_balance_type = "sequence"
training.moe_load_balance_coef = 3e-3
training.fp8 = False
training.benchmark = True

if __name__ == "__main__":
    launch(cfg)
