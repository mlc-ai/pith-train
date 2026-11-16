"""
Benchmark GPT-OSS-20B at 4K tokens on four 8-GPU H200 nodes: pp 1, cp 1, ep 8, micro-batch 4.
"""

from pathlib import Path

from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.pipeline_parallel_size = 1
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 8

training = cfg.training
training.model = Path("examples/pretrain_lm/gpt-oss-20b")
training.optimizer = make_muon_optimizer
training.scheduler = make_constant_scheduler
training.lr = 3.0e-4
training.max_steps = 12
training.micro_batch_size = 4
training.global_batch_size = 2048
training.sequence_length = 4096
training.moe_load_balance_type = "micro-batch"
training.moe_load_balance_coef = 1e-3
training.fp8 = False
training.benchmark = True

if __name__ == "__main__":
    launch(cfg)
