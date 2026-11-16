"""
Benchmark GPT-OSS-120B at 4K tokens on four 8-GPU H200 nodes: pp 2, cp 1, ep 16, micro-batch 2.
"""

from pathlib import Path

from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()
cfg.dataset = Path("/synthetic")

distributed = cfg.distributed
distributed.pipeline_parallel_size = 2
distributed.context_parallel_size = 1
distributed.expert_parallel_size = 16

training = cfg.training
training.model = Path("examples/pretrain_lm/gpt-oss-120b")
training.optimizer = make_muon_optimizer
training.scheduler = make_constant_scheduler
training.lr = 3.0e-4
training.max_steps = 12
training.micro_batch_size = 2
training.global_batch_size = 2048
training.sequence_length = 4096
training.moe_load_balance_type = "micro-batch"
training.moe_load_balance_coef = 1e-3
training.fp8 = False
training.benchmark = True

if __name__ == "__main__":
    launch(cfg)
