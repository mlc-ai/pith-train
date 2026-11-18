"""
Benchmark Qwen3.5-35B-A3B at 32K tokens on one 8-GPU H200 node: pp 1, cp 8, ep 8.
"""

from pathlib import Path

from pithtrain.modules.training import make_constant_scheduler, make_muon_optimizer
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()
cfg.dataset = Path("/synthetic")

distributed = cfg.distributed
distributed.pipeline_parallel_size = 1
distributed.context_parallel_size = 8
distributed.expert_parallel_size = 8

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-35b-a3b")
training.optimizer = make_muon_optimizer
training.scheduler = make_constant_scheduler
training.lr = 3.0e-4
training.max_steps = 12
training.micro_batch_size = 1
training.global_batch_size = 64
training.sequence_length = 32768
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8 = False
training.benchmark = True

if __name__ == "__main__":
    launch(cfg)
