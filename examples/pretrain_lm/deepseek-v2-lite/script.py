"""Pretrain DeepSeek-V2-Lite at 4K context on one 8-GPU H200 node: pp 4, ep 1, micro-batch 4."""

from functools import partial
from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg
from pithtrain.modules.training import make_muon_optimizer, make_wsd_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

cfg.dataset = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 4
distributed.expert_parallel_size = 1

training = cfg.training
training.model = Path("examples/pretrain_lm/deepseek-v2-lite")
training.optimizer = make_muon_optimizer
kwargs = dict(start_lr=1.0e-4, warmup_ratio=0.00, final_lr=1.0e-4)
training.scheduler = partial(make_wsd_scheduler, **kwargs)
training.lr = 1.0e-4

training.max_steps = 64
training.micro_batch_size = 4
training.global_batch_size = 1024
training.sequence_length = 4096
training.moe_load_balance_type = "sequence"
training.moe_load_balance_coef = 3e-3
training.fp8 = True

logging = cfg.logging
logging.wandb = LoggingWandbCfg()
logging.wandb.entity = "pithtrain"
logging.wandb.project = "hopper-deepgemm"
logging.wandb.name = "dsv2-fp8"

if __name__ == "__main__":
    launch(cfg)
