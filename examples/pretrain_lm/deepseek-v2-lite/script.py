"""Pretrain DeepSeek-V2-Lite on a single 8-GPU node with 8-way expert parallelism."""

from functools import partial
from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg  # noqa: F401
from pithtrain.modules.training import make_muon_optimizer, make_wsd_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 1
distributed.expert_parallel_size = 8

training = cfg.training
training.model = Path("examples/pretrain_lm/deepseek-v2-lite/config.json")
training.optimizer = make_muon_optimizer
kwargs = dict(start_lr=1.0e-5, warmup_ratio=0.03, final_lr=1.0e-5)
training.scheduler = partial(make_wsd_scheduler, **kwargs)
training.lr = 4.2e-4
training.max_steps = 4096
training.micro_batch_size = 1
training.global_batch_size = 1024
training.sequence_length = 2048
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")
training.moe_load_balance_type = "sequence"
training.moe_load_balance_coef = 3e-3
training.fp8_training = "disabled"
training.save_interval = 256
training.save_location = Path("workspace/checkpoints/deepseek-v2-lite")

# Wandb logging configuration. Comment out to disable.
logging = cfg.logging
logging.wandb = LoggingWandbCfg()
logging.wandb.entity = ""  # your wandb entity
logging.wandb.project = ""  # your wandb project
logging.wandb.name = "deepseek-v2-lite"

if __name__ == "__main__":
    launch(cfg)
