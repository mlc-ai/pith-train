"""Pretrain Qwen3-30B-A3B on a single 8-GPU H200/B200 node with 8-way expert parallelism."""

from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg  # noqa: F401
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

# Fits a single 8-GPU H200/B200 node. On 8xH100 (80GB) this does not fit; use
# two nodes with pp=2, ep=8 (16 GPUs).
distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 1
distributed.expert_parallel_size = 8

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3-30b-a3b/config.json")
training.optimizer = "Adam"
training.scheduler = "CosineAnnealing"
training.max_lr = 3.0e-4
training.min_lr = 1.0e-5
training.warmup_steps = 128
training.max_steps = 4096
training.micro_batch_size = 1
training.global_batch_size = 1024
training.sequence_length = 2048
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/qwen3")
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8_training = "disabled"
training.save_interval = 256
training.save_location = Path("workspace/checkpoints/qwen3-30b-a3b")

# Wandb logging configuration. Comment out to disable.
logging = cfg.logging
logging.wandb = LoggingWandbCfg()
logging.wandb.entity = ""  # your wandb entity
logging.wandb.project = ""  # your wandb project
logging.wandb.name = "qwen3-30b-a3b"

if __name__ == "__main__":
    launch(cfg)
