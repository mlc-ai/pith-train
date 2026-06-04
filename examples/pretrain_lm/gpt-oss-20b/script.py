from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg  # noqa: F401
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

cfg.distributed.context_parallel_size = 1
cfg.distributed.pipeline_parallel_size = 1
cfg.distributed.expert_parallel_size = 8

cfg.training.model = Path("examples/pretrain_lm/gpt-oss-20b/config.json")
cfg.training.optimizer = "Adam"
cfg.training.scheduler = "CosineAnnealing"
cfg.training.max_lr = 3.0e-4
cfg.training.min_lr = 1.0e-5
cfg.training.warmup_steps = 128
cfg.training.max_steps = 4096
cfg.training.micro_batch_size = 1
cfg.training.global_batch_size = 1024
cfg.training.sequence_length = 2048
cfg.training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/gpt-oss")
cfg.training.moe_load_balance_type = "global-batch"
cfg.training.moe_load_balance_coef = 1e-3
cfg.training.save_interval = 256
cfg.training.save_location = Path("workspace/checkpoints/gpt-oss-20b")

# cfg.logging.wandb = LoggingWandbCfg()
# cfg.logging.wandb.entity = "your-entity"
# cfg.logging.wandb.project = "gpt-oss-20b"
# cfg.logging.wandb.name = "gpt-oss-20b-pretrain"

if __name__ == "__main__":
    launch(cfg)
