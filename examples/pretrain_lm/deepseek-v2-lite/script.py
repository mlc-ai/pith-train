"""Pretrain DeepSeek-V2-Lite on a single 8-GPU node with 8-way expert parallelism."""

from functools import partial
from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg
from pithtrain.modules.training import make_muon_optimizer, make_wsd_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

cfg.data.dataset = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 2
distributed.expert_parallel_size = 2

training = cfg.training
training.model = Path("examples/pretrain_lm/deepseek-v2-lite")
training.optimizer = make_muon_optimizer
kwargs = dict(start_lr=1.0e-4, warmup_ratio=0.00, final_lr=1.0e-4)
training.scheduler = partial(make_wsd_scheduler, **kwargs)
training.lr = 1.0e-4

training.max_steps = 64
training.micro_batch_size = 1
training.global_batch_size = 1024
training.sequence_length = 2048
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
