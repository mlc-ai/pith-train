"""Pretrain Qwen3.5-35B-A3B across two 8-GPU H200/B200 nodes (pp=2, ep=8).

Qwen3.5-35B-A3B is a hybrid Gated-DeltaNet + full-attention MoE. It is far too
large for a single node, so this uses pipeline parallelism (pp=2) across the
two nodes on top of 8-way expert parallelism. Context parallelism is not yet
supported for this model (linear attention).
"""

from functools import partial
from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg  # noqa: F401
from pithtrain.modules.training import make_muon_optimizer, make_wsd_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

# Two 16-GPU nodes: pp=2 across nodes, ep=8 within. dp is inferred (=1).
distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 2
distributed.expert_parallel_size = 8

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-35b-a3b/config.json")
training.optimizer = make_muon_optimizer
kwargs = dict(start_lr=1.0e-5, warmup_ratio=0.03, final_lr=1.0e-5)
training.scheduler = partial(make_wsd_scheduler, **kwargs)
training.lr = 3.0e-4
training.max_steps = 4096
training.micro_batch_size = 1
training.global_batch_size = 1024
training.sequence_length = 2048
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/qwen3.5")
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8_training = "disabled"
training.save_interval = 256
training.save_location = Path("workspace/checkpoints/qwen3.5-35b-a3b")

# Wandb logging configuration. Comment out to disable.
logging = cfg.logging
logging.wandb = LoggingWandbCfg()
logging.wandb.entity = ""  # your wandb entity
logging.wandb.project = ""  # your wandb project
logging.wandb.name = "qwen3.5-35b-a3b"

if __name__ == "__main__":
    launch(cfg)
