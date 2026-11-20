"""Pretrain GLM-4.7-Flash at 4K context on one 8-GPU H200 node: pp 2, ep 4, micro-batch 2."""

from functools import partial
from pathlib import Path

from pithtrain.modules.training import make_muon_optimizer, make_wsd_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

cfg.dataset = Path("workspace/datasets/dclm-baseline/toktxt/glm-4.7")

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 2
distributed.expert_parallel_size = 4

training = cfg.training
training.model = Path("examples/pretrain_lm/glm-4.7-flash")
training.optimizer = make_muon_optimizer
kwargs = dict(start_lr=1.0e-5, warmup_ratio=0.03, final_lr=1.0e-5)
training.scheduler = partial(make_wsd_scheduler, **kwargs)
training.lr = 3.0e-4
training.max_steps = 4096
training.micro_batch_size = 2
training.global_batch_size = 1024
training.sequence_length = 4096
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8 = False
training.save_interval = 256
training.save_location = Path("workspace/checkpoints/glm-4.7-flash")

if __name__ == "__main__":
    launch(cfg)
