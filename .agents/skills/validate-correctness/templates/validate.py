from functools import partial
from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg
from pithtrain.modules.training import make_adamw_optimizer, make_wsd_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.pipeline_parallel_size = <pipeline-parallel-size>
distributed.expert_parallel_size = <expert-parallel-size>
distributed.context_parallel_size = <context-parallel-size>

training = cfg.training
training.model = Path("examples/pretrain_lm/<model>/config.json")
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/<tokenizer>")
training.optimizer = make_adamw_optimizer
training.moe_load_balance_type = "<moe-load-balance-type>"
training.moe_load_balance_coef = 1e-3
training.micro_batch_size = 1
training.sequence_length = <sequence-length>
training.global_batch_size = <global-batch-size>
training.fp8 = False
training.max_steps = 32
training.scheduler = partial(make_wsd_scheduler, start_lr=1e-6, warmup_ratio=1.0, decay_ratio=0.0)
training.lr = 1e-5
training.benchmark = False

wandb_cfg = LoggingWandbCfg()
wandb_cfg.entity = "<wandb-entity>"
wandb_cfg.project = "<wandb-project>"
wandb_cfg.name = Path(__file__).stem
cfg.logging.wandb = wandb_cfg

if __name__ == "__main__":
    launch(cfg)
