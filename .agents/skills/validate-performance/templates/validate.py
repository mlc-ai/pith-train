from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg
from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

# One identical runner supports the historical base and the current config API.
data_cfg = cfg.data if hasattr(cfg, "data") else cfg
data_cfg.dataset = Path("workspace/datasets/dclm-baseline/toktxt/<tokenizer>")

distributed = cfg.distributed
distributed.pipeline_parallel_size = <pipeline-parallel-size>
distributed.expert_parallel_size = <expert-parallel-size>
distributed.context_parallel_size = <context-parallel-size>

training = cfg.training
training.model = Path("examples/pretrain_lm/<model>")
training.optimizer = make_adamw_optimizer
training.scheduler = make_constant_scheduler
training.lr = 1e-6
training.moe_load_balance_type = "<moe-load-balance-type>"
training.moe_load_balance_coef = 1e-3
training.micro_batch_size = 1
training.sequence_length = 2048
training.global_batch_size = <global-batch-size>
training.fp8 = False
training.max_steps = 8
training.benchmark = True

wandb_cfg = LoggingWandbCfg()
wandb_cfg.entity = "PithTrain"
wandb_cfg.project = "<wandb-project>"
wandb_cfg.group = "performance/<model>"
wandb_cfg.name = Path(__file__).stem
cfg.logging.wandb = wandb_cfg

if __name__ == "__main__":
    launch(cfg)
