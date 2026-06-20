"""Run 5 warmup steps + 1 profiled step under nsys."""

import argparse
import os
from pathlib import Path

from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

MODELS = {
    "deepseek-v2-lite": {
        "config": "examples/pretrain_lm/deepseek-v2-lite/config.json",
        "dataset": "workspace/datasets/dclm-baseline/toktxt/deepseek-v2",
        "save_location": "workspace/checkpoints/deepseek-v2-lite",
        "moe_load_balance_type": "sequence",
        "moe_load_balance_coef": 1e-3,
    },
    "qwen3-30b-a3b": {
        "config": "examples/pretrain_lm/qwen3-30b-a3b/config.json",
        "dataset": "workspace/datasets/dclm-baseline/toktxt/qwen3",
        "save_location": "workspace/checkpoints/qwen3-30b-a3b",
        "moe_load_balance_type": "global-batch",
        "moe_load_balance_coef": 1e-3,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODELS))
parser.add_argument("--pipeline-parallel-size", type=int, required=True)
parser.add_argument("--expert-parallel-size", type=int, required=True)
parser.add_argument("--context-parallel-size", type=int, default=1)
parser.add_argument("--sequence-length", type=int, default=2048)
parsed = parser.parse_args()

specs = MODELS[parsed.model]
pp_size = parsed.pipeline_parallel_size
ep_size = parsed.expert_parallel_size
cp_size = parsed.context_parallel_size

# 32 microbatches per stage is more than enough for the pipeline to reach steady state.
# CP splits sequence not batch, so it does not multiply global_batch_size.
dp_size = int(os.environ["WORLD_SIZE"]) // (pp_size * cp_size * ep_size)
global_batch_size = 32 * dp_size * ep_size

cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.pipeline_parallel_size = pp_size
distributed.expert_parallel_size = ep_size
distributed.context_parallel_size = cp_size

training = cfg.training
training.model = Path(specs["config"])
training.optimizer = make_adamw_optimizer
training.scheduler = make_constant_scheduler
training.lr = 1e-6
training.max_steps = 6
training.micro_batch_size = 1
training.global_batch_size = global_batch_size
training.sequence_length = parsed.sequence_length
training.dataset = Path(specs["dataset"])
training.moe_load_balance_type = specs["moe_load_balance_type"]
training.moe_load_balance_coef = specs["moe_load_balance_coef"]
training.fp8_training = "disabled"
training.save_location = Path(specs["save_location"])
training.nsys_start = 5
training.nsys_stop = 6

if __name__ == "__main__":
    launch(cfg)
