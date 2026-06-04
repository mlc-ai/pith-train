"""Run training steps from a released checkpoint for correctness validation."""

import argparse
from pathlib import Path

from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

MODELS = {
    "deepseek-v2-lite": {
        "config": "examples/pretrain_lm/deepseek-v2-lite/config.json",
        "dataset": "workspace/datasets/dclm-baseline/toktxt/deepseek-v2",
        "save_location": "workspace/checkpoints/deepseek-v2-lite",
        "moe_load_balance_type": "sequence",
        "moe_load_balance_coef": 3e-3,
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
parser.add_argument("--max-steps", type=int, default=25)
parsed = parser.parse_args()

specs = MODELS[parsed.model]

cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.pipeline_parallel_size = parsed.pipeline_parallel_size
distributed.expert_parallel_size = parsed.expert_parallel_size
distributed.context_parallel_size = parsed.context_parallel_size

training = cfg.training
training.model = Path(specs["config"])
training.optimizer = "Adam"
training.scheduler = "Constant"
training.max_lr = 1e-6
training.min_lr = 1e-6
training.warmup_steps = 0
training.max_steps = parsed.max_steps
training.micro_batch_size = 1
training.global_batch_size = 1024
training.sequence_length = parsed.sequence_length
training.dataset = Path(specs["dataset"])
training.moe_load_balance_type = specs["moe_load_balance_type"]
training.moe_load_balance_coef = specs["moe_load_balance_coef"]
training.fp8_training = "disabled"
training.save_location = Path(specs["save_location"])

if __name__ == "__main__":
    launch(cfg)
