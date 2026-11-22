"""Benchmark the Qwen3.5 dense FSDP2 pathway on H200 GPUs."""

import argparse
from functools import partial
from pathlib import Path

from pithtrain.modules.dense_fsdp import DenseFSDPCfg
from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch


def reshard(value: str) -> bool | int:
    if value == "true":
        return True
    if value == "false":
        return False
    return int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--reshard", type=reshard, default=True)
    parser.add_argument("--prefetch", type=int, default=0)
    parser.add_argument("--reduce-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--checkpoint", choices=("none", "selective", "full"), default="none")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--unfused-adamw", action="store_true")
    args = parser.parse_args()

    cfg = PretrainLMCfg()
    train = cfg.training
    train.model = args.model
    train.hf_import_path = args.weights
    train.benchmark = True
    train.sequence_length = args.seq
    train.global_batch_size = args.batch
    train.micro_batch_size = args.micro_batch
    train.max_steps = args.steps
    train.lr = 1e-5
    train.optimizer = partial(make_adamw_optimizer, fused=not args.unfused_adamw)
    train.scheduler = make_constant_scheduler
    train.dense_fsdp = DenseFSDPCfg()
    train.dense_fsdp.reshard_after_forward = args.reshard
    train.dense_fsdp.prefetch = args.prefetch
    train.dense_fsdp.reduce_dtype = args.reduce_dtype
    train.dense_fsdp.activation_checkpoint = args.checkpoint
    train.dense_fsdp.compile_blocks = args.compile_blocks
    launch(cfg)
