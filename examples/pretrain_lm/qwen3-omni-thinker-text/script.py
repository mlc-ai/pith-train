"""Pretrain a selected native Thinker text config on tokenizer-compatible .bin data."""

import argparse
from functools import partial
from pathlib import Path

from pithtrain.modules.training import make_adamw_optimizer, make_wsd_scheduler
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Tiny/full config.json or its directory"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Omni bundle's tokens/train directory"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--checkpoints", type=Path)
    args = parser.parse_args()
    cfg = PretrainLMCfg()
    cfg.dataset = args.dataset
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.expert_parallel_size = args.ep
    cfg.distributed.context_parallel_size = args.cp
    t = cfg.training
    t.model = args.model
    t.optimizer = make_adamw_optimizer
    t.scheduler = partial(make_wsd_scheduler, start_lr=1e-6, warmup_ratio=0.25, decay_ratio=0)
    t.lr, t.max_steps, t.sequence_length = 1e-5, args.steps, args.sequence_length
    t.micro_batch_size, t.global_batch_size = 1, args.global_batch_size
    t.moe_load_balance_type, t.moe_load_balance_coef = "global-batch", 1e-3
    t.fp8 = False
    if args.checkpoints is not None:
        t.save_location, t.save_interval = args.checkpoints, 2
    launch(cfg)


if __name__ == "__main__":
    main()
