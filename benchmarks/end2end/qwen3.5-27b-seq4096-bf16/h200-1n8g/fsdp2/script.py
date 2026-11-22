"""Benchmark the Qwen3.5 dense FSDP2 pathway on H200 GPUs."""

import argparse
import os
from functools import partial
from pathlib import Path

import torch


def reshard(value: str) -> bool | int:
    if value == "true":
        return True
    if value == "false":
        return False
    return int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("examples/pretrain_lm/qwen3.5-27b"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", type=Path, help="Tokenized corpus directory for loss checks")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--blocks-per-group", type=int, default=1)
    parser.add_argument("--group-by-attention-type", action="store_true")
    parser.add_argument("--edge-sharding", choices=("separate", "root"), default="separate")
    parser.add_argument("--reshard", type=reshard, default=True)
    parser.add_argument("--keep-last-groups", type=int, default=0)
    parser.add_argument("--keep-last-group", dest="keep_last_groups", action="store_const", const=1)
    parser.add_argument("--keep-gathered-across-microbatches", action="store_true")
    parser.add_argument("--skip-embedding-backward-gather", action="store_true")
    parser.add_argument("--prefetch", type=int, default=0)
    parser.add_argument("--prefetch-forward-copy-out", action="store_true")
    parser.add_argument("--reduce-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--bf16-sharded-gradients", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--reduce-at-step-end",
        type=int,
        nargs="?",
        const=True,
        default=False,
        help="Defer reduce-scatter for all parameters or the first N decoder buckets",
    )
    parser.add_argument("--overlap-microbatch-reductions", action="store_true")
    parser.add_argument("--prefetch-data-during-optimizer", action="store_true")
    parser.add_argument("--reduce-scatter-buffers", type=int, default=1)
    parser.add_argument("--separate-reduce-scatter", action="store_true")
    parser.add_argument(
        "--collective-buffers", choices=("default", "process-group", "symmetric"), default="default"
    )
    parser.add_argument(
        "--checkpoint", choices=("none", "selective", "full", "offload"), default="none"
    )
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead"),
        default="default",
    )
    parser.add_argument("--nsys-start", type=int)
    parser.add_argument("--nsys-stop", type=int)
    parser.add_argument("--unfused-adamw", action="store_true")
    args = parser.parse_args()

    # PithTrain's kernel imports query the current GPU.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from pithtrain.modules.dense_fsdp import DenseFSDPCfg
    from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

    cfg = PretrainLMCfg()
    cfg.dataset = args.dataset if args.dataset is not None else Path(".")
    train = cfg.training
    train.model = args.model
    train.hf_import_path = args.weights
    train.benchmark = args.dataset is None
    train.sequence_length = args.seq
    train.global_batch_size = args.batch
    train.micro_batch_size = args.micro_batch
    train.max_steps = args.steps
    train.nsys_start = args.nsys_start
    train.nsys_stop = args.nsys_stop
    train.lr = 1e-5
    train.optimizer = partial(make_adamw_optimizer, fused=not args.unfused_adamw)
    train.scheduler = make_constant_scheduler
    train.dense_fsdp = DenseFSDPCfg()
    train.dense_fsdp.blocks_per_group = args.blocks_per_group
    train.dense_fsdp.group_by_attention_type = args.group_by_attention_type
    train.dense_fsdp.edge_sharding = args.edge_sharding
    train.dense_fsdp.reshard_after_forward = args.reshard
    train.dense_fsdp.keep_last_groups = args.keep_last_groups
    train.dense_fsdp.keep_gathered_across_microbatches = args.keep_gathered_across_microbatches
    train.dense_fsdp.skip_embedding_backward_gather = args.skip_embedding_backward_gather
    train.dense_fsdp.prefetch = args.prefetch
    train.dense_fsdp.prefetch_forward_copy_out = args.prefetch_forward_copy_out
    train.dense_fsdp.reduce_dtype = args.reduce_dtype
    train.dense_fsdp.bf16_sharded_gradients = args.bf16_sharded_gradients
    train.dense_fsdp.reduce_at_step_end = args.reduce_at_step_end
    train.dense_fsdp.overlap_microbatch_reductions = args.overlap_microbatch_reductions
    train.dense_fsdp.prefetch_data_during_optimizer = args.prefetch_data_during_optimizer
    train.dense_fsdp.reduce_scatter_buffers = args.reduce_scatter_buffers
    train.dense_fsdp.separate_reduce_scatter = args.separate_reduce_scatter
    train.dense_fsdp.collective_buffers = args.collective_buffers
    train.dense_fsdp.activation_checkpoint = args.checkpoint
    train.dense_fsdp.compile_blocks = args.compile_blocks
    train.dense_fsdp.compile_mode = args.compile_mode
    launch(cfg)
