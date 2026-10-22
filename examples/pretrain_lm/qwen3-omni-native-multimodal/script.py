"""Train a native Qwen3-Omni Thinker on one checkpointable modality mixture."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Full composite Thinker config")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Output / strict resume root"
    )
    parser.add_argument(
        "--initialize-from", type=Path, help="Canonical DCP root: load model weights into a new run"
    )
    parser.add_argument(
        "--initialize-step", type=int, default=0, help="Source DCP step (default: 0)"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=("text", "image", "audio", "video"),
        default=["text", "image", "audio", "video"],
    )
    parser.add_argument("--sampling-weights", type=json.loads)
    parser.add_argument("--epoch-samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    if (
        min(
            args.epoch_samples,
            args.sequence_length,
            args.global_batch_size,
            args.steps,
            args.save_interval,
            args.pp,
            args.ep,
        )
        <= 0
        or args.lr <= 0
    ):
        parser.error("Sizes, steps, parallelism and learning rate must be positive")
    if args.initialize_step < 0 or (args.initialize_from is None and args.initialize_step != 0):
        parser.error("--initialize-step must be nonnegative and requires --initialize-from")
    if args.epoch_samples % args.global_batch_size:
        parser.error("--epoch-samples must be a multiple of --global-batch-size")
    from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

    cfg = PretrainLMCfg()
    cfg.initialize_from, cfg.initialize_step = args.initialize_from, args.initialize_step
    cfg.data.dataset, cfg.data.format = args.dataset, "prepared_bundle"
    cfg.data.modalities = tuple(args.modalities)
    cfg.data.sampling_weights, cfg.data.epoch_samples = args.sampling_weights, args.epoch_samples
    cfg.data.num_workers = 0
    cfg.data.validate()
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.expert_parallel_size = args.ep
    t = cfg.training
    t.model, t.native_multimodal = args.model, True
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.lr, t.seed, t.max_steps = args.lr, args.seed, args.steps
    t.micro_batch_size, t.global_batch_size, t.sequence_length = (
        1,
        args.global_batch_size,
        args.sequence_length,
    )
    t.fp8, t.moe_load_balance_coef = False, 0.0
    t.save_interval, t.save_location, t.strict_resume = args.save_interval, args.checkpoint, True
    launch(cfg)


if __name__ == "__main__":
    main()
