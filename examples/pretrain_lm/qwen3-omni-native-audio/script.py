"""Train native Qwen3-Omni audio/text using an existing prepared audio bundle."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Composite Omni Thinker config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epoch-samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)
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
        parser.error("Sizes, step counts, parallel degrees and learning rate must be positive")
    if args.epoch_samples % args.global_batch_size:
        parser.error("--epoch-samples must be a multiple of --global-batch-size")
    # Keep --help usable without optional CUDA operator dependencies.
    from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

    cfg = PretrainLMCfg()
    cfg.data.dataset = args.dataset
    cfg.data.format = "prepared_bundle"
    cfg.data.modalities = ("audio",)
    cfg.data.epoch_samples = args.epoch_samples
    cfg.data.num_workers = 0
    cfg.data.validate()
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.expert_parallel_size = args.ep
    training = cfg.training
    training.model = args.model
    training.native_audio = True
    training.optimizer = make_adamw_optimizer
    training.scheduler = make_constant_scheduler
    training.lr = args.lr
    training.seed = args.seed
    training.max_steps = args.steps
    training.micro_batch_size = 1
    training.global_batch_size = args.global_batch_size
    training.sequence_length = args.sequence_length
    training.fp8 = False
    training.moe_load_balance_coef = 0.0
    training.save_interval = args.save_interval
    training.save_location = args.checkpoint
    launch(cfg)


if __name__ == "__main__":
    main()
