"""Train from a prepared Omni bundle through the normal pretrain_lm task."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared bundle root")
    parser.add_argument("--model", type=Path, required=True, help="Native model config directory")
    parser.add_argument("--stage", choices=("text", "image", "audio", "video"), default="text")
    parser.add_argument(
        "--sampling-weights", type=json.loads, help='JSON object, e.g. {"text":1,"image":1}'
    )
    parser.add_argument(
        "--epoch-samples", type=int, help="Media draws per epoch; multiple of global batch"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    if (
        min(
            args.sequence_length,
            args.global_batch_size,
            args.micro_batch_size,
            args.steps,
            args.save_interval,
            args.pp,
            args.cp,
            args.ep,
        )
        <= 0
    ):
        parser.error("Lengths, batch sizes, steps and parallel degrees must be positive")
    if args.sampling_weights is not None and not isinstance(args.sampling_weights, dict):
        parser.error("--sampling-weights must be a JSON object")

    # Keep --help usable on a machine without the CUDA training dependencies.
    from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
    from pithtrain.modules.training_data import OmniDataCfg
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

    cfg = PretrainLMCfg()
    cfg.dataset = args.dataset
    cfg.omni_data = OmniDataCfg()
    cfg.omni_data.stage = args.stage
    cfg.omni_data.sampling_weights = args.sampling_weights
    cfg.omni_data.epoch_samples = args.epoch_samples
    cfg.omni_data.num_workers = args.num_workers
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = args.cp
    cfg.distributed.expert_parallel_size = args.ep
    training = cfg.training
    training.model = args.model
    training.optimizer = make_adamw_optimizer
    training.scheduler = make_constant_scheduler
    training.lr = args.lr
    training.seed = args.seed
    training.max_steps = args.steps
    training.micro_batch_size = args.micro_batch_size
    training.global_batch_size = args.global_batch_size
    training.sequence_length = args.sequence_length
    training.fp8 = False
    training.moe_load_balance_coef = 0.0
    training.save_interval = args.save_interval
    training.save_location = args.checkpoint
    launch(cfg)


if __name__ == "__main__":
    main()
