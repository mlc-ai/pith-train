"""Native CP1/HSDP1 Qwen3-Omni GRPO with one CPU/Gloo-connected HF worker."""

import argparse
import json
import math
import os
from datetime import timedelta
from functools import partial
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Local directory containing config.json"
    )
    parser.add_argument(
        "--reference-hf",
        type=Path,
        required=True,
        help="Existing local HF safetensors; never downloaded",
    )
    parser.add_argument(
        "--hf-prefix",
        default="thinker.",
        help="Explicit HF component prefix; use empty string for standalone",
    )
    parser.add_argument("--scope", choices=("text", "thinker"), required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-request", type=int)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Read a prior checkpoint without writing new checkpoints (e.g. exact replay verification)",
    )
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--hf-device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--hf-compute-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--worker-timeout", type=float, default=120)
    parser.add_argument("--weight-limit-mib", type=int, default=512)
    parser.add_argument("--host-weight-limit-mib", type=int)
    parser.add_argument("--device-weight-limit-mib", type=int)
    parser.add_argument("--media-options", type=json.loads, default={})
    parser.add_argument("--reward-casefold", action="store_true")
    parser.add_argument("--reward-no-strip", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Validate local metadata and report limits; no models/GPU/processes",
    )
    args = parser.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.pp < 1 or args.ep < 1 or world % args.pp or (world // args.pp) % args.ep:
        parser.error("PP and EP must divide the native CP1 stage topology")
    if (
        args.requests < 1
        or not 1 <= args.updates <= args.requests
        or args.group_size < 2 * args.pp
        or args.group_size % 2
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        parser.error("Positive finite update limits and an even complete group >=2 required")
    if args.resume_request is not None and not 0 <= args.resume_request < args.requests:
        parser.error("Resume request must precede the configured request limit")
    return args


def main(argv=None, *, observer=None):
    args = parse_args(argv)

    from pithtrain.modules.omni_hf_transport import (
        DistributedHFThinkerRollout,
        _leader,
        initialize_native_policy,
        publish_native_policy,
    )
    from pithtrain.modules.omni_rollout_data import prepare_prompts
    from pithtrain.tasks.omni_hf_rollout import (
        ExactAnswerReward,
        LocalHFRolloutCfg,
        _config,
        _converter,
        configuration_identity,
        weight_memory_plan,
    )
    from pithtrain.tasks.omni_rl import OmniRLCfg, OmniRLTrainer, PithTrainPolicyAdapter

    raw = json.loads((args.model / "config.json").read_text())
    config = _config(raw, args.scope)
    cfg = LocalHFRolloutCfg(
        scope=args.scope,
        device=args.hf_device,
        compute_dtype=args.hf_compute_dtype,
        max_requests=args.requests,
        group_size=args.group_size,
        max_sequence_length=args.sequence_length,
        max_new_tokens=args.new_tokens,
        seed=args.seed,
        timeout_seconds=args.worker_timeout,
        max_weight_bytes=args.weight_limit_mib * (1 << 20),
        max_host_weight_bytes=None
        if args.host_weight_limit_mib is None
        else args.host_weight_limit_mib * (1 << 20),
        max_device_weight_bytes=None
        if args.device_weight_limit_mib is None
        else args.device_weight_limit_mib * (1 << 20),
    )
    cfg.validate()
    memory_plan = weight_memory_plan(config, cfg)
    if args.plan:
        print(
            json.dumps(
                dict(
                    status="PLAN_ONLY",
                    world=int(os.environ.get("WORLD_SIZE", "1")),
                    pp=args.pp,
                    ep=args.ep,
                    cp=1,
                    hsdp=1,
                    scope=args.scope,
                    canonical_fp32_bytes=memory_plan["canonical_fp32_bytes"],
                    memory_plan=memory_plan,
                    hf_device=args.hf_device,
                    hf_fp32_model_copies=2,
                    parent_reference_copy=True,
                    transient_snapshot_conversion_copies=False,
                    native_training_memory_additional=True,
                    requests=args.requests,
                    steps=args.updates,
                    worker_deadline_seconds=args.worker_timeout,
                    checkpoint_boundary="completed requests only",
                    gpu_validation="NOT_RUN",
                ),
                indent=2,
            )
        )
        return
    prompts, tokenizer = prepare_prompts(
        args.prompts,
        args.processor,
        config,
        scope=args.scope,
        max_length=args.sequence_length - args.new_tokens,
        media_options=args.media_options,
    )
    dp_size = int(os.environ.get("WORLD_SIZE", "1")) // args.pp
    if len(prompts.prompts) < args.requests * dp_size:
        raise ValueError("Not enough real prompts for bounded requests; no implicit cycling")
    # Only this execution path imports native CUDA operators. --help and --plan
    # are safe on CPU and do not initialize CUDA.
    import torch.distributed as dist

    from pithtrain.contexts import distributed, training
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.logging import setup_stdout
    from pithtrain.modules.training import (
        make_adamw_optimizer,
        make_constant_scheduler,
        setup_training,
    )
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg

    train = PretrainLMCfg()
    train.distributed.timeout = timedelta(seconds=args.worker_timeout + 30)
    train.distributed.pipeline_parallel_size, train.distributed.expert_parallel_size = (
        args.pp,
        args.ep,
    )
    t = train.training
    t.model, t.native_multimodal = args.model, args.scope == "thinker"
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.lr, t.seed, t.max_steps = args.lr, args.seed, args.updates
    t.global_batch_size, t.micro_batch_size, t.sequence_length = (
        args.group_size * dp_size,
        1,
        args.sequence_length,
    )
    t.fp8, t.moe_load_balance_coef, t.strict_resume = False, 0.0, True
    t.save_location = args.checkpoint
    setup_stdout()
    setup_distributed(train)
    group = dist.new_group(backend="gloo", timeout=train.distributed.timeout)
    reference = None

    def read_reference():
        nonlocal reference
        source_config = _config(
            json.loads((args.reference_hf / "config.json").read_text()), args.scope
        )
        if configuration_identity(source_config) != configuration_identity(config):
            raise ValueError("Reference/model configuration semantics differ")
        converter = _converter(args.scope)
        if args.scope == "text":
            hf, _, _ = converter.read_hf_text(
                args.reference_hf, config, prefix=args.hf_prefix, text_only=True
            )
        else:
            hf, _, _ = converter.read_hf(
                args.reference_hf, config, prefix=args.hf_prefix, thinker_only=True
            )
        reference = converter.hf_to_canonical(hf, config, prefix=args.hf_prefix)[0]
        reference = {name: value.float().contiguous() for name, value in reference.items()}
        return True

    _leader(read_reference, group, 1 << 20)
    reward = ExactAnswerReward(
        prompts,
        tokenizer,
        strip=not args.reward_no_strip,
        casefold=args.reward_casefold,
        dp_rank=distributed.dp_rank,
        dp_size=dp_size,
    )
    setup_training(train)
    rollout = DistributedHFThinkerRollout(
        cfg,
        group=group,
        dp_rank=distributed.dp_rank,
        pp_rank=distributed.pp_rank,
        pp_size=args.pp,
        model_config=raw,
        prompts=prompts,
        reference=reference,
        eos_token_id=tokenizer.eos_token_id,
        reward_config=reward.config,
        native_model=training.model,
    )
    if args.resume_request is None:
        initialize_native_policy(training.model, reference, config, cfg, group=group)
    publisher = partial(publish_native_policy, training.model, config, cfg, group)
    backend = PithTrainPolicyAdapter.from_training_context(export_policy=publisher, training_cfg=t)
    rlcfg = OmniRLCfg()
    rlcfg.max_requests, rlcfg.max_updates = args.requests, args.updates
    rlcfg.group_size = rlcfg.max_samples = args.group_size
    rlcfg.max_sequence_length, rlcfg.seed = args.sequence_length, args.seed
    trainer = OmniRLTrainer(
        rlcfg, contract=rollout.contract, backend=backend, rollout=rollout, reward=reward
    )
    try:
        if args.resume_request is not None:
            backend.load(args.checkpoint, trainer, request_index=args.resume_request)
        if observer is not None:
            observer("ready", trainer, None)
        while trainer.requests < args.requests and trainer.updates < args.updates:
            report = trainer.train_step()
            if not args.no_save:
                backend.save(args.checkpoint, trainer)
            if observer is not None:
                observer("update", trainer, report)
            print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        rollout.close()


if __name__ == "__main__":
    main()
