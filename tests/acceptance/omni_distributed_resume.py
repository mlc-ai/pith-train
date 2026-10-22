"""Real Omni + DualPipeV/FSDP update/save/fresh-process resume acceptance.

prepare is CPU-only. GPU modes require a separately coordinated torchrun job.
This entrance is prepared, not a claim that its multi-GPU gates have passed.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from tests.acceptance.omni_native_resume import equal, model_config, sha, verify_input
from tests.test_omni_training_acceptance import digest

REPO = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "produce", "restore", "negative"), required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--restore-root", type=Path, help="Strict checkpoint root rebuilt via HF bridge"
    )
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument(
        "--negative-case",
        choices=("missing-rank", "missing-cuda-rng", "shape", "topology", "optimizer-order"),
    )
    args = parser.parse_args(argv)
    if (args.mode == "negative") != (args.negative_case is not None):
        parser.error("Supply --negative-case exactly when mode is negative")
    if args.restore_root is not None and args.mode not in {"restore", "negative"}:
        parser.error("--restore-root applies only to restore/negative")
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    if (
        min(args.world_size, args.pp, args.ep) < 1
        or args.world_size % args.pp
        or (args.world_size // args.pp) % args.ep
    ):
        parser.error("Invalid PP/EP world topology")
    dp = args.world_size // args.pp
    if 8 % dp or 8 // dp < 2 * args.pp:
        parser.error("GB8 must provide at least 2*PP whole local microbatches")
    return args


def identity(args):
    return dict(
        commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        driver=sha(Path(__file__)),
        input=verify_input(args.dataset),
        pp=args.pp,
        ep=args.ep,
        world_size=args.world_size,
        sequence_length=32,
        global_batch_size=8,
        micro_batch_size=1,
        steps=3,
    )


def prepare(args):
    from pithtrain.modules.dataset import MemmapDataset
    from pithtrain.tasks.prepare_omni_data import processor_for

    assert not torch.cuda.is_initialized()
    assert not args.output.exists(), "Use a new evidence directory"
    assert not args.output.is_relative_to(args.dataset) and not args.dataset.is_relative_to(
        args.output
    )
    metadata = identity(args)
    recipe = json.loads((args.dataset / "bundle.json").read_text())["recipe"]
    _, thinker = processor_for(recipe, offline=True)
    config = model_config()
    assert config.vocab_size >= thinker.text_config.vocab_size
    shards = [MemmapDataset(path, 32) for path in (args.dataset / "tokens/train").rglob("*.bin")]
    assert shards and sum(map(len, shards)) >= 24
    assert all(
        0 <= int(shard.tokens.min()) <= int(shard.tokens.max()) < config.vocab_size
        for shard in shards
    )
    args.output.mkdir(parents=True)
    shutil.copytree(args.dataset, args.output / "data")
    config.save_pretrained(args.output / "model")
    assert verify_input(args.output / "data") == metadata["input"]
    (args.output / "prepare.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    print("CPU_PREPARE_PASS; CUDA_INITIALIZED=False; GPU_ACCEPTANCE=NOT_RUN")


class ObservedData:
    """Record the exact batches returned by the unchanged production source."""

    def __init__(self, source):
        self.source, self.batches = source, {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_batch(self, step, device):
        batches = self.source.get_batch(step, device)
        self.batches[step] = digest(
            [
                dict(
                    model_inputs=m.model_inputs,
                    objective_inputs=m.objective_inputs,
                    cu_seqlens=m.cu_seqlens,
                    sample_ids=m.sample_ids,
                )
                for m in batches
            ]
        )
        return batches


def snapshot(data):
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.distributed_checkpoint import _pack, _runtime

    values, _ = _runtime(
        training.model,
        training.optimizers,
        training.schedulers,
        data.checkpoint_state,
        distributed.device,
    )
    return _pack(values)


def verify_corruption_rejection(error, *, fault, world_size, model_key=None):
    """Accept only the injected fault's precise collective pre-mutation error."""
    from pithtrain.modules.distributed_checkpoint import DistributedCheckpointError

    expected_phase = "validate every rank before mutation"
    expected_type = "DistributedCheckpointError"
    expected_ranks = [world_size - 1]
    if fault == "missing-rank":
        expected_phase = "read and verify files"
        expected_message = "Missing/unexpected rank files"
        expected_ranks = list(range(world_size))
    elif fault == "missing-cuda-rng":
        expected_type = "StrictResumeError"
        expected_message = "app.resume.rng: missing=['cuda'], unexpected=[]"
    elif fault == "shape":
        assert isinstance(model_key, str) and model_key, "Missing injected model key"
        expected_message = f"state.model.{model_key}: local shard shape/dtype mismatch"
    elif fault == "topology":
        expected_message = "Rank topology, configuration, membership or model layout differs"
    elif fault == "optimizer-order":
        expected_message = "Optimizer group configuration/membership differs"
    else:
        raise AssertionError(f"Unknown corruption fault: {fault}")
    assert type(error) is DistributedCheckpointError, (
        f"Expected a collective rejection for {fault}, got {error!r}"
    )
    message = str(error)
    prefix = expected_phase + ": "
    assert message.startswith(prefix), f"Wrong rejection phase for {fault}: {message}"
    errors = []
    for item in message.removeprefix(prefix).split("; "):
        match = re.fullmatch(r"rank (\d+): (\w+): (.+)", item)
        assert match is not None, f"Malformed collective rejection: {message}"
        rank, error_type, reason = match.groups()
        assert error_type == expected_type and reason == expected_message, (
            f"Wrong rejection reason for {fault}: {item}"
        )
        errors.append(dict(rank=int(rank), error_type=error_type, message=reason))
    assert [item["rank"] for item in errors] == expected_ranks, (
        f"Wrong rejecting ranks for {fault}: {errors}"
    )
    return dict(
        fault=fault,
        reason_verified=True,
        error_type=type(error).__name__,
        phase=expected_phase,
        message=message,
        rank_errors=errors,
        expected=dict(
            phase=expected_phase,
            error_type=expected_type,
            message=expected_message,
            ranks=expected_ranks,
        ),
    )


def negative_case(args, cfg, data):
    from pithtrain.modules.checkpoint import load_checkpoint
    from pithtrain.modules.distributed_checkpoint import (
        MANIFEST,
        DistributedCheckpointError,
        _phase,
        resume_contract,
    )

    target = args.output / ("negative-" + args.negative_case)

    def corrupt_one_rank():
        if dist.get_rank() != 0:
            return
        shutil.copytree(
            (args.restore_root or args.output / "checkpoints") / "torch-dcp/00000002",
            target / "torch-dcp/00000002",
        )
        location = target / "torch-dcp/00000002"
        path = location / f"rank-{args.world_size - 1:05d}.pt"
        mutation = dict(target_rank=args.world_size - 1)
        if args.negative_case == "missing-rank":
            path.unlink()
            return mutation
        payload = torch.load(path, weights_only=True)
        if args.negative_case == "missing-cuda-rng":
            del payload["rng"]["cuda"]
        elif args.negative_case == "shape":
            key = next(iter(payload["state"]["model"]))
            original = payload["state"]["model"][key]["value"]
            assert original.shape != (1,), "Shape corruption must change the tensor shape"
            payload["state"]["model"][key]["value"] = original.new_zeros(1)
            mutation["model_key"] = key
        elif args.negative_case == "topology":
            assert payload["header"]["topology"]["ep_size"] != 999
            payload["header"]["topology"]["ep_size"] = 999
        elif args.negative_case == "optimizer-order":
            order = payload["state"]["optimizer"][0]["param_groups"][0]["params"]
            assert len(order) > 1 and order != list(reversed(order)), (
                "Optimizer-order corruption must change parameter order"
            )
            order.reverse()
        else:
            raise AssertionError(f"Unknown corruption fault: {args.negative_case}")
        torch.save(payload, path)
        manifest = json.loads((location / MANIFEST).read_text())
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (location / MANIFEST).write_text(json.dumps(manifest))
        return mutation

    mutation = [_phase("construct one-rank corruption", corrupt_one_rank)]
    dist.broadcast_object_list(mutation, src=0)
    before = snapshot(data)
    rejection = None
    try:
        load_checkpoint(
            target,
            2,
            data_state=data.checkpoint_state,
            strict=True,
            resume_config=resume_contract(cfg),
        )
    except DistributedCheckpointError as error:
        rejection = error
    evidence = _phase(
        "verify corruption rejection reason",
        lambda: verify_corruption_rejection(
            rejection,
            fault=args.negative_case,
            world_size=args.world_size,
            model_key=mutation[0].get("model_key"),
        ),
    )
    _phase(
        "verify corruption failure atomicity",
        lambda: equal(snapshot(data), before, "all-rank-failure-atomicity"),
    )
    evidence.update(atomicity_verified=True, corruption=mutation[0])
    return evidence


def run_gpu(args):
    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", (
        "Explicit scheduling opt-in required"
    )
    assert int(os.environ.get("WORLD_SIZE", "1")) == args.world_size
    assert json.loads((args.output / "prepare.json").read_text()) == identity(args)
    assert verify_input(args.output / "data") == verify_input(args.dataset)
    subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"], check=True)
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.checkpoint import find_checkpoint, load_checkpoint
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.distributed_checkpoint import resume_contract
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import make_adamw_optimizer, make_wsd_scheduler, setup_training
    from pithtrain.tasks import pretrain_lm

    cfg = pretrain_lm.PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size, cfg.distributed.expert_parallel_size = args.pp, args.ep
    cfg.distributed.timeout = timedelta(seconds=90)
    cfg.data.dataset, cfg.data.format, cfg.data.modalities = (
        args.output / "data",
        "prepared_bundle",
        ("text",),
    )
    cfg.data.num_workers = 0
    t = cfg.training
    t.model = args.output / "model"
    t.optimizer = make_adamw_optimizer
    t.scheduler = partial(make_wsd_scheduler, start_lr=1e-6, warmup_ratio=0.25, decay_ratio=0)
    t.lr, t.max_steps, t.sequence_length, t.global_batch_size, t.micro_batch_size = (
        1e-5,
        3,
        32,
        8,
        1,
    )
    t.strict_resume, t.save_interval = True, 2
    t.save_location = args.output / (
        "checkpoints" if args.mode == "produce" else "restored-checkpoints"
    )
    setup_logging(cfg)
    setup_distributed(cfg)
    assert "H100" in torch.cuda.get_device_name(), "Use the approved Flame target"
    data = ObservedData(pretrain_lm.setup_dataset(cfg))
    setup_training(cfg)
    rank = distributed.rank
    label = args.mode if args.mode != "negative" else "negative-" + args.negative_case
    report = args.output / f"{label}-rank-{rank:05d}.json"
    assert not report.exists(), "Retain previous evidence; choose a new output"
    expected_path = args.output / f"continuous-rank-{rank:05d}.pt"
    rejection = None
    if args.mode == "negative":
        rejection = negative_case(args, cfg, data)
    elif args.mode == "produce":
        for step in range(2):
            pretrain_lm.train_step(cfg, data, step)
        assert find_checkpoint(t.save_location) == 2
        expected = dict(boundary=snapshot(data), producer_pid=os.getpid())
        pretrain_lm.train_step(cfg, data, 2)
        expected.update(after=snapshot(data), next_batch=data.batches[2])
        torch.save(expected, expected_path)
    else:
        expected = torch.load(expected_path, weights_only=True)
        assert expected["producer_pid"] != os.getpid(), "Use a new torchrun process"
        assert all(not opt.state for opt in training.optimizers)
        with torch.no_grad():
            for parameter in training.model.parameters():
                parameter.zero_()
        torch.manual_seed(9917 + rank)
        load_checkpoint(
            args.restore_root or args.output / "checkpoints",
            2,
            data_state=data.checkpoint_state,
            strict=True,
            resume_config=resume_contract(cfg),
        )
        equal(snapshot(data), expected["boundary"], f"rank-{rank}-boundary")
        pretrain_lm.train_step(cfg, data, 2)
        equal(data.batches[2], expected["next_batch"], f"rank-{rank}-batch")
        equal(snapshot(data), expected["after"], f"rank-{rank}-after-native-update")
    assert data.consumed_samples == (0 if args.mode == "negative" else 24)
    dist.barrier()
    report.write_text(
        json.dumps(
            dict(
                status="PASSED",
                mode=args.mode,
                rank=rank,
                world_size=args.world_size,
                pp=args.pp,
                ep=args.ep,
                cp=1,
                native_updates=args.mode != "negative",
                negative_case=args.negative_case,
                rejection=rejection,
                all_ranks_rejected=args.mode == "negative",
                real_loader=True,
                fsdp=True,
                converted_checkpoint_root=str(args.restore_root) if args.restore_root else None,
                exact_restore=args.mode == "restore",
                consumed_samples=data.consumed_samples,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    args = parse_args()
    prepare(args) if args.mode == "prepare" else run_gpu(args)
