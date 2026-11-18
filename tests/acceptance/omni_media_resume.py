"""Strict fresh-process resume gates reusable by native multimodal consumers.

Select audio, video or the complete multimodal Thinker with --component.
All modes use the production prepared-bundle reader. All native model parameters,
optimizers, schedulers and RNG remain in strict snapshots. The reader is step-indexed:
snapshots and checkpoints hold only its static data fingerprint, and the completed
step selects the next batch, so no data cursor is saved or restored.

Each phase boundary prints one flushed OMNI_MEDIA_RESUME_PHASE JSON line to stderr (UTC
time, rank, pid, seconds since process start), so a run killed by an outer timeout still
shows the last phase it reached. The markers only print.
"""

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from pithtrain.modules.reproducibility import runtime_determinism_snapshot
from pithtrain.tasks.prepare_omni_data import verify_bundle
from tests.acceptance.omni_distributed_resume import (
    negative_case,
    snapshot,
    stored_data_fingerprint,
)
from tests.acceptance.omni_image_evidence import OptimizerStepEvidence
from tests.acceptance.omni_native_resume import equal, sha
from tests.test_omni_training_acceptance import digest

REPO = Path(__file__).resolve().parents[2]
PROCESS_NONCE = uuid.uuid4().hex


def process_age():
    """Seconds since this process started (Linux /proc), or None elsewhere."""
    try:
        with open("/proc/self/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        with open("/proc/uptime") as handle:
            uptime = float(handle.read().split()[0])
        return round(uptime - int(fields[19]) / os.sysconf("SC_CLK_TCK"), 3)
    except (OSError, ValueError, IndexError):
        return None


def phase_marker(phase, **fields):
    """Print one flushed stderr line marking a phase boundary; it changes no state."""
    record = dict(
        utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        rank=os.environ.get("RANK"),
        local_rank=os.environ.get("LOCAL_RANK"),
        pid=os.getpid(),
        process_age_seconds=process_age(),
        phase=phase,
        **fields,
    )
    print(
        "OMNI_MEDIA_RESUME_PHASE", json.dumps(record, sort_keys=True), file=sys.stderr, flush=True
    )


@dataclass(frozen=True)
class MediaSpec:
    name: str
    modalities: tuple[str, ...]
    training_flag: str
    required_components: tuple[str, ...]
    source_inputs: object
    training_groups: object
    hf_name: object
    validate_inputs: object
    sequence_length: int = 2048
    epoch_samples: int = 32
    seed: int = 1234

    def contract(self):
        assert self.modalities and self.required_components
        return dict(
            name=self.name,
            modalities=list(self.modalities),
            training_flag=self.training_flag,
            required_components=list(self.required_components),
            sequence_length=self.sequence_length,
            epoch_samples=self.epoch_samples,
            seed=self.seed,
            sources={
                str(Path(inspect.getsourcefile(function)).relative_to(REPO)): sha(
                    inspect.getsourcefile(function)
                )
                for function in (
                    self.source_inputs,
                    self.training_groups,
                    self.hf_name,
                    self.validate_inputs,
                )
            },
        )


def audio_spec():
    from pithtrain.models.qwen3_omni_audio_inputs import validate_audio_inputs
    from tests.acceptance.omni_audio_training import hf_name, source_inputs, training_groups

    return MediaSpec(
        "audio",
        ("audio",),
        "native_audio",
        ("audio_tower.",),
        source_inputs,
        training_groups,
        hf_name,
        validate_audio_inputs,
    )


def video_spec():
    from pithtrain.models.qwen3_omni_video_inputs import validate_video_inputs
    from tests.acceptance.omni_audio_training import hf_name
    from tests.acceptance.omni_image_evidence import training_groups
    from tests.acceptance.omni_video_training import source_inputs

    return MediaSpec(
        "video",
        ("video",),
        "native_video",
        ("visual.",),
        source_inputs,
        training_groups,
        hf_name,
        validate_video_inputs,
    )


def multimodal_spec():
    from pithtrain.models.qwen3_omni_multimodal_inputs import multimodal_position_ids
    from tests.acceptance.omni_audio_training import hf_name
    from tests.acceptance.omni_multimodal_training import component_groups, source_inputs

    return MediaSpec(
        "multimodal",
        ("text", "image", "audio", "video"),
        "native_multimodal",
        ("audio_tower.", "visual."),
        source_inputs,
        component_groups,
        hf_name,
        multimodal_position_ids,
    )


def component_spec(name):
    return {"audio": audio_spec, "video": video_spec, "multimodal": multimodal_spec}[name]()


def fixture_batches(spec, batches):
    """Retain every named mixed fixture instead of iterating over its dict keys."""
    fixtures = dict(batches) if isinstance(batches, Mapping) else dict(enumerate(batches))
    assert fixtures
    if spec.name == "multimodal":
        assert set(fixtures) == {"text", "image", "audio", "video", "av", "all"}, (
            "Complete Thinker preparation requires all six input classes"
        )
    return {str(name): batch for name, batch in fixtures.items()}


def reader_config(dataset, spec):
    from pithtrain.modules.data_config import DataCfg

    data = DataCfg()
    data.dataset, data.format, data.modalities = dataset, "prepared_bundle", spec.modalities
    data.num_workers, data.epoch_samples = 0, spec.epoch_samples
    return data, SimpleNamespace(
        global_batch_size=8,
        micro_batch_size=1,
        sequence_length=spec.sequence_length,
        seed=spec.seed,
    )


def require_input_components(batches, prefixes):
    """Every global step must exercise both towers when both are required."""
    keys = set().union(*(set(batch.media_inputs or {}) for batch in batches))
    for prefix in prefixes:
        present = (
            "input_features" in keys
            if prefix == "audio_tower."
            else bool({"pixel_values", "pixel_values_videos"} & keys)
        )
        assert prefix in {"audio_tower.", "visual."}
        assert present, f"No production input for component: {prefix}"


def cpu_reader_audit(dataset, spec, config, dp_size):
    """Read all DP partitions on CPU; this does not claim PP/GPU execution.

    The reader is step-indexed: after two completed steps, a fresh reader with the same
    static data fingerprint must return the continuous run's step-2 batch from the step
    alone. No reader cursor exists to save or restore.
    """
    from pithtrain.modules.training_data import PreparedBundleData

    data_cfg, training_cfg = reader_config(dataset, spec)

    def make_reader(rank):
        reader = PreparedBundleData(data_cfg, training_cfg, dp_rank=rank, dp_size=dp_size)
        reader.validate_model(
            SimpleNamespace(
                input_modalities=spec.modalities, validate_media_batch=spec.validate_inputs
            ),
            config,
        )
        return ObservedMediaData(reader)

    readers = [make_reader(rank) for rank in range(dp_size)]
    # Strict resume requires every rank to agree on this identity; it excludes the DP partition.
    fingerprint = readers[0].data_fingerprint
    assert all(reader.data_fingerprint == fingerprint for reader in readers)
    steps = []
    for step in range(3):
        global_batches = [reader.get_batch(step, "cpu") for reader in readers]
        require_input_components(
            [batch for part in global_batches for batch in part], spec.required_components
        )
        assert all(
            sum(int((batch.objective_inputs[0] != -100).sum()) for batch in part) > 0
            for part in global_batches
        )
        steps.append(
            [
                dict(
                    dp_rank=rank,
                    sample_ids=[batch.sample_ids for batch in part],
                    input_hashes=readers[rank].batches[step],
                )
                for rank, part in enumerate(global_batches)
            ]
        )
    for rank, reader in enumerate(readers):
        # Resume at the step-2 boundary: a fresh reader seeks the next step from the step.
        restored = make_reader(rank)
        assert restored.data_fingerprint == fingerprint, f"reader-{rank}-fingerprint"
        restored.get_batch(2, "cpu")
        equal(restored.batches[2], reader.batches[2], f"reader-{rank}-next-input")
    return dict(
        status="PASSED",
        scope="CPU production DP partitions and exact step-indexed next-data restore",
        gpu_acceptance="NOT_RUN",
        pp_execution="NOT_RUN",
        dp_size=dp_size,
        data_fingerprint=fingerprint,
        restored_step=2,
        next_step=2,
        steps=steps,
    )


def input_identity(root, modalities):
    bundle = verify_bundle(root)
    assert set(modalities) <= set(bundle["manifests"]["train"]), "Requested modality is absent"
    return dict(bundle=sha(root / "bundle.json"), checksums=sha(root / "checksums.sha256"))


def identity(args, spec):
    return dict(
        expected_hardware_family=getattr(args, "hardware_family", "H100"),
        # v2: strict state holds the static data fingerprint; v1 evidence restored a cursor.
        protocol="strict_multimodal_save_then_runtime_state_v2",
        commit=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        driver=sha(Path(__file__)),
        consumer=spec.contract(),
        input=input_identity(args.dataset, spec.modalities),
        pp=args.pp,
        ep=args.ep,
        world_size=args.world_size,
        dp=args.world_size // args.pp,
        cp=1,
        sequence_length=spec.sequence_length,
        global_batch_size=8,
        micro_batch_size=1,
        steps=3,
    )


def prepare(args, spec):
    assert not torch.cuda.is_initialized()
    assert not args.output.exists(), "Use a fresh evidence directory"
    assert not args.output.is_relative_to(args.dataset) and not args.dataset.is_relative_to(
        args.output
    )
    metadata = identity(args, spec)
    _, samples, batches, config = spec.source_inputs(args.dataset)
    assert samples and batches
    batches = fixture_batches(spec, batches)
    for batch in batches.values():
        assert (batch.labels != -100).any()
        spec.validate_inputs(batch.model_inputs["input_ids"], batch.model_inputs, config)
    args.output.mkdir(parents=True)
    shutil.copytree(args.dataset, args.output / "data")
    config.save_pretrained(args.output / "model")
    assert input_identity(args.output / "data", spec.modalities) == metadata["input"]
    metadata["sample_inputs"] = {
        name: digest(
            dict(model_inputs=batch.model_inputs, labels=batch.labels, sample_ids=batch.sample_ids)
        )
        for name, batch in batches.items()
    }
    metadata["production_reader"] = cpu_reader_audit(
        args.output / "data", spec, config, args.world_size // args.pp
    )
    (args.output / "prepare.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    print("CPU_PREPARE_PASS; CUDA_INITIALIZED=False; GPU_ACCEPTANCE=NOT_RUN")


class ObservedMediaData:
    """Hash every tensor returned by the unchanged production source, in step order."""

    def __init__(self, source):
        self.source, self.batches, self.steps_read = source, {}, []

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_batch(self, step, device):
        batches = self.source.get_batch(step, device)
        self.steps_read.append(step)
        self.batches[step] = digest(
            [
                dict(
                    model_inputs=mb.model_inputs,
                    objective_inputs=mb.objective_inputs,
                    cu_seqlens=mb.cu_seqlens,
                    sample_ids=mb.sample_ids,
                    media_context=mb.media_context,
                    media_inputs=mb.media_inputs,
                )
                for mb in batches
            ]
        )
        return batches


class ObjectiveObservation:
    """Observe exact detached objective sums; preserve the production objective."""

    def __init__(self, objective):
        self.objective, self.outputs = objective, []

    def __call__(self, model_outputs, objective_inputs):
        loss, detached = self.objective(model_outputs, objective_inputs)
        self.outputs.append(digest(dict(loss=detached, labels=objective_inputs)))
        return loss, detached


def require_components(keys, prefixes):
    assert keys, "Empty model state"
    for prefix in prefixes:
        assert any(key.startswith(prefix) for key in keys), (
            f"Missing checkpoint component: {prefix}"
        )


# Strict loading validates rank shards one owner at a time, so rank 0 stops the restore.
FINGERPRINT_REJECTION = dict(
    phase="validate every rank before mutation",
    error_type="DistributedCheckpointError",
    message=(
        "Checkpoint data fingerprint differs: check the dataset, modalities, sampling, "
        "seed, sequence length and batch sizes before resuming"
    ),
    ranks=[0],
)


def require_stored_fingerprint(root, rank, data_fingerprint):
    """This rank's committed step-2 strict shard must hold exactly the live data identity."""
    stored = stored_data_fingerprint(root, rank)
    assert stored == data_fingerprint, "Checkpoint data fingerprint differs from this run"
    return stored


def verify_fingerprint_rejection(error):
    """Accept only the strict loader's collective, pre-mutation data identity rejection."""
    from pithtrain.modules.distributed_checkpoint import DistributedCheckpointError

    expected = FINGERPRINT_REJECTION
    assert type(error) is DistributedCheckpointError, (
        f"Expected a collective data fingerprint rejection, got {error!r}"
    )
    message = str(error)
    prefix = expected["phase"] + ": "
    assert message.startswith(prefix), f"Wrong data fingerprint rejection phase: {message}"
    ranks = []
    for item in message.removeprefix(prefix).split("; "):
        match = re.fullmatch(r"rank (\d+): (\w+): (.+)", item)
        assert match is not None, f"Malformed collective rejection: {message}"
        rank, error_type, reason = match.groups()
        assert error_type == expected["error_type"] and reason == expected["message"], (
            f"Wrong data fingerprint rejection reason: {item}"
        )
        ranks.append(int(rank))
    assert ranks == expected["ranks"], f"Wrong data fingerprint rejecting ranks: {ranks}"
    return dict(reason_verified=True, message=message, expected=expected)


def fingerprint_rejection(root, step, data_fingerprint, resume_config, capture):
    """A mismatched live data fingerprint must fail on every rank before any state changes.

    capture() returns this rank's complete runtime snapshot and may run collectives.
    """
    from pithtrain.modules.distributed_checkpoint import (
        DistributedCheckpointError,
        _phase,
        load_checkpoint,
    )

    mismatched = hashlib.sha256(f"mismatched-{data_fingerprint}".encode()).hexdigest()
    assert mismatched != data_fingerprint
    before = capture()
    rejection = None
    try:
        load_checkpoint(
            root, step, data_fingerprint=mismatched, strict=True, resume_config=resume_config
        )
    except DistributedCheckpointError as error:
        rejection = error
    evidence = _phase(
        "verify data fingerprint rejection", lambda: verify_fingerprint_rejection(rejection)
    )
    after = capture()
    _phase(
        "verify data fingerprint rejection atomicity",
        lambda: equal(after, before, "data-fingerprint-rejection-atomicity"),
    )
    evidence.update(atomicity_verified=True, mismatched_fingerprint=mismatched)
    return evidence


def run_gpu(args, spec):
    phase_marker("run-gpu-start", mode=getattr(args, "mode", None), component=spec.name)
    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", "GPU execution is opt-in"
    assert int(os.environ.get("WORLD_SIZE", "1")) == args.world_size
    assert torch.cuda.is_available(), "Requires CUDA"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    phase_marker("cuda-device-set")
    from tests.acceptance.omni_hopper_hardware import require_hardware

    require_hardware(getattr(args, "hardware_family", "H100"))
    phase_marker("hardware-verified")
    prepared = json.loads((args.output / "prepare.json").read_text())
    assert {
        key: value
        for key, value in prepared.items()
        if key not in {"sample_inputs", "production_reader"}
    } == identity(args, spec)
    assert prepared["production_reader"]["status"] == "PASSED"
    assert input_identity(args.output / "data", spec.modalities) == prepared["input"]
    subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"], check=True)
    phase_marker("prepared-input-verified")
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.checkpoint import find_checkpoint
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.distributed_checkpoint import _phase, load_checkpoint, resume_contract
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import make_adamw_optimizer, make_wsd_scheduler, setup_training
    from pithtrain.tasks import pretrain_lm

    phase_marker("training-imports-done")

    cfg = pretrain_lm.PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size, cfg.distributed.expert_parallel_size = args.pp, args.ep
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.timeout = timedelta(seconds=90)
    cfg.data.dataset, cfg.data.format, cfg.data.modalities = (
        args.output / "data",
        "prepared_bundle",
        spec.modalities,
    )
    cfg.data.num_workers, cfg.data.epoch_samples = 0, spec.epoch_samples
    t = cfg.training
    t.model, t.seed = args.output / "model", spec.seed
    setattr(t, spec.training_flag, True)
    t.optimizer = make_adamw_optimizer
    t.scheduler = partial(make_wsd_scheduler, start_lr=1e-6, warmup_ratio=0.25, decay_ratio=0)
    t.lr, t.max_steps, t.sequence_length, t.global_batch_size, t.micro_batch_size = (
        1e-5,
        3,
        spec.sequence_length,
        8,
        1,
    )
    t.fp8, t.moe_load_balance_coef = False, 0.0
    t.strict_resume, t.save_interval = True, 2
    t.save_location = args.output / (
        "checkpoints" if args.mode == "produce" else "restored-checkpoints"
    )
    setup_logging(cfg)
    phase_marker("setup-distributed-start")
    setup_distributed(cfg)
    phase_marker("setup-distributed-done")
    data = ObservedMediaData(pretrain_lm.setup_dataset(cfg))
    phase_marker("dataset-ready")
    setup_training(cfg)
    phase_marker("training-ready", max_allocated_bytes=torch.cuda.max_memory_allocated())
    rank = distributed.rank
    label = args.mode if args.mode != "negative" else "negative-" + args.negative_case
    report_path = args.output / f"{label}-rank-{rank:05d}.json"
    assert not report_path.exists(), "Retain previous evidence; use a fresh output"
    expected_path = args.output / f"continuous-rank-{rank:05d}.pt"
    report = dict(
        status="RUNNING",
        mode=args.mode,
        rank=rank,
        source=prepared,
        process_nonce=PROCESS_NONCE,
        pid=os.getpid(),
        hardware=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        runtime_determinism=runtime_determinism_snapshot(),
    )
    actual = {
        spec.hf_name(name): parameter
        for module in training.model.module
        for name, parameter in module.named_parameters()
    }
    observer = OptimizerStepEvidence(
        actual, spec.training_groups(config_from_file(t.model)), reduce=dist.all_reduce
    )
    assert len(training.optimizers) == 1
    optimizer = training.optimizers[0]
    pre_hook, post_hook = (
        optimizer.register_step_pre_hook(observer.before_step),
        optimizer.register_step_post_hook(observer.after_step),
    )

    def observed_step(step):
        # Step 1 also writes the step-2 checkpoint (save_interval 2) inside train_step.
        phase_marker("train-step-start", step=step)
        observation = ObjectiveObservation(pretrain_lm.objective)
        with patch.object(pretrain_lm, "objective", observation):
            pretrain_lm.train_step(cfg, data, step)
        phase_marker(
            "train-step-done", step=step, max_allocated_bytes=torch.cuda.max_memory_allocated()
        )
        return observation.outputs

    def complete_snapshot():
        phase_marker("snapshot-start")
        state = snapshot(data)
        local_keys = sorted(state["model"])
        keys = [None] * args.world_size
        dist.all_gather_object(keys, local_keys)
        require_components(set().union(*(set(part) for part in keys)), spec.required_components)
        assert set(state) == {"model", "optimizer", "scheduler", "data", "rng"}
        assert set(state["rng"]) == {"torch_cpu", "python", "numpy", "cuda"}
        # Step-indexed data contributes only its static identity, never a reader cursor.
        assert state["data"] == {"fingerprint": data.data_fingerprint}
        phase_marker("snapshot-done")
        return state

    try:
        if args.mode == "negative":
            phase_marker("negative-case-start", case=args.negative_case)
            report["rejection"] = negative_case(args, cfg, data)
            report["negative_case"] = args.negative_case
            report["all_ranks_rejected"] = True
        elif args.mode == "produce":
            for step in range(2):
                observed_step(step)
            assert find_checkpoint(t.save_location) == 2
            phase_marker("checkpoint-found", step=2)
            report["checkpoint_data_fingerprint"] = require_stored_fingerprint(
                t.save_location, rank, data.data_fingerprint
            )
            expected = dict(
                boundary=complete_snapshot(), producer_nonce=PROCESS_NONCE, producer_pid=os.getpid()
            )
            objectives = observed_step(2)
            expected.update(
                after=complete_snapshot(),
                next_batch=data.batches[2],
                third_objectives=objectives,
                third_module_updates=observer.steps[-1],
            )
            phase_marker("expected-state-save-start")
            torch.save(expected, expected_path)
            phase_marker("expected-state-saved")
            assert len(observer.steps) == 3
        else:
            phase_marker("expected-state-load-start")
            expected = torch.load(expected_path, weights_only=True)
            assert expected["producer_nonce"] != PROCESS_NONCE, (
                "Use a fresh Python/torchrun process"
            )
            assert all(not opt.state for opt in training.optimizers)
            with torch.no_grad():
                for parameter in training.model.parameters():
                    parameter.zero_()
            torch.manual_seed(9917 + rank)
            root = args.restore_root or args.output / "checkpoints"
            # The checkpoint carries no cursor: prove that it holds this reader's identity
            # and that strict loading rejects any other identity before mutating state.
            phase_marker("data-identity-start", step=2)
            report["checkpoint_data_fingerprint"] = require_stored_fingerprint(
                root, rank, data.data_fingerprint
            )
            report["fingerprint_rejection"] = fingerprint_rejection(
                root, 2, data.data_fingerprint, resume_contract(cfg), complete_snapshot
            )
            phase_marker("data-identity-verified", step=2)
            phase_marker("checkpoint-load-start", step=2)
            load_checkpoint(
                root,
                2,
                data_fingerprint=data.data_fingerprint,
                strict=True,
                resume_config=resume_contract(cfg),
            )
            phase_marker("checkpoint-loaded", step=2)
            boundary = complete_snapshot()
            _phase(
                "exact complete boundary",
                lambda: equal(boundary, expected["boundary"], f"rank-{rank}-boundary"),
            )
            objectives = observed_step(2)
            after = complete_snapshot()
            _phase(
                "exact next media input",
                lambda: equal(data.batches[2], expected["next_batch"], f"rank-{rank}-input"),
            )
            _phase(
                "exact third objective",
                lambda: equal(objectives, expected["third_objectives"], f"rank-{rank}-objective"),
            )
            _phase(
                "exact third state", lambda: equal(after, expected["after"], f"rank-{rank}-after")
            )
            assert len(observer.steps) == 1
            _phase(
                "exact third component updates",
                lambda: equal(
                    observer.steps[0],
                    expected["third_module_updates"],
                    f"rank-{rank}-component-updates",
                ),
            )
            report["exact_component_updates"] = True
            report["exact_restore"] = True
            report["exact_third_step"] = True
        # The step alone selects data: the producer read steps 0-2, the restored process
        # only the next step (2), and a rejected restore reads nothing.
        expected_steps = dict(negative=[], produce=[0, 1, 2], restore=[2])[args.mode]
        assert data.steps_read == expected_steps, f"Data steps read: {data.steps_read}"
        if args.mode == "restore":
            report.update(restored_step=2, next_step=data.steps_read[0])
        report.update(
            status="PASSED",
            data_fingerprint=data.data_fingerprint,
            data_steps_read=data.steps_read,
            optimizer_step_evidence=observer.steps,
            required_components=spec.required_components,
            converted_checkpoint_root=str(args.restore_root) if args.restore_root else None,
            max_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        pre_hook.remove()
        post_hook.remove()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        phase_marker("rank-report-written", status=report["status"])
    dist.barrier()
    phase_marker("final-barrier-done")


def config_from_file(path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path, local_files_only=True)
    return getattr(config, "thinker_config", config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "produce", "restore", "negative"), required=True
    )
    parser.add_argument("--component", choices=("audio", "video", "multimodal"), default="audio")
    from tests.acceptance.omni_hopper_hardware import add_hardware_argument

    add_hardware_argument(parser)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore-root", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--pp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ep", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--negative-case",
        choices=(
            "missing-rank",
            "missing-cuda-rng",
            "shape",
            "topology",
            "optimizer-order",
            "fingerprint",
        ),
    )
    args = parser.parse_args(argv)
    if (args.mode == "negative") != (args.negative_case is not None):
        parser.error("--negative-case is required exactly for negative mode")
    if args.restore_root is not None and args.mode not in {"restore", "negative"}:
        parser.error("--restore-root requires restore/negative mode")
    if args.world_size < 1 or args.world_size % args.pp or (args.world_size // args.pp) % args.ep:
        parser.error("Invalid CP1 PP/EP world topology")
    dp = args.world_size // args.pp
    if dp not in (1, 2) or 8 // dp < 2 * args.pp:
        parser.error("Require DP1/2 and at least 2*PP local microbatches")
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    if args.restore_root is not None:
        args.restore_root = args.restore_root.resolve()
    return args


def execute(args, spec):
    prepare(args, spec) if args.mode == "prepare" else run_gpu(args, spec)


if __name__ == "__main__":
    phase_marker("entrypoint-imported")
    arguments = parse_args()
    execute(arguments, component_spec(arguments.component))
