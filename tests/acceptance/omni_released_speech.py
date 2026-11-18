"""Released-weight complete speech FSDP training with fresh-process resume.

Validation rows 20-22: text_and_codec, codec_only and waveform. No download,
allocation, upload or persistent-store API; the job script owns admission.

- ``plan`` prints the contract and sizes of one training mode.
- ``fixture`` (CPU) rebuilds the tiny speech gate's five real-media samples against
  the pinned released config, draws their synthetic RVQ labels over the full
  released codebook, and validates every scheduled rank/scan batch.
- ``stage`` (CPU) converts the pinned released snapshot to the standard speech DCP
  in a new job-owned scratch directory with the production initializer.
- ``produce`` initializes every component from that DCP, runs the tiny gate's
  four-update schedule (five scans for waveform) under the production
  deterministic policies and saves the native strict checkpoint at scan two.
- ``restore`` is a fresh torchrun: it initializes from the same DCP, strictly
  loads scan two and must reproduce every later complete state digest and input
  batch exactly.
- ``cleanup`` removes only the scratch directory that ``stage`` created.

Codec labels are synthetic and the waveform objective is test-defined. This is
a training and resume mechanism gate, not speech quality, and there is no
released-size HF parity.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import time
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.acceptance import omni_released_continuation as continuation
from tests.acceptance import omni_released_update_support as support
from tests.acceptance import omni_speech_chain as chain
from tests.acceptance import omni_speech_sharded as sharded

SCHEMA = "qwen3_omni_released_speech_v1"
MODEL = continuation.MODEL
REVISION = support.REVISION
CONFIG_SHA = support.CONFIG_SHA
METADATA_SHA = continuation.METADATA_SHA
GIB = support.GIB
DP = {"text_and_codec": 8, "codec_only": 2, "waveform": 2}
ROW = {"text_and_codec": 20, "codec_only": 21, "waveform": 22}
ACTIVE = {
    "text_and_codec": ("thinker", "talker"),
    "codec_only": ("talker",),
    "waveform": ("code2wav",),
}
# Exact element counts of the pinned public safetensors headers.
PARAMETERS = {"thinker": 31_719_205_488, "talker": 3_324_596_480, "code2wav": 216_016_577}
PAYLOAD_BYTES = 2 * sum(PARAMETERS.values())
SHARD_BYTES = 70_523_299_202
SAVE_SCAN = 2
UPDATES = 4
CODE_FRAMES = 4
LR = 1e-4
MAX_NORM = sharded.MAX_NORM
OBJECTIVE_ID = "speech-sine-prefix-mask-mse-v1"
SPEAKER = "ethan"
MAX_TENSOR_BYTES = 2**30
HOST_AVAILABLE_GIB = {"text_and_codec": 600, "codec_only": 320, "waveform": 320}
GPU_MARGIN = 8 * GIB
ACTIVATION_BYTES_PER_TOKEN = 8 * 2**20  # Rough text_and_codec estimate; diagnostic only.
MARKER = "owner.json"
SECONDS = {"initialization": 1800, "save": 1800, "load": 3600, "update": 900, "snapshot": 900}


def sizes(mode):
    """Byte budgets from the pinned header counts; no released payload is read."""
    total = sum(PARAMETERS.values())
    trainable = sum(PARAMETERS[name] for name in ACTIVE[mode])
    world = DP[mode]
    persistent = 4 * total + 12 * trainable  # FP32 masters, FP32 grads, two FP32 moments
    checkpoint = 4 * total + 8 * trainable  # FP32 masters plus two FP32 moments
    peak = max(2 * PAYLOAD_BYTES, PAYLOAD_BYTES + checkpoint)  # copy+DCP, then DCP+checkpoint
    return dict(
        parameters=dict(PARAMETERS),
        total_parameters=total,
        trainable_parameters=trainable,
        dp=world,
        payload_bytes=PAYLOAD_BYTES,
        persistent_bytes=persistent,
        persistent_bytes_per_gpu=math.ceil(persistent / world),
        checkpoint_bytes=checkpoint,
        checkpoint_bytes_per_rank=math.ceil(checkpoint / world),
        initialization_hash_reads_per_rank=2 * PAYLOAD_BYTES,
        strict_load_serialized_reads=2 * 3 * checkpoint,
        scratch_peak_bytes=peak,
        scratch_required_bytes=math.ceil(1.1 * peak),
        host_available_admission_bytes=HOST_AVAILABLE_GIB[mode] * GIB,
    )


def plan(mode):
    return dict(
        schema=SCHEMA,
        status="NOT_RUN",
        row=ROW[mode],
        training_mode=mode,
        topology=dict(dp=DP[mode], cp=1, ep=1, pp=1, hsdp_replica=1),
        scans=sharded.scans(mode),
        updates=UPDATES,
        save_scan=SAVE_SCAN,
        restore="fresh torchrun in the same allocation; initialize from the same DCP first",
        weights="pinned public released weights through the production speech initializer",
        precision="FP32 sharded masters/AdamW; BF16 compute; FP32 SUM gradients",
        determinism="configure_deterministic_training(True) before model construction",
        optimizer=dict(type="torch.optim.AdamW", lr=LR, weight_decay=0.0, foreach=False),
        scheduler="LambdaLR constant 1.0",
        max_grad_norm=MAX_NORM,
        codec_labels="synthetic RVQ labels over the full released codebook",
        waveform_objective=OBJECTIVE_ID,
        criteria=[
            "restored scan-2 state digest equals the producer's exactly",
            "every later scan's complete state digest and input batch equal exactly",
            "each update: finite loss, finite FP32 gradient for every trainable parameter,"
            " nonzero gradient and changed values in every trainable component",
            "frozen components bitwise unchanged; Adam steps equal completed updates",
            "waveform zero-target scan leaves model, optimizer, scheduler and RNG unchanged",
        ],
        sizes=sizes(mode),
        limitations=[
            "synthetic codec labels and test-defined waveform objective",
            "no released-size HF parity; no speech quality claim",
            "CP/EP/PP must equal 1",
        ],
    )


def code_layout(config):
    """One label range valid for Talker, residual predictor and code2wav."""
    talker, wave = config.talker_config, config.code2wav_config
    groups = talker.num_code_groups
    if groups != talker.code_predictor_config.num_code_groups or groups != wave.num_quantizers:
        raise chain.PreconditionError("Codec group counts differ between components")
    modulus = min(
        wave.codebook_size, talker.code_predictor_config.vocab_size, talker.text_config.vocab_size
    )
    return modulus, groups


def make_samples(batches, config):
    """The tiny gate's five samples; only the label range becomes the released one."""
    modulus, groups = code_layout(config)
    samples = []
    for index, kind in enumerate(chain.MEDIA):
        sample = chain.make_sample(batches["all" if kind == "mixed" else kind], config, index)
        generator = torch.Generator().manual_seed(chain.SEED + index)
        sample["codes"] = torch.randint(modulus, (1, CODE_FRAMES, groups), generator=generator)
        sample["code_lengths"] = torch.tensor([CODE_FRAMES])
        samples.append(sample)
    return samples


def scheduled_batches(samples, mode, scan, rank, world, modulus, device):
    """omni_speech_sharded.scheduled_batches for any DP world; odd ranks act as rank 1."""
    if type(world) is not int or world < 2 or type(rank) is not int or not 0 <= rank < world:
        raise ValueError("Invalid DP rank/world")
    if type(scan) is not int or not 0 <= scan < sharded.scans(mode):
        raise ValueError("Invalid scan")
    odd = rank % 2
    batches = [chain.move(item, device) for item in samples]
    for batch in batches:
        # Physical forward shape and order are identical on every rank; supervision
        # counts and the supplied RVQ label values differ explicitly.
        batch["codes"] = (batch["codes"] + rank + scan) % modulus
        keep = torch.ones_like(batch["codes"], dtype=torch.bool)
        if odd:
            keep[:, 1::2] = False
            positions = (batch["labels"] != -100).nonzero()
            for row, column in positions[1::2]:
                batch["labels"][row, column] = -100
        if (scan == 1 and odd) or scan == 2:
            batch["labels"].fill_(-100)
        batch["codebook_mask"] = keep
        batch["waveform_fraction"] = (
            0 if mode == "waveform" and (scan == 2 or (scan == 1 and odd)) else 2 - odd
        )
    return batches


class ReleasedData:
    """omni_speech_sharded.ScheduledData for any DP world."""

    def __init__(self, samples, fingerprint, mode, rank, world, modulus):
        self.samples, self.fingerprint, self.mode = samples, fingerprint, mode
        self.rank, self.world, self.modulus = rank, world, modulus
        self.scanned_batches = self.completed_updates = 0
        self.last_update_scan = -1
        self.observed = {}

    def get_batch(self, scan_index, device):
        if scan_index != self.scanned_batches:
            raise ValueError("Reader scan mismatch")
        result = scheduled_batches(
            self.samples, self.mode, scan_index, self.rank, self.world, self.modulus, device
        )
        self.scanned_batches += 1
        self.observed[scan_index] = chain.tensor_tree(result)
        return result

    def commit_update(self, *, scan_index, update_index):
        if (
            scan_index != self.scanned_batches - 1
            or update_index != self.completed_updates
            or sharded.updates_at(self.mode, scan_index + 1) != update_index + 1
        ):
            raise ValueError("Reader update mismatch")
        self.completed_updates += 1
        self.last_update_scan = scan_index

    def state_dict(self):
        return dict(
            fingerprint=self.fingerprint,
            mode=self.mode,
            rank=self.rank,
            world=self.world,
            modulus=self.modulus,
            scanned_batches=self.scanned_batches,
            completed_updates=self.completed_updates,
            last_update_scan=self.last_update_scan,
        )

    def validate_state_dict(self, state):
        if set(state) != set(self.state_dict()) or any(
            state[key] != getattr(self, key)
            for key in ("fingerprint", "mode", "rank", "world", "modulus")
        ):
            raise ValueError("Reader identity differs")
        scan, update, last = (
            state[k] for k in ("scanned_batches", "completed_updates", "last_update_scan")
        )
        if any(type(x) is not int for x in (scan, update, last)):
            raise ValueError("Reader counters must be integers")
        if not 0 <= scan <= sharded.scans(self.mode):
            raise ValueError("Reader scan outside the schedule")
        wanted_last = scan - 1 if scan else -1
        if self.mode == "waveform" and scan == 3:
            wanted_last = 1
        if update != sharded.updates_at(self.mode, scan) or last != wanted_last:
            raise ValueError("Reader update/scan schedule differs")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        for key in ("scanned_batches", "completed_updates", "last_update_scan"):
            setattr(self, key, state[key])


def validate_schedule(samples, config, mode, world):
    """Every rank/scan batch against the production contract; identical call plans."""
    from pithtrain.tasks.omni_speech_batch_contract import validate_speech_batch
    from pithtrain.tasks.omni_speech_fsdp import _local_collective_plan

    holder = SimpleNamespace(config=config)
    modulus, _ = code_layout(config)
    plans = []
    for scan in range(sharded.scans(mode)):
        per_rank = []
        for rank in range(world):
            batches = scheduled_batches(samples, mode, scan, rank, world, modulus, "cpu")
            for batch in batches:
                validate_speech_batch(batch, holder, mode, SPEAKER)
            per_rank.append(_local_collective_plan(batches, mode))
        if any(item != per_rank[0] for item in per_rank):
            raise chain.PreconditionError("Ranks would issue different FSDP call plans")
        plans.append(json.loads(json.dumps(per_rank[0])))
    return plans


def released_config(metadata):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    path = Path(metadata) / "config.json"
    if continuation.file_sha(path) != CONFIG_SHA:
        raise chain.PreconditionError("Pinned released config differs")
    return Qwen3OmniMoeConfig(**json.loads(path.read_text(encoding="utf-8")))


def fixture(args):
    from tests.acceptance.omni_multimodal_training import source_inputs

    if torch.cuda.is_initialized():
        raise chain.PreconditionError("Fixture preparation must not initialize CUDA")
    source = chain.source_identity(args.expected_commit)
    config = released_config(args.metadata)
    bundle, _, batches, _ = source_inputs(args.dataset)
    if bundle["recipe"]["revision"] != REVISION:
        raise chain.PreconditionError("Prepared bundle processor revision differs")
    samples = make_samples(batches, config)
    plans = {mode: validate_schedule(samples, config, mode, DP[mode]) for mode in DP}
    args.output.mkdir()
    path = args.output / "samples.pt"
    torch.save(samples, path)
    tokens = {kind: int(s["input_ids"].shape[1]) for kind, s in zip(chain.MEDIA, samples)}
    modulus, groups = code_layout(config)
    report = dict(
        schema=SCHEMA,
        status="PREPARED_NOT_TRAINED",
        source=source,
        seed=chain.SEED,
        media=list(chain.MEDIA),
        dataset_checksum=chain.sha(args.dataset / "checksums.sha256"),
        processor_revision=bundle["recipe"]["revision"],
        released_config_sha256=CONFIG_SHA,
        samples_sha256=continuation.file_sha(path),
        code_frames=CODE_FRAMES,
        code_groups=groups,
        codebook_modulus=modulus,
        tokens=tokens,
        tokens_per_rank_scan=sum(tokens.values()),
        text_and_codec_activation_estimate_bytes=ACTIVATION_BYTES_PER_TOKEN * sum(tokens.values()),
        collective_plans=plans,
        codec_provenance="synthetic RVQ labels over the full released codebook;"
        " not audio-encoder ground truth",
        waveform_objective=OBJECTIVE_ID,
        numerical_acceptance="NOT_RUN",
    )
    continuation.write_json(args.output / "fixture.json", report)
    return report


def load_fixture(path, digest, commit):
    path = Path(path)
    if continuation.file_sha(path / "fixture.json") != digest:
        raise chain.PreconditionError("Fixture report checksum differs")
    report = continuation.read_json(path / "fixture.json")
    if report["schema"] != SCHEMA or report["source"]["commit"] != commit:
        raise chain.PreconditionError("Fixture belongs to another source or contract")
    if report["released_config_sha256"] != CONFIG_SHA:
        raise chain.PreconditionError("Fixture released config differs")
    if continuation.file_sha(path / "samples.pt") != report["samples_sha256"]:
        raise chain.PreconditionError("Fixture samples checksum differs")
    return report, torch.load(path / "samples.pt", map_location="cpu", weights_only=True)


def mount_of(path, mountinfo=None):
    """The deepest mount containing path and its filesystem type."""
    path = Path(path).resolve()
    text = Path("/proc/self/mountinfo").read_text() if mountinfo is None else mountinfo
    best = None
    for line in text.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), left.split()[4])
        point = Path(point)
        if (path == point or point in path.parents) and (
            best is None or len(point.parts) > len(best[0].parts)
        ):
            best = (point, right.split()[0])
    if best is None:
        raise chain.PreconditionError("No mount contains the scratch path")
    return str(best[0]), best[1]


def scratch_admission(parent, required, mountinfo=None):
    parent = Path(parent)
    if not parent.is_dir() or parent.is_symlink():
        raise chain.PreconditionError("Scratch parent must be an existing real directory")
    mount, fstype = mount_of(parent, mountinfo)
    free = shutil.disk_usage(parent).free
    errors = []
    if fstype in ("tmpfs", "ramfs"):
        errors.append("Memory-backed scratch would be charged to RAM; not admitted")
    if free < required:
        errors.append(f"Scratch free bytes {free} below required {required}")
    return dict(
        parent=str(parent.resolve()),
        mount=mount,
        fstype=fstype,
        free_bytes=free,
        required_bytes=required,
        errors=errors,
    )


def released_files(metadata, source, receipt):
    """Pinned config/index identities and the receipt's 15 shard identities."""
    metadata, source = Path(metadata), Path(source)
    if continuation.file_sha(metadata / "manifest.json") != METADATA_SHA:
        raise chain.PreconditionError("Pinned metadata manifest differs")
    manifest = continuation.read_json(metadata / "manifest.json")
    if manifest["model"] != MODEL or manifest["revision"] != REVISION:
        raise chain.PreconditionError("Pinned metadata belongs to another release")
    pinned = {row["file"]: row for row in manifest["files"]}
    files = {}
    for name in ("config.json", "model.safetensors.index.json"):
        row, path = pinned[name], metadata / name
        if path.stat().st_size != row["bytes"] or continuation.file_sha(path) != row["sha256"]:
            raise chain.PreconditionError("Pinned metadata file differs: " + name)
        files[name] = dict(bytes=row["bytes"], sha256=row["sha256"])
    if files["config.json"]["sha256"] != CONFIG_SHA:
        raise chain.PreconditionError("Pinned config differs")
    if continuation.file_sha(source / "download.json") != receipt:
        raise chain.PreconditionError("Download receipt differs")
    download = continuation.read_json(source / "download.json")
    if (
        download["status"] != "DOWNLOADED_FROM_PINNED_PUBLIC_REVISION_NOT_YET_CONVERTED"
        or download["model"] != MODEL
        or download["revision"] != REVISION
    ):
        raise chain.PreconditionError("Download receipt identity differs")
    index = continuation.read_json(metadata / "model.safetensors.index.json")["weight_map"]
    rows = {row["name"]: row for row in download["files"]}
    if len(rows) != len(download["files"]) or len(rows) != 15 or set(rows) != set(index.values()):
        raise chain.PreconditionError("Receipt shard inventory differs from the index")
    for name, row in sorted(rows.items()):
        if Path(name).name != name:
            raise chain.PreconditionError("Invalid shard name")
        files[name] = dict(bytes=row["bytes"], sha256=row["observed_sha256"])
    if sum(row["bytes"] for row in rows.values()) != SHARD_BYTES:
        raise chain.PreconditionError("Receipt shard bytes differ")
    receipt_row = dict(bytes=(source / "download.json").stat().st_size, sha256=receipt)
    return files, receipt_row


def choose_snapshot(source, files, extra):
    """Use the shared source directly only when its inventory is exactly the pinned set."""
    source = Path(source)
    entries = {path.name: path for path in source.iterdir()}
    expected = dict(files, **extra)
    regular = all(path.is_file() and not path.is_symlink() for path in entries.values())
    if regular and set(entries) == set(expected):
        return "direct", expected
    return "copied", dict(files)


def copy_verified(origin, target, row):
    """Stream one pinned file into new scratch and verify the copied bytes."""
    digest, size = hashlib.sha256(), 0
    with Path(origin).open("rb") as reader, Path(target).open("xb") as writer:
        for block in iter(lambda: reader.read(8 * 2**20), b""):
            writer.write(block)
            digest.update(block)
            size += len(block)
    if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
        raise chain.PreconditionError("Copied snapshot file differs: " + Path(target).name)


def make_scratch(scratch, mode):
    scratch = Path(scratch)
    if not scratch.is_absolute() or scratch.exists():
        raise chain.PreconditionError("Use a new absolute scratch directory")
    scratch.mkdir()
    marker = dict(
        schema=SCHEMA,
        uuid=str(uuid.uuid4()),
        uid=os.getuid(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hostname=socket.gethostname(),
        created_unix=time.time(),
        training_mode=mode,
        contents="reconstructable speech DCP, snapshot copy and strict checkpoint only",
    )
    continuation.write_json(scratch / MARKER, marker)
    return marker


def stage_dcp(snapshot, expected_source, scratch, copied):
    """Production speech conversion into scratch; returns the stage fields."""
    from pithtrain.tasks.omni_speech_initialization import prepare_speech_initialization

    started = time.monotonic()
    result = prepare_speech_initialization(
        snapshot,
        Path(scratch) / "speech-dcp",
        expected_source=expected_source,
        max_tensor_bytes=MAX_TENSOR_BYTES,
    )
    seconds = time.monotonic() - started
    if result["full_model_replica"] is not False:
        raise AssertionError("Speech conversion must not build a full model replica")
    if copied:
        copy = Path(scratch) / "snapshot"
        if copy.is_symlink() or Path(snapshot) != copy:
            raise AssertionError("Refusing to remove anything but the owned snapshot copy")
        shutil.rmtree(copy)
    dcp = Path(scratch) / "speech-dcp"
    return dict(
        manifest_sha256=result["manifest_sha256"],
        canonical_tensors=result["canonical_tensors"],
        source_kind=result["source_kind"],
        dcp_bytes=sum(p.stat().st_size for p in dcp.rglob("*") if p.is_file()),
        conversion_seconds=seconds,
    )


def stage(args):
    if torch.cuda.is_initialized():
        raise chain.PreconditionError("Staging must not initialize CUDA")
    source = chain.source_identity(args.expected_commit)
    budget = sizes(args.training_mode)
    admission = scratch_admission(args.scratch.parent, budget["scratch_required_bytes"])
    if admission["errors"]:
        raise chain.PreconditionError("; ".join(admission["errors"]))
    files, receipt_row = released_files(args.metadata, args.source, args.download_receipt_sha256)
    mode, expected = choose_snapshot(args.source, files, {"download.json": receipt_row})
    args.output.mkdir()
    marker = make_scratch(args.scratch, args.training_mode)
    snapshot = args.source
    if mode == "copied":
        snapshot = args.scratch / "snapshot"
        snapshot.mkdir()
        for name, row in sorted(files.items()):
            pinned = name in ("config.json", "model.safetensors.index.json")
            copy_verified((args.metadata if pinned else args.source) / name, snapshot / name, row)
    expected_source = dict(
        kind="released",
        model_id=MODEL,
        revision=REVISION,
        config_sha256=CONFIG_SHA,
        files=expected,
    )
    fields = stage_dcp(snapshot, expected_source, args.scratch, mode == "copied")
    report = dict(
        schema=SCHEMA,
        status="PASS_RELEASED_SPEECH_DCP_STAGED",
        source=source,
        training_mode=args.training_mode,
        scratch=str(args.scratch),
        owner_uuid=marker["uuid"],
        slurm_job_id=marker["slurm_job_id"],
        hostname=marker["hostname"],
        snapshot=mode,
        expected_source_sha256=hashlib.sha256(
            json.dumps(expected_source, sort_keys=True).encode()
        ).hexdigest(),
        admission=admission,
        free_bytes_after=shutil.disk_usage(args.scratch).free,
        **fields,
    )
    continuation.write_json(args.output / "stage.json", report)
    return report


def load_stage(path, digest, commit, mode, scratch):
    if continuation.file_sha(path) != digest:
        raise chain.PreconditionError("Stage report checksum differs")
    report = continuation.read_json(path)
    if (
        report["schema"] != SCHEMA
        or report["status"] != "PASS_RELEASED_SPEECH_DCP_STAGED"
        or report["source"]["commit"] != commit
        or report["training_mode"] != mode
        or report["scratch"] != str(scratch)
    ):
        raise chain.PreconditionError("Stage report belongs to another source/mode/scratch")
    marker = continuation.read_json(Path(scratch) / MARKER)
    if marker["uuid"] != report["owner_uuid"]:
        raise chain.PreconditionError("Scratch owner marker differs")
    return report


def cleanup(args):
    """Remove only this job's reconstructable scratch, after its reports exist."""
    report = load_stage(
        args.stage, args.stage_sha256, args.expected_commit, args.training_mode, args.scratch
    )
    scratch = Path(args.scratch)
    marker = continuation.read_json(scratch / MARKER)
    if (
        scratch.is_symlink()
        or not scratch.is_dir()
        or scratch.stat().st_uid != os.getuid()
        or marker["uid"] != os.getuid()
        or marker["slurm_job_id"] != os.environ.get("SLURM_JOB_ID")
        or marker["hostname"] != socket.gethostname()
    ):
        raise chain.PreconditionError("Scratch is not owned by this job; nothing removed")
    removed = sum(
        p.stat().st_size for p in scratch.rglob("*") if p.is_file() and not p.is_symlink()
    )
    args.output.mkdir()
    shutil.rmtree(scratch)
    result = dict(
        schema=SCHEMA,
        status="REMOVED_OWNED_SCRATCH",
        scratch=str(scratch),
        owner_uuid=report["owner_uuid"],
        removed_bytes=removed,
        absent=not scratch.exists(),
    )
    continuation.write_json(args.output / "cleanup.json", result)
    return result


def build_meta_model(config):
    from pithtrain.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
    from pithtrain.models.qwen3_omni_multimodal import Qwen3OmniMoeThinkerMultimodalModel
    from pithtrain.models.qwen3_omni_speech import Qwen3OmniMoeSpeechModel
    from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration

    with torch.device("meta"):
        return Qwen3OmniMoeSpeechModel(
            config,
            Qwen3OmniMoeThinkerMultimodalModel(config.thinker_config, -1),
            Qwen3OmniMoeTalkerForConditionalGeneration(config.talker_config),
            Qwen3OmniMoeCode2Wav(config.code2wav_config),
        )


def configure_contexts(device):
    """The native speech context of omni_speech_chain.build_pair, without its HF model."""
    from torch import nn

    from pithtrain.contexts import distributed, training
    from pithtrain.operators.grouped_linear import GroupedLinear

    distributed.device = device
    distributed.pp_size = distributed.cp_size = distributed.ep_size = 1
    distributed.pp_rank = distributed.cp_rank = distributed.ep_rank = 0
    distributed.pp_group = distributed.cp_group = distributed.ep_group = None
    training.fp8, training.Linear, training.GroupedLinear = False, nn.Linear, GroupedLinear


def process_start():
    return Path("/proc/self/stat").read_text().split()[21]


def state_tree(model, optimizer, scheduler, data, state):
    from pithtrain.tasks import omni_speech as task

    return dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        data=data.state_dict(),
        task=state.state_dict(),
        rng=task._rng_state(),
    )


def logical_rows(rows):
    """Model, optimizer, scheduler and RNG rows; cursors advance even without an update."""
    return [row for row in rows if not row["path"] or row["path"][0][1] not in ("data", "task")]


def write_state(path, tree):
    return continuation.write_rows(path, continuation.state_rows(tree))


def validate_producer(producer, digest, *, rank, mode, source, fixture_sha, stage, world):
    producer = Path(producer)
    if continuation.file_sha(producer / "complete.json") != digest:
        raise chain.PreconditionError("Producer completion checksum differs")
    complete = continuation.read_json(producer / "complete.json")
    expected = dict(
        status="PASS_RELEASED_SPEECH_PRODUCER",
        training_mode=mode,
        source_commit=source["commit"],
        fixture_sha256=fixture_sha,
        stage_manifest_sha256=stage["manifest_sha256"],
        world=world,
    )
    if any(complete.get(key) != value for key, value in expected.items()):
        raise chain.PreconditionError("Producer completion identity differs")
    for item in complete["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise chain.PreconditionError("Unsafe producer file path")
        if continuation.file_sha(producer / relative) != item["sha256"]:
            raise chain.PreconditionError("Producer file changed: " + item["path"])
    prior = continuation.read_json(producer / f"rank-{rank}" / "result.json")
    if prior["status"] != "PASS" or prior["phase"] != "produce" or prior["rank"] != rank:
        raise chain.PreconditionError("Producer rank result differs")
    if (prior["pid"], prior["process_start"]) == (os.getpid(), process_start()):
        raise chain.PreconditionError("Restore must use a fresh process")
    return prior


def observed_scan(context, *, scan, checkpoint_root):
    """One task.launch scan with all-parameter evidence; returns this rank's report."""
    import torch.distributed as dist

    from pithtrain.tasks import omni_speech as task

    c = context
    count = scan + 1
    expected_update = sharded.updates_at(c.mode, count) > sharded.updates_at(c.mode, scan)
    before = {n: support.scan_tensor(p, digest=True) for n, p in c.trainable.items()}
    captured, losses = {}, []
    clip, codec, text = task.clip_sharded_grad_norm, task.speech_loss_sums, task.text_loss_sum

    def record(kind, item):
        values = item.loss_sum.detach().float().reshape(-1).cpu().tolist()
        losses.append(
            dict(
                kind=kind,
                loss_sum=[v if math.isfinite(v) else None for v in values],
                target_count=item.target_count.detach().reshape(-1).cpu().tolist(),
            )
        )
        return item

    def observed_clip(model, maximum, *, group):
        norm = clip(model, maximum, group=group)
        value = float(norm.item())
        captured["grad_norm"] = value if math.isfinite(value) else None
        return norm

    def pre_step(_optimizer, _args, _kwargs):
        captured["gradients"] = {
            n: None if p.grad is None else support.scan_tensor(p.grad)
            for n, p in c.trainable.items()
        }

    hook = c.optimizer.register_step_pre_hook(pre_step)
    task.clip_sharded_grad_norm = observed_clip
    task.speech_loss_sums = lambda *a, **k: record("codec", codec(*a, **k))
    task.text_loss_sum = lambda *a, **k: record("text", text(*a, **k))
    c.records.clear()
    try:
        c.cfg.max_batches, c.cfg.max_updates = count, UPDATES
        seconds = c.seconds["update"] + (c.seconds["save"] if checkpoint_root else 0)
        with continuation.deadline(seconds):
            task.launch(c.cfg, c.data, **c.options, checkpoint_root=checkpoint_root)
    finally:
        hook.remove()
        task.clip_sharded_grad_norm, task.speech_loss_sums, task.text_loss_sum = clip, codec, text
    for item in c.records:
        record("waveform", item)
    after = {n: support.scan_tensor(p, digest=True) for n, p in c.trainable.items()}
    gradients = captured.get("gradients")
    errors, components = [], {name: dict(gradient_nonzero=0, changed=0) for name in ACTIVE[c.mode]}
    rows = []
    for name, value in after.items():
        owner = name.split(".", 1)[0]
        gradient = None if gradients is None else gradients[name]
        changed = before[name]["sha256"] != value["sha256"]
        components[owner]["changed"] += int(changed)
        if not value["finite"] or value["dtype"] != "torch.float32":
            errors.append("nonfinite/non-FP32 parameter " + name)
        if expected_update:
            if gradient is None or not gradient["finite"] or gradient["dtype"] != "torch.float32":
                errors.append("missing/nonfinite/non-FP32 gradient " + name)
            else:
                components[owner]["gradient_nonzero"] += gradient["nonzero_elements"]
        rows.append(dict(key=name, before=before[name], after=value, gradient=gradient))
    if expected_update != (gradients is not None):
        errors.append("optimizer step presence differs from the explicit schedule")
    for item in losses:
        if None in item["loss_sum"] or min(item["target_count"]) < 0:
            errors.append("nonfinite loss sum or negative target count")
    if expected_update and captured.get("grad_norm") is None:
        errors.append("nonfinite or missing gradient norm")
    optimizer_rows = sharded.optimizer_evidence(
        c.trainable, c.optimizer, sharded.updates_at(c.mode, count)
    )
    errors += [
        f"optimizer state {n}" for n, row in optimizer_rows.items() if row["status"] != "PASS"
    ]
    if (c.state.scanned_batches, c.state.completed_updates) != (
        count,
        sharded.updates_at(c.mode, count),
    ):
        errors.append("task cursors differ from the explicit schedule")
    gathered = [None] * c.world
    dist.all_gather_object(gathered, dict(components=components, errors=bool(errors)))
    totals = {
        name: dict(
            gradient_nonzero=sum(g["components"][name]["gradient_nonzero"] for g in gathered),
            changed=sum(g["components"][name]["changed"] for g in gathered),
        )
        for name in ACTIVE[c.mode]
    }
    for name, total in totals.items():
        if expected_update and (total["gradient_nonzero"] <= 0 or total["changed"] <= 0):
            errors.append("component without nonzero gradient or actual update: " + name)
        if not expected_update and total["changed"]:
            errors.append("component changed during a zero-target scan: " + name)
    continuation.write_rows(c.rank_dir / f"update-{count}", rows)
    report = dict(
        scan=count,
        updates=sharded.updates_at(c.mode, count),
        expected_update=expected_update,
        saved_checkpoint=checkpoint_root is not None,
        losses=losses,
        grad_norm=captured.get("grad_norm"),
        components=components,
        global_components=totals,
        optimizer_rows_passed=sum(row["status"] == "PASS" for row in optimizer_rows.values()),
        optimizer_rows=len(optimizer_rows),
        errors=errors,
        status="FAIL" if errors else "PASS",
    )
    continuation.write_json(c.rank_dir / f"scan-{count}.json", report)
    statuses = [None] * c.world
    dist.all_gather_object(statuses, report["status"])
    if any(item != "PASS" for item in statuses):
        raise AssertionError(f"Scan {count} failed on at least one rank; reports retained")
    return report


def snapshot(context, name):
    c = context
    with continuation.deadline(c.seconds["snapshot"]):
        tree = state_tree(c.model, c.optimizer, c.scheduler, c.data, c.state)
        write_state(c.rank_dir / name, tree)
    rows = list(continuation.read_rows(c.rank_dir / name))
    if c.producer is not None:
        continuation.compare_rows(rows, continuation.read_rows(c.producer / c.rank_dir.name / name))
    return rows


def run_phase(phase, **kwargs):
    """produce/restore after the process group exists; any failure leaves failure.json."""
    import torch.distributed as dist

    rank_dir = Path(kwargs["output"]) / f"rank-{dist.get_rank()}"
    rank_dir.mkdir()
    try:
        return _phase_body(phase, **kwargs)
    except BaseException as exc:
        continuation.write_json(rank_dir / "failure.json", chain.failure_report(phase, exc))
        raise


def _phase_body(
    phase,
    *,
    mode,
    config,
    samples,
    fixture_sha,
    stage,
    scratch,
    output,
    source,
    runtime,
    device,
    build_model,
    seconds,
    producer=None,
    producer_sha256=None,
    contexts=True,
):
    """produce/restore body; device may be CPU (Gloo) in tests."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from pithtrain.modules.reproducibility import configure_deterministic_training
    from pithtrain.tasks import omni_speech as task
    from pithtrain.tasks.omni_speech_fsdp import apply_speech_fsdp, materialize_speech_shards
    from pithtrain.tasks.omni_speech_initialization import initialize_speech_from_dcp
    from pithtrain.tasks.sft_lm import SFTState

    assert phase in ("produce", "restore") and (producer is None) == (phase == "produce")
    rank, world = dist.get_rank(), dist.get_world_size()
    scratch, output = Path(scratch), Path(output)
    rank_dir = output / f"rank-{rank}"
    timings = {}
    if producer is not None:
        validate_producer(
            producer,
            producer_sha256,
            rank=rank,
            mode=mode,
            source=source,
            fixture_sha=fixture_sha,
            stage=stage,
            world=world,
        )
    configure_deterministic_training(True)  # Before construction: models capture the layout.
    if contexts:
        configure_contexts(device)
    started = time.monotonic()
    model = build_model(config)
    task.configure_trainable_components(model, mode)
    mesh = init_device_mesh(device.type, (world,), mesh_dim_names=("dp",))
    apply_speech_fsdp(model, mesh, mode=mode)
    materialize_speech_shards(model, device=device)
    local_bytes = sum(
        p.to_local().numel() * p.to_local().element_size() for p in model.parameters()
    )
    local_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    # The initializer requires one rank-uniform staging cap; uneven Shard(0) differs by rank.
    largest = torch.tensor([local_bytes], dtype=torch.int64, device=device)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    local_cap = int(largest.item())
    with continuation.deadline(seconds["initialization"]):
        receipt = initialize_speech_from_dcp(
            model,
            scratch / "speech-dcp",
            expected_manifest_sha256=stage["manifest_sha256"],
            rank=rank,
            world_size=world,
            max_local_bytes=local_cap,
            max_tensor_bytes=MAX_TENSOR_BYTES,
        )
    timings["initialization"] = time.monotonic() - started
    named = dict(model.named_parameters())
    trainable = {n: p for n, p in named.items() if p.requires_grad}
    frozen = {n: p for n, p in named.items() if not p.requires_grad}
    if {n.split(".", 1)[0] for n in trainable} != set(ACTIVE[mode]):
        raise AssertionError("Trainable components differ from the training mode")
    total = sum(p.numel() for p in named.values())
    checkpoint_bytes = 4 * total + 8 * sum(p.numel() for p in trainable.values())
    checkpoint = scratch / "checkpoint"
    if phase == "produce":
        if checkpoint.exists():
            raise chain.PreconditionError("Producer checkpoint root must be new")
        if shutil.disk_usage(scratch).free < math.ceil(1.1 * checkpoint_bytes):
            raise chain.PreconditionError("Scratch cannot hold the strict checkpoint")
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=LR, weight_decay=0.0, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    modulus, _ = code_layout(config)
    data = ReleasedData(samples, fixture_sha, mode, rank, world, modulus)
    state = SFTState()
    cfg = task.OmniSpeechCfg(
        mode=mode,
        max_updates=0,
        max_batches=0,
        max_grad_norm=MAX_NORM,
        waveform_objective_id=OBJECTIVE_ID,
    )
    contract = dict(
        schema=SCHEMA,
        source=source["commit"],
        fixture_sha256=fixture_sha,
        stage_manifest_sha256=stage["manifest_sha256"],
        training_mode=mode,
        dp=world,
        cp=1,
        ep=1,
        pp=1,
        hsdp_replica=1,
        runtime_lock_sha256=runtime.get("runtime_lock_sha256"),
        schedule=dict(scans=sharded.scans(mode), updates=UPDATES, save_scan=SAVE_SCAN),
    )
    records = []

    def objective(wave, batch):
        item = sharded.waveform_objective(wave, batch)
        records.append(item)
        return item

    options = dict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        state=state,
        token_group=mesh.get_group(),
        gradient_reduction="sum",
        waveform_objective=objective,
        checkpoint_contract=contract,
    )
    context = SimpleNamespace(
        mode=mode,
        world=world,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        state=state,
        cfg=cfg,
        options=options,
        trainable=trainable,
        records=records,
        seconds=seconds,
        rank_dir=rank_dir,
        producer=None if producer is None else Path(producer),
    )
    with continuation.deadline(seconds["snapshot"]):
        frozen_initial = {n: support.scan_tensor(p, digest=True) for n, p in frozen.items()}
    continuation.write_json(rank_dir / "frozen-initial.json", frozen_initial)
    if producer is not None:
        prior = continuation.read_json(Path(producer) / rank_dir.name / "frozen-initial.json")
        if prior != json.loads(json.dumps(frozen_initial)):
            raise AssertionError("Frozen initialization differs from the producer's")
    start, states = 0, {}
    if phase == "restore":
        cfg.max_batches, cfg.max_updates = SAVE_SCAN, sharded.updates_at(mode, SAVE_SCAN)
        started = time.monotonic()
        with continuation.deadline(seconds["load"]):
            task.launch(cfg, data, **options, checkpoint_root=checkpoint, resume_scan=SAVE_SCAN)
        timings["load"] = time.monotonic() - started
        if (state.scanned_batches, state.completed_updates) != (
            SAVE_SCAN,
            sharded.updates_at(mode, SAVE_SCAN),
        ):
            raise AssertionError("Restored task cursors differ")
        states[SAVE_SCAN] = snapshot(context, f"state-{SAVE_SCAN}")
        start = SAVE_SCAN
    for scan in range(start, sharded.scans(mode)):
        count = scan + 1
        save = phase == "produce" and count == SAVE_SCAN
        started = time.monotonic()
        report = observed_scan(context, scan=scan, checkpoint_root=checkpoint if save else None)
        timings[f"scan-{count}"] = time.monotonic() - started
        continuation.write_json(rank_dir / f"batch-{count}.json", data.observed[scan])
        if producer is not None:
            prior = continuation.read_json(Path(producer) / rank_dir.name / f"batch-{count}.json")
            if prior != json.loads(json.dumps(data.observed[scan])):
                raise AssertionError(f"Fresh-process input batch {count} differs")
        if count >= SAVE_SCAN:
            states[count] = snapshot(context, f"state-{count}")
        if not report["expected_update"]:
            if count - 1 not in states or logical_rows(states[count - 1]) != logical_rows(
                states[count]
            ):
                raise AssertionError("Zero-target scan changed model/optimizer/scheduler/RNG")
    with continuation.deadline(seconds["snapshot"]):
        frozen_final = {n: support.scan_tensor(p, digest=True) for n, p in frozen.items()}
    if frozen_final != frozen_initial:
        raise AssertionError("A frozen component changed")
    continuation.write_json(rank_dir / "frozen-final.json", frozen_final)
    result = dict(
        status="PASS",
        phase=phase,
        rank=rank,
        world=world,
        training_mode=mode,
        source_commit=source["commit"],
        fixture_sha256=fixture_sha,
        stage_manifest_sha256=stage["manifest_sha256"],
        runtime=runtime,
        pid=os.getpid(),
        process_start=process_start(),
        initialization=receipt,
        trainable_parameters=len(trainable),
        frozen_parameters=len(frozen),
        local_fp32_bytes=local_bytes,
        timings=timings,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
    )
    continuation.write_json(rank_dir / "result.json", result)
    dist.barrier()
    if rank == 0:
        files = [
            dict(path=str(p.relative_to(output)), sha256=continuation.file_sha(p))
            for p in sorted(output.rglob("*"))
            if p.is_file()
        ]
        continuation.write_json(
            output / "complete.json",
            dict(
                status="PASS_RELEASED_SPEECH_PRODUCER"
                if phase == "produce"
                else "PASS_RELEASED_SPEECH_FRESH_RESTORE",
                training_mode=mode,
                row=ROW[mode],
                source_commit=source["commit"],
                fixture_sha256=fixture_sha,
                stage_manifest_sha256=stage["manifest_sha256"],
                world=world,
                files=files,
                hf_parity="NOT_CLAIMED_AT_RELEASED_SIZE",
                codec_labels="SYNTHETIC",
            ),
        )
    dist.barrier()
    return result


def gpu_runtime(args, world, rank):
    import importlib.metadata
    import platform
    import subprocess

    if (
        int(os.environ.get("WORLD_SIZE", "0")) != world
        or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != world
        or int(os.environ.get("LOCAL_RANK", "-1")) != rank
    ):
        raise chain.PreconditionError("Use one single-node torchrun group of the planned DP size")
    if importlib.metadata.version("transformers") != "5.17.0":
        raise chain.PreconditionError("Transformers must be exactly 5.17.0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != world:
        raise chain.PreconditionError("Expose exactly the planned number of CUDA devices")
    if torch.cuda.get_device_name(rank) != args.hardware:
        raise chain.PreconditionError("Explicit hardware lock differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise chain.PreconditionError("Set CUBLAS_WORKSPACE_CONFIG before Python starts")
    observed = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        python=platform.python_version(),
        transformers=importlib.metadata.version("transformers"),
        packages=sorted(
            subprocess.check_output(
                [os.sys.executable, "-m", "pip", "freeze"], text=True, timeout=60
            ).splitlines()
        ),
    )
    lock = chain.verify_runtime_lock(args.runtime_lock, observed)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return dict(
        torch=observed["torch"],
        cuda=observed["cuda"],
        python=observed["python"],
        transformers=observed["transformers"],
        runtime_lock_sha256=lock,
        hardware=args.hardware,
        nvidia_smi=smi.stdout,
    )


def train(args):
    import torch.distributed as dist

    mode, phase = args.training_mode, args.phase
    world, rank = DP[mode], int(os.environ.get("RANK", "-1"))
    if int(os.environ.get("WORLD_SIZE", "0")) != world:
        raise chain.PreconditionError("WORLD_SIZE differs from the planned DP size")
    runtime = gpu_runtime(args, world, rank)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    source = chain.source_identity(args.expected_commit)
    report, samples = load_fixture(args.fixture, args.fixture_sha256, args.expected_commit)
    stage_report = load_stage(
        args.stage, args.stage_sha256, args.expected_commit, mode, args.scratch
    )
    config = released_config(args.metadata)
    budget = sizes(mode)
    resources = support.resource_inventory()
    if (resources["available_bytes"] or 0) < budget["host_available_admission_bytes"]:
        raise chain.PreconditionError("Host memory below the planned admission")
    if torch.cuda.mem_get_info(device)[0] < budget["persistent_bytes_per_gpu"] + GPU_MARGIN:
        raise chain.PreconditionError("Free GPU memory below the persistent state plus margin")
    seconds = dict(
        initialization=args.initialization_seconds,
        save=args.save_seconds,
        load=args.load_seconds,
        update=args.update_seconds,
        snapshot=args.snapshot_seconds,
    )
    timeout = continuation.collective_timeout_seconds(
        phase, seconds["initialization"], seconds["save"], seconds["load"]
    )
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout), device_id=device)
    try:
        if rank == 0:
            args.output.mkdir()
        dist.barrier()
        runtime = dict(runtime, host=resources, fixture_tokens=report["tokens_per_rank_scan"])
        return run_phase(
            phase,
            mode=mode,
            config=config,
            samples=samples,
            fixture_sha=args.fixture_sha256,
            stage=stage_report,
            scratch=args.scratch,
            output=args.output,
            source=source,
            runtime=runtime,
            device=device,
            build_model=build_meta_model,
            seconds=seconds,
            producer=args.producer,
            producer_sha256=args.producer_sha256,
        )
    finally:
        dist.destroy_process_group()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="phase", required=True)
    for name in ("plan", "fixture", "stage", "produce", "restore", "cleanup"):
        item = sub.add_parser(name)
        if name != "fixture":
            item.add_argument("--training-mode", required=True, choices=tuple(DP))
        if name == "plan":
            continue
        item.add_argument("--expected-commit", required=True)
        item.add_argument("--output", type=Path, required=True)
        if name in ("fixture", "stage", "produce", "restore"):
            item.add_argument("--metadata", type=Path, required=True)
        if name == "fixture":
            item.add_argument("--dataset", type=Path, required=True)
        if name == "stage":
            item.add_argument("--source", type=Path, required=True)
            item.add_argument("--download-receipt-sha256", required=True)
        if name in ("stage", "produce", "restore", "cleanup"):
            item.add_argument("--scratch", type=Path, required=True)
        if name in ("produce", "restore", "cleanup"):
            item.add_argument("--stage", type=Path, required=True)
            item.add_argument("--stage-sha256", required=True)
        if name in ("produce", "restore"):
            item.add_argument("--fixture", type=Path, required=True)
            item.add_argument("--fixture-sha256", required=True)
            item.add_argument("--runtime-lock", type=Path, required=True)
            item.add_argument("--hardware", required=True)
            for key, value in SECONDS.items():
                item.add_argument(f"--{key}-seconds", type=int, default=value)
        if name == "restore":
            item.add_argument("--producer", type=Path, required=True)
            item.add_argument("--producer-sha256", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.phase == "plan":
        print(json.dumps(plan(args.training_mode), indent=2))
        return
    if args.phase == "produce":
        args.producer = args.producer_sha256 = None
    function = dict(fixture=fixture, stage=stage, cleanup=cleanup).get(args.phase, train)
    function(args)


if __name__ == "__main__":
    main()
