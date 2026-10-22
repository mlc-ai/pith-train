"""Released Thinker initialization and strict two-update continuation.

No downloader, allocation, persistent store, or cleanup API. CPU prepare consumes
already downloaded public files. Produce and restore are separate torchrun jobs.
The original random-weight reference gates remain separate requirements.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from tests.acceptance import omni_released_update_support as original

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
METADATA_SHA = "4753ef18b35604e727ba2f3e3501ab8a8c277233dd892b68a9df4e70e421e886"
BLOCK = 2**20
ROWS_PER_FILE = 128


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_rows(path, rows):
    """Keep every row; split complete text proof into bounded individual files."""
    path = Path(path)
    path.mkdir()
    chunk, files, total = [], [], 0

    def flush():
        nonlocal chunk, total
        name = f"rows-{len(files):05d}.json"
        write_json(path / name, chunk)
        if (path / name).stat().st_size > 2 * 2**20:
            raise ValueError("Complete evidence chunk exceeds 2 MiB; do not truncate")
        files.append(dict(name=name, rows=len(chunk), sha256=file_sha(path / name)))
        total += len(chunk)
        chunk = []

    for row in rows:
        chunk.append(row)
        if len(chunk) == ROWS_PER_FILE:
            flush()
    if chunk:
        flush()
    write_json(path / "index.json", dict(schema=1, rows=total, files=files))
    return file_sha(path / "index.json")


def read_rows(path):
    path = Path(path)
    index = read_json(path / "index.json")
    assert index["schema"] == 1
    count = 0
    for i, row in enumerate(index["files"]):
        assert row["name"] == f"rows-{i:05d}.json"
        item = path / row["name"]
        assert not item.is_symlink() and file_sha(item) == row["sha256"]
        values = read_json(item)
        assert len(values) == row["rows"] <= ROWS_PER_FILE
        count += len(values)
        yield from values
    assert count == index["rows"]
    assert {p.name for p in path.iterdir()} == {r["name"] for r in index["files"]} | {"index.json"}


def tensor_digest(value):
    """All local bytes, at most one bounded CPU chunk; no full state clone."""
    import torch

    from pithtrain.modules.distributed_checkpoint import _describe

    local = original.local_tensor(value).detach()
    assert local.is_contiguous(), "Unexpected layout; refuse an unbounded contiguous copy"
    flat = local.reshape(-1)
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), BLOCK):
        chunk = flat[offset : offset + BLOCK]
        assert not chunk.is_floating_point() or torch.isfinite(chunk).all().item()
        digest.update(chunk.cpu().view(torch.uint8).numpy().tobytes())
    return dict(layout=_describe(value), local_shape=list(local.shape), sha256=digest.hexdigest())


def state_rows(value, path=()):
    """Typed complete state traversal: container shape/order and every leaf count."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        yield dict(path=path, type="tensor", **tensor_digest(value))
    elif isinstance(value, np.ndarray):
        assert value.dtype != object
        yield dict(
            path=path,
            type="ndarray",
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
    elif isinstance(value, dict):
        keys = [[type(k).__name__, k] for k in value]
        assert all(type(k) in (str, int) for k in value)
        yield dict(path=path, type="dict", keys=keys)
        for key in value:
            yield from state_rows(value[key], (*path, [type(key).__name__, key]))
    elif isinstance(value, (list, tuple)):
        yield dict(path=path, type=type(value).__name__, length=len(value))
        for i, item in enumerate(value):
            yield from state_rows(item, (*path, ["index", i]))
    else:
        assert type(value) in (type(None), str, int, bool, float), type(value)
        assert type(value) is not float or math.isfinite(value)
        yield dict(path=path, type=type(value).__name__, value=value)


def compare_rows(actual, expected):
    from itertools import zip_longest

    missing = object()
    for a, b in zip_longest(actual, expected, fillvalue=missing):
        assert a is not missing and b is not missing, "Complete state row count differs"
        # JSON normalizes tuples in shape/path metadata; never normalizes leaf types.
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (
            "Exact continuation state mismatch",
            a.get("path"),
            b.get("path"),
        )


def canonical_parts(key):
    """Independent pinned split-HF -> native Thinker mapping, no inverse converter."""
    if key.startswith(("visual.", "audio_tower.", "lm_head.")):
        base = "thinker." + key
    else:
        base = "thinker.model." + key
    if re.search(r"\.experts\.\d+\.gate_up_proj$", base):
        prefix = base.removesuffix("gate_up_proj")
        return [prefix + "gate_proj.weight", prefix + "up_proj.weight"]
    if re.search(r"\.experts\.\d+\.down_proj$", base):
        return [base + ".weight"]
    return [base]


def expected_fp32_record(key, shape, headers, source):
    import torch

    digest, elements = hashlib.sha256(), 0
    parts = canonical_parts(key)
    for part in parts:
        row = headers[part]
        with (source / row["file"]).open("rb") as stream:
            stream.seek(row["offset"])
            left = row["bytes"]
            while left:
                raw = bytearray(stream.read(min(2 * BLOCK, left)))
                assert len(raw) == min(2 * BLOCK, left)
                values = torch.frombuffer(raw, dtype=torch.bfloat16).float()
                assert torch.isfinite(values).all().item(), part
                digest.update(values.numpy().tobytes())
                elements += values.numel()
                left -= len(raw)
    assert elements == math.prod(shape), key
    return dict(
        key=key,
        shape=list(shape),
        dtype="torch.float32",
        sha256=digest.hexdigest(),
        public_keys=parts,
    )


def inspect_initialization_dcp(root, expected_shapes):
    """Strict inventory preflight before the public initializer's strict=False setter."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    location = Path(root) / "torch-dcp/00000000"
    assert not (location / "strict-manifest.json").exists()
    metadata = FileSystemReader(location).read_metadata().state_dict_metadata
    assert set(metadata) == {"app.model." + k for k in expected_shapes}
    for key, shape in expected_shapes.items():
        item = metadata["app.model." + key]
        assert tuple(item.size) == tuple(shape), key
        assert item.properties.dtype == torch.bfloat16, key
    return dict(keys=len(expected_shapes), metadata_sha256=file_sha(location / ".metadata"))


def verify_loaded_parameters(expected, output):
    """Gather one native parameter at a time; compare all selected released values."""
    import torch.distributed as dist

    from pithtrain.modules.checkpoint import iter_canonical_parameters

    seen, failures = set(), []

    def rows():
        for key, value in iter_canonical_parameters():
            seen.add(key)
            target = expected.get(key)
            record = tensor_digest(value)
            ok = (
                target is not None
                and record["layout"]["dtype"] == target["dtype"]
                and list(value.shape) == target["shape"]
                and record["sha256"] == target["sha256"]
            )
            if not ok:
                failures.append(key)
            yield dict(key=key, status="PASS" if ok else "FAIL", **record)

    write_rows(output, rows())
    all_seen, all_failures = [None] * dist.get_world_size(), [None] * dist.get_world_size()
    dist.all_gather_object(all_seen, sorted(seen))
    dist.all_gather_object(all_failures, failures)
    assert set().union(*(set(keys) for keys in all_seen)) == set(expected)
    assert not any(all_failures), all_failures


@contextmanager
def deadline(seconds):
    assert type(seconds) is int and seconds > 0

    def expired(*_):
        raise TimeoutError(f"Original phase deadline {seconds}s reached")

    previous_remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert interval == 0
    started = time.monotonic()
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(
        signal.ITIMER_REAL, min(seconds, previous_remaining) if previous_remaining else seconds
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_remaining:
            remaining = max(0.000001, previous_remaining - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining)


def collective_timeout_seconds(phase, initialization, save, load):
    """Owner-only I/O must fit the admitted cap while peers wait in agreement."""
    assert phase in ("produce", "restore")
    assert all(type(v) is int and v > 0 for v in (initialization, save, load))
    return max(600, initialization, save if phase == "produce" else load) + 10


def source_identity(expected):
    repo = Path(__file__).resolve().parents[2]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert re.fullmatch(r"[0-9a-f]{40}", expected) and actual == expected
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    return repo


def convert_thinker(source, root):
    """The existing public converter; config objects intentionally use setters."""
    from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, hf2dcp

    cfg = ConvertCheckpointCfg()
    cfg.operation = "hf2dcp"
    cfg.load_path, cfg.save_path = source, root / "torch-dcp/00000000"
    cfg.omni_multimodal = cfg.omni_thinker_only = True
    hf2dcp(cfg, logging.getLogger("released-continuation"))


def prepare(args):
    """Fresh CPU process: validate all cached/public bytes, convert, seal inventory."""
    import torch

    from pithtrain.tasks.convert_checkpoint import qwen3_omni_thinker as thinker

    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    assert file_sha(args.metadata / "manifest.json") == METADATA_SHA
    metadata = read_json(args.metadata / "manifest.json")
    assert metadata["model"] == MODEL and metadata["revision"] == original.REVISION
    for row in metadata["files"]:
        name = row["file"]
        assert Path(name).name == name
        path = args.metadata / name
        assert path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"]
    for name in ("config.json", "model.safetensors.index.json"):
        assert (args.source / name).read_bytes() == (args.metadata / name).read_bytes()
    assert file_sha(args.source / "config.json") == original.CONFIG_SHA
    assert file_sha(args.source / "download.json") == args.download_receipt_sha256
    download = read_json(args.source / "download.json")
    assert download["status"] == "DOWNLOADED_FROM_PINNED_PUBLIC_REVISION_NOT_YET_CONVERTED"
    assert download["model"] == MODEL and download["revision"] == original.REVISION
    assert download["application_retries"] == download["private_uploads"] == 0
    index = read_json(args.source / "model.safetensors.index.json")["weight_map"]
    observed = {row["name"]: row for row in download["files"]}
    assert len(observed) == len(download["files"]) == 15
    assert set(observed) == set(index.values())
    headers = {}
    for name in sorted(observed):
        assert Path(name).name == name
        path = args.source / name
        assert not path.is_symlink() and path.stat().st_size == observed[name]["bytes"]
        assert file_sha(path) == observed[name]["observed_sha256"]
        with path.open("rb") as stream:
            n = struct.unpack("<Q", stream.read(8))[0]
            assert 0 < n <= 16 * 2**20
            raw = stream.read(n)
        assert raw == (args.metadata / (name + ".header.json")).read_bytes()
        position = 0
        ranges = []
        for key, item in json.loads(raw).items():
            if key == "__metadata__":
                continue
            assert key not in headers and item["dtype"] == "BF16"
            start, end = item["data_offsets"]
            assert end - start == 2 * math.prod(item["shape"])
            headers[key] = dict(
                file=name, offset=8 + n + start, bytes=end - start, shape=item["shape"]
            )
            ranges.append((start, end))
        for start, end in sorted(ranges):
            assert start == position
            position = end
        assert path.stat().st_size == 8 + n + position
    assert {k: v["file"] for k, v in headers.items()} == index and len(index) == 28010
    assert sum(r["bytes"] for r in observed.values()) == 70523299202
    resources = original.resource_inventory()
    assert not original.resource_errors(resources)
    # Conversion's known engineering estimate is not a guaranteed peak; enforce
    # cgroup and leave OOM a failure, never silently try a smaller configuration.
    assert resources["available_bytes"] >= 230 * original.GIB
    assert shutil.disk_usage(args.work.parent).free >= 256 * original.GIB
    args.work.mkdir()
    args.output.mkdir()
    config = thinker.thinker_config(read_json(args.source / "config.json"))
    shapes = thinker.parameter_shapes(config)
    excluded = sorted(k for k in headers if not k.startswith("thinker."))
    expected_index = write_rows(
        args.output / "expected",
        (
            expected_fp32_record(k, shape, headers, args.source)
            for k, shape in sorted(shapes.items())
        ),
    )
    convert_thinker(args.source, args.work / "initialization")
    dcp = inspect_initialization_dcp(args.work / "initialization", shapes)
    (args.work / "model").mkdir()
    shutil.copyfile(args.source / "config.json", args.work / "model/config.json")
    write_json(
        args.output / "prepared.json",
        dict(
            status="PASS_RELEASED_THINKER_PREPARE_ONLY",
            source_commit=args.expected_commit,
            revision=original.REVISION,
            model=MODEL,
            config_sha256=original.CONFIG_SHA,
            metadata_sha256=METADATA_SHA,
            download_receipt_sha256=file_sha(args.source / "download.json"),
            publisher_lfs_verification=download["official_lfs_sha256"],
            public_files=download["files"],
            source_directory=str(args.source.resolve()),
            work_directory=str(args.work.resolve()),
            expected_index_sha256=expected_index,
            excluded_keys=excluded,
            full_component_conversion="SEPARATE_REQUIRED_GATE",
            dcp=dcp,
            numerical="NOT_RUN",
            checkpoint="NOT_RUN",
        ),
    )


def capture_state(data, device):
    from pithtrain.contexts import training
    from pithtrain.modules.distributed_checkpoint import _runtime

    return _runtime(
        training.model, training.optimizers, training.schedulers, data.checkpoint_state, device
    )[0]


def continuation(phase, cfg, data, device, output, producer, train_one):
    """Same transaction body exercised by the real CPU public-init regression."""
    import torch.distributed as dist

    from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint

    rank = dist.get_rank()
    contract = cfg["resume_contract"]
    checkpoint = cfg["checkpoint"]
    if phase == "produce":
        train_one(0)
        write_rows(output / f"boundary-{rank}", state_rows(capture_state(data, device)))
        with deadline(cfg["save_seconds"]):
            save_checkpoint(
                checkpoint, 1, strict=True, data_state=data.checkpoint_state, resume_config=contract
            )
    else:
        with deadline(cfg["load_seconds"]):
            load_checkpoint(
                checkpoint, 1, strict=True, data_state=data.checkpoint_state, resume_config=contract
            )
        compare_rows(
            state_rows(capture_state(data, device)), read_rows(producer / f"boundary-{rank}")
        )
        write_rows(output / f"boundary-{rank}", state_rows(capture_state(data, device)))
    train_one(1)
    if phase == "restore":
        compare_rows(state_rows(capture_state(data, device)), read_rows(producer / f"after-{rank}"))
    write_rows(output / f"after-{rank}", state_rows(capture_state(data, device)))


def observed_step(cfg, data, step, parameters, groups, output):
    import torch.distributed as dist

    from pithtrain.contexts import distributed, training
    from pithtrain.tasks import pretrain_lm

    rank, world = dist.get_rank(), dist.get_world_size()
    report = dict(status="RUNNING", step=step + 1, rank=rank, parameters={}, errors=[])
    optimizer = training.optimizers[0]
    old_step, old_batch = training.model.step, data.get_batch

    def batch(*args, **kwargs):
        result = old_batch(*args, **kwargs)
        # Includes actual sample IDs, labels, positions and full media tensor bytes.
        write_rows(
            output / f"batch-{step}-{rank}",
            state_rows(
                [
                    dict(
                        model_inputs=m.model_inputs,
                        objective_inputs=m.objective_inputs,
                        model_context=m.model_context,
                        media_inputs=m.media_inputs,
                        cu_seqlens=m.cu_seqlens,
                        sample_ids=m.sample_ids,
                    )
                    for m in result
                ]
            ),
        )
        return result

    def forward(*args, **kwargs):
        values = old_step(*args, **kwargs)
        report["objective_loss_sums"] = [float(v.detach().item()) for v in values]
        assert len(values) == (4 if distributed.pp_rank == 0 else 0)
        assert all(math.isfinite(v) for v in report["objective_loss_sums"])
        return values

    def before(opt, _args, _kwargs):
        assert (not opt.state) if step == 0 else bool(opt.state)
        failures = []
        for name, p in parameters.items():
            grad = original.scan_tensor(p.grad) if p.grad is not None else None
            value = original.scan_tensor(p, digest=True)
            report["parameters"][name] = dict(before=value, gradient=grad)
            if (
                not value["finite"]
                or value["dtype"] != "torch.float32"
                or grad is None
                or not grad["finite"]
                or grad["dtype"] != "torch.float32"
            ):
                failures.append(name)
        all_failures = [None] * world
        dist.all_gather_object(all_failures, failures)
        assert not any(all_failures), all_failures

    def after(opt, _args, _kwargs):
        for name, p in parameters.items():
            row = report["parameters"][name]
            row["after"] = original.scan_tensor(p, digest=True)
            row["changed"] = row["before"]["sha256"] != row["after"]["sha256"]
            state = opt.state.get(p, {})
            row["optimizer_state"] = dict(
                step=float(state["step"].item()) if "step" in state else None,
                **{
                    k: original.scan_tensor(state[k])
                    for k in ("exp_avg", "exp_avg_sq")
                    if k in state
                },
            )

    pre, post = optimizer.register_step_pre_hook(before), optimizer.register_step_post_hook(after)
    training.model.step, data.get_batch = forward, batch
    try:
        with deadline(600):
            pretrain_lm.train_step(cfg, data, step)
        assert data.consumed_samples == 16 * (step + 1)
        assert all(p.grad is None for p in parameters.values())
        gathered = [None] * world
        dist.all_gather_object(gathered, dict(rank=rank, parameters=report["parameters"]))
        if step == 0:
            report["components"], report["errors"] = original.validate_rows(gathered, groups)
        else:
            # Same all-parameter first-update requirements, with the required
            # second-update counter checked directly rather than rewriting rows.
            for item in gathered:
                for name, row in item["parameters"].items():
                    assert row["optimizer_state"]["step"] == 2, name
                    for field in ("before", "after", "gradient"):
                        assert row[field]["finite"] and row[field]["dtype"] == "torch.float32", name
                    for key in ("exp_avg", "exp_avg_sq"):
                        v = row["optimizer_state"][key]
                        assert v["finite"] and v["dtype"] == "torch.float32", name
            for prefix in groups:
                rows = [
                    row
                    for item in gathered
                    for name, row in item["parameters"].items()
                    if name.startswith(prefix)
                ]
                assert rows and sum(r["gradient"]["nonzero_elements"] for r in rows) > 0
                assert any(r["changed"] for r in rows)
        assert not report["errors"], report["errors"]
        report["status"] = "PASS"
    finally:
        pre.remove()
        post.remove()
        training.model.step, data.get_batch = old_step, old_batch
        rows = report.pop("parameters")
        write_rows(output / f"update-{step}-{rank}", (dict(key=k, **v) for k, v in rows.items()))
        write_json(output / f"update-{step}-{rank}.json", report)


def run_gpu(args):
    import importlib.metadata
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from pithtrain.contexts import training
    from pithtrain.modules.checkpoint import initialize_checkpoint, to_canonical_model
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.distributed_checkpoint import resume_contract
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import (
        make_adamw_optimizer,
        make_constant_scheduler,
        setup_training,
    )
    from pithtrain.tasks import pretrain_lm
    from tests.acceptance.omni_full_media import full_inventory, official_config

    assert os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1"
    assert importlib.metadata.version("transformers") == "5.17.0"
    assert importlib.metadata.version("safetensors") == "0.8.0"
    assert torch.__version__ == "2.13.0+cu130" and torch.get_default_dtype() == torch.float32
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8 and 0 <= rank < 8
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    assert "H200" in torch.cuda.get_device_name()
    assert file_sha(args.prepared / "prepared.json") == args.prepared_sha256
    prepared = read_json(args.prepared / "prepared.json")
    assert prepared["status"] == "PASS_RELEASED_THINKER_PREPARE_ONLY"
    assert prepared["source_commit"] == args.expected_commit
    assert (
        prepared["revision"] == original.REVISION
        and prepared["config_sha256"] == original.CONFIG_SHA
    )
    assert file_sha(args.prepared / "expected/index.json") == prepared["expected_index_sha256"]
    work = Path(prepared["work_directory"])
    assert not args.output.resolve().is_relative_to(work.resolve())
    assert not work.resolve().is_relative_to(args.output.resolve())
    assert file_sha(work / "model/config.json") == original.CONFIG_SHA
    expected_records = list(read_rows(args.prepared / "expected"))
    expected_all = {row["key"]: row for row in expected_records}
    assert len(expected_all) == len(expected_records)
    canonical_shapes = {k: v["shape"] for k, v in expected_all.items()}
    if args.phase == "produce":
        observed_dcp = inspect_initialization_dcp(work / "initialization", canonical_shapes)
        assert observed_dcp == prepared["dcp"]
    resources, plan = original.resource_inventory(), original.estimate(args.modality)
    assert not original.resource_errors(resources)
    # Keep room for the serialized cold rollback/payload plus allocator overhead.
    assert resources["available_bytes"] >= 192 * original.GIB
    disk_required = (12 * plan["parameters"] if args.phase == "produce" else 0) + 16 * original.GIB
    assert shutil.disk_usage(work).free >= disk_required
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    stage = 0 if rank < 4 else 1
    assert torch.cuda.mem_get_info()[0] >= plan[f"pp{stage}_initial_free_gpu_admission_bytes"]
    cfg = pretrain_lm.PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size, cfg.distributed.expert_parallel_size = 2, 4
    cfg.distributed.context_parallel_size = cfg.distributed.hsdp_replica = 1
    cfg.distributed.timeout = timedelta(
        seconds=collective_timeout_seconds(
            args.phase, args.initialization_seconds, args.save_seconds, args.load_seconds
        )
    )
    t = cfg.training
    t.model = work / "model"
    setattr(
        t,
        {"image": "native_vision", "audio": "native_audio", "video": "native_video"}[args.modality],
        True,
    )
    t.fp8, t.moe_load_balance_coef, t.benchmark = False, 0.0, False
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.lr, t.seed, t.max_steps = 1e-4, 1234, 2
    t.sequence_length, t.global_batch_size, t.micro_batch_size = args.sequence_length, 16, 1
    t.save_interval, t.save_location = None, work / ("checkpoint-" + args.modality)
    cfg.data.dataset, cfg.data.format, cfg.data.modalities = (
        args.dataset,
        "prepared_bundle",
        (args.modality,),
    )
    cfg.data.num_workers = 0
    setup_logging(cfg)
    setup_distributed(cfg)
    if rank == 0:
        args.output.mkdir()
    dist.barrier()
    write_json(
        args.output / f"runtime-{rank}.json",
        dict(
            source_commit=args.expected_commit,
            rank=rank,
            gpu_name=torch.cuda.get_device_name(),
            free_gpu_bytes=free_gpu,
            total_gpu_bytes=total_gpu,
            host=resources,
            distributions={
                d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
            },
            torch_cuda=torch.version.cuda,
            torch_git=torch.version.git_version,
            original_update_seconds=600,
            initialization_seconds=args.initialization_seconds,
            save_seconds=args.save_seconds,
            load_seconds=args.load_seconds,
            phase_seconds=args.phase_seconds,
            world=8,
            pp=2,
            ep=4,
            cp=1,
            attention_dp=4,
            expert_dp=1,
            global_batch=16,
            micro_batch=1,
            weights="pinned_public_released",
            hf_parity="SEPARATE_REQUIRED_GATE",
            full_speech="NOT_RUN",
        ),
    )
    data = pretrain_lm.setup_dataset(cfg)
    record_count = data.bundle["statistics"]["train"][args.modality]["samples"]
    required = 32 if args.data_epoch_policy == "single-epoch" else 16
    assert record_count >= required and data.epoch_samples >= required
    data_contract = dict(
        epoch_policy=args.data_epoch_policy,
        actual_source_records=record_count,
        data_configuration=cfg.data.to_json_dict(),
        recipe=data.recipe,
        bundle_sha256=file_sha(args.dataset / "bundle.json"),
        fingerprint=data.fingerprint,
        epoch_samples=data.epoch_samples,
        sampler_seed=data.seed,
        weights=data.weights,
        modalities=list(data.modalities),
        stage=data.stage,
        batch_configuration=data.batch_cfg,
        global_batch=data.global_batch_size,
        micro_batch=data.micro_batch_size,
        sequence_length=data.sequence_length,
        attention_dp=4,
        context_parallel=1,
        sample_ids="COMPLETE_PER_STEP_BATCH_RECORDS",
        records_fabricated=0,
    )
    if rank == 0:
        write_json(args.output / "data-contract.json", data_contract)
    config = official_config(work / "model/config.json", Path(__file__).resolve().parents[2])
    shapes, groups = (
        full_inventory(config, args.modality),
        original.component_groups(config, args.modality),
    )
    with deadline(args.initialization_seconds):
        setup_training(cfg)
        parameters = {}
        for module in training.model.module:
            for name, value in module.named_parameters():
                key = original.canonical(name)
                assert key not in parameters and key in shapes
                shape = list(shapes[key])
                if ".experts." in key:
                    shape[0] //= 4
                assert list(value.shape) == shape
                parameters[key] = value
        inventories = [None] * 8
        dist.all_gather_object(
            inventories,
            dict(
                keys=sorted(parameters),
                elements=sum(original.local_tensor(p).numel() for p in parameters.values()),
            ),
        )
        assert set().union(*(set(v["keys"]) for v in inventories)) == set(shapes)
        assert sum(v["elements"] for v in inventories) == plan["parameters"]
        for i, row in enumerate(inventories):
            assert row["elements"] == plan[f"pp{0 if i < 4 else 1}_sharded_elements"]
        assert len(training.optimizers) == 1
        original.inspect_production_adamw(training.optimizers[0], parameters, expected_lr=t.lr)
        # Advertised native canonical keys must be an exact selected-modal projection.
        local_keys = set(to_canonical_model(training.model.state_dict(), training.model))
        all_keys = [None] * 8
        dist.all_gather_object(all_keys, sorted(local_keys))
        selected = {
            k: v
            for k, v in expected_all.items()
            if not k.startswith("visual." if args.modality == "audio" else "audio_tower.")
        }
        assert set().union(*(set(v) for v in all_keys)) == set(selected)
        if args.phase == "produce":
            initialize_checkpoint(work / "initialization", 0)
            assert not training.optimizers[0].state and data.consumed_samples == 0
            verify_loaded_parameters(selected, args.output / f"initialized-{rank}")
    producer = args.producer
    if args.phase == "restore":
        assert file_sha(producer / "complete.json") == args.producer_sha256
        receipt = read_json(producer / "complete.json")
        assert (
            receipt["source_commit"] == args.expected_commit
            and receipt["modality"] == args.modality
        )
        assert receipt["status"] == "PASS_RELEASED_PRODUCER_TWO_UPDATES"
        assert receipt["prepared_sha256"] == args.prepared_sha256
        for item in receipt["files"]:
            relative = Path(item["path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            assert file_sha(producer / item["path"]) == item["sha256"]
        prior_rank = read_json(producer / f"rank-{rank}.json")
        identity = (os.getpid(), Path("/proc/self/stat").read_text().split()[21])
        assert identity != (prior_rank["pid"], prior_rank["process_start"]), (
            "Restore must use a fresh process"
        )
    contract = resume_contract(cfg)
    contract["released_data"] = data_contract
    continuation(
        args.phase,
        dict(
            resume_contract=contract,
            checkpoint=t.save_location,
            save_seconds=args.save_seconds,
            load_seconds=args.load_seconds,
        ),
        data,
        device,
        args.output,
        producer,
        lambda step: observed_step(cfg, data, step, parameters, groups, args.output),
    )
    if args.phase == "restore":
        compare_rows(
            read_rows(args.output / f"batch-1-{rank}"), read_rows(producer / f"batch-1-{rank}")
        )
    write_json(
        args.output / f"rank-{rank}.json",
        dict(
            status="PASS",
            rank=rank,
            phase=args.phase,
            source_commit=args.expected_commit,
            pid=os.getpid(),
            process_start=Path("/proc/self/stat").read_text().split()[21],
            prepared_sha256=args.prepared_sha256,
            weight_source="pinned_public_released_weights",
            updates=[1, 2] if args.phase == "produce" else [2],
        ),
    )
    dist.barrier()
    if rank == 0:
        files = [
            dict(path=str(p.relative_to(args.output)), sha256=file_sha(p))
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        ]
        write_json(
            args.output / "complete.json",
            dict(
                status="PASS_RELEASED_PRODUCER_TWO_UPDATES"
                if args.phase == "produce"
                else "PASS_RELEASED_FRESH_RESTORE_NEXT_UPDATE",
                source_commit=args.expected_commit,
                modality=args.modality,
                prepared_sha256=args.prepared_sha256,
                files=files,
                hf_parity="SEPARATE_REQUIRED_GATE",
                full_speech="NOT_RUN",
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "produce", "restore"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--download-receipt-sha256")
    for name in ("source", "metadata", "work", "prepared", "producer", "dataset"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("prepared", "producer"):
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--modality", choices=tuple(original.ENCODERS))
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--data-epoch-policy", choices=("single-epoch", "native-rollover"))
    for name in ("phase", "initialization", "save", "load"):
        parser.add_argument("--" + name + "-seconds", type=int, required=True)
    args = parser.parse_args()
    repo = source_identity(args.expected_commit)
    assert all(
        getattr(args, name + "_seconds") > 0 for name in ("phase", "initialization", "save", "load")
    )
    assert not args.output.exists()
    protected = [
        repo,
        *(
            getattr(args, k)
            for k in ("source", "metadata", "dataset", "prepared", "producer")
            if getattr(args, k) is not None
        ),
    ]
    for path in protected:
        assert not args.output.resolve().is_relative_to(path.resolve())
        assert not path.resolve().is_relative_to(args.output.resolve())
    if args.phase == "prepare":
        assert args.download_receipt_sha256 and re.fullmatch(
            r"[0-9a-f]{64}", args.download_receipt_sha256
        )
        assert all(getattr(args, k) is not None for k in ("source", "metadata", "work"))
        assert not args.work.exists()
        assert not args.output.resolve().is_relative_to(args.work.resolve())
        assert not args.work.resolve().is_relative_to(args.output.resolve())
        assert args.phase_seconds >= 600 + 1500 + 900
        for path in protected:
            assert not args.work.resolve().is_relative_to(path.resolve())
            assert not path.resolve().is_relative_to(args.work.resolve())
    else:
        assert args.modality and args.dataset and args.prepared and args.prepared_sha256
        assert args.data_epoch_policy is not None
        assert 0 < args.sequence_length <= 512
        if args.phase == "restore":
            assert args.producer and args.producer_sha256
            assert args.phase_seconds >= args.initialization_seconds + args.load_seconds + 600
        else:
            assert args.phase_seconds >= args.initialization_seconds + args.save_seconds + 1200
    with deadline(args.phase_seconds):
        prepare(args) if args.phase == "prepare" else run_gpu(args)


if __name__ == "__main__":
    main()
