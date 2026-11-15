"""Real CPU public model initialization and fresh-process strict continuation.

The distributed fixture is deliberately tiny: this verifies driver/checkpoint
mechanics, never released-weight values, native GPU math or full-size resources.
"""

import hashlib
import json
import logging
import os
import random
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from pithtrain.contexts import logging as log_context
from pithtrain.modules.checkpoint import initialize_checkpoint, iter_canonical_parameters
from pithtrain.modules.distributed_checkpoint import DistributedCheckpointError
from tests.acceptance import omni_released_continuation as driver
from tests.acceptance.omni_native_resume import equal
from tests.test_distributed_strict_checkpoint import (
    FINGERPRINT,
    advance,
    build,
    initialize_mesh,
    snapshot,
)


def test_complete_typed_streamed_state_and_tampering(tmp_path):
    value = {
        1: torch.arange(32).reshape(8, 4).float(),
        "1": (1, [2]),
        "bf16": torch.tensor([1.5], dtype=torch.bfloat16),
        "rng": np.arange(4, dtype=np.uint32),
    }
    root = tmp_path / "state"
    driver.write_rows(root, driver.state_rows(value))
    driver.compare_rows(driver.state_rows(value), driver.read_rows(root))
    with pytest.raises(AssertionError):
        driver.compare_rows(
            driver.state_rows({"1": value[1], 1: value["1"]}), driver.read_rows(root)
        )
    value[1][0, 0] = 20
    with pytest.raises(AssertionError, match="state mismatch"):
        driver.compare_rows(driver.state_rows(value), driver.read_rows(root))
    chunk = root / "rows-00000.json"
    chunk.write_text("[]")
    with pytest.raises(AssertionError):
        list(driver.read_rows(root))


def test_nested_deadline_keeps_outer_remaining_limit():
    with driver.deadline(2):
        with driver.deadline(1):
            time.sleep(0.01)
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0 < remaining < 2 and interval == 0
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_independent_split_mapping_reads_every_value(tmp_path):
    source = tmp_path / "weights.bin"
    gate = torch.arange(6).reshape(3, 2).bfloat16()
    up = gate + 10
    source.write_bytes(
        gate.view(torch.uint8).numpy().tobytes() + up.view(torch.uint8).numpy().tobytes()
    )
    key = "layers.0.mlp.experts.2.gate_up_proj"
    names = driver.canonical_parts(key)
    assert names == [
        "thinker.model.layers.0.mlp.experts.2.gate_proj.weight",
        "thinker.model.layers.0.mlp.experts.2.up_proj.weight",
    ]
    headers = {
        n: dict(file=source.name, offset=i * 12, bytes=12, shape=[3, 2])
        for i, n in enumerate(names)
    }
    record = driver.expected_fp32_record(key, (6, 2), headers, tmp_path)
    expected = torch.cat((gate, up)).float().numpy().tobytes()
    assert record["sha256"] == hashlib.sha256(expected).hexdigest()
    with pytest.raises(AssertionError):
        driver.expected_fp32_record(key, (5, 2), headers, tmp_path)


class Data:
    """The driver's view of step-indexed data: only its static fingerprint."""

    def __init__(self, fingerprint):
        self.data_fingerprint = fingerprint


def _worker(rank, directory, phase):
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / ('init-' + phase)}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        torch.set_num_threads(1)
        log_context.stdout = logging.getLogger("public-init-cpu-test")
        parts = build(*initialize_mesh("fsdp-layout", rank))
        model, optimizer, _, _ = parts
        seed = rank + (400 if phase == "produce" else 700)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        output = root / phase
        if rank == 0:
            output.mkdir()
        dist.barrier()
        if phase == "produce":
            # This is an actual canonical BF16 DCP payload, not a mocked loader.
            canonical = {k: v.detach().cpu().bfloat16() for k, v in iter_canonical_parameters()}
            expected = {
                k: dict(
                    key=k,
                    shape=list(v.shape),
                    dtype="torch.float32",
                    sha256=driver.tensor_digest(v.float())["sha256"],
                )
                for k, v in canonical.items()
            }
            if rank == 0:
                dcp.save(
                    {"app": {"model": canonical}},
                    checkpoint_id=root / "model-only/torch-dcp/00000000",
                    no_dist=True,
                )
            dist.barrier()
            shapes = {k: v.shape for k, v in canonical.items()}
            driver.inspect_initialization_dcp(root / "model-only", shapes)
            with pytest.raises(AssertionError):
                driver.inspect_initialization_dcp(root / "model-only", dict(shapes, surprise=(1,)))
            with torch.no_grad():
                for p in model.parameters():
                    p.to_local().fill_(-6)
            before_rng = torch.get_rng_state().clone()
            initialize_checkpoint(root / "model-only", 0)
            assert not optimizer.state
            assert torch.equal(torch.get_rng_state(), before_rng)
            driver.verify_loaded_parameters(expected, output / f"initialized-{rank}")
            bad = dict(expected)
            key = next(iter(bad))
            bad[key] = dict(bad[key], sha256="0" * 64)
            with pytest.raises(AssertionError):
                driver.verify_loaded_parameters(bad, output / f"negative-readback-{rank}")
        config = dict(
            checkpoint=root / "strict",
            save_seconds=40,
            load_seconds=40,
            resume_contract={"training": {"global_batch_size": 2}},
        )
        steps = []

        def train_one(step):
            # Step-indexed: the driver hands each update its step; no cursor exists.
            steps.append(step)
            advance(parts)

        def run(data):
            driver.continuation(
                phase, config, data, torch.device("cpu"), output, root / "produce", train_one
            )

        if phase == "restore":
            # Another data identity is rejected on every rank before any state changes.
            before = snapshot(parts)
            with pytest.raises(DistributedCheckpointError, match="data fingerprint differs"):
                run(Data("other-data"))
            equal(snapshot(parts), before)
            assert steps == []
        run(Data(FINGERPRINT))
        # Restore resumes at step 1, the update after the checkpoint, not at step 0.
        assert steps == ([0, 1] if phase == "produce" else [1])
        if phase == "produce":
            # The strict shard records exactly the live data fingerprint, never a cursor.
            shard = root / f"strict/torch-dcp/00000001/rank-{rank:05d}.pt"
            stored = torch.load(shard, weights_only=True)["state"]["data"]
            assert stored == {"fingerprint": FINGERPRINT}
        assert not torch.cuda.is_initialized()
        (output / f"result-{rank}.json").write_text(json.dumps(dict(status="PASS", steps=steps)))
    finally:
        dist.destroy_process_group()


def test_real_public_init_then_fresh_process_strict_next_update(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path), "produce"), nprocs=2, join=True)
    mp.spawn(_worker, args=(str(tmp_path), "restore"), nprocs=2, join=True)
    for phase in ("produce", "restore"):
        for rank in range(2):
            assert (
                json.loads((tmp_path / phase / f"result-{rank}.json").read_text())["status"]
                == "PASS"
            )


def test_inner_deadline_cannot_extend_outer_timeout():
    with pytest.raises(TimeoutError):
        with driver.deadline(1):
            with driver.deadline(2):
                time.sleep(1.2)
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_collective_timeout_covers_serialized_owner_io_budget():
    assert driver.collective_timeout_seconds("produce", 200, 1200, 300) == 1210
    assert driver.collective_timeout_seconds("restore", 200, 300, 1500) == 1510
    assert driver.collective_timeout_seconds("restore", 800, 300, 400) == 810
    with pytest.raises(AssertionError):
        driver.collective_timeout_seconds("restore", 800, 300, 0)


def test_actual_tiny_hf_public_converter_config(tmp_path):
    from tests import test_omni_thinker_converter as tiny

    config = tiny.configuration()
    state = tiny.source_state(config, "thinker.", "split", torch.bfloat16)
    source, converted = tmp_path / "hf", tmp_path / "native"
    tiny.fixtures.write_hf(source, config, state, sharded=True)
    driver.convert_thinker(source, converted)
    result = driver.inspect_initialization_dcp(converted, tiny.parameter_shapes(config))
    assert result["keys"] == len(tiny.parameter_shapes(config))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("length", [512, 2048])
def test_prepared_sequence_contract_preserves_original_recipe(tmp_path, length):
    bundle = {"recipe": {"batch": {"max_length": 2048, "max_video_frames": 64}}}
    path = tmp_path / "bundle.json"
    before = json.dumps(bundle).encode()
    path.write_bytes(before)
    result = driver.validate_dataset_sequence_length(tmp_path, length)
    assert result["prepared_max_length"] == 2048
    assert result["requested_sequence_length"] == length
    assert result["truncation"] is result["preprocessing_overrides"] is False
    assert result["gpu_peak_memory"] == "NOT_ESTABLISHED_BY_SEQUENCE_CONTRACT"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "declared,requested",
    [(512, 513), (2048, 2049), (4096, 2048), (True, 1), (2048, True), (2048, 0), (2048, 2048.0)],
)
def test_prepared_sequence_contract_rejects_unbound_length(tmp_path, declared, requested):
    (tmp_path / "bundle.json").write_text(
        json.dumps({"recipe": {"batch": {"max_length": declared}}})
    )
    with pytest.raises(AssertionError):
        driver.validate_dataset_sequence_length(tmp_path, requested)


def fake_host(monkeypatch, files, cpus):
    """Serve /proc and /sys reads from files and pin the visible CPU affinity."""
    monkeypatch.setattr(driver.original, "read_optional_text", lambda path: files.get(str(path)))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cpus)))


def test_resource_inventory_reads_slurm_ancestor_cgroups(monkeypatch):
    # A Slurm task without a cgroup namespace: the cgroup root and the task cgroup hold no
    # limit, the job and user cgroups carry the allocation, and CPUs are bound by affinity.
    gib = driver.original.GIB
    job = "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_3056658"
    mounts = [
        "26 1 259:1 / / rw,relatime shared:1 - ext4 /dev/root rw",
        "35 24 0:30 / /sys/fs/cgroup rw,nosuid,nodev shared:9 - cgroup2 cgroup2 rw,nsdelegate",
        "512 26 0:52 / /mnt/weka rw,relatime shared:300 - virtiofs fs-0 rw",
    ]
    files = {
        "/proc/self/cgroup": "0::/system.slice/slurmstepd.scope/job_3056658/step_0/user/task_0",
        "/proc/self/mountinfo": "\n".join(mounts),
        "/proc/meminfo": "MemTotal:       1651351256 kB\nMemAvailable:   1637442888 kB",
        f"{job}/memory.max": str(384 * gib),
        f"{job}/memory.current": str(gib),
        f"{job}/step_0/memory.max": "max",
        f"{job}/step_0/user/memory.max": str(384 * gib),
        f"{job}/step_0/user/memory.current": str(gib // 2),
        f"{job}/step_0/user/task_0/memory.max": "max",
        f"{job}/step_0/user/task_0/cpu.max": "max 100000",
    }
    fake_host(monkeypatch, files, cpus=32)
    resources = driver.original.resource_inventory()
    assert [row["path"] for row in resources["memory_limits"]] == [job, f"{job}/step_0/user"]
    assert resources["memory_limit_bytes"] == 384 * gib
    assert resources["memory_current_bytes"] == gib
    assert resources["available_bytes"] == 383 * gib
    assert resources["cpu_quota"] is None and resources["cpu_limit"] == 32
    assert driver.original.resource_errors(resources) == []


def test_resource_inventory_keeps_namespaced_container_limits(monkeypatch):
    gib = driver.original.GIB
    files = {
        "/proc/self/cgroup": "0::/",
        "/proc/self/mountinfo": "30 25 0:26 / /sys/fs/cgroup rw,nosuid - cgroup2 cgroup rw",
        "/proc/meminfo": "MemAvailable:   1048576000 kB",
        "/sys/fs/cgroup/memory.max": str(256 * gib),
        "/sys/fs/cgroup/memory.current": str(10 * gib),
        "/sys/fs/cgroup/memory.peak": str(12 * gib),
        "/sys/fs/cgroup/cpu.max": "1600000 100000",
    }
    fake_host(monkeypatch, files, cpus=64)
    resources = driver.original.resource_inventory()
    assert resources["memory_limit_bytes"] == 256 * gib
    assert resources["cgroup_lifetime_peak_bytes"] == 12 * gib
    assert resources["available_bytes"] == 246 * gib
    assert resources["cpu_quota"] == 16 and resources["cpu_limit"] == 16
    assert driver.original.resource_errors(resources) == []


def test_resource_inventory_reads_cgroup_v1_slurm_hierarchy(monkeypatch):
    gib = driver.original.GIB
    task = "/slurm/uid_4688/job_7/step_0/task_0"
    memory, cpu = "/sys/fs/cgroup/memory", "/sys/fs/cgroup/cpu,cpuacct"
    mounts = [
        f"31 25 0:27 / {memory} rw,nosuid - cgroup cgroup rw,memory",
        f"32 25 0:28 / {cpu} rw,nosuid - cgroup cgroup rw,cpu,cpuacct",
    ]
    files = {
        "/proc/self/cgroup": f"9:memory:{task}\n4:cpu,cpuacct:{task}",
        "/proc/self/mountinfo": "\n".join(mounts),
        "/proc/meminfo": "MemAvailable:   1048576000 kB",
        f"{memory}/slurm/uid_4688/job_7/memory.limit_in_bytes": str(384 * gib),
        f"{memory}/slurm/uid_4688/job_7/memory.usage_in_bytes": str(2 * gib),
        f"{memory}/slurm/uid_4688/job_7/memory.max_usage_in_bytes": str(3 * gib),
        f"{memory}{task}/memory.limit_in_bytes": "9223372036854771712",
        f"{memory}{task}/memory.usage_in_bytes": str(gib),
        f"{cpu}{task}/cpu.cfs_quota_us": "-1",
        f"{cpu}{task}/cpu.cfs_period_us": "100000",
    }
    fake_host(monkeypatch, files, cpus=32)
    resources = driver.original.resource_inventory()
    assert resources["memory_limit_bytes"] == 384 * gib
    assert resources["cgroup_lifetime_peak_bytes"] == 3 * gib
    assert resources["available_bytes"] == 382 * gib
    assert resources["cpu_quota"] is None and resources["cpu_limit"] == 32
    assert driver.original.resource_errors(resources) == []


def test_resource_errors_reject_missing_limit_usage_or_cpu_bound(monkeypatch):
    gib = driver.original.GIB
    job = "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_9"
    files = {
        "/proc/self/cgroup": "0::/system.slice/slurmstepd.scope/job_9/step_0/user/task_0",
        "/proc/self/mountinfo": "35 24 0:30 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw",
        "/proc/meminfo": "MemAvailable:   1637442888 kB",
    }
    fake_host(monkeypatch, files, cpus=128)
    resources = driver.original.resource_inventory()
    assert resources["memory_limit_bytes"] is None and resources["cpu_limit"] == 128
    assert driver.original.resource_errors(resources) == [
        "A readable finite host memory hard limit in [192,512] GiB is required",
        "Kernel CPU quota/affinity capacity in (0,32] is required",
    ]
    # A finite limit without its usage counter must not fall back to host-wide free memory.
    files[f"{job}/memory.max"] = str(384 * gib)
    fake_host(monkeypatch, files, cpus=32)
    resources = driver.original.resource_inventory()
    assert resources["memory_limit_bytes"] == 384 * gib and resources["available_bytes"] == 0
    assert driver.original.resource_errors(resources) == [
        "At least 192 GiB actual available cgroup/host RAM is required"
    ]
