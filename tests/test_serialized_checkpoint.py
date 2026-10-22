"""Actual two-rank Gloo checkpoint lifetimes and fresh continuation, no GPU claim."""

import gc
import hashlib
import json
import random
import weakref
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.modules import distributed_checkpoint as native
from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
from tests.acceptance.omni_native_resume import equal
from tests.test_distributed_strict_checkpoint import advance, build, initialize_mesh
from tests.test_task_distributed_checkpoint import StatefulTask, commit, envelope


def _snapshot(parts, data):
    values, _ = native._runtime(parts[0], (parts[1],), (parts[2],), data, torch.device("cpu"))
    return native._pack(values)


def _first_tensor(payload):
    return next(iter(payload["model"].values()))["value"]


def _worker(rank, directory, protocol, fault, phase):
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / ('rendezvous-' + phase)}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.set_num_threads(1)
        parts = build(*initialize_mesh("fsdp-layout", rank))
        task = StatefulTask(rank) if protocol == "bounded" else None
        data = (
            envelope(task, optimizer_policy="conditional_adamw_v1", bounded_load=True)
            if task
            else parts[3]
        )
        config = dict(training=dict(global_batch_size=2), protocol=protocol)
        seed = rank + (100 if phase == "produce" else 900)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        def step():
            commit(parts, task, True) if task else advance(parts)

        tracked = None
        reads = 0
        copies = 0
        leases = []
        marker = root / "live-cpu-payload"
        old_pack, old_load = native._pack, torch.load
        old_copy = native._restore_packed_inplace

        def acquire(kind):
            # The marker lives until an actual packed/loaded model tensor dies.
            # A different rank cannot begin staging while that tensor is held.
            with marker.open("x") as stream:
                stream.write(f"{kind}:{rank}")

        def release():
            marker.unlink()

        def track(value):
            tensor = _first_tensor(value)
            leases.append(weakref.ref(tensor))
            weakref.finalize(tensor, release)

        def packed(value):
            owned = tracked == "save" and isinstance(value, dict) and "model" in value
            if owned:
                acquire("save")
            try:
                result = old_pack(value)
            except BaseException:
                if owned:
                    release()
                raise
            if owned:
                track(result)
            return result

        location = root / "good" / "torch-dcp" / "00000001"

        def loaded(path, *args, **kwargs):
            nonlocal reads
            owned = tracked == "load" and Path(path).parent == location
            if not owned:
                return old_load(path, *args, **kwargs)
            reads += 1
            acquire("load")
            try:
                if rank == 1 and reads == 2 and fault in {"second-read", "manifest-rewrite"}:
                    assert (root / "rank-zero-copied").exists()
                    changed = old_load(path, *args, **kwargs)
                    _first_tensor(changed["state"]).add_(1)
                    torch.save(changed, path)
                    del changed
                    if fault == "manifest-rewrite":
                        manifest = json.loads((location / native.MANIFEST).read_text())
                        manifest["files"][Path(path).name] = hashlib.sha256(
                            Path(path).read_bytes()
                        ).hexdigest()
                        (location / native.MANIFEST).write_text(json.dumps(manifest))
                result = old_load(path, *args, **kwargs)
            except BaseException:
                release()
                raise
            track(result["state"])
            return result

        def copied(saved, template, live=None):
            nonlocal copies
            if isinstance(template, torch.Tensor):
                copies += 1
                if rank == 0:
                    (root / "rank-zero-copied").touch()
            return old_copy(saved, template, live)

        native._pack, torch.load, native._restore_packed_inplace = packed, loaded, copied
        if phase == "produce":
            step()
            boundary = _snapshot(parts, data)
            tracked = "save"
            save_checkpoint(root / "good", 1, data_state=data, strict=True, resume_config=config)
            tracked = None
            gc.collect()
            dist.barrier()
            assert not marker.exists() and all(ref() is None for ref in leases)
            raw = old_load(location / f"rank-{rank:05d}.pt", weights_only=True)
            assert set(raw) == {"header", "state", "rng"} | (
                {"task", "optimizer_schema"} if task else set()
            )
            assert raw["header"]["version"] == (
                native.TASK_STATE_VERSION if task else native.VERSION
            )
            del raw
            step()
            torch.save(
                dict(boundary=boundary, after=_snapshot(parts, data)), root / f"expected-{rank}.pt"
            )
        else:
            expected = old_load(root / f"expected-{rank}.pt", weights_only=True)
            if fault:
                # Nontrivial live state makes partial bounded copies observable.
                step()
                step()
            before = _snapshot(parts, data)
            if fault == "last-rank-preflight" and rank == 0:
                path = location / "rank-00001.pt"
                value = old_load(path, weights_only=True)
                entry = next(iter(value["state"]["model"].values()))
                entry["value"] = entry["value"].double()
                torch.save(value, path)
                manifest = json.loads((location / native.MANIFEST).read_text())
                manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                (location / native.MANIFEST).write_text(json.dumps(manifest))
                del value
            dist.barrier()
            tracked = "load"
            if fault:
                with pytest.raises(native.DistributedCheckpointError):
                    load_checkpoint(
                        root / "good", 1, data_state=data, strict=True, resume_config=config
                    )
                tracked = None
                equal(_snapshot(parts, data), before)
                if fault == "last-rank-preflight":
                    assert copies == 0 and not (root / "rank-zero-copied").exists()
            else:
                assert not parts[1].state
                load_checkpoint(
                    root / "good", 1, data_state=data, strict=True, resume_config=config
                )
                tracked = None
                assert reads == 2
                equal(_snapshot(parts, data), expected["boundary"])
                step()
                equal(_snapshot(parts, data), expected["after"])
            gc.collect()
            dist.barrier()
            assert not marker.exists() and all(ref() is None for ref in leases)
            assert not torch.cuda.is_initialized()
            (root / f"result-{rank}.json").write_text(
                json.dumps(
                    dict(status="PASS", protocol=protocol, fault=fault, reads=reads, copies=copies)
                )
            )
        native._pack, torch.load, native._restore_packed_inplace = old_pack, old_load, old_copy
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "protocol,fault",
    [
        ("legacy", None),
        ("bounded", None),
        ("bounded", "last-rank-preflight"),
        ("bounded", "second-read"),
        ("bounded", "manifest-rewrite"),
    ],
)
def test_two_rank_serialized_payload_lifetimes_and_fresh_restore(tmp_path, protocol, fault):
    mp.spawn(_worker, args=(str(tmp_path), protocol, fault, "produce"), nprocs=2, join=True)
    mp.spawn(_worker, args=(str(tmp_path), protocol, fault, "restore"), nprocs=2, join=True)
    assert all(
        json.loads((tmp_path / f"result-{rank}.json").read_text())["status"] == "PASS"
        for rank in range(2)
    )
