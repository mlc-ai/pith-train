"""Real two-process Gloo/DTensor protocol tests, not native GPU acceptance.

The small CPU model and cursor isolate rank-local storage, uneven expert shards,
collective rejection and rollback. The separate GPU entrance uses production
Omni, DualPipeV/FSDP and the real PR100 loader without these protocol fixtures.
"""

import hashlib
import json
import random
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from pithtrain.contexts import distributed, training
from pithtrain.modules.checkpoint import find_checkpoint, load_checkpoint, save_checkpoint
from pithtrain.modules.distributed_checkpoint import (
    MANIFEST,
    DistributedCheckpointError,
    _pack,
    _runtime,
)
from tests.acceptance.omni_native_resume import equal


class Cursor:
    """Protocol-only CPU state; never described as a real data-loader test."""

    def __init__(self):
        self.consumed = 0
        self.pending = self.fail_apply = self.malformed = False

    def state_dict(self):
        if self.pending:
            raise ValueError("uncommitted batch")
        value = dict(version=1, fingerprint="cpu-protocol", consumed_samples=self.consumed)
        if self.malformed:
            del value["fingerprint"]
        return value

    def load_state_dict(self, value):
        self.consumed = value["consumed_samples"]
        if self.fail_apply:
            self.fail_apply = False
            raise RuntimeError("injected one-rank apply failure")


def initialize_mesh(profile, rank):
    pp, ep = (2, 1) if profile == "pp" else (1, 2) if profile == "ep" else (1, 1)
    distributed.rank, distributed.world_size = rank, 2
    distributed.device = torch.device("cpu")
    distributed.attn_mesh = init_device_mesh(
        "cpu", (pp, 2 // pp, 1), mesh_dim_names=("pp", "dp", "cp")
    )
    distributed.expt_mesh = init_device_mesh(
        "cpu", (pp, 2 // (pp * ep), ep), mesh_dim_names=("pp", "dp", "ep")
    )
    distributed.pp_size, distributed.ep_size, distributed.cp_size = pp, ep, 1
    distributed.dp_size = 2 // pp
    distributed.pp_rank = distributed.attn_mesh.get_local_rank("pp")
    distributed.dp_rank = distributed.attn_mesh.get_local_rank("dp")
    distributed.ep_rank = distributed.expt_mesh.get_local_rank("ep")
    distributed.cp_rank = 0
    return distributed.attn_mesh["dp"], distributed.expt_mesh["dp"]


def build(attn_mesh, expt_mesh):
    model = nn.Module()
    model.module = nn.Sequential(nn.Module())
    part = model.module[0]
    part.register_parameter(
        "weight",
        nn.Parameter(
            distribute_tensor(
                torch.arange(10).reshape(5, 2).float() / 100 + 0.1, attn_mesh, [Shard(0)]
            )
        ),
    )
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts_per_rank = 3
    layer.mlp.experts = nn.Module()
    for name, shape in (("gate_up_proj", (3, 4, 2)), ("down_proj", (3, 2, 2))):
        values = torch.arange(np.prod(shape)).reshape(shape).float() / 100 + 0.1
        layer.mlp.experts.register_parameter(
            name, nn.Parameter(distribute_tensor(values, expt_mesh, [Shard(0)]))
        )
    part.layers = nn.ModuleDict({str(distributed.pp_rank): layer})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    sch = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.8)
    training.model, training.optimizers, training.schedulers = model, (opt,), (sch,)
    return model, opt, sch, Cursor()


def advance(parts):
    model, opt, sch, data = parts
    loss = sum((p.square().sum()).full_tensor() for p in model.parameters())
    loss.backward()
    opt.step()
    sch.step()
    opt.zero_grad(set_to_none=True)
    data.consumed += 2
    random.random()
    np.random.random()
    torch.rand(3)


def snapshot(parts):
    model, opt, sch, data = parts
    values, _ = _runtime(model, (opt,), (sch,), data, torch.device("cpu"))
    return _pack(values)


def mutate(root, case):
    location = root / "torch-dcp/00000001"
    path = location / "rank-00001.pt"
    if case == "manifest-bool":
        manifest = json.loads((location / MANIFEST).read_text())
        manifest["step"] = True
        (location / MANIFEST).write_text(json.dumps(manifest))
        return
    if case == "missing-rank":
        path.unlink()
        return
    if case == "missing-completion":
        (location / MANIFEST).unlink()
        return
    if case == "checksum":
        path.write_bytes(b"truncated")
        return
    payload = torch.load(path, weights_only=True)
    state = payload["state"]
    key = next(iter(state["model"]))
    if case == "header-bool":
        payload["header"]["step"] = True
    elif case == "missing-model":
        del state["model"][key]
    elif case == "extra-model":
        state["model"]["surprise"] = state["model"][key]
    elif case == "shape":
        state["model"][key]["value"] = torch.ones(1)
    elif case == "dtype":
        state["model"][key]["value"] = state["model"][key]["value"].double()
    elif case.startswith("moment-"):
        moment = state["optimizer"][0]["state"][0]["exp_avg"]
        if case == "moment-shape":
            moment["value"] = torch.ones(1)
        elif case == "moment-dtype":
            moment["value"] = moment["value"].double()
        elif case == "moment-mesh":
            moment["tensor"]["mesh"] = [999]
        elif case == "moment-nonfinite":
            moment["value"].flatten()[0] = float("nan")
        else:
            raise AssertionError(case)
    elif case == "topology":
        payload["header"]["topology"]["ep_size"] = 999
    elif case == "optimizer-order":
        state["optimizer"][0]["param_groups"][0]["params"].reverse()
    elif case == "missing-rng":
        del payload["rng"]["torch_cpu"]
    elif case == "cursor":
        state["data"]["consumed_samples"] = 3
    else:
        raise AssertionError(case)
    torch.save(payload, path)
    manifest = json.loads((location / MANIFEST).read_text())
    manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (location / MANIFEST).write_text(json.dumps(manifest))


def worker(rank, root, profile, mode):
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / ('rendezvous-' + mode)}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(1)
        attn, expt = initialize_mesh(profile, rank)
        parts = build(attn, expt)
        data = parts[-1]
        contract = dict(training=dict(global_batch_size=2), protocol="CPU-only", profile=profile)
        seed = (100 if mode == "save" else 900) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if mode == "save":
            advance(parts)
            save_checkpoint(root / "good", 1, data_state=data, strict=True, resume_config=contract)
            assert find_checkpoint(root / "good") == 1
            boundary = snapshot(parts)
            advance(parts)
            torch.save(dict(boundary=boundary, after=snapshot(parts)), root / f"expected-{rank}.pt")
        else:
            assert not parts[1].state
            load_checkpoint(root / "good", 1, data_state=data, strict=True, resume_config=contract)
            expected = torch.load(root / f"expected-{rank}.pt", weights_only=True)
            equal(snapshot(parts), expected["boundary"])
            advance(parts)
            equal(snapshot(parts), expected["after"])
            cases = (
                "manifest-bool",
                "header-bool",
                "missing-rank",
                "missing-completion",
                "checksum",
                "missing-model",
                "extra-model",
                "shape",
                "dtype",
                "topology",
                "optimizer-order",
                "moment-shape",
                "moment-dtype",
                "moment-mesh",
                "moment-nonfinite",
                "missing-rng",
                "cursor",
            )
            for case in cases:
                target = root / case
                if rank == 0:
                    shutil.copytree(root / "good", target)
                    mutate(target, case)
                dist.barrier()
                before = snapshot(parts)
                try:
                    load_checkpoint(target, 1, data_state=data, strict=True, resume_config=contract)
                except DistributedCheckpointError:
                    pass
                else:
                    raise AssertionError(f"Accepted {case}")
                equal(snapshot(parts), before)
            before = snapshot(parts)
            data.malformed = rank == 1
            with pytest.raises(DistributedCheckpointError, match="load contract"):
                load_checkpoint(
                    root / "good", 1, data_state=data, strict=True, resume_config=contract
                )
            data.malformed = False
            equal(snapshot(parts), before)
            before = snapshot(parts)
            data.fail_apply = rank == 1
            with pytest.raises(DistributedCheckpointError, match="apply strict checkpoint"):
                load_checkpoint(
                    root / "good", 1, data_state=data, strict=True, resume_config=contract
                )
            equal(snapshot(parts), before)
            data.pending = rank == 1
            with pytest.raises(DistributedCheckpointError, match="uncommitted"):
                save_checkpoint(
                    root / "uncommitted", 2, data_state=data, strict=True, resume_config=contract
                )
            data.pending = False
            assert not (root / "uncommitted/torch-dcp/00000002" / MANIFEST).exists()
            if rank == 1:
                parts[1].state.clear()
            with pytest.raises(DistributedCheckpointError, match="optimizer parameter state"):
                save_checkpoint(
                    root / "missing-optimizer",
                    2,
                    data_state=data,
                    strict=True,
                    resume_config=contract,
                )
            assert not (root / "missing-optimizer/torch-dcp/00000002" / MANIFEST).exists()
            distributed.cp_size = 2 if rank == 1 else 1
            with pytest.raises(DistributedCheckpointError, match="CP=1"):
                load_checkpoint(
                    root / "good", 1, data_state=data, strict=True, resume_config=contract
                )
            distributed.cp_size = 1
            assert not torch.cuda.is_initialized()
            (root / f"result-{rank}.json").write_text(
                json.dumps(
                    dict(
                        status="PASS",
                        profile=profile,
                        corruption_cases=len(cases),
                        apply_failure_rollback=True,
                        uncommitted_rejected=True,
                        cuda_initialized=False,
                    )
                )
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("profile", ["fsdp-layout", "pp", "ep"])
def test_real_gloo_dtensor_fresh_process_and_failures(tmp_path, profile):
    mp.spawn(worker, args=(str(tmp_path), profile, "save"), nprocs=2, join=True)
    mp.spawn(worker, args=(str(tmp_path), profile, "restore"), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f"result-{rank}.json").read_text())["status"] == "PASS"


def test_gpu_driver_rejects_unschedulable_mesh(tmp_path):
    from tests.acceptance.omni_distributed_resume import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "prepare",
                "--dataset",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--world-size",
                "8",
                "--pp",
                "2",
                "--ep",
                "2",
            ]
        )


def test_gpu_negative_requires_a_named_fault(tmp_path):
    from tests.acceptance.omni_distributed_resume import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "negative",
                "--dataset",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--world-size",
                "2",
            ]
        )


def test_gpu_combined_mesh_can_be_prepared_without_cuda(tmp_path):
    from tests.acceptance.omni_distributed_resume import parse_args

    args = parse_args(
        [
            "--mode",
            "prepare",
            "--dataset",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--world-size",
            "4",
            "--pp",
            "2",
            "--ep",
            "2",
        ]
    )
    assert args.world_size == 4 and args.pp == args.ep == 2
    assert not torch.cuda.is_initialized()


def _native_lr_factory(factor):
    return lambda step: 1.0 if step < 2 else factor


def scheduler_contract_worker(rank, directory):
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'lambda-rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.set_num_threads(1)
        meshes = initialize_mesh("fsdp-layout", rank)

        def make_parts():
            model, optimizer, _, data = build(*meshes)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _native_lr_factory(0.5))
            training.schedulers = (scheduler,)
            return model, optimizer, scheduler, data

        parts = make_parts()
        contract = dict(test="strict-lambda-v1", training=dict(global_batch_size=2))
        advance(parts)
        save_checkpoint(
            root / "lambda", 1, data_state=parts[3], strict=True, resume_config=contract
        )
        advance(parts)
        expected = snapshot(parts)
        restored = make_parts()
        if rank == 1:
            restored[2].lr_lambdas = [_native_lr_factory(0.1)]
        before = snapshot(restored)
        with pytest.raises(DistributedCheckpointError, match="configuration"):
            load_checkpoint(
                root / "lambda", 1, data_state=restored[3], strict=True, resume_config=contract
            )
        equal(snapshot(restored), before)
        restored[2].lr_lambdas = [_native_lr_factory(0.5)]
        load_checkpoint(
            root / "lambda", 1, data_state=restored[3], strict=True, resume_config=contract
        )
        advance(restored)
        equal(snapshot(restored), expected)
        assert restored[2].last_epoch == 2
        assert all(value["step"].item() == 2 for value in restored[1].state.values())
    finally:
        dist.destroy_process_group()


def test_native_lambda_semantics_collective_rejection_and_exact_next_update(tmp_path):
    mp.spawn(scheduler_contract_worker, args=(str(tmp_path),), nprocs=2, join=True)
