"""Native two-rank CPU task checkpoint protocol, not GPU model acceptance."""

import copy
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

from pithtrain.modules import distributed_checkpoint as native
from pithtrain.modules.checkpoint import find_checkpoint, load_checkpoint, save_checkpoint
from pithtrain.modules.task_checkpoint import TaskCheckpointState
from tests.acceptance.omni_native_resume import equal
from tests.test_distributed_strict_checkpoint import build, initialize_mesh


class StatefulTask:
    """Concrete local provider with variable cache and an independent RNG stream."""

    def __init__(self, rank):
        self.rank = rank
        self.requests = self.updates = self.data_cursor = 0
        self.cache = []
        self.generator = torch.Generator().manual_seed(400 + rank)
        self.pending = self.fail_apply = False

    def state_dict(self):
        if self.pending:
            raise ValueError("pending task transaction")
        return dict(
            schema=1,
            requests=self.requests,
            updates=self.updates,
            data_cursor=self.data_cursor,
            policy_version=self.updates,
            provider=dict(
                schema=1, rank=self.rank, cache=list(self.cache), rng=self.generator.get_state()
            ),
        )

    def validate_state_dict(self, state):
        if set(state) != {
            "schema",
            "requests",
            "updates",
            "data_cursor",
            "policy_version",
            "provider",
        }:
            raise ValueError("task schema")
        if type(state["schema"]) is not int or state["schema"] != 1:
            raise ValueError("task schema version")
        p = state["provider"]
        if (
            set(p) != {"schema", "rank", "cache", "rng"}
            or p["schema"] != 1
            or p["rank"] != self.rank
            or len(p["cache"]) != state["requests"]
            or state["data_cursor"] != state["requests"] * (self.rank + 2)
        ):
            raise ValueError("provider schema/rank/cursor")
        torch.Generator().set_state(p["rng"])

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.cache = list(state["provider"]["cache"])
        self.generator.set_state(state["provider"]["rng"])
        if self.fail_apply:
            self.fail_apply = False
            raise RuntimeError("injected provider restore failure after mutation")
        self.requests, self.updates, self.data_cursor = (
            state["requests"],
            state["updates"],
            state["data_cursor"],
        )


def envelope(task, **changes):
    options = dict(
        task_kind="omni_grpo",
        task_schema=1,
        contract={"algorithm": "cpu-fixture", "data": "rank-sharded-v1"},
        transaction_key="requests",
        update_key="updates",
        data_key="data_cursor",
        policy_version_key="policy_version",
    )
    options.update(changes)
    return TaskCheckpointState(task, **options)


def commit(parts, task, update):
    model, opt, scheduler, _ = parts
    if update:
        loss = sum(p.square().sum().full_tensor() for p in model.parameters())
        loss.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        task.updates += 1
    task.requests += 1
    task.data_cursor += task.rank + 2
    task.cache.append(torch.rand((), generator=task.generator).item())
    random.random()
    np.random.random()
    torch.rand(3)


def snapshot(parts, adapter):
    model, opt, sch, _ = parts
    values, _ = native._runtime(model, (opt,), (sch,), adapter, torch.device("cpu"))
    return native._pack(values)


def corrupt(root, case):
    folder = root / "torch-dcp/00000003"
    path = folder / "rank-00001.pt"
    payload = torch.load(path, weights_only=True)
    task = payload["task"]
    if case == "extra":
        task["extra"] = 1
    elif case == "kind":
        task["task_kind"] = "sft"
    elif case == "fingerprint":
        task["fingerprint"] = "0" * 64
    elif case == "schema-bool":
        task["task_schema"] = True
    elif case == "rng-schema":
        task["rng_metadata"]["version"] = True
    elif case == "transaction":
        task["cursors"]["transaction"] += 1
    elif case == "cursor-bool":
        task["cursors"]["updates"] = True
    elif case == "policy-version":
        task["provider_policy_version"] = 2
    elif case == "provider-rank":
        task["state"]["provider"]["rank"] = 999
    elif case == "provider-rng":
        task["state"]["provider"]["rng"] = torch.ones(1, dtype=torch.uint8)
    elif case == "nonfinite":
        task["state"]["provider"]["cache"][0] = float("nan")
    elif case == "rank-global-update":
        task["cursors"]["updates"] = task["provider_policy_version"] = 2
        task["state"]["updates"] = task["state"]["policy_version"] = 2
        for opt in payload["state"]["optimizer"]:
            for item in opt["state"].values():
                item["step"]["value"].fill_(2)
        for scheduler in payload["state"]["scheduler"]:
            scheduler["last_epoch"] = 2
    elif case == "update-counter":
        payload["state"]["optimizer"][0]["state"][0]["step"]["value"].fill_(3)
    else:
        raise AssertionError(case)
    torch.save(payload, path)
    manifest = json.loads((folder / native.MANIFEST).read_text())
    manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / native.MANIFEST).write_text(json.dumps(manifest))


def worker(rank, root, mode):
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / ('rendezvous-' + mode)}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        torch.set_num_threads(1)
        parts = build(*initialize_mesh("fsdp-layout", rank))
        task = StatefulTask(rank)
        adapter = envelope(task)
        config = {"protocol": "task-v2-cpu", "hsdp_replica": 1}  # No fictitious GB.
        seed = (100 if mode == "save" else 900) + rank
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if mode == "save":
            commit(parts, task, False)
            save_checkpoint(root / "good", 1, data_state=adapter, strict=True, resume_config=config)
            assert not parts[1].state and parts[2].last_epoch == 0
            zero = snapshot(parts, adapter)
            commit(parts, task, True)
            commit(parts, task, False)
            save_checkpoint(root / "good", 3, data_state=adapter, strict=True, resume_config=config)
            boundary = snapshot(parts, adapter)
            commit(parts, task, True)
            torch.save(
                dict(zero=zero, boundary=boundary, after=snapshot(parts, adapter)),
                root / f"expected-{rank}.pt",
            )
        else:
            expected = torch.load(root / f"expected-{rank}.pt", weights_only=True)
            load_checkpoint(root / "good", 1, data_state=adapter, strict=True, resume_config=config)
            equal(snapshot(parts, adapter), expected["zero"])
            assert not parts[1].state and parts[2].last_epoch == 0
            load_checkpoint(root / "good", 3, data_state=adapter, strict=True, resume_config=config)
            equal(snapshot(parts, adapter), expected["boundary"])
            commit(parts, task, True)
            equal(snapshot(parts, adapter), expected["after"])
            cases = (
                "extra",
                "kind",
                "fingerprint",
                "schema-bool",
                "rng-schema",
                "transaction",
                "cursor-bool",
                "policy-version",
                "provider-rank",
                "provider-rng",
                "nonfinite",
                "update-counter",
                "rank-global-update",
            )
            for case in cases:
                target = root / case
                if rank == 0:
                    shutil.copytree(root / "good", target)
                    corrupt(target, case)
                dist.barrier()
                before = snapshot(parts, adapter)
                with pytest.raises(native.DistributedCheckpointError):
                    load_checkpoint(
                        target, 3, data_state=adapter, strict=True, resume_config=config
                    )
                equal(snapshot(parts, adapter), before)
            before = snapshot(parts, adapter)
            task.fail_apply = rank == 1
            with pytest.raises(native.DistributedCheckpointError, match="apply strict checkpoint"):
                load_checkpoint(
                    root / "good", 3, data_state=adapter, strict=True, resume_config=config
                )
            equal(snapshot(parts, adapter), before)
            task.pending = rank == 1
            with pytest.raises(native.DistributedCheckpointError, match="pending task"):
                save_checkpoint(
                    root / "pending", 4, data_state=adapter, strict=True, resume_config=config
                )
            task.pending = False
            assert not (root / "pending").exists()
            original = native._write_tensor_file
            if rank == 1:

                def fail_write(*args):
                    raise OSError("injected rank shard write failure")

                native._write_tensor_file = fail_write
            with pytest.raises(native.DistributedCheckpointError, match="write shards"):
                save_checkpoint(
                    root / "partial", 4, data_state=adapter, strict=True, resume_config=config
                )
            native._write_tensor_file = original
            assert not (root / "partial/torch-dcp/00000004" / native.MANIFEST).exists()
            assert find_checkpoint(root / "partial") is None
            assert not torch.cuda.is_initialized()
            (root / f"result-{rank}.json").write_text(
                json.dumps(
                    dict(
                        status="PASS",
                        corruptions=len(cases),
                        zero_update_native_restore=True,
                        rank_local_rng_data_restored=True,
                        all_rank_apply_failure_rollback=True,
                        partial_save_unpublished=True,
                        cuda_initialized=False,
                    )
                )
            )
    finally:
        dist.destroy_process_group()


def test_native_task_real_two_rank_resume_preflight_and_atomicity(tmp_path):
    mp.spawn(worker, args=(str(tmp_path), "save"), nprocs=2, join=True)
    mp.spawn(worker, args=(str(tmp_path), "restore"), nprocs=2, join=True)
    assert all(
        json.loads((tmp_path / f"result-{rank}.json").read_text())["status"] == "PASS"
        for rank in range(2)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("task_schema", 0),
        ("fingerprint", "wrong"),
        ("task_kind", "sft"),
        ("provider_policy_version", True),
    ],
)
def test_envelope_rejects_schema_before_provider_mutation(field, value):
    task = StatefulTask(0)
    state = envelope(task)
    bad = copy.deepcopy(state.state_dict())
    bad[field] = value
    before = task.state_dict()
    with pytest.raises(ValueError):
        state.load_state_dict(bad)
    equal(task.state_dict(), before)


def conditional_commit(parts, task, selected):
    model, optimizer, scheduler, _ = parts
    if selected:
        parameters = list(model.parameters())
        loss = sum(parameters[i].square().sum().full_tensor() for i in selected)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        task.updates += 1
    commit(parts, task, False)


def corrupt_conditional(root, case):
    folder = root / "torch-dcp/00000003"
    path = folder / "rank-00001.pt"
    value = torch.load(path, weights_only=True)
    rows = value["optimizer_schema"]["optimizers"][0]
    state = value["state"]["optimizer"][0]["state"]
    if case == "schema-absent":
        del value["optimizer_schema"]
    elif case == "row-absent":
        rows.pop()
    elif case == "presence-flip":
        rows[0].update(present=False, step=None)
    elif case == "metadata-step-rollback":
        rows[0]["step"] -= 1
    elif case == "payload-step-rollback":
        state[0]["step"]["value"].sub_(1)
    elif case == "payload-state-absent":
        del state[0]
    elif case == "absent-counter":
        rows[2]["step"] = 0
    elif case == "metadata-step-bool":
        rows[0]["step"] = True
    elif case == "counter-exceeds-updates":
        rows[0]["step"] = 3
    elif case == "wrong-policy":
        value["task"]["optimizer_policy"] = "other"
    else:
        raise AssertionError(case)
    torch.save(value, path)
    manifest = json.loads((folder / native.MANIFEST).read_text())
    manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / native.MANIFEST).write_text(json.dumps(manifest))


def conditional_worker(rank, root):
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'conditional-rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.set_num_threads(1)
        meshes = initialize_mesh("fsdp-layout", rank)
        parts = build(*meshes)
        task = StatefulTask(rank)
        adapter = envelope(task, optimizer_policy="conditional_adamw_v1", bounded_load=True)
        config = {"protocol": "conditional-task-v2", "hsdp_replica": 1}
        original_unpack = native._unpack
        original_localize = native.to_localized_model

        def only_one_missing_tensor(saved, live, path="state"):
            assert isinstance(live, torch.Tensor), "bounded load allocated a whole shadow state"
            return original_unpack(saved, live, path)

        def no_full_expert_restack(*args):
            raise AssertionError("bounded load restacked the full model")

        native._unpack = only_one_missing_tensor
        native.to_localized_model = no_full_expert_restack
        conditional_commit(parts, task, [0])
        conditional_commit(parts, task, [0, 1])
        conditional_commit(parts, task, [])
        parameters = list(parts[0].parameters())
        assert parts[1].state[parameters[0]]["step"].item() == 2
        assert parts[1].state[parameters[1]]["step"].item() == 1
        assert parameters[2] not in parts[1].state
        save_checkpoint(root / "good", 3, data_state=adapter, strict=True, resume_config=config)
        boundary = snapshot(parts, adapter)
        conditional_commit(parts, task, [0, 2])
        continued = snapshot(parts, adapter)
        # Warm restore preserves exact per-parameter counters and absent state.
        load_checkpoint(root / "good", 3, data_state=adapter, strict=True, resume_config=config)
        equal(snapshot(parts, adapter), boundary)
        conditional_commit(parts, task, [0, 2])
        equal(snapshot(parts, adapter), continued)
        # Cold restore must not first initialize missing Adam entries.
        parts = build(*meshes)
        task = StatefulTask(rank)
        adapter = envelope(task, optimizer_policy="conditional_adamw_v1", bounded_load=True)
        assert not parts[1].state
        load_checkpoint(root / "good", 3, data_state=adapter, strict=True, resume_config=config)
        equal(snapshot(parts, adapter), boundary)
        for case in (
            "schema-absent",
            "row-absent",
            "presence-flip",
            "metadata-step-rollback",
            "payload-step-rollback",
            "payload-state-absent",
            "absent-counter",
            "metadata-step-bool",
            "counter-exceeds-updates",
            "wrong-policy",
        ):
            target = root / case
            if rank == 0:
                shutil.copytree(root / "good", target)
                corrupt_conditional(target, case)
            dist.barrier()
            before = snapshot(parts, adapter)
            with pytest.raises(native.DistributedCheckpointError):
                load_checkpoint(target, 3, data_state=adapter, strict=True, resume_config=config)
            equal(snapshot(parts, adapter), before)
        conditional_commit(parts, task, [0, 2])
        before = snapshot(parts, adapter)
        task.fail_apply = rank == 1
        with pytest.raises(native.DistributedCheckpointError, match="apply strict checkpoint"):
            load_checkpoint(root / "good", 3, data_state=adapter, strict=True, resume_config=config)
        equal(snapshot(parts, adapter), before)
        # A rank may fail after copying some model shards. CPU rollback must
        # restore every rank without allocating a third full-device state.
        original_copy = native._restore_packed_inplace
        copies = 0
        failed = False

        def fail_mid_copy(saved, template, live=None):
            nonlocal copies, failed
            if isinstance(template, torch.Tensor):
                copies += 1
                if rank == 1 and copies == 2 and not failed:
                    failed = True
                    raise RuntimeError("injected partial local tensor copy")
            return original_copy(saved, template, live)

        native._restore_packed_inplace = fail_mid_copy
        before = snapshot(parts, adapter)
        with pytest.raises(native.DistributedCheckpointError, match="apply strict checkpoint"):
            load_checkpoint(root / "good", 3, data_state=adapter, strict=True, resume_config=config)
        equal(snapshot(parts, adapter), before)
        native._restore_packed_inplace = original_copy
        native._unpack = original_unpack
        native.to_localized_model = original_localize
        # Default task-v2 and legacy strict policies remain complete-step only.
        default_adapter = envelope(task)
        with pytest.raises(native.DistributedCheckpointError, match="optimizer"):
            save_checkpoint(
                root / "default-must-reject",
                4,
                data_state=default_adapter,
                strict=True,
                resume_config=config,
            )
        assert not (root / "default-must-reject").exists()
    finally:
        dist.destroy_process_group()


def test_conditional_task_native_two_rank_cold_warm_and_atomicity(tmp_path):
    mp.spawn(conditional_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_conditional_policy_is_explicit_and_default_envelope_unchanged():
    task = StatefulTask(0)
    default = envelope(task)
    explicit = envelope(task, optimizer_policy="conditional_adamw_v1", bounded_load=True)
    assert "optimizer_policy" not in default.state_dict()
    assert explicit.state_dict()["optimizer_policy"] == "conditional_adamw_v1"
    assert default.fingerprint != explicit.fingerprint
    with pytest.raises(ValueError):
        default.validate_state_dict(explicit.state_dict())
    with pytest.raises(ValueError):
        envelope(task, optimizer_policy="unknown")
