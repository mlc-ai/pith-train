"""CPU recovery contract, not native CUDA backward or real PP/EP execution.

Fresh processes run a real HF Thinker text model with a stochastic test loss and
the converter's native-shaped checkpoint proxy. The small Stateful fixture is
test-only; PR100's actual loader remains a separately pinned integration gate.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch import nn

from pithtrain.contexts import distributed
from pithtrain.modules.checkpoint import CheckpointState, to_canonical_model, to_localized_model
from pithtrain.modules.dataset import MemmapDataset
from pithtrain.modules.strict_checkpoint import (
    StrictResumeError,
    StrictResumeState,
    load_strict_checkpoint,
    save_strict_checkpoint,
)
from pithtrain.tasks.convert_checkpoint import hf2dcp
from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import load_canonical
from tests import test_qwen3_omni_checkpoint as converter


class CommittedDataFixture:
    """PR100 state protocol test double using the existing MemmapDataset.

    Deliberately no new production loader/sampler. Real PR100 fingerprint,
    prefetch, staged-token handling and observation protocol need integration.
    """

    global_batch_size = 2

    def __init__(self, path):
        self.dataset = MemmapDataset(path, sequence_length=7)
        self.fingerprint = hashlib.sha256(path.read_bytes() + b"B2/S7/test-fixture").hexdigest()
        self.consumed_samples, self.pending_step = 0, None

    def state_dict(self):
        if self.pending_step is not None:
            raise RuntimeError("Only checkpoint data after the optimizer step has completed")
        return dict(version=1, fingerprint=self.fingerprint, consumed_samples=self.consumed_samples)

    def load_state_dict(self, state):
        if state["version"] != 1 or state["fingerprint"] != self.fingerprint:
            raise ValueError("Data fixture fingerprint/version differs")
        consumed = state["consumed_samples"]
        if type(consumed) is not int or consumed < 0 or consumed % self.global_batch_size:
            raise ValueError("Invalid checkpoint data consumption position")
        self.consumed_samples, self.pending_step = consumed, None

    def get_batch(self, step):
        assert self.pending_step is None and self.consumed_samples == step * 2
        self.pending_step = step
        ids = list(range(self.consumed_samples, self.consumed_samples + 2))
        pairs = [self.dataset[index] for index in ids]
        tokens, labels = (torch.stack(values) for values in zip(*pairs, strict=True))
        labels[:, 0] = -100
        return dict(sample_ids=ids, tokens=tokens, labels=labels, mask=labels != -100)

    def commit_step(self, step):
        assert self.pending_step == step
        self.consumed_samples += self.global_batch_size
        self.pending_step = None


def seed(value):
    random.seed(value)
    np.random.seed(value + 1)
    torch.manual_seed(value + 2)


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_tree_equal(left, right)
    else:
        assert type(actual) is type(expected) and actual == expected


def configure_cpu():
    torch.set_num_threads(1)
    for key in ("pp_size", "ep_size", "cp_size"):
        setattr(distributed, key, 1)
    distributed.ep_rank = 0


def native_shaped_cpu_model(config):
    hf = converter.model(config)
    body = hf["model"]
    body.lm_head = hf["lm_head"]
    for layer in body.layers:
        if hasattr(layer.mlp, "experts"):
            layer.mlp.experts_per_rank = config.num_experts
    return nn.ModuleDict({"module": nn.ModuleList([body])})


def make_optimizer(model, kind):
    if kind == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    return optimizer, scheduler


def resume_state(model, optimizer, scheduler, data, step):
    return StrictResumeState(
        model, (optimizer,), (scheduler,), data_state=data, step=step, global_batch_size=2
    )


def update(model, optimizer, scheduler, data, step):
    batch = data.get_batch(step)
    body = model["module"][0]
    # Exercise all three CPU RNG domains without changing supported model config.
    scale = 0.9 + 0.03 * (random.random() + float(np.random.random()) + torch.rand(()))
    hidden = body(input_ids=batch["tokens"], use_cache=False).last_hidden_state
    hidden = F.dropout(hidden, p=0.1, training=True)  # test-only stochastic loss input
    logits = body.lm_head(hidden)
    loss = F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(), reduction="sum")
    loss = loss / batch["mask"].sum() * scale
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    data.commit_step(step)
    assert any(not torch.equal(left, right) for left, right in zip(before, model.parameters()))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return {
        "batch": batch,
        "loss": loss.detach(),
        "state": copy.deepcopy(
            resume_state(model, optimizer, scheduler, data, step + 1).state_dict()
        ),
    }


def worker(root, mode, layout, optimizer_kind):
    configure_cpu()
    seed(2026 if mode == "save" else 901)
    config = converter.config.__wrapped__()
    model = native_shaped_cpu_model(config).train()
    data = CommittedDataFixture(root / "tokens.npy")
    optimizer, scheduler = make_optimizer(model, optimizer_kind)
    checkpoint = root / "training"
    if mode == "save":
        # Use the actual file converter to initialize the trainable proxy.
        source, disk = root / "hf", root / "imported"
        converter.write_hf(source, config, converter.synthetic_hf(config, layout=layout))
        hf2dcp(converter.make_cfg(load_path=source, save_path=disk), converter.LOGGER)
        canonical = load_canonical(disk, config)
        model.load_state_dict(to_localized_model(canonical, model), strict=True)
        update(model, optimizer, scheduler, data, 0)
        update(model, optimizer, scheduler, data, 1)
        state = resume_state(model, optimizer, scheduler, data, 2)
        save_strict_checkpoint(checkpoint, state)
        # Match PR100's save_then_runtime_state_v1 producer ordering.
        boundary = copy.deepcopy(state.state_dict())
        expected = update(model, optimizer, scheduler, data, 2)
        torch.save({"boundary": boundary, "next": expected}, root / "expected.pt")
    else:
        expected = torch.load(root / "expected.pt", weights_only=True)
        state = resume_state(model, optimizer, scheduler, data, 2)
        assert load_strict_checkpoint(checkpoint, state) == 2
        assert_tree_equal(state.state_dict(), expected["boundary"])
        actual = update(model, optimizer, scheduler, data, 2)
        assert_tree_equal(actual, expected["next"])
        (root / "result.json").write_text(
            json.dumps(
                {
                    "layout": layout,
                    "optimizer": optimizer_kind,
                    "completed_updates": 3,
                    "restored_boundary": 2,
                    "next_sample_ids": actual["batch"]["sample_ids"],
                    "loss": actual["loss"].item(),
                    "max_state_error": 0,
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "model": "HF Thinker eager CPU, native-shaped checkpoint proxy",
                    "data": "MemmapDataset + test-only PR100 Stateful contract fixture",
                },
                indent=2,
            )
        )
    assert not torch.cuda.is_initialized()


@pytest.fixture(autouse=True)
def cpu_context(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    for key, value in dict(pp_size=1, ep_size=1, cp_size=1, ep_rank=0).items():
        monkeypatch.setattr(distributed, key, value, raising=False)
    yield
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(previous)


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(113, dtype=np.int64) % 37)
    return path


@pytest.mark.parametrize("layout,optimizer_kind", [("split", "adamw"), ("fused", "sgd")])
def test_fresh_process_next_batch_and_continuous_update(data_path, layout, optimizer_kind):
    root = data_path.parent
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    for mode in ("save", "resume"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.test_strict_checkpoint",
                "--root",
                str(root),
                "--mode",
                mode,
                "--layout",
                layout,
                "--optimizer",
                optimizer_kind,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((root / "result.json").read_text())
    assert evidence["max_state_error"] == 0 and evidence["next_sample_ids"] == [4, 5]
    assert not evidence["cuda_initialized"]
    print("RESUME_EVIDENCE=" + json.dumps(evidence, sort_keys=True))


def linear_state(data_path, *, committed):
    model = nn.Linear(2, 3)
    optimizer, scheduler = make_optimizer(model, "adamw")
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    data = CommittedDataFixture(data_path)
    if committed:
        data.get_batch(0)
        data.commit_step(0)
    return resume_state(model, optimizer, scheduler, data, 1)


FAULTS = [
    "missing_optimizer",
    "missing_scheduler",
    "missing_data",
    "missing_resume",
    "missing_torch_cpu",
    "missing_python",
    "missing_numpy",
    "missing_cuda",
    "scheduler_count",
    "extra_scheduler",
    "model_missing",
    "model_unexpected",
    "model_shape",
    "model_dtype",
    "optimizer_missing",
    "optimizer_shape",
    "optimizer_membership",
    "optimizer_order",
    "fingerprint",
    "cursor",
    "cursor_bool",
    "step",
    "batch_size",
    "optimizer_type",
    "scheduler_type",
    "rng_invalid",
    "rng_python",
]


def corrupt(saved, fault):
    if fault in {"missing_optimizer", "missing_scheduler", "missing_data", "missing_resume"}:
        del saved[fault.removeprefix("missing_")]
    elif fault.startswith("missing_"):
        del saved["resume"]["rng"][fault.removeprefix("missing_")]
    elif fault == "scheduler_count":
        saved["scheduler"] = []
    elif fault == "extra_scheduler":
        saved["scheduler"].append(copy.deepcopy(saved["scheduler"][0]))
    elif fault == "model_missing":
        del saved["model"]["weight"]
    elif fault == "model_unexpected":
        saved["model"]["typo.weight"] = torch.ones(2)
    elif fault == "model_shape":
        saved["model"]["weight"] = saved["model"]["weight"].T.contiguous()
    elif fault == "model_dtype":
        saved["model"]["weight"] = saved["model"]["weight"].double()
    elif fault == "optimizer_missing":
        del saved["optimizer"]["state"]["weight"]["exp_avg"]
    elif fault == "optimizer_shape":
        saved["optimizer"]["state"]["weight"]["exp_avg"] = torch.ones(2, 3)
    elif fault == "optimizer_membership":
        saved["optimizer"]["param_groups"][0]["params"][0] = "missing.weight"
    elif fault == "optimizer_order":
        saved["optimizer"]["param_groups"][0]["params"].reverse()
    elif fault == "fingerprint":
        saved["data"]["fingerprint"] = "different-data"
    elif fault == "cursor":
        saved["data"]["consumed_samples"] = 4
    elif fault == "cursor_bool":
        saved["data"]["consumed_samples"] = True
    elif fault == "step":
        saved["resume"]["step"] = 7
    elif fault == "batch_size":
        saved["resume"]["global_batch_size"] = 1
    elif fault == "optimizer_type":
        saved["resume"]["optimizer_types"][0] = "torch.optim.SGD"
    elif fault == "scheduler_type":
        saved["resume"]["scheduler_types"][0] = "torch.optim.lr_scheduler.LambdaLR"
    elif fault == "rng_invalid":
        saved["resume"]["rng"]["torch_cpu"].zero_()
    elif fault == "rng_python":
        old = saved["resume"]["rng"]["python"]
        saved["resume"]["rng"]["python"] = (99, old[1], old[2])
    else:
        raise AssertionError(fault)


@pytest.mark.parametrize("fault", FAULTS)
def test_corrupt_checkpoint_rejected_before_live_state_changes(data_path, fault):
    saved = copy.deepcopy(linear_state(data_path, committed=True).state_dict())
    corrupt(saved, fault)
    root = data_path.parent / "corrupt"
    dcp.save({"app": saved}, checkpoint_id=root / "torch-dcp/00000001", no_dist=True)
    live = linear_state(data_path, committed=False)
    before = copy.deepcopy(live._snapshot())
    with pytest.raises(StrictResumeError):
        load_strict_checkpoint(root, live)
    assert_tree_equal(live._snapshot(), before)
    assert live.data_state.pending_step is None


def test_save_rejects_uncommitted_or_wrong_boundary_and_existing_target(data_path):
    state = linear_state(data_path, committed=False)
    root = data_path.parent / "checkpoint"
    with pytest.raises(StrictResumeError, match="position disagree"):
        save_strict_checkpoint(root, state)
    state.data_state.get_batch(0)
    with pytest.raises(RuntimeError, match="optimizer step"):
        save_strict_checkpoint(root, state)
    state.data_state.commit_step(0)
    location = save_strict_checkpoint(root, state)
    content = {p.name: p.read_bytes() for p in location.iterdir()}
    with pytest.raises(FileExistsError):
        save_strict_checkpoint(root, state)
    assert {p.name: p.read_bytes() for p in location.iterdir()} == content


def test_rejected_checkpoint_preserves_pristine_optimizer(data_path):
    saved = linear_state(data_path, committed=True).state_dict()
    del saved["optimizer"]
    root = data_path.parent / "pristine"
    dcp.save({"app": saved}, checkpoint_id=root / "torch-dcp/00000001", no_dist=True)
    model = nn.Linear(2, 3)
    optimizer, scheduler = make_optimizer(model, "adamw")
    state = resume_state(model, optimizer, scheduler, CommittedDataFixture(data_path), 1)
    before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    assert not optimizer.state
    with pytest.raises(StrictResumeError, match="missing"):
        load_strict_checkpoint(root, state)
    assert not optimizer.state, "Metadata preflight initialized the live optimizer"
    assert_tree_equal(model.state_dict(), before)
    assert_tree_equal(optimizer.state_dict(), optimizer_before)


def test_weight_initialization_is_not_strict_resume(data_path):
    state = linear_state(data_path, committed=False)
    root = data_path.parent / "weights-only"
    dcp.save(
        {"app": {"model": state.base.state_dict()["model"]}},
        checkpoint_id=root / "torch-dcp/00000001",
        no_dist=True,
    )
    with pytest.raises(StrictResumeError, match="missing"):
        load_strict_checkpoint(root, state)
    legacy = CheckpointState(
        state.base.model, state.base.optimizers, state.base.schedulers, model_only=True
    )
    dcp.load({"app": legacy}, checkpoint_id=root / "torch-dcp/00000001", no_dist=True)


def test_python_cached_gaussian_is_valid_resume_state(data_path):
    source = linear_state(data_path, committed=True)
    random.seed(125)
    random.gauss(0, 1)
    assert isinstance(random.getstate()[2], float)
    root = data_path.parent / "gaussian"
    save_strict_checkpoint(root, source)
    expected = random.gauss(0, 1)
    random.seed(991)
    restored = linear_state(data_path, committed=False)
    load_strict_checkpoint(root, restored)
    assert random.gauss(0, 1) == expected


@pytest.mark.parametrize("axis", ["pp_size", "ep_size", "cp_size"])
def test_distributed_resume_is_explicitly_not_certified(data_path, monkeypatch, axis):
    monkeypatch.setattr(distributed, axis, 2)
    with pytest.raises(NotImplementedError, match="PP=EP=CP=1"):
        linear_state(data_path, committed=False)


@pytest.mark.parametrize("pp,ep", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_cpu_pp_ep_model_and_optimizer_key_mapping(pp, ep, monkeypatch):
    """In-memory mapping only: no pipeline, collectives, FSDP or update equivalence."""
    config = converter.config.__wrapped__()
    config.num_hidden_layers, config.num_experts = 4, 4
    full = native_shaped_cpu_model(config)
    optimizer, scheduler = make_optimizer(full, "adamw")
    for i, parameter in enumerate(full.parameters()):
        parameter.grad = torch.full_like(parameter, (i + 1) / 1000)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    saved = copy.deepcopy(CheckpointState(full, (optimizer,), (scheduler,)).state_dict())
    observed = set()
    for pp_rank in range(pp):
        for ep_rank in range(ep):
            monkeypatch.setattr(distributed, "ep_rank", ep_rank)
            stages = (pp_rank, 2 * pp - 1 - pp_rank)
            chunks = []
            for stage in stages:
                body = native_shaped_cpu_model(config)["module"][0]
                # Explicit synthetic ownership tests global layer IDs and the
                # two local module.N prefixes; it is not layer_partition.
                selected = [i for i in range(4) if i % (2 * pp) == stage]
                chunk = nn.Module()
                chunk.layers = nn.ModuleDict({str(i): body.layers[i] for i in selected})
                if stage == 0:
                    chunk.embed_tokens = body.embed_tokens
                if stage == 2 * pp - 1:
                    chunk.norm, chunk.lm_head = body.norm, body.lm_head
                for layer in chunk.layers.values():
                    layer.mlp.experts_per_rank = config.num_experts // ep
                    start = ep_rank * layer.mlp.experts_per_rank
                    end = start + layer.mlp.experts_per_rank
                    for name, parameter in list(layer.mlp.experts.named_parameters()):
                        setattr(layer.mlp.experts, name, nn.Parameter(parameter[start:end].clone()))
                chunks.append(chunk)
            local = nn.ModuleDict({"module": nn.ModuleList(chunks)})
            local_opt, local_sched = make_optimizer(local, "adamw")
            state = CheckpointState(local, (local_opt,), (local_sched,))
            state.load_state_dict(saved)
            rebuilt = state.state_dict()
            for key, value in rebuilt["model"].items():
                observed.add(key)
                torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
                torch.testing.assert_close(
                    rebuilt["optimizer"]["state"][key],
                    saved["optimizer"]["state"][key],
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                to_canonical_model(local.state_dict(), local), rebuilt["model"], rtol=0, atol=0
            )
    assert observed == saved["model"].keys()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("save", "resume"), required=True)
    parser.add_argument("--layout", choices=("split", "fused"), required=True)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), required=True)
    args = parser.parse_args()
    worker(args.root, args.mode, args.layout, args.optimizer)
