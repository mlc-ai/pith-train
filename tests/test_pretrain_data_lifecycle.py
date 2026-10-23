"""Execute the actual task functions on CPU with a third data provider.

Model/GPU setup is unavailable on CPU: load the function ASTs without the GPU
imports, then mock pipeline, device metrics, and collectives. This checks task
lifecycle wiring and failure ordering, not GPU execution or numerical acceptance.
"""

import ast
import copy
import gc
import logging as std_logging
import random
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp

from pithtrain.contexts import logging, training
from pithtrain.modules.checkpoint import (
    CheckpointState,
    find_checkpoint,
    initialize_checkpoint,
    load_checkpoint,
)
from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.training_data import OmniPretrainData, global_loss_mean, global_target_count
from tests import test_training_data as data_fixtures
from tests.test_data_config import ROOT, example_modules
from tests.test_strict_checkpoint import assert_tree_equal


def task_functions(**namespace):
    path = Path(__file__).parents[1] / "pithtrain/tasks/pretrain_lm.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "get_global_batch",
            "train_step",
            "launch",
        }:
            node.decorator_list = []
            functions.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class OtherData:
    """No inheritance or Omni-specific fields; only the shared data contract."""

    def __init__(self, events):
        self.events = events
        self.checkpoint_state = SimpleNamespace(state_dict=lambda: {"completed": 1})

    def get_batch(self, step, device):
        self.events.append(("read", step))
        return [
            Microbatch(
                model_inputs=(torch.ones(1, 2),),
                cu_seqlens=None,
                objective_inputs=(torch.ones(1, 2, dtype=torch.long),),
            )
        ]

    def commit_step(self, step):
        self.events.append(("commit", step))


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("fail_optimizer", [False, True])
def test_task_commits_only_after_successful_optimizer_and_scheduler(
    monkeypatch, tmp_path, fail_optimizer, strict
):
    events = []
    data = OtherData(events)
    weight = torch.nn.Parameter(torch.tensor(1.0))
    model = SimpleNamespace(train=lambda: None, parameters=lambda: [weight])

    def model_step(batches, objective):
        loss = weight * batches[0].model_inputs[0].sum()
        loss.backward()
        return [loss.detach()]

    def optimizer_step():
        events.append("optimizer")
        if fail_optimizer:
            raise RuntimeError("optimizer failed")

    def save_checkpoint(root, step, *, data_state, **kwargs):
        assert kwargs == ({"strict": True, "resume_config": {"test": "contract"}} if strict else {})
        assert root == tmp_path and data_state is data.checkpoint_state
        assert events[-1] == ("commit", step - 1)
        events.append(("save", step))

    model.step = model_step
    optimizer = SimpleNamespace(step=optimizer_step, zero_grad=lambda **_: None)
    scheduler = SimpleNamespace(step=lambda: events.append("scheduler"))
    tc = SimpleNamespace(
        nsys_start=None,
        nsys_stop=None,
        memory_profile_start=None,
        memory_profile_stop=None,
        global_batch_size=1,
        micro_batch_size=1,
        moe_load_balance_coef=0,
        save_interval=1,
        strict_resume=strict,
        max_steps=1,
        save_location=tmp_path,
    )
    cfg = SimpleNamespace(training=tc, distributed=SimpleNamespace(hsdp_replica=False))
    mesh = type(
        "Mesh",
        (),
        {
            "__getitem__": lambda self, key: self,
            "_flatten": lambda self: self,
            "get_group": lambda self: None,
        },
    )()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda.memory, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_, **__: None)
    task = task_functions(
        torch=torch,
        time=time,
        gc=gc,
        objective=None,
        distributed=SimpleNamespace(dp_size=1, pp_rank=0, rank=1, attn_mesh=mesh),
        training=SimpleNamespace(model=model, optimizers=[optimizer], schedulers=[scheduler]),
        logging=SimpleNamespace(stdout=None),
        global_target_count=global_target_count,
        global_loss_mean=global_loss_mean,
        clip_grad_norm=lambda *_, **__: torch.tensor(1.0),
        MoELoadBalanceLossTracker=SimpleNamespace(
            reset=lambda: None, get_total_count_and_clear=lambda: (0, 0)
        ),
        save_checkpoint=save_checkpoint,
        resume_contract=lambda cfg: {"test": "contract"},
    )
    if fail_optimizer:
        with pytest.raises(RuntimeError, match="optimizer failed"):
            task.train_step(cfg, data, 0)
        assert events == [("read", 0), "optimizer"]
    else:
        task.train_step(cfg, data, 0)
        assert events == [("read", 0), "optimizer", "scheduler", ("commit", 0), ("save", 1)]


@pytest.mark.parametrize("has_data_state,strict", [(False, False), (True, False), (True, True)])
def test_launch_restores_through_provider_before_resuming(tmp_path, has_data_state, strict):
    events = []
    data = OtherData(events)
    if not has_data_state:
        data.checkpoint_state = None

    def load_checkpoint(root, step, *, data_state, **kwargs):
        assert kwargs == ({"strict": True, "resume_config": {"test": "contract"}} if strict else {})
        assert root == tmp_path and step == 2 and data_state is data.checkpoint_state
        events.append("restore")

    def step(cfg, source, number):
        assert source is data
        assert events[0] == "restore"
        events.append(number)

    # launch resolves train_step in its globals, so replace only that callback.
    task = task_functions(
        gc=SimpleNamespace(disable=lambda: None, enable=lambda: None),
        setup_logging=lambda cfg: None,
        setup_distributed=lambda cfg: None,
        setup_dataset=lambda cfg: data,
        setup_training=lambda cfg: None,
        logging=SimpleNamespace(stdout=SimpleNamespace(info=lambda *_: None)),
        find_checkpoint=lambda root: 2,
        load_checkpoint=load_checkpoint,
        resume_contract=lambda cfg: {"test": "contract"},
    )
    task.launch.__globals__["train_step"] = step
    task.launch(
        SimpleNamespace(
            training=SimpleNamespace(save_location=tmp_path, max_steps=4, strict_resume=strict)
        )
    )
    assert events == ["restore", 2, 3]


text_corpus = data_fixtures.text_corpus
text_bundle = data_fixtures.text_bundle


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def assert_rng_equal(before):
    assert random.getstate() == before["python"]
    current = np.random.get_state()
    assert current[0] == before["numpy"][0]
    np.testing.assert_array_equal(current[1], before["numpy"][1])
    assert current[2:] == before["numpy"][2:]
    assert torch.equal(torch.get_rng_state(), before["torch"])


@pytest.mark.parametrize("source_has_training_state", [False, True])
def test_public_launch_initializes_weights_with_fresh_prepared_data(
    monkeypatch, tmp_path, text_bundle, source_has_training_state
):
    """Real Torch DCP/public initializer and prepared reader; only GPU setup/step are CPU stand-ins."""
    root, tc = text_bundle
    data = OmniPretrainData(data_fixtures.prepared_config(root), tc)
    source = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(0.375)
        source.bias.fill_(-0.25)
    app = {"model": source.state_dict()}
    if source_has_training_state:
        app.update(
            optimizer={"stale": torch.tensor(99)},
            scheduler=[{"last_epoch": 99}],
            data={"consumed_samples": 99},
        )
    init_root = tmp_path / "converted"
    location = init_root / "torch-dcp/00000007"
    dcp.save({"app": app}, checkpoint_id=location, no_dist=True)
    # Deliberately invalid: model initialization must never read an old RNG sidecar.
    (location / "rng-rank-00000.pt").write_bytes(b"not a training RNG state")
    events, snapshots = [], {}

    def setup(cfg):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        for key, value in dict(
            model=model, optimizers=(optimizer,), schedulers=(scheduler,)
        ).items():
            monkeypatch.setattr(training, key, value, raising=False)
        snapshots.update(
            optimizer=copy.deepcopy(optimizer.state_dict()),
            scheduler=copy.deepcopy(scheduler.state_dict()),
            data=copy.deepcopy(data.state_dict()),
            rng=rng_state(),
        )

    def first_update(cfg, actual_data, step):
        assert actual_data is data and step == 0  # The source identifier was 7.
        model, (optimizer,), (scheduler,) = training.model, training.optimizers, training.schedulers
        assert_tree_equal(model.state_dict(), source.state_dict())
        assert not optimizer.state
        assert_tree_equal(optimizer.state_dict(), snapshots["optimizer"])
        assert_tree_equal(scheduler.state_dict(), snapshots["scheduler"])
        assert_tree_equal(data.state_dict(), snapshots["data"])
        assert_rng_equal(snapshots["rng"])
        assert not torch.cuda.is_initialized()
        batch = data.get_batch(step, "cpu")[0]
        model(batch.model_inputs[0][..., :2].float() / 10000).square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        data.commit_step(step)
        assert all(value["step"].item() == 1 for value in optimizer.state.values())
        events.append(step)

    monkeypatch.setattr(logging, "stdout", std_logging.getLogger(__name__), raising=False)
    task = task_functions(
        gc=SimpleNamespace(disable=lambda: None, enable=lambda: None),
        setup_logging=lambda cfg: None,
        setup_distributed=lambda cfg: None,
        setup_dataset=lambda cfg: data,
        setup_training=setup,
        logging=logging,
        find_checkpoint=find_checkpoint,
        initialize_checkpoint=initialize_checkpoint,
        load_checkpoint=load_checkpoint,
    )
    task.launch.__globals__["train_step"] = first_update
    task.launch(
        SimpleNamespace(
            initialize_from=init_root,
            initialize_step=7,
            training=SimpleNamespace(
                save_location=tmp_path / "new-run", max_steps=1, strict_resume=True
            ),
        )
    )
    assert events == [0]
    assert data.state_dict() != snapshots["data"]
    assert not torch.cuda.is_initialized()


def test_public_resume_still_rejects_model_only_without_prepared_state(
    monkeypatch, tmp_path, text_bundle
):
    root, tc = text_bundle
    data = OmniPretrainData(data_fixtures.prepared_config(root), tc)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    location = tmp_path / "resume/torch-dcp/00000000"
    dcp.save(
        {"app": {"model": {k: torch.zeros_like(v) for k, v in model.state_dict().items()}}},
        checkpoint_id=location,
        no_dist=True,
    )
    for key, value in dict(model=model, optimizers=(optimizer,), schedulers=(scheduler,)).items():
        monkeypatch.setattr(training, key, value, raising=False)
    monkeypatch.setattr(logging, "stdout", std_logging.getLogger(__name__), raising=False)
    before = copy.deepcopy(
        (model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), data.state_dict())
    )
    rng = rng_state()
    with pytest.raises(ValueError, match="no data state"):
        load_checkpoint(tmp_path / "resume", 0, data_state=data.checkpoint_state)
    assert_tree_equal(
        (model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), data.state_dict()),
        before,
    )
    assert_rng_equal(rng)
    assert not torch.cuda.is_initialized()


def test_model_only_state_does_not_initialize_optimizer_slots():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    before = copy.deepcopy(optimizer.state_dict())
    state = CheckpointState(model, (optimizer,), (), model_only=True)
    assert set(state.state_dict()) == {"model"}
    assert not optimizer.state
    assert_tree_equal(optimizer.state_dict(), before)


@pytest.mark.parametrize(
    "source,source_step,saved_step,match",
    [
        ("source", 0, 0, "new output run"),
        ("source", -1, None, "nonnegative"),
        ("source", True, None, "nonnegative"),
        (None, 3, None, "requires initialize_from"),
    ],
)
def test_launch_rejects_conflicting_initialization_before_setup(
    source, source_step, saved_step, match
):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid initialization reached training setup")

    task = task_functions(setup_logging=unexpected, find_checkpoint=lambda root: saved_step)
    cfg = SimpleNamespace(
        initialize_from=source,
        initialize_step=source_step,
        training=SimpleNamespace(save_location=Path("output")),
    )
    with pytest.raises(ValueError, match=match):
        task.launch(cfg)


@pytest.mark.parametrize("initialize", [False, True])
def test_native_multimodal_recipe_exposes_distinct_initialization(
    monkeypatch, tmp_path, initialize
):
    calls = example_modules(monkeypatch)
    path = ROOT / "examples/pretrain_lm/qwen3-omni-native-multimodal/script.py"
    args = [
        str(path),
        "--dataset",
        str(tmp_path / "bundle"),
        "--model",
        str(tmp_path / "model"),
        "--checkpoint",
        str(tmp_path / "output"),
        "--epoch-samples",
        "8",
    ]
    if initialize:
        args += ["--initialize-from", str(tmp_path / "converted"), "--initialize-step", "7"]
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(str(path), run_name="__main__")
    assert len(calls) == 1
    cfg = calls[0]
    assert cfg.initialize_from == (tmp_path / "converted" if initialize else None)
    assert cfg.initialize_step == (7 if initialize else 0)
    assert cfg.training.strict_resume is True
    assert cfg.training.save_location == tmp_path / "output"
    assert cfg.data.format == "prepared_bundle"
    assert not torch.cuda.is_initialized()
