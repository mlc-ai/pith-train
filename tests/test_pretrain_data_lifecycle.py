"""Execute the actual task functions on CPU with a third data provider.

Model/GPU setup is unavailable on CPU: load the function ASTs without the GPU
imports, then mock pipeline, device metrics, and collectives. This checks task
lifecycle wiring and failure ordering, not GPU execution or numerical acceptance.
"""

import ast
import gc
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.optimizer import local_grads, scale_grads_
from pithtrain.modules.training_data import global_loss_mean, global_target_count


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


@pytest.mark.parametrize("fail_stage", [None, "optimizer", "scheduler"])
@pytest.mark.parametrize("fp8", [False, True])
def test_task_commits_only_after_successful_optimizer_and_scheduler(
    monkeypatch, tmp_path, fail_stage, fp8
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
        torch.testing.assert_close(weight.grad, torch.tensor(1.0))
        events.append("optimizer")
        if fail_stage == "optimizer":
            raise RuntimeError("optimizer failed")

    def scheduler_step():
        events.append("scheduler")
        if fail_stage == "scheduler":
            raise RuntimeError("scheduler failed")

    def save_checkpoint(root, step, *, data_state):
        assert root == tmp_path and data_state is data.checkpoint_state
        assert events[-1] == ("commit", step - 1)
        events.append(("save", step))

    model.step = model_step
    optimizer = SimpleNamespace(step=optimizer_step, zero_grad=lambda **_: None)
    scheduler = SimpleNamespace(step=scheduler_step)
    tc = SimpleNamespace(
        nsys_start=None,
        nsys_stop=None,
        memory_profile_start=None,
        memory_profile_stop=None,
        global_batch_size=1,
        micro_batch_size=1,
        moe_load_balance_coef=0,
        save_interval=1,
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
        training=SimpleNamespace(
            model=model, optimizers=[optimizer], schedulers=[scheduler], fp8=fp8
        ),
        logging=SimpleNamespace(stdout=None),
        global_target_count=global_target_count,
        global_loss_mean=global_loss_mean,
        local_grads=local_grads,
        scale_grads_=scale_grads_,
        precompute_fp8_param_scales=lambda: events.append("fp8-scales"),
        clip_grad_norm=lambda *_, **__: torch.tensor(1.0),
        MoELoadBalanceLossTracker=SimpleNamespace(
            reset=lambda: None, get_total_count_and_clear=lambda: (0, 0)
        ),
        save_checkpoint=save_checkpoint,
    )
    if fail_stage:
        with pytest.raises(RuntimeError, match=f"{fail_stage} failed"):
            task.train_step(cfg, data, 0)
        expected = [("read", 0), "optimizer"]
        if fail_stage == "scheduler":
            expected.append("scheduler")
        assert events == expected
    else:
        task.train_step(cfg, data, 0)
        expected = [("read", 0), "optimizer", "scheduler"]
        if fp8:
            expected.append("fp8-scales")
        assert events == [*expected, ("commit", 0), ("save", 1)]


@pytest.mark.parametrize("has_data_state", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_launch_restores_through_provider_before_resuming(tmp_path, has_data_state, fp8):
    events = []
    data = OtherData(events)
    if not has_data_state:
        data.checkpoint_state = None

    def load_checkpoint(root, step, *, data_state):
        assert root == tmp_path and step == 2 and data_state is data.checkpoint_state
        events.append("restore")

    def step(cfg, source, number):
        assert source is data
        assert events[0] == "restore"
        events.append(number)

    # launch resolves train_step in its globals, so replace only that callback.
    task = task_functions(
        gc=SimpleNamespace(
            collect=lambda: None, freeze=lambda: None, disable=lambda: None, enable=lambda: None
        ),
        setup_logging=lambda cfg: None,
        setup_distributed=lambda cfg: None,
        setup_dataset=lambda cfg: data,
        setup_training=lambda cfg: None,
        logging=SimpleNamespace(stdout=SimpleNamespace(info=lambda *_: None)),
        find_checkpoint=lambda root: 2,
        load_checkpoint=load_checkpoint,
        training=SimpleNamespace(fp8=fp8),
        precompute_fp8_param_scales=lambda: events.append("fp8-scales"),
        tune_kernels=lambda: events.append("tune"),
    )
    task.launch.__globals__["train_step"] = step
    task.launch(SimpleNamespace(training=SimpleNamespace(save_location=tmp_path, max_steps=4)))
    expected = ["restore", "fp8-scales"] if fp8 else ["restore"]
    assert events == [*expected, "tune", 2, "tune", 3]


@pytest.mark.parametrize("fp8", [False, True])
def test_launch_imports_hf_weights_before_first_data_step(tmp_path, fp8):
    events = []
    data = OtherData(events)
    model = object()

    def load_hf(path, target):
        assert path == tmp_path / "hf" and target is model
        events.append("hf-import")

    task = task_functions(
        gc=SimpleNamespace(
            collect=lambda: None, freeze=lambda: None, disable=lambda: None, enable=lambda: None
        ),
        setup_logging=lambda cfg: None,
        setup_distributed=lambda cfg: None,
        setup_dataset=lambda cfg: data,
        setup_training=lambda cfg: None,
        logging=SimpleNamespace(stdout=SimpleNamespace(info=lambda *_: None)),
        find_checkpoint=lambda root: None,
        load_hf_into_model=load_hf,
        training=SimpleNamespace(fp8=fp8, model=model),
        precompute_fp8_param_scales=lambda: events.append("fp8-scales"),
        tune_kernels=lambda: events.append("tune"),
    )

    def step(cfg, source, number):
        assert source is data
        events.append(number)

    task.launch.__globals__["train_step"] = step
    task.launch(
        SimpleNamespace(
            training=SimpleNamespace(
                save_location=tmp_path, hf_import_path=tmp_path / "hf", max_steps=1
            )
        )
    )
    expected = ["hf-import", "fp8-scales"] if fp8 else ["hf-import"]
    assert events == [*expected, "tune", 0, "tune"]
