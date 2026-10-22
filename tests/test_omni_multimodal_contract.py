"""Exercise real selection/entrypoint ASTs without importing GPU-only operators."""

import ast
import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pithtrain.modules.model_config import resolve_text_model_config
from tests.test_data_config import task_config
from tests.test_qwen3_omni_multimodal import config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "flag", ("native_vision", "native_audio", "native_video", "native_multimodal")
)
def test_all_native_modes_have_explicit_selection_and_no_ambiguous_combination(flag):
    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "model_class_for_config"
    )
    classes = {
        "native_vision": ("Qwen3OmniMoeThinkerImageModel", object()),
        "native_audio": ("Qwen3OmniMoeThinkerAudioModel", object()),
        "native_video": ("Qwen3OmniMoeThinkerVideoModel", object()),
        "native_multimodal": ("Qwen3OmniMoeThinkerMultimodalModel", object()),
    }
    text = object()
    namespace = {name: value for name, value in classes.values()}
    namespace.update(
        Qwen3OmniMoeThinkerTextModel=text, resolve_text_model_config=resolve_text_model_config
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "registration", "exec"), namespace)
    choose = namespace[function.name]
    assert choose(config()) is text
    assert choose(config(), **{flag: True}) is classes[flag][1]
    other = "native_audio" if flag == "native_multimodal" else "native_multimodal"
    with pytest.raises(ValueError, match="Select one"):
        choose(config(), **{flag: True, other: True})


def test_actual_dataset_entry_forwards_every_native_selection_flag(tmp_path):
    observed = []

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        def validate_model(self, *args):
            pass

    def choose(source, **flags):
        observed.append(flags)
        return object()

    task = task_config(
        OmniPretrainData=Provider,
        AutoConfig=SimpleNamespace(from_pretrained=lambda _: object()),
        model_class_for_config=choose,
    )
    cfg = task.PretrainLMCfg()
    cfg.data.dataset, cfg.data.format = tmp_path, "prepared_bundle"
    cfg.training.model, cfg.training.native_multimodal = "composite", True
    task.setup_dataset(cfg)
    assert observed == [
        dict(native_vision=False, native_audio=False, native_video=False, native_multimodal=True)
    ]


def test_unified_production_has_no_transformers_execution_dependency():
    for filename in ("qwen3_omni_multimodal.py", "qwen3_omni_multimodal_inputs.py"):
        tree = ast.parse((ROOT / "pithtrain/models" / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("transformers")


def optimizer_probe_block(driver, root=ROOT):
    """Execute each real probe block without importing native CUDA operators."""
    path = root / "tests/acceptance" / driver
    tree = ast.parse(path.read_text())
    name = "staged" if driver == "omni_multimodal_training.py" else "run"
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = next(
        i
        for i, n in enumerate(entry.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "before"
    )
    end_target = "(contract, checkpoint)" if name == "staged" else "zero"
    end = next(
        i
        for i in range(start + 1, len(entry.body))
        if isinstance(entry.body[i], ast.Assign)
        and ast.unparse(entry.body[i].targets[0]) == end_target
    )
    return compile(ast.Module(body=entry.body[start:end], type_ignores=[]), str(path), "exec")


def probe_fixture():
    """Use the actual AdamW factory with only CPU model and collective fixtures."""
    from tests.acceptance.omni_image_evidence import local_tensor, module_update_evidence
    from tests.acceptance.omni_optimizer_probe import isolated_optimizer_probe

    torch.manual_seed(317)
    model = nn.ModuleDict({name: nn.Linear(3, 2) for name in ("text", "visual", "audio")})
    context = SimpleNamespace(model=model)
    config = SimpleNamespace(lr=1e-4)
    namespace = dict(
        training=context,
        AdamW=torch.optim.AdamW,
        Optimizer=torch.optim.Optimizer,
        TrainingCfg=SimpleNamespace,
    )
    path = ROOT / "pithtrain/modules/training.py"
    tree = ast.parse(path.read_text())
    factory = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_adamw_optimizer"
    )
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(path), "exec"), namespace)
    make_optimizer = namespace["make_adamw_optimizer"]
    context.optimizers = make_optimizer(config)
    context.schedulers = (torch.optim.lr_scheduler.LambdaLR(context.optimizers[0], lambda _: 1),)
    context.consumed_samples = 0
    for parameter in model.parameters():
        parameter.grad = parameter.detach().clone()
    namespace.update(
        actual=dict(model.named_parameters()),
        t=config,
        config=config,
        make_adamw_optimizer=make_optimizer,
        local_tensor=local_tensor,
        module_update_evidence=module_update_evidence,
        isolated_optimizer_probe=isolated_optimizer_probe,
        torch=torch,
        component_groups=lambda _: ("text.", "visual.", "audio."),
        training_groups_fn=lambda _: ("text.", "visual.", "audio."),
        dist=SimpleNamespace(all_reduce=lambda _: None),
    )
    return namespace


@pytest.mark.parametrize("driver", ["omni_multimodal_training.py", "omni_image_staged.py"])
def test_probe_preserves_formal_start_and_checkpoint_steps(driver):
    from pithtrain.modules.distributed_checkpoint import (
        DistributedCheckpointError,
        _apply,
        _check_training_state,
        _optimizer_template,
        _runtime,
    )
    from pithtrain.modules.strict_checkpoint import _rng_state
    from tests.acceptance.omni_native_resume import equal
    from tests.test_distributed_strict_checkpoint import Cursor

    namespace = probe_fixture()
    training = namespace["training"]
    optimizer, scheduler = training.optimizers[0], training.schedulers[0]
    parameters = tuple(training.model.parameters())
    device = torch.device("cpu")
    data = Cursor()
    formal_before = copy.deepcopy(
        dict(
            model=training.model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            data=data.state_dict(),
            rng=_rng_state(device),
        )
    )
    # The update witness must still describe a real, unchanged AdamW update,
    # although the formal training weights are restored before the first step.
    reference = copy.deepcopy(training.model)
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=namespace["t"].lr, weight_decay=0.1
    )
    for parameter in reference.parameters():
        parameter.grad = parameter.detach().clone()
    reference_optimizer.step()
    exec(optimizer_probe_block(driver), namespace)
    assert training.optimizers == (optimizer,) and training.schedulers == (scheduler,)
    formal_after = dict(
        model=training.model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        data=data.state_dict(),
        rng=_rng_state(device),
    )
    equal(formal_after, formal_before, "formal start unchanged by update probe")
    assert optimizer.state == {} and all(parameter.grad is None for parameter in parameters)
    updates = namespace["updates" if driver == "omni_multimodal_training.py" else "modules"]
    assert set(updates) == {"text.", "visual.", "audio."}
    for group in updates:
        expected_sq = sum(
            (value.double() - formal_before["model"][name].double()).square().sum().item()
            for name, value in reference.state_dict().items()
            if name.startswith(group)
        )
        assert updates[group]["update_squared_sum"] == expected_sq > 0
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in parameters
    }

    def advance():
        sum((p * torch.randn_like(p)).sum() for p in parameters).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        data.consumed += 8
        random.random()
        np.random.random()

    def captured():
        # Exercise the production checkpoint capture/apply protocol on CPU.
        values, _ = _runtime(training.model, (optimizer,), (scheduler,), data, device)
        return copy.deepcopy(values)

    for step in (1, 2):
        advance()
        assert set(optimizer.state) == set(parameters)
        assert all(state["step"].item() == step for state in optimizer.state.values())
        assert scheduler.last_epoch == step and data.consumed == step * 8
        _check_training_state(captured(), [_optimizer_template(optimizer)], step)
    boundary = captured()
    malformed = copy.deepcopy(boundary)
    for state in malformed["optimizer"][0]["state"].values():
        state["step"].add_(1)
    with pytest.raises(DistributedCheckpointError, match="Incomplete optimizer update counters"):
        _check_training_state(malformed, [_optimizer_template(optimizer)], 2)
    advance()
    expected_next = captured()
    _apply(boundary, training.model, (optimizer,), (scheduler,), data, device)
    equal(captured(), boundary, "production CPU checkpoint state restored")
    advance()
    equal(captured(), expected_next, "CPU probe isolation resumed next update")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fail", [False, True])
def test_optimizer_probe_restores_aliases_gradients_buffers_rng_and_modes(fail):
    from pithtrain.modules.strict_checkpoint import _rng_state
    from tests.acceptance.omni_native_resume import equal
    from tests.acceptance.omni_optimizer_probe import isolated_optimizer_probe

    model = nn.Module()
    model.weight = nn.Parameter(torch.arange(6).reshape(2, 3).float() + 1)
    model.alias = model.weight
    model.other = nn.Parameter(torch.ones(2, 3))
    model.unused = nn.Parameter(torch.ones(2))
    model.register_buffer("buffer", torch.tensor([7.0, -0.0]))
    model.register_buffer("buffer_alias", model.buffer)
    model.weight.grad = torch.ones_like(model.weight)
    gradient = model.weight.grad
    model.other.grad = gradient
    before = copy.deepcopy(model.state_dict())
    expected_grad = gradient.clone()
    rng = _rng_state(torch.device("cpu"))
    factory = lambda: (torch.optim.AdamW(model.parameters(), lr=1e-4),)

    def probe():
        with isolated_optimizer_probe(model, factory) as optimizers:
            optimizers[0].step()
            assert not torch.equal(model.weight, before["weight"])
            model.buffer.add_(4)
            model.weight.grad = None
            model.other.grad.zero_()
            model.unused.grad = torch.ones_like(model.unused)
            torch.rand(3)
            random.random()
            np.random.random()
            model.eval()
            if fail:
                raise ValueError("observer failed")

    if fail:
        with pytest.raises(ValueError, match="observer failed"):
            probe()
    else:
        probe()
    equal(model.state_dict(), before, "probe local values")
    assert model.weight is model.alias and model.buffer is model.buffer_alias
    assert model.weight.grad is model.other.grad is gradient
    assert model.unused.grad is None and model.training
    torch.testing.assert_close(gradient, expected_grad, rtol=0, atol=0)
    equal(_rng_state(torch.device("cpu")), rng, "probe RNG")
    assert not torch.cuda.is_initialized()


def test_optimizer_probe_restores_dtensor_local_shards_on_failure(tmp_path, monkeypatch):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    from tests.acceptance.omni_optimizer_probe import isolated_optimizer_probe

    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = nn.Module()
        model.weight = nn.Parameter(
            distribute_tensor(torch.arange(6).reshape(2, 3).float(), mesh, [Shard(0)])
        )
        model.register_buffer("buffer", distribute_tensor(torch.ones(2), mesh, [Shard(0)]))
        model.weight.grad = torch.ones_like(model.weight)
        parameter, gradient, buffer = model.weight, model.weight.grad, model.buffer
        before = parameter.to_local().detach().clone()
        grad_before = gradient.to_local().clone()
        buffer_before = buffer.to_local().clone()

        def forbidden_gather(*args, **kwargs):
            raise AssertionError("Probe must not gather DTensors")

        monkeypatch.setattr(DTensor, "full_tensor", forbidden_gather)
        with pytest.raises(ValueError, match="DTensor observer failed"):
            with isolated_optimizer_probe(
                model, lambda: (torch.optim.AdamW(model.parameters(), lr=1e-4),)
            ) as optimizers:
                optimizers[0].step()
                assert not torch.equal(parameter.to_local(), before)
                buffer.to_local().add_(3)
                parameter.grad = None
                raise ValueError("DTensor observer failed")
        assert model.weight is parameter and model.buffer is buffer and parameter.grad is gradient
        torch.testing.assert_close(parameter.to_local(), before, rtol=0, atol=0)
        torch.testing.assert_close(gradient.to_local(), grad_before, rtol=0, atol=0)
        torch.testing.assert_close(buffer.to_local(), buffer_before, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
    assert not torch.cuda.is_initialized()
