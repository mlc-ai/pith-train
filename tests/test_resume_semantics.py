"""Strict numerical contracts must distinguish executable policy, not just counters."""

import functools
import math

import pytest
import torch

from pithtrain.modules.resume_semantics import callable_contract, scheduler_contract


def schedule(step, factor):
    return max(factor, math.cos(step))


def test_callable_partial_arguments_and_code_are_bound():
    left = callable_contract(functools.partial(schedule, factor=0.1))
    assert left == callable_contract(functools.partial(schedule, factor=0.1))
    assert left != callable_contract(functools.partial(schedule, factor=0.2))


def test_opaque_callable_and_dynamic_global_access_rejected():
    class Opaque:
        def __call__(self, step):
            return 1.0

    for function in (Opaque(), lambda step: globals()["factor"]):
        with pytest.raises(ValueError, match="Unverifiable"):
            callable_contract(function)


def test_scheduler_counters_do_not_change_stable_contract():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)
    expected = scheduler_contract(scheduler)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    assert scheduler_contract(scheduler) == expected
    scheduler.lr_lambdas = [lambda step: 0.1**step]
    assert scheduler_contract(scheduler) != expected


@pytest.mark.parametrize("factory_name", ["make_wsd_scheduler", "make_constant_scheduler"])
def test_production_scheduler_factory_binds_all_arguments(factory_name, monkeypatch):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    from pithtrain.contexts import training

    # Execute the unchanged factory definitions from the production source.
    # This CPU test isolates them from training.py's optional Triton model imports.
    path = Path(__file__).parents[1] / "pithtrain/modules/training.py"
    parsed = ast.parse(path.read_text())
    definitions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"make_wsd_scheduler", "make_constant_scheduler"}
    ]
    assert len(definitions) == 2
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    namespace = dict(
        __name__="pithtrain.modules.training",
        training=training,
        math=math,
        LambdaLR=torch.optim.lr_scheduler.LambdaLR,
    )
    exec(compile(module, str(path), "exec"), namespace)

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.02)
    monkeypatch.setattr(training, "optimizers", (optimizer,), raising=False)
    factory = namespace[factory_name]
    first = factory(SimpleNamespace(lr=0.02, max_steps=10))[0]
    second = factory(SimpleNamespace(lr=0.02, max_steps=20))[0]
    left, right = scheduler_contract(first), scheduler_contract(second)
    assert left != right
    assert left["identity"]["factory"]["identity"]["arguments"]["s:max_steps"] == 10
    assert right["identity"]["factory"]["identity"]["arguments"]["s:max_steps"] == 20


@pytest.mark.parametrize(
    "left,right",
    [({1: 0.5}, {"1": 0.5}), ((0.5,), [0.5]), ({1: 0.5, "1": 0.1}, {"1": 0.1})],
)
def test_callable_contract_preserves_container_and_mapping_key_types(left, right):
    def factory(value):
        return lambda step: value

    assert callable_contract(factory(left)) != callable_contract(factory(right))
    assert callable_contract(factory(left)) == callable_contract(factory(left))


def test_scheduler_mapping_key_change_cannot_share_policy_contract():
    def make(table):
        return lambda step: table.get(step, 1.0)

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.02)
    integer = torch.optim.lr_scheduler.LambdaLR(optimizer, make({1: 0.5}))
    textual = torch.optim.lr_scheduler.LambdaLR(optimizer, make({"1": 0.5}))
    assert integer.lr_lambdas[0](1) == 0.5
    assert textual.lr_lambdas[0](1) == 1.0
    assert scheduler_contract(integer) != scheduler_contract(textual)
