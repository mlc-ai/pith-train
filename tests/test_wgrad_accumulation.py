"""CPU regressions for manual-backward accumulation, using real FSDP parameter methods."""

import ast
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest
import torch
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup


@pytest.fixture
def attach():
    # The operator module imports GPU GEMMs. Execute its actual accumulation helpers unchanged.
    path = Path(__file__).resolve().parents[1] / "pithtrain/operators/wgrad_accumulation.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_attach_main_grads", "_accumulate_autograd_grad"}
    ]
    namespace = dict(
        torch=torch,
        partial=partial,
        Any=Any,
        List=List,
        Tuple=Tuple,
        FSDPParam=FSDPParam,
        FSDPParamGroup=FSDPParamGroup,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_attach_main_grads"]


def fsdp_parameter(dtype=torch.bfloat16, requires_grad=True):
    parameter = object.__new__(FSDPParam)
    parameter.sharded_state = ShardedState.UNSHARDED
    parameter._unsharded_param = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad)
    parameter.unsharded_accumulated_grad = None
    parameter.reduce_dtype = torch.float32
    return parameter


def group_for(parameters):
    reduced = []
    group = SimpleNamespace(fsdp_params=parameters, reduce_grads=False, pre_backward=lambda: None)

    def post_backward():
        if group.reduce_grads:
            reduced.append(
                [
                    None
                    if p.unsharded_accumulated_grad is None
                    else p.unsharded_accumulated_grad.clone()
                    for p in parameters
                ]
            )
            for p in parameters:
                p.unsharded_accumulated_grad = None

    group.post_backward = post_backward
    return group, reduced


def test_manual_microbatches_match_fp32_master_across_steps(attach):
    # Actual BF16 contributions from the failed SFT q_norm coordinate; cancellation changes sign.
    contributions = torch.tensor(
        [
            [-0.00138092041015625, 0.00128936767578125, -0.004913330078125, -0.0017242431640625],
            [-0.0079345703125, 0.0086669921875, 0.0030517578125, 0.002960205078125],
        ],
        dtype=torch.bfloat16,
    )
    expected = contributions.float().sum(dim=1)
    actual = []
    for values, reference in zip(contributions, expected):
        parameter, unused = fsdp_parameter(), fsdp_parameter(requires_grad=False)
        group, reduced = group_for([parameter, unused])
        attach(group, [])  # A norm-only group has no custom linear main_grad pair.
        for step in range(2):
            if step:
                parameter._unsharded_param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            group.reduce_grads = False
            group.pre_backward()
            group.pre_backward()  # Repeated scheduling must not double-install the hook.
            for contribution in values:
                (parameter.unsharded_param * contribution).sum().backward()
                assert parameter.unsharded_param.grad is None
            torch.testing.assert_close(
                parameter.unsharded_accumulated_grad, reference.reshape(1), rtol=0, atol=0
            )
            assert unused.unsharded_accumulated_grad is None
            group.reduce_grads = True
            group.post_backward()
        assert len(reduced) == 2
        actual.append(reduced[-1][0])
    torch.testing.assert_close(sum(actual), contributions.float().sum().reshape(1), rtol=0, atol=0)


def test_autograd_and_deferred_main_grad_are_combined_once(attach):
    parameter = fsdp_parameter()
    group, reduced = group_for([parameter])
    main_grad = torch.empty(1, dtype=torch.float32)
    main_grad.untyped_storage().resize_(0)
    attach(group, [(parameter, main_grad)])
    for _ in range(2):
        group.reduce_grads = False
        group.pre_backward()
        for value in (1.0, -0.5):
            (parameter.unsharded_param * value).sum().backward()
        main_grad.add_(2.0)  # Deferred GEMM contribution to the same weight.
        group.reduce_grads = True
        group.post_backward()
        torch.testing.assert_close(reduced[-1][0], torch.tensor([2.5]), rtol=0, atol=0)
        assert main_grad.untyped_storage().size() == 0


def test_fp32_compute_leaf_keeps_ordinary_autograd_accumulation(attach):
    parameter = fsdp_parameter(torch.float32)
    group, _ = group_for([parameter])
    attach(group, [])
    group.pre_backward()
    for value in (1.0, 0.125, -0.5):
        (parameter.unsharded_param * value).sum().backward()
    torch.testing.assert_close(
        parameter.unsharded_param.grad, torch.tensor([0.625]), rtol=0, atol=0
    )
    assert parameter.unsharded_accumulated_grad is None
