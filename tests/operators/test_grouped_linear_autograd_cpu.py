"""Real CPU grouped_mm/autograd contract; CUDA kernels remain a separate gate.

Load the unchanged production class definitions without importing CUDA-only
dependencies. No grouped matmul or autograd operation is substituted.
"""

import ast
import queue
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def classes():
    root = Path(__file__).resolve().parents[2]
    namespace = dict(
        torch=torch,
        nn=nn,
        F=F,
        partial=partial,
        queue=queue,
        List=List,
        Optional=Optional,
        Callable=Callable,
    )
    for name, selected in [
        ("pithtrain/pipeline/execution.py", {"WeightGradStore"}),
        ("pithtrain/operators/grouped_linear.py", {"GroupedLinearFunc", "GroupedLinear"}),
    ]:
        source = root / name
        tree = ast.parse(source.read_text())
        definitions = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in selected
        ]
        assert {node.name for node in definitions} == selected
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["GroupedLinear"], namespace["WeightGradStore"]


def inputs():
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(3, 16, 16, generator=generator)
    data = torch.randn(32, 16, generator=generator, dtype=torch.bfloat16)
    dy = torch.randn(32, 16, generator=generator, dtype=torch.bfloat16)
    return weight, data, dy, torch.tensor([16, 16, 32], dtype=torch.int32)


def test_cast_weight_returns_gradient_to_fp32_master_and_hooks():
    grouped, _ = classes()
    weight, data, dy, offsets = inputs()
    layer = grouped(3, 16, 16)
    layer.weight = nn.Parameter(weight.clone())
    master = nn.Parameter(weight.clone())
    x, reference_x = data.clone().requires_grad_(), data.clone().requires_grad_()
    calls = []
    layer.weight.register_hook(lambda grad: calls.append(grad.clone()))
    for _ in range(2):
        output = torch.func.functional_call(
            layer, {"weight": layer.weight.bfloat16()}, (x, offsets)
        )
        expected = F.grouped_mm(reference_x, master.bfloat16().transpose(1, 2), offs=offsets)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.backward(dy)
        expected.backward(dy)
    assert layer.weight.grad is not None and layer.weight.grad.dtype == torch.float32
    assert len(calls) == 2
    torch.testing.assert_close(layer.weight.grad, master.grad, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    assert torch.count_nonzero(layer.weight.grad[1]) == 0
    assert not torch.cuda.is_initialized()


def test_leaf_deferred_weight_gradient_matches_immediate_and_accumulates():
    grouped, store = classes()
    weight, data, dy, offsets = inputs()
    eager, deferred = grouped(3, 16, 16), grouped(3, 16, 16)
    eager.weight = nn.Parameter(weight.bfloat16().clone())
    deferred.weight = nn.Parameter(weight.bfloat16().clone())
    x, reference_x = data.clone().requires_grad_(), data.clone().requires_grad_()
    try:
        for index in range(2):
            store.enabled = False
            eager(reference_x, offsets).backward(dy)
            previous = None if deferred.weight.grad is None else deferred.weight.grad.clone()
            store.enabled = True
            deferred(x, offsets).backward(dy)
            if index == 0:
                assert deferred.weight.grad is None
            else:
                torch.testing.assert_close(deferred.weight.grad, previous, rtol=0, atol=0)
            store.flush()
            store.pop()
            torch.testing.assert_close(deferred.weight.grad, eager.weight.grad, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
    finally:
        store.enabled = False
        store.clear()
    assert not torch.cuda.is_initialized()
