"""CPU coverage for the real CE wrapper and training objective.

The native CuTe/TIRx kernels remain GPU-only. A Torch kernel substitute exercises
the production autograd/reduction interface and SM90 launch cache without claiming
CUDA equivalence.
"""

import ast
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_nodes(relative, names, namespace):
    path = ROOT / relative
    nodes = [
        node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names
    ]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TorchKernel:
    """Independent FP32 CE oracle for the wrapper's destructive kernel interface."""

    def __init__(self):
        self.reductions = []

    def __call__(self, inp, target, ignore_index, reduction):
        self.reductions.append(reduction)
        with torch.enable_grad():
            logits = inp.detach().float().requires_grad_()
            per_row = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
            divisor = (target != ignore_index).sum().clamp_min(1)
            loss = per_row.sum() / (divisor if reduction == "mean" else 1)
            gradient = torch.autograd.grad(loss, logits)[0]
        inp.copy_(gradient.to(inp.dtype))
        return loss.detach(), inp


def wrapper():
    kernel = TorchKernel()
    namespace = dict(torch=torch, cross_entropy_fwd_bwd=kernel)
    load_nodes(
        "pithtrain/operators/cross_entropy.py",
        {"CrossEntropy", "cross_entropy"},
        namespace,
    )
    return namespace["cross_entropy"], kernel


@pytest.mark.parametrize("first", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sm90_launch_cache_keeps_reductions_separate(first, dtype):
    kernel, compiled = TorchKernel(), []

    def compile_kernel(_dtype, _target_dtype, num_cols, *_layout_and_reduction):
        reduction = _layout_and_reduction[-1]
        compiled.append(reduction)

        def launch(flat, target, _row_loss, _count, output, ignore_index):
            loss, _ = kernel(flat.reshape(-1, num_cols), target, ignore_index, reduction)
            output.copy_(loss)

        return launch

    namespace = load_nodes(
        "pithtrain/kernels/cross_entropy_sm90.py",
        {"cross_entropy_fwd_bwd_sm90"},
        dict(
            torch=torch,
            TARGET_TYPES={torch.int64: object()},
            COMPILED={},
            pick_layout=lambda *_: (1, 2, True, 256),
            compile_kernel=compile_kernel,
        ),
    )
    values = torch.linspace(-2, 2, 40, dtype=dtype).reshape(5, 8)
    target = torch.tensor([0, 1, -100, 3, 4])
    reductions = [first, "sum" if first == "mean" else "mean", first]
    for reduction in reductions:
        expected = values.detach().float().clone().requires_grad_()
        reference = F.cross_entropy(expected, target, reduction=reduction)
        reference.backward()
        inp = values.clone()
        loss, gradient = namespace["cross_entropy_fwd_bwd_sm90"](inp, target, -100, reduction)
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected.grad.to(dtype), rtol=0, atol=0)
        assert gradient.data_ptr() == inp.data_ptr()
    assert compiled == reductions[:2]
    assert kernel.reductions == reductions


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("valid_count", [0, 1, 3, 62, 257])
def test_real_wrapper_reduction_loss_and_backward_arity(reduction, dtype, valid_count):
    ce, kernel = wrapper()
    values = torch.linspace(-2, 2, 32, dtype=dtype).repeat(257, 1)
    labels = torch.arange(257) % 32
    labels[valid_count:] = -100
    inputs = values.clone().requires_grad_()
    expected = values.float().requires_grad_()
    expected_loss = F.cross_entropy(expected, labels, reduction="none").sum()
    if reduction == "mean":
        expected_loss = expected_loss / max(valid_count, 1)
    expected_loss.backward()
    loss = ce(inputs, labels, reduction=reduction)
    loss.backward()
    assert kernel.reductions == [reduction]
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, expected.grad.to(dtype), rtol=0, atol=0)
    assert inputs.grad[labels == -100].count_nonzero() == 0
    assert not torch.cuda.is_initialized()


def test_default_remains_mean_and_invalid_reductions_fail_before_kernel():
    ce, kernel = wrapper()
    values = torch.linspace(-1, 1, 40).reshape(5, 8).to(torch.bfloat16)
    labels = torch.tensor([0, 1, -100, 3, 4])
    default, explicit = values.clone().requires_grad_(), values.clone().requires_grad_()
    a, b = ce(default, labels), ce(explicit, labels, reduction="mean")
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(default.grad, explicit.grad, rtol=0, atol=0)
    assert kernel.reductions == ["mean", "mean"]
    for invalid in ("none", "batchmean", None, 0):
        with pytest.raises(ValueError, match="reduction must be"):
            ce(values.clone(), labels, reduction=invalid)
    assert kernel.reductions == ["mean", "mean"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("valid_count", [0, 1, 3, 62, 257])
def test_real_objective_requests_sum_before_low_precision_gradient_storage(dtype, valid_count):
    ce, kernel = wrapper()
    objective = load_nodes(
        "pithtrain/tasks/pretrain_lm.py",
        {"objective"},
        dict(torch=torch, Tuple=Tuple, cross_entropy=ce),
    )["objective"]
    values = torch.linspace(-2, 2, 32, dtype=dtype).repeat(1, 257, 1)
    labels = (torch.arange(257) % 32).reshape(1, 257)
    labels[:, valid_count:] = -100
    actual, expected = values.clone().requires_grad_(), values.float().requires_grad_()
    loss, detached = objective((actual,), (labels,))
    loss.backward()
    reference = F.cross_entropy(expected.flatten(0, 1), labels.flatten(), reduction="none").sum()
    reference.backward()
    assert kernel.reductions == ["sum"]
    assert detached.requires_grad is False
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    torch.testing.assert_close(detached, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual.grad, expected.grad.to(dtype), rtol=0, atol=0)
