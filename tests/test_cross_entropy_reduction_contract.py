"""CPU coverage for the real CE wrapper and training objective.

The native Triton kernel remains GPU-only. A Torch kernel substitute exercises
the production autograd/reduction interface without claiming CUDA equivalence.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
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

    def __getitem__(self, grid):
        def launch(**kwargs):
            reduction = kwargs["REDUCTION"]
            self.reductions.append(reduction)
            with torch.enable_grad():
                logits = kwargs["X_ptr"].detach().float().requires_grad_()
                labels = kwargs["Y_ptr"]
                per_row = F.cross_entropy(
                    logits, labels, ignore_index=kwargs["ignore_idx"], reduction="none"
                )
                divisor = (labels != kwargs["ignore_idx"]).sum().clamp_min(1)
                loss = per_row.sum() / (divisor if reduction == "mean" else 1)
                gradient = torch.autograd.grad(loss, logits)[0]
            kwargs["X_ptr"].copy_(gradient.to(kwargs["X_ptr"].dtype))
            kwargs["loss_ptr"].copy_(per_row.detach())

        return launch


class TorchRegisterKernel:
    """Same independent Torch oracle, adapted to the fused register ABI."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def launch(**kwargs):
            return self.kernel[grid](
                X_ptr=kwargs["X"],
                Y_ptr=kwargs["Y"],
                loss_ptr=kwargs["LOSS"],
                ignore_idx=kwargs["IGNORE"],
                REDUCTION="sum",
            )

        return launch


def wrapper():
    kernel = TorchKernel()
    namespace = load_nodes(
        "pithtrain/operators/cross_entropy.py",
        {"CrossEntropy", "cross_entropy"},
        dict(
            torch=torch,
            triton=SimpleNamespace(
                next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                cdiv=lambda n, d: (n + d - 1) // d,
            ),
            cross_entropy_fwd=kernel,
            cross_entropy_register_sum=TorchRegisterKernel(kernel),
        ),
    )
    return namespace["cross_entropy"], kernel


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_strict_sum_preserves_destructive_api_and_reference_rounding(dtype, all_ignored):
    ce, kernel = wrapper()
    generator = torch.Generator().manual_seed(37)
    values = torch.randn((7, 3072), generator=generator).to(dtype)
    labels = torch.randint(3072, (7,), generator=generator)
    labels[1::3] = -100
    if all_ignored:
        labels.fill_(-100)
    expected = values.detach().float().clone().requires_grad_()
    expected_loss = F.cross_entropy(expected, labels, reduction="sum")
    expected_loss.backward()
    prior = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        actual = values.clone().requires_grad_()
        loss = ce(actual, labels, reduction="sum")
        # The public destructive API and BF16 gradient storage remain intact.
        torch.testing.assert_close(actual.detach(), expected.grad.to(dtype), rtol=0, atol=0)
        loss.backward()
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(actual.grad, expected.grad.to(dtype), rtol=0, atol=0)
        assert kernel.reductions == []
        torch.use_deterministic_algorithms(False)
        ce(values.clone(), labels, reduction="sum")
        assert kernel.reductions == ["sum"]
    finally:
        torch.use_deterministic_algorithms(prior, warn_only=warn_only)
