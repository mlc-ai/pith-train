"""CPU mathematical and real custom-op/compiled observer controls."""

from typing import Optional, Tuple

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.acceptance import omni_attention_math as gate


@torch.library.custom_op("omni_math_contract::forward", mutates_args=())
def forward_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = gate.mathematical_attention(q, k, v, scale)
    return output, torch.zeros(q.shape[:2], dtype=q.dtype, device=q.device)


@forward_op.register_fake
def forward_fake(q, k, v, scale):
    return torch.empty_like(q), torch.empty(q.shape[:2], dtype=q.dtype, device=q.device)


@torch.library.custom_op("omni_math_contract::backward", mutates_args=())
def backward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    cotangent: torch.Tensor,
    scale: float,
    causal: bool,
    left: Optional[int],
    right: Optional[int],
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Analytic CPU derivative; no nested autograd inside a custom operator.
    groups = q.shape[2] // k.shape[2]
    a = q.transpose(1, 2)
    b = k.repeat_interleave(groups, 2).transpose(1, 2)
    c = v.repeat_interleave(groups, 2).transpose(1, 2)
    dy = cotangent.transpose(1, 2)
    mask = torch.ones(q.shape[1], q.shape[1], dtype=torch.bool).triu(1)
    p = ((a @ b.transpose(-1, -2)) * scale).masked_fill(mask, -torch.inf).softmax(-1)
    dp = dy @ c.transpose(-1, -2)
    ds = p * (dp - (dp * p).sum(-1, keepdim=True)) * scale
    dq = (ds @ b).transpose(1, 2).contiguous()
    dk = (
        (ds.transpose(-1, -2) @ a)
        .transpose(1, 2)
        .reshape(*k.shape[:2], k.shape[2], groups, k.shape[3])
        .sum(3)
    )
    dv = (
        (p.transpose(-1, -2) @ dy)
        .transpose(1, 2)
        .reshape(*v.shape[:2], v.shape[2], groups, v.shape[3])
        .sum(3)
    )
    return dq, dk, dv


@backward_op.register_fake
def backward_fake(q, k, v, output, lse, cotangent, scale, causal, left, right, deterministic):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def setup_context(ctx, inputs, output):
    q, k, v, scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.scale = scale


def backward(ctx, cotangent, unused):
    q, k, v, output, lse = ctx.saved_tensors
    return (*backward_op(q, k, v, output, lse, cotangent, ctx.scale, True, None, None, False), None)


forward_op.register_autograd(backward, setup_context=setup_context)


def tensors():
    generator = torch.Generator().manual_seed(107)
    return tuple(
        torch.randn(1, 7, heads, 8, dtype=torch.float64, generator=generator).requires_grad_()
        for heads in (4, 2, 2)
    )


def independent_record(lengths=None):
    q, k, v = tensors()
    if lengths is not None:
        q, k, v = (value[0].detach().requires_grad_() for value in (q, k, v))
    output = gate.mathematical_attention(q, k, v, 8**-0.5, lengths)
    cotangent = torch.cos(output.detach() * 3)
    gradients = torch.autograd.grad(output, (q, k, v), cotangent)
    return dict(
        kind="varlen" if lengths else "dense",
        lengths=lengths,
        scale=8**-0.5,
        deterministic=False,
        **dict(
            zip(
                ("q", "k", "v", "output", "cotangent", "dq", "dk", "dv"),
                (q, k, v, output, cotangent, *gradients),
                strict=True,
            )
        ),
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_observer_preserves_real_customop_outputs_all_grads_and_fullgraph(monkeypatch, compiled):
    monkeypatch.setitem(gate.BWD_NAMES, "omni_math_contract.backward.default", "dense")

    def operation(q, k, v):
        return forward_op(q, k, v, 8**-0.5)[0].sin()

    if compiled:
        operation = torch.compile(operation, fullgraph=True)
    q, k, v = tensors()
    output = operation(q, k, v)
    baseline = torch.autograd.grad(output.sum(), (q, k, v))
    observer = gate.ActualBackwardCapture()
    second = operation(q, k, v)
    with observer:
        observed = torch.autograd.grad(second.sum(), (q, k, v))
    assert len(observer.records) == 1
    torch.testing.assert_close(output, second, rtol=0, atol=0)
    for left, right in zip(baseline, observed, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    report = gate.check_record(observer.records[0])
    assert report["status"] == "PASSED"
    assert max(report["nse"].values()) < 1e-28
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("lengths", [None, [2, 5], [1, 1, 5]])
def test_fp64_causal_gqa_output_and_all_input_gradient_oracles(lengths):
    report = gate.check_record(independent_record(lengths))
    assert report["status"] == "PASSED"
    assert report["nse"] == dict(output=0, dq=0, dk=0, dv=0)


@pytest.mark.parametrize("tensor", ["output", "dq", "dk", "dv"])
def test_actual_math_fault_cannot_pass_by_omitting_a_tensor(tensor):
    record = independent_record()
    record[tensor] = record[tensor] * -1
    report = gate.check_record(record)
    assert report["status"] == "FAILED"
    assert report["nse"][tensor] == 2
    assert report["thresholds"] == {"output": 1e-3, "dq": 1e-2, "dk": 1e-2, "dv": 1e-2}


def test_packed_reference_does_not_allow_cross_document_attention():
    record = independent_record([2, 5])
    record["lengths"] = [7]
    assert gate.check_record(record)["status"] == "FAILED"


def test_nonfinite_or_missing_gradient_is_not_pass():
    record = independent_record()
    record["dk"][0, 0, 0, 0] = torch.nan
    with pytest.raises(AssertionError, match="Nonfinite"):
        gate.check_record(record)
    record = independent_record()
    del record["dv"]
    with pytest.raises(KeyError):
        gate.check_record(record)


def test_exact_observer_control_rejects_parameter_or_objective_change():
    before = ([torch.ones(2)], torch.tensor(3.0), {"weight": torch.ones(2)}, [torch.ones(3)])
    assert gate.exact_observer_control(before, before)["status"] == "EXACT"
    with pytest.raises(AssertionError, match="parameter gradient"):
        gate.exact_observer_control(
            before, (before[0], before[1], {"weight": torch.zeros(2)}, before[3])
        )
    with pytest.raises(AssertionError, match="objective"):
        gate.exact_observer_control(before, (before[0], torch.tensor(4.0), before[2], before[3]))
    with pytest.raises(AssertionError, match="audio input gradient"):
        gate.exact_observer_control(before, (*before[:3], [torch.zeros(3)]))


def test_unrelated_dispatch_operations_are_returned_without_capture():
    with gate.ActualBackwardCapture() as observer:
        value = torch.arange(8).square()
    assert torch.equal(value, torch.tensor([0, 1, 4, 9, 16, 25, 36, 49]))
    assert observer.records == []
    assert isinstance(observer, TorchDispatchMode)


@pytest.mark.parametrize("lengths", [[0, 7], [2, 4], [-1, 8]])
def test_invalid_packed_lengths_rejected(lengths):
    q, k, v = tensors()
    with pytest.raises(ValueError, match="Invalid packed"):
        gate.mathematical_attention(q[0], k[0], v[0], 1.0, lengths)


@pytest.mark.parametrize("lengths", [[14, 16], [247, 273], [386, 1631], [1260, 14]])
def test_actual_prepared_fixture_lengths_are_preserved_without_truncation(lengths):
    original = list(lengths)
    gate.validate_geometry_lengths(lengths)
    assert lengths == original


@pytest.mark.parametrize("lengths", [[], [10], [0, 4], [10, 2049], [10, 4096]])
def test_unbounded_or_unrelated_geometry_rejected(lengths):
    with pytest.raises(ValueError, match="actual prepared fixture lengths"):
        gate.validate_geometry_lengths(lengths)


def test_nondeterministic_repeat_is_capture_precondition_not_math_failure():
    baseline = ([torch.ones(2)], torch.tensor(3.0), {"weight": torch.ones(2)}, [])
    repeat = (baseline[0], baseline[1], {"weight": torch.zeros(2)}, [])
    with pytest.raises(gate.CaptureTransparencyPreconditionError):
        gate.exact_observer_control(baseline, repeat)
    assert gate.OUTPUT_NSE == 1e-3
    assert gate.GRADIENT_NSE == 1e-2


def test_cli_preserves_capture_precondition_not_run_with_nonzero_exception(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace

    output = tmp_path / "report.json"
    (tmp_path / "checksums.sha256").write_text("CPU contract fixture\\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "omni_attention_math",
            "--component",
            "image",
            "--dataset",
            str(tmp_path),
            "--output",
            str(output),
            "--expected-commit",
            "a" * 40,
        ],
    )
    monkeypatch.setenv("PITHTRAIN_RUN_OMNI_GPU_TESTS", "1")
    monkeypatch.setattr(gate, "verify_source", lambda *args: {"commit": "a" * 40})
    monkeypatch.setattr(gate, "verify_local_fa4", lambda: {"scope": "CPU_CONTRACT_SIMULATION"})
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    batch = SimpleNamespace(
        model_inputs={"input_ids": torch.ones(1, 2, dtype=torch.long)},
        labels=torch.ones(1, 2, dtype=torch.long),
    )
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=1))
    monkeypatch.setattr(
        gate, "fixtures", lambda *args: (None, config, [batch], [batch], "CPU_CONTRACT_SIMULATION")
    )
    monkeypatch.setattr(gate, "prepare_native", lambda *args: object())
    baseline = ([torch.ones(2)], torch.tensor(3.0), {"weight": torch.ones(2)}, [])
    repeat = (baseline[0], baseline[1], {"weight": torch.zeros(2)}, [])
    executions = iter((baseline, repeat))
    monkeypatch.setattr(gate, "execute_original", lambda *args, **kwargs: next(executions))
    with pytest.raises(gate.CaptureTransparencyPreconditionError):
        gate.main()
    report = json.loads(output.read_text())
    assert report["status"] == "NOT_RUN_CAPTURE_TRANSPARENCY_PRECONDITION"
    assert report["mathematical_comparison"] == "NOT_RUN"
    assert report["failure_phase"] == "capture_transparency_precondition"
    assert report["actual_captures"] == report["geometry_cases"] == []
    assert not torch.cuda.is_initialized()
