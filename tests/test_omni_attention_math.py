"""CPU mathematical and real custom-op/compiled observer controls."""

import ast
from pathlib import Path
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
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    learnable_sink: Optional[torch.Tensor],
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # pithtrain::flash_attn4_mha_bwd's schema with an analytic CPU derivative; no nested autograd
    # inside a custom operator. Without a learnable sink the sink gradient is empty, as in FA4's op.
    groups = q.shape[2] // k.shape[2]
    a = q.transpose(1, 2)
    b = k.repeat_interleave(groups, 2).transpose(1, 2)
    c = v.repeat_interleave(groups, 2).transpose(1, 2)
    dy = do.transpose(1, 2)
    mask = torch.ones(q.shape[1], q.shape[1], dtype=torch.bool).triu(1)
    p = ((a @ b.transpose(-1, -2)) * softmax_scale).masked_fill(mask, -torch.inf).softmax(-1)
    dp = dy @ c.transpose(-1, -2)
    ds = p * (dp - (dp * p).sum(-1, keepdim=True)) * softmax_scale
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
    return dq, dk, dv, q.new_empty(0)


@backward_op.register_fake
def backward_fake(
    q,
    k,
    v,
    o,
    lse,
    do,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    learnable_sink,
    deterministic,
):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), q.new_empty(0)


def setup_context(ctx, inputs, output):
    q, k, v, scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.scale = scale


def backward(ctx, cotangent, unused):
    q, k, v, output, lse = ctx.saved_tensors
    dq, dk, dv, _ = backward_op(
        q, k, v, output, lse, cotangent, ctx.scale, True, None, None, None, False
    )
    return dq, dk, dv, None


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


VARLEN_GRADIENTS = []


@torch.library.custom_op("omni_math_contract::varlen_backward", mutates_args=())
def varlen_backward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    learnable_sink: Optional[torch.Tensor],
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # pithtrain::flash_attn4_varlen_bwd's schema; returns the gradients the test set up.
    return tuple(value.clone() for value in VARLEN_GRADIENTS[-1])


@torch.library.custom_op("omni_math_contract::legacy_backward", mutates_args=())
def legacy_backward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # FA4's backward schema before learnable_sink, which the capture no longer accepts.
    return q.clone(), k.clone(), v.clone()


def varlen_arguments(record, **changes):
    """Positional varlen backward arguments for a packed independent record."""
    cu_seqlens = torch.tensor([0, *record["lengths"]], dtype=torch.int32).cumsum(0).int()
    call = dict(
        q=record["q"].detach(),
        k=record["k"].detach(),
        v=record["v"].detach(),
        o=record["output"].detach(),
        lse=torch.zeros(record["q"].shape[:2], dtype=record["q"].dtype),
        do=record["cotangent"],
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(record["lengths"]),
        max_seqlen_k=max(record["lengths"]),
        softmax_scale=record["scale"],
        causal=True,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        deterministic=False,
    )
    call.update(changes)
    return [call[name] for name in gate.BWD_ARGUMENTS["varlen"]]


def packed_record(monkeypatch, sink_gradient=None):
    operation = torch.ops.omni_math_contract.varlen_backward.default
    monkeypatch.setitem(gate.BWD_NAMES, str(operation), "varlen")
    record = independent_record([3, 4])
    if sink_gradient is None:
        sink_gradient = record["q"].new_empty(0)
    VARLEN_GRADIENTS.append((record["dq"], record["dk"], record["dv"], sink_gradient))
    return record


FA4_SOURCE = Path(__file__).resolve().parents[1] / "pithtrain" / "operators" / "flash_attn_v4.py"


def fa4_custom_op_signatures():
    """{registered name: (argument names, return count)} of the FA4 custom ops, from source.

    torch.library.custom_op infers each schema from its function's signature, so this reads
    them without importing the operators, whose import probes the CUDA device
    (pithtrain.kernels.arch).
    """
    signatures = {}
    for node in ast.parse(FA4_SOURCE.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and ast.unparse(decorator.func) == "torch.library.custom_op"
            ):
                continue
            namespace, operation = decorator.args[0].value.split("::")
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            returns = node.returns
            count = (
                len(returns.slice.elts)
                if isinstance(returns, ast.Subscript) and isinstance(returns.slice, ast.Tuple)
                else 1
            )
            signatures[f"{namespace}.{operation}.default"] = (
                tuple(argument.arg for argument in arguments),
                count,
            )
    return signatures


def test_capture_schema_matches_the_fa4_backward_operation_sources():
    signatures = fa4_custom_op_signatures()
    real = {name: kind for name, kind in gate.BWD_NAMES.items() if name.startswith("pithtrain.")}
    assert sorted(real.values()) == ["dense", "varlen"]
    for name, kind in real.items():
        assert signatures[name] == (gate.BWD_ARGUMENTS[kind], 4)


def test_capture_schema_matches_the_registered_fa4_backward_operations():
    # Importing the operators runs pithtrain.kernels.arch, which probes the CUDA device; without
    # one, the source check above covers the schema.
    if not torch.cuda.is_available():
        pytest.skip("importing the FA4 operators probes a CUDA device")
    pytest.importorskip("pithtrain.operators.flash_attn_v4")
    real = {name: kind for name, kind in gate.BWD_NAMES.items() if name.startswith("pithtrain.")}
    assert sorted(real.values()) == ["dense", "varlen"]
    for name, kind in real.items():
        namespace, operation, _ = name.split(".")
        schema = getattr(getattr(torch.ops, namespace), operation).default._schema
        assert tuple(argument.name for argument in schema.arguments) == gate.BWD_ARGUMENTS[kind]
        assert len(schema.returns) == 4


def test_varlen_capture_binds_packed_lengths_and_the_actual_tensors(monkeypatch):
    record = packed_record(monkeypatch)
    with gate.ActualBackwardCapture() as observer:
        result = varlen_backward_op(*varlen_arguments(record))
    (captured,) = observer.records
    assert captured["kind"] == "varlen" and captured["lengths"] == [3, 4]
    assert captured["scale"] == record["scale"] and captured["deterministic"] is False
    for name in ("q", "k", "v", "output", "cotangent", "dq", "dk", "dv"):
        assert torch.equal(captured[name], record[name].detach())
    assert torch.equal(result[0], record["dq"]) and result[3].numel() == 0
    assert gate.check_record(captured)["status"] == "PASSED"


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(learnable_sink=torch.zeros(4, dtype=torch.float64)), "learnable sink"),
        (dict(window_size_left=2), "causal/full-window"),
        (dict(causal=False), "causal/full-window"),
        (dict(max_seqlen_k=5), "self-attention"),
    ],
)
def test_capture_rejects_calls_outside_the_independent_gate(monkeypatch, change, message):
    record = packed_record(monkeypatch)
    with pytest.raises(RuntimeError, match=message), gate.ActualBackwardCapture() as observer:
        varlen_backward_op(*varlen_arguments(record, **change))
    assert observer.records == []


def test_capture_rejects_a_sink_gradient_without_a_sink(monkeypatch):
    record = packed_record(monkeypatch, sink_gradient=torch.zeros(4, dtype=torch.float64))
    with (
        pytest.raises(RuntimeError, match="sink gradient"),
        gate.ActualBackwardCapture() as observer,
    ):
        varlen_backward_op(*varlen_arguments(record))
    assert observer.records == []


def test_capture_rejects_the_backward_schema_from_before_learnable_sink(monkeypatch):
    operation = torch.ops.omni_math_contract.legacy_backward.default
    monkeypatch.setitem(gate.BWD_NAMES, str(operation), "dense")
    q, k, v = (value.detach() for value in tensors())
    with pytest.raises(RuntimeError, match="Unexpected FA4 backward schema"):
        with gate.ActualBackwardCapture() as observer:
            legacy_backward_op(q, k, v, q, q[..., 0], q, 8**-0.5, True, None, None, False)
    assert observer.records == []
