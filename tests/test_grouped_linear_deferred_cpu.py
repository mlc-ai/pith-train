"""CPU regressions using complete production modules and native F.grouped_mm.

No GEMM/autograd/WeightGradStore/FSDP method is replaced or AST-extracted.
Only unavailable GPU dependency modules and the eager pipeline package entry
point are isolated during import. Their callable shims fail on use. This does
not exercise CUDA main_grad kernels, streams, collectives, or native FSDP setup.

Normal runs load this checkout. For the three saved-nonleaf negative controls
only, PITHTRAIN_GROUPED_LINEAR_CPU_BASELINE_SOURCE may name the frozen e2 file;
its exact SHA256 is required below. The override never changes production files.
"""

import contextlib
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState

BASELINE_ENV = "PITHTRAIN_GROUPED_LINEAR_CPU_BASELINE_SOURCE"
BASELINE_SHA256 = "4060b42f94e0147ad9c248f227cb8e530b2635bd004a1d2bafd1ea35b6625409"
WIDTH = 16


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _empty(store):
    assert not store.cache and store.funcs_queue.empty()


@pytest.fixture
def production(request, record_property):
    """Load whole modules with reversible, function-local import isolation."""
    assert str(torch.__version__).split("+")[0].startswith("2.13."), (
        "Requires project-supported PyTorch 2.13"
    )
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert not torch.cuda.is_initialized()
    repo = Path(__file__).resolve().parents[1]
    source = repo / "pithtrain/operators/grouped_linear.py"
    override = os.environ.get(BASELINE_ENV)
    if override:
        assert request.node.originalname == "test_saved_nonleaf_lifecycle", (
            "Baseline override is restricted to the three saved-nonleaf negative controls"
        )
        source = Path(override)
        assert source.is_absolute() and source.is_file()
        assert _sha(source) == BASELINE_SHA256, "Baseline operator is not frozen e2"
    native_grouped_mm = F.grouped_mm
    torch_root = Path(torch.__file__).resolve().parent
    assert Path(native_grouped_mm.__code__.co_filename).resolve().is_relative_to(torch_root)
    gpu_calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            gpu_calls.append(name)
            raise AssertionError(f"GPU-only dependency used by CPU test: {name}")

        return fail

    def shim(name, attributes):
        module = ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        for attribute in attributes:
            setattr(module, attribute, forbidden(f"{name}.{attribute}"))
        return module

    class UnavailableArchitecture:
        def __ge__(self, other):
            return forbidden("linear.ARCH_MAJOR comparison")(other)

        def __lt__(self, other):
            return forbidden("linear.ARCH_MAJOR comparison")(other)

    class UnavailableBF16Linear(nn.Module):
        def __init__(self, *args, **kwargs):
            forbidden("linear.BF16Linear construction")(*args, **kwargs)

    # These real package __init__ files have no CUDA initialization. The pipeline
    # __init__, in contrast, eagerly imports DualPipeV and unrelated GPU modules.
    pithtrain = importlib.import_module("pithtrain")
    operators = importlib.import_module("pithtrain.operators")
    assert Path(pithtrain.__file__).resolve() == (repo / "pithtrain/__init__.py").resolve()
    with pytest.MonkeyPatch.context() as patch:
        pipeline = ModuleType("pithtrain.pipeline")
        pipeline.__path__ = [str(repo / "pithtrain/pipeline")]
        pipeline.__spec__ = importlib.machinery.ModuleSpec(
            "pithtrain.pipeline", loader=None, is_package=True
        )
        patch.setitem(sys.modules, "pithtrain.pipeline", pipeline)
        patch.setattr(pithtrain, "pipeline", pipeline, raising=False)
        deep_gemm = shim(
            "deep_gemm",
            [
                "k_grouped_bf16_gemm_tn_contiguous",
                "m_grouped_fp8_fp4_gemm_nt_contiguous",
                "m_grouped_fp8_gemm_nt_contiguous",
                "k_grouped_fp8_gemm_tn_contiguous",
                "k_grouped_fp8_gemm_nt_contiguous",
            ],
        )
        quantize = shim(
            "pithtrain.operators.deepgemm_quantize",
            [
                "fp8cast_blockwise_transpose_batched",
                "fp8cast_rowwise_colwise",
                "fp8cast_rowwise_kmajor",
            ],
        )
        linear = shim("pithtrain.operators.linear", [])
        linear.ARCH_MAJOR = UnavailableArchitecture()
        linear.BF16Linear = UnavailableBF16Linear
        for module in (deep_gemm, quantize, linear):
            patch.setitem(sys.modules, module.__name__, module)
            if module.__name__.startswith("pithtrain.operators."):
                patch.setattr(operators, module.__name__.rsplit(".", 1)[1], module, raising=False)

        def load(name, path, parent):
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, name, module)
            patch.setattr(parent, name.rsplit(".", 1)[1], module, raising=False)
            spec.loader.exec_module(module)  # Whole file, never selected AST nodes.
            assert Path(module.__file__).resolve() == path.resolve()
            return module

        execution = load(
            "pithtrain.pipeline.execution", repo / "pithtrain/pipeline/execution.py", pipeline
        )
        grouped = load("pithtrain.operators.grouped_linear", source, operators)
        accumulation = load(
            "pithtrain.operators.wgrad_accumulation",
            repo / "pithtrain/operators/wgrad_accumulation.py",
            operators,
        )
        assert grouped.F is F and grouped.F.grouped_mm is native_grouped_mm
        assert grouped.WeightGradStore is execution.WeightGradStore
        assert grouped.GroupedLinearFunc.__module__ == grouped.__name__
        assert execution.run_backward.__module__ == execution.__name__
        assert hasattr(grouped, "FP8GroupedLinear")  # Full module body was executed.
        assert accumulation.FSDPParam is FSDPParam
        fsdp_source = Path(importlib.import_module(FSDPParam.__module__).__file__).resolve()
        native_source = Path(native_grouped_mm.__code__.co_filename).resolve()
        assert fsdp_source.is_relative_to(torch_root)
        sources = {
            name: {
                "path": str(Path(module.__file__).resolve()),
                "sha256": _sha(Path(module.__file__)),
            }
            for name, module in (
                ("grouped_linear", grouped),
                ("execution", execution),
                ("wgrad_accumulation", accumulation),
            )
        }
        record_property(
            "cpu_source_coverage",
            json.dumps(
                {
                    "sources": sources,
                    "baseline_operator_override": bool(override),
                    "full_module_import": True,
                    "AST_extraction": False,
                    "native_F_grouped_mm": True,
                    "torch": str(torch.__version__),
                    "torch_path": torch.__file__,
                    "torch_git_version": getattr(torch.version, "git_version", None),
                    "native_functional_source": {
                        "path": str(native_source),
                        "sha256": _sha(native_source),
                    },
                    "installed_FSDPParam_source": {
                        "path": str(fsdp_source),
                        "sha256": _sha(fsdp_source),
                    },
                    "threads": torch.get_num_threads(),
                    "interop_threads": torch.get_num_interop_threads(),
                    "pipeline_package_init_bypassed": True,
                    "GPU_dependency_shims": [
                        deep_gemm.__name__,
                        quantize.__name__,
                        linear.__name__,
                    ],
                },
                sort_keys=True,
            ),
        )
        store = execution.WeightGradStore
        _empty(store)
        try:
            yield SimpleNamespace(
                grouped=grouped,
                store=store,
                backward=execution.run_backward,
                accumulation=accumulation,
                record=record_property,
            )
        finally:
            # Test isolation after pytest has captured any original exception;
            # this is not production rollback or evidence of safe retry.
            record_property(
                "queue_at_teardown",
                json.dumps({"cache": len(store.cache), "batches": store.funcs_queue.qsize()}),
            )
            store.enabled = False
            store.clear()
            assert F.grouped_mm is native_grouped_mm
            assert not gpu_calls, f"Unexpected GPU dependency calls: {gpu_calls}"
            assert not torch.cuda.is_initialized()


def _owner(dtype=torch.bfloat16, requires_grad=True):
    # Use the existing CPU grouped_mm tests' 16-wide layout. Zero padding leaves
    # the original 2x2 active values, coefficients, pullback and exact oracle intact.
    values = torch.zeros((1, WIDTH, WIDTH), dtype=dtype, device="cpu")
    values[0, :2, :2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device="cpu")
    owner = nn.Module()
    owner.weight = nn.Parameter(values, requires_grad=requires_grad)
    return owner


def _basis(row, column, coefficient):
    result = torch.zeros((1, WIDTH, WIDTH), dtype=torch.float32, device="cpu")
    result[0, row, column] = coefficient
    return result


class _Observer:
    def __init__(self, parameter, fault=False):
        self.phase = "F"
        self.events = []
        self.sink = torch.zeros_like(parameter, dtype=torch.float32)
        self.fault = fault
        self.w_count = 0
        self.handle = parameter.register_post_accumulate_grad_hook(self.hook)

    def hook(self, parameter):
        grad = None if parameter.grad is None else parameter.grad.detach().clone()
        event = dict(
            phase=self.phase,
            grad=grad,
            promoted=False,
            multithreading=torch.autograd.is_multithreading_enabled(),
        )
        self.events.append(event)
        if grad is None:  # Legal Torch 2.13 B callback when weight grad is None.
            return
        if self.phase == "W":
            self.w_count += 1
            if self.fault and self.w_count == 2:
                raise _InjectedHookError("second defined W hook")
        self.sink.add_(grad.float())
        parameter.grad = None
        event["promoted"] = True

    def reset(self):
        self.events.clear()
        self.sink.zero_()
        self.w_count = 0
        self.phase = "F"

    def assert_counts(self, b_defined, b_none, w_defined):
        assert sum(e["phase"] == "B" and e["grad"] is not None for e in self.events) == b_defined
        assert sum(e["phase"] == "B" and e["grad"] is None for e in self.events) == b_none
        assert sum(e["phase"] == "W" and e["grad"] is not None for e in self.events) == w_defined
        assert not any(e["phase"] == "W" and e["grad"] is None for e in self.events)


class _InjectedHookError(RuntimeError):
    pass


def _graph(api, calls, side=False):
    loss = None
    inputs = []
    for weight, out_row, in_column, coefficient in calls:
        x = torch.zeros((WIDTH, WIDTH), dtype=torch.bfloat16, device="cpu")
        x[0, in_column] = 1
        x.requires_grad_()
        dy = torch.zeros_like(x)
        dy[0, out_row] = coefficient
        output = api.grouped.GroupedLinearFunc.apply(
            x, weight, torch.tensor([WIDTH], dtype=torch.int32, device="cpu")
        )
        # CPU matmul is an independent tiny analytic reference, never a substitute
        # for production grouped_mm. All production forward/B/W calls stay native.
        _same(output, (x.detach().float() @ weight.detach()[0].float().T).bfloat16())
        term = (output.float() * dy.float()).sum()
        loss = term if loss is None else loss + term
        inputs.append((x, (dy.float() @ weight.detach()[0].float()).bfloat16()))
    if side:
        loss = loss + calls[0][0].float().sum() * 0
    return loss, inputs


def _backward_and_flush(api, loss, inputs, observer=None):
    if observer is not None:
        observer.phase = "B"
    api.store.enabled = True
    try:
        api.backward([loss], [torch.ones_like(loss)])
    finally:
        api.store.enabled = False
    for x, expected in inputs:
        assert x.grad is not None
        _same(x.grad, expected)
    queued = len(api.store.cache)
    api.store.flush()  # Empty lists are still a real batch that must be popped.
    assert not api.store.cache and api.store.funcs_queue.qsize() == 1
    if observer is not None:
        observer.phase = "W"
    return queued


@pytest.mark.parametrize("kind", ["consumed", "shared", "fresh"])
def test_saved_nonleaf_lifecycle(production, kind):
    api = production
    owner = _owner(torch.float32)
    observer = _Observer(owner.weight)
    for epoch in range(2):
        owner.zero_grad(set_to_none=True)
        observer.reset()
        _empty(api.store)
        scale = 1.0 if epoch == 0 else 0.5
        shared = owner.weight.square().bfloat16()
        coefficients = [2.0] if kind == "consumed" else [2.0, 3.0]
        calls = [
            (owner.weight.square().bfloat16() if kind == "fresh" else shared, 0, 0, c * scale)
            for c in coefficients
        ]
        loss, inputs = _graph(api, calls, side=kind == "consumed")
        queued = _backward_and_flush(api, loss, inputs, observer)
        api.record(f"{kind}_epoch{epoch}_queued", queued)
        # Pop BEFORE checking the desired route: the frozen e2 negative control
        # must expose its real W saved-graph RuntimeError, not an early assertion.
        api.store.pop()
        assert queued == 0
        _empty(api.store)
        observer.assert_counts(b_defined=1, b_none=0, w_defined=0)
        _same(observer.sink, _basis(0, 0, 2 * sum(coefficients) * scale))
        assert owner.weight.grad is None


@pytest.mark.parametrize("ambient,no_grad_w", [(True, False), (True, True), (False, False)])
def test_raw_leaf_deferred_queue_and_fp32_sum(production, ambient, no_grad_w):
    api = production
    owner = _owner()
    observer = _Observer(owner.weight)
    with torch.autograd.set_multithreading_enabled(ambient):
        for epoch in range(2):
            owner.zero_grad(set_to_none=True)
            observer.reset()
            _empty(api.store)
            scale = 1.0 if epoch == 0 else 0.5
            calls = [(owner.weight, 0, 0, c * scale) for c in (128.0, 0.5, -128.0)]
            loss, inputs = _graph(api, calls)
            assert _backward_and_flush(api, loss, inputs, observer) == 3
            assert owner.weight.grad is None and not observer.sink.any()
            observer.assert_counts(b_defined=0, b_none=1, w_defined=0)
            assert torch.autograd.is_multithreading_enabled() == ambient
            with torch.no_grad() if no_grad_w else contextlib.nullcontext():
                api.store.pop()
            _empty(api.store)
            observer.assert_counts(b_defined=0, b_none=1, w_defined=3)
            assert all(
                e["multithreading"] == (False if e["phase"] == "B" else ambient)
                for e in observer.events
            )
            assert torch.autograd.is_multithreading_enabled() == ambient
            _same(observer.sink, _basis(0, 0, 0.5 * scale))
            assert owner.weight.grad is None


@pytest.mark.parametrize("kind", ["cast", "view"])
def test_shared_nonleaf_exact_two_round_oracle(production, kind):
    api = production
    owner = _owner(torch.float32 if kind == "cast" else torch.bfloat16)
    observer = _Observer(owner.weight)
    for epoch in range(2):
        owner.zero_grad(set_to_none=True)
        observer.reset()
        _empty(api.store)
        scale = 1.0 if epoch == 0 else 0.5
        if kind == "cast":
            shared = owner.weight.bfloat16()
            calls = [(shared, 0, 0, c * scale) for c in (128.0, 0.5, -128.0)]
            expected = _basis(0, 0, 0.5 * scale)
        else:
            shared, transposed = owner.weight.view_as(owner.weight), owner.weight.transpose(1, 2)
            calls = [
                (shared, 0, 0, 128.0 * scale),
                (shared, 0, 0, -128.0 * scale),
                (transposed, 1, 0, 0.5 * scale),
                (transposed, 1, 0, 0.25 * scale),
            ]
            expected = _basis(0, 1, 0.75 * scale)
        loss, inputs = _graph(api, calls)
        queued = _backward_and_flush(api, loss, inputs, observer)
        api.store.pop()
        assert queued == 0
        _empty(api.store)
        observer.assert_counts(b_defined=1, b_none=0, w_defined=0)
        _same(observer.sink, expected)  # No weakened tolerance for shared BF16 accumulation.
        assert owner.weight.grad is None


def test_frozen_alias_stays_without_gradient(production):
    api = production
    owner = _owner(requires_grad=False)
    for epoch in range(2):
        owner.zero_grad(set_to_none=True)
        alias = owner.weight.view_as(owner.weight)
        scale = 1.0 if epoch == 0 else 0.5
        loss, inputs = _graph(api, [(alias, 0, 0, c * scale) for c in (128.0, 0.5, -128.0)])
        _backward_and_flush(api, loss, inputs)
        api.store.pop()
        _empty(api.store)
        assert owner.weight.grad is None and alias.grad is None


def test_w_hook_error_preserves_partial_state(production):
    api = production
    owner = _owner()
    observer = _Observer(owner.weight, fault=True)
    loss, inputs = _graph(api, [(owner.weight, 0, 0, c) for c in (128.0, 0.5, -128.0)])
    assert _backward_and_flush(api, loss, inputs, observer) == 3
    with pytest.raises(_InjectedHookError, match="second defined W hook"):
        api.store.pop()
    observer.assert_counts(b_defined=0, b_none=1, w_defined=2)
    contributions = [e for e in observer.events if e["phase"] == "W"]
    assert [e["promoted"] for e in contributions] == [True, False]
    _same(observer.sink, contributions[0]["grad"].float())
    _same(owner.weight.grad, contributions[1]["grad"])
    # The batch was dequeued before callbacks ran. Empty != all contributions done.
    _empty(api.store)


def test_empty_batch_and_empty_queue_contract(production):
    store = production.store
    for _ in range(2):
        _empty(store)
        store.flush()
        assert store.funcs_queue.qsize() == 1
        store.pop()
        _empty(store)
    with pytest.raises(AssertionError, match="Pop empty queue"):
        store.pop()


def test_deferred_w_uses_real_fsdp_accumulation_methods(production):
    api = production
    owner = _owner()
    parameter = object.__new__(FSDPParam)
    parameter.sharded_state = ShardedState.UNSHARDED
    parameter._unsharded_param = owner.weight
    parameter.unsharded_accumulated_grad = None
    parameter.reduce_dtype = torch.float32
    assert (
        parameter.accumulate_unsharded_grad_if_needed.__func__
        is FSDPParam.accumulate_unsharded_grad_if_needed
    )
    assert (
        parameter.to_accumulated_grad_if_needed.__func__ is FSDPParam.to_accumulated_grad_if_needed
    )
    fsdp_source = Path(importlib.import_module(FSDPParam.__module__).__file__).resolve()
    assert (
        Path(parameter.accumulate_unsharded_grad_if_needed.__func__.__code__.co_filename).resolve()
        == fsdp_source
    )
    assert (
        Path(parameter.to_accumulated_grad_if_needed.__func__.__code__.co_filename).resolve()
        == fsdp_source
    )
    group = SimpleNamespace(
        fsdp_params=[parameter],
        reduce_grads=False,
        pre_backward=lambda: None,
        post_backward=lambda: None,
    )
    api.accumulation._attach_main_grads(group, [])
    for epoch in range(2):
        owner.zero_grad(set_to_none=True)
        parameter.unsharded_accumulated_grad = None
        group.pre_backward()
        hook_count = len(owner.weight._post_accumulate_grad_hooks)
        group.pre_backward()
        assert len(owner.weight._post_accumulate_grad_hooks) == hook_count == 1
        scale = 1.0 if epoch == 0 else 0.5
        loss, inputs = _graph(api, [(owner.weight, 0, 0, c * scale) for c in (128.0, 0.5, -128.0)])
        assert _backward_and_flush(api, loss, inputs) == 3
        assert owner.weight.grad is None and parameter.unsharded_accumulated_grad is None
        api.store.pop()
        _empty(api.store)
        assert owner.weight.grad is None
        assert parameter.unsharded_accumulated_grad.dtype == torch.float32
        _same(parameter.unsharded_accumulated_grad, _basis(0, 0, 0.5 * scale))
