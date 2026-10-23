"""CPU contracts for FA4 policy transport, not FA4 numerical acceptance.

The real torch.library schema, fake kernels and registered autograd run here.
Only the external CUDA-only FA4 entry points are replaced with CPU test doubles.
"""

import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pithtrain.contexts import training
from tests.cpu_helpers import setup_training_fixture as setup_training_fixture

ROOT = Path(__file__).parents[1]


@pytest.fixture
def wrapper(monkeypatch):
    calls = []
    failure = []

    def forward(q, k, v, **kwargs):
        calls.append(("forward", kwargs))
        shape = (q.shape[0], q.shape[2], q.shape[1]) if q.ndim == 4 else (q.shape[1], q.shape[0])
        return q.clone(), torch.zeros(shape, dtype=torch.float32)

    def backward(q, k, v, out, dout, lse, **kwargs):
        calls.append(("backward", kwargs))
        if failure:
            raise NotImplementedError(failure[0])
        return torch.ones_like(q), torch.full_like(k, 2), torch.full_like(v, 3)

    fake = types.ModuleType("flash_attn.cute.interface")
    fake._flash_attn_fwd, fake._flash_attn_bwd = forward, backward
    for name in ("flash_attn", "flash_attn.cute"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", fake)
    path = ROOT / "pithtrain/operators/flash_attn_v4.py"
    namespace = "test_fa4_policy_" + uuid.uuid4().hex
    source = path.read_text().replace('"pithtrain::', f'"{namespace}::')
    module = types.ModuleType(namespace)
    monkeypatch.setitem(sys.modules, namespace, module)
    exec(compile(source, str(path), "exec"), module.__dict__)
    monkeypatch.setattr(training, "deterministic_attention", False)
    return module, calls, failure


def invoke(module, packed, policy=None):
    shape = (5, 2, 4) if packed else (1, 5, 2, 4)
    q, k, v = (torch.randn(shape, requires_grad=True) for _ in range(3))
    kwargs = dict(softmax_scale=0.5, causal=True, window_size=(3, 0))
    if policy is not None:
        kwargs["deterministic"] = policy
    if packed:
        offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
        out = module.flash_attn_varlen_func(q, k, v, offsets, 5, **kwargs)
    else:
        out = module.flash_attn_func(q, k, v, **kwargs)
    return out, (q, k, v)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "strict,explicit,expected",
    [(False, None, False), (True, None, True), (False, True, True), (True, False, False)],
)
def test_forward_captures_policy_for_backward(
    wrapper, monkeypatch, packed, strict, explicit, expected
):
    module, calls, _ = wrapper
    monkeypatch.setattr(training, "deterministic_attention", strict)
    out, inputs = invoke(module, packed, explicit)
    # Changing runtime state after forward must not alter its backward contract.
    monkeypatch.setattr(training, "deterministic_attention", not expected)
    out.sum().backward()
    forward = next(options for name, options in calls if name == "forward")
    backward = next(options for name, options in calls if name == "backward")
    assert "deterministic" not in forward  # Upstream forward has no such option.
    assert backward["deterministic"] is expected
    for options in (forward, backward):
        assert options["softmax_scale"] == 0.5
        assert options["causal"] is True
        assert options["window_size_left"] == 3
        assert options["window_size_right"] == 0
        if packed:
            assert torch.equal(options["cu_seqlens_q"], torch.tensor([0, 2, 5], dtype=torch.int32))
            assert options["max_seqlen_q"] == options["max_seqlen_k"] == 5
    for index, value in enumerate(inputs, 1):
        torch.testing.assert_close(value.grad, torch.full_like(value, index), rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_unsupported_determinism_propagates_without_retry(wrapper, packed):
    module, calls, failure = wrapper
    failure.append("FA4 deterministic backward unsupported")
    out, _ = invoke(module, packed, True)
    with pytest.raises(NotImplementedError, match="deterministic backward unsupported"):
        out.sum().backward()
    assert [options["deterministic"] for name, options in calls if name == "backward"] == [True]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_fullgraph_aot_autograd_preserves_policy(wrapper, monkeypatch, packed, strict):
    module, calls, _ = wrapper
    monkeypatch.setattr(training, "deterministic_attention", strict)
    shape = (5, 2, 4) if packed else (1, 5, 2, 4)
    inputs = tuple(torch.randn(shape, requires_grad=True) for _ in range(3))
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32)

    def forward(q, k, v):
        if packed:
            return module.flash_attn_varlen_func(q, k, v, offsets, 5, 0.5, causal=True)
        return module.flash_attn_func(q, k, v, 0.5, causal=True)

    compiled = torch.compile(forward, fullgraph=True, backend="aot_eager")
    compiled(*inputs).sum().backward()
    assert [options["deterministic"] for name, options in calls if name == "backward"] == [strict]


@pytest.mark.parametrize("strict", [None, False, True])
def test_setup_training_sets_policy_before_model_construction(setup_training, strict):
    runtime = SimpleNamespace(deterministic_attention=not strict)
    observed = []
    cfg, setup = setup_training(
        runtime, lambda *_: observed.append(runtime.deterministic_attention), lambda _: None
    )
    if strict is not None:
        cfg.training.strict_resume = strict
    setup(cfg)
    assert observed == [bool(strict)]
    assert runtime.deterministic_attention is bool(strict)
