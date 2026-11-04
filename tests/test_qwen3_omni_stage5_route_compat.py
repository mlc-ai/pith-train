"""CPU reference aggregation only; original default Stage5 gate is unchanged."""

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pithtrain.models.qwen3_omni_eager_compat import eager_stage5_from_routes


@pytest.mark.parametrize("seed", [82, 83])
@pytest.mark.parametrize("k", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("shared_residual", [False, True])
def test_actual_hf_oracle_with_original_unsorted_router_slots(
    seed, k, dtype, permuted, shared_residual
):
    import transformers
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf

    assert transformers.__version__ == "5.17.0"
    assert hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest() == (
        "1c1fc322dfa547c697576e8ad3c88b817aa382abc9db5d966755be3d9fabf7ce"
    )
    torch.manual_seed(seed)
    n, experts, hidden, width = 11, 16, 32, 24
    config = SimpleNamespace(
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=width,
        hidden_act="silu",
        _experts_implementation="eager",
    )
    model = hf.Qwen3OmniMoeThinkerTextExperts(config).to(dtype)
    with torch.no_grad():
        model.gate_up_proj.normal_(0, 0.2)
        model.down_proj.normal_(0, 0.2)
    x = torch.randn(n, hidden, dtype=dtype, requires_grad=True)
    weights, ids = torch.randn(n, experts).softmax(-1).topk(k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype).detach().requires_grad_()
    pointers = {model.down_proj[e].data_ptr(): e for e in range(experts)}
    observed = []
    linear = torch.nn.functional.linear

    def observe(value, weight, bias=None):
        output = linear(value, weight, bias)
        expert = pointers.get(weight.data_ptr())
        if expert is not None:
            slot, token = torch.where((ids == expert).T)
            observed.append((token * k + slot, output))
        return output

    with patch.object(torch.nn.functional, "linear", observe):
        expected = model(x, ids, weights)
    positions = torch.cat([p for p, _ in observed])
    assert torch.equal(positions.sort().values, torch.arange(n * k))
    slots = x.new_zeros(n * k, hidden).index_copy(0, positions, torch.cat([v for _, v in observed]))
    returned = torch.randperm(n * k) if permuted else None
    outputs = slots if returned is None else slots[returned]
    residual = (
        x.view(1, n, hidden)
        if shared_residual
        else torch.randn(1, n, hidden, dtype=dtype, requires_grad=True)
    )
    expected = residual + expected.view_as(residual)
    actual = eager_stage5_from_routes(
        outputs, weights, ids, residual, num_experts=experts, returned_indices=returned
    )
    assert torch.equal(actual, expected)
    targets = (x, weights, model.gate_up_proj, model.down_proj)
    if not shared_residual:
        targets += (residual,)
    cotangent = torch.randn_like(expected)
    a = torch.autograd.grad(actual, targets, cotangent, retain_graph=True)
    b = torch.autograd.grad(expected, targets, cotangent)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sparse_returns_match_independent_per_token_oracle(dtype):
    torch.manual_seed(24)
    ids = torch.tensor([[7, 1, 4], [6, 2, 3], [5, 0, 2]])
    returned = torch.tensor([8, 2, 0, 1])  # token1 has no local output.
    outputs = torch.randn(4, 8, dtype=dtype, requires_grad=True)
    weights = torch.randn(3, 3, dtype=dtype, requires_grad=True)
    residual = torch.randn(1, 3, 8, dtype=dtype, requires_grad=True)
    expected = []
    for token in range(3):
        rows = [
            (int(ids[token, int(pos) % 3]), row, int(pos) % 3)
            for row, pos in enumerate(returned)
            if int(pos) // 3 == token
        ]
        value = outputs.new_zeros(8)
        for _, row, slot in sorted(rows):
            value = value + outputs[row] * weights[token, slot]
        expected.append(value)
    expected = residual + torch.stack(expected).view_as(residual)
    actual = eager_stage5_from_routes(
        outputs, weights, ids, residual, num_experts=8, returned_indices=returned
    )
    assert torch.equal(actual, expected)
    targets = (outputs, weights, residual)
    cotangent = torch.randn_like(residual)
    a = torch.autograd.grad(actual, targets, cotangent, retain_graph=True)
    b = torch.autograd.grad(expected, targets, cotangent)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [0, 3])
def test_empty_returns_have_explicit_zero_vjps(dtype, n):
    outputs = torch.empty(0, 8, dtype=dtype, requires_grad=True)
    weights = torch.ones(n, 2, dtype=dtype, requires_grad=True)
    ids = torch.tensor([[0, 1]]).expand(n, -1)
    residual = torch.randn(1, n, 8, dtype=dtype, requires_grad=True)
    actual = eager_stage5_from_routes(
        outputs,
        weights,
        ids,
        residual,
        num_experts=2,
        returned_indices=torch.empty(0, dtype=torch.int64),
    )
    assert torch.equal(actual, residual)
    grads = torch.autograd.grad(actual, (outputs, weights, residual), torch.ones_like(actual))
    assert torch.equal(grads[0], torch.zeros_like(outputs))
    assert torch.equal(grads[1], torch.zeros_like(weights))
    assert torch.equal(grads[2], torch.ones_like(residual))


def test_argument_contract_rejects_ambiguous_or_invalid_mappings_and_preserves_inputs():
    outputs, weights, ids, residual = (
        torch.ones(4, 8),
        torch.ones(2, 2),
        torch.tensor([[1, 0], [0, 1]]),
        torch.ones(1, 2, 8),
    )
    args = (outputs, weights, ids, residual)
    old = [x.clone() for x in args]
    versions = [x._version for x in args]
    rng = torch.random.get_rng_state().clone()
    eager_stage5_from_routes(*args, num_experts=2)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(
        torch.equal(x, y) and x._version == v for x, y, v in zip(args, old, versions, strict=True)
    )
    for mapping in (
        torch.tensor([0, 0, 1, 2]),
        torch.tensor([-1, 0, 1, 2]),
        torch.tensor([0, 1, 2, 4]),
        torch.tensor([0, 1]),
    ):
        with pytest.raises(ValueError):
            eager_stage5_from_routes(*args, num_experts=2, returned_indices=mapping)
    with pytest.raises(TypeError):
        eager_stage5_from_routes(*args, num_experts=2, returned_indices=torch.arange(4).int())
    with pytest.raises(ValueError):
        eager_stage5_from_routes(outputs[:1], weights, ids, residual, num_experts=2)
    with pytest.raises(ValueError):
        eager_stage5_from_routes(outputs, weights, torch.zeros_like(ids), residual, num_experts=2)
    with pytest.raises(ValueError):
        eager_stage5_from_routes(*args, num_experts=True)
    with pytest.raises(TypeError):
        eager_stage5_from_routes(outputs.double(), weights, ids, residual, num_experts=2)
    with pytest.raises(ValueError):
        eager_stage5_from_routes(outputs.to("meta"), weights, ids, residual, num_experts=2)
