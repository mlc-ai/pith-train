"""Check production compiled text stage5 and its eager arithmetic body.

Extracting these pure methods avoids unrelated CUDA-only kernel imports. The
production bodies are executed without replacement in both controls.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.cpu_helpers import layer_type


@pytest.mark.parametrize("is_moe,ep", [(False, 1), (True, 1), (True, 2)])
def test_text_stage5_fullgraph_output_and_all_gradients(is_moe, ep):
    torch.manual_seed(19)
    layer = layer_type(ep)()
    layer.is_moe = is_moe
    layer.mlp = SimpleNamespace(deterministic_token_layout=False)
    residual = torch.randn(1, 4, 8, requires_grad=True)
    weights = torch.randn(4, 2, requires_grad=True) if is_moe else None
    indexes = torch.tensor([6, 1, 4, 3, 2]) if ep == 2 else torch.arange(8)
    output = (
        torch.randn(len(indexes), 8, requires_grad=True)
        if is_moe
        else torch.randn_like(residual, requires_grad=True)
    )
    if is_moe:
        # Independent per-token weighted sum also covers sparse, unordered EP2
        # assignments and tokens with no local experts.
        contributions = []
        for token in range(4):
            selected = indexes // 2 == token
            contributions.append(
                (output[selected] * weights.reshape(-1)[indexes[selected], None]).sum(0)
            )
        expected = residual + torch.stack(contributions).reshape_as(residual)
    else:
        expected = residual + output
    values = (residual, output, weights) if is_moe else (residual, output)
    expected_grads = torch.autograd.grad(expected.square().sum(), values)

    def eager(*args):
        return type(layer).forward_stage5.__wrapped__(layer, *args)

    for call in (eager, layer.forward_stage5):
        actual = call(output, indexes if is_moe else None, weights, residual)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        actual_grads = torch.autograd.grad(actual.square().sum(), values)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(
    "top_k,dtype",
    [(1, torch.bfloat16), (2, torch.bfloat16), (8, torch.bfloat16), (8, torch.float32)],
)
@pytest.mark.parametrize("deterministic_token_layout", [False, True])
def test_default_combine_matches_actual_hf_grouped_mm_output_and_all_gradients(
    top_k, dtype, deterministic_token_layout, monkeypatch
):
    import transformers
    import transformers.integrations.moe as hf_moe
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextExperts,
    )

    assert transformers.__version__ == "5.17.0"
    torch.manual_seed(82)
    n, experts, hidden, width = 11, 16, 32, 24
    config = SimpleNamespace(
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=width,
        hidden_act="silu",
        _experts_implementation="grouped_mm",  # HF 5.17.0 default for this class
    )
    hf = Qwen3OmniMoeThinkerTextExperts(config).to(dtype)
    with torch.no_grad():
        hf.gate_up_proj.normal_(0, 0.2)
        hf.down_proj.normal_(0, 0.2)
    inputs = torch.randn(n, hidden, dtype=dtype, requires_grad=True)
    weights, ids = torch.randn(n, experts).softmax(-1).topk(top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype).detach().requires_grad_()
    observed = []
    grouped_linear = hf_moe._grouped_linear

    def observe(value, weight, offsets, *args, **kwargs):
        output = grouped_linear(value, weight, offsets, *args, **kwargs)
        if weight is hf.down_proj:
            observed.append(output)
        return output  # The actual HF forward and every gradient remain unchanged.

    with monkeypatch.context() as patch:
        patch.setattr(hf_moe, "_grouped_linear", observe)
        expected = hf(inputs, ids, weights)
    (rows,) = observed
    assert rows.shape == (n * top_k, hidden)
    # HF groups rows with this exact sort; undo it to recover router-slot order.
    _, by_expert = torch.sort(ids.reshape(-1))
    slots = rows.new_zeros(rows.shape).index_copy(0, by_expert, rows)
    if top_k > 1:
        assert not torch.equal(ids, ids.sort(dim=-1).values)  # production slots stay unsorted
    targets = (inputs, weights, hf.gate_up_proj, hf.down_proj)
    cotangent = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, targets, cotangent, retain_graph=True)
    for ep in (1, 2):
        layer = layer_type(ep)()
        layer.is_moe = True
        layer.mlp = SimpleNamespace(deterministic_token_layout=deterministic_token_layout)
        # Any complete EP return permutation must restore the same token/slot values;
        # this CPU check deliberately does not assume stable dispatch ordering.
        returned_indices = torch.randperm(n * top_k) if ep == 2 else None
        if returned_indices is not None:
            assert not torch.equal(returned_indices, torch.arange(n * top_k))
        outputs = slots if returned_indices is None else slots[returned_indices]

        # Preserve the original oracle/control matrix: eager for each dtype;
        # compiled for the BF16 top-8 production geometry. Do not double-compile.
        def eager(*args):
            return type(layer).forward_stage5.__wrapped__(layer, *args)

        calls = [eager]
        if top_k == 8 and dtype == torch.bfloat16:
            calls.append(layer.forward_stage5)
        for call_index, call in enumerate(calls):
            actual = call(outputs, returned_indices, weights, torch.zeros_like(expected))
            assert torch.equal(actual, expected)
            # The final compiled backward may donate buffers; release its graph normally.
            retain_graph = not (ep == 2 and call_index == len(calls) - 1)
            actual_grad = torch.autograd.grad(actual, targets, cotangent, retain_graph=retain_graph)
            assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad, strict=True))
    if top_k == 8 and dtype == torch.bfloat16:
        # HF eager's serial BF16 index_add_ in expert-ID order is the opt-in
        # hf_eager_expert_id_serial contract, not the default; this oracle must reject it.
        products = slots.view(n, top_k, hidden) * weights[..., None]
        order = ids.argsort(dim=-1)
        serial = torch.zeros_like(expected)
        for slot in range(top_k):
            serial = serial + products[torch.arange(n), order[:, slot]]
        assert not torch.equal(serial, expected)


@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("deterministic_token_layout", [False, True])
def test_talker_keeps_routed_shared_residual_addition_order(ep, deterministic_token_layout):
    torch.manual_seed(83)
    source = Path(__file__).parents[1] / "pithtrain/models/qwen3_omni_talker_pipeline.py"
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3OmniMoeTalkerPipelineDecoderLayer"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_stage5"
    )
    parent = layer_type(ep)()
    namespace = dict(parent._forward_stage5.__globals__)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    layer = SimpleNamespace(
        mlp=SimpleNamespace(deterministic_token_layout=deterministic_token_layout)
    )
    outputs = torch.randn(24, 8, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    shared = torch.randn_like(residual, requires_grad=True)
    indexes = torch.randperm(24) if ep == 2 else None
    returned = outputs[indexes] if ep == 2 else outputs
    routed = outputs.new_zeros(3, 8)
    for slot in range(8):
        routed = routed + outputs.view(3, 8, 8)[:, slot] * weights[:, slot, None]
    expected = residual + (routed + shared)
    actual = namespace["forward_stage5"](
        layer, returned, indexes, weights, torch.cat((residual, shared), -1)
    )
    assert torch.equal(actual, expected)
    values = (outputs, weights, residual, shared)
    a = torch.autograd.grad(actual.float().square().sum(), values, retain_graph=True)
    b = torch.autograd.grad(expected.float().square().sum(), values)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
