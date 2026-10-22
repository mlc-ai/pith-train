"""Check production eager text stage5 and its explicit CPU fullgraph control.

Extracting these pure methods avoids unrelated CUDA-only kernel imports. The
production bodies are executed without replacement in both controls.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def layer_type(ep):
    source = Path(__file__).parents[1] / "pithtrain/models/qwen3_omni_moe.py"
    tree = ast.parse(source.read_text())
    name = "Qwen3OmniMoeThinkerTextDecoderLayer"
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = dict(torch=torch, distributed=SimpleNamespace(ep_size=ep))
    helper = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_combine_expert_order"
    )
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    cls = type(name, (torch.nn.Module,), {})
    for method in ("_forward_stage5", "forward_stage5"):
        node = next(
            n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == method
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        setattr(cls, method, namespace[method])
    return cls


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
    compiled = torch.compile(
        layer.forward_stage5, fullgraph=True, options={"emulate_precision_casts": True}
    )
    for call in (layer.forward_stage5, compiled):
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
def test_expert_order_matches_actual_hf_output_and_all_gradients(
    top_k, dtype, deterministic_token_layout, monkeypatch
):
    import transformers
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
        _experts_implementation="eager",
    )
    hf = Qwen3OmniMoeThinkerTextExperts(config).to(dtype)
    with torch.no_grad():
        hf.gate_up_proj.normal_(0, 0.2)
        hf.down_proj.normal_(0, 0.2)
    inputs = torch.randn(n, hidden, dtype=dtype, requires_grad=True)
    weights, ids = torch.randn(n, experts).softmax(-1).topk(top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(dtype).detach().requires_grad_()
    pointers = {hf.down_proj[e].data_ptr(): e for e in range(experts)}
    observed = []
    linear = torch.nn.functional.linear

    def observe(value, weight, bias=None):
        output = linear(value, weight, bias)
        expert = pointers.get(weight.data_ptr())
        if expert is not None:
            slot, token = torch.where((ids == expert).transpose(0, 1))
            observed.append((token * top_k + slot, output))
        return output  # The actual HF forward and every gradient remain unchanged.

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.functional, "linear", observe)
        expected = hf(inputs, ids, weights)
    indexes = torch.cat([item[0] for item in observed])
    assert torch.equal(indexes.sort().values, torch.arange(n * top_k))
    slots = (
        inputs.new_zeros(n * top_k, hidden)
        .index_copy(0, indexes, torch.cat([item[1] for item in observed]))
        .view(n, top_k, hidden)
    )
    ordered_ids, order = ids.sort(dim=-1)
    if top_k > 1:
        assert not torch.equal(ids, ordered_ids)
    ordered_weights = weights.gather(1, order)
    ordered_outputs = slots.gather(1, order[..., None].expand(-1, -1, hidden))
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
        outputs = ordered_outputs.flatten(0, 1)
        if returned_indices is not None:
            outputs = outputs[returned_indices]
        calls = [layer.forward_stage5]
        if top_k == 8 and dtype == torch.bfloat16:
            calls.append(
                torch.compile(
                    layer.forward_stage5, fullgraph=True, options={"emulate_precision_casts": True}
                )
            )
        for call_index, call in enumerate(calls):
            actual = call(outputs, returned_indices, ordered_weights, torch.zeros_like(expected))
            assert torch.equal(actual, expected)
            # The final compiled backward may donate buffers; release its graph normally.
            retain_graph = not (ep == 2 and call_index == len(calls) - 1)
            actual_grad = torch.autograd.grad(actual, targets, cotangent, retain_graph=retain_graph)
            assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad, strict=True))
    if top_k == 8 and dtype == torch.bfloat16:
        # Sorting alone and an unordered serial loop are not the HF BF16 contract.
        assert not torch.equal((ordered_outputs * ordered_weights[..., None]).sum(1), expected)


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
    namespace = dict(parent.forward_stage5.__globals__)
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
