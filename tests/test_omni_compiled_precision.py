"""Production media eager and image compiled stages preserve BF16 arithmetic.

Stage5 executes its complete original body. Stage1 fixes the attention output
to isolate residual, norm and routing arithmetic; it is not FA4 acceptance.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.test_omni_stage5_compile import layer_type

ROOT = Path(__file__).parents[1]


def value_and_grads(call, layer, sources, indices, cotangent):
    outputs, weights, residual = [
        None if value is None else value.detach().clone().requires_grad_() for value in sources
    ]
    result = call(layer, outputs, indices, weights, residual)
    values = (outputs, residual) if weights is None else (outputs, weights, residual)
    gradients = torch.autograd.grad((result.float() * cotangent.float()).sum(), values)
    return result, gradients


def reference_text_stage5(layer, outputs, _indices, weights, residual):
    """Independent dense/top-2 arithmetic with each BF16 product rounded first."""
    if weights is None:
        return residual + outputs
    contributions = outputs.reshape(*weights.shape, -1) * weights.unsqueeze(-1)
    combined = contributions[:, 0] + contributions[:, 1]
    return residual + combined.reshape_as(residual)


@pytest.mark.parametrize("kind", ["media", "image"])
@pytest.mark.parametrize("is_moe,ep", [(False, 1), (True, 1), (True, 2)])
def test_stage5_preserves_bf16_output_and_every_input_gradient(kind, is_moe, ep):
    torch.manual_seed(19)
    image = layer_type(ep, media=True)
    layer = (image if kind == "image" else image.__bases__[0])()
    layer.is_moe, layer.idx, layer.text_hidden_size, layer.deepstack_count = is_moe, 1, 128, 2
    residual = torch.randn(1, 69, 384 if kind == "image" else 128, dtype=torch.bfloat16) * 0.02
    weights = torch.randn(69, 2, dtype=torch.bfloat16) if is_moe else None
    outputs = torch.randn((138, 128) if is_moe else (1, 69, 128), dtype=torch.bfloat16) * 0.02
    cotangent = torch.randn_like(residual)
    sources = (outputs, weights, residual)
    indexes = torch.arange(138) if is_moe else None
    production = type(layer).forward_stage5
    reference = production.__wrapped__ if kind == "image" else reference_text_stage5
    calls = [production]
    if kind == "media":
        # Retain the previous CPU compiled-operator coverage as a separate
        # control; the production media entry itself executes eagerly.
        calls.append(
            torch.compile(production, fullgraph=True, options={"emulate_precision_casts": True})
        )
    expected, target_gradients = value_and_grads(reference, layer, sources, indexes, cotangent)
    for call in calls:
        actual, gradients = value_and_grads(call, layer, sources, indexes, cotangent)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for gradient, target in zip(gradients, target_gradients, strict=True):
            torch.testing.assert_close(gradient, target, rtol=0, atol=0)


def controlled_stage1_type():
    namespace = dict(torch=torch, nn=nn, F=F)
    path = ROOT / "pithtrain/models/qwen3_omni_moe.py"
    tree = ast.parse(path.read_text())
    for name in ("Qwen3OmniMoeThinkerTextRMSNorm", "Qwen3OmniMoeThinkerTextTopKRouter"):
        node = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    decoder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3OmniMoeThinkerMediaDecoderLayer"
    )
    method = next(
        node
        for node in decoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_stage1_compute"
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    class FixedAttention(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = nn.Parameter(value)

        def forward(self, _hidden, _rotary, _cu):
            return self.value

    class ControlledLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_moe = True
            self.input_layernorm = nn.Identity()
            self.self_attn = FixedAttention(torch.randn(1, 69, 128, dtype=torch.bfloat16) * 0.02)
            self.post_attention_layernorm = namespace["Qwen3OmniMoeThinkerTextRMSNorm"](
                128, 1e-6
            ).to(torch.bfloat16)
            self.mlp = nn.Module()
            self.mlp.gate = namespace["Qwen3OmniMoeThinkerTextTopKRouter"](
                SimpleNamespace(
                    hidden_size=128, num_experts_per_tok=2, num_experts=4, norm_topk_prob=True
                )
            ).to(torch.bfloat16)
            nn.init.normal_(self.mlp.gate.weight, std=0.02)

    ControlledLayer.forward_stage1_compute = namespace["forward_stage1_compute"]
    return ControlledLayer


def reference_stage1(layer, hidden):
    """HF eager residual, RMSNorm and top-k expressions, independently composed."""
    residual = hidden + layer.self_attn.value
    value = residual.to(torch.float32)
    variance = value.pow(2).mean(-1, keepdim=True)
    value = value * torch.rsqrt(variance + layer.post_attention_layernorm.variance_epsilon)
    hidden = layer.post_attention_layernorm.weight * value.to(residual.dtype)
    if not layer.is_moe:
        return hidden, residual, None, None, None
    router = layer.mlp.gate
    logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), router.weight)
    probabilities = F.softmax(logits, dtype=torch.float32, dim=-1)
    weights, indices = torch.topk(probabilities, router.top_k, dim=-1)
    if router.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return hidden, residual, indices, weights.to(logits.dtype), None


@pytest.mark.parametrize("is_moe", [False, True])
def test_stage1_preserves_bf16_residual_norm_boundaries(is_moe):
    torch.manual_seed(739)
    layer = controlled_stage1_type()()
    layer.is_moe = is_moe
    source = torch.randn(1, 69, 128, dtype=torch.bfloat16) * 0.02
    probes = (torch.randn_like(source), torch.randn_like(source))
    # Preserve the existing independent residual/norm objective. The additional
    # regression below also covers router outputs and every parameter gradient.
    parameters = (layer.self_attn.value, layer.post_attention_layernorm.weight)
    actual_input, expected_input = [source.detach().clone().requires_grad_() for _ in range(2)]
    actual = layer.forward_stage1_compute(actual_input, None)
    expected = reference_stage1(layer, expected_input)
    for value, target in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    objectives = [
        sum(
            (value.float() * probe.float()).sum()
            for value, probe in zip(result[:2], probes, strict=True)
        )
        for result in (actual, expected)
    ]
    grads = torch.autograd.grad(objectives[0], (actual_input, *parameters))
    targets = torch.autograd.grad(objectives[1], (expected_input, *parameters))
    for gradient, target in zip(grads, targets, strict=True):
        torch.testing.assert_close(gradient, target, rtol=0, atol=0)

    # Keep the old compiled residual/norm contract in addition to production.
    compiled = torch.compile(
        type(layer).forward_stage1_compute,
        fullgraph=True,
        options={"emulate_precision_casts": True},
    )
    compiled_input = source.detach().clone().requires_grad_()
    controlled = compiled(layer, compiled_input, None)
    for value, target in zip(controlled[:2], expected[:2], strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    objective = sum(
        (value.float() * probe.float()).sum()
        for value, probe in zip(controlled[:2], probes, strict=True)
    )
    compiled_grads = torch.autograd.grad(objective, (compiled_input, *parameters))
    for gradient, target in zip(compiled_grads, targets, strict=True):
        torch.testing.assert_close(gradient, target, rtol=0, atol=0)


def test_stage1_preserves_router_probabilities_and_full_vjp():
    # This retained seed/geometry had one router probability differ under the
    # previous compiled entry, including its emulate_precision_casts policy.
    torch.manual_seed(739)
    layer = controlled_stage1_type()()
    source = torch.randn(1, 69, 128, dtype=torch.bfloat16) * 0.02
    probes = (
        torch.randn_like(source),
        torch.randn_like(source),
        torch.randn(69, 2, dtype=torch.bfloat16),
    )
    actual_input, expected_input = [source.detach().clone().requires_grad_() for _ in range(2)]
    actual = layer.forward_stage1_compute(actual_input, None)
    expected = reference_stage1(layer, expected_input)
    for value, target in zip(actual[:4], expected[:4], strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    assert actual[4] is expected[4] is None
    objectives = [
        sum(
            (value.float() * probe.float()).sum()
            for value, probe in zip((result[0], result[1], result[3]), probes, strict=True)
        )
        for result in (actual, expected)
    ]
    parameters = tuple(layer.parameters())
    gradients = torch.autograd.grad(objectives[0], (actual_input, *parameters))
    targets = torch.autograd.grad(objectives[1], (expected_input, *parameters))
    for gradient, target in zip(gradients, targets, strict=True):
        torch.testing.assert_close(gradient, target, rtol=0, atol=0)
