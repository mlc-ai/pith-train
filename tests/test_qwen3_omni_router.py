"""CPU checks for Omni's router math and main's eager replay callback contract.

Load only the production router class, since importing the complete native
model eagerly probes CUDA through its kernel imports. This does not exercise
native attention, grouped GEMM, FSDP or the pipeline.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, replay_indices


def router_class():
    source = Path(__file__).parents[1] / "pithtrain/models/qwen3_omni_moe.py"
    tree = ast.parse(source.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "Qwen3OmniMoeThinkerTextTopKRouter"
    )
    namespace = {
        "torch": torch,
        "F": F,
        "nn": nn,
        "MoELoadBalanceLossInjector": MoELoadBalanceLossInjector,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_replay_keeps_callback_eager_and_routes_differentiable(normalized, dtype):
    config = SimpleNamespace(
        num_experts_per_tok=2, num_experts=4, hidden_size=3, norm_topk_prob=normalized
    )
    router = router_class()(config).to(dtype)
    with torch.no_grad():
        router.weight.copy_(torch.arange(12, dtype=dtype).reshape(4, 3) / 16)
    calls = []

    def callback(num_tokens, top_k):
        assert not torch.compiler.is_compiling()
        calls.append((num_tokens, top_k))
        # Consecutive invocations replay different routes with identical shape.
        return (torch.arange(num_tokens * top_k).reshape(num_tokens, top_k) + len(calls)) % 4

    router.router_replay = callback
    compiled = torch.compile(router, backend="eager", fullgraph=True)
    hidden = torch.linspace(-0.8, 0.6, 18, dtype=dtype).reshape(2, 3, 3)
    for _ in range(2):
        replay = replay_indices(router, hidden, config.num_experts_per_tok)
        indices, weights, loss = compiled(hidden, replay)
        assert calls[-1] == (6, 2)
        assert loss is None
        torch.testing.assert_close(indices, replay, rtol=0, atol=0)

        reference_weight = router.weight.detach().clone().requires_grad_()
        logits = hidden.reshape(6, 3) @ reference_weight.T
        expected = logits.softmax(-1, dtype=torch.float32).gather(-1, replay)
        if normalized:
            expected = expected / expected.sum(-1, keepdim=True)
        expected = expected.to(dtype)
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)
        factors = torch.arange(12, dtype=dtype).reshape(6, 2) / 8
        (weights * factors).sum().backward()
        (expected * factors).sum().backward()
        torch.testing.assert_close(router.weight.grad, reference_weight.grad, rtol=0, atol=0)
        router.zero_grad(set_to_none=True)
    assert calls == [(6, 2), (6, 2)]


def test_no_replay_keeps_original_topk_probabilities():
    config = SimpleNamespace(
        num_experts_per_tok=2, num_experts=4, hidden_size=3, norm_topk_prob=True
    )
    router = router_class()(config)
    with torch.no_grad():
        router.weight.copy_(torch.arange(12).reshape(4, 3) / 16)
    hidden = torch.tensor([[[0.2, -0.4, 0.7]]])
    assert replay_indices(router, hidden, 2) is None
    indices, weights, loss = router(hidden)
    scores = (hidden.reshape(1, 3) @ router.weight.T).softmax(-1)
    expected_weight, expected_indices = scores.topk(2, dim=-1)
    expected_weight = expected_weight / expected_weight.sum(-1, keepdim=True)
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weight, rtol=0, atol=0)
    assert loss is None
