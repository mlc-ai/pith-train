"""
Check replayed expert choices, live router weights and gradients for each routing formula.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from pithtrain.models.deepseek_v2 import DeepSeekV2MoEGate
from pithtrain.models.gpt_oss import GptOssTopKRouter
from pithtrain.models.qwen3_moe import Qwen3MoeGate
from pithtrain.models.qwen35_moe import Qwen35MoeTopKRouter
from pithtrain.modules.load_balance import force_balance, replay_indices

HIDDEN, EXPERTS, GROUPS = 256, 16, 4
GATES = ["deepseek-v2-lite", "deepseek-v2", "qwen3", "qwen3.5", "gpt-oss"]


def build_gate(name: str) -> torch.nn.Module:
    torch.manual_seed(0)
    match name:
        case "deepseek-v2-lite":
            config = DeepseekV2Config(hidden_size=HIDDEN, n_routed_experts=EXPERTS, num_experts_per_tok=6, topk_method="greedy")  # fmt: skip
            gate = DeepSeekV2MoEGate(config)
        case "deepseek-v2":
            config = DeepseekV2Config(hidden_size=HIDDEN, n_routed_experts=EXPERTS, num_experts_per_tok=2, topk_method="group_limited_greedy", n_group=GROUPS, topk_group=1, routed_scaling_factor=16.0)  # fmt: skip
            gate = DeepSeekV2MoEGate(config)
        case "qwen3":
            config = Qwen3MoeConfig(hidden_size=HIDDEN, num_experts=EXPERTS, num_experts_per_tok=4, norm_topk_prob=True)  # fmt: skip
            gate = Qwen3MoeGate(config)
        case "qwen3.5":
            config = Qwen3_5MoeTextConfig(hidden_size=HIDDEN, num_experts=EXPERTS, num_experts_per_tok=4)  # fmt: skip
            gate = Qwen35MoeTopKRouter(config)
        case "gpt-oss":
            config = GptOssConfig(hidden_size=HIDDEN, num_local_experts=EXPERTS, num_experts_per_tok=4)  # fmt: skip
            gate = GptOssTopKRouter(config)
            torch.nn.init.normal_(gate.bias, std=0.1)
    torch.nn.init.xavier_uniform_(gate.weight)
    return gate.to("cuda", torch.bfloat16)


def live_weights(name: str, gate: torch.nn.Module, hidden: torch.Tensor, idx: torch.Tensor):
    """
    Independent reference for the live weights at replayed expert ids.
    """
    hidden = hidden.view(-1, HIDDEN)
    if name.startswith("deepseek"):
        scores = F.linear(hidden.float(), gate.weight.float()).softmax(dim=-1)
        return scores.gather(-1, idx) * gate.routed_scaling_factor
    if name == "gpt-oss":
        logits = F.linear(hidden, gate.weight, gate.bias)
        return logits.gather(-1, idx).softmax(dim=-1, dtype=torch.float32)
    weight = F.linear(hidden, gate.weight).softmax(dim=-1, dtype=torch.float32).gather(-1, idx)
    return weight / weight.sum(dim=-1, keepdim=True)


def router_grad(gate: torch.nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """
    Use a fixed random readout; summing normalized weights would give zero gradient.
    """
    probe = torch.randn(weight.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(1))  # fmt: skip
    return torch.autograd.grad((weight * probe).sum(), gate.weight)[0]


@pytest.mark.parametrize("name", GATES)
def test_replay_of_live_choice_is_exact(name):
    gate = build_gate(name)
    hidden = torch.randn(512, HIDDEN, device="cuda", dtype=torch.bfloat16)
    live_idx, live_weight, _ = gate(hidden)
    if name == "deepseek-v2":
        group = live_idx // (EXPERTS // GROUPS)
        assert (group == group[:, :1]).all()
    idx, weight, _ = gate(hidden, live_idx.clone())
    assert torch.equal(idx, live_idx)
    assert torch.equal(weight, live_weight)
    assert torch.equal(router_grad(gate, weight), router_grad(gate, live_weight))


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32], ids=["int64", "int32"])
@pytest.mark.parametrize("name", GATES)
def test_replay_overrides_choice(name, dtype):
    gate = build_gate(name)
    top_k = gate.top_k if name.startswith("deepseek") else gate.num_experts_per_tok
    hidden = torch.randn(4, 128, HIDDEN, device="cuda", dtype=torch.bfloat16)
    routes = torch.rand(512, EXPERTS, device="cuda").topk(top_k).indices.to(dtype)

    def router_replay(num_tokens: int, k: int) -> torch.Tensor:
        assert (num_tokens, k) == (512, top_k)
        return routes

    gate.router_replay = router_replay
    idx, weight, _ = gate(hidden, replay_indices(gate, hidden, top_k))
    assert torch.equal(idx, routes)
    expected = live_weights(name, gate, hidden, routes)
    torch.testing.assert_close(weight, expected)
    torch.testing.assert_close(router_grad(gate, weight), router_grad(gate, expected))


@pytest.mark.parametrize("num_experts,top_k", [(128, 8), (64, 6), (32, 4), (128, 4), (256, 8)])
def test_force_balance(num_experts, top_k):
    """
    Check equal load and distinct ids per token for each shipped expert count and top-k.
    """
    num_tokens = 24 * num_experts
    idx = force_balance(num_experts)(num_tokens, top_k)
    assert idx.shape == (num_tokens, top_k)
    counts = torch.bincount(idx.flatten(), minlength=num_experts)
    assert (counts == num_tokens * top_k // num_experts).all()
    assert (idx.sort(dim=-1).values.diff(dim=-1) > 0).all()
