"""
Check replayed expert choices, live router weights and gradients for each routing formula.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from pithtrain.models.glm4_moe_lite import Glm4MoeLiteTopkRouter
from pithtrain.models.gpt_oss import GptOssTopKRouter
from pithtrain.models.qwen3_moe import Qwen3MoeGate
from pithtrain.models.qwen35_moe import Qwen35MoeTopKRouter
from pithtrain.modules.load_balance import replay_indices

HIDDEN, EXPERTS, GROUPS = 256, 16, 4
GATES = ["qwen3", "qwen3.5", "gpt-oss", "glm-4.7"]


def build_gate(name: str) -> torch.nn.Module:
    torch.manual_seed(0)
    match name:
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
        case "glm-4.7":
            config = Glm4MoeLiteConfig(hidden_size=HIDDEN, n_routed_experts=EXPERTS, num_experts_per_tok=4, n_group=GROUPS, topk_group=2, norm_topk_prob=True, routed_scaling_factor=1.8)  # fmt: skip
            gate = Glm4MoeLiteTopkRouter(config)
            torch.nn.init.normal_(gate.e_score_correction_bias, std=0.1)
    torch.nn.init.xavier_uniform_(gate.weight)
    return gate.to("cuda", torch.bfloat16)


def live_weights(name: str, gate: torch.nn.Module, hidden: torch.Tensor, idx: torch.Tensor):
    """
    Independent reference for the live weights at replayed expert ids.
    """
    hidden = hidden.view(-1, HIDDEN)
    if name == "gpt-oss":
        logits = F.linear(hidden, gate.weight, gate.bias)
        return logits.gather(-1, idx).softmax(dim=-1, dtype=torch.float32)
    if name == "glm-4.7":
        weight = F.linear(hidden.float(), gate.weight.float()).sigmoid().gather(-1, idx)
        return weight / (weight.sum(dim=-1, keepdim=True) + 1e-20) * gate.routed_scaling_factor
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
    if name == "glm-4.7":
        # The chosen experts come from at most topk_group groups.
        group = live_idx // (EXPERTS // GROUPS)
        distinct = (group.sort(dim=-1).values.diff(dim=-1) != 0).sum(dim=-1) + 1
        assert (distinct <= gate.topk_group).all()
    idx, weight, _ = gate(hidden, live_idx.clone())
    assert torch.equal(idx, live_idx)
    assert torch.equal(weight, live_weight)
    assert torch.equal(router_grad(gate, weight), router_grad(gate, live_weight))


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32], ids=["int64", "int32"])
@pytest.mark.parametrize("name", GATES)
def test_replay_overrides_choice(name, dtype):
    gate = build_gate(name)
    top_k = gate.top_k if name == "glm-4.7" else gate.num_experts_per_tok
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
