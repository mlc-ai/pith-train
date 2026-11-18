"""Exercise actual inherited model methods without importing CUDA-only kernels."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.cpu_helpers import load_definitions


class Gate(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = torch.tensor(indices, dtype=torch.int64)
        self.weights = nn.Parameter(torch.ones(1, 8, dtype=torch.bfloat16))

    def forward(self, hidden, replay=None):
        count = hidden.numel() // hidden.shape[-1]
        return self.indices[None].expand(count, -1), self.weights.expand(count, -1), None


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = nn.Parameter(torch.tensor([1.0] + [0.00390625] * 7, dtype=torch.bfloat16))

    def forward(self, tokens, expert_ids, ks, sizes):
        return tokens * self.values[expert_ids, None]


class ZeroGate(nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


def classes():
    def scatter(tokens, indices, experts, *, deterministic):
        assert deterministic
        order = indices.argsort(stable=True)
        return tokens[order], order.argsort(), indices[order], None, None

    namespace = dict(
        torch=torch,
        nn=nn,
        replay_indices=lambda *_: None,
        MoELoadBalanceLossTracker=SimpleNamespace(add=lambda _: None),
        scatter_for_grouped_gemm=scatter,
        padded_index_gather=lambda value, indices: value[indices],
    )
    load_definitions(
        "pithtrain/models/qwen3_omni_moe.py",
        ("_combine_expert_order", "_combine_expert_sum", "Qwen3OmniMoeThinkerTextSparseMoeBlock"),
        namespace,
    )
    load_definitions(
        "pithtrain/models/qwen3_omni_talker_pipeline.py",
        ("Qwen3OmniMoeTalkerPipelineMoeBlock",),
        namespace,
    )
    return namespace


def block(name, indices):
    kind = classes()[name]
    model = kind.__new__(kind)
    nn.Module.__init__(model)
    model.num_experts_per_tok = model.experts_per_rank = 8
    model.gate, model.experts = Gate(indices), Experts()
    model.shared_expert_gate, model.shared_expert = ZeroGate(), nn.Identity()
    return model


def test_thinker_keeps_grouped_k_sum():
    model = block("Qwen3OmniMoeThinkerTextSparseMoeBlock", [7, 0, 6, 1, 5, 2, 4, 3])
    hidden = torch.ones(1, 1, 1, dtype=torch.bfloat16)
    assert model.reference_forward(hidden).item() == 1.03125


@pytest.mark.parametrize("indices", [list(range(8)), [7, 0, 6, 1, 5, 2, 4, 3]])
def test_talker_reference_keeps_expert_id_serial_rounding_and_shared_branch(indices):
    model = block("Qwen3OmniMoeTalkerPipelineMoeBlock", indices)
    hidden = torch.ones(1, 1, 1, dtype=torch.bfloat16, requires_grad=True)
    output = model.reference_forward(hidden)
    # Eager HF's expert0 product1.0 is followed by seven additions below half
    # its BF16 ULP. The separate shared branch is0.5, after routed accumulation.
    assert output.item() == 1.5
    output.sum().backward()
    assert torch.equal(model.experts.values.grad, torch.ones_like(model.experts.values))
    expected_gate_grad = model.experts.values.detach()[torch.tensor(indices)][None]
    assert torch.equal(model.gate.weights.grad, expected_gate_grad)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
