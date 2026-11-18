"""Original grouped-HF BF16 reduction, EP return order and Talker isolation."""

import pytest
import torch

from tests.cpu_helpers import load_definitions

NS = load_definitions(
    "pithtrain/models/qwen3_omni_moe.py",
    ("_combine_expert_order", "_combine_expert_sum"),
    dict(torch=torch),
)


def test_thinker_bf16_reduction_keeps_small_contributions_and_talker_unchanged():
    values = torch.tensor([1.0] + [0.00390625] * 7, dtype=torch.bfloat16).view(8, 1)
    weights = torch.ones(1, 8, dtype=torch.bfloat16)
    thinker = NS["_combine_expert_sum"](values, weights, None, torch.bfloat16)
    talker = NS["_combine_expert_order"](values, weights, None, torch.bfloat16)
    assert thinker.item() == 1.03125 and talker.item() == 1.0


@pytest.mark.parametrize("returned", [None, [9, 2, 5, 4, 10, 1], []])
def test_thinker_original_slots_sparse_EP_and_every_gradient(returned):
    torch.manual_seed(739)
    indexes = None if returned is None else torch.tensor(returned, dtype=torch.int64)
    count = 12 if indexes is None else len(indexes)
    values = torch.randn(count, 16, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    actual = NS["_combine_expert_sum"](values, weights, indexes, torch.bfloat16)
    slots = values if indexes is None else values.new_zeros(12, 16).index_copy(0, indexes, values)
    expected = (slots.reshape(3, 4, 16) * weights[..., None]).sum(1)
    assert torch.equal(actual, expected)
    a = torch.autograd.grad(actual.float().square().sum(), (values, weights), retain_graph=True)
    b = torch.autograd.grad(expected.float().square().sum(), (values, weights))
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
