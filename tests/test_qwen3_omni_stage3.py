"""CPU contract for Omni stage 3 and the scatter wrapper's new host-count argument.

The production layer and autograd wrappers run unchanged. Only the CUDA scatter
kernel is replaced with an independent padded CPU scatter; this is not GPU acceptance.
"""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tests.cpu_helpers import load_definitions


@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("known_counts", [False, True])
def test_stage3_keeps_host_counts_policy_and_gradients(ep, deterministic, known_counts):
    ids = torch.tensor([3, 1, 1, 3, 0])
    counts = torch.bincount(ids, minlength=4).tolist() if known_counts else None
    seen = []

    def scatter(tokens, expert_idxs, num_groups, alignment, group_counts, *, deterministic):
        seen.append((group_counts, deterministic))
        sizes = [
            ((expert_idxs == group).sum().item() + alignment - 1) // alignment * alignment
            for group in range(num_groups)
        ]
        output = tokens.new_zeros(sum(sizes), tokens.shape[1])
        reverse = torch.empty_like(expert_idxs)
        start = 0
        for group, size in enumerate(sizes):
            rows = torch.where(expert_idxs == group)[0]
            output[start : start + rows.numel()] = tokens[rows]
            reverse[rows] = torch.arange(start, start + rows.numel())
            start += size
        sizes_tensor = torch.tensor(sizes, dtype=torch.int32)
        return output, reverse, sizes_tensor.cumsum(0).int(), sizes_tensor, sizes

    namespace = dict(torch=torch, Optional=Optional, scatter_by_expert=scatter)
    load_definitions(
        "pithtrain/operators/token_scatter.py",
        (
            "ScatterForGroupedGemm",
            "scatter_for_grouped_gemm",
            "PaddedIndexGather",
            "padded_index_gather",
        ),
        namespace,
    )
    namespace.update(
        distributed=SimpleNamespace(ep_size=ep),
        training=SimpleNamespace(deterministic_token_layout=deterministic),
    )
    load_definitions(
        "pithtrain/models/qwen3_omni_moe.py",
        ("forward_stage3",),
        namespace,
        class_name="Qwen3OmniMoeThinkerTextDecoderLayer",
    )
    layer = SimpleNamespace(
        is_moe=True,
        mlp=SimpleNamespace(experts_per_rank=4, experts=lambda values, *_: values.square()),
    )
    expand = torch.tensor([2, 0, 2, 1, 0]) if ep > 1 else None
    inputs = torch.arange((3 if ep > 1 else 5) * 8, dtype=torch.float32).reshape(-1, 8)
    actual, expected = inputs.clone().requires_grad_(), inputs.clone().requires_grad_()
    result = namespace["forward_stage3"](layer, actual, ids, expand, counts)
    reference = (expected[expand] if expand is not None else expected).square()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    result.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert seen == [(counts, deterministic)]

    # The pipeline also supplies the metadata argument to dense layers.
    dense = SimpleNamespace(is_moe=False, mlp=lambda values: values.square())
    torch.testing.assert_close(
        namespace["forward_stage3"](dense, inputs, None, None, None), inputs.square()
    )
