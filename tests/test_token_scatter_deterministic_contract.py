"""Exercise the actual stable Torch fallback on CPU; this is not a CUDA kernel test."""

from typing import Optional

import pytest
import torch

from tests.cpu_helpers import load_definitions


@pytest.mark.parametrize("rows", [0, 1, 5])
@pytest.mark.parametrize("host_counts", [False, True])
@pytest.mark.parametrize("alignment", [3, 128])
def test_stable_fallback_preserves_order_bytes_padding_and_host_counts(
    rows, host_counts, alignment
):
    namespace = load_definitions(
        "pithtrain/kernels/token_scatter_sm100.py",
        ("scatter_by_expert_torch", "scatter_by_expert_sm100"),
        dict(torch=torch, Optional=Optional, MAX_GROUPS=1024, GEMM_ALLOC_ALIGNMENT=1024),
    )
    scatter = namespace["scatter_by_expert_sm100"]
    ids = torch.tensor([3, 0, 3, 1, 0])[:rows]
    wide = torch.arange(5 * 16, dtype=torch.float32).reshape(5, 16)
    wide[0, 1:5] = torch.tensor([float("nan"), float("inf"), -0.0, 0.0])
    values = wide[:rows, 1:9]  # non-contiguous rows and misaligned base
    before = values.clone()
    counts = [(ids == group).sum().item() for group in range(4)]
    out, reverse, offsets, sizes, host_sizes = scatter(
        values, ids, 4, alignment, counts if host_counts else None, deterministic=True
    )
    if rows == 0:
        assert out.shape == (0, 8)
        assert reverse.numel() == offsets.numel() == sizes.numel() == 0
        assert host_sizes == []
        return

    expected_sizes = [(count + alignment - 1) // alignment * alignment for count in counts]
    assert host_sizes == sizes.tolist() == expected_sizes
    assert offsets.tolist() == torch.tensor(expected_sizes).cumsum(0).tolist()
    expected_reverse = torch.empty_like(ids)
    start = 0
    for group, size in enumerate(expected_sizes):
        group_rows = [row for row, expert in enumerate(ids.tolist()) if expert == group]
        expected_reverse[group_rows] = torch.arange(start, start + len(group_rows))
        start += size
    assert torch.equal(reverse, expected_reverse)
    assert torch.equal(out[reverse].contiguous().view(torch.uint8), before.view(torch.uint8))
    assert torch.equal(values.contiguous().view(torch.uint8), before.view(torch.uint8))

    allocated_rows = (sum(host_sizes) + 1023) // 1024 * 1024
    allocation = torch.as_strided(out, (allocated_rows, 8), (8, 1))
    padding = torch.ones(allocated_rows, dtype=torch.bool)
    padding[reverse] = False
    assert allocation[padding].view(torch.uint8).count_nonzero() == 0
