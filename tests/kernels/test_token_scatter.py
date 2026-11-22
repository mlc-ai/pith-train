"""
scatter_by_expert at the token_scatter cases, exact against PyTorch.
"""

import pytest
import torch

from pithtrain.kernels import scatter_by_expert
from tests.kernels.cases import params, seed_of
from tests.kernels.common import (
    BF16,
    BLOCK,
    REALISTIC,
    ROUTING_CHUNK,
    bits,
    check_layout,
    check_placement,
    stage3_input,
)


@pytest.fixture(autouse=True)
def empty_cache():
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()


@pytest.mark.parametrize("case", params("token_scatter", keep=lambda c: REALISTIC(c) and not c["axes"]["bf16_fused"]))  # fmt: skip
def test_token_scatter(case: dict) -> None:
    a = case["axes"]
    G, H, counts = a["G"], a["hidden"], a["group_sizes"]
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    inp = stage3_input(case, gen)
    expert_idxs = inp["expert_idxs"]
    m = expert_idxs.numel()
    x = torch.randn(m, H, generator=gen, **BF16)
    ids_before = expert_idxs.clone()
    out, reverse, offs, ks_tensor, ks = scatter_by_expert(x, expert_idxs, G, BLOCK, inp["group_counts"])  # fmt: skip
    check_layout(offs, ks_tensor, ks, counts)
    M = sum(ks)
    assert M == a["fp8_rows"] and out.shape == (M, H) and out.dtype == x.dtype and out.stride() == (H, 1)  # fmt: skip
    check_placement(reverse, expert_idxs, counts, ks, M)
    for start in range(0, m, ROUTING_CHUNK):
        end = min(start + ROUTING_CHUNK, m)
        assert torch.equal(bits(out[reverse[start:end]]), bits(x[start:end])), "rows not copied exactly"  # fmt: skip
    padding = torch.ones(M, dtype=torch.bool, device="cuda")
    padding[reverse] = False
    assert not bool(bits(out[padding]).any()), "padding rows must be zero bytes"
    rows_alloc = -(-M // 1024) * 1024
    assert out.storage_offset() == 0 and out.untyped_storage().nbytes() >= rows_alloc * H * 2
    tail = torch.as_strided(out, (rows_alloc - M, H), (H, 1), M * H)
    assert not bool(bits(tail).any()), "allocation tail rows must be zero bytes"
    assert torch.equal(expert_idxs, ids_before), "expert_idxs must not change"
