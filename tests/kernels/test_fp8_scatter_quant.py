"""
scatter_quant and fp8_group_indices at the fp8_scatter_quant cases, byte-exact against PyTorch.
"""

import pytest
import torch

from pithtrain.kernels import (
    fp8_group_indices,
    scatter_quant,
)
from tests.kernels.cases import params, seed_of
from tests.kernels.common import (
    BF16,
    BLOCK,
    I64,
    REALISTIC,
    ROUTING_CHUNK,
    assert_fp8_equal,
    assert_scale_equal,
    check_layout,
    check_placement,
    fp8_quant_grad_reference,
    stage3_input,
)


@pytest.fixture(autouse=True)
def empty_cache():
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()


def routing_row_chunks(sizes: list[int], chunk: int = ROUTING_CHUNK):
    """
    Yield (group, group start, chunk start, chunk end) over grouped rows, chunks within a group.
    """
    start = 0
    for group, size in enumerate(sizes):
        for a in range(start, start + size, chunk):
            yield group, start, a, min(a + chunk, start + size)
        start += size


def received_tokens(rows: int, H: int, gen: torch.Generator) -> torch.Tensor:
    """
    bf16 tokens with row magnitudes from 2^-8 to 2^8 and some zero rows.
    """
    x = torch.randn(rows, H, generator=gen, device="cuda")
    x *= torch.exp2(torch.randint(-8, 9, (rows, 1), generator=gen, device="cuda").float())
    x[3::97] = 0
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("case", params("fp8_scatter_quant", keep=REALISTIC))
def test_scatter_quant(case: dict) -> None:
    a = case["axes"]
    G, H, counts = a["G"], a["hidden"], a["group_sizes"]
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    inp = stage3_input(case, gen)
    expert_idxs, expand_idx = inp["expert_idxs"], inp["expand_idx"]
    gathered = received_tokens(inp["num_src"], H, gen)
    quantized, reverse, offs, ks, ks_tensor = scatter_quant(gathered, expert_idxs, G, expand_idx, group_counts=inp["group_counts"])  # fmt: skip
    check_layout(offs, ks_tensor, ks, counts)
    M = sum(ks)
    assert M == a["fp8_rows"]
    check_placement(reverse, expert_idxs, counts, ks, M)
    x_fp8, x_scale, x_ch, x_ch_scale = quantized
    assert x_fp8.shape == (M, H) and x_scale.shape == (M, -(-H // BLOCK))
    assert x_ch.shape == (M * H,) and x_ch_scale.shape == (H, M // BLOCK)  # K-major per group
    src = torch.full((M,), -1, **I64)
    src[reverse] = expand_idx if expand_idx is not None else torch.arange(expert_idxs.numel(), **I64)  # fmt: skip
    for group, group_start, start, end in routing_row_chunks(ks):
        rows = src[start:end]
        dense = torch.zeros(end - start, H, **BF16)
        valid = rows >= 0
        dense[valid] = gathered[rows[valid]]
        q, s = fp8_quant_grad_reference(dense)
        assert_fp8_equal(x_fp8[start:end], q, "x_fp8")
        assert_scale_equal(x_scale[start:end], s, "x_scale")
        q, s = fp8_quant_grad_reference(dense.T)  # per channel: 128 rows x 1 column blocks
        size, first, last = ks[group], start - group_start, end - group_start
        data = x_ch[group_start * H : (group_start + size) * H].view(H, size)[:, first:last]
        scale = x_ch_scale[:, start // BLOCK : end // BLOCK]
        assert_fp8_equal(data, q, "x_ch_fp8")
        assert_scale_equal(scale, s, "x_ch_scale")
    hint = fp8_group_indices(offs, M)
    want = torch.repeat_interleave(torch.arange(G, device="cuda", dtype=torch.int32), ks_tensor, output_size=M)  # fmt: skip
    assert hint.dtype == torch.int32 and hint.shape == (M,) and torch.equal(hint, want)
