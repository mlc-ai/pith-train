"""
segment_sum at the segment_sum cases, within the analytic fp32 bound.
"""

import pytest
import torch

from pithtrain.kernels import segment_sum
from tests.kernels.cases import params, seed_of
from tests.kernels.common import (
    BLOCK,
    F64,
    REALISTIC,
    ROUTING_CHUNK,
    bits,
    cumsum_list,
    grouped_rows,
    pad,
    record,
)


@pytest.fixture(autouse=True)
def empty_cache():
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()


U32 = 2.0**-24


def check_segment_sum(case: dict, what: str, x: torch.Tensor, ks: list[int]) -> None:
    offs = torch.tensor(cumsum_list(ks)[1:], device="cuda", dtype=torch.int32)
    G, n = len(ks), x.shape[1]
    got32 = segment_sum(x, offs, torch.float32)
    got16 = segment_sum(x, offs, torch.bfloat16)
    assert got32.shape == got16.shape == (G, n)
    assert got32.dtype == torch.float32 and got16.dtype == torch.bfloat16
    assert torch.equal(bits(got16), bits(got32.to(torch.bfloat16))), f"{what}: bf16 is not the fp32 sum rounded once"  # fmt: skip
    worst = 0.0
    for group, (size, start) in enumerate(zip(ks, cumsum_list(ks))):
        ref = torch.zeros(n, device="cuda", dtype=F64)
        absref = torch.zeros_like(ref)
        for a in range(start, start + size, ROUTING_CHUNK):
            part = x[a : min(a + ROUTING_CHUNK, start + size)].to(F64)
            ref += part.sum(0)
            absref += part.abs().sum(0)
        depth = -(-size // BLOCK) + 126
        gamma = depth * U32 / (1 - depth * U32)
        err = (got32[group].to(F64) - ref).abs()
        worst = max(worst, float((err / (gamma * absref + 1e-300)).max()))
    record(case["id"], f"segment_sum {what}", ratio=worst)
    assert worst <= 1, f"segment_sum {what}: error {worst:.3g}x the fp32 accumulation bound"


@pytest.mark.parametrize("case", params("segment_sum", keep=lambda c: REALISTIC(c) and c["axes"]["bias"]))  # fmt: skip
def test_segment_sum(case: dict) -> None:
    a = case["axes"]
    H, inter, counts = a["hidden"], a["inter"], a["group_sizes"]
    ks = [pad(c) for c in counts]
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    widths = [("down", H)] + ([("gate_up", 2 * inter)] if not a["bf16_fused"] else [])
    for what, width in widths:
        check_segment_sum(case, what, grouped_rows(counts, width, gen), ks)
