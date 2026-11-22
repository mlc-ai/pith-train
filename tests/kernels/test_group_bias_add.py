"""
group_bias_add at the group_bias_add cases, bit-exact against PyTorch.
"""

import pytest
import torch

from pithtrain.kernels import group_bias_add
from tests.kernels.cases import params, seed_of
from tests.kernels.common import BF16, REALISTIC, ROUTING_CHUNK, bits, grouped_rows, pad


@pytest.fixture(autouse=True)
def empty_cache():
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()


@pytest.mark.parametrize("case", params("group_bias_add", keep=lambda c: REALISTIC(c) and c["axes"]["bias"]))  # fmt: skip
def test_group_bias_add(case: dict) -> None:
    a = case["axes"]
    G, H, counts = a["G"], a["hidden"], a["group_sizes"]
    ks = [pad(c) for c in counts]
    M = sum(ks)
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    groups = torch.repeat_interleave(torch.arange(G, device="cuda", dtype=torch.int32), torch.tensor(ks, device="cuda"), output_size=M)  # fmt: skip
    y = grouped_rows(counts, H, gen, scale=4.0)
    bias = torch.randn(G, H, generator=gen, **BF16)
    before = y.clone()
    assert group_bias_add(y, bias, groups) is None
    for start in range(0, M, ROUTING_CHUNK):
        end = min(start + ROUTING_CHUNK, M)
        want = (before[start:end].float() + bias[groups[start:end].long()].float()).to(torch.bfloat16)  # fmt: skip
        assert torch.equal(bits(y[start:end]), bits(want)), "group_bias_add is not y + bias[group]"
