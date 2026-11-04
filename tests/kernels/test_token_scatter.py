"""
Row placement within a group is unspecified, so the test checks the row set: each token lands, bit
for bit, in its expert's first count rows, and every other row is zero.
"""

import pytest
import torch

from pithtrain.kernels import scatter_by_expert


def check(x, expert_idxs, G, alignment=128, group_counts=None):
    m, H = x.shape
    counts = torch.bincount(expert_idxs, minlength=G).cpu()
    ks_want = ((counts + alignment - 1) // alignment * alignment).tolist()
    offs_want = torch.tensor(ks_want, dtype=torch.int64).cumsum(0)

    out, reverse_idxs, offs, ks_tensor, ks = scatter_by_expert(x, expert_idxs, G, alignment, group_counts)  # fmt: skip
    assert ks == ks_want
    assert ks_tensor.dtype == torch.int32 and ks_tensor.tolist() == ks_want
    assert offs.dtype == torch.int32 and offs.tolist() == offs_want.tolist()
    assert reverse_idxs.dtype == torch.int64 and reverse_idxs.shape == (m,)
    assert out.shape == (sum(ks_want), H) and out.dtype == x.dtype
    assert out.stride() == (H, 1)

    starts = (offs_want - torch.tensor(ks_want)).to(x.device)
    pos = reverse_idxs - starts[expert_idxs]
    assert torch.all(pos >= 0) and torch.all(pos < counts.to(x.device)[expert_idxs])
    assert torch.unique(reverse_idxs).numel() == m
    assert torch.equal(out[reverse_idxs], x)
    pad = torch.ones(out.shape[0], dtype=torch.bool, device=x.device)
    pad[reverse_idxs] = False
    assert torch.all(out[pad] == 0)
    return out, reverse_idxs, offs, ks_tensor, ks


def routing(m, G, skew=0.0, seed=0, empty=()):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    probs = torch.rand(G, generator=gen, device="cuda") ** (1 + 8 * skew)
    for expert_idx in empty:
        probs[expert_idx] = 0
    return torch.multinomial(probs, m, replacement=True, generator=gen)


# (tokens * top-k, hidden, local experts): the four models at EP1 / EP8.
MODELS = [
    (4096 * 6, 2048, 64),  # deepseek-v2-lite, top-6 of 64
    (4096 * 8, 2048, 128),  # qwen3-30b-a3b, top-8 of 128
    (4096 * 8, 2048, 32),  # qwen3.5-35b-a3b at EP8 (256 experts)
    (4096 * 4, 2880, 32),  # gpt-oss-20b, top-4 of 32
    (4096 * 4, 2880, 4),  # gpt-oss-20b at EP8
]


@pytest.mark.parametrize("m,H,G", MODELS)
def test_models(m, H, G):
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    check(x, routing(m, G, skew=0.5), G)


@pytest.mark.parametrize("alignment", [1, 16, 128, 256])
def test_ragged_and_empty_groups(alignment):
    G = 16
    expert_idxs = routing(3001, G, skew=1.0, seed=1, empty=(0, 5, 15))
    x = torch.randn(3001, 1000, device="cuda").to(torch.bfloat16)
    _, _, _, _, ks = check(x, expert_idxs, G, alignment)
    assert ks[0] == ks[5] == ks[15] == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("H", [1, 100, 256, 257, 2880])
def test_hidden_sizes_and_dtypes(dtype, H):
    expert_idxs = routing(777, 8, seed=2)
    x = (torch.randn(777, H, device="cuda") * 100).to(dtype)
    check(x, expert_idxs, 8)


def test_group_counts_on_host():
    """
    With group_counts the host sizes need no device copy; results are the same.
    """
    G = 32
    expert_idxs = routing(10000, G, skew=0.7, seed=3, empty=(3,))
    x = torch.randn(10000, 2048, device="cuda").to(torch.bfloat16)
    counts = torch.bincount(expert_idxs, minlength=G).tolist()
    without_counts = check(x, expert_idxs, G)
    with_counts = check(x, expert_idxs, G, group_counts=counts)
    assert without_counts[4] == with_counts[4] and torch.equal(without_counts[2], with_counts[2]) and torch.equal(without_counts[3], with_counts[3])  # fmt: skip


def test_single_token_single_group_and_one_group_all():
    x = torch.randn(1, 2048, device="cuda").to(torch.bfloat16)
    check(x, torch.tensor([3], device="cuda"), 8)
    x = torch.randn(4096, 2048, device="cuda").to(torch.bfloat16)
    check(x, torch.full((4096,), 7, device="cuda"), 8)  # exact multiple of 128, no padding
    check(x[:4000].contiguous(), torch.full((4000,), 0, device="cuda"), 1)


def test_special_values_copied_bitwise():
    x = torch.randn(512, 300, device="cuda").to(torch.bfloat16)
    x[0, :3] = torch.tensor([float("inf"), float("-inf"), -0.0])
    x[1] = 0
    out, reverse_idxs, *_ = scatter_by_expert(x, routing(512, 4, seed=4), 4)
    assert torch.equal(out[reverse_idxs].view(torch.int16), x.view(torch.int16))


def test_empty_input():
    x = torch.empty(0, 2048, device="cuda", dtype=torch.bfloat16)
    out, reverse_idxs, offs, ks_tensor, ks = scatter_by_expert(
        x, torch.empty(0, dtype=torch.long, device="cuda"), 8
    )
    assert out.shape == (0, 2048) and reverse_idxs.shape == (0,) and ks == []
    # With no tokens the offsets and sizes come back empty, not (G,) zeros.
    assert offs.numel() == 0 and ks_tensor.numel() == 0


def test_length_mismatch_raises():
    x = torch.randn(100, 64, device="cuda").to(torch.bfloat16)
    with pytest.raises(ValueError):
        scatter_by_expert(x, torch.zeros(50, dtype=torch.long, device="cuda"), 4)
