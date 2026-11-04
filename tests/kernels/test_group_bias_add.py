"""
Contract: y += bias[group_indices] in place, bit-identical to y + bias[group_indices] in PyTorch
(one fp32 add rounded once to y's dtype). This holds for the kernel's inputs (same dtype, M a
multiple of 128, contiguous) and for every other input, which takes a PyTorch fallback.
"""

import pytest
import torch

from pithtrain.kernels import group_bias_add


def group_ids(counts, index_dtype=torch.int32):
    return torch.repeat_interleave(
        torch.arange(len(counts), device="cuda"), torch.tensor(counts, device="cuda")
    ).to(index_dtype)


def check(y, bias, group_indices):
    want = (y + bias[group_indices]).to(y.dtype)
    ptr = y.data_ptr()
    assert group_bias_add(y, bias, group_indices) is None
    assert y.data_ptr() == ptr
    assert torch.equal(y, want)


# Hidden sizes: gpt-oss-20b 2880 (the caller, not a multiple of the 256-column block), the other
# models' 2048, a narrow 100 and a wide 4096.
@pytest.mark.parametrize("H", [2880, 2048, 100, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_ragged_groups(H, dtype, index_dtype):
    gen = torch.Generator(device="cuda").manual_seed(0)
    counts = [128, 0, 384, 128, 0, 0, 1024, 256]  # 128-aligned grouped rows, empty groups
    group_indices = group_ids(counts, index_dtype)
    y = (torch.randn(sum(counts), H, generator=gen, device="cuda") * 4).to(dtype)
    bias = torch.randn(len(counts), H, generator=gen, device="cuda").to(dtype)
    check(y, bias, group_indices)


def test_gpt_oss_experts():
    """
    32 local experts (EP1), random routing padded to a multiple of 128 rows per expert.
    """
    gen = torch.Generator(device="cuda").manual_seed(1)
    raw = torch.randint(0, 400, (32,), generator=gen, device="cuda")
    counts = ((raw + 127) // 128 * 128).tolist()
    group_indices = group_ids(counts)
    y = torch.randn(sum(counts), 2880, generator=gen, device="cuda").to(torch.bfloat16)
    bias = torch.randn(32, 2880, generator=gen, device="cuda").to(torch.bfloat16)
    check(y, bias, group_indices)


def test_large_negative_and_zero():
    gen = torch.Generator(device="cuda").manual_seed(2)
    counts = [256, 128]
    group_indices = group_ids(counts)
    y = (torch.randn(384, 2880, generator=gen, device="cuda") * 1e30).to(torch.bfloat16)
    y[:64] = 0
    bias = (torch.randn(2, 2880, generator=gen, device="cuda") * 1e30).to(torch.bfloat16)
    bias[1] = -bias[1]
    check(y, bias, group_indices)
    # Overflow to inf exactly as PyTorch.
    y = torch.full((128, 256), 3e38, device="cuda", dtype=torch.bfloat16)
    check(y, torch.full((1, 256), 3e38, device="cuda", dtype=torch.bfloat16), group_ids([128]))


def test_empty():
    y = torch.empty(0, 2880, device="cuda", dtype=torch.bfloat16)
    check(y, torch.randn(4, 2880, device="cuda").to(torch.bfloat16), group_ids([0, 0, 0, 0]))


def test_rows_not_multiple_of_128():
    gen = torch.Generator(device="cuda").manual_seed(3)
    counts = [5, 0, 130, 1]
    y = torch.randn(sum(counts), 2880, generator=gen, device="cuda").to(torch.bfloat16)
    bias = torch.randn(4, 2880, generator=gen, device="cuda").to(torch.bfloat16)
    check(y, bias, group_ids(counts))


def test_mixed_dtypes():
    """
    A bf16 y with an fp32 bias: y + bias[group_indices] promotes to fp32, rounded back into y once.
    """
    gen = torch.Generator(device="cuda").manual_seed(4)
    counts = [128, 256]
    y = torch.randn(384, 2880, generator=gen, device="cuda").to(torch.bfloat16)
    bias = torch.randn(2, 2880, generator=gen, device="cuda")
    check(y, bias, group_ids(counts))


def test_non_contiguous():
    gen = torch.Generator(device="cuda").manual_seed(5)
    counts = [128, 128]
    wide = torch.randn(256, 3000, generator=gen, device="cuda").to(torch.bfloat16)
    y = wide[:, :2880]
    before = wide.clone()
    bias = torch.randn(2, 2880, generator=gen, device="cuda").to(torch.bfloat16)
    check(y, bias, group_ids(counts))
    assert torch.equal(wide[:, 2880:], before[:, 2880:])
    y = torch.randn(256, 2880, generator=gen, device="cuda").to(torch.bfloat16)
    bias_t = torch.randn(2880, 2, generator=gen, device="cuda").to(torch.bfloat16).t()
    check(y, bias_t, group_ids(counts))
