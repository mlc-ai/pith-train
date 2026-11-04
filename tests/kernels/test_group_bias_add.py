"""
Contract: y += bias[group_indices] in place, bit-identical to y + bias[group_indices] in PyTorch
(one fp32 add rounded once to y's dtype), signed zeros and infinities included. This holds for the
kernel's inputs and for every other input, which takes a PyTorch fallback.
"""

import pytest
import torch

from pithtrain.kernels import group_bias_add


def group_ids(counts, index_dtype=torch.int32):
    return torch.repeat_interleave(
        torch.arange(len(counts), device="cuda"), torch.tensor(counts, device="cuda")
    ).to(index_dtype)


def bits(x):
    return x.view({2: torch.int16, 4: torch.int32}[x.element_size()])


def check(y, bias, group_indices):
    want = (y + bias[group_indices]).to(y.dtype)
    ptr = y.data_ptr()
    assert group_bias_add(y, bias, group_indices) is None
    assert y.data_ptr() == ptr
    assert torch.equal(bits(y), bits(want))


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_misaligned_contiguous(dtype):
    """
    Contiguous views whose base is one element past an allocation, so not 16-byte aligned.
    """
    gen = torch.Generator(device="cuda").manual_seed(7)
    counts = [128, 256]
    y = torch.randn(384 * 2880 + 1, generator=gen, device="cuda").to(dtype)[1:].view(384, 2880)
    bias = torch.randn(2 * 2880 + 1, generator=gen, device="cuda").to(dtype)[1:].view(2, 2880)
    assert y.is_contiguous() and bias.is_contiguous()
    check(y, bias, group_ids(counts))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_subnormals_and_signed_zeros(dtype):
    """
    Subnormal inputs and sums are kept (not flushed to zero); -0 + -0 is -0, -0 + 0 is +0.
    """
    tiny = torch.finfo(dtype).smallest_normal
    counts = [128, 128]
    y = torch.full((256, 2880), -0.0, device="cuda", dtype=dtype)
    y[:, 1::4] = tiny / 4
    y[:, 2::4] = -tiny / 2
    y[:, 3::4] = tiny
    bias = torch.zeros(2, 2880, device="cuda", dtype=dtype)
    bias[0] = -0.0
    bias[1, 1::4] = tiny / 8
    bias[1, 2::4] = -tiny / 8
    bias[1, 3::4] = -tiny / 2
    check(y, bias, group_ids(counts))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_signed_infinities(dtype):
    """
    Positive and negative overflow of finite inputs, and explicit +/-inf in y and in bias, give the
    same signed infinities as PyTorch. Columns cycle through the cases; group 1 adds a finite bias.
    """
    gen = torch.Generator(device="cuda").manual_seed(10)
    big, inf = torch.finfo(dtype).max, float("inf")
    y_cases = [big, -big, inf, -inf, 1.0, -1.0, inf, -inf]
    bias_cases = [big, -big, 2.0, -2.0, inf, -inf, inf, -inf]
    pattern = torch.tensor(y_cases, device="cuda").repeat(2880 // 8)
    y = pattern.expand(256, 2880).to(dtype).contiguous()
    bias = torch.stack(
        [
            torch.tensor(bias_cases, device="cuda").repeat(2880 // 8),
            torch.randn(2880, generator=gen, device="cuda"),
        ]
    ).to(dtype)
    group_indices = group_ids([128, 128])
    check(y, bias, group_indices)
    assert torch.isinf(y[:128]).all() and torch.isinf(y[128:, 2::8]).all()
    assert (y[:128, 0::8] > 0).all() and (y[:128, 1::8] < 0).all()


def test_deterministic():
    gen = torch.Generator(device="cuda").manual_seed(8)
    counts = [512, 0, 640, 128]
    group_indices = group_ids(counts)
    y = torch.randn(sum(counts), 2880, generator=gen, device="cuda").to(torch.bfloat16)
    bias = torch.randn(4, 2880, generator=gen, device="cuda").to(torch.bfloat16)
    first, second = y.clone(), y.clone()
    group_bias_add(first, bias, group_indices)
    group_bias_add(second, bias, group_indices)
    assert torch.equal(bits(first), bits(second))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_contract_shape_single_launch(dtype, index_dtype):
    """
    The caller's shape (gpt-oss-20b, 4096 tokens, EP4: (16384, 2880), 8 experts) runs as one kernel
    launch, not the gather / add / copy of the PyTorch fallback, for every supported value and index
    dtype, with the output bit-identical to PyTorch.
    """
    gen = torch.Generator(device="cuda").manual_seed(9)
    counts = [1792, 2048, 2304, 1920, 2176, 2048, 1664, 2432]
    group_indices = group_ids(counts, index_dtype)
    y = torch.randn(sum(counts), 2880, generator=gen, device="cuda").to(dtype)
    bias = torch.randn(8, 2880, generator=gen, device="cuda").to(dtype)
    want = (y + bias[group_indices]).to(dtype)
    warmup, timed = y.clone(), y.clone()
    group_bias_add(warmup, bias, group_indices)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        group_bias_add(timed, bias, group_indices)
        torch.cuda.synchronize()
    kernels = [
        event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == 1, kernels
    assert torch.equal(bits(timed), bits(want))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_side_stream_ordering(index_dtype):
    """
    Called inside torch.cuda.stream(s), as DualPipeV does, the kernel runs on s in order: y is written
    on s behind a ~50 ms sleep, so a launch on any other stream would read y before that write lands.
    """
    gen = torch.Generator(device="cuda").manual_seed(10)
    counts = [1792, 2048, 2304, 1920, 2176, 2048, 1664, 2432]
    group_indices = group_ids(counts, index_dtype)
    src = torch.randn(sum(counts), 2880, generator=gen, device="cuda").to(torch.bfloat16)
    bias = torch.randn(8, 2880, generator=gen, device="cuda").to(torch.bfloat16)
    want = (src + bias[group_indices]).to(torch.bfloat16)
    y = torch.zeros_like(src)
    group_bias_add(y.clone(), bias, group_indices)  # compile outside the timed ordering
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(100_000_000)
        y.copy_(src)
        group_bias_add(y, bias, group_indices)
    side.synchronize()
    assert torch.equal(bits(y), bits(want))
