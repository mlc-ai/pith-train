"""
Verifier of scatter_by_expert against plain PyTorch. Every token must land, byte for byte, in its
expert's first count rows (reverse_shuffle_idxs a bijection onto the token rows), every padding row
and every row of the allocation tail up to a multiple of 1024 rows must be zero bytes, and the offsets
and sizes must be exact. The order of a group's tokens is not part of the architecture-neutral
contract (token order on SM90, atomic order on SM100), so it is not checked here; consumers locate a
token only through reverse_shuffle_idxs. With deterministic=True, token order is checked too.
"""

import pytest
import torch

from pithtrain.kernels import scatter_by_expert


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def routing(m, G, skew=0.0, seed=0, empty=()):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    probs = torch.rand(G, generator=gen, device="cuda") ** (1 + 8 * skew)
    for expert_idx in empty:
        probs[expert_idx] = 0
    return torch.multinomial(probs, m, replacement=True, generator=gen)


def check(x, expert_idxs, G, alignment=128, group_counts=None, deterministic=False):
    m, H = x.shape
    counts = torch.bincount(expert_idxs, minlength=G).cpu()
    ks_want = ((counts + alignment - 1) // alignment * alignment).tolist()
    offs_want = torch.tensor(ks_want, dtype=torch.int64).cumsum(0)
    x_before, expert_idxs_before = x.clone(), expert_idxs.clone()

    out, reverse_idxs, offs, ks_tensor, ks = scatter_by_expert(x, expert_idxs, G, alignment, group_counts, deterministic=deterministic)  # fmt: skip
    assert ks == ks_want
    assert ks_tensor.dtype == torch.int32 and ks_tensor.shape == (G,) and ks_tensor.tolist() == ks_want  # fmt: skip
    assert offs.dtype == torch.int32 and offs.shape == (G,) and offs.tolist() == offs_want.tolist()
    assert reverse_idxs.dtype == torch.int64 and reverse_idxs.shape == (m,)
    assert out.shape == (sum(ks_want), H) and out.dtype == x.dtype
    assert out.stride() == (H, 1)

    # each token inside its group's first count rows, the rows a bijection onto the tokens
    starts = (offs_want - torch.tensor(ks_want)).to(x.device)
    pos = reverse_idxs - starts[expert_idxs]
    assert torch.all(pos >= 0) and torch.all(pos < counts.to(x.device)[expert_idxs])
    assert torch.unique(reverse_idxs).numel() == m
    assert torch.equal(as_bytes(out[reverse_idxs]), as_bytes(x))
    pad = torch.ones(out.shape[0], dtype=torch.bool, device=x.device)
    pad[reverse_idxs] = False
    assert not bool(as_bytes(out[pad]).any()), "padding rows must be zero bytes"

    # the allocation behind the prefix view: zero bytes up to a multiple of 1024 rows
    rows_alloc = (out.shape[0] + 1023) // 1024 * 1024
    assert out.storage_offset() == 0
    assert out.untyped_storage().nbytes() >= rows_alloc * H * x.element_size()
    tail = torch.as_strided(out, (rows_alloc - out.shape[0], H), (H, 1), out.shape[0] * H)
    assert not bool(as_bytes(tail).any()), "tail rows must be zero bytes"

    assert torch.equal(as_bytes(x), as_bytes(x_before)) and torch.equal(expert_idxs, expert_idxs_before), "inputs must not change"  # fmt: skip
    if deterministic:
        for group in range(G):
            rows = reverse_idxs[expert_idxs == group]
            assert torch.all(rows[1:] > rows[:-1])
    return out, reverse_idxs, offs, ks_tensor, ks


# (tokens per rank * top-k, hidden, local experts): the four models at EP 2 / 4 / 8.
MODEL_SHAPES = [
    (top_k, H, E // ep)
    for top_k, H, E in [
        (8, 2048, 128),
        (8, 2048, 256),
        (6, 2048, 64),
        (4, 2880, 32),
    ]  # qwen3, qwen3.5, deepseek-v2-lite, gpt-oss-20b
    for ep in (2, 4, 8)
]


@pytest.mark.parametrize("T", [4096, 8192])
@pytest.mark.parametrize("top_k,H,G", MODEL_SHAPES)
def test_models(T, top_k, H, G):
    m = T * top_k
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    check(x, routing(m, G, skew=0.3, seed=T + G), G)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("alignment", [1, 16, 128, 256])
def test_ragged_and_empty_groups(alignment, deterministic):
    G = 16
    expert_idxs = routing(3001, G, skew=1.0, seed=1, empty=(0, 5, 15))
    x = torch.randn(3001, 1000, device="cuda").to(torch.bfloat16)
    _, _, _, _, ks = check(x, expert_idxs, G, alignment, deterministic=deterministic)
    assert ks[0] == ks[5] == ks[15] == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("H", [1, 3, 100, 256, 257, 2880])
def test_hidden_sizes_and_dtypes(dtype, H):
    expert_idxs = routing(777, 8, seed=2)
    x = (torch.randn(777, H, device="cuda") * 100).to(dtype)
    check(x, expert_idxs, 8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("H", [64, 100, 2880])
def test_strided_rows(dtype, H):
    """
    Row strides other than H (wider rows, a stride that is not a multiple of 16 bytes), a base that
    is not 16-byte aligned, and stride-0 rows (one row expanded).
    """
    expert_idxs = routing(1500, 12, skew=0.5, seed=3)
    wide = (torch.randn(1500, H + 9, device="cuda") * 10).to(dtype)
    check(wide[:, :H], expert_idxs, 12)
    misaligned = wide[:, 1 : 1 + H]
    assert misaligned.data_ptr() % 16 != 0
    check(misaligned, expert_idxs, 12)
    odd_stride = (torch.randn(1500, H + 1, device="cuda") * 10).to(dtype)[:, :H]
    assert (odd_stride.stride(0) * odd_stride.element_size()) % 16 != 0
    check(odd_stride, expert_idxs, 12)
    one_row = (torch.randn(1, H, device="cuda") * 10).to(dtype).expand(1500, H)
    assert one_row.stride(0) == 0
    check(one_row, expert_idxs, 12)


@pytest.mark.parametrize(
    "G,m", [(1, 5000), (256, 20000), (1024, 4096), (2048, 5000), (4096, 300), (4096, 70000)]
)
def test_group_counts_range(G, m):
    """
    One group, many groups, and group counts beyond what one thread block covers in one pass,
    including a large G with few tokens (most groups empty).
    """
    x = torch.randn(m, 64, device="cuda").to(torch.bfloat16)
    check(x, routing(m, G, skew=0.5, seed=G + m), G)
    check(x, routing(m, G, skew=0.5, seed=G + m), G, alignment=16)


@pytest.mark.parametrize("deterministic", [False, True])
def test_group_counts_on_host(deterministic):
    """
    With group_counts the host sizes need no device copy; results are the same.
    """
    G = 32
    expert_idxs = routing(10000, G, skew=0.7, seed=3, empty=(3, 31))
    x = torch.randn(10000, 2048, device="cuda").to(torch.bfloat16)
    counts = torch.bincount(expert_idxs, minlength=G).tolist()
    without_counts = check(x, expert_idxs, G, deterministic=deterministic)
    with_counts = check(x, expert_idxs, G, group_counts=counts, deterministic=deterministic)
    assert without_counts[4] == with_counts[4] and torch.equal(without_counts[2], with_counts[2]) and torch.equal(without_counts[3], with_counts[3])  # fmt: skip


def test_single_token_single_group_and_one_group_all():
    x = torch.randn(1, 2048, device="cuda").to(torch.bfloat16)
    check(x, torch.tensor([3], device="cuda"), 8)
    x = torch.randn(4096, 2048, device="cuda").to(torch.bfloat16)
    check(x, torch.full((4096,), 7, device="cuda"), 8)  # exact multiple of 128, no padding
    check(x[:4000].contiguous(), torch.full((4000,), 0, device="cuda"), 1)


@pytest.mark.parametrize("dtype,bits", [(torch.bfloat16, torch.int16), (torch.float16, torch.int16), (torch.float32, torch.int32)])  # fmt: skip
@pytest.mark.parametrize("deterministic", [False, True])
def test_special_values_copied_bitwise(dtype, bits, deterministic):
    """
    Signed zeros, infinities and NaNs with distinct payloads keep their bytes.
    """
    x = (torch.randn(512, 300, device="cuda") * 10).to(dtype)
    raw = x.view(bits)
    specials = {torch.int16: [0x7FC1, -1, 0x7F81, -0x8000] if dtype == torch.bfloat16 else [0x7E01, -1, 0x7C01, -0x8000], torch.int32: [0x7FC00001, -1, 0x7F800001, -0x80000000]}[bits]  # fmt: skip
    raw[0, : len(specials)] = torch.tensor(specials, dtype=bits, device="cuda")
    x[1, :3] = torch.tensor([float("inf"), float("-inf"), -0.0], dtype=dtype, device="cuda")
    x[2] = 0
    check(x, routing(512, 4, seed=4), 4, deterministic=deterministic)


@pytest.mark.parametrize("deterministic", [False, True])
def test_empty_input(deterministic):
    x = torch.empty(0, 2048, device="cuda", dtype=torch.bfloat16)
    out, reverse_idxs, offs, ks_tensor, ks = scatter_by_expert(
        x, torch.empty(0, dtype=torch.long, device="cuda"), 8, deterministic=deterministic
    )
    assert out.shape == (0, 2048) and reverse_idxs.shape == (0,) and ks == []
    # With no tokens the offsets and sizes come back empty, not (G,) zeros.
    assert offs.shape == (0,) and ks_tensor.shape == (0,)
    assert out.dtype == x.dtype and reverse_idxs.dtype == torch.int64
    assert offs.dtype == torch.int32 and ks_tensor.dtype == torch.int32


def test_length_mismatch_raises():
    """
    An expert_idxs length other than the input row count raises ValueError, before any launch, also
    with no input rows.
    """
    x = torch.randn(100, 64, device="cuda").to(torch.bfloat16)
    with pytest.raises(ValueError):
        scatter_by_expert(x, torch.zeros(50, dtype=torch.long, device="cuda"), 4)
    with pytest.raises(ValueError):
        scatter_by_expert(x[:0], torch.zeros(3, dtype=torch.long, device="cuda"), 4)


CALL_SHAPES = [(100, 256, 4), (70000, 128, 64), (5000, 64, 2048), (1, 32, 1), (20000, 2880, 16), (4096, 100, 300), (70000, 64, 8)]  # fmt: skip


def make_call(idx, m, H, G):
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    expert_idxs = routing(m, G, skew=0.5, seed=100 + idx)
    return x, expert_idxs, G


def check_result(x, expert_idxs, G, result):
    out, reverse_idxs, offs, ks_tensor, ks = result
    counts = torch.bincount(expert_idxs, minlength=G)
    ks_want = ((counts + 127) // 128 * 128).tolist()
    assert ks == ks_want and ks_tensor.tolist() == ks_want and offs.tolist() == torch.tensor(ks_want).cumsum(0).tolist()  # fmt: skip
    starts = (torch.tensor(ks_want).cumsum(0) - torch.tensor(ks_want)).to(x.device)
    pos = reverse_idxs - starts[expert_idxs]
    assert torch.all(pos >= 0) and torch.all(pos < counts[expert_idxs])
    assert torch.unique(reverse_idxs).numel() == x.shape[0]
    assert torch.equal(as_bytes(out[reverse_idxs]), as_bytes(x))
    pad = torch.ones(out.shape[0], dtype=torch.bool, device=x.device)
    pad[reverse_idxs] = False
    assert not bool(as_bytes(out[pad]).any())


def test_back_to_back_calls():
    """
    Many calls with no synchronization in between, varying m and G (the prep scratch grows),
    alternating with and without host counts.
    """
    calls = [make_call(i, *shape) for i, shape in enumerate(CALL_SHAPES * 2)]
    torch.cuda.synchronize()
    results = []
    for i, (x, expert_idxs, G) in enumerate(calls):
        counts = torch.bincount(expert_idxs, minlength=G).tolist() if i % 2 else None
        results.append(scatter_by_expert(x, expert_idxs, G, 128, counts))
    for (x, expert_idxs, G), result in zip(calls, results):
        check_result(x, expert_idxs, G, result)


def test_two_streams():
    """
    Calls issued on two streams that run concurrently (host counts, so the host never waits).
    """
    calls = [make_call(i, *shape) for i, shape in enumerate(CALL_SHAPES * 2)]
    counts = [torch.bincount(expert_idxs, minlength=G).tolist() for _, expert_idxs, G in calls]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    results = []
    for i, (x, expert_idxs, G) in enumerate(calls):
        with torch.cuda.stream(streams[i % 2]):
            results.append(scatter_by_expert(x, expert_idxs, G, 128, counts[i]))
    torch.cuda.synchronize()
    for (x, expert_idxs, G), result in zip(calls, results):
        check_result(x, expert_idxs, G, result)


@pytest.mark.parametrize("H", [1, 7, 64])
def test_bytes(H):
    """
    One-byte elements: odd hidden sizes, an odd row stride and a misaligned base take the narrowest
    vectors.
    """
    expert_idxs = routing(2000, 6, skew=0.5, seed=8)
    wide = torch.randint(0, 256, (2000, H + 3), dtype=torch.uint8, device="cuda")
    check(wide[:, :H].contiguous(), expert_idxs, 6)
    check(wide[:, 1 : 1 + H], expert_idxs, 6)


def test_irregular_group_count_low_experts():
    """
    A group count that is not a multiple of 4 and beyond what shared memory holds, over several
    chunks, with every token routed to a few low experts (long runs of empty groups after them).
    """
    G, m = 2051, 9000
    expert_idxs = routing(m, 5, skew=0.5, seed=9)
    x = torch.randn(m, 96, device="cuda").to(torch.bfloat16)
    _, _, offs, _, ks = check(x, expert_idxs, G)
    assert offs[-1].item() == sum(ks[:5])


def test_scratch_growth_on_fresh_stream():
    """
    On a fresh stream, calls with no synchronization in between grow the per-stream scratch in the
    number of chunks (past 64 chunks of 2048 tokens) and in the group count, while earlier calls
    may still be running.
    """
    stream = torch.cuda.Stream()
    shapes = [(1000, 32, 4), (140000, 16, 8), (150000, 16, 2051), (3000, 32, 64), (140000, 16, 8)]
    calls = [make_call(50 + i, m, H, G) for i, (m, H, G) in enumerate(shapes)]
    counts = [torch.bincount(expert_idxs, minlength=G).tolist() for _, expert_idxs, G in calls]
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        results = [
            scatter_by_expert(x, expert_idxs, G, 128, counts[i])
            for i, (x, expert_idxs, G) in enumerate(calls)
        ]
    torch.cuda.synchronize()
    for (x, expert_idxs, G), result in zip(calls, results):
        check_result(x, expert_idxs, G, result)


def test_device_sizes_from_other_streams():
    """
    Without group_counts the sizes come back through the device-to-host copy from whichever
    stream issued the call.
    """
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    calls = [make_call(70 + i, *shape) for i, shape in enumerate(CALL_SHAPES)]
    torch.cuda.synchronize()
    results = []
    for i, (x, expert_idxs, G) in enumerate(calls):
        with torch.cuda.stream(streams[i % 2]):
            results.append(scatter_by_expert(x, expert_idxs, G))
    torch.cuda.synchronize()
    for (x, expert_idxs, G), result in zip(calls, results):
        check_result(x, expert_idxs, G, result)
