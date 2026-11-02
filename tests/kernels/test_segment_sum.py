"""
Verifier of segment_sum against plain PyTorch: run-to-run equality, equality with each
group summed alone (which takes the kernel's other internal path when the full call has many
groups), closeness to a float64 sum, and the documented summation order.
"""

import pytest
import torch

from pithtrain.kernels import fp8_group_indices, segment_sum


def offsets(counts):
    return torch.tensor(counts, device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)


def reference(x, offs):
    M, H = x.shape
    ends = offs.long().clamp(max=M).tolist()
    out, start = torch.zeros(len(ends), H, dtype=torch.float64, device=x.device), 0
    for group_idx, end in enumerate(ends):
        if end > start:
            out[group_idx] = x[start:end].double().sum(0)
        start = max(start, end)
    return out


def tolerance(x, offs):
    """
    Bound on the fp32 accumulation error of each group's sum: gamma_(T + 126) * sum|x| for a group
    of T tiles, the depth of any order inside a 128-row tile (at most 127 adds) followed by the
    T - 1 adds of the tile fold, so it holds on every architecture.
    """
    M = x.shape[0]
    ends = offs.long().clamp(max=M).tolist()
    starts = [0] + [max(0, end) for end in ends[:-1]]
    num_tiles = [-(-max(end - start, 0) // 128) for start, end in zip(starts, ends)]
    unit = torch.finfo(torch.float32).eps / 2
    depth = torch.tensor([count + 126 for count in num_tiles], dtype=torch.float64, device=x.device)
    return (depth * unit / (1 - depth * unit))[:, None] * reference(x.abs(), offs) + 1e-30


def check(counts, H, dtype, out_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    M = sum(counts)
    x = (torch.randn(M, H, generator=gen, device="cuda") * 3).to(dtype)
    offs = offsets(counts)
    got = segment_sum(x, offs, out_dtype)
    assert got.dtype == out_dtype and got.shape == (len(counts), H)
    assert torch.equal(got, segment_sum(x, offs, out_dtype))
    # Close to the exact sum (fp32 accumulation, then one rounding to out_dtype).
    got32 = segment_sum(x, offs, torch.float32)
    want = reference(x, offs)
    assert ((got32.double() - want).abs() <= tolerance(x, offs)).all()
    assert torch.equal(got, got32.to(out_dtype))
    # Same bits as each group summed alone (the other internal path when there are many groups).
    start = 0
    for group_idx, count in enumerate(counts):
        # A fresh copy, since a sliced base pointer can change Triton's specialization.
        alone = segment_sum(x[start : start + count].clone(), offsets([count]), torch.float32)  # fmt: skip
        assert torch.equal(alone[0], got32[group_idx]), f"group {group_idx} differs when summed alone"  # fmt: skip
        start += count
    return got32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_experts", [4, 8, 16, 32])
def test_balanced_128_aligned(dtype, num_experts):
    # EP8 / EP4 / EP2 / EP1 of gpt-oss-20b, grouped rows padded to 128.
    check([2048] * num_experts, 2880, dtype, dtype)


@pytest.mark.parametrize("H", [2880, 5760, 100, 2048])
def test_uneven_and_empty_segments(H):
    check([0, 1, 127, 128, 129, 0, 5000, 3, 0], H, torch.bfloat16, torch.bfloat16, seed=1)


@pytest.mark.parametrize("H", [2048, 2880])
def test_many_ragged_groups(H):
    # 64 / 128 local experts with ragged counts, many empty (deepseek / qwen sizes).
    gen = torch.Generator().manual_seed(7)
    for G in (64, 128):
        counts = torch.randint(0, 300, (G,), generator=gen)
        counts[torch.randperm(G, generator=gen)[: G // 4]] = 0
        check(counts.tolist(), H, torch.bfloat16, torch.float32, seed=G)


def test_long_segment():
    check([70000, 128], 2880, torch.bfloat16, torch.float32, seed=2)


def test_float16_input():
    check([300, 0, 129], 1000, torch.float16, torch.float16, seed=3)


def test_all_empty_and_trailing_padding():
    x = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    for counts in ([0, 0, 0], [100, 0]):
        offs = offsets(counts)
        got = check_tensor(x, counts)
        if counts == [0, 0, 0]:
            assert torch.all(got == 0)
        else:
            assert torch.all(got[1] == 0)  # rows 100..255 belong to no group
    # An offset past M is clamped to M: group 1 is rows 128..255.
    offs = torch.tensor([128, 512], device="cuda", dtype=torch.int32)
    got = segment_sum(x, offs, torch.float32)
    assert ((got.double() - reference(x, offs)).abs() <= tolerance(x, offs)).all()
    alone = segment_sum(x[128:].clone(), offsets([128]), torch.float32)
    assert torch.equal(got[1], alone[0])


def test_zero_rows():
    x = torch.empty(0, 2880, device="cuda", dtype=torch.bfloat16)
    got = segment_sum(x, offsets([0, 0, 0, 0]), torch.bfloat16)
    assert got.shape == (4, 2880) and torch.all(got == 0)


def test_large_and_cancelling_values():
    gen = torch.Generator(device="cuda").manual_seed(8)
    x = (torch.randn(1000, 300, generator=gen, device="cuda") * 1e4).to(torch.bfloat16)
    x[500:] = -x[:500]  # group 0 sums to zero: only the fp32 rounding bound is left
    got = check_tensor(x, [1000, 0])
    assert torch.all(got[1] == 0)


def check_tensor(x, counts):
    offs = offsets(counts)
    got = segment_sum(x, offs, torch.float32)
    assert ((got.double() - reference(x, offs)).abs() <= tolerance(x, offs)).all()
    return got


def test_order_is_tile_order():
    """
    The kernel sums in the documented order: a group's tile sums added in row order reproduce its
    bits, and summing its halves separately does not.
    """
    counts = [8192] * 4
    gen = torch.Generator(device="cuda").manual_seed(9)
    x = (torch.randn(sum(counts), 2880, generator=gen, device="cuda") * 3).to(torch.bfloat16)
    got = segment_sum(x, offsets(counts), torch.float32)
    tiles = segment_sum(x[:8192].clone(), offsets([128] * 64), torch.float32)
    acc = torch.zeros(2880, device="cuda")
    for tile_sum in tiles:
        acc = acc + tile_sum
    assert torch.equal(acc, got[0])
    halves = torch.stack(
        [
            segment_sum(x[i * 8192 : i * 8192 + 4096], offsets([4096]), torch.float32)[0]
            + segment_sum(x[i * 8192 + 4096 : (i + 1) * 8192], offsets([4096]), torch.float32)[0]
            for i in range(4)
        ]
    )
    assert not torch.equal(halves, got)


def documented_order(x, counts, out_dtype, block_rows=16):
    """
    Plain-PyTorch segment_sum in the SM90 order: per tile, block sums in row order from zero, a
    pairwise tree over the blocks, then the tiles of each group added in row order from zero.
    """
    H, out, start = x.shape[1], [], 0
    for count in counts:
        num_tiles = -(-count // 128)
        rows = torch.zeros(num_tiles * 128, H, dtype=torch.float32, device=x.device)
        rows[:count] = x[start : start + count].float()
        blocks = rows.view(num_tiles, 128 // block_rows, block_rows, H)
        acc = torch.zeros(num_tiles, 128 // block_rows, H, dtype=torch.float32, device=x.device)
        for row in range(block_rows):
            acc = acc + blocks[:, :, row]
        sums = list(acc.unbind(1))
        while len(sums) > 1:
            sums = [sums[2 * i] + sums[2 * i + 1] for i in range(len(sums) // 2)]
        group_sum = torch.zeros(H, dtype=torch.float32, device=x.device)
        for tile_sum in sums[0] if sums else torch.zeros(0, H, device=x.device):
            group_sum = group_sum + tile_sum
        out.append(group_sum)
        start += count
    return (
        torch.stack(out).to(out_dtype)
        if out
        else torch.zeros(0, H, dtype=out_dtype, device=x.device)
    )


def has_block_order():
    """
    Whether this device sums inside a tile in the 16-row-block order (SM90's; SM100's is Triton's
    tl.sum, which plain PyTorch does not reproduce), read from an output as the layout checks do.
    """
    gen = torch.Generator(device="cuda").manual_seed(18)
    x = (torch.randn(128, 512, generator=gen, device="cuda") * 3).to(torch.bfloat16)
    return torch.equal(
        segment_sum(x, offsets([128]), torch.float32), documented_order(x, [128], torch.float32)
    )


BLOCK_ORDER = has_block_order()
# What the device must do, independent of the kernel under test: the library's documented
# SM90-only group-id hint (None on SM100) marks the device whose in-tile order is the block order.
EXPECT_BLOCK_ORDER = (
    fp8_group_indices(torch.tensor([128], dtype=torch.int32, device="cuda"), 128) is not None
)


@pytest.mark.skipif(not EXPECT_BLOCK_ORDER, reason="the in-tile order is Triton's tl.sum here")
@pytest.mark.parametrize(
    "counts",
    [
        [2048] * 8,
        [7000, 128, 300, 0, 1],
        [0, 1, 127, 128, 129, 0, 5000, 3, 0],
        [4095],
        [33408 - 4 * 128, 128, 128, 128, 128],
    ],
)
@pytest.mark.parametrize(
    "dtype, out_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float32, torch.float32),
    ],
)
def test_exact_documented_order(counts, dtype, out_dtype):
    gen = torch.Generator(device="cuda").manual_seed(10)
    x = (torch.randn(sum(counts), 1000, generator=gen, device="cuda") * 3).to(dtype)
    got = segment_sum(x, offsets(counts), out_dtype)
    assert torch.equal(got, documented_order(x, counts, out_dtype))


@pytest.mark.skipif(not EXPECT_BLOCK_ORDER, reason="the in-tile order is Triton's tl.sum here")
def test_exact_order_tells_orders_apart():
    """
    A different order inside the tile (128 rows one at a time) gives other bits on random data, so
    the exact-order test pins the order.
    """
    gen = torch.Generator(device="cuda").manual_seed(11)
    counts = [4096, 1000]
    x = (torch.randn(sum(counts), 2880, generator=gen, device="cuda") * 3).to(torch.bfloat16)
    got = segment_sum(x, offsets(counts), torch.float32)
    assert torch.equal(got, documented_order(x, counts, torch.float32))
    assert not torch.equal(got, documented_order(x, counts, torch.float32, block_rows=128))


def test_no_groups_or_no_columns():
    x = torch.randn(300, 64, device="cuda", dtype=torch.bfloat16)
    assert segment_sum(x, offsets([]), torch.bfloat16).shape == (0, 64)
    got = segment_sum(
        torch.empty(300, 0, device="cuda", dtype=torch.bfloat16), offsets([100, 200]), torch.float32
    )
    assert got.shape == (2, 0) and got.dtype == torch.float32


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_zero_rows_any_out_dtype(out_dtype):
    got = segment_sum(
        torch.empty(0, 100, device="cuda", dtype=torch.bfloat16), offsets([0, 0]), out_dtype
    )
    assert got.dtype == out_dtype and got.shape == (2, 100) and torch.all(got == 0)


def test_offsets_dtype_and_stride():
    """
    int64 and non-contiguous offsets give the bits of contiguous int32 ones.
    """
    counts = [300, 0, 129, 1000]
    x = torch.randn(sum(counts), 2880, device="cuda", dtype=torch.bfloat16)
    want = segment_sum(x, offsets(counts), torch.float32)
    offs64 = torch.tensor(counts, device="cuda").cumsum(0)
    assert torch.equal(segment_sum(x, offs64, torch.float32), want)
    strided = torch.stack([offs64, offs64], 1)[:, 0]
    assert not strided.is_contiguous()
    assert torch.equal(segment_sum(x, strided, torch.float32), want)
    assert torch.equal(segment_sum(x, strided.to(torch.int32), torch.float32), want)


@pytest.mark.parametrize(
    "dtype, H",
    [
        (torch.float32, 7),
        (torch.float32, 1001),
        (torch.bfloat16, 101),
        (torch.bfloat16, 100),
        (torch.float16, 3),
    ],
)
def test_narrow_and_odd_rows(dtype, H):
    check([300, 0, 129, 70], H, dtype, dtype, seed=12)


def test_misaligned_base_matches_aligned():
    """
    A base pointer that only allows narrow loads gives the bits of an aligned copy.
    """
    M, H = 1000, 2880
    storage = torch.randn(M * H + 1, device="cuda").to(torch.bfloat16)
    x = storage[1:].view(M, H)
    assert x.data_ptr() % 16 != 0
    counts = [500, 0, 500]
    got = segment_sum(x, offsets(counts), torch.float32)
    assert torch.equal(got, segment_sum(x.clone(), offsets(counts), torch.float32))
    assert (
        (got.double() - reference(x, offsets(counts))).abs() <= tolerance(x, offsets(counts))
    ).all()


def test_many_small_groups():
    # more groups than threads per CTA (several scan rounds), many tiles per CTA spanning groups
    gen = torch.Generator().manual_seed(13)
    counts = torch.randint(0, 40, (1000,), generator=gen)
    counts[torch.randperm(1000, generator=gen)[:300]] = 0
    check(counts.tolist(), 256, torch.bfloat16, torch.float32, seed=13)


def test_one_group_spans_everything():
    # one group of every row next to empty ones, alone vs with others
    check([0, 50000, 0], 2880, torch.bfloat16, torch.bfloat16, seed=14)


def test_concurrent_streams():
    counts = [3000, 0, 5000, 129]
    x = torch.randn(sum(counts), 2880, device="cuda", dtype=torch.bfloat16)
    offs = offsets(counts)
    want = segment_sum(x, offs, torch.bfloat16)
    streams = [torch.cuda.Stream() for _ in range(2)]
    torch.cuda.synchronize()
    outs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            outs.append([segment_sum(x, offs, torch.bfloat16) for _ in range(4)])
    torch.cuda.synchronize()
    assert all(torch.equal(out, want) for stream_outs in outs for out in stream_outs)


def test_exact_tile_fold():
    """
    Tiles whose sums are exact in any in-tile order (small integers times a per-tile power of two),
    but whose fold rounds, so the bits pin the inter-tile order on every device: tile sums added in
    row order from zero.
    """
    gen = torch.Generator(device="cuda").manual_seed(19)
    counts = [128 * 37 + 5, 0, 128 * 3]
    M = sum(counts)
    scale = (
        2.0
        ** torch.randint(
            -20, 20, (-(-counts[0] // 128) + 3, 1), generator=gen, device="cuda"
        ).float()
    )
    rows = torch.randint(-64, 64, (M, 640), generator=gen, device="cuda").float()
    tile_of_row = torch.cat(
        [
            torch.arange(counts[0], device="cuda") // 128,
            -(-counts[0] // 128) + torch.arange(counts[2], device="cuda") // 128,
        ]
    )
    x = (rows * scale[tile_of_row]).to(torch.bfloat16)
    got = segment_sum(x, offsets(counts), torch.float32)
    assert torch.equal(got, documented_order(x, counts, torch.float32))


def test_wide_offsets_clamp_without_wrapping():
    """
    int64 ends past the int32 range clamp to M (not wrap), contiguous or strided, and narrower
    integer dtypes give the bits of int32.
    """
    x = torch.randn(300, 64, device="cuda", dtype=torch.bfloat16)
    want = segment_sum(x, offsets([100, 200, 0]), torch.float32)
    wide = torch.tensor([100, 2**33 + 5, 2**40], device="cuda", dtype=torch.int64)
    assert torch.equal(segment_sum(x, wide, torch.float32), want)
    strided = torch.stack([wide, -wide], 1)[:, 0]
    assert torch.equal(segment_sum(x, strided, torch.float32), want)
    small = torch.tensor([100, 300, 300], device="cuda", dtype=torch.int16)
    assert torch.equal(segment_sum(x, small, torch.float32), want)
    for dtype in (torch.uint32, torch.uint64):
        wide_unsigned = torch.tensor([100, 2**31 + 5, 2**32 - 1], device="cuda", dtype=dtype)
        assert torch.equal(segment_sum(x, wide_unsigned, torch.float32), want)
    top = torch.tensor([100, 2**63 + 5, 2**64 - 1], dtype=torch.uint64).cuda()
    assert torch.equal(segment_sum(x, top, torch.float32), want)


def test_block_order_is_the_expected_one():
    """
    The in-tile order the kernel shows is the one the device must have: a wrong SM90 order fails
    here instead of making the exact-order tests skip.
    """
    assert BLOCK_ORDER == EXPECT_BLOCK_ORDER


def gamma_tolerance(x, counts):
    """
    Standard forward-error bound of the documented order, gamma_(T + 17) * sum|x| per group: 15
    serial adds in a 16-row block, a 3-level tree over 8 blocks, T - 1 adds folding T tiles.
    """
    unit = torch.finfo(torch.float32).eps / 2
    depth = torch.tensor(
        [-(-count // 128) + 17 for count in counts], dtype=torch.float64, device=x.device
    )
    return (depth * unit / (1 - depth * unit))[:, None] * reference(
        x.abs(), offsets(counts)
    ) + 1e-30


@pytest.mark.skipif(not EXPECT_BLOCK_ORDER, reason="the in-tile order is Triton's tl.sum here")
def test_adversarial_rounding_within_gamma_bound():
    """
    Each block is 1.0 then fifteen 2^-24: every add rounds back to 1.0, so the documented order
    gives exactly 8.0 where the exact sum is 8 + 120 * 2^-24, outside a 4 eps * sum|x| bound but
    inside the operation-depth bounds.
    """
    block = torch.full((16,), 2.0**-24)
    block[0] = 1.0
    x = block.repeat(8)[:, None].cuda()
    got = segment_sum(x, offsets([128]), torch.float32)
    assert got.item() == 8.0
    assert torch.equal(got, documented_order(x, [128], torch.float32))
    assert ((got.double() - reference(x, offsets([128]))).abs() <= gamma_tolerance(x, [128])).all()
    assert (
        (got.double() - reference(x, offsets([128]))).abs() <= tolerance(x, offsets([128]))
    ).all()


def test_long_segment_wide_rows():
    check([70000, 128], 5760, torch.bfloat16, torch.float32, seed=20)


def test_side_stream():
    """
    segment_sum runs on the caller's current stream: inputs written on a side stream behind a sleep
    give the reference result when the call is made on that stream.
    """
    counts = [3000, 0, 5000, 129]
    src = (torch.randn(sum(counts), 2880, device="cuda") * 3).to(torch.bfloat16)
    offs_src = offsets(counts)
    x, offs = torch.zeros_like(src), torch.zeros_like(offs_src)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
        x.copy_(src), offs.copy_(offs_src)
        got = segment_sum(x, offs, torch.float32)
    stream.synchronize()
    assert ((got.double() - reference(src, offs_src)).abs() <= tolerance(src, offs_src)).all()
