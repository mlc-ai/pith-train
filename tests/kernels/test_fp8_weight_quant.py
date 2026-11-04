"""
Verifier of the fp8_weight_quant feature against a PyTorch 128x128 blockwise FP8 quantization (one
E8M0 scale per 128x128 tile, from the tile amax).

- fp8_quant_weight and fp8_quant_weight_grouped: the blockwise FP8 of W and
  of W.T (per expert for a (G, N, K) weight), with row-major scales.
- The FP8 parameter all-gather kernels, driven as the FP8 parameter gather drives them. FSDP dim-0
  shards of an fp32 master (rows of a 2D weight, experts of a 3D one) each write the per-tile amax
  of the bf16-rounded shard into a zeroed grid (fp8_shard_amax). The grids are MAX-reduced, the
  scales follow (fp8_scales), every shard is cast with them (fp8_shard_cast) and the gathered FP8
  weight is packed with its transpose (fp8_pack_transpose). The result must equal the blockwise
  quantization of the full bf16 weight.
- fp8_shard_amax_multi and fp8_scales_multi return one launch for a list of weights (a callable on
  the H200) whose results must equal the per-tensor calls.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import (
    fp8_pack_transpose,
    fp8_quant_weight,
    fp8_quant_weight_grouped,
    fp8_scales,
    fp8_scales_multi,
    fp8_shard_amax,
    fp8_shard_amax_multi,
    fp8_shard_cast,
)

BLOCK = 128


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def e8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """
    Return 2 ** ceil(log2(max(amax, 1e-4) / 448)), computed exactly with frexp.
    """
    ratio = amax.float().clamp(min=1e-4) / 448.0
    mantissa, exponent = torch.frexp(ratio)  # ratio = mantissa * 2**exponent, mantissa in [0.5, 1)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    return ((exponent + 127) << 23).to(torch.int32).view(torch.float32)


def tiles(x: torch.Tensor) -> torch.Tensor:
    G, N, K = x.shape
    R, C = cdiv(N, BLOCK), cdiv(K, BLOCK)
    x_pad = F.pad(x.float(), (0, C * BLOCK - K, 0, R * BLOCK - N))
    return x_pad.view(G, R, BLOCK, C, BLOCK)


def ref_amax(x: torch.Tensor) -> torch.Tensor:
    """
    Return the (G, R, C) per-tile amax of a (G, N, K) tensor.
    """
    x_tiles = tiles(x)
    if x_tiles.numel() == 0:
        return torch.zeros(x_tiles.shape[0], x_tiles.shape[1], x_tiles.shape[3], device=x.device)
    return x_tiles.abs().amax(dim=(2, 4))


def ref_blockwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a (G, N, K) tensor in 128x128 blocks: (fp8 (G, N, K), scale (G, R, C)).
    """
    G, N, K = x.shape
    scale = e8m0_scale(ref_amax(x))
    fp8 = (tiles(x) / scale[:, :, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    fp8 = fp8.view(G, scale.shape[1] * BLOCK, scale.shape[2] * BLOCK)[:, :N, :K]
    return fp8.contiguous(), scale


def assert_fp8_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float8_e4m3fn, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    out_bytes, ref_bytes = out.view(torch.uint8), ref.view(torch.uint8)
    assert torch.equal(out_bytes, ref_bytes), f"{name}: {(out_bytes != ref_bytes).sum().item()} mismatching bytes"  # fmt: skip


def assert_f32_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float32, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    assert torch.equal(out, ref), f"{name}: {(out != ref).sum().item()} mismatching values"


def make_weight(
    shape: tuple, dtype: torch.dtype = torch.bfloat16, kind: str = "randn"
) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(repr((shape, kind)).encode()))
    x = torch.randn(shape, device="cuda", generator=gen, dtype=torch.float32)
    if kind == "outliers":
        mask = torch.rand(shape, device="cuda", generator=gen) < 1e-3
        x = torch.where(mask, x * 3e4, x)
        x.view(-1)[0] = 6e4
    elif kind == "zeros":
        x[..., : shape[-1] // 2] = 0.0
        if x.dim() == 3:
            x[0] = 0.0
    elif kind == "tiny":
        x = x * 1e-6
    elif kind == "wide":
        x = x * torch.exp2(torch.randint(-30, 30, shape, device="cuda", generator=gen).float())
    elif kind == "huge":
        x = x * 1e30
    elif kind == "negative":
        x = -x.abs()
    elif kind == "pow2":
        x = torch.exp2(torch.randint(-20, 20, shape, device="cuda", generator=gen).float()) * 448.0
    return x.to(dtype)


KINDS = ["randn", "outliers", "zeros", "tiny", "wide", "huge", "negative", "pow2"]


def check_blockwise_transpose(x: torch.Tensor) -> None:
    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(x)
    ref_fp8, ref_scale = ref_blockwise(x.unsqueeze(0))
    ref_fp8_t, ref_scale_t = ref_blockwise(x.T.unsqueeze(0))
    assert_fp8_equal("w_fp8", w_fp8, ref_fp8[0])
    assert_f32_equal("w_scale", w_scale, ref_scale[0])
    assert_fp8_equal("wt_fp8", wt_fp8, ref_fp8_t[0])
    assert_f32_equal("wt_scale", wt_scale, ref_scale_t[0])
    assert w_scale.is_contiguous() and wt_scale.is_contiguous()  # row-major by contract
    assert torch.equal(w_scale, wt_scale.T)


def check_blockwise_transpose_batched(x: torch.Tensor) -> None:
    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight_grouped(x)
    ref_fp8, ref_scale = ref_blockwise(x)
    ref_fp8_t, ref_scale_t = ref_blockwise(x.transpose(1, 2))
    assert_fp8_equal("w_fp8", w_fp8, ref_fp8)
    assert_f32_equal("w_scale", w_scale, ref_scale)
    assert_fp8_equal("wt_fp8", wt_fp8, ref_fp8_t)
    assert_f32_equal("wt_scale", wt_scale, ref_scale_t)
    assert w_scale.is_contiguous() and wt_scale.is_contiguous()
    assert torch.equal(w_scale, wt_scale.transpose(1, 2))


# (N, K) weights of the four models (fused gate_up / down / dense MLP / attention projections),
# tile edges, unaligned and tiny shapes.
SHAPES_2D = [
    (4096, 2048),
    (1536, 2048),
    (2048, 768),
    (2816, 2048),
    (2048, 1408),
    (21888, 2048),
    (2048, 10944),
    (5760, 2880),
    (2880, 2880),
    (512, 2048),
    (576, 10944),
    (128, 128),
    (256, 256),
    (300, 384),
    (100, 300),
    (64, 64),
    (7, 300),
    (130, 7),
    (1, 1),
    (256, 7168),
]
# (G, N, K) expert weights: gate_up (G, 2I, H) and down (G, H, I) of each model, plus odd shapes.
SHAPES_3D = [
    (4, 1536, 2048),
    (4, 2048, 768),
    (4, 2816, 2048),
    (4, 2048, 1408),
    (4, 1024, 2048),
    (4, 2048, 512),
    (2, 5760, 2880),
    (2, 2880, 2880),
    (1, 128, 128),
    (4, 256, 128),
    (8, 256, 512),
    (2, 300, 384),
    (4, 64, 64),
    (5, 384, 192),
    (3, 100, 260),
    (1, 129, 136),
]


@pytest.mark.parametrize("shape", SHAPES_2D, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("kind", KINDS)
def test_blockwise_transpose(shape: tuple, kind: str) -> None:
    check_blockwise_transpose(make_weight(shape, kind=kind))


@pytest.mark.parametrize("shape", [(512, 384), (200, 136), (2048, 768)])
@pytest.mark.parametrize("kind", ["randn", "wide"])
def test_blockwise_transpose_fp32(shape: tuple, kind: str) -> None:
    check_blockwise_transpose(make_weight(shape, dtype=torch.float32, kind=kind))


def test_blockwise_transpose_all_zeros() -> None:
    check_blockwise_transpose(torch.zeros(256, 384, device="cuda", dtype=torch.bfloat16))
    check_blockwise_transpose(torch.zeros(128, 256, device="cuda", dtype=torch.bfloat16))


def test_blockwise_transpose_strided() -> None:
    check_blockwise_transpose(make_weight((640, 2304), kind="outliers")[:, :2048])


@pytest.mark.parametrize("shape", SHAPES_3D, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("kind", KINDS)
def test_blockwise_transpose_batched(shape: tuple, kind: str) -> None:
    check_blockwise_transpose_batched(make_weight(shape, kind=kind))


def test_blockwise_transpose_batched_fp32_and_zeros() -> None:
    check_blockwise_transpose_batched(make_weight((3, 260, 384), dtype=torch.float32, kind="wide"))
    check_blockwise_transpose_batched(torch.zeros(4, 128, 256, device="cuda", dtype=torch.bfloat16))


def test_blockwise_transpose_empty() -> None:
    for shape in [(0, 256), (256, 0)]:
        x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(x)
        N, K = shape
        assert w_fp8.shape == (N, K) and wt_fp8.shape == (K, N)
        assert w_scale.shape == (cdiv(N, BLOCK), cdiv(K, BLOCK))
        assert wt_scale.shape == (cdiv(K, BLOCK), cdiv(N, BLOCK))


def test_blockwise_transpose_batched_empty() -> None:
    for shape in [(0, 256, 128), (2, 0, 128), (2, 128, 0)]:
        x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight_grouped(x)
        G, N, K = shape
        assert w_fp8.shape == (G, N, K) and wt_fp8.shape == (G, K, N)
        assert w_scale.shape == (G, cdiv(N, BLOCK), cdiv(K, BLOCK))
        assert wt_scale.shape == (G, cdiv(K, BLOCK), cdiv(N, BLOCK))


def make_shards(weight: torch.Tensor, num_shards: int) -> list[tuple[torch.Tensor, int, int]]:
    """
    Return the FSDP dim-0 shards of an fp32 master as the FP8 parameter gather sees them: (local
    (g_local, rows, K), row_offset, group_offset). A rank past the end holds an empty shard.
    """
    chunk = cdiv(weight.shape[0], num_shards)
    out = []
    for rank in range(num_shards):
        start, end = min(rank * chunk, weight.shape[0]), min(rank * chunk + chunk, weight.shape[0])
        local = weight[start:end].contiguous()
        if weight.dim() == 2:
            out.append((local.view(1, *local.shape), start, 0))
        else:
            out.append((local, 0, start))
    return out


def grid_of(shape: tuple) -> tuple[int, int, int]:
    G, N, K = (1, *shape) if len(shape) == 2 else shape
    return G, cdiv(N, BLOCK), cdiv(K, BLOCK)


def as_3d(w: torch.Tensor) -> torch.Tensor:
    return w.unsqueeze(0) if w.dim() == 2 else w


def shard_alone(shape: tuple, local: torch.Tensor, row_offset: int, group_offset: int) -> torch.Tensor:  # fmt: skip
    """
    Return the full (G, N, K) bf16-rounded weight with everything but this shard zeroed.
    """
    full = torch.zeros(as_3d(torch.empty(shape, device="meta")).shape, device=local.device)
    num_local_groups, rows, _ = local.shape
    full[group_offset : group_offset + num_local_groups, row_offset : row_offset + rows] = local.bfloat16().float()  # fmt: skip
    return full


# (weight shape, number of shards): 2D weights shard their rows, 3D weights their experts; tiles
# straddle shards where the shard size is not a multiple of 128, and some ranks hold nothing.
SHARDED = [
    ((4096, 2048), 16),
    ((512, 2048), 16),
    ((2048, 4096), 16),
    ((1536, 2048), 8),
    ((2048, 1408), 8),
    ((1000, 2880), 4),
    ((5760, 2880), 8),
    ((576, 10944), 5),
    ((300, 260), 7),
    ((300, 258), 7),
    ((6, 256), 4),
    ((16, 768, 2048), 2),
    ((16, 2048, 768), 2),
    ((8, 1536, 2048), 4),
    ((4, 2048, 1408), 4),
    ((2, 5760, 2880), 2),
    ((5, 384, 192), 2),
    ((5, 384, 192), 4),
    ((3, 130, 66), 2),
]


@pytest.mark.parametrize(
    "shape,n", SHARDED, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v)
)
@pytest.mark.parametrize("kind", ["wide", "outliers"])
def test_param_gather_pipeline(shape: tuple, n: int, kind: str) -> None:
    master = make_weight(shape, dtype=torch.float32, kind=kind)
    shards = make_shards(master, n)
    G, R, C = grid_of(shape)
    full_fp8, full_scale = ref_blockwise(as_3d(master.bfloat16()))
    full_fp8_t, full_scale_t = ref_blockwise(as_3d(master.bfloat16()).transpose(1, 2))

    amax = torch.zeros(G, R, C, device="cuda")
    for local, row_offset, group_offset in shards:
        shard_amax = torch.zeros(G, R, C, device="cuda")
        fp8_shard_amax(local, shard_amax, row_offset, group_offset)
        assert_f32_equal("shard amax", shard_amax, ref_amax(shard_alone(shape, local, row_offset, group_offset)))  # fmt: skip
        amax = torch.maximum(amax, shard_amax)
    assert_f32_equal("reduced amax", amax, ref_amax(as_3d(master.bfloat16())))

    scale = torch.full((G, R, C), -1.0, device="cuda")
    scale_t = torch.full((G, C, R), -1.0, device="cuda")
    fp8_scales(amax, scale, scale_t)
    assert_f32_equal("scale", scale, full_scale)
    assert_f32_equal("scale_t", scale_t, full_scale_t)

    parts = []
    for local, row_offset, group_offset in shards:
        out = torch.empty(local.shape, dtype=torch.float8_e4m3fn, device="cuda")
        fp8_shard_cast(local, amax, out, row_offset, group_offset)
        parts.append(out)
    gathered = torch.cat(parts, dim=1 if len(shape) == 2 else 0)
    assert_fp8_equal("gathered fp8", gathered, full_fp8)

    numel = gathered.numel()
    packed = torch.full((2 * numel,), 0x55, dtype=torch.uint8, device="cuda")
    fp8_pack_transpose(gathered, packed)
    assert_fp8_equal("packed fp8", packed[:numel].view(torch.float8_e4m3fn).view_as(full_fp8), full_fp8)  # fmt: skip
    packed_t = packed[numel:].view(torch.float8_e4m3fn).view(G, *full_fp8_t.shape[1:])
    assert_fp8_equal("packed fp8_t", packed_t, full_fp8_t)

    if len(shape) == 2:
        ref = fp8_quant_weight(master.bfloat16())
        ref = [t.unsqueeze(0) for t in ref]
    else:
        ref = fp8_quant_weight_grouped(master.bfloat16())
    assert_fp8_equal("vs blockwise fp8", gathered, ref[0])
    assert_f32_equal("vs blockwise scale", scale, ref[1])
    assert_fp8_equal("vs blockwise fp8_t", packed_t, ref[2])
    assert_f32_equal("vs blockwise scale_t", scale_t, ref[3])


def test_shard_amax_rounds_to_bf16() -> None:
    # 448 * (1 + 2**-12) has scale 2 in fp32 but rounds to 448 (scale 1) in bf16.
    weight = torch.full((1, 128, 128), 0.5, device="cuda")
    weight[0, 3, 5] = 448.0 * (1 + 2**-12)
    amax = torch.zeros(1, 1, 1, device="cuda")
    fp8_shard_amax(weight, amax, 0, 0)
    assert amax.item() == 448.0
    scale, scale_t = torch.empty(1, 1, 1, device="cuda"), torch.empty(1, 1, 1, device="cuda")
    fp8_scales(amax, scale, scale_t)
    assert scale.item() == 1.0 and scale_t.item() == 1.0
    out = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device="cuda")
    fp8_shard_cast(weight, amax, out, 0, 0)
    assert_fp8_equal("cast", out, weight.bfloat16().float().to(torch.float8_e4m3fn))


@pytest.mark.parametrize(
    "grid", [(1, 32, 16), (16, 6, 16), (16, 16, 6), (2, 5, 86), (3, 1, 1), (1, 171, 23)]
)
def test_scales(grid: tuple) -> None:
    amax = make_weight(grid, dtype=torch.float32, kind="wide").abs()
    amax.view(-1)[0] = 0.0
    amax.view(-1)[-1] = 448.0 * 2.0**-5  # exact power of two times 448
    G, R, C = grid
    scale, scale_t = torch.empty(G, R, C, device="cuda"), torch.empty(G, C, R, device="cuda")
    fp8_scales(amax, scale, scale_t)
    assert_f32_equal("scale", scale, e8m0_scale(amax))
    assert_f32_equal("scale_t", scale_t, e8m0_scale(amax).transpose(1, 2))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4096, 2048),
        (1, 512, 2048),
        (1, 2048, 4096),
        (16, 768, 2048),
        (16, 2048, 768),
        (1, 576, 10944),
        (2, 5760, 2880),
        (5, 384, 192),
        (3, 100, 260),
        (1, 129, 136),
        (2, 24, 40),
    ],
)
def test_pack_transpose(shape: tuple) -> None:
    x = make_weight(shape, kind="wide").to(torch.float8_e4m3fn)
    numel = x.numel()
    out = torch.full((2 * numel,), 0x55, dtype=torch.uint8, device="cuda")
    fp8_pack_transpose(x, out)
    assert torch.equal(out[:numel], x.view(torch.uint8).reshape(-1))
    assert torch.equal(out[numel:], x.transpose(1, 2).contiguous().view(torch.uint8).reshape(-1))


# The FP8 weights of one FSDP shard group: 2D and 3D, tiles straddling shards, an unaligned K
# (258, no 16-byte rows), and a weight whose last rank holds nothing.
GROUP = [
    (1536, 2048),
    (2048, 768),
    (1000, 2880),
    (4, 1536, 2048),
    (4, 2048, 768),
    (300, 258),
    (6, 256),
    (3, 130, 66),
]
RANKS = 4


def packed_grids(device: str = "cuda") -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Return one zeroed amax buffer holding every weight's grid back to back, as the FP8 parameter
    gather packs it, and a view per grid.
    """
    sizes = [grid_of(shape) for shape in GROUP]
    buffer = torch.zeros(sum(groups * rows * cols for groups, rows, cols in sizes), device=device)
    views, start = [], 0
    for groups, rows, cols in sizes:
        views.append(buffer[start : start + groups * rows * cols].view(groups, rows, cols))
        start += groups * rows * cols
    return buffer, views


def test_multi_launch_param_gather() -> None:
    masters = [make_weight(shape, dtype=torch.float32, kind="wide") for shape in GROUP]
    per_rank = [[make_shards(master, RANKS)[rank] for master in masters] for rank in range(RANKS)]

    reduced, reduced_views = packed_grids()
    launches = []
    for rank in range(RANKS):
        shards = per_rank[rank]
        buffer, amaxes = packed_grids()
        launch = fp8_shard_amax_multi(
            [shard[0] for shard in shards],
            amaxes,
            [shard[1] for shard in shards],
            [shard[2] for shard in shards],
        )
        assert launch is not None and callable(launch), "H200 must have a one-launch amax"
        launch()
        for (local, row_offset, group_offset), amax, shape in zip(shards, amaxes, GROUP):
            single = torch.zeros_like(amax)
            fp8_shard_amax(local, single, row_offset, group_offset)
            assert_f32_equal(f"multi amax {shape} rank {rank}", amax, single)
        launches.append((launch, buffer, amaxes))
        torch.maximum(reduced, buffer, out=reduced)

    for master, amax, shape in zip(masters, reduced_views, GROUP):
        assert_f32_equal(f"reduced amax {shape}", amax, ref_amax(as_3d(master.bfloat16())))

    scales = [torch.full(grid_of(shape), -1.0, device="cuda") for shape in GROUP]
    scales_t = [
        torch.full(grid_of(shape), -1.0, device="cuda").transpose(1, 2).contiguous()
        for shape in GROUP
    ]
    scales_launch = fp8_scales_multi(reduced_views, scales, scales_t)
    assert scales_launch is not None and callable(scales_launch), (
        "H200 must have a one-launch scales"
    )
    scales_launch()
    for master, amax, scale, scale_t, shape in zip(masters, reduced_views, scales, scales_t, GROUP):
        single, single_t = torch.empty_like(scale), torch.empty_like(scale_t)
        fp8_scales(amax, single, single_t)
        assert_f32_equal(f"multi scale {shape}", scale, single)
        assert_f32_equal(f"multi scale_t {shape}", scale_t, single_t)
        ref_fp8, ref_scale = ref_blockwise(as_3d(master.bfloat16()))
        assert_f32_equal(f"scale {shape}", scale, ref_scale)
        assert_f32_equal(f"scale_t {shape}", scale_t, ref_scale.transpose(1, 2))
        parts = []
        for rank in range(RANKS):
            local, row_offset, group_offset = per_rank[rank][GROUP.index(shape)]
            out = torch.empty(local.shape, dtype=torch.float8_e4m3fn, device="cuda")
            fp8_shard_cast(local, amax, out, row_offset, group_offset)
            parts.append(out)
        assert_fp8_equal(f"gathered {shape}", torch.cat(parts, dim=1 if len(shape) == 2 else 0), ref_fp8)  # fmt: skip

    # Launch objects are built once and reused: change the masters in place, re-zero the buffers
    # and launch again, as precompute_fp8_param_scales does after each optimizer step.
    for master in masters:
        master.mul_(-3.0).add_(0.25)
    per_rank_new = [[make_shards(master, RANKS)[rank] for master in masters] for rank in range(RANKS)]  # fmt: skip
    for rank, (launch, buffer, amaxes) in enumerate(launches):
        for (old, _, _), (new, _, _) in zip(per_rank[rank], per_rank_new[rank]):
            old.copy_(new)
        buffer.zero_()
        launch()
        for (local, row_offset, group_offset), amax, shape in zip(per_rank[rank], amaxes, GROUP):
            single = torch.zeros_like(amax)
            fp8_shard_amax(local, single, row_offset, group_offset)
            assert_f32_equal(f"relaunch amax {shape} rank {rank}", amax, single)
    reduced.zero_()
    for _, buffer, _ in launches:
        torch.maximum(reduced, buffer, out=reduced)
    scales_launch()
    for master, scale, scale_t, shape in zip(masters, scales, scales_t, GROUP):
        _, ref_scale = ref_blockwise(as_3d(master.bfloat16()))
        assert_f32_equal(f"relaunch scale {shape}", scale, ref_scale)
        assert_f32_equal(f"relaunch scale_t {shape}", scale_t, ref_scale.transpose(1, 2))


@pytest.mark.parametrize("grid", [(1, 32, 16), (16, 6, 16), (2, 5, 86), (3, 1, 1)])
def test_scales_multi(grid: tuple) -> None:
    amax = make_weight(grid, dtype=torch.float32, kind="wide").abs()
    amax.view(-1)[0] = 0.0
    G, R, C = grid
    grids = [amax, amax * 3, amax[:1].contiguous()]
    outs = [(torch.empty_like(grid_amax), torch.empty(grid_amax.shape[0], C, R, device="cuda")) for grid_amax in grids]  # fmt: skip
    launch = fp8_scales_multi(grids, [out_pair[0] for out_pair in outs], [out_pair[1] for out_pair in outs])  # fmt: skip
    assert launch is not None and callable(launch)
    launch()
    for grid_amax, (scale, scale_t) in zip(grids, outs):
        assert_f32_equal("scale", scale, e8m0_scale(grid_amax))
        assert_f32_equal("scale_t", scale_t, e8m0_scale(grid_amax).transpose(1, 2))
