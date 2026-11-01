"""
Verifier of the fp8_weight_quant feature against a PyTorch 128x128 blockwise FP8 quantization (one
E8M0 scale per 128x128 tile, from the tile amax).

- fp8_quant_weight and fp8_quant_weight_grouped: the blockwise FP8 of W and
  of W.T (per expert for a (G, N, K) weight), with row-major scales, on bf16 and fp32 inputs of
  any strides (transposed, row-padded, column-strided, expanded and misaligned views), with fresh
  outputs, tiles holding inf (either E8M0 result of an infinite amax, as the other FP8 verifiers
  accept), and called inside torch.compile(fullgraph=True, options={"cpp_wrapper": True}) as the
  models call them.
- The FP8 parameter all-gather kernels, driven as the FP8 parameter gather drives them. FSDP dim-0
  shards of an fp32 master (rows of a 2D weight, experts of a 3D one) each write the per-tile amax
  of the bf16-rounded shard into a zeroed grid (fp8_shard_amax). The grids are MAX-reduced, the
  scales follow (fp8_scales), every shard is cast with them (fp8_shard_cast) and the gathered FP8
  weight is packed with its transpose (fp8_pack_transpose). The result must equal the blockwise
  quantization of the full bf16 weight.
- fp8_shard_amax_multi and fp8_scales_multi return one launch for a list of weights (a callable on
  the H200) whose results must equal the per-tensor calls, or None where the device has no such
  launch, in which case the tests run the per-tensor loop callers fall back to.
"""

import warnings
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_blockwise_transpose_empty(dtype: torch.dtype) -> None:
    for shape in [(0, 256), (256, 0), (0, 0)]:
        x = torch.empty(shape, device="cuda", dtype=dtype)
        w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(x)
        N, K = shape
        assert w_fp8.shape == (N, K) and wt_fp8.shape == (K, N)
        assert w_scale.shape == (cdiv(N, BLOCK), cdiv(K, BLOCK))
        assert wt_scale.shape == (cdiv(K, BLOCK), cdiv(N, BLOCK))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_blockwise_transpose_batched_empty(dtype: torch.dtype) -> None:
    for shape in [(0, 256, 128), (2, 0, 128), (2, 128, 0), (0, 0, 0)]:
        x = torch.empty(shape, device="cuda", dtype=dtype)
        w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight_grouped(x)
        G, N, K = shape
        assert w_fp8.shape == (G, N, K) and wt_fp8.shape == (G, K, N)
        assert w_scale.shape == (G, cdiv(N, BLOCK), cdiv(K, BLOCK))
        assert wt_scale.shape == (G, cdiv(K, BLOCK), cdiv(N, BLOCK))


@pytest.mark.parametrize(
    "shape", [(512, 384), (200, 136), (2048, 768), (7, 300), (130, 7), (1, 1), (4097, 2047)]
)
@pytest.mark.parametrize("kind", ["randn", "wide", "huge", "tiny"])
def test_blockwise_transpose_fp32_partial(shape: tuple, kind: str) -> None:
    check_blockwise_transpose(make_weight(shape, dtype=torch.float32, kind=kind))


@pytest.mark.parametrize("shape", [(3, 260, 384), (4, 768, 2048), (2, 100, 300), (1, 129, 136)])
@pytest.mark.parametrize("kind", ["randn", "wide", "huge"])
def test_blockwise_transpose_batched_fp32(shape: tuple, kind: str) -> None:
    check_blockwise_transpose_batched(make_weight(shape, dtype=torch.float32, kind=kind))


def misaligned(shape: tuple, dtype: torch.dtype, kind: str) -> torch.Tensor:
    """
    A contiguous view one element past a fresh allocation (data_ptr % 16 == element size).
    """
    numel = 1
    for dim in shape:
        numel *= dim
    return make_weight((numel + 1,), dtype=dtype, kind=kind)[1:].view(shape)


def strided(shape: tuple, stride: tuple, dtype: torch.dtype, kind: str) -> torch.Tensor:
    """
    An as_strided view of shape and stride (in elements) into a fresh buffer.
    """
    span = 1 + sum((size - 1) * step for size, step in zip(shape, stride))
    return make_weight((span + 3,), dtype=dtype, kind=kind).as_strided(shape, stride, 3)


# 2-D views of every stride pattern the kernels take: transposed (unit stride along N, with and
# without a storage offset), row-padded, column-strided, misaligned, expanded (zero strides),
# singleton dims with arbitrary strides, odd shapes.
VIEWS_2D = {
    "transposed": lambda d: make_weight((2048, 4096), d, "outliers").T,
    "transposed_partial": lambda d: make_weight((300, 130), d, "wide").T,
    "transposed_offset": lambda d: make_weight((301, 384), d, "wide")[1:].T,
    "transposed_misaligned": lambda d: misaligned((260, 300), d, "wide").T,
    "rowstrided": lambda d: make_weight((640, 2304), d, "outliers")[:, :2048],
    "rowstrided_partial": lambda d: make_weight((300, 400), d, "wide")[:, :260],
    "colstrided": lambda d: make_weight((640, 4096), d, "wide")[:, ::2],
    "colstrided_partial": lambda d: make_weight((300, 520), d, "wide")[:, ::2],
    "both_strided": lambda d: strided((300, 260), (1000, 3), d, "wide"),
    "misaligned": lambda d: misaligned((1024, 2048), d, "outliers"),
    "misaligned_partial": lambda d: misaligned((300, 260), d, "wide"),
    "expanded_rows": lambda d: make_weight((1, 384), d, "wide").expand(300, 384),
    "expanded_cols": lambda d: make_weight((300, 1), d, "wide").expand(300, 260),
    "row_singleton": lambda d: strided((1, 256), (7, 1), d, "wide"),
    "col_singleton": lambda d: strided((300, 1), (3, 5), d, "wide"),
    "odd": lambda d: make_weight((4097, 2047), d, "wide"),
}
# 3-D views: transposed experts, padded rows, every other expert, independent group / row / column
# strides (including a permuted layout), misaligned, expanded experts, singleton dims, odd shapes.
VIEWS_3D = {
    "transposed": lambda d: make_weight((4, 2048, 768), d, "outliers").transpose(1, 2),
    "transposed_partial": lambda d: make_weight((3, 300, 130), d, "wide").transpose(1, 2),
    "transposed_offset": lambda d: make_weight((4, 301, 384), d, "wide")[:, 1:].transpose(1, 2),
    "rowstrided": lambda d: make_weight((4, 768, 2304), d, "outliers")[:, :, :2048],
    "every_other_expert": lambda d: make_weight((8, 300, 260), d, "wide")[::2],
    "every_other_expert_aligned": lambda d: make_weight((8, 256, 512), d, "wide")[::2],
    "independent_strides": lambda d: make_weight((6, 300, 528), d, "wide")[::2, :, ::2],
    "permuted": lambda d: make_weight((260, 3, 200), d, "wide").permute(1, 0, 2),
    "permuted_aligned": lambda d: make_weight((256, 4, 512), d, "wide").permute(1, 0, 2),
    "misaligned": lambda d: misaligned((4, 384, 512), d, "outliers"),
    "misaligned_partial": lambda d: misaligned((3, 260, 300), d, "wide"),
    "expanded_experts": lambda d: make_weight((1, 256, 384), d, "wide").expand(4, 256, 384),
    "singletons": lambda d: strided((1, 300, 1), (5, 7, 3), d, "wide"),
    "odd": lambda d: make_weight((8, 1000, 1004), d, "wide"),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("view", list(VIEWS_2D))
def test_blockwise_transpose_views(view: str, dtype: torch.dtype) -> None:
    check_blockwise_transpose(VIEWS_2D[view](dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("view", list(VIEWS_3D))
def test_blockwise_transpose_batched_views(view: str, dtype: torch.dtype) -> None:
    check_blockwise_transpose_batched(VIEWS_3D[view](dtype))


def e8m0_bits_scale(amax: torch.Tensor, native: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The recipe's scale of an fp32 amax with the E8M0 arithmetic of the device, and its exact
    reciprocal (2^254 / scale as bits): the SM90 arithmetic gives an infinite amax the scale inf
    (reciprocal -inf); native saturates it to 2^127 as SM100's cvt does. Both equal e8m0_scale on
    every finite amax.
    """
    ratio = torch.maximum(amax, torch.full_like(amax, 1e-4)) * torch.full_like(amax, 1.0 / 448.0)
    bits = ratio.view(torch.int32)
    bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    scale = torch.maximum(bits.view(torch.float32), torch.full_like(amax, 2.0**-126))
    if native:
        scale = torch.minimum(scale, torch.full_like(scale, 2.0**127))
    return scale, ((254 << 23) - scale.view(torch.int32)).view(torch.float32)


def inf_tile_refs(x: torch.Tensor) -> list:
    """
    (fp8 bytes (G, N, K) uint8, scale (G, R, C)) of a (G, N, K) x under both E8M0 results.
    """
    G, N, K = x.shape
    refs = []
    for native in (False, True):
        scale, rcp = e8m0_bits_scale(ref_amax(x), native)
        fp8 = (tiles(x) * rcp[:, :, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        fp8 = fp8.view(G, scale.shape[1] * BLOCK, scale.shape[2] * BLOCK)[:, :N, :K]
        refs.append((fp8.contiguous().view(torch.uint8), scale))
    return refs


def assert_inf_tiles(name: str, out: torch.Tensor, scale: torch.Tensor, x: torch.Tensor) -> None:
    """
    out / scale of the (G, N, K) x: in a tile whose amax is infinite, either E8M0 result element by
    element (NaN bytes of either sign are one value); every other tile exactly the recipe.
    """
    (sm90_fp8, sm90_scale), (native_fp8, native_scale) = inf_tile_refs(x)
    got = out.reshape(sm90_fp8.shape).view(torch.uint8)
    nan = (got & 0x7F) == 0x7F
    ok = (got == sm90_fp8) | (got == native_fp8) | (nan & ((sm90_fp8 & 0x7F) == 0x7F))
    inf_tile = torch.isinf(ref_amax(x))
    exact = (got == sm90_fp8) & (sm90_fp8 == native_fp8)
    per_elem_inf = inf_tile.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)
    per_elem_inf = per_elem_inf[:, : x.shape[1], : x.shape[2]]
    assert bool(torch.where(per_elem_inf, ok, exact).all()), f"{name}: bytes differ"
    got_scale = scale.reshape(sm90_scale.shape)
    assert torch.equal(got_scale[~inf_tile], sm90_scale[~inf_tile]), f"{name}: finite-tile scales"
    in_inf = got_scale[inf_tile]
    assert bool(((in_inf == sm90_scale[inf_tile]) | (in_inf == native_scale[inf_tile])).all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("view", ["contiguous", "transposed"])
def test_blockwise_transpose_inf_tiles(view: str, dtype: torch.dtype) -> None:
    """
    Tiles holding +inf or -inf (2-D and 3-D, a partial tile among them, a contiguous and a
    transposed view): an infinite amax takes either E8M0 result, every other tile is exact.
    """
    w = make_weight((300, 260), dtype, "randn")
    w[3, 5], w[200, 130], w[299, 259] = float("inf"), float("-inf"), float("inf")
    w3 = make_weight((3, 260, 300), dtype, "randn")
    w3[0, 1, 2], w3[2, 259, 299] = float("-inf"), float("inf")
    if view == "transposed":
        w, w3 = w.T.contiguous().T, w3.transpose(1, 2).contiguous().transpose(1, 2)
    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(w)
    assert_inf_tiles("w_fp8", w_fp8, w_scale, w.unsqueeze(0))
    assert_inf_tiles("wt_fp8", wt_fp8, wt_scale, w.T.unsqueeze(0))
    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight_grouped(w3)
    assert_inf_tiles("grouped w_fp8", w_fp8, w_scale, w3)
    assert_inf_tiles("grouped wt_fp8", wt_fp8, wt_scale, w3.transpose(1, 2))


def assert_outputs_match(got: tuple, x: torch.Tensor) -> None:
    """
    got, the four outputs of fp8_quant_weight[_grouped] of the 3-D x (or 2-D squeezed), equal the
    recipe of x and of its per-expert transpose.
    """
    x3 = x if x.dim() == 3 else x.unsqueeze(0)
    want_fp8, want_scale = ref_blockwise(x3)
    want_fp8_t, want_scale_t = ref_blockwise(x3.transpose(1, 2))
    shape = (lambda t: t) if x.dim() == 3 else (lambda t: t[0])
    for name, out, ref in zip(
        ["w_fp8", "w_scale", "wt_fp8", "wt_scale"],
        got,
        [want_fp8, want_scale, want_fp8_t, want_scale_t],
    ):
        (assert_fp8_equal if out.dtype == torch.float8_e4m3fn else assert_f32_equal)(
            name, out, shape(ref)
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_blockwise_transpose_side_stream(compiled: bool) -> None:
    """
    The entries launch on the current stream, eager and inside a torch.compile(fullgraph=True,
    cpp_wrapper=True) region: on a fresh side stream, behind a long sleep, an input written there is
    quantized correctly (a launch on the default stream would read the zeros it held before), and
    the work is still queued on the side stream when the calls return.
    """
    src2, src3 = (
        make_weight((1536, 2048), kind="outliers"),
        make_weight((4, 768, 2048), kind="wide"),
    )
    x2, x3 = torch.zeros_like(src2), torch.zeros_like(src3)
    entry2, entry3 = fp8_quant_weight, fp8_quant_weight_grouped
    if compiled:
        torch._dynamo.reset()
        options = {"cpp_wrapper": True}
        entry2 = torch.compile(lambda w: fp8_quant_weight(w), fullgraph=True, options=options)
        entry3 = torch.compile(
            lambda w: fp8_quant_weight_grouped(w), fullgraph=True, options=options
        )
    # warm both entries (first-call builds and compiles) so the sleep below outlasts the calls
    entry2(torch.zeros_like(src2)), entry3(torch.zeros_like(src3))
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(100_000_000)
        x2.copy_(src2)
        x3.copy_(src3)
        got2, got3 = entry2(x2), entry3(x3)
        done = torch.cuda.Event()
        done.record()
    assert not done.query(), (
        "the side stream finished before the sleep could: the sleep is too short"
    )
    side.synchronize()
    assert_outputs_match(got2, src2)
    assert_outputs_match(got3, src3)
    if compiled:
        torch._dynamo.reset()


def storages(tensors) -> list:
    return [t.untyped_storage() for t in tensors]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "entry, shape",
    [
        (fp8_quant_weight, (300, 260)),
        (fp8_quant_weight, (0, 256)),
        (fp8_quant_weight, (256, 0)),
        (fp8_quant_weight_grouped, (3, 260, 300)),
        (fp8_quant_weight_grouped, (0, 128, 256)),
        (fp8_quant_weight_grouped, (2, 0, 128)),
    ],
    ids=["2d", "2d_no_rows", "2d_no_cols", "3d", "3d_no_experts", "3d_no_rows"],
)
def test_blockwise_transpose_fresh_outputs(entry, shape: tuple, dtype: torch.dtype) -> None:
    """
    The four outputs are fresh: contiguous tensors that are not views, each with its own storage,
    shared neither with x nor with each other nor with the outputs of an earlier call kept alive
    (checked by storage object, so empty outputs count too).
    """
    x = (
        make_weight(shape, dtype, "wide")
        if 0 not in shape
        else torch.empty(shape, device="cuda", dtype=dtype)
    )
    first, second = entry(x), entry(x)
    for outputs in (first, second):
        assert all(t.is_contiguous() and t._base is None for t in outputs)
    owned = storages(first) + storages(second) + [x.untyped_storage()]
    for i, a in enumerate(owned):
        for b in owned[i + 1 :]:
            assert a._cdata != b._cdata, "two tensors share a storage"
            if a.nbytes() and b.nbytes():
                assert a.data_ptr() != b.data_ptr(), "two storages share memory"


def assert_same_outputs(got: tuple, want: tuple) -> None:
    for name, out, ref in zip(["w_fp8", "w_scale", "wt_fp8", "wt_scale"], got, want):
        assert out.is_contiguous(), name
        if out.dtype == torch.float8_e4m3fn:
            assert_fp8_equal(name, out, ref)
        else:
            assert_f32_equal(name, out, ref)


# (entry, input, view applied inside the compiled region): contiguous, a strided slice and a
# transposed view, bf16 and fp32, 2-D and 3-D.
COMPILE_CASES = {
    "2d": (fp8_quant_weight, (1536, 2048), lambda w: w),
    "2d_rowstrided": (fp8_quant_weight, (640, 2304), lambda w: w[:, :2048]),
    "2d_transposed": (fp8_quant_weight, (2048, 768), lambda w: w.T),
    "3d": (fp8_quant_weight_grouped, (4, 768, 2048), lambda w: w),
    "3d_rowstrided": (fp8_quant_weight_grouped, (4, 300, 400), lambda w: w[:, :, :260]),
    "3d_transposed": (fp8_quant_weight_grouped, (4, 2048, 768), lambda w: w.transpose(1, 2)),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("case", list(COMPILE_CASES))
def test_blockwise_transpose_compiled(case: str, dtype: torch.dtype) -> None:
    """
    The entries inside torch.compile with the models' options: the first call (the op first seen
    inside the region) and a second, cached call on new values equal the eager outputs byte for
    byte, with the weight requiring grad as a model's does.
    """
    entry, shape, view = COMPILE_CASES[case]
    torch._dynamo.reset()

    @torch.compile(fullgraph=True, options={"cpp_wrapper": True})
    def region(w: torch.Tensor) -> tuple:
        return entry(view(w))

    for kind in ["outliers", "wide"]:
        w = make_weight(shape, dtype, kind).requires_grad_(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = region(w)
        assert not [str(c.message) for c in caught if "autograd" in str(c.message)], caught
        assert_same_outputs(got, entry(view(w)))
        assert not any(out.requires_grad for out in got)
    torch._dynamo.reset()


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


def amax_multi_or_loop(shards, amaxes, row_offsets, group_offsets):
    """
    fp8_shard_amax_multi's one-launch callable, or, where it returns None (no such path on the
    device), the per-shard loop over fp8_shard_amax its contract tells callers to run.
    """
    launch = fp8_shard_amax_multi(shards, amaxes, row_offsets, group_offsets)
    if launch is None:
        args = list(zip(shards, amaxes, row_offsets, group_offsets))
        return lambda: [fp8_shard_amax(*arg) for arg in args]
    assert callable(launch)
    return launch


def scales_multi_or_loop(amaxes, scales, scales_t):
    """
    fp8_scales_multi's one-launch callable, or the loop over fp8_scales where it returns None.
    """
    launch = fp8_scales_multi(amaxes, scales, scales_t)
    if launch is None:
        args = list(zip(amaxes, scales, scales_t))
        return lambda: [fp8_scales(*arg) for arg in args]
    assert callable(launch)
    return launch


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
        launch = amax_multi_or_loop(
            [shard[0] for shard in shards],
            amaxes,
            [shard[1] for shard in shards],
            [shard[2] for shard in shards],
        )
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
    scales_launch = scales_multi_or_loop(reduced_views, scales, scales_t)
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
    launch = scales_multi_or_loop(grids, [out_pair[0] for out_pair in outs], [out_pair[1] for out_pair in outs])  # fmt: skip
    launch()
    for grid_amax, (scale, scale_t) in zip(grids, outs):
        assert_f32_equal("scale", scale, e8m0_scale(grid_amax))
        assert_f32_equal("scale_t", scale_t, e8m0_scale(grid_amax).transpose(1, 2))
