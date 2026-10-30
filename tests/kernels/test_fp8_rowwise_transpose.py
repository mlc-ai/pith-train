"""
FP8 bytes and scale values must match the PyTorch reference exactly; scale strides are unspecified,
so scales are compared by logical value.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import fp8_quant_grad

BLOCK = 128


def e8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """
    Return 2 ** ceil(log2(max(amax, 1e-4) / 448)), computed exactly with frexp.
    """
    ratio = amax.float().clamp(min=1e-4) / 448.0
    mantissa, exponent = torch.frexp(ratio)  # ratio = mantissa * 2**exponent, mantissa in [0.5, 1)
    exponent = torch.where(mantissa == 0.5, exponent - 1, exponent)
    return ((exponent + 127) << 23).to(torch.int32).view(torch.float32)


def ref_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor rowwise in 1x128 blocks: (fp8 (M, K), scale (M, ceil(K/128))).
    """
    M, K = x.shape
    num_blocks = (K + BLOCK - 1) // BLOCK
    x_blocks = F.pad(x.float(), (0, num_blocks * BLOCK - K)).view(M, num_blocks, BLOCK)
    scale = e8m0_scale(x_blocks.abs().amax(dim=-1)) if M * num_blocks else torch.empty(M, num_blocks, device=x.device)  # fmt: skip
    fp8 = (x_blocks / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return fp8.view(M, num_blocks * BLOCK)[:, :K].contiguous(), scale


def assert_fp8_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float8_e4m3fn, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    out_bytes, ref_bytes = out.view(torch.uint8), ref.view(torch.uint8)
    assert torch.equal(out_bytes, ref_bytes), f"{name}: {(out_bytes != ref_bytes).sum().item()} mismatching bytes"  # fmt: skip


def assert_scale_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float32, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    assert torch.equal(out, ref), f"{name}: {(out != ref).sum().item()} mismatching scales"


def check(x: torch.Tensor) -> None:
    x_fp8, x_scale, xt_fp8, xt_scale = fp8_quant_grad(x)
    ref_fp8, ref_scale = ref_rowwise(x)
    ref_fp8_t, ref_scale_t = ref_rowwise(x.T)
    assert_fp8_equal("x_fp8", x_fp8, ref_fp8)
    assert_scale_equal("x_scale", x_scale, ref_scale)
    assert_fp8_equal("xt_fp8", xt_fp8, ref_fp8_t)
    assert_scale_equal("xt_scale", xt_scale, ref_scale_t)


# (M, N): model hidden / intermediate / dense sizes, tile edges, unaligned sizes.
SHAPES = [
    (4096, 4096),
    (4096, 2048),
    (4096, 1408),
    (4096, 768),
    (4096, 512),
    (4096, 2880),
    (1000, 10944),
    (4100, 576),
    (128, 128),
    (128, 256),
    (256, 256),
    (64, 256),
    (1, 128),
    (7, 64),
    (7, 300),
    (16, 8),
    (300, 200),
    (130, 1000),
    (257, 129),
    (64, 4104),
    (333, 1001),
    (96, 1028),
    (258, 1030),
    (4101, 512),
    (136, 384),
    (2056, 2048),
    (8, 4096),
    (256, 7168),
]


def make_input(
    m: int, n: int, kind: str, seed: int = 0, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, n, device="cuda", dtype=torch.float32, generator=gen)
    if kind == "outliers":
        mask = torch.rand(m, n, device="cuda", generator=gen) < 0.001
        x = torch.where(mask, x * 1e4, x)
        x[:, ::7] *= 1e-3
        x[::5, :] *= 30.0
    elif kind == "zeros":
        x[:, : max(1, n // 3)] = 0.0
        x[: max(1, m // 4), :] = 0.0
    elif kind == "tiny":
        x = x * 1e-7
    elif kind == "pow2":
        # amax exactly 448 * 2**k takes the exact-power-of-two (no round-up) branch of the ceil.
        x = x.clamp(-1, 1)
        x[:, 0] = 448.0 * 2.0**-3
        x[0, :] = -448.0
    elif kind == "wide":
        x = x * torch.exp2(torch.randint(-40, 40, (m, n), device="cuda", generator=gen).float())
    elif kind == "huge":
        x = x * 1e30
    elif kind == "negative":
        x = -x.abs()
    elif kind == "mantissa":
        # every one of the 24 significand bits set at random, so no bf16 / fp16 rounding is exact
        x = x + torch.randn(m, n, device="cuda", dtype=torch.float32, generator=gen) * 2.0**-12
    elif kind == "fp32max":
        x = x.clamp(-1, 1) * 3.0e38
    elif kind == "subnormal":
        # fp32 subnormals mixed with normals, and whole rows / columns of subnormals only
        mask = torch.rand(m, n, device="cuda", generator=gen) < 0.5
        x = torch.where(mask, x * 1e-40, x)
        x[::3, :] *= 1e-40
        x[:, ::5] *= 1e-40
    if dtype == torch.float16:
        x = x.clamp(-65504.0, 65504.0)
    return x.to(dtype)


KINDS = ["randn", "outliers", "zeros", "tiny", "pow2", "wide", "huge", "negative"]
# fp32 only: values bf16 cannot hold (full significand, near the fp32 max, subnormals)
FP32_KINDS = KINDS + ["mantissa", "fp32max", "subnormal"]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", KINDS)
def test_matches_reference(shape: tuple[int, int], kind: str) -> None:
    check(make_input(*shape, kind))


def test_all_zeros() -> None:
    check(torch.zeros(64, 256, device="cuda", dtype=torch.bfloat16))


def test_misaligned_base_pointer() -> None:
    base = make_input(512, 1032, "outliers")
    for off in (3, 4, 8):
        check(base.view(-1)[off : off + 512 * 1024].view(512, 1024))


def test_non_contiguous() -> None:
    check(make_input(256, 512, "randn")[:, :384])
    check(make_input(512, 300, "outliers").T)


DTYPES = [torch.bfloat16, torch.float16, torch.float32]
DTYPE_IDS = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def kinds_of(dtype: torch.dtype) -> list[str]:
    return FP32_KINDS if dtype == torch.float32 else KINDS


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", FP32_KINDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=DTYPE_IDS.get)
def test_dtypes(shape: tuple[int, int], kind: str, dtype: torch.dtype) -> None:
    if kind not in kinds_of(dtype):
        pytest.skip("fp32-only kind")
    check(make_input(*shape, kind, dtype=dtype))


def misaligned_rows(m: int, n: int, kind: str, dtype: torch.dtype, offset: int) -> torch.Tensor:
    """
    (m, n) view whose base sits offset elements into its storage and whose rows are n + 3 apart.
    """
    flat = make_input(m + 1, n + 3, kind, dtype=dtype).view(-1)
    return flat[offset : offset + m * (n + 3)].view(m, n + 3)[:, :n]


# (m, n, kind, dtype) -> an (m, n) view of each stride class the kernel must take as is.
VIEWS = {
    "rowstrided_even": lambda m, n, k, d: make_input(m, n + 64, k, dtype=d)[:, :n],
    "rowstrided_odd": lambda m, n, k, d: make_input(m, n + 3, k, dtype=d)[:, :n],
    "transposed": lambda m, n, k, d: make_input(n, m, k, dtype=d).T,
    "transposed_rowstrided": lambda m, n, k, d: make_input(n, m + 5, k, dtype=d)[:, :m].T,
    "doubly_strided": lambda m, n, k, d: make_input(2 * m, 3 * n, k, dtype=d)[::2, ::3],
    "colstride17": lambda m, n, k, d: make_input(m, 17 * n, k, dtype=d)[:, ::17],
    "misaligned_odd1": lambda m, n, k, d: misaligned_rows(m, n, k, d, 1),
    "misaligned_odd2": lambda m, n, k, d: misaligned_rows(m, n, k, d, 2),
    "misaligned_odd3": lambda m, n, k, d: misaligned_rows(m, n, k, d, 3),
    "stride0_rows": lambda m, n, k, d: make_input(1, n, k, dtype=d).expand(m, n),
    "stride0_cols": lambda m, n, k, d: make_input(m, 1, k, dtype=d).expand(m, n),
    "stride0_both": lambda m, n, k, d: make_input(1, 1, k, dtype=d).expand(m, n),
}


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("view", list(VIEWS))
def test_strided_views(shape: tuple[int, int], kind: str, view: str) -> None:
    x = VIEWS[view](*shape, kind, torch.bfloat16)
    assert x.shape == shape
    check(x)


CROSS_SHAPES = [(4096, 2048), (4100, 576), (257, 129), (333, 1001), (7, 300), (16, 8)]
CROSS_VIEWS = ["rowstrided_odd", "transposed", "transposed_rowstrided", "doubly_strided", "misaligned_odd1", "misaligned_odd2", "misaligned_odd3"]  # fmt: skip


@pytest.mark.parametrize("shape", CROSS_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", FP32_KINDS)
@pytest.mark.parametrize("view", CROSS_VIEWS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=DTYPE_IDS.get)
def test_dtype_views(shape: tuple[int, int], kind: str, view: str, dtype: torch.dtype) -> None:
    if kind not in kinds_of(dtype):
        pytest.skip("fp32-only kind")
    check(VIEWS[view](*shape, kind, dtype))


def strided_storage(shape: tuple[int, int], stride: tuple[int, int], kind: str, dtype: torch.dtype) -> torch.Tensor:  # fmt: skip
    """
    shape / stride view of fresh storage, with a 3-element storage offset.
    """
    size = 3 + sum((dim - 1) * step for dim, step in zip(shape, stride)) + 1
    return make_input(1, size, kind, dtype=dtype).view(-1).as_strided(shape, stride, 3)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS.get)
def test_size_one_dims(shape: tuple[int, int], kind: str, dtype: torch.dtype) -> None:
    """
    The stride of a size-1 dim is arbitrary in torch (odd, huge, or unit on the wrong axis).
    """
    m, n = shape
    for size, stride in (((1, n), (7, 1)), ((1, n), (2 * n + 1, 2)), ((1, n), (1, 1)), ((m, 1), (1, 7)), ((m, 1), (3, 5)), ((m, 1), (1, 1)), ((1, 1), (5, 9))):  # fmt: skip
        check(strided_storage(size, stride, kind, dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS.get)
def test_fp32_boundaries(dtype: torch.dtype) -> None:
    """
    Block amaxes at 448 * 2**k and 1..16 ulps either side (the ceil of the E8M0 scale), and every
    midpoint between two adjacent e4m3 values (round half to even) at scale 1.
    """
    rows = []
    if dtype == torch.float32:
        for k in (-20, -3, 0, 5, 30, 100):
            center = torch.tensor(448.0 * 2.0**k, dtype=torch.float32)
            bits = center.view(torch.int32).item()
            for delta in range(-16, 17):
                amax = torch.tensor(bits + delta, dtype=torch.int32).view(torch.float32).item()
                row = torch.linspace(-amax, amax, BLOCK, dtype=torch.float32) * 0.5
                row[delta % BLOCK] = amax if delta % 2 else -amax
                rows.append(row)
    e4m3 = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mids = (e4m3[:-1] + e4m3[1:]) / 2
    ties = torch.cat([mids, -mids, torch.tensor([448.0])])
    ties = F.pad(ties, (0, -ties.numel() % BLOCK), value=1.0).view(-1, BLOCK).clone()
    ties[:, 0] = 448.0
    rows += list(ties)
    x = torch.stack(rows).to(dtype).cuda()
    check(x)
    check(x.T.contiguous())


def test_reference_matches_recipe() -> None:
    """
    The reference's divide by 448 and the recipe's multiply by fp32(1 / 448) give the same E8M0 scale
    at the exponent boundaries the kernels round at.
    """
    centers = torch.tensor([448.0 * 2.0**k for k in range(-30, 120)], dtype=torch.float32)
    deltas = torch.arange(-200, 201, dtype=torch.int32)
    amax = (centers.view(torch.int32)[:, None] + deltas[None, :]).view(torch.float32).flatten()
    amax = torch.cat([amax, torch.tensor([0.0, 1e-4, 5e-5, 3.0e38])])
    recipe = amax.clamp(min=1e-4) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    bits = recipe.view(torch.int32)
    round_up = torch.where((bits & 0x007FFFFF) != 0, 0x00800000, 0).to(torch.int32)
    exp_bits = (bits & 0x7F800000) + round_up
    recipe_scale = exp_bits.clamp(min=0x00800000).view(torch.float32)
    assert torch.equal(e8m0_scale(amax), recipe_scale)


@pytest.mark.parametrize("shape", [(0, 256), (256, 0), (0, 0)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS.get)
def test_empty_dtypes(shape: tuple[int, int], dtype: torch.dtype) -> None:
    m, n = shape
    x = torch.empty(m, n, device="cuda", dtype=dtype)
    out = fp8_quant_grad(x)
    expected = [(m, n), (m, (n + BLOCK - 1) // BLOCK), (n, m), (n, (m + BLOCK - 1) // BLOCK)]
    assert [tuple(t.shape) for t in out] == expected
    assert [t.dtype for t in out] == [torch.float8_e4m3fn, torch.float32] * 2
    for scale in out[1::2]:
        rows = scale.shape[0]
        # strides are unspecified: row-major or MN-major (columns of align(rows, 4))
        assert scale.stride() in ((scale.shape[1], 1), (1, (rows + 3) // 4 * 4)), scale.stride()


@pytest.mark.parametrize("dtype", [torch.int32, torch.float64, torch.uint8], ids=str)
@pytest.mark.parametrize("shape", [(64, 256), (0, 256)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_unsupported_dtype(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    x = torch.zeros(shape, device="cuda", dtype=dtype)
    with pytest.raises(TypeError, match="bfloat16"):
        fp8_quant_grad(x)


@pytest.mark.parametrize("shape", [(256,), (2, 64, 128)], ids=str)
def test_not_2d(shape: tuple[int, ...]) -> None:
    with pytest.raises(AssertionError):
        fp8_quant_grad(torch.zeros(shape, device="cuda", dtype=torch.bfloat16))


def test_empty() -> None:
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    out = fp8_quant_grad(x)
    assert [tuple(t.shape) for t in out] == [(0, 256), (0, 2), (256, 0), (256, 0)]
