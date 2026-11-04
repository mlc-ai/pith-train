"""
Contract: the SwiGLU silu(g) * u (or its two gradients), rounded to bf16 as the plain kernels
silu_mul_fwd and silu_mul_bwd round it, then quantized as fp8_quant_grouped does on SM90:
rowwise 1x128 FP8 plus the per-channel FP8 laid out K-major per group.

The kernels' sigmoid uses ex2.approx, which PyTorch cannot reproduce bit for bit, so the test checks
that the FP8 bytes and scale values equal, exactly, a PyTorch quantization of the bf16 output of
silu_mul_fwd and silu_mul_bwd, and that this output is within one bf16 rounding (plus fp32 noise) of
a float64 PyTorch evaluation of the same formula.

Recipe: per 128-element block, amax = max |x| ignoring NaN, scale = 2^ceil(log2(max(amax, 1e-4) /
448)) and fp8 = e4m3_satfinite(x * (1 / scale)), rounded to nearest even.
"""

import pytest
import torch

from pithtrain.kernels import (
    fp8_fused_experts_supported,
    fp8_silu_mul_bwd,
    fp8_silu_mul_fwd,
    fp8_silu_mul_gu_bwd,
    fp8_silu_mul_gu_fwd,
    silu_mul_bwd,
    silu_mul_fwd,
)

pytestmark = pytest.mark.skipif(
    not fp8_fused_experts_supported(),
    reason="needs the fused FP8 expert path (fp8_fused_experts_supported)",
)

BLOCK = 128


def e8m0_scale_rcp(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the recipe's power-of-two scale of an fp32 amax, and its exact reciprocal.
    """
    clamped = torch.maximum(amax, torch.full_like(amax, 1e-4))
    ratio = clamped * torch.full_like(clamped, 1.0 / 448.0)
    bits = ratio.view(torch.int32)
    bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    scale = torch.maximum(bits.view(torch.float32), torch.full_like(clamped, 2.0**-126))
    rcp = ((254 << 23) - scale.view(torch.int32)).view(torch.float32)
    return scale, rcp


def absmax_input(x: torch.Tensor) -> torch.Tensor:
    magnitude = x.abs()
    return torch.where(torch.isnan(magnitude), torch.zeros_like(magnitude), magnitude)


def quantize_bytes(x: torch.Tensor, rcp: torch.Tensor) -> torch.Tensor:
    return (x * rcp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def quantize_grouped_ref(x: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Return fp8_quant_grouped on SM90 in PyTorch: x_fp8 (M, K) as uint8, x_scale (M,
    ceil(K/128)), x_ch flat (M*K,) uint8 K-major per group, and x_ch_scale (K, M/128).
    """
    M, K = x.shape
    num_blocks = -(-K // BLOCK)
    x_pad = torch.zeros(M, num_blocks * BLOCK, dtype=torch.float32, device=x.device)
    x_pad[:, :K] = x.float()
    scale_row, rcp_row = e8m0_scale_rcp(absmax_input(x_pad).view(M, num_blocks, BLOCK).amax(-1))
    fp8_row = quantize_bytes(x_pad.view(M, num_blocks, BLOCK), rcp_row[..., None]).view(M, num_blocks * BLOCK)[:, :K]  # fmt: skip
    x_col = x_pad[:, :K].reshape(M // BLOCK, BLOCK, K)
    scale_ch, rcp_ch = e8m0_scale_rcp(absmax_input(x_col).amax(1))
    fp8_ch = quantize_bytes(x_col, rcp_ch[:, None, :]).view(M, K)
    flat = torch.zeros(M * K, dtype=torch.uint8, device=x.device)
    start = 0
    for end in offs.tolist():
        flat[start * K : end * K].view(K, end - start).copy_(fp8_ch[start:end].T)
        start = end
    return fp8_row.contiguous(), scale_row, flat, scale_ch.T.contiguous()


def assert_bytes_equal(got: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    """
    A NaN may carry either sign.
    """
    got = got.view(torch.uint8)
    nan = (ref & 0x7F) == 0x7F
    ok = (got == ref) | (nan & ((got & 0x7F) == 0x7F))
    if not bool(ok.all()):
        bad = (~ok).nonzero()
        raise AssertionError(f"{name}: {bad.shape[0]} byte mismatches, first at {bad[:4].tolist()}")


def assert_quant_equal(got: tuple, ref: tuple) -> None:
    names = ("x_fp8", "x_scale", "x_ch_fp8", "x_ch_scale")
    assert len(got) == 4
    for name, got_part, ref_part in zip(names, got, ref):
        assert got_part.shape == ref_part.shape, (name, got_part.shape, ref_part.shape)
        if name.endswith("fp8"):
            assert got_part.dtype == torch.float8_e4m3fn, name
            assert_bytes_equal(got_part, ref_part, name)
        else:
            assert got_part.dtype == torch.float32, name
            if not torch.equal(got_part, ref_part):
                bad = (got_part != ref_part).nonzero()
                raise AssertionError(f"{name}: {bad.shape[0]} mismatches, first {bad[:4].tolist()}")


def assert_bf16_close(
    got: torch.Tensor, ref64: torch.Tensor, mag: torch.Tensor, floor: torch.Tensor
) -> None:
    """
    Check that bf16 got rounds a value fp32-close to the float64 ref64: within one bf16 rounding,
    fp32 noise relative to the terms' magnitude mag, and floor times the fp32 sigmoid's absolute
    underflow (sigmoid(x) flushes to 0 for x below about -88).
    """
    got64 = got.double()
    special = ~torch.isfinite(ref64.to(torch.bfloat16).double())
    nan = torch.isnan(ref64)
    assert torch.equal(torch.isnan(got64), nan)
    assert torch.equal(got64[special & ~nan], ref64.to(torch.bfloat16).double()[special & ~nan])
    finite = ~special
    err = (got64[finite] - ref64[finite]).abs()
    tol = 2.0**-8 * ref64[finite].abs() + 2.0**-20 * mag[finite] + 1e-37 * floor[finite] + 1e-37
    if not bool((err <= tol).all()):
        worst = (err - tol).argmax()
        raise AssertionError(f"activation off: got {got64[finite][worst]}, want {ref64[finite][worst]}")  # fmt: skip


def check_fwd_activation(g: torch.Tensor, u: torch.Tensor, h: torch.Tensor) -> None:
    g64, u64 = g.double(), u.double()
    silu = g64 * torch.sigmoid(g64)
    assert_bf16_close(h, silu * u64, (silu * u64).abs(), (g64 * u64).abs())


def check_bwd_activation(dh, g, u, dg, du) -> None:
    dh64, g64, u64 = dh.double(), g.double(), u.double()
    sig = torch.sigmoid(g64)
    silu = g64 * sig
    assert_bf16_close(du, dh64 * silu, (dh64 * silu).abs(), (dh64 * g64).abs())
    front = (dh64 * u64 * sig).abs()
    floor = (dh64 * u64).abs() * (1 + g64.abs())
    assert_bf16_close(dg, dh64 * u64 * sig * (1.0 + g64 - silu), front * (1 + g64.abs()), floor)


def make_offs(sizes: list[int]) -> torch.Tensor:
    return torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)


def make_inputs(M: int, N: int, kind: str, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    xs = [torch.randn(M, N, device="cuda", generator=gen) for _ in range(3)]
    if kind == "outliers":
        for x in xs:
            mask = torch.rand(M, N, device="cuda", generator=gen) < 1e-3
            x[mask] *= 1e4
            x[torch.rand(M, N, device="cuda", generator=gen) < 0.05] = 0.0
            x[:128, :64] = 0.0  # whole zero blocks, so amax clamps to 1e-4
            x[128:256] *= 1e-30  # tiny products and flushed FP8
    elif kind == "wide":
        for x in xs:
            x.mul_(torch.exp2(torch.randint(-20, 20, (M, 1), device="cuda", generator=gen)))
    elif kind == "large":  # fp32 gradients up to about 1e36
        for x in xs:
            x.mul_(1e12)
    return [x.to(torch.bfloat16) for x in xs]


GROUPS = {
    "balanced16x2048": [2048] * 16,
    "imbalanced": [128, 0, 384, 1024, 0, 0, 256, 128, 2048, 0, 640],
    "single": [256],
    "empty_first_last": [0, 512, 128, 0],
    "many_small": [128] * 40 + [0] * 8,
    "experts128": [0, 256] * 32 + [128] * 64,
    "over_cap": [0] * 1100 + [256],  # > 1024 groups
    "all_empty_but_one": [0] * 7 + [128],
}

# (groups, N, kind). N: the expert intermediates of qwen3-30b-a3b (768), deepseek-v2-lite (1408),
# qwen3.5-35b-a3b (512), deepseek-v2-lite's dense MLP (10944), plus ragged / small ones.
CASES = [
    ("balanced16x2048", 768, "normal"),
    ("imbalanced", 768, "normal"),
    ("imbalanced", 768, "outliers"),
    ("imbalanced", 768, "wide"),
    ("imbalanced", 1408, "normal"),
    ("imbalanced", 512, "normal"),
    ("single", 10944, "normal"),
    ("single", 768, "normal"),
    ("single", 768, "large"),
    ("empty_first_last", 2048, "normal"),
    ("many_small", 768, "outliers"),
    ("imbalanced", 200, "outliers"),  # ragged N, partial last column block
    ("single", 72, "normal"),  # N < 128
    ("empty_first_last", 1416, "wide"),
    ("single", 100, "normal"),  # N % 8 != 0
    ("experts128", 768, "normal"),
    ("over_cap", 768, "normal"),
    ("all_empty_but_one", 512, "negative"),
]


def make_case(groups: str, N: int, kind: str, seed: int):
    sizes = GROUPS[groups]
    M = int(sum(sizes))
    dh, g, u = make_inputs(M, N, kind, seed=seed + N + M)
    if kind == "negative":
        g, u = -g.abs(), -u.abs()
    return make_offs(sizes), dh, g, u


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_fp8_silu_mul_fwd(groups: str, N: int, kind: str) -> None:
    offs, _, g, u = make_case(groups, N, kind, seed=0)
    h = silu_mul_fwd(g, u)
    check_fwd_activation(g, u, h)
    assert_quant_equal(fp8_silu_mul_fwd(g, u, offs), quantize_grouped_ref(h, offs))


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_fp8_silu_mul_gu_fwd(groups: str, N: int, kind: str) -> None:
    offs, _, g, u = make_case(groups, N, kind, seed=1)
    gu = torch.cat([g, u], 1).contiguous()
    h = silu_mul_fwd(g, u)
    assert_quant_equal(fp8_silu_mul_gu_fwd(gu, offs), quantize_grouped_ref(h, offs))


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_fp8_silu_mul_bwd(groups: str, N: int, kind: str) -> None:
    offs, dh, g, u = make_case(groups, N, kind, seed=2)
    dg, du = silu_mul_bwd(dh, g, u)
    check_bwd_activation(dh, g, u, dg, du)
    got_g, got_u = fp8_silu_mul_bwd(dh, g, u, offs)
    assert_quant_equal(got_g, quantize_grouped_ref(dg, offs))
    assert_quant_equal(got_u, quantize_grouped_ref(du, offs))


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_fp8_silu_mul_gu_bwd(groups: str, N: int, kind: str) -> None:
    offs, dh, g, u = make_case(groups, N, kind, seed=3)
    gu = torch.cat([g, u], 1).contiguous()
    dg, du = silu_mul_bwd(dh, g, u)
    got = fp8_silu_mul_gu_bwd(dh, gu, offs)
    assert_quant_equal(got, quantize_grouped_ref(torch.cat([dg, du], 1), offs))


def test_exhaustive_bf16_gate() -> None:
    """
    Check every finite BF16 gate value against several up scales, on all four entry points.
    """
    bits = torch.arange(0, 1 << 16, device="cuda", dtype=torch.int32).to(torch.int16)
    g_all = bits.view(torch.bfloat16)
    g_all = g_all[torch.isfinite(g_all)]
    N = 256
    M = (g_all.numel() + N - 1) // N
    M = (M + 127) // 128 * 128
    g = torch.zeros(M * N, device="cuda", dtype=torch.bfloat16)
    g[: g_all.numel()] = g_all
    g = g.view(M, N)
    offs = make_offs([M // 2, 0, M // 2])
    gen = torch.Generator(device="cuda").manual_seed(0)
    for scale in (1.0, 1e-3, 1e3):
        u = (torch.randn(M, N, device="cuda", generator=gen) * scale).to(torch.bfloat16)
        dh = torch.randn(M, N, device="cuda", generator=gen).to(torch.bfloat16)
        gu = torch.cat([g, u], 1).contiguous()
        h = silu_mul_fwd(g, u)
        check_fwd_activation(g, u, h)
        ref_h = quantize_grouped_ref(h, offs)
        assert_quant_equal(fp8_silu_mul_fwd(g, u, offs), ref_h)
        assert_quant_equal(fp8_silu_mul_gu_fwd(gu, offs), ref_h)
        dg, du = silu_mul_bwd(dh, g, u)
        check_bwd_activation(dh, g, u, dg, du)
        got_g, got_u = fp8_silu_mul_bwd(dh, g, u, offs)
        assert_quant_equal(got_g, quantize_grouped_ref(dg, offs))
        assert_quant_equal(got_u, quantize_grouped_ref(du, offs))
        got = fp8_silu_mul_gu_bwd(dh, gu, offs)
        assert_quant_equal(got, quantize_grouped_ref(torch.cat([dg, du], 1), offs))


def test_zeros() -> None:
    """
    All-zero inputs give zero bytes and every scale at the floor 2^ceil(log2(1e-4 / 448)).
    """
    M, N = 384, 768
    zeros = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    offs = make_offs([128, 256])
    floor = e8m0_scale_rcp(torch.zeros(1, device="cuda"))[0].item()
    for out in (
        fp8_silu_mul_fwd(zeros, zeros, offs),
        fp8_silu_mul_gu_fwd(torch.cat([zeros, zeros], 1), offs),
        *fp8_silu_mul_bwd(zeros, zeros, zeros, offs),
        fp8_silu_mul_gu_bwd(zeros, torch.cat([zeros, zeros], 1), offs),
    ):
        assert not bool(out[0].view(torch.uint8).any()) and not bool(out[2].view(torch.uint8).any())
        assert bool((out[1] == floor).all()) and bool((out[3] == floor).all())


def test_empty_input() -> None:
    """
    M == 0 (no tokens routed to this rank) returns correctly shaped empty outputs.
    """
    N = 768
    g = torch.empty(0, N, device="cuda", dtype=torch.bfloat16)
    gu = torch.empty(0, 2 * N, device="cuda", dtype=torch.bfloat16)
    offs = torch.zeros(4, device="cuda", dtype=torch.int32)

    def shapes(out, K):
        assert out[0].shape == (0, K) and out[0].dtype == torch.float8_e4m3fn
        assert out[1].shape == (0, -(-K // BLOCK)) and out[1].dtype == torch.float32
        assert out[2].numel() == 0 and out[2].dtype == torch.float8_e4m3fn
        assert out[3].shape == (K, 0) and out[3].dtype == torch.float32

    shapes(fp8_silu_mul_fwd(g, g, offs), N)
    shapes(fp8_silu_mul_gu_fwd(gu, offs), N)
    grad_gate_quant, grad_up_quant = fp8_silu_mul_bwd(g, g, g, offs)
    shapes(grad_gate_quant, N)
    shapes(grad_up_quant, N)
    shapes(fp8_silu_mul_gu_bwd(g, gu, offs), 2 * N)


QWEN35_COUNTS = [[1024] * 16, [0, 128, 256, 0, 640, 128, 1024, 0], [128]]


@pytest.mark.parametrize("counts", QWEN35_COUNTS)
@pytest.mark.parametrize("N", [512, 768, 520])
def test_gu_qwen35_cases(counts: list[int], N: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    M, offs = sum(counts), make_offs(counts)
    gu = torch.randn(M, 2 * N, device="cuda", generator=gen).to(torch.bfloat16) * 3
    gu[:5, :7] = 0
    gu[7, 9] = 300.0
    dh = torch.randn(M, N, device="cuda", generator=gen).to(torch.bfloat16)
    g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
    h = silu_mul_fwd(g, u)
    check_fwd_activation(g, u, h)
    assert_quant_equal(fp8_silu_mul_gu_fwd(gu, offs), quantize_grouped_ref(h, offs))
    dg, du = silu_mul_bwd(dh, g, u)
    check_bwd_activation(dh, g, u, dg, du)
    got = fp8_silu_mul_gu_bwd(dh, gu, offs)
    assert_quant_equal(got, quantize_grouped_ref(torch.cat([dg, du], 1), offs))
