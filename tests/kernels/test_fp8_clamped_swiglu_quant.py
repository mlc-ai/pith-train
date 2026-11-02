"""
Contract: gpt-oss's clamped SwiGLU of gate_up + bias[group] (gate at even and up at odd columns; the
bf16 bias added in fp32 and rounded to bf16), rounded to bf16 as clamped_swiglu_fwd /
clamped_swiglu_bwd round it, then quantized as fp8_quant_grouped does: rowwise 1x128 plus
per-channel FP8 in fp8_quant_grouped's layout (K-major per group on SM90, (M, C) with scale (M / 128,
C) on SM100; the checks read the layout the output has). With padded=True the K-major pair has
ceil(C / 256) * 256 channels (C = N forward, 2N backward) where the fused SM90 kernel runs and C
elsewhere; the extra channels are unwritten. The backward also returns the per-group bias gradient: fp32 sums of
each 128-row block of the bf16 gradient, added in block order and rounded once to the bias dtype.

Reference: the kernels' sigmoid (Triton's 1 / (1 + ex2.approx(...))) cannot be reproduced bit for
bit in PyTorch. So the FP8 bytes and scale values must equal exactly a PyTorch quantization of the
plain kernels' bf16 output on the PyTorch-biased input, and those bf16 activations must lie within
one bf16 rounding (plus fp32 noise) of a float64 evaluation of the formula. The bias add is
compared exactly. The bias gradient must lie within one bf16 rounding of the fp32 block sums, plus
fp32 reassociation inside a block, whose summation order is the kernel's.

Recipe: per 128-element block amax = max |x| (NaN ignored),
scale = 2^ceil(log2(max(amax, 1e-4) / 448)) computed as the kernels do (an fp32 product with
1 / 448, then a power-of-two ceil by bit arithmetic), and q = e4m3_satfinite(x * (1 / scale))
rounding to nearest even.

Known SM100 difference: in a block whose amax is infinite the SM90 E8M0 arithmetic gives scale inf
(reciprocal -inf, bytes -/+448), SM100's native cvt.rp.satfinite.ue8m0x2 saturates the scale to
2^127 (bytes +/-0). Inputs with inf mean training has already diverged, so either is accepted,
element by element, in such blocks only; everywhere else the bytes and scales must match exactly.
"""

import pytest
import torch

from pithtrain.kernels import (
    clamped_swiglu_bwd,
    clamped_swiglu_fwd,
    fp8_clamped_swiglu_bwd,
    fp8_clamped_swiglu_kmajor,
    fp8_clamped_swiglu_rowwise,
    fp8_group_indices,
    fp8_quant_grouped,
    k_grouped_fp8_gemm,
)

BLOCK = 128
ALPHA, LIMIT = 1.702, 7.0


def e8m0_scale(amax: torch.Tensor, native: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The recipe's power-of-two scale of an fp32 amax, and its exact reciprocal. native saturates the
    scale of an infinite amax to 2^127 as SM100's cvt does (the SM90 arithmetic gives inf).
    """
    amax_floor = torch.maximum(amax, torch.full_like(amax, 1e-4))
    raw_scale = amax_floor * torch.full_like(amax_floor, 1.0 / 448.0)
    bits = raw_scale.view(torch.int32)
    # round up to a power of two: keep the exponent, bump it when any mantissa bit is set
    bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    scale = torch.maximum(bits.view(torch.float32), torch.full_like(amax_floor, 2.0**-126))
    if native:
        scale = torch.minimum(scale, torch.full_like(scale, 2.0**127))
    rcp = ((254 << 23) - scale.view(torch.int32)).view(torch.float32)
    return scale, rcp


def nan_free_abs(x: torch.Tensor) -> torch.Tensor:
    x_abs = x.abs()
    return torch.where(torch.isnan(x_abs), torch.zeros_like(x_abs), x_abs)


def quantize_e4m3(x: torch.Tensor, rcp: torch.Tensor) -> torch.Tensor:
    return (x * rcp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def quantize_ref(x: torch.Tensor, native: bool) -> tuple[torch.Tensor, ...]:
    M, K = x.shape
    nb = -(-K // BLOCK)
    xf = torch.zeros(M, nb * BLOCK, dtype=torch.float32, device=x.device)
    xf[:, :K] = x.float()
    s_row, r_row = e8m0_scale(nan_free_abs(xf).view(M, nb, BLOCK).amax(-1), native)
    q_row = quantize_e4m3(xf.view(M, nb, BLOCK), r_row[..., None]).view(M, nb * BLOCK)[:, :K]
    xc = xf[:, :K].reshape(M // BLOCK, BLOCK, K)
    s_ch, r_ch = e8m0_scale(nan_free_abs(xc).amax(1), native)
    q_ch = quantize_e4m3(xc, r_ch[:, None, :]).view(M, K)
    return q_row.contiguous(), s_row, q_ch.contiguous(), s_ch


def quantize_grouped_ref(x: torch.Tensor, offs: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], ...]:  # fmt: skip
    """
    fp8_quant_grouped in PyTorch, in logical form: x_fp8 (M, K) as uint8, x_scale (M, ceil(K /
    128)), x_ch (M, K) uint8 (128x1 blocks), x_ch_scale (M / 128, K). Each entry is a pair (the SM90
    E8M0 arithmetic, the SM100-native variant), which differ only in blocks whose amax is infinite.
    """
    return tuple(zip(quantize_ref(x, False), quantize_ref(x, True)))


def bytes_match(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    A NaN may carry either sign.
    """
    got = got.view(torch.uint8)
    nan = (ref & 0x7F) == 0x7F
    return (got == ref) | (nan & ((got & 0x7F) == 0x7F))


def assert_quant_equal(got: tuple, ref: tuple, names: tuple[str, ...]) -> None:
    """
    ref holds a pair per output (see quantize_grouped_ref); either is accepted element by element.
    """
    assert len(got) == len(ref) == len(names)
    for name, a, (b, b_alt) in zip(names, got, ref):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        if name.endswith("fp8"):
            assert a.dtype == torch.float8_e4m3fn, name
            ok = bytes_match(a, b) | bytes_match(a, b_alt)
        else:
            assert a.dtype == torch.float32, name
            ok = (a == b) | (a == b_alt)
        if not bool(ok.all()):
            bad = (~ok).nonzero()
            raise AssertionError(f"{name}: {bad.shape[0]} mismatches, first at {bad[:4].tolist()}")


ROW = ("x_fp8", "x_scale")
KMAJOR = ("x_ch_fp8", "x_ch_scale")


def padded_channels(c: int) -> int:
    return -(-c // 256) * 256


def assert_kmajor(got: tuple, ref: tuple, offs: torch.Tensor, K: int, padded: bool = False) -> None:
    """
    Assert the per-channel pair got equals ref exactly. In the K-major layout (flat data) it is laid
    out with K channels (or, when padded, K or padded_channels(K) as the scale's row count says) and
    compared in its first K channels, the extra ones unspecified; otherwise it is (M, K) with scale
    (M / 128, K) whether padded or not.
    """
    fp8, scale = got
    if fp8.ndim == 2:
        assert fp8.shape[1] == K and scale.shape == (fp8.shape[0] // BLOCK, K), (fp8.shape, scale.shape)  # fmt: skip
        assert_quant_equal(got, ref, KMAJOR)
        return
    KP = scale.shape[0] if scale.ndim == 2 else K
    assert KP in ((K, padded_channels(K)) if padded else (K,)), (scale.shape, K, padded)
    M = scale.shape[1] * BLOCK if scale.ndim == 2 else 0
    assert fp8.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert fp8.numel() == M * KP, (fp8.shape, scale.shape, KP)
    raw = fp8.view(torch.uint8)
    real = torch.empty(M, K, dtype=torch.uint8, device=raw.device)
    start = 0
    for end in offs.tolist():
        blk = raw[start * KP : end * KP].view(KP, end - start)
        real[start:end].copy_(blk[:K].T)
        start = end
    assert_quant_equal((real.view(torch.float8_e4m3fn), scale[:K].T), ref, KMAJOR)


def group_ids(offs: torch.Tensor, M: int) -> torch.Tensor:
    ids = torch.arange(M, device=offs.device)
    return torch.searchsorted(offs.long(), ids, right=True).clamp_(max=offs.shape[0] - 1)


def biased_ref(gate_up: torch.Tensor, bias: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return (gate_up.float() + bias.float()[group_ids(offs, gate_up.shape[0])]).to(torch.bfloat16)


def assert_bf16_close(got: torch.Tensor, ref64: torch.Tensor, mag: torch.Tensor) -> None:
    """
    Assert the bf16 got is the bf16 rounding of a value fp32-close to the float64 ref64.
    """
    got64 = got.double()
    assert bool(torch.isfinite(ref64).all()) and bool(torch.isfinite(got64).all())
    err = (got64 - ref64).abs()
    tol = 2.0**-8 * ref64.abs() + 2.0**-20 * mag + 1e-30
    if not bool((err <= tol).all()):
        worst = (err - tol).argmax()
        raise AssertionError(f"activation off: got {got64.flatten()[worst]}, want {ref64.flatten()[worst]}")  # fmt: skip


def to_f32(v: float) -> float:
    return torch.tensor(v, dtype=torch.float32).item()


def check_fwd_activation(x: torch.Tensor, h: torch.Tensor) -> None:
    alpha, limit = to_f32(ALPHA), to_f32(LIMIT)
    gate, up = x[:, 0::2].double(), x[:, 1::2].double()
    gc, uc = gate.clamp(max=limit), up.clamp(-limit, limit)
    glu = gc * torch.sigmoid(alpha * gc)
    assert_bf16_close(h, (uc + 1) * glu, (uc.abs() + 1) * glu.abs())


def check_bwd_activation(dh: torch.Tensor, x: torch.Tensor, dx: torch.Tensor) -> None:
    alpha, limit = to_f32(ALPHA), to_f32(LIMIT)
    dh64, gate, up = dh.double(), x[:, 0::2].double(), x[:, 1::2].double()
    gc, uc = gate.clamp(max=limit), up.clamp(-limit, limit)
    sig = torch.sigmoid(alpha * gc)
    glu = gc * sig
    front = dh64 * (uc + 1) * sig
    dg = torch.where(gate <= limit, front * (1 + alpha * gc * (1 - sig)), 0.0)
    du = torch.where((up >= -limit) & (up <= limit), dh64 * glu, 0.0)
    assert_bf16_close(dx[:, 0::2], dg, (dh64 * (uc.abs() + 1) * sig).abs() * (1 + abs(alpha) * gc.abs()))  # fmt: skip
    assert_bf16_close(dx[:, 1::2], du, (dh64 * glu).abs())


def bias_grad_blocks(dgu: torch.Tensor, offs: torch.Tensor, G: int) -> tuple:
    """
    fp32 per-group sums of the 128-row block sums in block order, and the per-group sums of |dgu|
    (the reassociation bound).
    """
    M, C = dgu.shape
    blocks = dgu.float().view(M // BLOCK, BLOCK, C).sum(1)
    abs_blocks = dgu.float().abs().view(M // BLOCK, BLOCK, C).sum(1)
    expected = torch.zeros(G, C, device=dgu.device)
    abs_sum = torch.zeros(G, C, device=dgu.device)
    start = 0
    for group_idx, end in enumerate((offs // BLOCK).tolist()):
        for block_idx in range(start, end):
            expected[group_idx] += blocks[block_idx]
        abs_sum[group_idx] = abs_blocks[start:end].sum(0)
        start = end
    return expected, abs_sum


def check_bias_grad(got: torch.Tensor, dgu: torch.Tensor, offs: torch.Tensor, bias) -> None:
    """
    Assert the bf16 bias gradient lies within one bf16 rounding of the fp32 block-order sums, plus
    fp32 reassociation inside a block, and is exactly zero for empty groups.
    """
    assert got.dtype == bias.dtype and got.shape == bias.shape
    expected, abs_sum = bias_grad_blocks(dgu, offs, bias.shape[0])
    err = (got.double() - expected.double()).abs()
    tol = 2.0**-8 * expected.double().abs() + 2.0**-16 * abs_sum.double()
    assert bool((err <= tol).all()), f"bias grad off by {(err - tol).max().item()}"
    empty = abs_sum == 0
    assert bool((got[empty] == 0).all())


def make_offs(M: int, G: int, gen: torch.Generator) -> torch.Tensor:
    blocks = M // BLOCK
    cuts = torch.sort(torch.randint(0, blocks + 1, (G - 1,), generator=gen)).values
    ends = torch.cat([cuts, torch.tensor([blocks])]) * BLOCK
    return ends.to(torch.int32).cuda()


def make_inputs(M: int, N: int, G: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    gu = torch.randn(M, 2 * N, generator=gen) * 4
    gu[torch.rand(M, 2 * N, generator=gen) < 0.01] *= 100  # outliers past the clamp
    gu[torch.rand(M, 2 * N, generator=gen) < 0.01] = 0
    if M >= 256:
        gu[128:256] = 0
    gu[:, 0:16:2] = -gu[:, 0:16:2].abs()  # negative gates
    bias = torch.randn(G, 2 * N, generator=gen)
    dh = torch.randn(M, N, generator=gen)
    dh[torch.rand(M, N, generator=gen) < 0.01] *= 1000
    bf = dict(dtype=torch.bfloat16, device="cuda")
    return gu.to(**bf), bias.to(**bf), dh.to(**bf), make_offs(M, G, gen)


# (M, N, G): gpt-oss-20b's 2880 (hidden = intermediate), small / ragged N, 32 experts.
SHAPES = [
    (128, 2880, 1),
    (1024, 2880, 4),
    (4096, 2880, 32),
    (2048, 2880, 16),
    (512, 64, 3),
    (640, 136, 5),  # N and 2N not multiples of 128
    (256, 8, 2),
    (384, 100, 3),  # N % 8 != 0: unfused fallback
]


@pytest.mark.parametrize("M,N,G", SHAPES)
def test_quant_rowwise(M: int, N: int, G: int) -> None:
    gu, bias, _, offs = make_inputs(M, N, G, seed=M + N + G)
    x = biased_ref(gu, bias, offs)
    h = clamped_swiglu_fwd(x, ALPHA, LIMIT)
    check_fwd_activation(x, h)
    ref = quantize_grouped_ref(h, offs)
    got = fp8_clamped_swiglu_rowwise(gu, bias, offs, ALPHA, LIMIT)
    assert_quant_equal(got, ref[:2], ROW)


KMAJOR_CASES = [pytest.param(*shape, padded) for shape in SHAPES for padded in (False, True)]


@pytest.mark.parametrize("M,N,G,padded", KMAJOR_CASES)
def test_quant_kmajor(M: int, N: int, G: int, padded: bool) -> None:
    gu, bias, _, offs = make_inputs(M, N, G, seed=3 * M + N + G)
    h = clamped_swiglu_fwd(biased_ref(gu, bias, offs), ALPHA, LIMIT)
    ref = quantize_grouped_ref(h, offs)
    got = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=padded)
    assert_kmajor(got, ref[2:], offs, N, padded)


@pytest.mark.parametrize("M,N,G,padded", KMAJOR_CASES)
def test_bwd_quant(M: int, N: int, G: int, padded: bool) -> None:
    gu, bias, dh, offs = make_inputs(M, N, G, seed=7 * M + N + G)
    x = biased_ref(gu, bias, offs)
    dgu = clamped_swiglu_bwd(dh, x, ALPHA, LIMIT)
    check_bwd_activation(dh, x, dgu)
    ref = quantize_grouped_ref(dgu, offs)
    got, d_bias = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT, padded=padded)
    assert_quant_equal(got[:2], ref[:2], ROW)
    assert_kmajor(got[2:], ref[2:], offs, 2 * N, padded)
    check_bias_grad(d_bias, dgu, offs, bias)


@pytest.mark.parametrize("backward", [False, True])
def test_padded_layout_feeds_k_grouped_gemm(backward: bool) -> None:
    """
    With padded=True the fused kernels lay out padded_channels(C) channels, the extra ones
    unwritten (the allocator is dirtied first); k_grouped_fp8_gemm drops them, so the weight
    gradient equals the one from the unpadded layout.
    """
    M, N, G = 1024, 2880, 4
    gu, bias, dh, offs = make_inputs(M, N, G, seed=17)
    torch.cuda.empty_cache()
    torch.empty(64 << 20, dtype=torch.uint8, device="cuda").fill_(0x7F)  # FP8 NaN bytes
    if backward:
        C = 2 * N
        pad = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT, padded=True)[0][2:]
        flat = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)[0][2:]
    else:
        C = N
        pad = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=True)
        flat = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT)
    if pad[0].ndim == 1:  # K-major: the fused SM90 kernels pad
        assert pad[1].shape[0] == padded_channels(C) and flat[1].shape[0] == C
    else:  # padded is ignored
        assert pad[1].shape == flat[1].shape == (M // BLOCK, C)
    other = torch.randn(M, 2048, device="cuda").to(torch.bfloat16)
    b = fp8_quant_grouped(other, offs)[2:]
    ks = [int(size) for size in torch.diff(offs, prepend=offs.new_zeros(1)).tolist()]
    kt = torch.tensor(ks, dtype=torch.int32, device="cuda")
    got = torch.zeros(G, C, 2048, device="cuda")
    want = torch.zeros(G, C, 2048, device="cuda")
    k_grouped_fp8_gemm(pad, b, got, ks, kt)
    k_grouped_fp8_gemm(flat, b, want, ks, kt)
    assert bool(torch.isfinite(got).all())
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backward", [False, True])
def test_every_bf16_gate_and_up(backward: bool) -> None:
    """
    Every BF16 bit pattern (inf and NaN included) as gate and, shuffled, as up, with a random bias;
    the block amax ignores NaN.
    """
    M, N, G = 512, 128, 2
    gen = torch.Generator().manual_seed(3)
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.bfloat16)
    up = bits[torch.randperm(65536, generator=gen)].view(torch.bfloat16)
    gu = torch.stack([gate, up], dim=-1).view(M, 2 * N).cuda()
    bias = (torch.randn(G, 2 * N, generator=gen) * 0.1).to(torch.bfloat16).cuda()
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    x = biased_ref(gu, bias, offs)
    if not backward:
        ref = quantize_grouped_ref(clamped_swiglu_fwd(x, ALPHA, LIMIT), offs)
        assert_quant_equal(fp8_clamped_swiglu_rowwise(gu, bias, offs, ALPHA, LIMIT), ref[:2], ROW)
        for padded in (False, True):
            got = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=padded)
            assert_kmajor(got, ref[2:], offs, N, padded)
        return
    dh = torch.randn(M, N, generator=gen).to(torch.bfloat16).cuda()
    dgu = clamped_swiglu_bwd(dh, x, ALPHA, LIMIT)
    out, _ = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
    ref = quantize_grouped_ref(dgu, offs)
    assert_quant_equal(out[:2], ref[:2], ROW)
    assert_kmajor(out[2:], ref[2:], offs, 2 * N)


def test_zeros() -> None:
    """
    Zero inputs and a zero bias give zero bytes, the scale floor 2^ceil(log2(1e-4 / 448)) and a
    zero bias gradient, from every entry point.
    """
    M, N, G = 384, 2880, 3
    gu = torch.zeros(M, 2 * N, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(G, 2 * N, dtype=torch.bfloat16, device="cuda")
    dh = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([128, 128, 384], dtype=torch.int32, device="cuda")
    floor = e8m0_scale(torch.zeros(1, device="cuda"))[0].item()
    row = fp8_clamped_swiglu_rowwise(gu, bias, offs, ALPHA, LIMIT)
    km = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT)
    (q, d_bias) = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
    for fp8, scale in (row, km, q[:2], q[2:]):
        assert not bool(fp8.view(torch.uint8).any())
        assert bool((scale == floor).all())
    assert not bool(d_bias.any())


def test_empty_input() -> None:
    """
    M == 0 (no tokens routed to this rank): correctly shaped empty outputs, zero bias grad.
    """
    N, G = 2880, 4
    gu = torch.empty(0, 2 * N, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(G, 2 * N, device="cuda").to(torch.bfloat16)
    dh = torch.empty(0, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.zeros(G, dtype=torch.int32, device="cuda")
    q, s = fp8_clamped_swiglu_rowwise(gu, bias, offs, ALPHA, LIMIT)
    assert q.shape == (0, N) and s.shape == (0, -(-N // BLOCK))
    km, kms = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT)
    assert km.numel() == 0 and kms.shape in ((N, 0), (0, N))
    out, d_bias = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
    assert out[0].shape == (0, 2 * N) and out[2].numel() == 0
    assert out[3].shape in ((2 * N, 0), (0, 2 * N))
    assert d_bias.shape == bias.shape and d_bias.dtype == bias.dtype and not bool(d_bias.any())


@pytest.mark.parametrize("N", [2880, 136])
def test_inf_nan_blocks(N: int) -> None:
    """
    inf and NaN in gate_up and dh through every entry point: NaN is ignored by the amax, an infinite
    amax takes either E8M0 result (see the module docstring), every other block is exact.
    """
    M, G = 512, 3
    gu, bias, dh, offs = make_inputs(M, N, G, seed=29)
    inf, nan = float("inf"), float("nan")
    gu[3, 1], gu[130, 4], gu[260, 9], gu[261, 2 * N - 1] = inf, -inf, inf, -inf  # up, gate
    gu[7, 0], gu[300, 3], gu[400, 2 * N - 2] = nan, nan, nan
    dh[11, 5], dh[290, N - 1], dh[450, 0], dh[451, 1] = inf, -inf, nan, nan
    x = biased_ref(gu, bias, offs)
    h = clamped_swiglu_fwd(x, ALPHA, LIMIT)
    ref = quantize_grouped_ref(h, offs)
    assert_quant_equal(fp8_clamped_swiglu_rowwise(gu, bias, offs, ALPHA, LIMIT), ref[:2], ROW)
    for padded in (False, True):
        got = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=padded)
        assert_kmajor(got, ref[2:], offs, N, padded)
    dgu = clamped_swiglu_bwd(dh, x, ALPHA, LIMIT)
    assert not bool(torch.isfinite(dgu).all())
    ref = quantize_grouped_ref(dgu, offs)
    out, _ = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
    assert_quant_equal(out[:2], ref[:2], ROW)
    assert_kmajor(out[2:], ref[2:], offs, 2 * N)


def kmajor_channels_of(pair: tuple) -> int | None:
    """
    The channel count a K-major per-channel pair is laid out for (its scale's row count), or None
    where the pair is (M, C) (no K-major layout, so no unfused fallback to tell apart).
    """
    return pair[1].shape[0] if pair[0].ndim == 1 else None


def check_fwd_activation_fp32(x: torch.Tensor, h: torch.Tensor) -> None:
    alpha, limit = to_f32(ALPHA), to_f32(LIMIT)
    gate, up = x[:, 0::2].double(), x[:, 1::2].double()
    gc, uc = gate.clamp(max=limit), up.clamp(-limit, limit)
    want = (uc + 1) * gc * torch.sigmoid(alpha * gc)
    torch.testing.assert_close(h.double(), want, rtol=1e-5, atol=1e-6 * want.abs().max().item())


def ref64_fwd(x: torch.Tensor) -> torch.Tensor:
    """
    float64 clamped SwiGLU of the interleaved x (gate even, up odd), alpha / limit as fp32 values.
    """
    alpha, limit = to_f32(ALPHA), to_f32(LIMIT)
    gate, up = x[:, 0::2].double(), x[:, 1::2].double()
    gc, uc = gate.clamp(max=limit), up.clamp(-limit, limit)
    return (uc + 1) * gc * torch.sigmoid(alpha * gc)


def ref64_bwd(dh: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    float64 gradient of ref64_fwd with respect to x, interleaved like x; zero where the clamp saturates.
    """
    alpha, limit = to_f32(ALPHA), to_f32(LIMIT)
    dh64, gate, up = dh.double(), x[:, 0::2].double(), x[:, 1::2].double()
    gc, uc = gate.clamp(max=limit), up.clamp(-limit, limit)
    sig = torch.sigmoid(alpha * gc)
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    out[:, 0::2] = torch.where(
        gate <= limit, dh64 * (uc + 1) * sig * (1 + alpha * gc * (1 - sig)), 0.0
    )
    out[:, 1::2] = torch.where((up >= -limit) & (up <= limit), dh64 * gc * sig, 0.0)
    return out


STABLE_DELTA = (
    2.0**-14
)  # relative; far above the kernels' fp32 error (~1e-6), far below bf16's 2^-8 spacing


def round_as(v64: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    v64 rounded as the fallback stores the activation: to fp32, then to bf16 when the path is bf16.
    """
    v32 = v64.float()
    return v32.to(dtype) if dtype == torch.bfloat16 else v32


def unstable(v64: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Elements of the float64 activation whose stored value or FP8 encoding could depend on the kernels'
    fp32 error: those whose rounding to bf16 (when dtype is bf16), e4m3 byte (rowwise or per-channel) or
    block scale changes
    under a relative perturbation of STABLE_DELTA (for a changed scale, the element setting the block's
    amax), and nonzero values small enough for the kernels' flush-to-zero sigmoid to matter.
    """
    M, K = v64.shape
    mid = round_as(v64, dtype)
    ref = quantize_ref(mid, False)
    bad = (v64 != 0) & (v64.abs() < 2.0**-60)
    for factor in (1 - STABLE_DELTA, 1 + STABLE_DELTA):
        alt = round_as(v64 * factor, dtype)
        q = quantize_ref(alt, False)
        if (
            dtype == torch.bfloat16
        ):  # an fp32 activation is compared to tolerance, only its FP8 must be stable
            bad |= alt != mid
        bad |= (q[0] != ref[0]) | (q[2] != ref[2])
        pad = -(-K // BLOCK) * BLOCK - K
        mag = torch.nn.functional.pad(mid.float().abs(), (0, pad)).view(M, -1, BLOCK)
        rows, blocks = (q[1] != ref[1]).nonzero(as_tuple=True)
        cols = blocks * BLOCK + mag[rows, blocks].argmax(-1)
        bad[rows, cols.clamp(max=K - 1)] = True
        mag_ch = mid.float().abs().view(M // BLOCK, BLOCK, K)
        row_blocks, chans = (q[3] != ref[3]).nonzero(as_tuple=True)
        bad[row_blocks * BLOCK + mag_ch[row_blocks, :, chans].argmax(-1), chans] = True
    return bad


def promoted_input(gu: torch.Tensor, bias: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """
    gate_up + bias[group] in the promoted dtype, as the fallback adds it: one fp32 add, rounded to bf16
    when both are bf16.
    """
    x = gu.float() + bias.float()[group_ids(offs, gu.shape[0])]
    return x.to(torch.bfloat16) if bias.dtype == torch.bfloat16 else x


def stable_inputs(gu, bias, dh, offs, gen: torch.Generator, max_rounds: int = 30):
    """
    gu and dh with every (gate, up, dh) triple resampled (randn * 4, randn) until no element of the
    forward activation or the backward gradient is unstable, so the exact expectations built from
    float64 PyTorch below cannot depend on the kernels' approximate sigmoid.
    """
    gu, dh = gu.clone(), dh.clone()
    for _ in range(max_rounds):
        x = promoted_input(gu, bias, offs)
        bad = unstable(ref64_fwd(x), x.dtype)
        bad_bwd = unstable(ref64_bwd(dh.to(x.dtype), x), x.dtype)
        bad |= bad_bwd[:, 0::2] | bad_bwd[:, 1::2]
        rows, cols = bad.nonzero(as_tuple=True)
        if rows.numel() == 0:
            return gu, dh
        n = rows.numel()
        fresh = (torch.randn(n, 2, generator=gen) * 4).to(gu.dtype).to(gu.device)
        gu[rows, 2 * cols], gu[rows, 2 * cols + 1] = fresh[:, 0], fresh[:, 1]
        dh[rows, cols] = torch.randn(n, generator=gen).to(dh.dtype).to(dh.device)
    raise AssertionError("could not make the inputs stable")


def bias_grad_ref(
    dgu: torch.Tensor, offs: torch.Tensor, G: int, dtype: torch.dtype
) -> torch.Tensor:
    """
    The bias gradient recipe: fp32 sums of each 128-row block of the stored gradient, added from zero
    in block order per group, rounded once to dtype.
    """
    M, C = dgu.shape
    blocks = dgu.float().view(M // BLOCK, BLOCK, C).sum(1)
    out, start = torch.zeros(G, C, dtype=torch.float32, device=dgu.device), 0
    for group_idx, end in enumerate((offs // BLOCK).tolist()):
        for block_idx in range(start, end):
            out[group_idx] = out[group_idx] + blocks[block_idx]
        start = end
    return out.to(dtype)


NONFUSABLE = ["n_not_multiple_of_8", "fp32_bias", "int64_offsets", "strided_dh", "fp32_dh"]


@pytest.mark.parametrize("case", NONFUSABLE)
def test_nonfusable_fallback(case: str) -> None:
    """
    Inputs the fused SM90 kernels do not cover take the unfused sequence (bias add in the promoted
    dtype, the clamped_swiglu kernels, fp8_quant_grouped). With padded=True its K-major pair keeps
    C channels where the fused kernel would pad to padded_channels(C), which shows the fallback ran.

    Expected values come from PyTorch only: the float64 activation and gradient of the promoted input
    (bias add, clamp masks, dh cast as autograd does), rounded to the stored dtype, then the exact FP8
    recipe (quantize_grouped_ref) and the bias gradient recipe. The inputs are made stable first
    (stable_inputs), so the kernels' approximate sigmoid cannot move any stored value, e4m3 byte or
    scale: the bytes and scales must match exactly, and the bias gradient exactly where it is bf16
    (to fp32 tolerance for the fp32 bias, whose stored gradient is fp32).
    """
    M, G = 384, 3
    N = 100 if case == "n_not_multiple_of_8" else 200
    gu, bias, dh, offs = make_inputs(M, N, G, seed=41)
    case_bias = bias.float() if case == "fp32_bias" else bias
    case_offs = offs.long() if case == "int64_offsets" else offs
    gu, dh = stable_inputs(gu, case_bias, dh, offs, torch.Generator().manual_seed(43))
    probe = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=True)
    if kmajor_channels_of(probe) is None:
        pytest.skip("only the K-major layout has an unfused fallback to tell apart")
    if case != "n_not_multiple_of_8":
        assert kmajor_channels_of(probe) == padded_channels(N)  # these inputs alone run fused
    bias, offs = case_bias, case_offs
    dh_arg = dh
    if case == "strided_dh":
        dh_arg = torch.cat([dh, dh[:, :8]], dim=1)[:, :N]
        assert not dh_arg.is_contiguous()
    if case == "fp32_dh":
        dh_arg = dh.float()
    x = promoted_input(gu, bias, offs)
    dh_x = dh_arg.to(x.dtype)  # as autograd casts it
    h_ref, dgu_ref = round_as(ref64_fwd(x), x.dtype), round_as(ref64_bwd(dh_x, x), x.dtype)
    if case in ("n_not_multiple_of_8", "fp32_bias", "int64_offsets"):
        ref = quantize_grouped_ref(h_ref, offs)
        assert_quant_equal(fp8_clamped_swiglu_rowwise(gu, bias, offs, ALPHA, LIMIT), ref[:2], ROW)
        got = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=True)
        assert kmajor_channels_of(got) == N
        assert_kmajor(got, ref[2:], offs, N, padded=True)
        # Diagnostic: the plain kernel's activation against the float64 formula.
        h = clamped_swiglu_fwd(x, ALPHA, LIMIT)
        if x.dtype == torch.bfloat16:
            check_fwd_activation(x, h)
        else:
            check_fwd_activation_fp32(x, h)
    got, d_bias = fp8_clamped_swiglu_bwd(dh_arg, gu, bias, offs, ALPHA, LIMIT, padded=True)
    assert kmajor_channels_of(got[2:]) == 2 * N
    ref = quantize_grouped_ref(dgu_ref, offs)
    assert_quant_equal(got[:2], ref[:2], ROW)
    assert_kmajor(got[2:], ref[2:], offs, 2 * N, padded=True)
    want_bias = bias_grad_ref(dgu_ref, offs, G, bias.dtype)
    assert d_bias.dtype == bias.dtype and d_bias.shape == bias.shape
    if bias.dtype == torch.bfloat16:
        assert torch.equal(d_bias, want_bias), "bias gradient differs from the recipe"
    else:
        scale = dgu_ref.abs().float().view(M // BLOCK, BLOCK, -1).sum(1).max().item()
        torch.testing.assert_close(d_bias, want_bias, rtol=1e-5, atol=1e-6 * scale)
    # Diagnostic: the plain kernel's gradient against the float64 formula.
    dgu = clamped_swiglu_bwd(dh_x.contiguous(), x, ALPHA, LIMIT)
    if x.dtype == torch.bfloat16:
        check_bwd_activation(dh_x, x, dgu)
    else:
        torch.testing.assert_close(
            dgu.double(), ref64_bwd(dh_x, x), rtol=1e-5, atol=1e-6 * dgu.abs().max().item()
        )


# Bias gradient (block_segment_sum's fold of the 128-row block partials). The fold order is fixed:
# each group's blocks are added in row order from zero, so check_bias_grad needs no fold term. On
# the SM90 unfused fallback the partials are the plain PyTorch block sums of the bf16 gradient, so
# there the gradient is checked bit for bit; fp8_group_indices' documented SM90-only hint (None on
# SM100) marks that device, independently of the kernel under test.
SM90 = fp8_group_indices(torch.tensor([BLOCK], dtype=torch.int32, device="cuda"), BLOCK) is not None


def block_fold(dgu: torch.Tensor, offs: torch.Tensor, dtype, reverse: bool = False) -> torch.Tensor:
    """
    Each group's fp32 block sums of dgu added one at a time from zero, in increasing block order
    (or decreasing with reverse), rounded once to dtype.
    """
    M, C = dgu.shape
    blocks = dgu.float().view(M // BLOCK, BLOCK, C).sum(1)
    out, start = [], 0
    for end in (offs // BLOCK).tolist():
        acc = torch.zeros(C, device=dgu.device)
        for block_idx in range(end - 1, start - 1, -1) if reverse else range(start, end):
            acc = acc + blocks[block_idx]
        out.append(acc)
        start = end
    return torch.stack(out).to(dtype)


def inputs_for_blocks(block_counts: list, N: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    M = BLOCK * sum(block_counts)
    gu = torch.randn(M, 2 * N, generator=gen) * 4
    bias = torch.randn(len(block_counts), 2 * N, generator=gen)
    dh = torch.randn(M, N, generator=gen)
    offs = (torch.tensor(block_counts).cumsum(0) * BLOCK).to(torch.int32).cuda()
    bf = dict(dtype=torch.bfloat16, device="cuda")
    return gu.to(**bf), bias.to(**bf), dh.to(**bf), offs


def skewed_blocks(G: int, num_blocks: int, seed: int) -> list:
    gen = torch.Generator().manual_seed(seed)
    weights = torch.exp(0.3 * torch.randn(G, generator=gen))
    counts = torch.floor(weights / weights.sum() * num_blocks).long()
    counts[: num_blocks - int(counts.sum())] += 1
    return counts.tolist()


# (block counts per group, N): gpt-oss contract M / G (balanced and lognormal(0.3) skew; N = 2880 runs
# fused, N = 100 the unfused fallback), ragged with empty groups, N not a multiple of 128 or 8, and one
# group of 545 blocks (a 70000-row group padded to 128).
BIAS_CASES = [
    ([16] * 8, 2880),
    (skewed_blocks(4, 259, 1), 2880),
    (skewed_blocks(16, 138, 2), 2880),
    ([0, 3, 0, 1, 40, 0, 2], 2880),
    ([0, 3, 0, 1, 40, 0, 2], 100),
    ([5, 0, 7], 136),
    ([1], 2880),
    ([2, 545, 0, 1], 2880),
    ([2, 545, 0, 1], 100),
    (skewed_blocks(64, 300, 3), 100),
]


@pytest.mark.parametrize("block_counts,N", BIAS_CASES)
def test_bias_grad(block_counts: list, N: int) -> None:
    gu, bias, dh, offs = inputs_for_blocks(block_counts, N, seed=sum(block_counts) + N)
    dgu = clamped_swiglu_bwd(dh, biased_ref(gu, bias, offs), ALPHA, LIMIT)
    _, d_bias = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
    check_bias_grad(d_bias, dgu, offs, bias)
    if SM90 and N % 8 != 0:
        # the fallback's partials are dgu's PyTorch block sums: the fold is exact, and its order shows
        assert torch.equal(d_bias, block_fold(dgu, offs, bias.dtype))


@pytest.mark.parametrize("N", [2880, 100])
def test_bias_grad_block_order(N: int) -> None:
    """
    Group 1's blocks are B0, B1 = -B0 exactly (the same rows with dh negated, scaled by 2^16) and a
    small B2: the fold in row order from zero gives B2's sum, any other order loses it in the
    rounding of the large terms. Fused (N = 2880) and fallback (N = 100) paths alike.
    """
    gu, bias, dh, offs = inputs_for_blocks([2, 3, 1], N, seed=N + 5)
    rows = slice(2 * BLOCK, 3 * BLOCK)
    gu[3 * BLOCK : 4 * BLOCK] = gu[rows]
    dh[rows] = dh[rows] * 2.0**16
    dh[3 * BLOCK : 4 * BLOCK] = -dh[rows]
    dgu = clamped_swiglu_bwd(dh, biased_ref(gu, bias, offs), ALPHA, LIMIT)
    _, d_bias = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
    small = dgu[4 * BLOCK : 5 * BLOCK].float()
    want = small.sum(0)
    tol = 2.0**-8 * want.abs() + 2.0**-16 * small.abs().sum(0)
    assert bool(((d_bias[1].float() - want).abs() <= tol).all())
    big = dgu[rows].float().abs().sum(0)
    assert bool((big > 2.0**12 * small.abs().sum(0)).any()), "the large blocks must dominate"


@pytest.mark.parametrize("N", [2880, 100])
def test_bias_grad_side_stream(N: int) -> None:
    """
    The backward runs on the caller's current stream: inputs written on a side stream behind a sleep
    give the right bias gradient when the call is made on that stream.
    """
    gu_src, bias_src, dh_src, offs = inputs_for_blocks([3, 0, 9, 4], N, seed=N)
    gu, bias, dh = torch.zeros_like(gu_src), torch.zeros_like(bias_src), torch.zeros_like(dh_src)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
        gu.copy_(gu_src), bias.copy_(bias_src), dh.copy_(dh_src)
        _, d_bias = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
    stream.synchronize()
    dgu = clamped_swiglu_bwd(dh_src, biased_ref(gu_src, bias_src, offs), ALPHA, LIMIT)
    check_bias_grad(d_bias, dgu, offs, bias_src)
