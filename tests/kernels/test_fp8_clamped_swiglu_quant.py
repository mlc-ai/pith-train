"""
Contract: gpt-oss's clamped SwiGLU of gate_up + bias[group] (gate at even and up at odd columns; the
bf16 bias added in fp32 and rounded to bf16), rounded to bf16 as clamped_swiglu_fwd /
clamped_swiglu_bwd round it, then quantized as fp8_quant_grouped does on SM90: rowwise 1x128
plus per-channel FP8 K-major per group. With padded=True the K-major pair has ceil(C / 256) * 256
channels (C = N forward, 2N backward) where the fused kernel runs and C on the unfused fallback; the
extra channels are unwritten. The backward also returns the per-group bias gradient: fp32 sums of
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
"""

import pytest
import torch

from pithtrain.kernels import (
    clamped_swiglu_bwd,
    clamped_swiglu_fwd,
    fp8_clamped_swiglu_bwd,
    fp8_clamped_swiglu_kmajor,
    fp8_clamped_swiglu_rowwise,
    fp8_fused_experts_supported,
    fp8_quant_grouped,
    k_grouped_fp8_gemm,
)

pytestmark = pytest.mark.skipif(
    not fp8_fused_experts_supported(),
    reason="needs the fused FP8 expert path (fp8_fused_experts_supported)",
)

BLOCK = 128
ALPHA, LIMIT = 1.702, 7.0


def e8m0_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The recipe's power-of-two scale of an fp32 amax, and its exact reciprocal.
    """
    amax_floor = torch.maximum(amax, torch.full_like(amax, 1e-4))
    raw_scale = amax_floor * torch.full_like(amax_floor, 1.0 / 448.0)
    bits = raw_scale.view(torch.int32)
    # round up to a power of two: keep the exponent, bump it when any mantissa bit is set
    bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    scale = torch.maximum(bits.view(torch.float32), torch.full_like(amax_floor, 2.0**-126))
    rcp = ((254 << 23) - scale.view(torch.int32)).view(torch.float32)
    return scale, rcp


def nan_free_abs(x: torch.Tensor) -> torch.Tensor:
    x_abs = x.abs()
    return torch.where(torch.isnan(x_abs), torch.zeros_like(x_abs), x_abs)


def quantize_e4m3(x: torch.Tensor, rcp: torch.Tensor) -> torch.Tensor:
    return (x * rcp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def quantize_grouped_ref(x: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    fp8_quant_grouped on SM90 in PyTorch: (x_fp8 (M, K) as uint8, x_scale (M, ceil(K / 128)),
    x_ch flat (M * K,) uint8 K-major per group, x_ch_scale (K, M / 128)).
    """
    M, K = x.shape
    nb = -(-K // BLOCK)
    xf = torch.zeros(M, nb * BLOCK, dtype=torch.float32, device=x.device)
    xf[:, :K] = x.float()
    s_row, r_row = e8m0_scale(nan_free_abs(xf).view(M, nb, BLOCK).amax(-1))
    q_row = quantize_e4m3(xf.view(M, nb, BLOCK), r_row[..., None]).view(M, nb * BLOCK)[:, :K]
    xc = xf[:, :K].reshape(M // BLOCK, BLOCK, K)
    s_ch, r_ch = e8m0_scale(nan_free_abs(xc).amax(1))
    q_ch = quantize_e4m3(xc, r_ch[:, None, :]).view(M, K)
    flat = torch.zeros(M * K, dtype=torch.uint8, device=x.device)
    start = 0
    for end in offs.tolist():
        flat[start * K : end * K].view(K, end - start).copy_(q_ch[start:end].T)
        start = end
    return q_row.contiguous(), s_row, flat, s_ch.T.contiguous()


def assert_bytes(got: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    """
    A NaN may carry either sign.
    """
    got = got.view(torch.uint8)
    nan = (ref & 0x7F) == 0x7F
    ok = (got == ref) | (nan & ((got & 0x7F) == 0x7F))
    if not bool(ok.all()):
        bad = (~ok).nonzero()
        raise AssertionError(f"{name}: {bad.shape[0]} byte mismatches, first at {bad[:4].tolist()}")


def assert_quant_equal(got: tuple, ref: tuple, names: tuple[str, ...]) -> None:
    assert len(got) == len(ref) == len(names)
    for name, a, b in zip(names, got, ref):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        if name.endswith("fp8"):
            assert a.dtype == torch.float8_e4m3fn, name
            assert_bytes(a, b, name)
        else:
            assert a.dtype == torch.float32, name
            if not torch.equal(a, b):
                bad = (a != b).nonzero()
                raise AssertionError(f"{name}: {bad.shape[0]} mismatches, first {bad[:4].tolist()}")


ROW = ("x_fp8", "x_scale")
KMAJOR = ("x_ch_fp8", "x_ch_scale")
ALL = ROW + KMAJOR


def padded_channels(c: int) -> int:
    return -(-c // 256) * 256


def assert_kmajor(got: tuple, ref: tuple, offs: torch.Tensor, K: int, padded: bool = False) -> None:
    """
    Assert the K-major pair got, laid out with K channels (or, when padded, K or padded_channels(K)
    as the scale's row count says), equals the unpadded ref exactly in its first K channels. Extra
    channels are unspecified.
    """
    fp8, scale = got
    KP = scale.shape[0] if scale.ndim == 2 else K
    assert KP in ((K, padded_channels(K)) if padded else (K,)), (scale.shape, K, padded)
    M = scale.shape[1] * BLOCK if scale.ndim == 2 else 0
    assert fp8.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert fp8.numel() == M * KP, (fp8.shape, scale.shape, KP)
    raw = fp8.view(torch.uint8)
    real = torch.empty(M * K, dtype=torch.uint8, device=raw.device)
    start = 0
    for end in offs.tolist():
        blk = raw[start * KP : end * KP].view(KP, end - start)
        real[start * K : end * K].view(K, end - start).copy_(blk[:K])
        start = end
    assert_quant_equal((real.view(torch.float8_e4m3fn), scale[:K]), ref, KMAJOR)


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
    assert pad[1].shape[0] == padded_channels(C) and flat[1].shape[0] == C
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
    assert_quant_equal(out, quantize_grouped_ref(dgu, offs), ALL)


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
    assert km.numel() == 0 and kms.shape == (N, 0)
    out, d_bias = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
    assert out[0].shape == (0, 2 * N) and out[2].numel() == 0 and out[3].shape == (2 * N, 0)
    assert d_bias.shape == bias.shape and d_bias.dtype == bias.dtype and not bool(d_bias.any())
