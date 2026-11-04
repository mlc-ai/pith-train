"""
Verifies the six grouped GEMMs of the BF16 CuTe experts (SM90 only) against a float64 PyTorch
reference of the same math.

Forward, per expert e over its grouped rows [cu[e], cu[e + 1]):

    preact = round(x[a_idx] @ [W_gate | W_up]^T (+ bias))     interleaved (gate, up) pairs
    h      = round(act(preact))                              SwiGLU, or gpt-oss clamped SwiGLU
    out[slot_of[r]] = round(h @ W_down^T (+ b_down))[r]       scattered to received-slot order

Backward:

    dh   = dy[slot_of] @ W_down
    dpre = round(dact(preact) * dh)                           h re-emitted bit for bit
    dx   = round(dg @ W_gate + du @ W_up)                     dpre in the weights' address order
    mg_gate, mg_up += dg^T x[a_idx], du^T x[a_idx]            fp32, in place
    mg_down        += dy[slot_of]^T h                          fp32, in place

Every bf16 output is the fp32 accumulation rounded once: it must lie within half a bf16 ulp of the
float64 reference plus 1e-5 of the sum of absolute products. A main grad accumulation from a
nonzero start must lie within 1e-5 of the sum of absolute products plus 2 fp32 ulps of the result.
Gate and up are tested as two weights in both address orders, as the two halves of one fused
[gate; up] weight, and (gpt-oss) as the interleaved rows of one weight; their main grads likewise.
Shapes are those of qwen3-30b-a3b (2048 / 768; 16, 32, 64 local experts), deepseek-v2-lite
(2048 / 1408), qwen3.5 (2048 / 512) and gpt-oss-20b (2880 / 2880, 32 experts), with ragged and
empty groups, more than 65536 rows, and 0 rows.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import (
    bf16_expert_down_dgrad,
    bf16_expert_down_fwd,
    bf16_expert_down_wgrad,
    bf16_expert_gate_up_dgrad,
    bf16_expert_gate_up_fwd,
    bf16_expert_gate_up_wgrad,
    bf16_experts_supported,
    tune_kernels,
)

pytestmark = pytest.mark.skipif(
    not bf16_experts_supported(),
    reason="needs the fused BF16 CuTe experts (bf16_experts_supported)",
)

BF16 = dict(device="cuda", dtype=torch.bfloat16)
F64 = torch.float64


def assert_within(out, ref, absref, what, bf16=True):
    """
    rel is half a bf16 ulp for a bf16 output or 2 fp32 ulps for an fp32 one; absref bounds the
    accumulation-order error.
    """
    rel = 2.0**-8 if bf16 else 2.0**-22
    err = (out.to(F64) - ref).abs()
    tol = rel * ref.abs() + 1e-5 * absref + 1e-30
    worst = (err / tol).max().item() if err.numel() else 0.0
    assert worst <= 1.0, f"{what}: error {worst:.3f}x the tolerance"


def gmm(a, w, counts, trans=True):
    """
    Per group a_g @ w[g]^T (or a_g @ w[g] with trans=False) in float64, and |a_g| @ |w[g]|.
    """
    outs, abs_outs, row_start = [], [], 0
    for group_idx, count in enumerate(counts):
        a_group = a[row_start : row_start + count].to(F64)
        w_group = w[group_idx].to(F64)
        w_group = w_group.T if trans else w_group
        outs.append(a_group @ w_group)
        abs_outs.append(a_group.abs() @ w_group.abs())
        row_start += count
    return torch.cat(outs), torch.cat(abs_outs)


def per_row_bias(bias, counts):
    return torch.cat([bias[group_idx].to(F64).expand(count, -1) for group_idx, count in enumerate(counts)])  # fmt: skip


def kgrouped(a, b, counts):
    """
    Per group a_g^T @ b_g in float64, and |a_g|^T @ |b_g|, each (G, a cols, b cols).
    """
    outs, abs_outs, row_start = [], [], 0
    for count in counts:
        a_group, b_group = a[row_start : row_start + count].to(F64), b[row_start : row_start + count].to(F64)  # fmt: skip
        outs.append(a_group.T @ b_group)
        abs_outs.append(a_group.abs().T @ b_group.abs())
        row_start += count
    return torch.stack(outs), torch.stack(abs_outs)


def swiglu_ref(g, u):
    return F.silu(g) * u


def dswiglu_ref(g, u, dh):
    sig = torch.sigmoid(g)
    return dh * u * sig * (1 + g * (1 - sig)), dh * g * sig


def oai_ref(g, u, alpha, limit):
    gc, uc = g.clamp(max=limit), u.clamp(-limit, limit)
    return gc * torch.sigmoid(alpha * gc) * (uc + 1)


def doai_ref(g, u, dh, alpha, limit):
    gc, uc = g.clamp(max=limit), u.clamp(-limit, limit)
    sig = torch.sigmoid(alpha * gc)
    dg = dh * (uc + 1) * (sig + alpha * gc * sig * (1 - sig)) * (g <= limit)
    du = dh * gc * sig * (u.abs() <= limit)
    return dg, du


def cu_of(counts):
    return torch.tensor([0] + list(counts), device="cuda").cumsum(0).to(torch.int32)


def ragged(G, total, seed, empty=0.0):
    """
    G group sizes summing to about total, some empty, the rest uneven.
    """
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(G, generator=gen) ** 2
    weights[torch.rand(G, generator=gen) < empty] = 0
    weights[0] = max(weights[0].item(), 1e-3)
    return [int(size) for size in (weights / weights.sum() * total).round().long()]


def pair(G, n, k, layout, dtype=torch.bfloat16, scale=1.0):
    """
    (gate, up), each (G, n, k), laid out as layout: separate (two weights, gate lower), swapped
    (two weights, up lower), halves (one fused (G, 2n, k) weight [gate; up]) or interleaved (one
    (G, 2n, k) weight with gate / up rows alternating, gpt-oss).
    """
    if layout in ("separate", "swapped"):
        buf = (torch.randn(2, G, n, k, device="cuda") * scale).to(dtype)
        return (buf[0], buf[1]) if layout == "separate" else (buf[1], buf[0])
    buf = (torch.randn(G, 2 * n, k, device="cuda") * scale).to(dtype)
    if layout == "halves":
        return buf[:, :n], buf[:, n:]
    assert layout == "interleaved"
    return buf[:, 0::2], buf[:, 1::2]


def inputs(counts, H, seed):
    torch.manual_seed(seed)
    M = sum(counts)
    T = max(M // 2, 1)
    x = torch.randn(T, H, **BF16)
    a_idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
    slot_of = torch.randperm(M, device="cuda").to(torch.int32)
    return x, a_idx, slot_of, cu_of(counts)


def lanes(dpre, swapped):
    """
    (dg, du) from dpre, whose pairs are in the weights' address order.
    """
    return (dpre[:, 1::2], dpre[:, 0::2]) if swapped else (dpre[:, 0::2], dpre[:, 1::2])


# name: (H, I, counts)
CASES = {
    "qwen3-g16-ragged": (2048, 768, ragged(16, 16384, 0, empty=0.2)),
    "qwen3-g32-balanced": (2048, 768, [256] * 32),
    "qwen3-g64-ragged": (2048, 768, ragged(64, 20000, 1, empty=0.3)),
    "qwen3-edges": (2048, 768, [0, 1, 300, 129, 0, 1000, 7, 512, 1, 0, 2048, 33]),
    "qwen3-imbalanced": (2048, 768, [6000, 3, 0, 120, 1, 250, 0, 9]),
    "qwen3-large": (2048, 768, [9000] * 8),  # more than 65536 rows in one call
    "dsv2-g8-ragged": (2048, 1408, ragged(8, 12288, 2, empty=0.2)),
    "dsv2-g16": (2048, 1408, [768] * 15 + [0]),
    "q35-g32": (2048, 512, [512] * 32),
    "q35-g64-ragged": (2048, 512, ragged(64, 16384, 3, empty=0.2)),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("layout", ["separate", "swapped", "halves"])
def test_swiglu_forward_backward(case, layout):
    H, inter, counts = CASES[case]
    G, M = len(counts), sum(counts)
    x, a_idx, slot_of, cu = inputs(counts, H, seed=list(CASES).index(case))
    wg, wu = pair(G, inter, H, layout, scale=H**-0.5)
    wd = (torch.randn(G, H, inter, device="cuda") * inter**-0.5).to(torch.bfloat16)
    xg = x[a_idx.long()]

    # preact (gate, up) pairs whatever the address order; h from rounded preact
    pre, h = bf16_expert_gate_up_fwd(x, wg, wu, cu, a_idx)
    assert pre.shape == (M, 2 * inter) and h.shape == (M, inter)
    g_ref, g_abs = gmm(xg, wg, counts)
    u_ref, u_abs = gmm(xg, wu, counts)
    assert_within(pre[:, 0::2], g_ref, g_abs, "preact gate")
    assert_within(pre[:, 1::2], u_ref, u_abs, "preact up")
    g16, u16 = pre[:, 0::2].to(F64), pre[:, 1::2].to(F64)
    h_ref = swiglu_ref(g16, u16)
    assert_within(h, h_ref, h_ref.abs(), "h")

    out = bf16_expert_down_fwd(h, wd, cu, slot_of)
    y_ref, y_abs = gmm(h, wd, counts)
    ref, ref_abs = torch.empty_like(y_ref), torch.empty_like(y_abs)
    ref[slot_of.long()], ref_abs[slot_of.long()] = y_ref, y_abs
    assert_within(out, ref, ref_abs, "down")

    # dy read by slot_of, dpre in the weights' address order
    dy = torch.randn(M, H, **BF16)
    dpre, h2, swapped = bf16_expert_down_dgrad(dy, wd, pre, cu, slot_of, wg, wu)
    assert swapped == (wu.data_ptr() < wg.data_ptr()) == (layout == "swapped")
    assert torch.equal(h, h2), "the backward must re-emit the forward's h bit for bit"
    dh, dh_abs = gmm(dy[slot_of.long()], wd, counts, trans=False)
    dg_ref, du_ref = dswiglu_ref(g16, u16, dh)
    dg_abs, du_abs = dswiglu_ref(g16, u16, dh_abs)
    dg, du = lanes(dpre, swapped)
    assert_within(dg, dg_ref, dg_abs.abs(), "dgate")
    assert_within(du, du_ref, du_abs.abs(), "dup")

    # one GEMM over K = 2I summing the gate and up halves in fp32
    dx = bf16_expert_gate_up_dgrad(dpre, wg, wu, cu, swapped)
    a, a_abs = gmm(dg, wg, counts, trans=False)
    b, b_abs = gmm(du, wu, counts, trans=False)
    assert_within(dx, a + b, a_abs + b_abs, "dgrad")

    pre2, h3 = bf16_expert_gate_up_fwd(x, wg, wu, cu, a_idx)
    assert torch.equal(pre, pre2) and torch.equal(h, h3)
    assert torch.equal(out, bf16_expert_down_fwd(h, wd, cu, slot_of))
    assert torch.equal(dpre, bf16_expert_down_dgrad(dy, wd, pre, cu, slot_of, wg, wu)[0])
    assert torch.equal(dx, bf16_expert_gate_up_dgrad(dpre, wg, wu, cu, swapped))


WGRAD_CASES = [
    "qwen3-edges",
    "qwen3-imbalanced",
    "qwen3-large",
    "qwen3-g64-ragged",
    "dsv2-g8-ragged",
    "q35-g32",
]


@pytest.mark.parametrize("case", WGRAD_CASES)
@pytest.mark.parametrize(
    "w_layout,mg_layout",
    [
        ("separate", "separate"),
        ("separate", "swapped"),
        ("swapped", "separate"),
        ("swapped", "swapped"),
        ("halves", "halves"),
    ],
)
def test_swiglu_wgrads(case, w_layout, mg_layout):
    H, inter, counts = CASES[case]
    G, M = len(counts), sum(counts)
    x, a_idx, slot_of, cu = inputs(counts, H, seed=1)
    wg, wu = pair(G, inter, H, w_layout)
    swapped = wu.data_ptr() < wg.data_ptr()
    dpre = torch.randn(M, 2 * inter, **BF16)
    h = torch.randn(M, inter, **BF16)
    dy = torch.randn(M, H, **BF16)
    mg_g, mg_u = pair(G, inter, H, mg_layout, dtype=torch.float32, scale=3.0)
    mg_d = torch.randn(G, H, inter, device="cuda") * 3.0
    start = [t.clone() for t in (mg_g, mg_u, mg_d)]
    bf16_expert_gate_up_wgrad(x, dpre, swapped, mg_g, mg_u, cu, a_idx)
    bf16_expert_down_wgrad(dy, h, mg_d, cu, slot_of)

    xg = x[a_idx.long()]
    dg, du = lanes(dpre, swapped)
    for mg, s0, (inc, inc_abs), what in (
        (mg_g, start[0], kgrouped(dg, xg, counts), "mg_gate"),
        (mg_u, start[1], kgrouped(du, xg, counts), "mg_up"),
        (mg_d, start[2], kgrouped(dy[slot_of.long()], h, counts), "mg_down"),
    ):
        assert_within(mg, s0.to(F64) + inc, inc_abs, what, bf16=False)

    # deterministic: the same accumulation from the same start gives the same bits
    before = [t.clone() for t in (mg_g, mg_u, mg_d)]
    for t, s0 in zip((mg_g, mg_u, mg_d), start):
        t.copy_(s0)
    bf16_expert_gate_up_wgrad(x, dpre, swapped, mg_g, mg_u, cu, a_idx)
    bf16_expert_down_wgrad(dy, h, mg_d, cu, slot_of)
    for a, b in zip(before, (mg_g, mg_u, mg_d)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("case", ["qwen3-edges", "qwen3-imbalanced", "q35-g32", "dsv2-g16"])
def test_down_scattered(case):
    """
    Bias in fp32 before the rounding; the scattered rows equal the grouped rows reordered.
    """
    H, inter, counts = CASES[case]
    G, M = len(counts), sum(counts)
    torch.manual_seed(4)
    h = torch.randn(M, inter, **BF16)
    wd = (torch.randn(G, H, inter, device="cuda") * inter**-0.5).to(torch.bfloat16)
    cu = cu_of(counts)
    slot_of = torch.randperm(M, device="cuda").to(torch.int32)
    ident = torch.arange(M, device="cuda", dtype=torch.int32)
    y_ref, y_abs = gmm(h, wd, counts)
    for bias in (None, (torch.randn(G, H, device="cuda") * 4).to(torch.bfloat16)):
        grouped = bf16_expert_down_fwd(h, wd, cu, ident, bias)
        ref, ref_abs = y_ref, y_abs
        if bias is not None:
            rows = per_row_bias(bias, counts)
            ref, ref_abs = ref + rows, ref_abs + rows.abs()
        assert_within(grouped, ref, ref_abs, f"down bias={bias is not None}")
        out = bf16_expert_down_fwd(h, wd, cu, slot_of, bias)
        expect = torch.empty_like(grouped)
        expect[slot_of.long()] = grouped
        assert torch.equal(out, expect)


GPTOSS = {
    "g32-ragged": ragged(32, 16384, 5, empty=0.15),
    "g8-edges": [700, 0, 1, 2048, 300, 129, 900, 64],
}


@pytest.mark.parametrize("counts_case", list(GPTOSS))
@pytest.mark.parametrize("limit", [7.0, 2.0])
def test_gptoss(counts_case, limit):
    """
    gpt-oss: gate / up the interleaved rows of one (G, 2I, H) weight, a per-expert interleaved bias
    added in fp32 before the one rounding, the clamped SwiGLU (gradients zeroed where the clamp
    saturates), a bias on the down projection, and main grads that are the interleaved rows of one
    fp32 (G, 2I, H) buffer.
    """
    counts = GPTOSS[counts_case]
    H = inter = 2880
    alpha = 1.702
    act = (alpha, limit)
    G, M = len(counts), sum(counts)
    x, a_idx, slot_of, cu = inputs(counts, H, seed=3)
    wg, wu = pair(G, inter, H, "interleaved", scale=3 * H**-0.5)  # preacts beyond the limit
    bias = (torch.randn(G, 2 * inter, device="cuda") * 2).to(torch.bfloat16)
    wd = (torch.randn(G, H, inter, device="cuda") * inter**-0.5).to(torch.bfloat16)
    bd = (torch.randn(G, H, device="cuda") * 2).to(torch.bfloat16)
    xg = x[a_idx.long()]

    pre, h = bf16_expert_gate_up_fwd(x, wg, wu, cu, a_idx, bias, act)
    for lane, w, what in ((0, wg, "gate"), (1, wu, "up")):
        ref, ref_abs = gmm(xg, w, counts)
        rows = per_row_bias(bias[:, lane::2], counts)
        assert_within(pre[:, lane::2], ref + rows, ref_abs + rows.abs(), f"preact {what}")
    g16, u16 = pre[:, 0::2].to(F64), pre[:, 1::2].to(F64)
    assert (g16 > limit).any() and (u16.abs() > limit).any(), "the clamp must be exercised"
    h_ref = oai_ref(g16, u16, alpha, limit)
    assert_within(h, h_ref, h_ref.abs(), "h")

    out = bf16_expert_down_fwd(h, wd, cu, slot_of, bd)
    y_ref, y_abs = gmm(h, wd, counts)
    rows = per_row_bias(bd, counts)
    ref, ref_abs = torch.empty_like(y_ref), torch.empty_like(y_abs)
    ref[slot_of.long()], ref_abs[slot_of.long()] = y_ref + rows, y_abs + rows.abs()
    assert_within(out, ref, ref_abs, "down")

    dy = torch.randn(M, H, **BF16)
    dpre, h2, swapped = bf16_expert_down_dgrad(dy, wd, pre, cu, slot_of, wg, wu, act)
    assert not swapped
    assert torch.equal(h, h2)
    dh, dh_abs = gmm(dy[slot_of.long()], wd, counts, trans=False)
    dg_ref, du_ref = doai_ref(g16, u16, dh, alpha, limit)
    dg_abs, du_abs = doai_ref(g16, u16, dh_abs, alpha, limit)
    dg, du = lanes(dpre, swapped)
    assert_within(dg, dg_ref, dg_abs.abs(), "dgate")
    assert_within(du, du_ref, du_abs.abs(), "dup")
    assert torch.all(dg[g16 > limit] == 0) and torch.all(du[u16.abs() > limit] == 0)

    dx = bf16_expert_gate_up_dgrad(dpre, wg, wu, cu, swapped)
    a, a_abs = gmm(dg, wg, counts, trans=False)
    b, b_abs = gmm(du, wu, counts, trans=False)
    assert_within(dx, a + b, a_abs + b_abs, "dgrad")

    mg_g, mg_u = pair(G, inter, H, "interleaved", dtype=torch.float32)
    mg_d = torch.randn(G, H, inter, device="cuda")
    start = [t.clone() for t in (mg_g, mg_u, mg_d)]
    bf16_expert_gate_up_wgrad(x, dpre, swapped, mg_g, mg_u, cu, a_idx)
    bf16_expert_down_wgrad(dy, h, mg_d, cu, slot_of)
    for mg, s0, (inc, inc_abs), what in (
        (mg_g, start[0], kgrouped(dg, xg, counts), "mg_gate"),
        (mg_u, start[1], kgrouped(du, xg, counts), "mg_up"),
        (mg_d, start[2], kgrouped(dy[slot_of.long()], h, counts), "mg_down"),
    ):
        assert_within(mg, s0.to(F64) + inc, inc_abs, what, bf16=False)


@pytest.mark.parametrize(
    "H,inter,G,gptoss",
    [
        (2048, 768, 16, False),
        (2048, 1408, 8, False),
        (2048, 512, 32, False),
        (2880, 2880, 32, True),
    ],
)
def test_zero_rows(H, inter, G, gptoss):
    cu = torch.zeros(G + 1, dtype=torch.int32, device="cuda")
    idx = torch.empty(0, dtype=torch.int32, device="cuda")
    x = torch.randn(3, H, **BF16)
    layout = "interleaved" if gptoss else "separate"
    wg, wu = pair(G, inter, H, layout)
    wd = torch.randn(G, H, inter, **BF16)
    bias = torch.randn(G, 2 * inter, **BF16) if gptoss else None
    bd = torch.randn(G, H, **BF16) if gptoss else None
    act = (1.702, 7.0) if gptoss else ()
    pre, h = bf16_expert_gate_up_fwd(x, wg, wu, cu, idx, bias, act)
    assert pre.shape == (0, 2 * inter) and h.shape == (0, inter)
    out = bf16_expert_down_fwd(h, wd, cu, idx, bd)
    assert out.shape == (0, H)
    dy = torch.empty(0, H, **BF16)
    dpre, h2, swapped = bf16_expert_down_dgrad(dy, wd, pre, cu, idx, wg, wu, act)
    assert dpre.shape == (0, 2 * inter) and h2.shape == (0, inter)
    dx = bf16_expert_gate_up_dgrad(dpre, wg, wu, cu, swapped)
    assert dx.shape == (0, H)
    mg_g, mg_u = pair(G, inter, H, layout, dtype=torch.float32)
    mg_d = torch.randn(G, H, inter, device="cuda")
    start = [t.clone() for t in (mg_g, mg_u, mg_d)]
    bf16_expert_gate_up_wgrad(x, dpre, swapped, mg_g, mg_u, cu, idx)
    bf16_expert_down_wgrad(dy, h, mg_d, cu, idx)
    for a, b in zip(start, (mg_g, mg_u, mg_d)):
        assert torch.equal(a, b)


def test_wrong_group_offsets_fail():
    """
    The reference check is sharp: a GEMM run with shifted group offsets fails it.
    """
    H, inter, counts = CASES["qwen3-edges"]
    G = len(counts)
    x, a_idx, _, cu = inputs(counts, H, seed=0)
    wg, wu = pair(G, inter, H, "separate", scale=H**-0.5)
    g_ref, g_abs = gmm(x[a_idx.long()], wg, counts)
    bad = cu.clone()
    bad[1:-1] += 1
    pre, _ = bf16_expert_gate_up_fwd(x, wg, wu, bad, a_idx)
    with pytest.raises(AssertionError):
        assert_within(pre[:, 0::2], g_ref, g_abs, "preact gate")


def test_after_tuning(tmp_path, monkeypatch):
    """
    After a tuning round, the GEMMs run on their tuned configs and still pass every check.
    """
    monkeypatch.setenv("PITHTRAIN_TUNING_DIR", str(tmp_path))
    test_swiglu_forward_backward("qwen3-g32-balanced", "separate")
    test_swiglu_wgrads("qwen3-g32-balanced", "separate", "swapped")
    test_gptoss("g8-edges", 7.0)
    tune_kernels()
    test_swiglu_forward_backward("qwen3-g32-balanced", "separate")
    test_swiglu_wgrads("qwen3-g32-balanced", "separate", "swapped")
    test_gptoss("g8-edges", 7.0)
