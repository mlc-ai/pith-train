"""
Routed expert GEMMs and fused FP8 activation quantizations of one EP rank at the spec's moe_experts
cases. FP8 GEMMs against the fp32 product of the dequantized operands, BF16 GEMMs against float64,
both within a per-element bound plus calc_diff (about 3x the worst measured on the H200); the FP8
activations within half an e4m3 ulp of float64, with the recipe's scales.
"""

import json
import os

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
    fp8_clamped_swiglu_bwd,
    fp8_clamped_swiglu_kmajor,
    fp8_clamped_swiglu_rowwise,
    fp8_grouped_gemm,
    fp8_grouped_wgrad_empty,
    fp8_quant_grouped,
    fp8_quant_weight_grouped,
    fp8_silu_mul_bwd,
    fp8_silu_mul_fwd,
    fp8_silu_mul_gu_bwd,
    fp8_silu_mul_gu_fwd,
    k_grouped_fp8_gemm,
)
from tests.kernels.cases import params, select

BLOCK = 128
CHUNK = 32768
F64 = torch.float64
BF16 = dict(device="cuda", dtype=torch.bfloat16)

# Bounds per output kind, about 3x the worst measured on the H200.
FP8_REL = {"bf16": 2.0**-8, "fp32": 2.0**-20}
FP8_COEF = {"bf16": 2.0**-11, "fp32": 2.0**-9}
FP8_CALC_DIFF = {"bf16": 5e-6, "fp32": 3e-8}
BF16_REL = {"bf16": 2.0**-8, "fp32": 2.0**-22}
BF16_COEF = {"bf16": 7e-6, "fp32": 1.6e-4}
BF16_CALC_DIFF = {"bf16": 5e-6, "fp32": 5e-7}
FP8 = lambda case: case["axes"]["routing"] != "boundary"  # boundary cases are for the BF16 picker
ALPHA = 1.702  # gpt-oss clamped SwiGLU, asserted against the case
LIMIT = 7.0


def record(case: dict, what: str, **values) -> None:
    """
    Append one measurement to $PITHTRAIN_SPEC_MEASURE.<pid> for calibrating the bounds.
    """
    path = os.environ.get("PITHTRAIN_SPEC_MEASURE")
    if path:
        with open(f"{path}.{os.getpid()}", "a") as f:
            f.write(json.dumps(dict(case=case["id"], what=what, **values)) + "\n")


@pytest.fixture(autouse=True)
def no_tf32():
    """
    Disable TF32 and empty the allocator cache around each test.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous
    torch.cuda.empty_cache()


class Check:
    """
    Running error statistics of one output: calc_diff, the smallest passing coef, finiteness.
    """

    def __init__(self, what: str, rel: float):
        self.what, self.rel = what, rel
        self.xy = self.xx = self.yy = 0.0
        self.coef = 0.0
        self.finite = True

    def add(self, out: torch.Tensor, ref: torch.Tensor, absref: torch.Tensor) -> None:
        if out.numel() == 0:
            return
        out, ref = out.to(F64), ref.to(F64)
        self.finite &= bool(torch.isfinite(out).all())
        self.xy += float((out * ref).sum())
        self.xx += float((out * out).sum())
        self.yy += float((ref * ref).sum())
        excess = ((out - ref).abs() - self.rel * ref.abs()).clamp(min=0)
        self.coef = max(self.coef, float((excess / (absref.to(F64) + 1e-30)).max()))

    @property
    def calc_diff(self) -> float:
        den = self.xx + self.yy
        return 0.0 if den == 0 else 1 - 2 * self.xy / den

    def verify(self, case: dict, coef: float, calc_diff: float | None) -> None:
        record(case, self.what, calc_diff=self.calc_diff, coef=self.coef)
        assert self.finite, f"{self.what}: non-finite output"
        assert self.coef <= coef, f"{self.what}: per-element error needs coef {self.coef:.3g} > {coef:.3g}"  # fmt: skip
        if calc_diff is not None:
            assert self.calc_diff <= calc_diff, f"{self.what}: calc_diff {self.calc_diff:.3g} > {calc_diff:.3g}"  # fmt: skip


def row_chunks(counts: list[int]):
    """
    Yield (group, group start, chunk start, chunk end) over grouped rows, chunks within a group.
    """
    start = 0
    for group, count in enumerate(counts):
        for chunk in range(start, start + count, CHUNK):
            yield group, start, chunk, min(chunk + CHUNK, start + count)
        start += count


def gate_up_layout(case: dict) -> str:
    """
    The case's gate_up_layout: separate, halves or interleaved.
    """
    return case["axes"]["gate_up_layout"]


# FP8 ---------------------------------------------------------------------------------------------


def dq_rows(fp8: torch.Tensor, scale: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """
    Rows [start, end) of a 1x128 rowwise FP8 operand, dequantized to fp32.
    """
    K = fp8.shape[1]
    return fp8[start:end].float() * scale[start:end].repeat_interleave(BLOCK, dim=1)[:, :K]


def dq_weight(fp8: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """
    One expert of a 128x128 blockwise FP8 weight (G, N, K), dequantized to fp32.
    """
    N, K = fp8.shape[1:]
    full = scale[group].repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return fp8[group].float() * full[:N, :K]


def dq_channels(pair, counts: list[int], group_start: int, group: int, start: int, end: int, channels: int):  # fmt: skip
    """
    Rows [start, end) of one group's per-channel FP8, dequantized to fp32 (channels, rows).
    """
    data, scale = pair
    first, last = (start - group_start), (end - group_start)
    rows = counts[group]
    block = data[group_start * channels : (group_start + rows) * channels].view(channels, rows)
    block_scale = scale[:, start // BLOCK : end // BLOCK].repeat_interleave(BLOCK, dim=1)
    return block[:, first:last].float() * block_scale


def check_m_grouped(case, what, out, a_pair, w_pair, counts, transposed=False):
    """
    Check out[rows of g] against dq(a)[rows] @ dq(w[g])^T.
    """
    check = Check(what, FP8_REL["bf16"])
    w_dq, w_group = None, -1
    for group, _, start, end in row_chunks(counts):
        if group != w_group:
            w_dq, w_group = dq_weight(*w_pair, group), group
        a = dq_rows(*a_pair, start, end)
        check.add(out[start:end], a @ w_dq.T, a.abs() @ w_dq.abs().T)
    check.verify(case, FP8_COEF["bf16"], FP8_CALC_DIFF["bf16"])


def check_k_grouped(case, what, d, a_pair, b_pair, counts, N, K):
    """
    Check d[g] against dq(a)_g @ dq(b)_g^T over the rows of group g.
    """
    check = Check(what, FP8_REL["fp32"])
    group_start = 0
    for group, count in enumerate(counts):
        ref = torch.zeros(N, K, device="cuda", dtype=torch.float32)
        absref = torch.zeros_like(ref)
        for start in range(group_start, group_start + count, CHUNK):
            end = min(start + CHUNK, group_start + count)
            a = dq_channels(a_pair, counts, group_start, group, start, end, N)
            b = dq_channels(b_pair, counts, group_start, group, start, end, K)
            ref += a @ b.T
            absref += a.abs() @ b.abs().T
        check.add(d[group], ref, absref)
        group_start += count
    check.verify(case, FP8_COEF["fp32"], FP8_CALC_DIFF["fp32"])


def grouped_input(rows: int, width: int, counts: list[int], padded: list[int], gen) -> torch.Tensor:
    """
    bf16 (sum(padded), width): random real rows per group, zero padding.
    """
    x = torch.zeros(sum(padded), width, **BF16)
    start = 0
    for count, size in zip(counts, padded):
        if count:
            x[start : start + count] = torch.randn(count, width, generator=gen, **BF16)
        start += size
    return x


@pytest.mark.parametrize("case", params("moe_experts", keep=FP8))
def test_fp8_expert_gemms(case: dict) -> None:
    axes = case["axes"]
    G, H, inter = axes["G"], axes["hidden"], axes["inter"]
    counts = axes["group_sizes"]
    ks = [-(-count // BLOCK) * BLOCK for count in counts]
    assert sum(ks) == axes["fp8_rows"]
    offs = torch.tensor(ks, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    ks_tensor = torch.tensor(ks, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(sum(counts) + G)
    # one 2I gate_up GEMM, or separate gate and up of I each
    gu_widths = [2 * inter] if gate_up_layout(case) != "separate" else [inter]
    M = sum(ks)

    # forward: x -> gate_up, h -> down
    x = grouped_input(M, H, counts, ks, gen)
    x_fp8, x_scale, x_ch, x_ch_scale = fp8_quant_grouped(x, offs)
    del x
    for n in gu_widths:
        w = (torch.randn(G, n, H, device="cuda", generator=gen) * H**-0.5).to(torch.bfloat16)
        w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight_grouped(w)
        del w
        out = fp8_grouped_gemm((x_fp8, x_scale), (w_fp8, w_scale), n, offs)
        assert out.shape == (M, n) and out.dtype == torch.bfloat16
        check_m_grouped(case, f"gate_up fwd n={n}", out, (x_fp8, x_scale), (w_fp8, w_scale), ks)
        del out
        # dgrad: dpre (M, n) through the transposed pair to (M, H)
        dpre = grouped_input(M, n, counts, ks, gen)
        dp_fp8, dp_scale, dp_ch, dp_ch_scale = fp8_quant_grouped(dpre, offs)
        del dpre
        dx = fp8_grouped_gemm((dp_fp8, dp_scale), (wt_fp8, wt_scale), H, offs)
        assert dx.shape == (M, H)
        check_m_grouped(case, f"gate_up dgrad n={n}", dx, (dp_fp8, dp_scale), (wt_fp8, wt_scale), ks)  # fmt: skip
        del dx, dp_fp8, dp_scale, w_fp8, w_scale, wt_fp8, wt_scale
        # wgrad: d (G, n, H) += dpre_g^T x_g
        d = fp8_grouped_wgrad_empty(torch.Size((G, n, H)), torch.device("cuda"))
        d.zero_()
        k_grouped_fp8_gemm((dp_ch, dp_ch_scale), (x_ch, x_ch_scale), d, ks, ks_tensor)
        check_k_grouped(case, f"gate_up wgrad n={n}", d, (dp_ch, dp_ch_scale), (x_ch, x_ch_scale), ks, n, H)  # fmt: skip
        del d, dp_ch, dp_ch_scale
    del x_fp8, x_scale, x_ch, x_ch_scale

    h = grouped_input(M, inter, counts, ks, gen)
    h_fp8, h_scale, h_ch, h_ch_scale = fp8_quant_grouped(h, offs)
    del h
    w = (torch.randn(G, H, inter, device="cuda", generator=gen) * inter**-0.5).to(torch.bfloat16)
    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight_grouped(w)
    del w
    y = fp8_grouped_gemm((h_fp8, h_scale), (w_fp8, w_scale), H, offs)
    assert y.shape == (M, H)
    check_m_grouped(case, "down fwd", y, (h_fp8, h_scale), (w_fp8, w_scale), ks)
    del y, h_fp8, h_scale

    dy = grouped_input(M, H, counts, ks, gen)
    dy_fp8, dy_scale, dy_ch, dy_ch_scale = fp8_quant_grouped(dy, offs)
    del dy
    dh = fp8_grouped_gemm((dy_fp8, dy_scale), (wt_fp8, wt_scale), inter, offs)
    assert dh.shape == (M, inter)
    check_m_grouped(case, "down dgrad", dh, (dy_fp8, dy_scale), (wt_fp8, wt_scale), ks)
    del dh, dy_fp8, dy_scale
    d = fp8_grouped_wgrad_empty(torch.Size((G, H, inter)), torch.device("cuda"))
    d.zero_()
    k_grouped_fp8_gemm((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), d, ks, ks_tensor)
    check_k_grouped(case, "down wgrad", d, (dy_ch, dy_ch_scale), (h_ch, h_ch_scale), ks, H, inter)


# BF16 --------------------------------------------------------------------------------------------


def gate_up_pair(case: dict, G: int, n: int, k: int, gen, dtype=torch.bfloat16, scale: float = 1.0):  # fmt: skip
    """
    (gate, up), each (G, n, k), in the model's gate_up layout.
    """
    layout = gate_up_layout(case)
    if layout == "separate":
        return tuple((torch.randn(G, n, k, device="cuda", generator=gen) * scale).to(dtype) for _ in range(2))  # fmt: skip
    buf = (torch.randn(G, 2 * n, k, device="cuda", generator=gen) * scale).to(dtype)
    if layout == "halves":
        return buf[:, :n], buf[:, n:]
    return buf[:, 0::2], buf[:, 1::2]


def act_ref(g, u):
    return F.silu(g) * u


def dact_ref(g, u, dh):
    sig = torch.sigmoid(g)
    return dh * u * sig * (1 + g * (1 - sig)), dh * g * sig


def clamped_ref(g, u):
    gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
    return gc * torch.sigmoid(ALPHA * gc) * (uc + 1)


def dclamped_ref(g, u, dh):
    gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
    sig = torch.sigmoid(ALPHA * gc)
    dg = dh * (uc + 1) * (sig + ALPHA * gc * sig * (1 - sig)) * (g <= LIMIT)
    du = dh * gc * sig * (u.abs() <= LIMIT)
    return dg, du


def bf16_check(what: str, wgrad: bool = False) -> Check:
    return Check(f"bf16 {what}", BF16_REL["fp32" if wgrad else "bf16"])


@pytest.mark.parametrize("case", params("moe_experts", keep=lambda c: c["axes"]["bf16_fused"]))  # fmt: skip
def test_bf16_expert_gemms(case: dict) -> None:
    axes = case["axes"]
    G, H, inter = axes["G"], axes["hidden"], axes["inter"]
    if not bf16_experts_supported(G):
        pytest.skip(f"bf16_experts_supported({G}) is False here")
    counts = axes["group_sizes"]
    M = sum(counts)
    clamped, bias = axes["act"] == "clamped_swiglu", axes["bias"]
    act, dact = (clamped_ref, dclamped_ref) if clamped else (act_ref, dact_ref)
    act_params = (ALPHA, LIMIT) if clamped else ()
    assert not clamped or (axes["swiglu_alpha"], axes["swiglu_limit"]) == act_params
    gen = torch.Generator(device="cuda").manual_seed(M + G)
    received = max(M // 2, 1)  # received (deduplicated) tokens, each read by about two slots
    x = torch.randn(received, H, generator=gen, **BF16)
    a_idx = torch.randint(0, received, (M,), device="cuda", generator=gen, dtype=torch.int32)
    slot_of = torch.randperm(M, device="cuda", generator=gen).to(torch.int32)
    cu = torch.tensor([0] + counts, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    wg, wu = gate_up_pair(case, G, inter, H, gen, scale=(3 if clamped else 1) * H**-0.5)
    wd = (torch.randn(G, H, inter, device="cuda", generator=gen) * inter**-0.5).to(torch.bfloat16)
    b_gu = (torch.randn(G, 2 * inter, device="cuda", generator=gen) * 2).to(torch.bfloat16) if bias else None  # fmt: skip
    b_d = (torch.randn(G, H, device="cuda", generator=gen) * 2).to(torch.bfloat16) if bias else None  # fmt: skip
    a_idx_l, slot_l = a_idx.long(), slot_of.long()

    pre, h = bf16_expert_gate_up_fwd(x, wg, wu, cu, a_idx, b_gu, act_params)
    assert pre.shape == (M, 2 * inter) and h.shape == (M, inter)
    out = bf16_expert_down_fwd(h, wd, cu, slot_of, b_d)
    assert out.shape == (M, H)
    dy = torch.randn(M, H, generator=gen, **BF16)
    dpre, h2, swapped = bf16_expert_down_dgrad(dy, wd, pre, cu, slot_of, wg, wu, act_params)
    assert torch.equal(h, h2), "the backward must re-emit the forward's h bit for bit"
    dx = bf16_expert_gate_up_dgrad(dpre, wg, wu, cu, swapped)
    assert dx.shape == (M, H)
    dg_all, du_all = (dpre[:, 1::2], dpre[:, 0::2]) if swapped else (dpre[:, 0::2], dpre[:, 1::2])
    mg_g, mg_u = gate_up_pair(case, G, inter, H, gen, dtype=torch.float32)
    mg_d = torch.randn(G, H, inter, device="cuda", generator=gen)
    start_g, start_u, start_d = mg_g.clone(), mg_u.clone(), mg_d.clone()
    bf16_expert_gate_up_wgrad(x, dpre, swapped, mg_g, mg_u, cu, a_idx)
    bf16_expert_down_wgrad(dy, h, mg_d, cu, slot_of)

    names = ["preact gate", "preact up", "h", "down fwd", "dgate", "dup", "gate_up dgrad"]
    checks = {name: bf16_check(name) for name in names}
    checks.update({name: bf16_check(name, wgrad=True) for name in ("mg_gate", "mg_up", "mg_down")})  # fmt: skip
    kinds = {name: "fp32" if name.startswith("mg_") else "bf16" for name in checks}
    wgrad = {name: [None, None] for name in ("mg_gate", "mg_up", "mg_down")}
    group_w = -1
    for group, group_start, start, end in row_chunks(counts):
        if group != group_w:
            for name, (inc, inc_abs) in wgrad.items():
                if inc is not None:
                    target, start_value = {"mg_gate": (mg_g, start_g), "mg_up": (mg_u, start_u), "mg_down": (mg_d, start_d)}[name]  # fmt: skip
                    checks[name].add(target[group_w], start_value[group_w].to(F64) + inc, inc_abs)
            wgrad = {name: [0, 0] for name in wgrad}
            Wg, Wu, Wd = wg[group].to(F64), wu[group].to(F64), wd[group].to(F64)
            group_w = group
        xg = x[a_idx_l[start:end]].to(F64)
        g_ref, u_ref = xg @ Wg.T, xg @ Wu.T
        g_abs, u_abs = xg.abs() @ Wg.abs().T, xg.abs() @ Wu.abs().T
        if bias:
            bg, bu = b_gu[group, 0::2].to(F64), b_gu[group, 1::2].to(F64)
            g_ref, u_ref, g_abs, u_abs = g_ref + bg, u_ref + bu, g_abs + bg.abs(), u_abs + bu.abs()
        checks["preact gate"].add(pre[start:end, 0::2], g_ref, g_abs)
        checks["preact up"].add(pre[start:end, 1::2], u_ref, u_abs)
        g16, u16 = pre[start:end, 0::2].to(F64), pre[start:end, 1::2].to(F64)
        h_ref = act(g16, u16)
        checks["h"].add(h[start:end], h_ref, h_ref.abs())
        hc = h[start:end].to(F64)
        y_ref, y_abs = hc @ Wd.T, hc.abs() @ Wd.abs().T
        if bias:
            bd = b_d[group].to(F64)
            y_ref, y_abs = y_ref + bd, y_abs + bd.abs()
        checks["down fwd"].add(out[slot_l[start:end]], y_ref, y_abs)
        dyc = dy[slot_l[start:end]].to(F64)
        dh, dh_abs = dyc @ Wd, dyc.abs() @ Wd.abs()
        dg_ref, du_ref = dact(g16, u16, dh)
        dg_abs, du_abs = dact(g16, u16, dh_abs)
        dg, du = dg_all[start:end], du_all[start:end]
        checks["dgate"].add(dg, dg_ref, dg_abs.abs())
        checks["dup"].add(du, du_ref, du_abs.abs())
        dg64, du64 = dg.to(F64), du.to(F64)
        dx_ref = dg64 @ Wg + du64 @ Wu
        dx_abs = dg64.abs() @ Wg.abs() + du64.abs() @ Wu.abs()
        checks["gate_up dgrad"].add(dx[start:end], dx_ref, dx_abs)
        for name, (a, b) in (("mg_gate", (dg64, xg)), ("mg_up", (du64, xg)), ("mg_down", (dyc, hc))):  # fmt: skip
            wgrad[name][0] = wgrad[name][0] + a.T @ b
            wgrad[name][1] = wgrad[name][1] + a.abs().T @ b.abs()
    for name, (inc, inc_abs) in wgrad.items():
        if group_w >= 0 and not isinstance(inc, int):
            target, start_value = {"mg_gate": (mg_g, start_g), "mg_up": (mg_u, start_u), "mg_down": (mg_d, start_d)}[name]  # fmt: skip
            checks[name].add(target[group_w], start_value[group_w].to(F64) + inc, inc_abs)
    # groups without rows keep their main grads bit for bit
    for group, count in enumerate(counts):
        if count == 0:
            for target, start_value in ((mg_g, start_g), (mg_u, start_u), (mg_d, start_d)):
                assert torch.equal(target[group], start_value[group]), f"empty group {group} changed"  # fmt: skip
    if clamped:
        assert bool((pre[:, 0::2].float() > LIMIT).any()), "the clamp must be exercised"
    for name, check in checks.items():
        check.verify(case, BF16_COEF[kinds[name]], BF16_CALC_DIFF[kinds[name]])


# FP8 fused activation + quantization --------------------------------------------------------------


def e8m0(amax: torch.Tensor) -> torch.Tensor:
    """
    The recipe's E8M0 scale of a block amax.
    """
    raw = amax.float().clamp(min=1e-4) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    bits = raw.view(torch.int32)
    bump = torch.where((bits & 0x007FFFFF) != 0, 0x00800000, 0).to(torch.int32)
    return ((bits & 0x7F800000) + bump).view(torch.float32)


class QuantCheck:
    """
    An FP8 activation operand against float64: within half an e4m3 ulp, with the recipe's scales.
    """

    def __init__(self, what: str):
        self.what, self.bad, self.bad_scales, self.blocks, self.worst = what, 0, 0, 0, 0.0

    def add(self, deq: torch.Tensor, scale_full: torch.Tensor, ref: torch.Tensor) -> None:
        """
        Check aligned deq and ref; scale_full repeats each block's scale.
        """
        if deq.numel() == 0:
            return
        ref = ref.to(F64)
        tol = (2.0**-4 + 2.0**-8) * ref.abs() + scale_full.to(F64) * 2.0**-10
        ratio = (deq.to(F64) - ref).abs() / tol
        self.worst = max(self.worst, float(ratio.max()))
        self.bad += int((ratio > 1).sum())

    def add_scales(self, scale: torch.Tensor, amax: torch.Tensor) -> None:
        """
        Check per-block scales against the recipe's scale of the bf16 amax.
        """
        lo, hi = e8m0(amax * (1 - 2.0**-7)), e8m0(amax * (1 + 2.0**-7))
        near = lo != hi
        self.blocks += scale.numel()
        self.bad_scales += int(((scale != e8m0(amax)) & ~near).sum())

    def verify(self, case: dict) -> None:
        record(case, self.what, worst=self.worst, bad=self.bad, bad_scales=self.bad_scales)
        assert self.bad == 0, f"{self.what}: {self.bad} elements out of bounds (worst {self.worst:.3g}x)"  # fmt: skip
        assert self.bad_scales == 0, f"{self.what}: {self.bad_scales} of {self.blocks} block scales differ from the recipe"  # fmt: skip


def check_rowwise(check: QuantCheck, pair, ref_fn, M: int) -> None:
    """
    Check a 1x128 rowwise (fp8, scale) pair against the float64 rows ref_fn(start, end).
    """
    fp8, scale = pair
    N = fp8.shape[1]
    nb = -(-N // BLOCK)
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        ref = ref_fn(start, end)
        rounded = ref.to(torch.bfloat16).float()
        padded = torch.zeros(end - start, nb * BLOCK, device="cuda")
        padded[:, :N] = rounded.abs()
        amax = padded.view(end - start, nb, BLOCK).amax(-1)
        sc = scale[start:end]
        check.add_scales(sc, amax)
        full = sc.repeat_interleave(BLOCK, dim=1)[:, :N]
        check.add(fp8[start:end].float() * full, full, ref)


def check_channels(check: QuantCheck, pair, ref_fn, counts: list[int], channels: int) -> None:
    """
    Check a per-channel (fp8, scale) pair (SM90 K-major) against the float64 rows of ref_fn.
    """
    data, scale = pair
    for group, group_start, start, end in row_chunks(counts):
        ref = ref_fn(start, end)  # (rows, channels)
        rounded = ref.to(torch.bfloat16).float().abs()
        amax = rounded.view(-1, BLOCK, channels).amax(1)  # (rows / 128, channels)
        sc = scale[:, start // BLOCK : end // BLOCK].T
        check.add_scales(sc, amax)
        deq = dq_channels(pair, counts, group_start, group, start, end, channels).T
        check.add(deq, sc.repeat_interleave(BLOCK, dim=0), ref)


def silu_ref(g, u):
    return F.silu(g) * u


def dsilu_ref(g, u, dh):
    sig = torch.sigmoid(g)
    return dh * u * sig * (1 + g * (1 - sig)), dh * g * sig


@pytest.mark.parametrize("case", params("moe_experts", keep=FP8))
def test_fp8_expert_activation(case: dict) -> None:
    """
    The fused activation + FP8 quantization kernels in the model's layout against float64.
    """
    axes = case["axes"]
    G, inter = axes["G"], axes["inter"]
    counts = axes["group_sizes"]
    ks = [-(-count // BLOCK) * BLOCK for count in counts]
    M = sum(ks)
    offs = torch.tensor(ks, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    gen = torch.Generator(device="cuda").manual_seed(M + 7 * G)
    layout = gate_up_layout(case)
    gu = grouped_input(M, 2 * inter, counts, ks, gen)
    dh = grouped_input(M, inter, counts, ks, gen)
    checks = []

    if layout == "interleaved":
        assert (case["axes"]["swiglu_alpha"], case["axes"]["swiglu_limit"]) == (ALPHA, LIMIT)
        gu = gu * 4  # reach the clamp
        bias = (torch.randn(G, 2 * inter, device="cuda", generator=gen)).to(torch.bfloat16)
        row_bias = torch.repeat_interleave(bias, torch.tensor(ks, device="cuda"), dim=0, output_size=M)  # fmt: skip

        def z(start, end):  # gate_up + bias in fp32, rounded to bf16
            return (gu[start:end].float() + row_bias[start:end].float()).to(torch.bfloat16).to(F64)  # fmt: skip

        h_fn = lambda s, e: clamped_ref(z(s, e)[:, 0::2], z(s, e)[:, 1::2])
        rows = fp8_clamped_swiglu_rowwise(gu, bias, offs, ALPHA, LIMIT)
        ch = fp8_clamped_swiglu_kmajor(gu, bias, offs, ALPHA, LIMIT)
        checks.append(QuantCheck("clamped h rowwise"))
        check_rowwise(checks[-1], rows, h_fn, M)
        checks.append(QuantCheck("clamped h channels"))
        check_channels(checks[-1], ch, h_fn, ks, inter)
        del rows, ch

        def dgu_fn(start, end):
            zz = z(start, end)
            dg, du = dclamped_ref(zz[:, 0::2], zz[:, 1::2], dh[start:end].to(F64))
            out = torch.empty(end - start, 2 * inter, device="cuda", dtype=F64)
            out[:, 0::2], out[:, 1::2] = dg, du
            return out

        dgu_quant, d_bias = fp8_clamped_swiglu_bwd(dh, gu, bias, offs, ALPHA, LIMIT)
        checks.append(QuantCheck("clamped dgate_up rowwise"))
        check_rowwise(checks[-1], dgu_quant[:2], dgu_fn, M)
        checks.append(QuantCheck("clamped dgate_up channels"))
        check_channels(checks[-1], dgu_quant[2:], dgu_fn, ks, 2 * inter)
        bias_check = Check("clamped d_bias", BF16_REL["bf16"])
        group_start = 0
        for group, size in enumerate(ks):
            ref = torch.zeros(2 * inter, device="cuda", dtype=F64)
            absref = torch.zeros_like(ref)
            for start in range(group_start, group_start + size, CHUNK):
                part = dgu_fn(start, min(start + CHUNK, group_start + size))
                ref += part.sum(0)
                absref += part.abs().sum(0)
            # one bf16 rounding per summand
            bias_check.add(d_bias[group], ref, absref)
            group_start += size
        bias_check.verify(case, 2.0**-8, None)
        assert bool((gu.float() > LIMIT).any()), "the clamp must be exercised"
    else:
        g = gu[:, :inter].contiguous() if layout == "halves" else gu[:, 0::2].contiguous()
        u = gu[:, inter:].contiguous() if layout == "halves" else gu[:, 1::2].contiguous()
        h_fn = lambda s, e: silu_ref(g[s:e].to(F64), u[s:e].to(F64))
        quant = fp8_silu_mul_gu_fwd(gu, offs) if layout == "halves" else fp8_silu_mul_fwd(g, u, offs)  # fmt: skip
        checks.append(QuantCheck("silu h rowwise"))
        check_rowwise(checks[-1], quant[:2], h_fn, M)
        checks.append(QuantCheck("silu h channels"))
        check_channels(checks[-1], quant[2:], h_fn, ks, inter)
        del quant
        grads = lambda s, e: dsilu_ref(g[s:e].to(F64), u[s:e].to(F64), dh[s:e].to(F64))
        if layout == "halves":
            dgu_quant = fp8_silu_mul_gu_bwd(dh, gu, offs)
            cat_fn = lambda s, e: torch.cat(grads(s, e), dim=1)
            checks.append(QuantCheck("silu dgate_up rowwise"))
            check_rowwise(checks[-1], dgu_quant[:2], cat_fn, M)
            checks.append(QuantCheck("silu dgate_up channels"))
            check_channels(checks[-1], dgu_quant[2:], cat_fn, ks, 2 * inter)
        else:
            dg_quant, du_quant = fp8_silu_mul_bwd(dh, g, u, offs)
            for lane, quant in ((0, dg_quant), (1, du_quant)):
                fn = lambda s, e, lane=lane: grads(s, e)[lane]
                name = ("dgate", "dup")[lane]
                checks.append(QuantCheck(f"silu {name} rowwise"))
                check_rowwise(checks[-1], quant[:2], fn, M)
                checks.append(QuantCheck(f"silu {name} channels"))
                check_channels(checks[-1], quant[2:], fn, ks, inter)
    for check in checks:
        check.verify(case)


@pytest.mark.smoke
def test_fp8_check_is_sharp() -> None:
    """
    The FP8 GEMM check rejects a wrong-expert GEMM and an unquantized bf16 product.
    """
    case = select("moe_experts", "smoke")[0]
    axes = case["axes"]
    G, H, n = axes["G"], axes["hidden"], axes["inter"]
    ks = [-(-count // BLOCK) * BLOCK for count in axes["group_sizes"]]
    offs = torch.tensor(ks, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = grouped_input(sum(ks), H, axes["group_sizes"], ks, gen)
    w = (torch.randn(G, n, H, device="cuda", generator=gen) * H**-0.5).to(torch.bfloat16)
    x_pair = fp8_quant_grouped(x, offs)[:2]
    w_pair = fp8_quant_weight_grouped(w)[:2]
    order = [(g + 1) % G for g in range(G)]
    rolled = tuple(t[order].contiguous() for t in w_pair)
    wrong = fp8_grouped_gemm(x_pair, rolled, n, offs)
    with pytest.raises(AssertionError):
        check_m_grouped(case, "wrong expert", wrong, x_pair, w_pair, ks)
    unquantized = torch.cat([x[a:b] @ w[g].T for g, (a, b) in enumerate(zip([0] + offs.tolist()[:-1], offs.tolist()))])  # fmt: skip
    with pytest.raises(AssertionError):
        check_m_grouped(case, "unquantized", unquantized, x_pair, w_pair, ks)
