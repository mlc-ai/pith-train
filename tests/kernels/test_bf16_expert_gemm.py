"""
The fused BF16 expert GEMMs at the bf16_expert_gemm cases, against float64.
"""

import pytest
import torch

from pithtrain.kernels import (
    bf16_expert_down_dgrad,
    bf16_expert_down_fwd,
    bf16_expert_down_wgrad,
    bf16_expert_gate_up_dgrad,
    bf16_expert_gate_up_fwd,
    bf16_expert_gate_up_wgrad,
    bf16_experts_supported,
)
from tests.kernels.cases import params
from tests.kernels.common import (
    ALPHA,
    BF16,
    BF16_REL,
    F64,
    LIMIT,
    Check,
    clamped_swiglu_bwd_reference,
    clamped_swiglu_reference,
    gate_up_layout,
    row_chunks,
    silu_mul_bwd_reference,
    silu_mul_reference,
)


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


BF16_COEF = {"bf16": 7e-6, "fp32": 1.6e-4}
BF16_CALC_DIFF = {"bf16": 5e-6, "fp32": 5e-7}


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


def bf16_check(what: str, wgrad: bool = False) -> Check:
    return Check(f"bf16 {what}", BF16_REL["fp32" if wgrad else "bf16"])


@pytest.mark.parametrize("case", params("bf16_expert_gemm", keep=lambda c: c["axes"]["bf16_fused"]))  # fmt: skip
def test_bf16_expert_gemms(case: dict) -> None:
    axes = case["axes"]
    G, H, inter = axes["G"], axes["hidden"], axes["inter"]
    if not bf16_experts_supported(G):
        pytest.skip(f"bf16_experts_supported({G}) is False here")
    counts = axes["group_sizes"]
    M = sum(counts)
    clamped, bias = axes["act"] == "clamped_swiglu", axes["bias"]
    act, dact = (
        (clamped_swiglu_reference, clamped_swiglu_bwd_reference)
        if clamped
        else (silu_mul_reference, silu_mul_bwd_reference)
    )
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
    for group, count in enumerate(counts):
        if count == 0:
            for target, start_value in ((mg_g, start_g), (mg_u, start_u), (mg_d, start_d)):
                assert torch.equal(target[group], start_value[group]), f"empty group {group} changed"  # fmt: skip
    if clamped:
        assert bool((pre[:, 0::2].float() > LIMIT).any()), "the clamp must be exercised"
    for name, check in checks.items():
        check.verify(case, BF16_COEF[kinds[name]], BF16_CALC_DIFF[kinds[name]])
