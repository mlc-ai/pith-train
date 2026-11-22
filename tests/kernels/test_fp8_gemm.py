"""
FP8 dense and grouped GEMMs, their activation quantizations and the fused FP8 SwiGLU MLP at the
fp8_gemm cases, against fp32 of the exactly dequantized operands.
"""

import pytest
import torch

from pithtrain.kernels import (
    fp8_fused_experts_supported,
    fp8_gemm_bwd,
    fp8_gemm_fwd,
    fp8_grouped_gemm,
    fp8_grouped_wgrad_empty,
    fp8_quant_act,
    fp8_quant_grad,
    fp8_quant_grouped,
    fp8_quant_weight,
    fp8_quant_weight_grouped,
    k_grouped_fp8_gemm,
)
from pithtrain.operators.fp8_swiglu_mlp import fp8_swiglu_mlp
from pithtrain.operators.linear import FP8Linear
from pithtrain.operators.silu_mul import silu_mul
from tests.kernels.cases import MIN_LOCAL, params, select
from tests.kernels.common import (
    BLOCK,
    CHUNK,
    DEV,
    FP8,
    Check,
    assert_fp8_equal,
    assert_scale_equal,
    dq_channels,
    fp8_quant_grad_reference,
    fp8_quant_weight_reference,
    gate_up_layout,
    grouped_input,
    half_ulp_bf16,
    record,
    row_chunks,
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


# Bounds per output kind, about 3x the worst measured on the H200.
FP8_REL = {"bf16": 2.0**-8, "fp32": 2.0**-20}
FP8_COEF = {"bf16": 2.0**-11, "fp32": 2.0**-9}
FP8_CALC_DIFF = {"bf16": 5e-6, "fp32": 3e-8}


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


@pytest.mark.parametrize(
    "case", params("fp8_gemm", keep=lambda c: "group_sizes" in c["axes"] and FP8(c))
)
def test_fp8_expert_gemms(case: dict) -> None:
    axes = case["axes"]
    G, H, inter = axes["G"], axes["hidden"], axes["inter"]
    counts = axes["group_sizes"]
    ks = [-(-count // BLOCK) * BLOCK for count in counts]
    assert sum(ks) == axes["fp8_rows"]
    offs = torch.tensor(ks, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()
    ks_tensor = torch.tensor(ks, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(sum(counts) + G)
    gu_widths = [2 * inter] if gate_up_layout(case) != "separate" else [inter]
    M = sum(ks)

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
        dpre = grouped_input(M, n, counts, ks, gen)
        dp_fp8, dp_scale, dp_ch, dp_ch_scale = fp8_quant_grouped(dpre, offs)
        del dpre
        dx = fp8_grouped_gemm((dp_fp8, dp_scale), (wt_fp8, wt_scale), H, offs)
        assert dx.shape == (M, H)
        check_m_grouped(case, f"gate_up dgrad n={n}", dx, (dp_fp8, dp_scale), (wt_fp8, wt_scale), ks)  # fmt: skip
        del dx, dp_fp8, dp_scale, w_fp8, w_scale, wt_fp8, wt_scale
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


def test_fp8_check_is_sharp() -> None:
    """
    The FP8 GEMM check rejects a wrong-expert GEMM and an unquantized bf16 product.
    """
    case = next(c for c in select("fp8_gemm") if "group_sizes" in c["axes"])
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


FP8_LINEAR_ROWS_PER_CHUNK = 4096
FP8_LINEAR_ACCUM = 3e-4  # x sum |a_k b_k|, beyond the output's bf16 rounding
FP8_LINEAR_CALC_DIFF = 3e-6  # bf16 rounding alone gives 1.39e-6
FP8_LINEAR_BF16_CALC_DIFF = 2e-3  # from quantizing both operands to e4m3


def fp8_linear_input(shape: tuple[int, int], seed: int, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=gen) * scale).to(torch.bfloat16)


class FP8LinearStats:
    """
    calc_diff and the worst elementwise excess accumulated over row chunks.
    """

    def __init__(self):
        self.xy = self.xx = self.yy = 0.0
        self.excess = 0.0

    def add(self, out: torch.Tensor, ref: torch.Tensor, absref: torch.Tensor | None) -> None:
        x, y = out.double(), ref.double()
        self.xy += float((x * y).sum())
        self.xx += float((x * x).sum())
        self.yy += float((y * y).sum())
        if absref is not None:
            excess = ((x - y).abs() - half_ulp_bf16(y)).clamp(min=0) / (absref.double() + 1e-30)
            self.excess = max(self.excess, float(excess.max()))

    @property
    def calc_diff(self) -> float:
        den = self.xx + self.yy
        return 0.0 if den == 0 else 1 - 2 * self.xy / den


def compare_fp8_linear(
    out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, with_abs: bool
) -> FP8LinearStats:
    """
    out (m, n) against a @ b.T in fp32 (a (m, k), b (n, k) fp32), in row chunks.
    """
    stats = FP8LinearStats()
    for start in range(0, out.shape[0], FP8_LINEAR_ROWS_PER_CHUNK):
        rows = slice(start, start + FP8_LINEAR_ROWS_PER_CHUNK)
        ref = a[rows] @ b.T
        absref = a[rows].abs() @ b.abs().T if with_abs else None
        stats.add(out[rows], ref, absref)
    return stats


def check_fp8_linear(
    case_id: str, name: str, out: torch.Tensor, exact: tuple, truth: tuple
) -> None:
    exact_stats = compare_fp8_linear(out, *exact, with_abs=True)
    truth_stats = compare_fp8_linear(out, *truth, with_abs=False)
    record(case_id, name, calc_diff=exact_stats.calc_diff, accum=exact_stats.excess, bf16_calc_diff=truth_stats.calc_diff)  # fmt: skip
    assert exact_stats.excess <= FP8_LINEAR_ACCUM, f"{name}: elementwise excess {exact_stats.excess:.3g} > {FP8_LINEAR_ACCUM:.3g} x sum|ab|"  # fmt: skip
    assert exact_stats.calc_diff <= FP8_LINEAR_CALC_DIFF, f"{name}: calc_diff {exact_stats.calc_diff:.3g} vs exact dequant"  # fmt: skip
    assert truth_stats.calc_diff <= FP8_LINEAR_BF16_CALC_DIFF, f"{name}: calc_diff {truth_stats.calc_diff:.3g} vs bf16 product"  # fmt: skip


@pytest.mark.parametrize("case", params("fp8_gemm", keep=lambda c: "projection" in c["axes"]))
def test_fp8_linear(case: dict) -> None:
    axes = case["axes"]
    M, N, K = axes["M"], axes["N"], axes["K"]
    x, w, dy = (
        fp8_linear_input((M, K), 1),
        fp8_linear_input((N, K), 2, 0.02),
        fp8_linear_input((M, N), 3),
    )

    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(w)
    ref_w, ref_w_scale = fp8_quant_weight_reference(w)
    assert_fp8_equal(w_fp8, ref_w, "w_fp8")
    assert_scale_equal(w_scale, ref_w_scale, "w_scale")
    assert_fp8_equal(wt_fp8, ref_w.T.contiguous(), "wt_fp8")
    assert_scale_equal(wt_scale, ref_w_scale.T.contiguous(), "wt_scale")

    x_fp8, x_scale, _, _ = fp8_quant_act(x)
    ref_x, ref_x_scale = fp8_quant_grad_reference(x)
    assert_fp8_equal(x_fp8, ref_x, "x_fp8")
    assert_scale_equal(x_scale, ref_x_scale, "x_scale")
    del x_fp8, x_scale

    y, xt_fp8, xt_scale = fp8_gemm_fwd(x, w_fp8, w_scale)
    assert y.shape == (M, N) and y.dtype == torch.bfloat16
    ref_xt, ref_xt_scale = fp8_quant_weight_reference(x.T)
    assert_fp8_equal(xt_fp8, ref_xt, "xt_fp8")
    assert_scale_equal(xt_scale, ref_xt_scale, "xt_scale")
    w_dq = dequant_blockwise(ref_w, ref_w_scale)
    check_fp8_linear(
        case["id"], "y", y, (dequant_rowwise(ref_x, ref_x_scale), w_dq), (x.float(), w.float())
    )
    del y

    dy_fp8, dy_scale, dyt_fp8, dyt_scale = fp8_quant_grad(dy)
    ref_dy, ref_dy_scale = fp8_quant_grad_reference(dy)
    ref_dyt, ref_dyt_scale = fp8_quant_grad_reference(dy.T)
    assert_fp8_equal(dy_fp8, ref_dy, "dy_fp8")
    assert_scale_equal(dy_scale, ref_dy_scale, "dy_scale")
    assert_fp8_equal(dyt_fp8, ref_dyt, "dyt_fp8")
    assert_scale_equal(dyt_scale, ref_dyt_scale, "dyt_scale")
    del dy_fp8, dy_scale, dyt_fp8, dyt_scale

    dx, dw = fp8_gemm_bwd(dy, wt_fp8, wt_scale, xt_fp8, xt_scale, K)
    assert dx.shape == (M, K) and dx.dtype == torch.bfloat16
    assert dw.shape == (N, K) and dw.dtype == torch.bfloat16
    dy_dq = dequant_rowwise(ref_dy, ref_dy_scale)
    check_fp8_linear(case["id"], "dx", dx, (dy_dq, w_dq.T.contiguous()), (dy.float(), w.float().T.contiguous()))  # fmt: skip
    del dx, dy_dq, w_dq
    xt_dq = dequant_blockwise(ref_xt, ref_xt_scale)
    check_fp8_linear(case["id"], "dw", dw, (dequant_rowwise(ref_dyt, ref_dyt_scale), xt_dq), (dy.float().T.contiguous(), x.float().T.contiguous()))  # fmt: skip


WGRAD_CASES = params(
    "fp8_gemm", keep=lambda c: "projection" in c["axes"] and c["axes"]["M"] == MIN_LOCAL
)


@pytest.mark.parametrize("case", WGRAD_CASES)
def test_wgrad_bf16_rounding(case: dict) -> None:
    """
    The fp32 sum of bf16 weight gradients over micro-batches stays within their bf16 roundings.
    """
    microbatches = 4
    axes = case["axes"]
    M, N, K = min(axes["M"], 4096), axes["N"], axes["K"]
    w = fp8_linear_input((N, K), 2, 0.02)
    _, _, wt_fp8, wt_scale = fp8_quant_weight(w)
    main_grad = torch.zeros(N, K, dtype=torch.float32, device=DEV)
    exact = torch.zeros(N, K, dtype=torch.float64, device=DEV)
    bound = torch.zeros(N, K, dtype=torch.float64, device=DEV)
    for step in range(microbatches):
        x, dy = fp8_linear_input((M, K), 10 + step), fp8_linear_input((M, N), 20 + step)
        _, xt_fp8, xt_scale = fp8_gemm_fwd(x, *quant_weight_pair(w))
        _, dw = fp8_gemm_bwd(dy, wt_fp8, wt_scale, xt_fp8, xt_scale, K)
        main_grad += dw.float()
        ref_dyt, ref_dyt_scale = fp8_quant_grad_reference(dy.T)
        term = dequant_rowwise(ref_dyt, ref_dyt_scale).double() @ dequant_blockwise(xt_fp8, xt_scale).double().T  # fmt: skip
        absterm = dequant_rowwise(ref_dyt, ref_dyt_scale).abs().double() @ dequant_blockwise(xt_fp8, xt_scale).abs().double().T  # fmt: skip
        exact += term
        bound += half_ulp_bf16(term) + FP8_LINEAR_ACCUM * absterm
    err = (main_grad.double() - exact).abs()
    rel = float((main_grad.double() - exact).norm() / exact.norm())
    record(case["id"], "wgrad_accumulated", rel_l2=rel, worst=float((err / (bound + 1e-30)).max()))
    assert bool((err <= bound * 1.0001 + 1e-30).all()), f"accumulated wgrad error {float((err / bound).max()):.3g}x the per-term bf16 bound"  # fmt: skip


def quant_weight_pair(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w_fp8, w_scale, _, _ = fp8_quant_weight(w)
    return w_fp8, w_scale


FP8_SWIGLU_MLP_CALC_DIFF = 7.5e-3


def fp8_swiglu_mlp_calc_diff(got: torch.Tensor, exact: torch.Tensor) -> float:
    got = got.double()
    return float(1 - 2 * (got * exact).sum() / (got.square().sum() + exact.square().sum()))


@pytest.mark.parametrize(
    "case", params("fp8_gemm", keep=lambda c: c["axes"].get("fp8_swiglu_mlp", False))
)
def test_fp8_swiglu_mlp(case: dict) -> None:
    if not fp8_fused_experts_supported():
        pytest.skip("the fused FP8 MLP needs fp8_fused_experts_supported()")
    M, inter, hidden = case["axes"]["M"], case["axes"]["inter"], case["axes"]["hidden"]
    gen = torch.Generator(device=DEV).manual_seed(M + inter + hidden)
    layers = [FP8Linear(hidden, inter, bias=False), FP8Linear(hidden, inter, bias=False)]
    layers.append(FP8Linear(inter, hidden, bias=False))
    for layer in layers:
        layer.to(device=DEV, dtype=torch.bfloat16)
        with torch.no_grad():
            fan_in = layer.weight.shape[1]
            layer.weight.copy_(torch.randn(layer.weight.shape, device=DEV, generator=gen) * fan_in**-0.5)  # fmt: skip
    gate, up, down = layers
    x = torch.randn(M, hidden, device=DEV, generator=gen).to(torch.bfloat16)
    dy = torch.randn(M, hidden, device=DEV, generator=gen).to(torch.bfloat16)

    def run(fused: bool) -> list[torch.Tensor]:
        for layer in layers:
            layer.weight.grad = None
        leaf = x.clone().requires_grad_(True)
        if fused:
            y = fp8_swiglu_mlp(gate, up, down, leaf)
        else:
            quantized = FP8Linear.quantize_input(leaf)
            y = down(silu_mul(gate(leaf, quantized), up(leaf, quantized)))
        y.backward(dy)
        return [y.detach(), leaf.grad] + [layer.weight.grad for layer in layers]

    names = ["y", "dx", "dw_gate", "dw_up", "dw_down"]
    fused, unfused = run(True), run(False)
    for name, a, b in zip(names, fused, unfused):
        assert a.dtype == b.dtype and torch.equal(a, b), f"{name} differs from the unfused FP8 path"

    xd = x.double().requires_grad_(True)
    wd = [layer.weight.detach().double().requires_grad_(True) for layer in layers]
    g, u = xd @ wd[0].T, xd @ wd[1].T
    y = (g * torch.sigmoid(g) * u) @ wd[2].T
    y.backward(dy.double())
    exact = [y.detach(), xd.grad] + [w.grad for w in wd]
    metrics = {name: fp8_swiglu_mlp_calc_diff(a, b) for name, a, b in zip(names, fused, exact)}
    record(case["id"], "fp8_swiglu_mlp", **metrics)
    for name, value in metrics.items():
        assert value <= FP8_SWIGLU_MLP_CALC_DIFF, (
            f"{name}: fp8_swiglu_mlp_calc_diff {value:.3g} against float64"
        )


def dequant_rowwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.repeat_interleave(BLOCK, dim=-1)[..., : q.shape[-1]]


def dequant_blockwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    R, C = q.shape[-2:]
    full = scale.repeat_interleave(BLOCK, dim=-2).repeat_interleave(BLOCK, dim=-1)
    return q.float() * full[..., :R, :C]
