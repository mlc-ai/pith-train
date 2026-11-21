"""
The FP8 dense linear at every training shape of the spec (family dense_gemm: M = tokens per GPU, N
and K of each training.Linear), called as FP8Linear calls it (pithtrain/operators/linear.py):

    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(w)
    y, xt_fp8, xt_scale = fp8_gemm_fwd(x, w_fp8, w_scale)           x quantized by fp8_quant_act
    dx, dw = fp8_gemm_bwd(dy, wt_fp8, wt_scale, xt_fp8, xt_scale, K)  dy quantized by fp8_quant_grad

Checked against references written here (tests/spec/fp8_reference.py), not against other kernels:

- the quantized operands, bit-exact: fp8_quant_weight's w (128x128) and w.T, fp8_quant_act's x
  (1x128) and x.T (128x128), fp8_quant_grad's dy and dy.T (1x128);
- each GEMM output against the fp32 product of the exactly dequantized FP8 operands (the GEMM's
  own input), elementwise: |out - ref| <= half a bf16 ulp of ref + ACCUM * (|a| @ |b|)_ij, the
  bf16 rounding of the result plus the accumulation error of FP8 tensor-core GEMMs, proportional
  to the sum of absolute products; and calc_diff(out, ref) <= EXACT_CALC_DIFF;
- each GEMM output against the fp32 product of the unquantized bf16 operands (what the layer
  approximates): calc_diff <= BF16_CALC_DIFF, the error of quantizing both operands to e4m3.

Bounds, about 3x the largest errors measured on the H200 over every smoke and spec case (seed 0,
recorded with PITHTRAIN_SPEC_ERRORS): see ACCUM, EXACT_CALC_DIFF and BF16_CALC_DIFF below.

Weight-gradient precision: fp8_gemm_bwd returns weight_grad in bf16 (the dtype of dy), and FP8Linear
hands it to autograd as the bf16 weight's gradient. Unlike BF16Linear, GroupedLinear and
FP8GroupedLinear, FP8Linear gets no fp32 main_grad from install_wgrad_accumulation
(modules/wgrad_accumulation.py), so every micro-batch's dense FP8 weight gradient is rounded to bf16
before the fp32 sum across micro-batches: a relative L2 error of 1.7e-3 per micro-batch term
(calc_diff 1.39e-6, all of it the bf16 rounding; the GEMM's own accumulation error is an order
smaller). test_wgrad_bf16_rounding pins that cost down: the accumulated gradient stays within the
sum of the per-term bf16 roundings (measured at most 0.41 of it).
"""

import pytest
import torch

from pithtrain.kernels import (
    fp8_gemm_bwd,
    fp8_gemm_fwd,
    fp8_quant_act,
    fp8_quant_grad,
    fp8_quant_weight,
)
from tests.spec.cases import spec_params
from tests.spec.fp8_reference import (
    assert_fp8_equal,
    assert_scale_equal,
    dequant_blockwise,
    dequant_rowwise,
    half_ulp_bf16,
    quant_blockwise,
    quant_rowwise,
)
from tests.spec.measure import record

DEV = "cuda"
ROWS_PER_CHUNK = 4096

ACCUM = 3e-4
"""
Accumulation error per unit of sum |a_k b_k|, beyond the output's bf16 rounding; measured at most
9.9e-5 (y), 9.1e-5 (dx), 3.1e-5 (dw) on the H200.
"""
EXACT_CALC_DIFF = 3e-6
"""
calc_diff against the exact-dequant product: the bf16 rounding of the output alone gives 1.39e-6
(measured 1.39e-6 at every shape, so the accumulation adds nothing visible); about twice that.
"""
BF16_CALC_DIFF = 2e-3
"""
calc_diff against the bf16 product: e4m3 keeps 3 mantissa bits (relative error up to 2^-4, about
2.6% rms for normal inputs), so each operand contributes about 3.4e-4, the product about 7e-4
(measured at most 7.15e-4).
"""


def make(shape: tuple[int, int], seed: int, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=gen) * scale).to(torch.bfloat16)


class Stats:
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


def compare(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, with_abs: bool) -> Stats:
    """
    out (m, n) against a @ b.T in fp32 (a (m, k), b (n, k) fp32), in row chunks.
    """
    stats = Stats()
    for start in range(0, out.shape[0], ROWS_PER_CHUNK):
        rows = slice(start, start + ROWS_PER_CHUNK)
        ref = a[rows] @ b.T
        absref = a[rows].abs() @ b.abs().T if with_abs else None
        stats.add(out[rows], ref, absref)
    return stats


def check(case_id: str, name: str, out: torch.Tensor, exact: tuple, truth: tuple) -> None:
    exact_stats = compare(out, *exact, with_abs=True)
    truth_stats = compare(out, *truth, with_abs=False)
    record(case_id, name, calc_diff=exact_stats.calc_diff, accum=exact_stats.excess, bf16_calc_diff=truth_stats.calc_diff)  # fmt: skip
    assert exact_stats.excess <= ACCUM, f"{name}: elementwise excess {exact_stats.excess:.3g} > {ACCUM:.3g} x sum|ab|"  # fmt: skip
    assert exact_stats.calc_diff <= EXACT_CALC_DIFF, f"{name}: calc_diff {exact_stats.calc_diff:.3g} vs exact dequant"  # fmt: skip
    assert truth_stats.calc_diff <= BF16_CALC_DIFF, f"{name}: calc_diff {truth_stats.calc_diff:.3g} vs bf16 product"  # fmt: skip


@pytest.fixture(autouse=True)
def no_tf32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("case", spec_params("dense_gemm"))
def test_fp8_linear(case: dict) -> None:
    axes = case["axes"]
    M, N, K = axes["M"], axes["N"], axes["K"]
    x, w, dy = make((M, K), 1), make((N, K), 2, 0.02), make((M, N), 3)

    # weight: 128x128 tiles of w and of w.T
    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(w)
    ref_w, ref_w_scale = quant_blockwise(w)
    assert_fp8_equal(w_fp8, ref_w, "w_fp8")
    assert_scale_equal(w_scale, ref_w_scale, "w_scale")
    assert_fp8_equal(wt_fp8, ref_w.T.contiguous(), "wt_fp8")
    assert_scale_equal(wt_scale, ref_w_scale.T.contiguous(), "wt_scale")

    # activation: 1x128 rows of x (checked through fp8_quant_act, which fp8_gemm_fwd calls) and
    # 128x128 tiles of x.T (returned by fp8_gemm_fwd)
    x_fp8, x_scale, _, _ = fp8_quant_act(x)
    ref_x, ref_x_scale = quant_rowwise(x)
    assert_fp8_equal(x_fp8, ref_x, "x_fp8")
    assert_scale_equal(x_scale, ref_x_scale, "x_scale")
    del x_fp8, x_scale

    y, xt_fp8, xt_scale = fp8_gemm_fwd(x, w_fp8, w_scale)
    assert y.shape == (M, N) and y.dtype == torch.bfloat16
    ref_xt, ref_xt_scale = quant_blockwise(x.T)
    assert_fp8_equal(xt_fp8, ref_xt, "xt_fp8")
    assert_scale_equal(xt_scale, ref_xt_scale, "xt_scale")
    w_dq = dequant_blockwise(ref_w, ref_w_scale)
    check(case["id"], "y", y, (dequant_rowwise(ref_x, ref_x_scale), w_dq), (x.float(), w.float()))
    del y

    # gradient: 1x128 rows of dy and of dy.T
    dy_fp8, dy_scale, dyt_fp8, dyt_scale = fp8_quant_grad(dy)
    ref_dy, ref_dy_scale = quant_rowwise(dy)
    ref_dyt, ref_dyt_scale = quant_rowwise(dy.T)
    assert_fp8_equal(dy_fp8, ref_dy, "dy_fp8")
    assert_scale_equal(dy_scale, ref_dy_scale, "dy_scale")
    assert_fp8_equal(dyt_fp8, ref_dyt, "dyt_fp8")
    assert_scale_equal(dyt_scale, ref_dyt_scale, "dyt_scale")
    del dy_fp8, dy_scale, dyt_fp8, dyt_scale

    dx, dw = fp8_gemm_bwd(dy, wt_fp8, wt_scale, xt_fp8, xt_scale, K)
    assert dx.shape == (M, K) and dx.dtype == torch.bfloat16
    assert dw.shape == (N, K) and dw.dtype == torch.bfloat16
    dy_dq = dequant_rowwise(ref_dy, ref_dy_scale)
    check(case["id"], "dx", dx, (dy_dq, w_dq.T.contiguous()), (dy.float(), w.float().T.contiguous()))  # fmt: skip
    del dx, dy_dq, w_dq
    xt_dq = dequant_blockwise(ref_xt, ref_xt_scale)
    check(case["id"], "dw", dw, (dequant_rowwise(ref_dyt, ref_dyt_scale), xt_dq), (dy.float().T.contiguous(), x.float().T.contiguous()))  # fmt: skip


WGRAD_CASES = [c for c in spec_params("dense_gemm") if c.values[0]["tier"] == "smoke"]


@pytest.mark.parametrize("case", WGRAD_CASES)
def test_wgrad_bf16_rounding(case: dict) -> None:
    """
    Accumulating MICROBATCHES micro-batches of weight gradients in fp32 from fp8_gemm_bwd's bf16
    weight_grad (training today) against accumulating the exact-dequant fp32 products: the error
    must stay within the bf16 rounding of each term (half an ulp of each partial product, summed),
    i.e. the bf16 hand-off loses precision but does not bias or blow up. Records the relative L2
    error of the accumulated gradient so the cost is visible.
    """
    microbatches = 4
    axes = case["axes"]
    M, N, K = min(axes["M"], 4096), axes["N"], axes["K"]
    w = make((N, K), 2, 0.02)
    _, _, wt_fp8, wt_scale = fp8_quant_weight(w)
    main_grad = torch.zeros(N, K, dtype=torch.float32, device=DEV)
    exact = torch.zeros(N, K, dtype=torch.float64, device=DEV)
    bound = torch.zeros(N, K, dtype=torch.float64, device=DEV)
    for step in range(microbatches):
        x, dy = make((M, K), 10 + step), make((M, N), 20 + step)
        _, xt_fp8, xt_scale = fp8_gemm_fwd(x, *quant_weight_pair(w))
        _, dw = fp8_gemm_bwd(dy, wt_fp8, wt_scale, xt_fp8, xt_scale, K)
        main_grad += dw.float()
        ref_dyt, ref_dyt_scale = quant_rowwise(dy.T)
        term = dequant_rowwise(ref_dyt, ref_dyt_scale).double() @ dequant_blockwise(xt_fp8, xt_scale).double().T  # fmt: skip
        absterm = dequant_rowwise(ref_dyt, ref_dyt_scale).abs().double() @ dequant_blockwise(xt_fp8, xt_scale).abs().double().T  # fmt: skip
        exact += term
        bound += half_ulp_bf16(term) + ACCUM * absterm
    err = (main_grad.double() - exact).abs()
    rel = float((main_grad.double() - exact).norm() / exact.norm())
    record(case["id"], "wgrad_accumulated", rel_l2=rel, worst=float((err / (bound + 1e-30)).max()))
    assert bool((err <= bound * 1.0001 + 1e-30).all()), f"accumulated wgrad error {float((err / bound).max()):.3g}x the per-term bf16 bound"  # fmt: skip


def quant_weight_pair(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w_fp8, w_scale, _, _ = fp8_quant_weight(w)
    return w_fp8, w_scale
