"""
silu_mul_fwd / silu_mul_bwd at every swiglu_dense shape, within one bf16 rounding of float64; and
the fused FP8 MLP (fp8_swiglu_mlp), bit-identical to the unfused path and close to float64.
"""

import pytest
import torch

from pithtrain.kernels import fp8_fused_experts_supported, silu_mul_bwd, silu_mul_fwd
from pithtrain.operators.fp8_swiglu_mlp import fp8_swiglu_mlp
from pithtrain.operators.linear import FP8Linear
from pithtrain.operators.silu_mul import silu_mul
from tests.kernels.cases import params
from tests.kernels.fp8_reference import half_ulp_bf16
from tests.kernels.measure import record

DEV = "cuda"
ROWS_PER_CHUNK = 4096
SLACK = 2.0**-20
"""
Relative fp32 error of the kernels' sigmoid and products (a few fp32 ulps).
"""


def ratio(out: torch.Tensor, exact: torch.Tensor, magnitude: torch.Tensor) -> float:
    err = (out.double() - exact).abs()
    return float((err / (half_ulp_bf16(exact) + SLACK * magnitude + 1e-300)).max())


@pytest.mark.parametrize("case", params("swiglu_dense"))
def test_silu_mul(case: dict) -> None:
    M, inter = case["axes"]["M"], case["axes"]["inter"]
    gen = torch.Generator(device=DEV).manual_seed(M + inter)
    gate = (torch.randn(M, inter, device=DEV, generator=gen) * 3).to(torch.bfloat16)
    up = torch.randn(M, inter, device=DEV, generator=gen).to(torch.bfloat16)
    dout = torch.randn(M, inter, device=DEV, generator=gen).to(torch.bfloat16)
    out = silu_mul_fwd(gate, up)
    dg, du = silu_mul_bwd(dout, gate, up)
    for t in (out, dg, du):
        assert t.shape == (M, inter) and t.dtype == torch.bfloat16 and t.is_contiguous()
    worst = dict(out=0.0, dg=0.0, du=0.0)
    for start in range(0, M, ROWS_PER_CHUNK):
        rows = slice(start, start + ROWS_PER_CHUNK)
        g, u, d = gate[rows].double(), up[rows].double(), dout[rows].double()
        s = torch.sigmoid(g)
        silu = g * s
        worst["out"] = max(worst["out"], ratio(out[rows], silu * u, (silu * u).abs()))
        dsilu = s * (1 + g * (1 - s))
        mag = (d * u).abs() * (s + (g * s).abs() * (1 + s))
        worst["dg"] = max(worst["dg"], ratio(dg[rows], d * u * dsilu, mag))
        worst["du"] = max(worst["du"], ratio(du[rows], d * silu, (d * silu).abs()))
    record(case["id"], "silu_mul", **worst)
    for name, value in worst.items():
        assert value <= 1.0, f"{name}: error {value:.3g}x one bf16 rounding"


FP8_CALC_DIFF = 7.5e-3
"""
calc_diff of the fused FP8 MLP's outputs against the float64 MLP.
"""


def calc_diff(got: torch.Tensor, exact: torch.Tensor) -> float:
    got = got.double()
    return float(1 - 2 * (got * exact).sum() / (got.square().sum() + exact.square().sum()))


@pytest.mark.parametrize("case", params("swiglu_dense", keep=lambda c: c["axes"]["fp8_swiglu_mlp"]))
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
    metrics = {name: calc_diff(a, b) for name, a, b in zip(names, fused, exact)}
    record(case["id"], "fp8_swiglu_mlp", **metrics)
    for name, value in metrics.items():
        assert value <= FP8_CALC_DIFF, f"{name}: calc_diff {value:.3g} against float64"
