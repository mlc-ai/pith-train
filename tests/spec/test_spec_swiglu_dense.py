"""
silu_mul_fwd / silu_mul_bwd (the eager SwiGLU of the dense and shared-expert MLPs) at every training
shape of the spec (family swiglu_dense: M = tokens per GPU, the MLP's intermediate width), against
float64 references written here:

    out = silu(g) * u,  dg = dout * u * s * (1 + g (1 - s)),  du = dout * silu(g),  s = sigmoid(g)

Each output is computed in fp32 and rounded once to bf16 (the contract), so it must lie within half
a bf16 ulp of the float64 value plus SLACK x the magnitude of its terms, the fp32 error of the
approximate sigmoid (ex2.approx, div.full) that can flip a rounding near a tie.
"""

import pytest
import torch

from pithtrain.kernels import silu_mul_bwd, silu_mul_fwd
from tests.spec.cases import spec_params
from tests.spec.fp8_reference import half_ulp_bf16
from tests.spec.measure import record

DEV = "cuda"
ROWS_PER_CHUNK = 4096
SLACK = 2.0**-20
"""
Relative fp32 error of the kernels' sigmoid and products (a few fp32 ulps).
"""


def ratio(out: torch.Tensor, exact: torch.Tensor, magnitude: torch.Tensor) -> float:
    err = (out.double() - exact).abs()
    return float((err / (half_ulp_bf16(exact) + SLACK * magnitude + 1e-300)).max())


@pytest.mark.parametrize("case", spec_params("swiglu_dense"))
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
