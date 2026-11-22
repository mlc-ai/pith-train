"""
silu_mul_fwd / _bwd at the silu_mul cases, within one bf16 rounding of float64.
"""

import pytest
import torch

from pithtrain.kernels import silu_mul_bwd, silu_mul_fwd
from tests.kernels.cases import params
from tests.kernels.common import DEV, half_ulp_bf16, record

SILU_MUL_ROWS_PER_CHUNK = 4096
SILU_MUL_SLACK = 2.0**-20  # a few fp32 ulps of the kernels' sigmoid and products


def silu_mul_ratio(out: torch.Tensor, exact: torch.Tensor, magnitude: torch.Tensor) -> float:
    err = (out.double() - exact).abs()
    return float((err / (half_ulp_bf16(exact) + SILU_MUL_SLACK * magnitude + 1e-300)).max())


@pytest.mark.parametrize("case", params("silu_mul"))
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
    for start in range(0, M, SILU_MUL_ROWS_PER_CHUNK):
        rows = slice(start, start + SILU_MUL_ROWS_PER_CHUNK)
        g, u, d = gate[rows].double(), up[rows].double(), dout[rows].double()
        s = torch.sigmoid(g)
        silu = g * s
        worst["out"] = max(worst["out"], silu_mul_ratio(out[rows], silu * u, (silu * u).abs()))
        dsilu = s * (1 + g * (1 - s))
        mag = (d * u).abs() * (s + (g * s).abs() * (1 + s))
        worst["dg"] = max(worst["dg"], silu_mul_ratio(dg[rows], d * u * dsilu, mag))
        worst["du"] = max(worst["du"], silu_mul_ratio(du[rows], d * silu, (d * silu).abs()))
    record(case["id"], "silu_mul", **worst)
    for name, value in worst.items():
        assert value <= 1.0, f"{name}: error {value:.3g}x one bf16 rounding"
