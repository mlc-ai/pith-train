"""
causal_conv1d_fwd / _bwd at the causal_conv1d cases, within one bf16 rounding plus fp32 slack.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import causal_conv1d_bwd, causal_conv1d_empty, causal_conv1d_fwd
from tests.kernels.cases import params, seed_of

BF16_HALF_ULP = 2.0**-8
FP32_EPS = 2.0**-24
DW_ACCUMULATION = 2.0**-16


def causal_conv1d_inputs(case: dict) -> tuple[torch.Tensor, ...]:
    axes = case["axes"]
    B, S, C, K = axes["B"], axes["T_local"], axes["C"], axes["width"]
    offset = K - 1 if axes["cp"] > 1 else 0
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    x = torch.randn(B, offset + S, C, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(C, 1, K, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    grad_out = torch.randn(B, S, C, generator=gen, device="cuda").to(torch.bfloat16)
    return x, weight, grad_out, offset


def causal_conv1d_reference(x, weight, grad_out, offset):
    """
    float64 y, dx, dw of the definition, each with the sum of its terms' magnitudes.
    """
    B, L, C = x.shape
    K, S = weight.shape[-1], L - offset
    w = weight.double().view(C, K)
    out = [torch.empty(B, S, C, dtype=torch.float64, device=x.device) for _ in range(2)]
    out += [torch.zeros(B, L, C, dtype=torch.float64, device=x.device) for _ in range(2)]
    out += [torch.empty(C, K, dtype=torch.float64, device=x.device) for _ in range(2)]
    y, y_abs, dx, dx_abs, dw, dw_abs = out
    for c0 in range(0, C, 2048):
        ch = slice(c0, c0 + 2048)
        xp = F.pad(x[..., ch].double(), (0, 0, K - 1, 0))  # x[t'] at xp[t' + K - 1]
        g = grad_out[..., ch].double()
        y[..., ch], y_abs[..., ch] = 0, 0
        for k in range(K):
            tap = xp[:, offset + k : offset + k + S]  # x[t + offset - (K - 1) + k]
            y[..., ch] += w[ch, k] * tap
            y_abs[..., ch] += (w[ch, k] * tap).abs()
            # dx[t + offset - (K - 1) + k] += w[k] g[t]
            contrib = torch.zeros_like(xp)
            contrib[:, offset + k : offset + k + S] = w[ch, k] * g
            dx[..., ch] += contrib[:, K - 1 :]
            dx_abs[..., ch] += contrib[:, K - 1 :].abs()
            dw[ch, k] = (g * tap).sum((0, 1))
            dw_abs[ch, k] = (g * tap).abs().sum((0, 1))
    return y, y_abs, dx, dx_abs, dw.view(weight.shape), dw_abs.view(weight.shape)


@pytest.mark.parametrize("case", params("causal_conv1d"))
def test_causal_conv1d_at_training_shapes(case: dict) -> None:
    x, weight, grad_out, offset = causal_conv1d_inputs(case)
    K = weight.shape[-1]
    y = causal_conv1d_fwd(x, weight, offset)
    dx, dw = causal_conv1d_bwd(grad_out, x, weight, offset)
    assert y.shape == grad_out.shape and y.dtype == torch.bfloat16
    fake = causal_conv1d_empty(x.to("meta"), offset)  # the operator's fake implementation
    assert (fake.shape, fake.stride(), fake.dtype) == (y.shape, y.stride(), y.dtype)
    assert dx.shape == x.shape and dw.shape == weight.shape and dw.dtype == weight.dtype
    y_ref, y_abs, dx_ref, dx_abs, dw_ref, dw_abs = causal_conv1d_reference(
        x, weight, grad_out, offset
    )
    ratios = {
        "y": worst_ratio(y, y_ref, BF16_HALF_ULP * y_ref.abs() + K * FP32_EPS * y_abs),
        "dx": worst_ratio(dx, dx_ref, BF16_HALF_ULP * dx_ref.abs() + K * FP32_EPS * dx_abs),
        "dw": worst_ratio(dw, dw_ref, BF16_HALF_ULP * dw_ref.abs() + DW_ACCUMULATION * dw_abs),
    }
    print(f"MEASURED {case['id']} " + " ".join(f"{n}={r:.3f}" for n, r in ratios.items()))
    assert all(torch.isfinite(t).all() for t in (y, dx, dw))
    bad = {name: ratio for name, ratio in ratios.items() if ratio > 1.0}
    assert not bad, f"error / bound above 1: {bad}"


def worst_ratio(out: torch.Tensor, ref: torch.Tensor, tol: torch.Tensor) -> float:
    """
    max |out - ref| / tol elementwise (<= 1 passes); tol broadcasts against ref.
    """
    if out.numel() == 0:
        return 0.0
    err = (out.double() - ref.double()).abs()
    return float((err / (tol.double() + 1e-30)).max())
