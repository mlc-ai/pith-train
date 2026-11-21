"""
The Gated DeltaNet's causal depthwise conv at every training shape of
tools/kernel_cases/kernel_cases.json (family causal_conv1d): causal_conv1d_fwd and
causal_conv1d_bwd on the (B = mbs, T_local, C) mixed q / k / v channels (C = 8192 for
Qwen3.5-35B-A3B, 10240 for Qwen3.5-27B), width 4, bf16 x and weight, called as
pithtrain/operators/causal_conv1d.py and Qwen35MoeGatedDeltaNet.block call them:

- cp == 1: offset 0, the conv's zero padding at the sequence start;
- cp > 1: offset = width - 1, x led by the width - 1 tokens prepend_conv_state fetches from the
  previous rank (random history here, as a middle rank holds), whose input gradient the kernel
  returns too.

Reference: the convolution as its definition in float64, y[b, t, c] = sum_k w[c, k] x[b, t + offset
- (K - 1) + k, c] (taps at negative positions zero), dx and dw its exact adjoints, each with the sum
of its terms' magnitudes.

Bounds, elementwise: y and dx are K-term fp32 sums rounded once to bf16, so |got - ref| <= 2^-8
|ref| (half a bf16 ulp, relative) + K 2^-24 sum|terms| (fp32 accumulation); dw is a B T-term fp32
reduction rounded once to the weight's bf16, so |got - ref| <= 2^-8 |ref| + 2^-16 sum|terms|: fp32
blocked reductions of up to 131072 terms stay far inside 2^-16 of the magnitude sum. Largest error / bound
measured on the H200 over the 95 cases: y 0.996, dx 0.996, dw 0.932, the rounding term dominating
(the outputs are correctly rounded fp32 sums); each case prints its ratios on a MEASURED line.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import causal_conv1d_bwd, causal_conv1d_fwd
from tests.spec.cases import spec_params
from tests.spec.references import worst_ratio

BF16_HALF_ULP = 2.0**-8
FP32_EPS = 2.0**-24
DW_ACCUMULATION = 2.0**-16


def make(case: dict) -> tuple[torch.Tensor, ...]:
    axes = case["axes"]
    B, S, C, K = axes["B"], axes["T_local"], axes["C"], axes["width"]
    offset = K - 1 if axes["cp"] > 1 else 0
    gen = torch.Generator(device="cuda").manual_seed(int(case["id"].rsplit("/", 1)[1]) + S)
    x = torch.randn(B, offset + S, C, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(C, 1, K, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    grad_out = torch.randn(B, S, C, generator=gen, device="cuda").to(torch.bfloat16)
    return x, weight, grad_out, offset


def reference(x, weight, grad_out, offset):
    """
    float64 (y, |y| terms, dx, |dx| terms, dw, |dw| terms) of the definition, channel-blocked so a
    32k x 10240 sequence stays within a few GB.
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
            # dx[t + offset - (K - 1) + k] += w[k] g[t]: rows of xp shifted back by K - 1
            contrib = torch.zeros_like(xp)
            contrib[:, offset + k : offset + k + S] = w[ch, k] * g
            dx[..., ch] += contrib[:, K - 1 :]
            dx_abs[..., ch] += contrib[:, K - 1 :].abs()
            dw[ch, k] = (g * tap).sum((0, 1))
            dw_abs[ch, k] = (g * tap).abs().sum((0, 1))
    return y, y_abs, dx, dx_abs, dw.view(weight.shape), dw_abs.view(weight.shape)


@pytest.mark.parametrize("case", spec_params("causal_conv1d"))
def test_causal_conv1d_at_training_shapes(case: dict) -> None:
    x, weight, grad_out, offset = make(case)
    K = weight.shape[-1]
    y = causal_conv1d_fwd(x, weight, offset)
    dx, dw = causal_conv1d_bwd(grad_out, x, weight, offset)
    assert y.shape == grad_out.shape and y.dtype == torch.bfloat16
    assert dx.shape == x.shape and dw.shape == weight.shape and dw.dtype == weight.dtype
    y_ref, y_abs, dx_ref, dx_abs, dw_ref, dw_abs = reference(x, weight, grad_out, offset)
    ratios = {
        "y": worst_ratio(y, y_ref, BF16_HALF_ULP * y_ref.abs() + K * FP32_EPS * y_abs),
        "dx": worst_ratio(dx, dx_ref, BF16_HALF_ULP * dx_ref.abs() + K * FP32_EPS * dx_abs),
        "dw": worst_ratio(dw, dw_ref, BF16_HALF_ULP * dw_ref.abs() + DW_ACCUMULATION * dw_abs),
    }
    print(f"MEASURED {case['id']} " + " ".join(f"{n}={r:.3f}" for n, r in ratios.items()))
    assert all(torch.isfinite(t).all() for t in (y, dx, dw))
    bad = {name: ratio for name, ratio in ratios.items() if ratio > 1.0}
    assert not bad, f"error / bound above 1: {bad}"
