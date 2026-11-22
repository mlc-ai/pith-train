"""
clamped_swiglu_fwd / _bwd at the clamped_swiglu cases, against float64.
"""

import pytest
import torch

from pithtrain.kernels import (
    clamped_swiglu_bwd,
    clamped_swiglu_fwd,
)
from tests.kernels.cases import params, seed_of
from tests.kernels.common import (
    ALPHA,
    F64,
    LIMIT,
    REALISTIC,
    ROUTING_CHUNK,
    bits,
    grouped_rows,
    half_ulp_bf16,
    record,
)


@pytest.fixture(autouse=True)
def empty_cache():
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()


REL = 2.0**-18  # clamped SwiGLU fp32 error, about 3x the worst measured (2^-19.8)


@pytest.mark.parametrize("case", params("clamped_swiglu", keep=lambda c: REALISTIC(c) and c["axes"]["act"] == "clamped_swiglu" and not c["axes"]["bf16_fused"]))  # fmt: skip
def test_clamped_swiglu(case: dict) -> None:
    a = case["axes"]
    assert (a["swiglu_alpha"], a["swiglu_limit"]) == (ALPHA, LIMIT)
    inter, counts = a["inter"], a["group_sizes"]
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    gate_up = grouped_rows(counts, 2 * inter, gen, scale=4.0)  # reaches the clamps (7.0 is exact)
    dh = grouped_rows(counts, inter, gen)
    M = gate_up.shape[0]
    assert M == a["fp8_rows"]
    out = clamped_swiglu_fwd(gate_up, ALPHA, LIMIT)
    grad = clamped_swiglu_bwd(dh, gate_up, ALPHA, LIMIT)
    assert out.shape == (M, inter) and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert grad.shape == (M, 2 * inter) and grad.dtype == torch.bfloat16
    worst = dict(fwd=0.0, dgate=0.0, dup=0.0)
    needed = dict(fwd=0.0, dgate=0.0, dup=0.0)  # the smallest REL that would pass, for calibration

    def check(name, got, ref, scale):
        err, half = (got.to(F64) - ref).abs(), half_ulp_bf16(ref)
        worst[name] = max(worst[name], float((err / (half + 2 * REL * scale)).max()))
        needed[name] = max(needed[name], float(((err - half).clamp(min=0) / (2 * scale + 1e-300)).max()))  # fmt: skip

    for start in range(0, M, ROUTING_CHUNK // 2):
        end = min(start + ROUTING_CHUNK // 2, M)
        g, u, d = gate_up[start:end, 0::2].to(F64), gate_up[start:end, 1::2].to(F64), dh[start:end].to(F64)  # fmt: skip
        gc, uc = g.clamp(max=LIMIT), u.clamp(-LIMIT, LIMIT)
        sig = torch.sigmoid(ALPHA * gc)
        h = (uc + 1) * gc * sig
        check("fwd", out[start:end], h, h.abs())
        dgate = d * (uc + 1) * sig * (1 + ALPHA * gc * (1 - sig)) * (g <= LIMIT)
        dup = d * gc * sig * (u.abs() <= LIMIT)
        check("dgate", grad[start:end, 0::2], dgate, d.abs() * (uc + 1).abs() * sig * (1 + ALPHA * gc.abs()))  # fmt: skip
        check("dup", grad[start:end, 1::2], dup, dup.abs())
        assert not bool(bits(grad[start:end, 0::2][g > LIMIT]).any()), (
            "gate gradient past the limit"
        )
        assert not bool(bits(grad[start:end, 1::2][u.abs() > LIMIT]).any()), "up gradient past the limit"  # fmt: skip
    for name, value in worst.items():
        record(case["id"], f"clamped_swiglu {name}", ratio=value, rel_needed=needed[name])
    assert bool((gate_up[:, 0::2].float() > LIMIT).any()), "the clamp must be exercised"
    for name, value in worst.items():
        assert value <= 1, f"clamped_swiglu {name}: error {value:.3g}x the bound"
