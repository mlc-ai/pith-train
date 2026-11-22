"""
FlashAttention-4 at every attention launch of the spec (dense, varlen and ring-step calls), forward
and backward, against fp32 attention in plain PyTorch. Relative L2 error (max abs error for lse)
within BOUNDS, about 3x measured on the H200.
"""

import json
import os

import pytest
import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from tests.kernels.attention_reference import attention_reference
from tests.kernels.cases import params, seed_of
from tests.kernels.references import rel_l2

BOUNDS = {
    "out": 7e-3,  # measured 2.31e-3
    "lse": 9e-6,  # max abs error, measured 2.86e-6
    "dq": 7.5e-3,  # measured 2.49e-3
    "dk": 7.5e-3,  # measured 2.48e-3
    "dv": 7.5e-3,  # measured 2.37e-3
    "dsink": 1e-2,  # measured 3.03e-3
}
LSE_SHIFT = 0.5
ERROR_LOG = os.environ.get("PITHTRAIN_SPEC_ERROR_LOG")


def randn(gen: torch.Generator, *shape: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32).to(dtype)


def check(case: dict, got: dict, want: dict) -> None:
    """
    Logs each output's error against want, then asserts the bounds.
    """
    errors = {}
    for name, value in got.items():
        value = value.detach()
        assert torch.isfinite(value).all(), f"{name} has non-finite values"
        ref = want[name]
        if name == "lse":
            errors[name] = float((value.float() - ref.float()).abs().max())
        else:
            errors[name] = rel_l2(value, ref)
    if ERROR_LOG:
        with open(ERROR_LOG, "a") as f:
            f.write(json.dumps(dict(id=case["id"], **errors)) + "\n")
    failed = {name: err for name, err in errors.items() if err > BOUNDS[name]}
    assert not failed, f"errors above BOUNDS: {failed} (all: {errors})"


def run_dense(case: dict, gen: torch.Generator) -> None:
    a = case["axes"]
    B, S, Hq, Hkv, D, Dv = a["B"], a["Sq"], a["Hq"], a["Hkv"], a["D"], a["Dv"]
    scale = D**-0.5
    window = (a["window"], 0) if a["window"] is not None else (None, None)
    q = randn(gen, B, S, Hq, D).requires_grad_(True)
    k = randn(gen, B, S, Hkv, D).requires_grad_(True)
    v = randn(gen, B, S, Hkv, Dv).requires_grad_(True)
    dout = randn(gen, B, S, Hq, Dv)
    sink = randn(gen, Hq).requires_grad_(True) if a["sinks"] else None
    out = flash_attn_func(q, k, v, scale, causal=True, window_size=window, learnable_sink=sink)
    out.backward(dout)
    got = dict(out=out, dq=q.grad, dk=k.grad, dv=v.grad)
    if sink is not None:
        got["dsink"] = sink.grad
    want = {name: [] for name in ("out", "dq", "dk", "dv")}
    dsink = 0
    for b in range(B):
        ref = attention_reference(q[b].detach(), k[b].detach(), v[b].detach(), scale, True, a["window"], sink.detach() if sink is not None else None, dout[b])  # fmt: skip
        for name in want:
            want[name].append(ref[name])
        if sink is not None:
            dsink = dsink + ref["dsink"]
    want = {name: torch.stack(value) for name, value in want.items()}
    want["dsink"] = dsink
    check(case, got, want)


def run_varlen(case: dict, gen: torch.Generator) -> None:
    a = case["axes"]
    T, Hq, Hkv, D, Dv = a["Sq"], a["Hq"], a["Hkv"], a["D"], a["Dv"]
    scale = D**-0.5
    window = (a["window"], 0) if a["window"] is not None else (None, None)
    cu = a["cu_seqlens"]
    cu_t = torch.tensor(cu, dtype=torch.int32, device="cuda")
    q = randn(gen, T, Hq, D).requires_grad_(True)
    k = randn(gen, T, Hkv, D).requires_grad_(True)
    v = randn(gen, T, Hkv, Dv).requires_grad_(True)
    dout = randn(gen, T, Hq, Dv)
    sink = randn(gen, Hq).requires_grad_(True) if a["sinks"] else None
    out = flash_attn_varlen_func(q, k, v, cu_t, T, scale, causal=True, window_size=window, learnable_sink=sink)  # fmt: skip
    out.backward(dout)
    got = dict(out=out, dq=q.grad, dk=k.grad, dv=v.grad)
    if sink is not None:
        got["dsink"] = sink.grad
    want = {name: [] for name in ("out", "dq", "dk", "dv")}
    dsink = 0
    for start, end in zip(cu, cu[1:]):
        ref = attention_reference(q[start:end].detach(), k[start:end].detach(), v[start:end].detach(), scale, True, a["window"], sink.detach() if sink is not None else None, dout[start:end])  # fmt: skip
        for name in want:
            want[name].append(ref[name])
        if sink is not None:
            dsink = dsink + ref["dsink"]
    want = {name: torch.cat(value) for name, value in want.items()}
    want["dsink"] = dsink
    check(case, got, want)


def run_ring_step(case: dict, gen: torch.Generator) -> None:
    """
    One zigzag ring step's FA4 forward and backward, called as the ring calls them.
    """
    a = case["axes"]
    kind, B, Hq, Hkv, D, Dv = a["kind"], a["B"], a["Hq"], a["Hkv"], a["D"], a["Dv"]
    scale = D**-0.5
    if kind == "ring_diag":
        S = a["Sq"]
        q, k, v = randn(gen, B, S, Hq, D), randn(gen, B, S, Hkv, D), randn(gen, B, S, Hkv, Dv)
        q_call, k_call, v_call, causal = q, k, v, True
    elif kind == "ring_front":
        S, block = a["Sq"], a["Skv"]
        q = randn(gen, B, S, Hq, D)
        k, v = randn(gen, B, 2 * block, Hkv, D), randn(gen, B, 2 * block, Hkv, Dv)
        q_call, k_call, v_call, causal = q, k[:, :block], v[:, :block], False
    else:
        block, S = a["Sq"], a["Skv"]
        q = randn(gen, B, 2 * block, Hq, D)
        k, v = randn(gen, B, S, Hkv, D), randn(gen, B, S, Hkv, Dv)
        q_call, k_call, v_call, causal = q[:, block:], k, v, False
    out, lse, *_ = _flash_attn_fwd(q_call, k_call, v_call, softmax_scale=scale, causal=causal, return_lse=True)  # fmt: skip
    dout = randn(gen, *out.shape)
    lse_in = lse + LSE_SHIFT
    if kind == "ring_back":  # contiguous copies, as the ring passes them
        dq, dk, dv = _flash_attn_bwd(q_call.contiguous(), k_call, v_call, out.contiguous(), dout.contiguous(), lse_in.contiguous(), softmax_scale=scale, causal=False)  # fmt: skip
    else:
        dq, dk, dv = _flash_attn_bwd(q_call, k_call, v_call, out, dout, lse_in, softmax_scale=scale, causal=causal)  # fmt: skip
    got = dict(out=out, lse=lse, dq=dq, dk=dk, dv=dv)
    want = {name: [] for name in got}
    for b in range(B):
        fwd = attention_reference(q_call[b], k_call[b], v_call[b], scale, causal)
        bwd = attention_reference(q_call[b], k_call[b], v_call[b], scale, causal, dout=dout[b], o_in=out[b], lse_in=lse_in[b])  # fmt: skip
        want["out"].append(fwd["out"])
        want["lse"].append(fwd["lse"])
        for name in ("dq", "dk", "dv"):
            want[name].append(bwd[name])
    check(case, got, {name: torch.stack(value) for name, value in want.items()})


@pytest.mark.parametrize("case", params("attention"))
def test_attention(case: dict) -> None:
    gen = torch.Generator(device="cuda").manual_seed(seed_of(case))
    kind = case["axes"]["kind"]
    if kind == "dense":
        run_dense(case, gen)
    elif kind == "varlen":
        run_varlen(case, gen)
    else:
        run_ring_step(case, gen)
