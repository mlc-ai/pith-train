import json
import os
from dataclasses import dataclass, fields
from itertools import accumulate

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from pithtrain.contexts import distributed
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from pithtrain.operators.ring_attention import (
    mla_ring_attention_func,
    ring_attention_func,
    ring_attention_varlen_func,
)
from tests.kernels.attention_reference import attention_reference
from tests.kernels.cases import params, seed_of
from tests.kernels.references import rel_l2
from tests.utilities import cosine_error, launch


@dataclass
class Request:
    B: int
    S: int
    HQ: int
    HK: int
    D: int
    atol: float = 1e-5


@dataclass
class Result:
    out: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


def extract_zigzag(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    chunks = x.chunk(2 * cp_size, dim=1)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]], dim=1).contiguous()


def record(req: Request) -> tuple[Result, Result]:
    cp_group = distributed.cp_group
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()
    softmax_scale = req.D**-0.5

    torch.manual_seed(42)
    q_full = torch.randn(req.B, req.S, req.HQ, req.D, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(req.B, req.S, req.HK, req.D, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(req.B, req.S, req.HK, req.D, device=device, dtype=torch.bfloat16)

    q_ref = q_full.clone().requires_grad_(True)
    k_ref = k_full.clone().requires_grad_(True)
    v_ref = v_full.clone().requires_grad_(True)
    out_ref = flash_attn_func(q_ref, k_ref, v_ref, softmax_scale, causal=True)
    out_ref.sum().backward()
    out_ref = extract_zigzag(out_ref, cp_rank, cp_size)
    dq_ref = extract_zigzag(q_ref.grad, cp_rank, cp_size)
    dk_ref = extract_zigzag(k_ref.grad, cp_rank, cp_size)
    dv_ref = extract_zigzag(v_ref.grad, cp_rank, cp_size)
    ref = Result(out_ref, dq_ref, dk_ref, dv_ref)

    q_imp = extract_zigzag(q_full, cp_rank, cp_size).clone().requires_grad_(True)
    k_imp = extract_zigzag(k_full, cp_rank, cp_size).clone().requires_grad_(True)
    v_imp = extract_zigzag(v_full, cp_rank, cp_size).clone().requires_grad_(True)
    out_imp = ring_attention_func(q_imp, k_imp, v_imp, softmax_scale, cp_group)
    out_imp.sum().backward()
    imp = Result(out_imp, q_imp.grad, k_imp.grad, v_imp.grad)

    return ref, imp


def verify(req: Request) -> None:
    ref, imp = record(req)
    for f in fields(ref):
        error = cosine_error(getattr(ref, f.name), getattr(imp, f.name))
        if error >= req.atol:
            raise AssertionError(f"{f.name} diverged: {error=:.2e} >= {req.atol=}")


REQUESTS = []
REQUESTS.append(pytest.param(2, Request(B=1, S=2048, HQ=4, HK=4, D=64), id="CP2-MHA-S2048"))
REQUESTS.append(pytest.param(2, Request(B=2, S=2048, HQ=8, HK=2, D=64), id="CP2-GQA-S2048"))
REQUESTS.append(pytest.param(4, Request(B=1, S=4096, HQ=12, HK=4, D=128), id="CP4-GQA-S4096"))


@pytest.mark.parametrize("cp_size,req", REQUESTS)
def test_ring_attention_vs_dense(cp_size: int, req: Request) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify, req)


@dataclass
class VarlenRequest:
    seqlens: tuple[int, ...]
    HQ: int
    HK: int
    D: int
    atol: float = 1e-5


def extract_zigzag_varlen(x: torch.Tensor, cu_seqlens: torch.Tensor, cp_rank: int, cp_size: int):
    bounds = cu_seqlens.tolist()
    docs = [x[start:end].unsqueeze(0) for start, end in zip(bounds[:-1], bounds[1:]) if end > start]
    return torch.cat([extract_zigzag(doc, cp_rank, cp_size).squeeze(0) for doc in docs])


def record_varlen(req: VarlenRequest) -> tuple[Result, Result]:
    cp_group = distributed.cp_group
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()
    softmax_scale = req.D**-0.5
    cu_seqlens = torch.tensor([0, *accumulate(req.seqlens)], dtype=torch.int32, device=device)
    T = sum(req.seqlens)

    torch.manual_seed(42)
    q_full = torch.randn(T, req.HQ, req.D, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(T, req.HK, req.D, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(T, req.HK, req.D, device=device, dtype=torch.bfloat16)
    grad_out = torch.randn(T, req.HQ, req.D, device=device, dtype=torch.bfloat16)

    def shard(x: torch.Tensor) -> torch.Tensor:
        return extract_zigzag_varlen(x, cu_seqlens, cp_rank, cp_size)

    q_ref = q_full.clone().requires_grad_(True)
    k_ref = k_full.clone().requires_grad_(True)
    v_ref = v_full.clone().requires_grad_(True)
    out_ref = flash_attn_varlen_func(q_ref, k_ref, v_ref, cu_seqlens, T, softmax_scale, causal=True)
    out_ref.backward(grad_out)
    ref = Result(shard(out_ref), shard(q_ref.grad), shard(k_ref.grad), shard(v_ref.grad))

    q_imp = shard(q_full).requires_grad_(True)
    k_imp = shard(k_full).requires_grad_(True)
    v_imp = shard(v_full).requires_grad_(True)
    out_imp = ring_attention_varlen_func(q_imp, k_imp, v_imp, cu_seqlens, softmax_scale, cp_group)
    out_imp.backward(shard(grad_out))
    imp = Result(out_imp, q_imp.grad, k_imp.grad, v_imp.grad)

    return ref, imp


def verify_varlen(req: VarlenRequest) -> None:
    ref, imp = record_varlen(req)
    for f in fields(ref):
        expected, actual = (getattr(result, f.name).double() for result in (ref, imp))
        error = float(
            (expected - actual).square().sum()
            / (expected.square() + actual.square()).sum().clamp_min(1e-12)
        )
        if not error < req.atol:
            raise AssertionError(f"{f.name} diverged: {error=:.2e} >= {req.atol=}")


# Cover single-token blocks, repeated boundaries and multi-tile documents. CP4 exercises
# front-step updates followed by back-block accumulation on the same rank; CP2 cannot.
VARLEN_REQUESTS = []
VARLEN_REQUESTS.append(pytest.param(2, VarlenRequest((1024,), HQ=4, HK=4, D=64), id="CP2-MHA-1doc"))
VARLEN_REQUESTS.append(
    pytest.param(2, VarlenRequest((4, 60, 512, 196, 252), HQ=8, HK=2, D=64), id="CP2-GQA-5docs")
)
VARLEN_REQUESTS.append(
    pytest.param(
        2, VarlenRequest((0, 4, 60, 0, 512, 196, 0, 0), HQ=8, HK=2, D=64), id="CP2-GQA-empty-docs"
    )
)
VARLEN_REQUESTS.append(
    pytest.param(4, VarlenRequest((8, 120, 1024, 392, 504), HQ=12, HK=4, D=128), id="CP4-GQA-5docs")
)
VARLEN_REQUESTS.append(
    pytest.param(4, VarlenRequest((8,) * 32, HQ=8, HK=2, D=64), id="CP4-GQA-32x8")
)


@pytest.mark.parametrize("cp_size,req", VARLEN_REQUESTS)
def test_ring_attention_varlen_vs_full(cp_size: int, req: VarlenRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_varlen, req)


@dataclass
class MLARequest:
    B: int
    S: int
    H: int
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    atol: float = 1e-3
    atol_weight: float = 1e-2
    # Relative L2 tolerance for the kv_b weight gradient. Cosine is scale-invariant and would hide a
    # uniform magnitude bug such as a missed cross-CP sum. The bf16 noise floor at the tested sizes
    # (S<=2048, H<=16, R=512) is about 5e-3.
    rtol_weight: float = 1e-2


@dataclass
class MLAResult:
    out: torch.Tensor
    dq_nope: torch.Tensor
    dq_pe: torch.Tensor
    d_normed_kv: torch.Tensor
    d_k_pe: torch.Tensor


def record_mla(req: MLARequest):
    cp_group = distributed.cp_group
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()
    H, R = req.H, req.kv_lora_rank
    nope, rope, vdim = req.qk_nope_head_dim, req.qk_rope_head_dim, req.v_head_dim
    softmax_scale = (nope + rope) ** -0.5
    out_features = H * (nope + vdim)

    torch.manual_seed(42)
    q_nope_full = torch.randn(req.B, req.S, H, nope, device=device, dtype=torch.bfloat16)
    q_pe_full = torch.randn(req.B, req.S, H, rope, device=device, dtype=torch.bfloat16)
    normed_kv_full = torch.randn(req.B, req.S, R, device=device, dtype=torch.bfloat16)
    k_pe_full = torch.randn(req.B, req.S, 1, rope, device=device, dtype=torch.bfloat16)
    w_full = torch.randn(out_features, R, device=device, dtype=torch.bfloat16) / (R**0.5)

    q_nope_r = q_nope_full.clone().requires_grad_(True)
    q_pe_r = q_pe_full.clone().requires_grad_(True)
    normed_kv_r = normed_kv_full.clone().requires_grad_(True)
    k_pe_r = k_pe_full.clone().requires_grad_(True)
    w_r = w_full.clone().requires_grad_(True)
    kv = F.linear(normed_kv_r, w_r).view(req.B, req.S, H, nope + vdim)
    k_nope, value = torch.split(kv, [nope, vdim], dim=-1)
    q_r = torch.cat([q_nope_r, q_pe_r], dim=-1)
    k_r = torch.cat([k_nope, k_pe_r.expand(-1, -1, H, -1)], dim=-1)
    out_r = flash_attn_func(
        q_r,
        k_r,
        value.contiguous(),
        softmax_scale=softmax_scale,
        causal=True,
    )
    out_r.sum().backward()
    ref = MLAResult(
        extract_zigzag(out_r, cp_rank, cp_size),
        extract_zigzag(q_nope_r.grad, cp_rank, cp_size),
        extract_zigzag(q_pe_r.grad, cp_rank, cp_size),
        extract_zigzag(normed_kv_r.grad, cp_rank, cp_size),
        extract_zigzag(k_pe_r.grad, cp_rank, cp_size),
    )
    dW_ref = w_r.grad

    q_nope_i = extract_zigzag(q_nope_full, cp_rank, cp_size).clone().requires_grad_(True)
    q_pe_i = extract_zigzag(q_pe_full, cp_rank, cp_size).clone().requires_grad_(True)
    normed_kv_i = extract_zigzag(normed_kv_full, cp_rank, cp_size).clone().requires_grad_(True)
    k_pe_i = extract_zigzag(k_pe_full, cp_rank, cp_size).clone().requires_grad_(True)
    w_i = w_full.clone().requires_grad_(True)
    out_i = mla_ring_attention_func(
        q_nope_i,
        q_pe_i,
        normed_kv_i,
        k_pe_i,
        w_i,
        sm_scale=softmax_scale,
        qk_nope_head_dim=nope,
        v_head_dim=vdim,
        cp_group=cp_group,
    )
    out_i.sum().backward()
    imp = MLAResult(out_i, q_nope_i.grad, q_pe_i.grad, normed_kv_i.grad, k_pe_i.grad)
    # kv_b weight grad is global: sum the per-rank partials across CP (FSDP does this in training).
    dW_imp = w_i.grad.clone()
    torch.distributed.all_reduce(dW_imp, group=cp_group)

    return ref, imp, dW_ref, dW_imp


def verify_mla(req: MLARequest) -> None:
    ref, imp, dW_ref, dW_imp = record_mla(req)
    for f in fields(ref):
        error = cosine_error(getattr(ref, f.name), getattr(imp, f.name))
        if error >= req.atol:
            raise AssertionError(f"{f.name} diverged: {error=:.2e} >= {req.atol=}")
    # Compare dW in fp32: both gradients reach the test as bf16 (autograd casts the op's fp32 dW to
    # the parameter's dtype), and norms and dot products in bf16 are imprecise at small magnitudes.
    dW_ref_f = dW_ref.to(torch.float32)
    dW_imp_f = dW_imp.to(torch.float32)
    werr = cosine_error(dW_ref_f, dW_imp_f)
    if werr >= req.atol_weight:
        raise AssertionError(f"kv_b_weight_grad diverged: {werr=:.2e} >= {req.atol_weight=}")
    rel = ((dW_imp_f - dW_ref_f).norm() / dW_ref_f.norm()).item()
    if rel >= req.rtol_weight:
        raise AssertionError(
            f"kv_b_weight_grad relative error too large: {rel=:.2e} >= {req.rtol_weight=}"
        )


MLA_REQUESTS = []
MLA_REQUESTS.append(pytest.param(2, MLARequest(B=1, S=2048, H=16), id="CP2-MLA-S2048"))
MLA_REQUESTS.append(pytest.param(4, MLARequest(B=1, S=2048, H=16), id="CP4-MLA-S2048"))


@pytest.mark.parametrize("cp_size,req", MLA_REQUESTS)
def test_mla_ring_attention_vs_dense(cp_size: int, req: MLARequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_mla, req)


def record_mla_fp8(req: MLARequest):
    """
    Like record_mla, but with kv_b in FP8. The reference decompresses the full-sequence latent with
    an FP8Linear, so its kv_b quantization error matches the implementation, which passes the same
    quantized kv_b weight into the ring as kv_b_quant. Comparing fp8 with fp8 isolates the in-ring
    math from the large e4m3 quantization error, so a tight tolerance is meaningful.
    """
    from pithtrain.operators.linear import FP8Linear

    cp_group = distributed.cp_group
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()
    H, R = req.H, req.kv_lora_rank
    nope, rope, vdim = req.qk_nope_head_dim, req.qk_rope_head_dim, req.v_head_dim
    softmax_scale = (nope + rope) ** -0.5
    out_features = H * (nope + vdim)

    torch.manual_seed(42)
    q_nope_full = torch.randn(req.B, req.S, H, nope, device=device, dtype=torch.bfloat16)
    q_pe_full = torch.randn(req.B, req.S, H, rope, device=device, dtype=torch.bfloat16)
    normed_kv_full = torch.randn(req.B, req.S, R, device=device, dtype=torch.bfloat16)
    k_pe_full = torch.randn(req.B, req.S, 1, rope, device=device, dtype=torch.bfloat16)
    w_full = torch.randn(out_features, R, device=device, dtype=torch.bfloat16) / (R**0.5)

    ref_lin = FP8Linear(R, out_features, bias=False).to(device).to(torch.bfloat16)
    ref_lin.weight.data.copy_(w_full)
    q_nope_r = q_nope_full.clone().requires_grad_(True)
    q_pe_r = q_pe_full.clone().requires_grad_(True)
    normed_kv_r = normed_kv_full.clone().requires_grad_(True)
    k_pe_r = k_pe_full.clone().requires_grad_(True)
    kv = ref_lin(normed_kv_r).view(req.B, req.S, H, nope + vdim)
    k_nope, value = torch.split(kv, [nope, vdim], dim=-1)
    q_r = torch.cat([q_nope_r, q_pe_r], dim=-1)
    k_r = torch.cat([k_nope, k_pe_r.expand(-1, -1, H, -1)], dim=-1)
    out_r = flash_attn_func(
        q_r,
        k_r,
        value.contiguous(),
        softmax_scale=softmax_scale,
        causal=True,
    )
    out_r.sum().backward()
    ref = MLAResult(
        extract_zigzag(out_r, cp_rank, cp_size),
        extract_zigzag(q_nope_r.grad, cp_rank, cp_size),
        extract_zigzag(q_pe_r.grad, cp_rank, cp_size),
        extract_zigzag(normed_kv_r.grad, cp_rank, cp_size),
        extract_zigzag(k_pe_r.grad, cp_rank, cp_size),
    )
    dW_ref = ref_lin.weight.grad

    imp_lin = FP8Linear(R, out_features, bias=False).to(device).to(torch.bfloat16)
    imp_lin.weight.data.copy_(w_full)
    kv_b_quant = imp_lin.get_quantized_weight()
    q_nope_i = extract_zigzag(q_nope_full, cp_rank, cp_size).clone().requires_grad_(True)
    q_pe_i = extract_zigzag(q_pe_full, cp_rank, cp_size).clone().requires_grad_(True)
    normed_kv_i = extract_zigzag(normed_kv_full, cp_rank, cp_size).clone().requires_grad_(True)
    k_pe_i = extract_zigzag(k_pe_full, cp_rank, cp_size).clone().requires_grad_(True)
    out_i = mla_ring_attention_func(
        q_nope_i,
        q_pe_i,
        normed_kv_i,
        k_pe_i,
        imp_lin.weight,
        sm_scale=softmax_scale,
        qk_nope_head_dim=nope,
        v_head_dim=vdim,
        cp_group=cp_group,
        kv_b_quant=kv_b_quant,
    )
    out_i.sum().backward()
    imp = MLAResult(out_i, q_nope_i.grad, q_pe_i.grad, normed_kv_i.grad, k_pe_i.grad)
    # kv_b weight grad is global: sum the per-rank partials across CP (FSDP does this in training).
    dW_imp = imp_lin.weight.grad.clone()
    torch.distributed.all_reduce(dW_imp, group=cp_group)

    return ref, imp, dW_ref, dW_imp


def _relerr(ref: torch.Tensor, imp: torch.Tensor) -> float:
    r = ref.to(torch.float32)
    return ((imp.to(torch.float32) - r).norm() / (r.norm() + 1e-12)).item()


def verify_mla_fp8(req: MLARequest) -> None:
    ref, imp, dW_ref, dW_imp = record_mla_fp8(req)
    # The fp8 reference differs only in the flash recompute order (ring vs dense) and in activation
    # quant blocking (per hop vs whole sequence). Cosine is scale-invariant and would hide a uniform
    # fp8 scale bug, so relative L2 error is checked too.
    atol_fp8 = 2e-2
    # Per-hop e4m3 block quant gives a relative error of a few percent, worst for dW (about 3.6e-2)
    # since it sums over all CP hops; a broken fp8 GEMM gives O(1).
    rtol_fp8 = 6e-2
    for f in fields(ref):
        a, b = getattr(ref, f.name), getattr(imp, f.name)
        cos, rel = cosine_error(a, b), _relerr(a, b)
        if cos >= atol_fp8 or rel >= rtol_fp8:
            raise AssertionError(
                f"[fp8] {f.name} diverged: cosine={cos:.2e} (>={atol_fp8}) relerr={rel:.2e} (>={rtol_fp8})"
            )
    cos, rel = cosine_error(dW_ref, dW_imp), _relerr(dW_ref, dW_imp)
    if cos >= atol_fp8 or rel >= rtol_fp8:
        raise AssertionError(
            f"[fp8] kv_b_weight_grad diverged: cosine={cos:.2e} (>={atol_fp8}) relerr={rel:.2e} (>={rtol_fp8})"
        )


@pytest.mark.parametrize("cp_size,req", MLA_REQUESTS)
def test_mla_ring_attention_fp8_vs_dense(cp_size: int, req: MLARequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_mla_fp8, req)


# The spec's CP cases (tests/kernels/cases.py, family ring_attention): every rank against fp32
# causal attention over the whole sequence, relative L2 error within BOUNDS (about 3x measured).

BOUNDS = {
    "out": 9e-3,  # measured 2.84e-3
    "dq": 9e-3,  # measured 2.97e-3
    "dk": 9e-3,  # measured 2.97e-3
    "dv": 9e-3,  # measured 2.95e-3
    "dq_nope": 1e-2,  # measured 3.11e-3
    "dq_pe": 1.15e-2,  # measured 3.69e-3
    "dnormed_kv": 1.05e-2,  # measured 3.40e-3
    "dk_pe": 1.25e-2,  # measured 4.06e-3
    "dkv_b": 8e-3,  # measured 2.56e-3
}
NOPE, ROPE, VDIM, LORA = 128, 64, 128, 512  # DeepSeek-V2-Lite's MLA dims


def zigzag_rows(rank: int, cp: int, seq: int, device) -> torch.Tensor:
    """
    The global indices of rank's zigzag tokens.
    """
    block = seq // (2 * cp)
    front = torch.arange(rank * block, (rank + 1) * block, device=device)
    back = torch.arange((2 * cp - 1 - rank) * block, (2 * cp - rank) * block, device=device)
    return torch.cat([front, back])


def randn(gen: torch.Generator, *shape: int, scale: float = 1.0) -> torch.Tensor:
    x = torch.randn(*shape, generator=gen, device=gen.device, dtype=torch.float32)
    return (x * scale).to(torch.bfloat16)


def check(case: dict, rank: int, got: dict, want: dict) -> None:
    errors = {}
    for name, value in got.items():
        value = value.detach()
        assert torch.isfinite(value).all(), f"rank {rank}: {name} has non-finite values"
        errors[name] = rel_l2(value, want[name])
    log = os.environ.get("PITHTRAIN_SPEC_ERROR_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(dict(id=case["id"], rank=rank, **errors)) + "\n")
    failed = {name: err for name, err in errors.items() if err > BOUNDS[name]}
    assert not failed, f"rank {rank}: errors above BOUNDS: {failed} (all: {errors})"


def gqa_worker(case: dict) -> None:
    a = case["axes"]
    group = distributed.cp_group
    rank, cp = group.rank(), group.size()
    B, seq, Hq, Hkv, D, Dv = a["B"], a["seq"], a["Hq"], a["Hkv"], a["D"], a["Dv"]
    scale = D**-0.5
    gen = torch.Generator(device=f"cuda:{torch.cuda.current_device()}").manual_seed(seed_of(case))
    q, k, v = randn(gen, B, seq, Hq, D), randn(gen, B, seq, Hkv, D), randn(gen, B, seq, Hkv, Dv)
    dout = randn(gen, B, seq, Hq, Dv)
    rows = zigzag_rows(rank, cp, seq, q.device)

    q_loc = q[:, rows].contiguous().requires_grad_(True)
    k_loc = k[:, rows].contiguous().requires_grad_(True)
    v_loc = v[:, rows].contiguous().requires_grad_(True)
    out = ring_attention_func(q_loc, k_loc, v_loc, scale, group)
    out.backward(dout[:, rows])
    got = dict(out=out, dq=q_loc.grad, dk=k_loc.grad, dv=v_loc.grad)

    want = {name: [] for name in got}
    for b in range(B):
        ref = attention_reference(q[b, rows], k[b], v[b], scale, True, dout=dout[b, rows], q_positions=rows)  # fmt: skip
        dk, dv = ref["dk"].contiguous(), ref["dv"].contiguous()
        dist.all_reduce(dk, group=group)
        dist.all_reduce(dv, group=group)
        want["out"].append(ref["out"])
        want["dq"].append(ref["dq"])
        want["dk"].append(dk[rows])
        want["dv"].append(dv[rows])
        del ref, dk, dv
    check(case, rank, got, {name: torch.stack(value) for name, value in want.items()})


def mla_worker(case: dict) -> None:
    a = case["axes"]
    group = distributed.cp_group
    rank, cp = group.rank(), group.size()
    B, seq, H = a["B"], a["seq"], a["Hq"]
    assert (a["D"], a["Dv"]) == (NOPE + ROPE, VDIM)
    scale = a["D"] ** -0.5
    gen = torch.Generator(device=f"cuda:{torch.cuda.current_device()}").manual_seed(seed_of(case))
    q_nope, q_pe = randn(gen, B, seq, H, NOPE), randn(gen, B, seq, H, ROPE)
    normed_kv, k_pe = randn(gen, B, seq, LORA), randn(gen, B, seq, 1, ROPE)
    weight = randn(gen, H * (NOPE + VDIM), LORA, scale=LORA**-0.5)
    dout = randn(gen, B, seq, H, VDIM)
    rows = zigzag_rows(rank, cp, seq, q_nope.device)

    leaves = [x[:, rows].contiguous().requires_grad_(True) for x in (q_nope, q_pe, normed_kv, k_pe)]  # fmt: skip
    w = weight.clone().requires_grad_(True)
    out = mla_ring_attention_func(*leaves, w, sm_scale=scale, qk_nope_head_dim=NOPE, v_head_dim=VDIM, cp_group=group)  # fmt: skip
    out.backward(dout[:, rows])
    dw = w.grad.float()
    dist.all_reduce(dw, group=group)
    got = dict(out=out, dq_nope=leaves[0].grad, dq_pe=leaves[1].grad, dnormed_kv=leaves[2].grad, dk_pe=leaves[3].grad, dkv_b=dw)  # fmt: skip

    w32 = weight.float()
    want = {name: [] for name in got if name != "dkv_b"}
    dw_ref = torch.zeros_like(w32)
    for b in range(B):
        kv = (normed_kv[b].float() @ w32.T).view(seq, H, NOPE + VDIM)
        k_full = torch.cat([kv[..., :NOPE], k_pe[b].float().expand(seq, H, ROPE)], dim=-1)
        q_rows = torch.cat([q_nope[b, rows], q_pe[b, rows]], dim=-1)
        ref = attention_reference(q_rows, k_full, kv[..., NOPE:], scale, True, dout=dout[b, rows], q_positions=rows)  # fmt: skip
        del kv, k_full
        dk, dv = ref["dk"].contiguous(), ref["dv"].contiguous()
        dist.all_reduce(dk, group=group)
        dist.all_reduce(dv, group=group)
        dkv = torch.cat([dk[..., :NOPE], dv], dim=-1).view(seq, H * (NOPE + VDIM))
        want["out"].append(ref["out"])
        want["dq_nope"].append(ref["dq"][..., :NOPE])
        want["dq_pe"].append(ref["dq"][..., NOPE:])
        want["dnormed_kv"].append((dkv[rows] @ w32))
        want["dk_pe"].append(dk[rows, :, NOPE:].sum(1, keepdim=True))
        dw_ref += dkv.T @ normed_kv[b].float()
        del ref, dk, dv, dkv
    want = {name: torch.stack(value) for name, value in want.items()}
    want["dkv_b"] = dw_ref
    check(case, rank, got, want)


def worker(case: dict) -> None:
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        (mla_worker if case["axes"]["mla"] else gqa_worker)(case)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def run_spec_case(case: dict) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = case["axes"]["cp"]
    launch(cfg, worker, case)


@pytest.mark.parametrize("case", params("ring_attention"))
def test_ring_attention_at_training_shapes(case: dict) -> None:
    run_spec_case(case)
