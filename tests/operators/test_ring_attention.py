"""Test the correctness of ring attention under context parallelism."""

from dataclasses import dataclass, fields

import pytest
import torch
import torch.nn.functional as F

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx
from pithtrain.operators.flash_attn_v4 import flash_attn_func, mla_flash_attn_func
from pithtrain.operators.ring_attention import mla_ring_attention_func, ring_attention_func
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
    """
    Extract this rank's zigzag-local slice along the sequence dim (dim=1).

    The global sequence is split into 2*cp_size equal chunks; rank r holds
    chunk r (front) concatenated with chunk 2*cp_size - r - 1 (mirror back).
    """
    chunks = x.chunk(2 * cp_size, dim=1)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]], dim=1).contiguous()


def record(ctx: DistributedCtx, req: Request) -> tuple[Result, Result]:
    """
    Record the forward output and the input gradients dQ, dK, dV for both the
    baseline and the implementation. The baseline is flash_attn_func run on the
    full sequence with no CP communication; the implementation is the zigzag
    ring_attention_func run on this rank's zigzag-local slice of Q/K/V with K/V
    rotated around the CP ring during the forward and backward passes.
    """
    cp_group = ctx.device_mesh.get_group("cp")
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


def verify(ctx: DistributedCtx, req: Request) -> None:
    ref, imp = record(ctx, req)
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


# ---------------------------------------------------------------------------
# MLA "pass the latent" ring attention.
# ---------------------------------------------------------------------------


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
    # Relative-error tolerance for the kv_b weight gradient. Cosine is scale-invariant
    # and would hide a uniform magnitude bug (e.g. a missed cross-CP sum), so we ALSO
    # require ||dW_imp - dW_ref|| / ||dW_ref|| < rtol_weight. Bf16-noise floor at the
    # sizes we test (S<=2048, H<=16, R=512) is ~5e-3.
    rtol_weight: float = 1e-2


@dataclass
class MLAResult:
    out: torch.Tensor
    dq_nope: torch.Tensor
    dq_pe: torch.Tensor
    d_normed_kv: torch.Tensor
    d_k_pe: torch.Tensor


def record_mla(ctx: DistributedCtx, req: MLARequest):
    """
    Reference: full-sequence MLA attention (decompress the latent via kv_b, then dense MLA
    flash attention) with no CP. Implementation: rotate the compressed latent around the
    zigzag ring and decompress on each rank. We compare the forward output and the input
    gradients (dq_nope, dq_pe, d_normed_kv, d_k_pe) on this rank's zigzag slice, plus the
    kv_b weight gradient -- which is global, so the per-rank partials are summed across CP.
    """
    cp_group = ctx.device_mesh.get_group("cp")
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

    # Reference: dense full-sequence MLA, no CP.
    q_nope_r = q_nope_full.clone().requires_grad_(True)
    q_pe_r = q_pe_full.clone().requires_grad_(True)
    normed_kv_r = normed_kv_full.clone().requires_grad_(True)
    k_pe_r = k_pe_full.clone().requires_grad_(True)
    w_r = w_full.clone().requires_grad_(True)
    kv = F.linear(normed_kv_r, w_r).view(req.B, req.S, H, nope + vdim)
    k_nope, value = torch.split(kv, [nope, vdim], dim=-1)
    out_r = mla_flash_attn_func(
        q_nope_r,
        q_pe_r,
        k_nope.contiguous(),
        k_pe_r,
        value.contiguous(),
        softmax_scale=softmax_scale,
        qk_nope_head_dim=nope,
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

    # Implementation: zigzag-local latent rotated around the ring.
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


def verify_mla(ctx: DistributedCtx, req: MLARequest) -> None:
    ref, imp, dW_ref, dW_imp = record_mla(ctx, req)
    for f in fields(ref):
        error = cosine_error(getattr(ref, f.name), getattr(imp, f.name))
        if error >= req.atol:
            raise AssertionError(f"{f.name} diverged: {error=:.2e} >= {req.atol=}")
    # Promote to fp32 for the dW comparison: the implementation now returns dW in fp32
    # (kept full-precision through FSDP's fp32 reduce-scatter), while the reference is
    # bf16 from autograd through F.linear -- norm/dot are dtype-sensitive at small mags.
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


# ---------------------------------------------------------------------------
# FP8 in-ring kv_b decompression (pass-latent CP with fp8_training="deep-gemm").
# ---------------------------------------------------------------------------

# deep_gemm is a required dependency (GPU-only project), so no import guard; gate only on the
# hardware the FP8 GEMM actually needs.
requires_fp8 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9),
    reason="FP8 kv_b path requires Hopper (SM90)+",
)


def record_mla_fp8(ctx: DistributedCtx, req: MLARequest):
    """
    Like record_mla but exercises the FP8 in-ring kv_b path. The reference decompresses the
    latent via an FP8Linear on the *full* sequence (so its kv_b quantization error matches the
    implementation); the implementation passes the same FP8-quantized kv_b weight into the ring
    via ``kv_b_quant``. Comparing fp8-vs-fp8 isolates the in-ring math from the (large) e4m3
    quantization error, so a tight tolerance is meaningful.
    """
    from pithtrain.layers.deepgemm_fp8_linear import FP8Linear

    cp_group = ctx.device_mesh.get_group("cp")
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

    # Reference: dense full-sequence MLA, no CP, kv_b decompressed via FP8Linear.
    ref_lin = FP8Linear(R, out_features, bias=False).to(device).to(torch.bfloat16)
    ref_lin.weight.data.copy_(w_full)
    q_nope_r = q_nope_full.clone().requires_grad_(True)
    q_pe_r = q_pe_full.clone().requires_grad_(True)
    normed_kv_r = normed_kv_full.clone().requires_grad_(True)
    k_pe_r = k_pe_full.clone().requires_grad_(True)
    kv = ref_lin(normed_kv_r).view(req.B, req.S, H, nope + vdim)
    k_nope, value = torch.split(kv, [nope, vdim], dim=-1)
    out_r = mla_flash_attn_func(
        q_nope_r,
        q_pe_r,
        k_nope.contiguous(),
        k_pe_r,
        value.contiguous(),
        softmax_scale=softmax_scale,
        qk_nope_head_dim=nope,
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

    # Implementation: zigzag-local latent rotated around the ring, FP8 in-ring decompress.
    imp_lin = FP8Linear(R, out_features, bias=False).to(device).to(torch.bfloat16)
    imp_lin.weight.data.copy_(w_full)
    kv_b_quant = imp_lin._get_quantized_weight()
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


def verify_mla_fp8(ctx: DistributedCtx, req: MLARequest) -> None:
    ref, imp, dW_ref, dW_imp = record_mla_fp8(ctx, req)
    # fp8-vs-fp8 reference: the only differences are the ring vs dense flash recompute order
    # and per-hop vs whole-sequence activation quant blocking -- both small. Check BOTH cosine
    # (direction) AND relative L2 error (magnitude): cosine is scale-invariant and would hide a
    # uniform fp8 scale bug, so relerr is the one that actually pins down magnitude. Tolerances
    # are looser than the bf16 test (e4m3 noise) but a broken/transposed fp8 GEMM gives O(1)
    # error on both, so this still catches it.
    atol_fp8 = 2e-2  # cosine
    # relerr is a few percent under fp8 (per-hop e4m3 block quant; dW, summed over all CP hops,
    # is the worst at ~3.6e-2 observed). A broken fp8 GEMM gives O(1) relerr, so 6e-2 still
    # catches it while leaving headroom for legitimate quant noise.
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


@requires_fp8
@pytest.mark.parametrize("cp_size,req", MLA_REQUESTS)
def test_mla_ring_attention_fp8_vs_dense(cp_size: int, req: MLARequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_mla_fp8, req)
