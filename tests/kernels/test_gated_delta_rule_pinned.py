"""
Verifier of the pinned dense chunked gated delta rule (gated_delta_rule_pinned_supported,
chunk_gdr_fwd_pinned, chunk_gdr_bwd_pinned), the dense path of Qwen3.5's Gated DeltaNet (CuTe-DSL
kernels on the H200 at every T, a ragged variant of each when T is not a multiple of 64; TIRx
kernels on the B200).

The recurrence, per value head with state S of shape (K, V) and scale = K ** -0.5:

    S = S * exp(g_t);  u = beta_t * (v_t - k_t @ S);  S = S + outer(k_t, u);  o_t = scale * q_t @ S

q and k are L2-normalized per head, H key heads serve groups of HV // H consecutive value heads,
g is the raw (natural-log) gate in fp32 and beta is bf16.

Checked, each test on its own (none skips on the H200):

- the capability query: True at the Qwen3.5 geometry (batch 1, 32 value heads, 16 or
  32 key heads, dim 128, bf16 q / k / v / beta, fp32 gate), False off it;
- the forward and backward (dq, dk per value head, dv, dg for the raw gate, dbeta) against a naive
  token-by-token fp32 recurrence and its autograd, at aligned and ragged lengths (T = 1 .. 8191,
  last-chunk sizes around the 8- and 16-row copy blocks), H = 16 and 32, the model's gate and a
  mild one, and packed-document resets (g = -128) on and next to chunk boundaries, inside a
  partial last chunk, on the final token, and 64 in a row ending on it. The kernels keep some chunk
  intermediates in bf16, so the check is calc_diff per output below BOUNDS plus a max absolute
  error within MAX_ABS of the reference's largest magnitude, and every value finite. The bounds are
  about 2.5 to 3 times the largest error measured on the H200 over every case here at the test's
  seed and three more: calc_diff 8.1e-6 (o), 8.1e-6 (dq), 1.1e-5 (dk), 6.1e-6 (dv),
  3.6e-5 (dg, at T = 2), 7.7e-6 (dbeta), and a largest absolute error of 9.3e-3 of the reference's
  largest magnitude (dg at T = 2; 8.7e-3 elsewhere);
- at long and ragged T (4096 .. 8192, with resets), the forward and backward against the
  recurrence in fp64, in its chunked form (itself checked against the token-by-token one), with the
  same bounds;
- determinism: repeated calls, and a call in a fresh process (cold compile), are bit-identical, at
  aligned and ragged T; the same results on non-default streams;
- which kernels run: only the CuTe kernels, once each, in both directions at aligned and ragged T,
  and one compile per kernel and variant over a sweep of lengths;
- that the comparisons see errors: a perturbed gate and a dg without its in-chunk reverse cumsum
  fail them.
"""

import functools
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from pithtrain.kernels import (
    chunk_gdr_bwd_pinned,
    chunk_gdr_fwd_pinned,
    gated_delta_rule_pinned_supported,
)

HV, D = 32, 128  # Qwen3.5-35B-A3B: 32 value heads, 16 key heads, head dims 128
BT = 64  # the chunk size; T a multiple of it runs the CuTe kernels on the H200
CUTE_KERNELS = (
    "WyRepresentation",
    "ChunkStateOutput",
    "BwdPrologue",
    "StateRecompute",
    "StateGradient",
    "ChunkGradients",
)
GRADS = ("dq", "dk", "dv", "dg", "dbeta")


def make_inputs(
    S: int,
    H: int = 16,
    B: int = 1,
    gate: str = "model",
    seed: int = 0,
    resets: tuple[int, ...] = (),
    hv: int = HV,
) -> tuple[torch.Tensor, ...]:
    """
    Return q / k (B, S, H, D) L2-normalized bf16, v (B, S, HV, D) bf16, g (B, S, HV) fp32, beta
    (B, S, HV) bf16 and do like v. gate="model" draws the gate as the model computes it,
    -exp(A_log) * softplus(a + dt_bias) with A_log = log U(0, 16); "mild" keeps a long memory.
    resets are packed-document starts, where g = -128 as the model sets it. hv value heads (HV
    unless a folded batch is wanted: b samples fold to H 16 b / HV 32 b).
    """
    HV = hv
    gen = torch.Generator(device="cuda").manual_seed(seed)
    device, bf16 = "cuda", torch.bfloat16

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=gen)

    q = F.normalize(randn(B, S, H, D).to(bf16), dim=-1)
    k = F.normalize(randn(B, S, H, D).to(bf16), dim=-1)
    v = randn(B, S, HV, D).to(bf16)
    if gate == "model":
        a_log = torch.log(torch.rand(HV, device=device, generator=gen) * 16)
        g = -a_log.exp() * F.softplus(randn(B, S, HV) + 1.0)
    else:
        g = -torch.rand(B, S, HV, device=device, generator=gen) * 0.1
    if resets:
        g[:, list(resets)] = -128.0
    beta = randn(B, S, HV).to(bf16).sigmoid()
    do = randn(B, S, HV, D).to(bf16)
    return q, k, v, g, beta, do


def pinned_path_available() -> bool:
    q, k, v, g, beta, _ = make_inputs(64)
    return gated_delta_rule_pinned_supported(q, k, v, g, beta)


pytestmark = pytest.mark.skipif(not pinned_path_available(), reason="no pinned path on this device")


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


# calc_diff bound per output and the largest absolute error over the reference's largest magnitude:
# about 2.5 to 3 times the largest measured errors listed in the module docstring.
BOUNDS = {"o": 2e-5, "dq": 2e-5, "dk": 3e-5, "dv": 2e-5, "dg": 1e-4, "dbeta": 2e-5}
MAX_ABS = 2e-2


def naive_gdr(q, k, v, g, beta):
    """
    q / k are per value head, (B, T, HV, D).
    """
    B, T, num_heads, K = q.shape
    scale = K**-0.5
    state = q.new_zeros(B, num_heads, K, v.shape[-1])
    out = []
    for t in range(T):
        state = state * g[:, t].exp()[..., None, None]
        k_t = k[:, t]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k_t, state))
        state = state + k_t[..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state))
    return torch.stack(out, 1)


def per_value_head(x: torch.Tensor, hv: int) -> torch.Tensor:
    return x.repeat_interleave(hv // x.shape[2], dim=2)


def naive_reference(q, k, v, g, beta, do) -> tuple[torch.Tensor, dict]:
    """
    The fp32 recurrence's output and its autograd gradients (dq, dk per value head).
    """
    hv = v.shape[2]
    leaves = [
        per_value_head(q, hv).float(),
        per_value_head(k, hv).float(),
        v.float(),
        g,
        beta.float(),
    ]
    leaves = [leaf_value.detach().clone().requires_grad_() for leaf_value in leaves]
    ref = naive_gdr(*leaves)
    ref.backward(do.float())
    return ref.detach(), {name: leaf.grad for name, leaf in zip(GRADS, leaves)}


def assert_close_to(got: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    assert got.shape == ref.shape, name
    assert torch.isfinite(got).all(), name
    diff_value = calc_diff(got, ref)
    assert diff_value < BOUNDS[name], f"{name}: calc_diff {diff_value:.2e} >= {BOUNDS[name]:.0e}"
    err = float((got.float() - ref).abs().max())
    bound = MAX_ABS * float(ref.abs().max())
    assert err <= bound, f"{name}: max abs err {err:.3e} > {bound:.3e}"


def test_supported_at_the_qwen35_geometry() -> None:
    for H in (16, 32):
        for S in (64, 1000, 4096):
            q, k, v, g, beta, _ = make_inputs(S, H=H)
            assert gated_delta_rule_pinned_supported(q, k, v, g, beta), (H, S)


def test_not_supported_off_the_geometry() -> None:
    q, k, v, g, beta, _ = make_inputs(256)
    bf16 = torch.bfloat16
    off = {
        "batch 2": make_inputs(256, B=2)[:5],
        "16 value heads": (q, k, v[:, :, :16], g[:, :, :16], beta[:, :, :16]),
        "key heads not dividing value heads": (q[:, :, :12], k[:, :, :12], v, g, beta),
        "key dim 64": (q[..., :64].contiguous(), k[..., :64].contiguous(), v, g, beta),
        "value dim 64": (q, k, v[..., :64].contiguous(), g, beta),
        "bf16 gate": (q, k, v, g.to(bf16), beta),
        "fp32 q / k / v": (q.float(), k.float(), v.float(), g, beta),
        "fp32 beta": (q, k, v, g, beta.float()),
        "q and k differ": (q[:, :, :8], k, v, g, beta),
        "shorter v": (q, k, v[:, :128], g, beta),
        "shorter gate": (q, k, v, g[:, :128], beta),
        "beta for 16 value heads": (q, k, v, g, beta[:, :, :16]),
        "empty sequence": (q[:, :0], k[:, :0], v[:, :0], g[:, :0], beta[:, :0]),
        "gate on the host": (q, k, v, g.cpu(), beta),
        "rank-3 q / k": (q[0], k[0], v, g, beta),
        "rank-2 gate": (q, k, v, g[0], beta),
        "48 value heads over 16 key heads": make_inputs(256, H=16, hv=48)[:5],
        "96 value heads over 32 key heads": make_inputs(256, H=32, hv=96)[:5],
        "64 value heads over 64 key heads": make_inputs(256, H=64, hv=64)[:5],
        "non-contiguous v": (q, k, v.transpose(1, 2).contiguous().transpose(1, 2), g, beta),
    }
    for name, args in off.items():
        assert not gated_delta_rule_pinned_supported(*args), name


# (S, H, gate, resets): every aligned length with both H; resets at, before and after chunk
# boundaries and on the final token. Then ragged lengths (the ragged kernel variants): resets at row
# 0, the row before the partial last chunk, its first row, its middle and the final token, and 64 in
# a row ending on the final token; last-chunk sizes around the copies' 8- and 16-row blocks.
CASES = [
    (64, 16, "model", ()),
    (64, 32, "mild", (0, 31, 63)),
    (128, 32, "model", (0, 63, 64, 127)),
    (128, 16, "mild", ()),
    (192, 16, "model", (0, 100, 191)),
    (192, 32, "mild", ()),
    (256, 32, "model", ()),
    (256, 16, "mild", (63, 64, 127, 255)),
    (512, 16, "model", (0, 200)),
    (512, 32, "mild", ()),
    (2048, 16, "model", (0, 700, 1024)),
    (2048, 32, "mild", ()),
    (4096, 16, "mild", ()),
    (4096, 32, "model", (63, 64, 127, 4095)),
    (1, 16, "model", ()),
    (1, 32, "mild", (0,)),
    (2, 32, "model", ()),
    (3, 16, "mild", (2,)),
    (37, 16, "model", (0, 18, 36)),
    (37, 32, "mild", ()),
    (63, 32, "model", (0, 31, 62)),
    (63, 16, "mild", ()),
    (65, 16, "model", (0, 63, 64)),
    (65, 32, "mild", (64,)),
    (100, 16, "mild", ()),
    (100, 32, "model", (0, 63, 64, 81, 99)),
    (127, 32, "model", (0, 63, 64, 95, 126)),
    (127, 16, "model", tuple(range(63, 127))),
    (129, 16, "model", (127, 128)),
    (129, 32, "mild", ()),
    (300, 16, "mild", (0, 77, 128, 250)),
    (300, 32, "model", (0, 255, 256, 278, 299)),
    (1000, 16, "mild", ()),
    (1000, 32, "model", (0, 999)),
    (1000, 16, "model", tuple(range(936, 1000))),
    (4097, 16, "model", (0, 4095, 4096)),
    (4097, 32, "mild", ()),
    (8191, 16, "mild", ()),
    (8191, 32, "model", (0, 8127, 8128, 8160, 8190)),
    *(
        (64 + tail, (16, 32)[i % 2], ("model", "mild")[i // 2 % 2], (63, 64, 63 + tail))
        for i, tail in enumerate((7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63))
    ),
]


@pytest.mark.parametrize("S,H,gate,resets", CASES)
def test_against_naive_recurrence(S: int, H: int, gate: str, resets: tuple[int, ...]) -> None:
    q, k, v, g, beta, do = make_inputs(S, H=H, gate=gate, seed=S + H, resets=resets)
    assert gated_delta_rule_pinned_supported(q, k, v, g, beta)

    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    assert o.shape == v.shape and o.dtype == torch.bfloat16
    assert g_cum.shape == g.shape and g_cum.dtype == torch.float32
    assert A.shape == (1, S, HV, 64)
    grads = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    dq, dk, dv, dg, dbeta = grads
    assert dq.shape == dk.shape == (1, S, HV, D)
    assert dv.shape == v.shape and dg.shape == g.shape and dbeta.shape == beta.shape
    assert dg.dtype == torch.float32 and dbeta.dtype == dv.dtype == torch.bfloat16
    assert all(grad.is_contiguous() for grad in grads)

    ref, ref_grads = naive_reference(q, k, v, g, beta, do)
    assert_close_to(o, ref, "o")
    for name, got in zip(GRADS, grads):
        assert_close_to(got, ref_grads[name], name)


# (S, b, gate, resets): b samples of H 16 / HV 32 folded into the heads (H 16 b / HV 32 b, value head
# j reading key head j // 2), as gated_delta_rule folds a batch; aligned and ragged, with resets.
FOLDED_CASES = [
    (64, 3, "model", ()),
    (64, 2, "mild", (0, 63)),
    (65, 3, "model", (64,)),
    (1000, 2, "model", (0, 999)),
    (2048, 3, "mild", ()),
    (4000, 3, "model", (0, 63, 64, 3999)),
    (4096, 3, "model", (63, 64, 4095)),
    (4096, 2, "mild", ()),
    (8192, 3, "model", (0, 8191)),
]


@pytest.mark.parametrize("S,b,gate,resets", FOLDED_CASES)
def test_folded_batch_against_naive_recurrence(
    S: int, b: int, gate: str, resets: tuple[int, ...]
) -> None:
    """
    The folded geometry (the SM90 widening of the capability query; skipped where it is False):
    forward and backward against the fp32 recurrence, as test_against_naive_recurrence.
    """
    q, k, v, g, beta, do = make_inputs(S, H=16 * b, gate=gate, seed=S + b, resets=resets, hv=32 * b)
    if not gated_delta_rule_pinned_supported(q, k, v, g, beta):
        pytest.skip("the folded geometry runs the pinned kernels on SM90 only")
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    assert o.shape == v.shape and A.shape == (1, S, 32 * b, 64)
    grads = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    assert grads[0].shape == grads[1].shape == (1, S, 32 * b, D)
    ref, ref_grads = naive_reference(q, k, v, g, beta, do)
    assert_close_to(o, ref, "o")
    for name, got in zip(GRADS, grads):
        assert_close_to(got, ref_grads[name], name)


@pytest.mark.parametrize("S", [64, 4000])
def test_folded_batch_is_per_sample(S: int) -> None:
    """
    Folding is exact: each sample's slice of the folded call's outputs and gradients equals the
    pinned call on that sample alone, bit for bit (every head's recurrence is independent).
    """
    b = 3
    q, k, v, g, beta, do = make_inputs(
        S, H=16 * b, gate="model", seed=7, resets=(0, S // 2), hv=32 * b
    )
    if not gated_delta_rule_pinned_supported(q, k, v, g, beta):
        pytest.skip("the folded geometry runs the pinned kernels on SM90 only")
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    folded = (o, g_cum, A, *chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do))
    for i in range(b):
        keys, values = slice(16 * i, 16 * (i + 1)), slice(32 * i, 32 * (i + 1))
        qi, ki = q[:, :, keys].contiguous(), k[:, :, keys].contiguous()
        vi, gi, bi, doi = (x[:, :, values].contiguous() for x in (v, g, beta, do))
        oi, gci, Ai = chunk_gdr_fwd_pinned(qi, ki, vi, gi, bi)
        single = (oi, gci, Ai, *chunk_gdr_bwd_pinned(qi, ki, vi, gci, bi, Ai, doi))
        for name, x, y in zip(("o", "g_cum", "A", *GRADS), folded, single):
            assert torch.equal(x[:, :, values], y), (i, name)


@pytest.mark.parametrize("S", [256, 1000])
def test_zero_and_tiny_beta(S: int) -> None:
    """
    beta exactly 0 and below 1e-9 (a sigmoid far in its tail) on a sixth of the positions: every
    gradient, dbeta at those positions included, against the fp32 recurrence with the usual bounds.
    """
    q, k, v, g, beta, do = make_inputs(S, H=16, gate="mild", seed=S + 5)
    gen = torch.Generator(device="cuda").manual_seed(S)
    pick = torch.rand(beta.shape, generator=gen, device="cuda")
    beta = torch.where(pick < 1 / 12, 0.0, beta)
    beta = torch.where((pick >= 1 / 12) & (pick < 1 / 6), 2.0**-40, beta).to(torch.bfloat16)
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    grads = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    ref, ref_grads = naive_reference(q, k, v, g, beta, do)
    assert_close_to(o, ref, "o")
    for name, got in zip(GRADS, grads):
        assert_close_to(got, ref_grads[name], name)
    small = pick < 1 / 6
    dbeta, want = grads[4].float()[small], ref_grads["dbeta"][small]
    err = float((dbeta - want).abs().max())
    assert err <= MAX_ABS * float(ref_grads["dbeta"].abs().max()), f"dbeta at small beta: {err:.3e}"


def test_comparison_is_sensitive() -> None:
    """
    A 10% weaker gate in the forward, a 30% weaker one in the backward (on its saved tensors), a dg
    without its in-chunk reverse cumsum, or one cumulated over the whole sequence instead, all fail
    the comparisons above.
    """
    q, k, v, g, beta, do = make_inputs(256, gate="model", seed=3)
    ref, ref_grads = naive_reference(q, k, v, g, beta, do)
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g * 0.9, beta)
    assert calc_diff(o, ref) > BOUNDS["o"]
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g * 0.7, beta)
    for name, got in zip(GRADS, chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)):
        assert calc_diff(got, ref_grads[name]) > BOUNDS[name], name
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    dg = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)[3]
    chunks = dg.view(1, -1, BT, HV)
    no_cumsum = (chunks - F.pad(chunks[:, :, 1:], (0, 0, 0, 1))).view_as(dg)
    whole_sequence = dg.flip(1).cumsum(1).flip(1)
    for wrong in (no_cumsum, whole_sequence):
        assert calc_diff(wrong, ref_grads["dg"]) > BOUNDS["dg"]


def chunked_reference(q, k, v, g, beta, do, chunk: int = BT) -> tuple[torch.Tensor, dict]:
    """
    The recurrence's output and autograd gradients (dq, dk per value head) in fp64, in its chunked
    form, whose autograd stays small at long lengths: within a chunk entering with state S0 and
    cumulative gate G, U solves (I + A) U = beta * (V - exp(G) * (K S0)) with A[t, i] = beta_t
    exp(G_t - G_i) k_t . k_i for i < t; o = scale * (exp(G) * (Q S0) + (exp(G_t - G_i) q_t .
    k_i)_{i <= t} U); the state leaving is exp(G_L) S0 + sum_i exp(G_L - G_i) outer(k_i, u_i).
    """
    hv = v.shape[2]
    leaves = [per_value_head(q, hv), per_value_head(k, hv), v, g, beta]
    leaves = [x.detach().double().requires_grad_() for x in leaves]
    qd, kd, vd, gd, bd = leaves
    scale = qd.shape[-1] ** -0.5
    state = qd.new_zeros(hv, qd.shape[-1], vd.shape[-1])
    out = []
    for c0 in range(0, qd.shape[1], chunk):
        span = slice(c0, min(c0 + chunk, qd.shape[1]))
        qc, kc, vc = (t[0, span].transpose(0, 1) for t in (qd, kd, vd))
        gc, bc = gd[0, span].transpose(0, 1).cumsum(-1), bd[0, span].transpose(0, 1)
        L = gc.shape[-1]
        lower = torch.ones(L, L, dtype=torch.bool, device=qd.device).tril()
        decay = torch.where(lower, gc[:, :, None] - gc[:, None, :], float("-inf")).exp()
        strict = decay * (~torch.eye(L, dtype=torch.bool, device=qd.device))
        A = bc[:, :, None] * strict * (kc @ kc.transpose(-1, -2))
        rhs = bc[..., None] * (vc - gc.exp()[..., None] * (kc @ state))
        eye = torch.eye(L, dtype=qd.dtype, device=qd.device)
        u = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
        o = scale * (gc.exp()[..., None] * (qc @ state) + (decay * (qc @ kc.transpose(-1, -2))) @ u)  # fmt: skip
        out.append(o.transpose(0, 1))
        k_decayed = kc * (gc[:, -1:] - gc).exp()[..., None]
        state = gc[:, -1].exp()[:, None, None] * state + k_decayed.transpose(-1, -2) @ u
    ref = torch.cat(out, 0).unsqueeze(0)
    ref.backward(do.double())
    return ref.detach(), {name: leaf.grad for name, leaf in zip(GRADS, leaves)}


def test_chunked_reference_is_the_recurrence() -> None:
    """
    The fp64 chunked reference equals the token-by-token recurrence (in fp64) to rounding, ragged
    last chunk and resets included.
    """
    q, k, v, g, beta, do = make_inputs(150, H=16, gate="model", seed=9, resets=(0, 64, 100))
    ref, ref_grads = chunked_reference(q, k, v, g, beta, do)
    leaves = [per_value_head(q, HV), per_value_head(k, HV), v, g, beta]
    leaves = [x.detach().double().requires_grad_() for x in leaves]
    naive = naive_gdr(*leaves)
    naive.backward(do.double())
    torch.testing.assert_close(ref, naive.detach(), rtol=1e-9, atol=1e-9)
    for name, leaf in zip(GRADS, leaves):
        torch.testing.assert_close(ref_grads[name], leaf.grad, rtol=1e-8, atol=1e-8)


# (S, H, gate, resets): long and ragged lengths, against the fp64 chunked recurrence.
LONG_CASES = [
    (4096, 16, "mild", ()),
    (4096, 32, "model", (0, 63, 64, 127, 4095)),
    (4096, 16, "model", (0, 300, 1777, 2048, 4091)),
    (4096, 32, "mild", (0, 300, 1777, 2048, 4091)),
    (8192, 16, "mild", ()),
    (8192, 32, "mild", ()),
    (1000, 16, "model", (0, 959, 960, 980, 999)),
    (4097, 32, "model", (0, 4095, 4096)),
    (4097, 16, "mild", ()),
    (300, 32, "mild", (0, 77, 128, 250)),
    (8191, 16, "mild", (8190,)),
]


@pytest.mark.parametrize("S,H,gate,resets", LONG_CASES)
def test_long_against_chunked_reference(S: int, H: int, gate: str, resets) -> None:
    """
    Forward and backward at long and ragged T against the fp64 chunked recurrence, with the bounds
    of test_against_naive_recurrence (BOUNDS, MAX_ABS).
    """
    q, k, v, g, beta, do = make_inputs(S, H=H, gate=gate, seed=S + H, resets=resets)
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    grads = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    ref, ref_grads = chunked_reference(q, k, v, g, beta, do)
    assert_close_to(o, ref, "o")
    for name, got in zip(GRADS, grads):
        assert_close_to(got, ref_grads[name], name)


COLD_RUN = """
import sys, torch
sys.path.insert(0, {tests!r})
from test_gated_delta_rule_pinned import make_inputs
from pithtrain.kernels import chunk_gdr_bwd_pinned, chunk_gdr_fwd_pinned
q, k, v, g, beta, do = make_inputs({S}, H=16, gate="model", seed=5)
o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
torch.save([o.cpu(), *(x.cpu() for x in chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do))], {path!r})
"""


@pytest.mark.parametrize("S", [256, 1000, 4097, 8192])
def test_deterministic(tmp_path, S: int) -> None:
    """
    Two calls in this process and one in a fresh process (its kernels compiled cold) agree bit for bit.
    """
    q, k, v, g, beta, do = make_inputs(S, H=16, gate="model", seed=5)
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    first = (o, *chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do))
    second = (
        chunk_gdr_fwd_pinned(q, k, v, g, beta)[0],
        *chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do),
    )
    for name, x, y in zip(("o", *GRADS), first, second):
        assert torch.equal(x, y), name
    path = str(tmp_path / "cold.pt")
    code = COLD_RUN.format(tests=os.path.dirname(__file__), S=S, path=path)
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
    )
    for name, x, y in zip(("o", *GRADS), first, torch.load(path)):
        assert torch.equal(x.cpu(), y), name


@pytest.mark.parametrize("S", [1, 63, 64, 65, 127, 128, 192])
def test_short_pipelines_repeat(S: int) -> None:
    """
    Fewer chunks than the state-gradient ring's depth (1, 2, 3 chunks, the partial last chunk the
    first one the reverse recurrence processes): 20 back-to-back backward launches agree bit for
    bit, so no barrier or ring-stage race shows at short lengths.
    """
    q, k, v, g, beta, do = make_inputs(S, H=16, gate="model", seed=S)
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    first = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    for _ in range(20):
        again = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
        for name, x, y in zip(GRADS, first, again):
            assert torch.equal(x, y), name


def kernel_names(fn) -> list[str]:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]


@functools.cache
def runs_cute_kernels() -> bool:
    """
    Whether the pinned forward launches the CuTe kernels (the H200's implementation); the kernel-set
    and compile-count checks below are about those kernels, and skip where another implementation
    (the B200's TIRx kernels) runs.
    """
    q, k, v, g, beta, _ = make_inputs(64, H=16)
    names = kernel_names(lambda: chunk_gdr_fwd_pinned(q, k, v, g, beta))
    return any(cute in name for name in names for cute in CUTE_KERNELS)


@pytest.mark.parametrize("S", [1, 63, 65, 1000, 4096, 4097, 8192])
def test_only_cute_kernels(S: int) -> None:
    """
    Each direction launches exactly its CuTe kernels, once each, and nothing else, at aligned and
    ragged T.
    """
    if not runs_cute_kernels():
        pytest.skip("the pinned kernels here are not the CuTe ones")
    q, k, v, g, beta, do = make_inputs(S, H=16)
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    for direction, fn, expected in (
        ("forward", lambda: chunk_gdr_fwd_pinned(q, k, v, g, beta), CUTE_KERNELS[:2]),
        ("backward", lambda: chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do), CUTE_KERNELS[2:]),
    ):
        names = kernel_names(fn)
        launched = sorted(
            next((cute for cute in CUTE_KERNELS if cute in name), name) for name in names
        )
        assert launched == sorted(expected), f"{direction} at T={S} launches {launched}"


@pytest.mark.parametrize("S", [65, 4097])
def test_side_streams(S: int) -> None:
    """
    The compiled kernels launch on the caller's current stream (resolved per call, not cached): on
    two non-default streams, after the default stream compiled and ran them, the same results.
    """
    q, k, v, g, beta, do = make_inputs(S, H=16, gate="model", seed=9, resets=(0, S - 1))
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    first = (o, g_cum, A, *chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do))
    torch.cuda.synchronize()
    for _ in range(2):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10_000_000)  # the stream is still busy when the kernels are enqueued
            o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
            again = (o, g_cum, A, *chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do))
        torch.cuda.current_stream().wait_stream(stream)
        for name, x, y in zip(("o", "g_cum", "A", *GRADS), first, again):
            assert torch.equal(x, y), name


COMPILE_COUNT = """
import sys, torch
import cutlass.cute as cute
sys.path.insert(0, {tests!r})
compiles = []
compile_kernel = cute.compile
cute.compile = lambda kernel, *args, **kwargs: compiles.append(type(kernel).__name__) or compile_kernel(kernel, *args, **kwargs)
from test_gated_delta_rule_pinned import make_inputs
from pithtrain.kernels import chunk_gdr_bwd_pinned, chunk_gdr_fwd_pinned
for S in range(1, 201):
    q, k, v, g, beta, do = make_inputs(S, H=16, gate="model", seed=S)
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
torch.cuda.synchronize()
print(" ".join(sorted(compiles)))
"""


def test_one_compile_per_kernel_variant() -> None:
    """
    A sweep of T = 1 .. 200 in a fresh process compiles each of the six kernels twice, the aligned
    and the ragged variant, and nothing per length (T stays a runtime argument).
    """
    if not runs_cute_kernels():
        pytest.skip("the pinned kernels here are not the CuTe ones")
    code = COMPILE_COUNT.format(tests=os.path.dirname(__file__))
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
    )
    assert out.stdout.split() == sorted(2 * CUTE_KERNELS), out.stdout
