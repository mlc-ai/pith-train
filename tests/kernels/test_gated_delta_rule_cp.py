"""
Verifier of one rank's share of the chunked gated delta rule under context parallelism
(gated_delta_rule_cp_supported, gated_delta_rule_cp_plan, chunk_gdr_cp_fwd_pre / _post,
chunk_gdr_cp_bwd_pre / _post), the CP path of Qwen3.5's Gated DeltaNet on the B200 (SM90 has no
kernel: the capability query is False there and the whole file skips).

The recurrence, per value head with state S of shape (K, V) and scale = K ** -0.5:

    S = S * exp(g_t);  u = beta_t * (v_t - k_t @ S);  S = S + outer(k_t, u);  o_t = scale * q_t @ S

with S reset to zero at every packed-document start. q and k are L2-normalized per head on 16 key
heads and repeated to the 32 value heads, as the operator hands them over under CP.

The CP group is emulated in one process, as in the harness the kernels were evolved in: the whole
sequence is cut into world_size contiguous shards, each rank's CP context is derived from the
global document boundaries as FLA's build_cp_context derives it, every rank's stage before the
all-gather runs, the payloads are stacked in rank order as the all-gather returns them, and every
rank's stage after it runs on that stack.

Checked:

- the capability query: True at the Qwen3.5 geometry (batch 1, 32 heads per value head, dim 128,
  bf16 q / k / v / beta, fp32 gate), False off it;
- the reference: the recurrence in its chunked form (64-token chunks, the intra-chunk system solved
  as a unit lower-triangular one), in fp64 PyTorch, whose autograd stays small at long lengths;
  itself checked against the naive token-by-token recurrence on a short packed sequence;
- the shards' outputs and gradients (dq, dk per value head, dv, dg for the raw gate, dbeta),
  concatenated, against that reference over the whole sequence and its autograd: calc_diff below
  1e-4 (output) and 1e-3 (gradients) plus a max absolute error within 2e-2 of the reference's
  largest magnitude, the bounds of the dense verifier. Cases cover every
  rank position, partial chunks, documents ending on, inside and across rank boundaries (on and
  off the 64-token chunk grid), gate resets, and lengths that take the backward's split and packed
  h recompute and the forward's folded last-document pass;
- the payloads' shapes and dtypes, and that the ones no rank reads are zero;
- sensitivity: a perturbed gate, or the payloads stacked out of rank order, fails the comparison.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import (
    chunk_gdr_cp_bwd_post,
    chunk_gdr_cp_bwd_pre,
    chunk_gdr_cp_fwd_post,
    chunk_gdr_cp_fwd_pre,
    gated_delta_rule_cp_plan,
    gated_delta_rule_cp_supported,
)

H, HV, D = 16, 32, 128  # Qwen3.5-35B-A3B: 16 key heads, 32 value heads, head dims 128


def meta_inputs(S: int = 256, B: int = 1, heads: int = HV, dim: int = D) -> list[torch.Tensor]:
    bf16 = torch.bfloat16
    qkv = [torch.empty(B, S, heads, dim, dtype=bf16, device="meta") for _ in range(3)]
    g = torch.empty(B, S, heads, dtype=torch.float32, device="meta")
    beta = torch.empty(B, S, heads, dtype=bf16, device="meta")
    return [*qkv, g, beta]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not gated_delta_rule_cp_supported(*meta_inputs()),
    reason="no CP kernels on this device",
)


def make_inputs(
    T: int, gate: str = "model", seed: int = 0, resets: tuple[int, ...] = ()
) -> tuple[torch.Tensor, ...]:
    """
    The whole sequence: q / k (1, T, HV, D) L2-normalized bf16 on H key heads repeated to HV, v
    (1, T, HV, D) bf16, g (1, T, HV) fp32, beta (1, T, HV) bf16 and do like v. gate="model" draws
    the gate as the model computes it, -exp(A_log) * softplus(a + dt_bias) with A_log = log U(0,
    16); "mild" keeps a long memory. resets get g = -128, as the model sets it at document starts.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    device, bf16 = "cuda", torch.bfloat16

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=gen)

    q = F.normalize(randn(1, T, H, D).to(bf16), dim=-1).repeat_interleave(HV // H, dim=2)
    k = F.normalize(randn(1, T, H, D).to(bf16), dim=-1).repeat_interleave(HV // H, dim=2)
    v = randn(1, T, HV, D).to(bf16)
    if gate == "model":
        a_log = torch.log(torch.rand(HV, device=device, generator=gen) * 16)
        g = -a_log.exp() * F.softplus(randn(1, T, HV) + 1.0)
    else:
        g = -torch.rand(1, T, HV, device=device, generator=gen) * 0.1
    if resets:
        g[:, list(resets)] = -128.0
    beta = randn(1, T, HV).to(bf16).sigmoid()
    do = randn(1, T, HV, D).to(bf16)
    return q.contiguous(), k.contiguous(), v, g, beta, do


def rank_context(cu: list[int], world: int, rank: int) -> tuple:
    """
    FLA's CP context of rank for global document boundaries cu over world contiguous shards:
    (local boundaries, is_first_rank, is_last_rank, pre_num_ranks, post_num_ranks).
    """
    part = cu[-1] // world
    lo, hi = rank * part, (rank + 1) * part
    local = sorted({min(max(x, lo), hi) - lo for x in cu})
    first_start = max(x for x in cu[:-1] if x <= lo)
    last_end = min(x for x in cu[1:] if x >= hi)
    pre, post = rank - first_start // part, (last_end - 1) // part - rank
    return local, pre == 0, post == 0, pre, post


def make_plans(cu: list[int], world: int) -> list[dict]:
    plans = []
    for rank in range(world):
        local, first, last, pre, post = rank_context(cu, world, rank)
        cu_cpu = torch.tensor(local, dtype=torch.int32)
        plans.append(gated_delta_rule_cp_plan(cu_cpu.cuda(), cu_cpu, rank, first, last, pre, post))
    return plans


def stack(payloads: list[torch.Tensor]) -> torch.Tensor:
    """
    What the all-gather returns: every rank's payload in rank order.
    """
    return torch.stack(payloads)


def run_cp(q, k, v, g, beta, do, world: int, cu: list[int], gather=stack) -> dict:
    """
    One emulated CP step over the whole sequence: the shards' results concatenated, and the
    payloads.
    """
    S = q.shape[1] // world
    plans = make_plans(cu, world)

    def shard(t: torch.Tensor, rank: int) -> torch.Tensor:
        return t[:, rank * S : (rank + 1) * S].contiguous()

    pre = [chunk_gdr_cp_fwd_pre(*(shard(t, r) for t in (q, k, v, g, beta)), plans[r]) for r in range(world)]  # fmt: skip
    gathered = gather([payload for payload, _ in pre])
    fwd = [chunk_gdr_cp_fwd_post(gathered, saved, plans[r]) for r, (_, saved) in enumerate(pre)]
    out = {"o": torch.cat([o for o, *_ in fwd], 1), "hm": [payload for payload, _ in pre]}
    out["fwd"] = fwd
    if do is None:
        return out
    bpre = []
    for r, (_, g_cum, A, initial_state) in enumerate(fwd):
        args = [shard(t, r) for t in (q, k, v)] + [g_cum, shard(beta, r), A, shard(do, r)]
        bpre.append(chunk_gdr_cp_bwd_pre(*args, initial_state, plans[r]))
    gathered = gather([payload for payload, _ in bpre])
    grads = [chunk_gdr_cp_bwd_post(gathered, saved, plans[r]) for r, (_, saved) in enumerate(bpre)]
    for i, name in enumerate(("dq", "dk", "dv", "dg", "dbeta")):
        out[name] = torch.cat([grad[i] for grad in grads], 1)
    out["dhm"] = [payload for payload, _ in bpre]
    return out


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def naive_gdr(q, k, v, g, beta, cu: list[int]):
    """
    q / k are per value head, (1, T, HV, D); the state resets at every document start in cu.
    """
    T, K = q.shape[1], q.shape[-1]
    scale = K**-0.5
    starts = set(cu[:-1])
    state = q.new_zeros(1, q.shape[2], K, v.shape[-1])
    out = []
    for t in range(T):
        if t in starts:
            state = state * 0
        state = state * g[:, t].exp()[..., None, None]
        k_t = k[:, t]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k_t, state))
        state = state + k_t[..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state))
    return torch.stack(out, 1)


def chunked_gdr(q, k, v, g, beta, cu: list[int], chunk: int = 64):
    """
    naive_gdr in chunked form: within a chunk entering with state S0 and cumulative gate G (Gamma =
    exp(G)), U solves (I + A) U = beta * (V - Gamma * (K S0)) with A[t, i] = beta_t exp(G_t - G_i)
    k_t . k_i for i < t; o = scale * (Gamma * (Q S0) + (exp(G_t - G_i) q_t . k_i)_{i <= t} U); the
    state leaving is exp(G_L) S0 + sum_i exp(G_L - G_i) outer(k_i, u_i).
    """
    scale = q.shape[-1] ** -0.5
    out = []
    for bos, eos in zip(cu[:-1], cu[1:]):
        state = q.new_zeros(q.shape[2], q.shape[-1], v.shape[-1])
        for c0 in range(bos, eos, chunk):
            span = slice(c0, min(c0 + chunk, eos))
            qc, kc, vc = (t[0, span].transpose(0, 1) for t in (q, k, v))
            gc, bc = g[0, span].transpose(0, 1).cumsum(-1), beta[0, span].transpose(0, 1)
            L = gc.shape[-1]
            lower = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
            decay = torch.where(lower, gc[:, :, None] - gc[:, None, :], float("-inf")).exp()
            strict = decay * (~torch.eye(L, dtype=torch.bool, device=q.device))
            A = bc[:, :, None] * strict * (kc @ kc.transpose(-1, -2))
            rhs = bc[..., None] * (vc - gc.exp()[..., None] * (kc @ state))
            eye = torch.eye(L, device=q.device)
            u = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
            o = scale * (gc.exp()[..., None] * (qc @ state) + (decay * (qc @ kc.transpose(-1, -2))) @ u)  # fmt: skip
            out.append(o.transpose(0, 1))
            kd = kc * (gc[:, -1:] - gc).exp()[..., None]
            state = gc[:, -1].exp()[:, None, None] * state + kd.transpose(-1, -2) @ u
    return torch.cat(out, 0).unsqueeze(0)


def assert_close_to(got: torch.Tensor, ref: torch.Tensor, name: str, diff: float) -> None:
    assert got.shape == ref.shape, name
    assert torch.isfinite(got).all(), name
    diff_value = calc_diff(got, ref)
    assert diff_value < diff, f"{name}: calc_diff {diff_value:.2e}"
    err = float((got.float() - ref).abs().max())
    bound = 2e-2 * float(ref.abs().max())
    assert err <= bound, f"{name}: max abs err {err:.3e} > {bound:.3e}"


def test_supported_at_the_qwen35_geometry() -> None:
    for S in (1, 100, 1024, 8192):
        assert gated_delta_rule_cp_supported(*meta_inputs(S)), S


def test_not_supported_off_the_geometry() -> None:
    q, k, v, g, beta = meta_inputs()
    off = {
        "batch 2": meta_inputs(B=2),
        "16 heads": meta_inputs(heads=16),
        "64 value heads (folded batch)": meta_inputs(heads=64),
        "dim 64": meta_inputs(dim=64),
        "empty": meta_inputs(S=0),
        "unrepeated key heads": (q[:, :, :H], k[:, :, :H], v, g, beta),
        "value dim 64": (q, k, v[..., :64], g, beta),
        "bf16 gate": (q, k, v, g.to(torch.bfloat16), beta),
        "fp32 q / k / v": (q.float(), k.float(), v.float(), g, beta),
        "fp32 beta": (q, k, v, g, beta.float()),
    }
    for name, args in off.items():
        assert not gated_delta_rule_cp_supported(*args), name


# (world size, tokens per rank, global document boundaries or None, gate, resets)
CASES = [
    (2, 256, None, "mild", ()),
    (3, 100, None, "model", ()),
    (4, 512, None, "model", ()),
    (4, 256, [0, 100, 512, 600, 1024], "model", ()),
    (4, 300, [0, 64, 1150, 1200], "mild", ()),
    (4, 256, [0, 64, 512, 640, 1024], "model", ()),
    (2, 1000, None, "model", (0, 77, 1000, 1500)),
    (2, 2560, [0, 1500, 5120], "mild", ()),
    (2, 4096, None, "mild", ()),
    (2, 6144, None, "model", ()),
]


@pytest.mark.parametrize("world,S,cu,gate,resets", CASES)
def test_against_naive_recurrence(world: int, S: int, cu, gate: str, resets) -> None:
    T = world * S
    cu = cu or [0, T]
    q, k, v, g, beta, do = make_inputs(T, gate=gate, seed=T + world, resets=resets)
    assert gated_delta_rule_cp_supported(*(t[:, :S] for t in (q, k, v, g, beta)))
    got = run_cp(q, k, v, g, beta, do, world, cu)

    for o, g_cum, A, initial_state in got["fwd"]:
        assert o.shape == (1, S, HV, D) and o.dtype == torch.bfloat16
        assert g_cum.shape == (1, S, HV) and g_cum.dtype == torch.float32
        assert A.shape == (1, S, HV, 64) and A.dtype == torch.bfloat16
        assert initial_state.shape == (1, HV, D, D) and initial_state.dtype == torch.float32
    for r in range(world):
        _, first, last, _, _ = rank_context(cu, world, r)
        for name, payload, unread in (("hm", got["hm"][r], last), ("dhm", got["dhm"][r], first)):
            assert payload.shape == (HV, D, 2 * D) and payload.dtype == torch.float32, name
            if unread:
                assert not payload.any(), f"{name} of rank {r} is read by no rank but is not zero"
    for name in ("dq", "dk", "dv", "dbeta"):
        assert got[name].dtype == torch.bfloat16, name
    assert got["dg"].dtype == torch.float32

    leaves = [t.double().requires_grad_() for t in (q, k, v, g, beta)]
    ref = chunked_gdr(*leaves, cu)
    ref.backward(do.double())

    assert_close_to(got["o"], ref.detach(), "o", 1e-4)
    for name, leaf in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves):
        assert_close_to(got[name], leaf.grad, name, 1e-3)


def test_reference_is_the_recurrence() -> None:
    T, cu = 300, [0, 100, 164, 300]
    q, k, v, g, beta, do = make_inputs(T, gate="model", seed=5, resets=(200,))
    leaves = [t.double().requires_grad_() for t in (q, k, v, g, beta)]
    twins = [t.detach().clone().requires_grad_() for t in leaves]
    naive, chunked = naive_gdr(*leaves, cu), chunked_gdr(*twins, cu)
    naive.backward(do.double())
    chunked.backward(do.double())
    assert calc_diff(chunked, naive) < 1e-10
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), twins, leaves):
        assert calc_diff(a.grad, b.grad) < 1e-9, name


def test_comparison_is_sensitive() -> None:
    world, S = 4, 256
    q, k, v, g, beta, _ = make_inputs(world * S, gate="mild", seed=3)
    cu = [0, world * S]
    ref = chunked_gdr(q.double(), k.double(), v.double(), g.double(), beta.double(), cu)
    assert calc_diff(run_cp(q, k, v, g, beta, None, world, cu)["o"], ref) < 1e-4
    assert calc_diff(run_cp(q, k, v, g * 0.9, beta, None, world, cu)["o"], ref) > 1e-4
    # The merge reads every predecessor's payload by its rank: a stack out of rank order fails.
    reverse = run_cp(q, k, v, g, beta, None, world, cu, gather=lambda p: torch.stack(p[::-1]))
    assert calc_diff(reverse["o"], ref) > 1e-4
