"""
Verifier of the context-parallel chunked gated delta rule pieces (gated_delta_rule_cp_supported,
gdr_cp_fwd_local, gdr_cp_merge, gdr_cp_fwd_output, gdr_cp_bwd_local, gdr_cp_bwd_states,
gdr_cp_bwd_output): FLA 0.5.2's context-parallel structure on CuTe-DSL kernels, SM90 only (every
test skips where the capability query is False).

A sequence is split into W consecutive rank slices. Rank r's affine state map (h_end = M h_start +
he) comes from its own inputs. The maps are gathered and folded into each rank's initial state h0,
and the backward does the same in reverse for the gradient of each rank's final state.

Checked against PyTorch (FLA is no longer a dependency, so its kernels no longer serve as
oracles; the references below evaluate FLA's chunk algorithm in float64 or run the fp32
recurrence):

- the capability query: True at Qwen3.5's geometry and its batch folds (H 16 b / HV 32 b), False
  off it;
- single GPU, synthetic inputs, model and mild gates, resets, aligned and ragged slices, HV 32 and
  96:
  - the forward map hm (he and the M chain) against its float64 evaluation from the rank's own w,
    u and g_cum: calc_diff below 1e-5 and max-abs within 2e-2 of the largest magnitude;
  - the merge in both directions against the float64 fold of the gathered maps, never reading the
    slots outside its ranks (they are NaN-poisoned), and seeing a wrong order or membership;
  - the backward map's M^T and dh halves against their float64 evaluation, the same bounds;
  - the states and v_new recomputed from h0 against the float64 state recurrence, and the output
    from h0 against the fp32 recurrence (calc_diff below 1e-4);
  - the gradients from h0 and dht against the fp32 recurrence (with h0, and with <h_T, dht> added
    to the loss): calc_diff below 1e-3 and max-abs within 2e-2 of the largest magnitude;
- a reset (g = -128) at chunk residue 0, 1 or 63 zeroes M and M^T exactly, and a merge through
    such a rank yields exactly that rank's he;
- the whole chain emulated in one process (W ranks run one after another, the all-gather a stack)
  against the fp32 recurrence over the whole sequence, output and the five gradients, documents
  crossing rank boundaries, aligned and ragged slices, batch 1 and 3 folded;
- determinism (repeated calls bit-identical), only the CuTe kernels launching, and one compile
  per kernel variant over a sweep of lengths, ranks and group sizes.

The SM100 TIRx family (chunk_gdr_cp_supported, gated_delta_rule_cp_plan, chunk_gdr_cp_fwd_pre /
_post, chunk_gdr_cp_bwd_pre / _post; every such test skips where its query is False). Verifier of
one rank's share of the chunked gated delta rule under context parallelism
(chunk_gdr_cp_supported, gated_delta_rule_cp_plan, chunk_gdr_cp_fwd_pre / _post,
chunk_gdr_cp_bwd_pre / _post), the CP path of Qwen3.5's Gated DeltaNet on the B200.

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

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from pithtrain.kernels import (
    chunk_gdr_cp_bwd_post,
    chunk_gdr_cp_bwd_pre,
    chunk_gdr_cp_fwd_post,
    chunk_gdr_cp_fwd_pre,
    chunk_gdr_cp_supported,
    gated_delta_rule_cp_plan,
    gated_delta_rule_cp_supported,
    gdr_cp_bwd_local,
    gdr_cp_bwd_output,
    gdr_cp_bwd_states,
    gdr_cp_fwd_local,
    gdr_cp_fwd_output,
    gdr_cp_merge,
)

D, BT = 128, 64
GRADS = ("dq", "dk", "dv", "dg", "dbeta")
RCP_LN2 = 1.4426950216


def make_inputs(
    T: int, b: int = 1, gate: str = "model", seed: int = 0, resets: tuple[int, ...] = ()
) -> tuple[torch.Tensor, ...]:
    """
    One slice of b samples folded into the heads: q / k (1, T, 16 b, D) L2-normalized bf16, v (1,
    T, 32 b, D) bf16, the raw gate g (1, T, 32 b) fp32, beta bf16 and do like v. gate="model" as
    the model computes it, "mild" a long memory (the transition M of order 1). resets: g = -128.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    H, HV = 16 * b, 32 * b

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=gen)

    q = F.normalize(randn(1, T, H, D).to(torch.bfloat16), dim=-1)
    k = F.normalize(randn(1, T, H, D).to(torch.bfloat16), dim=-1)
    v = randn(1, T, HV, D).to(torch.bfloat16)
    if gate == "model":
        a_log = torch.log(torch.rand(HV, device="cuda", generator=gen) * 16)
        g = -a_log.exp() * F.softplus(randn(1, T, HV) + 1.0)
    else:
        g = -torch.rand(1, T, HV, device="cuda", generator=gen) * 0.02
    if resets:
        g[:, list(resets)] = -128.0
    beta = randn(1, T, HV).to(torch.bfloat16).sigmoid()
    return q, k, v, g, beta, randn(1, T, HV, D).to(torch.bfloat16)


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    return gated_delta_rule_cp_supported(*make_inputs(64)[:5])


SM90 = pytest.mark.skipif(not available(), reason="the CuTe CP kernels run on SM90 only")


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def max_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.double() - y.double()).abs().max())


def per_value_head(x: torch.Tensor, hv: int) -> torch.Tensor:
    return x.repeat_interleave(hv // x.shape[2], dim=2).contiguous()


def chunk_transitions_float64(k, w, g_cum) -> list[torch.Tensor]:
    """
    Each chunk's state transition in float64, (HV, K, K), h_end = M_c h_start for the part of the
    chunk recurrence that is linear in h: M_c = exp2(g_last) I - (k * exp2(g_last - g))^T w, with
    k per value head and g_cum in log2 units (FLA's chunk algorithm without its roundings).
    """
    T, HV = w.shape[1], w.shape[2]
    k, w, g = per_value_head(k, HV)[0].double(), w[0].double(), g_cum[0].double()
    eye = torch.eye(D, dtype=torch.float64, device="cuda")
    out = []
    for start in range(0, T, BT):
        rows = slice(start, min(start + BT, T))
        g_c, g_last = g[rows], g[rows][-1]
        kd = k[rows] * torch.exp2(g_last - g_c)[..., None]
        out.append(torch.exp2(g_last)[:, None, None] * eye - torch.einsum("thk,thj->hkj", kd, w[rows]))  # fmt: skip
    return out


def chunk_states_float64(k, w, u, g_cum, h0=None) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:  # fmt: skip
    """
    FLA's forward state recurrence in float64 from h0 (HV, K, V) or zero: (the chunk-entry states,
    each (HV, K, V); v_new = u - w h per chunk, (T, HV, V); the final state).
    """
    T, HV = u.shape[1], u.shape[2]
    k, w, u, g = per_value_head(k, HV)[0].double(), w[0].double(), u[0].double(), g_cum[0].double()
    h = torch.zeros(HV, D, D, dtype=torch.float64, device="cuda") if h0 is None else h0.double()
    states, v_new = [], []
    for start in range(0, T, BT):
        rows = slice(start, min(start + BT, T))
        g_c, g_last = g[rows], g[rows][-1]
        states.append(h)
        vn = u[rows] - torch.einsum("thk,hkv->thv", w[rows], h)
        v_new.append(vn)
        kd = k[rows] * torch.exp2(g_last - g_c)[..., None]
        h = h * torch.exp2(g_last)[:, None, None] + torch.einsum("thk,thv->hkv", kd, vn)
    return states, torch.cat(v_new), h


def forward_map_float64(k, w, u, g_cum) -> torch.Tensor:
    """
    The rank's forward map in float64, (HV, K, V + K): he, the final state from zero (columns :V),
    and M, the product of the chunk transitions M_L ... M_1 (columns V:).
    """
    M = None
    for M_c in chunk_transitions_float64(k, w, g_cum):
        M = M_c if M is None else M_c @ M
    return torch.cat([chunk_states_float64(k, w, u, g_cum)[2], M], -1)


def merge_float64(ag, sources) -> torch.Tensor:
    """
    h = M_s h + he_s over the source ranks in order, from zero, in float64: (HV, K, V).
    """
    h = torch.zeros(ag.shape[1], D, D, dtype=torch.float64, device="cuda")
    for s_rank in sources:
        h = ag[s_rank, :, :, D:].double() @ h + ag[s_rank, :, :, :D].double()
    return h


def assert_near(got: torch.Tensor, ref: torch.Tensor, name: str, diff: float) -> None:
    """
    calc_diff below diff and the largest absolute error within 2e-2 of the reference's largest
    magnitude (the dense verifier's bounds), every value finite.
    """
    assert got.shape == ref.shape, name
    assert torch.isfinite(got).all(), name
    assert calc_diff(got, ref) < diff, f"{name}: calc_diff {calc_diff(got, ref):.2e}"
    bound = 2e-2 * float(ref.abs().max())
    assert max_abs(got, ref) <= bound, f"{name}: max abs err {max_abs(got, ref):.3e} > {bound:.3e}"


def chunk_dh_float64(q, k, w, do, dv_local, g_cum) -> torch.Tensor:
    """
    The reverse state-gradient recurrence from zero in float64, FLA's chunk algorithm without its
    bf16 roundings: dh (HV, K, V), the gradient of the slice's initial state.
    """
    T, HV = do.shape[1], do.shape[2]
    q, k = (per_value_head(x, HV)[0].double() for x in (q, k))
    w, do, dv, g = w[0].double(), do[0].double(), dv_local[0].double(), g_cum[0].double()
    dh = torch.zeros(HV, D, D, dtype=torch.float64, device="cuda")
    for start in reversed(range(0, T, BT)):
        rows = slice(start, min(start + BT, T))
        g_c, g_last = g[rows], g[rows][-1]
        du = dv[rows] + torch.einsum("thk,hkv->thv", k[rows], dh) * torch.exp2(g_last - g_c)[..., None]  # fmt: skip
        qg = q[rows] * torch.exp2(g_c)[..., None]
        dh = dh * torch.exp2(g_last)[:, None, None] + D**-0.5 * torch.einsum("thk,thv->hkv", qg, do[rows]) - torch.einsum("thk,thv->hkv", w[rows], du)  # fmt: skip
    return dh


def naive_gdr(q, k, v, g, beta, h0=None):
    """
    The fp32 recurrence per value head (q / k (1, T, HV, D)) from h0 (HV, K, V) or zero: (output,
    final state).
    """
    state = q.new_zeros(1, v.shape[2], D, D) if h0 is None else h0[None].clone()
    out = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state))
        state = state + k[:, t][..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * D**-0.5, state))
    return torch.stack(out, 1), state[0]


@SM90
def test_supported_at_qwen35_and_folded_geometries() -> None:
    for b in (1, 2, 3):
        for T in (1, 64, 4000):
            assert gated_delta_rule_cp_supported(*make_inputs(T, b)[:5]), (b, T)


@SM90
def test_not_supported_off_the_geometry() -> None:
    q, k, v, g, beta, _ = make_inputs(256)
    off = {
        "key dim 64": (q[..., :64].contiguous(), k[..., :64].contiguous(), v, g, beta),
        "value dim 64": (q, k, v[..., :64].contiguous(), g, beta),
        "fp32 q / k / v": (q.float(), k.float(), v.float(), g, beta),
        "fp32 beta": (q, k, v, g, beta.float()),
        "bf16 gate": (q, k, v, g.bfloat16(), beta),
        "non-contiguous v": (q, k, v.transpose(1, 2).contiguous().transpose(1, 2), g, beta),
        "value heads not a multiple of key heads": (
            q[:, :, :12].contiguous(),
            k[:, :, :12].contiguous(),
            v,
            g,
            beta,
        ),  # fmt: skip
        "value heads not twice the key heads": (
            q[:, :, :8].contiguous(),
            k[:, :, :8].contiguous(),
            v,
            g,
            beta,
        ),  # fmt: skip
        "16 value heads": (
            q[:, :, :8].contiguous(),
            k[:, :, :8].contiguous(),
            v[:, :, :16].contiguous(),
            g[:, :, :16].contiguous(),
            beta[:, :, :16].contiguous(),
        ),  # fmt: skip
        "batch 2 unfolded": tuple(torch.cat([x, x]) for x in (q, k, v, g, beta)),
        "rank-3 q / k": (q[0], k[0], v, g, beta),
        "on the host": (q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu()),
        "empty slice": (q[:, :0], k[:, :0], v[:, :0], g[:, :0], beta[:, :0]),
    }
    for name, args in off.items():
        assert not gated_delta_rule_cp_supported(*args), name


# (T, b, gate, resets): rank slices of the contract lengths, batch 1 and the folded batch 3
TRANSITION_CASES = [
    (64, 3, "model", ()),
    (64, 1, "mild", (0, 63)),
    (2048, 1, "mild", (0, 1, 63)),
    (2048, 3, "model", ()),
    (4000, 1, "model", (0, 64, 3999)),
    (4000, 3, "mild", (63,)),
    (4096, 1, "model", ()),
    (4096, 1, "mild", (4095,)),
    (8192, 1, "mild", ()),
    (8192, 3, "model", (0, 4096)),
]


@pytest.mark.parametrize("T,b,gate,resets", TRANSITION_CASES)
@SM90
def test_forward_map(T: int, b: int, gate: str, resets: tuple[int, ...]) -> None:
    """
    he and M against their float64 evaluation from the rank's own w, u and g_cum (M a TF32 chain).
    """
    q, k, v, g, beta, _ = make_inputs(T, b, gate, seed=T + b, resets=resets)
    g_cum, A, w, u, hm = gdr_cp_fwd_local(k, v, g, beta, True)
    assert hm.shape == (32 * b, D, 2 * D) and hm.dtype == torch.float32
    ref = forward_map_float64(k, w, u, g_cum)
    assert_near(hm[..., :D], ref[..., :D], "he", 1e-5)
    assert_near(hm[..., D:], ref[..., D:], "M", 1e-5)


@pytest.mark.parametrize("T,b,gate,resets", TRANSITION_CASES)
@SM90
def test_backward_map(T: int, b: int, gate: str, resets: tuple[int, ...]) -> None:
    """
    The dh half and M^T (the transposed chain M_1^T ... M_L^T) against their float64 evaluation of
    the chunk algorithm from the rank's own w, do, dv_local and g_cum.
    """
    q, k, v, g, beta, do = make_inputs(T, b, gate, seed=T + b + 1, resets=resets)
    g_cum, A, *_ = gdr_cp_fwd_local(k, v, g, beta, False)
    w, u, dv_local, dhm = gdr_cp_bwd_local(q, k, v, g_cum, beta, A, do, True)
    M = None
    for M_c in chunk_transitions_float64(k, w, g_cum):
        M = M_c if M is None else M_c @ M
    assert_near(dhm[..., D:], M.transpose(-1, -2), "M^T", 1e-5)
    assert_near(dhm[..., :D], chunk_dh_float64(q, k, w, do, dv_local, g_cum), "dh", 1e-5)


def gathered_maps(W: int, T: int, b: int, gate: str, backward: bool) -> torch.Tensor:
    """
    W ranks' real maps (independent slices), (W, HV, K, V + K).
    """
    maps = []
    for r in range(W):
        q, k, v, g, beta, do = make_inputs(T, b, gate, seed=100 + r)
        g_cum, A, w, u, hm = gdr_cp_fwd_local(k, v, g, beta, True)
        maps.append(gdr_cp_bwd_local(q, k, v, g_cum, beta, A, do, True)[3] if backward else hm)
    return torch.stack(maps)


@pytest.mark.parametrize("gate", ["model", "mild"])
@pytest.mark.parametrize("backward", [False, True])
@SM90
def test_merge_and_reads_only_its_ranks(gate: str, backward: bool) -> None:
    W, b = 4, 1
    ag = gathered_maps(W, 512, b, gate, backward)
    for rank in range(W):
        count = W - 1 - rank if backward else rank
        sources = range(rank + 1, rank + 1 + count) if backward else range(rank - count, rank)
        poisoned = ag.clone()
        for s in set(range(W)) - set(sources):
            poisoned[s] = float("nan")
        got = gdr_cp_merge(poisoned, rank, count, backward)
        assert got.shape == (32 * b, D, D) and got.dtype == torch.float32
        if count == 0:
            assert torch.equal(got, torch.zeros_like(got)), rank
        else:
            order = reversed(sources) if backward else sources
            assert_near(got, merge_float64(ag, order), f"rank {rank}", 1e-6)


@SM90
def test_merge_order_and_membership_are_checked() -> None:
    """
    The merge comparison sees a fold in the wrong order and a fold that includes the rank itself.
    """
    ag = gathered_maps(4, 512, 1, "mild", False)
    reference = merge_float64(ag, range(3))
    assert_near(gdr_cp_merge(ag, 3, 3, False), reference, "merge", 1e-6)
    reversed_order = gdr_cp_merge(ag[[2, 1, 0, 3]].contiguous(), 3, 3, False)
    including_self = gdr_cp_merge(ag, 4, 4, False)  # ranks 0 .. 3, the rank 3 itself among them
    assert calc_diff(reversed_order, reference) > 1e-4
    assert calc_diff(including_self, reference) > 1e-4


@pytest.mark.parametrize("T,gate", [(64, "model"), (4000, "mild"), (4096, "model")])
@SM90
def test_states_and_output_from_h0(T: int, gate: str) -> None:
    """
    The chunk-entry states and v_new against FLA's state recurrence in float64 from the same w, u
    and h0, and the output against the fp32 token-by-token recurrence from h0.
    """
    q, k, v, g, beta, _ = make_inputs(T, 1, gate, seed=T + 7)
    g_cum, A, w, u, _ = gdr_cp_fwd_local(k, v, g, beta, False)
    h0 = torch.randn(32, D, D, device="cuda") * 0.1
    h, v_new = gdr_cp_bwd_states(k, w, u, g_cum, h0)
    states, v_new_ref, _ = chunk_states_float64(k, w, u, g_cum, h0)
    assert_near(h[0], torch.stack(states).transpose(-1, -2), "h", 1e-5)  # stored V-first
    assert_near(v_new[0], v_new_ref, "v_new", 1e-5)
    o = gdr_cp_fwd_output(q, k, w, u, g_cum, h0)
    o_ref, _ = naive_gdr(per_value_head(q, 32).float(), per_value_head(k, 32).float(), v.float(), g, beta.float(), h0=h0)  # fmt: skip
    assert_near(o, o_ref, "o", 1e-4)


@pytest.mark.parametrize("T,gate", [(256, "model"), (1000, "mild"), (2048, "model")])
@SM90
def test_gradients_from_h0_and_dht(T: int, gate: str) -> None:
    """
    The five gradients from h0 and dht against the fp32 recurrence's autograd (with h0, and with
    <h_T, dht> added to the loss), with the dense verifier's bounds.
    """
    q, k, v, g, beta, do = make_inputs(T, 1, gate, seed=T + 9)
    gen = torch.Generator(device="cuda").manual_seed(T)
    h0 = torch.randn(32, D, D, device="cuda", generator=gen) * 0.1
    dht = torch.randn(32, D, D, device="cuda", generator=gen) * 0.1
    g_cum, A, w, u, _ = gdr_cp_fwd_local(k, v, g, beta, False)
    w, u, dv_local, _ = gdr_cp_bwd_local(q, k, v, g_cum, beta, A, do, False)
    h, v_new = gdr_cp_bwd_states(k, w, u, g_cum, h0)
    ours = gdr_cp_bwd_output(q, k, v, w, v_new, do, dv_local, h, g_cum, beta, A, u, dht)
    q_rep, k_rep = per_value_head(q, 32), per_value_head(k, 32)
    leaves = [x.detach().float().requires_grad_() for x in (q_rep, k_rep, v, g, beta)]
    o_ref, ht_ref = naive_gdr(*leaves, h0=h0)
    ((o_ref * do.float()).sum() + (ht_ref * dht).sum()).backward()
    for name, got, leaf in zip(GRADS, ours, leaves):
        assert_near(got, leaf.grad, name, 1e-3)


@pytest.mark.parametrize("residue", [0, 1, 63])
@SM90
def test_reset_zeroes_the_transition(residue: int) -> None:
    T = 1000
    q, k, v, g, beta, do = make_inputs(T, 1, "mild", seed=residue, resets=(5 * BT + residue,))
    g_cum, A, w, u, hm = gdr_cp_fwd_local(k, v, g, beta, True)
    *_, dhm = gdr_cp_bwd_local(q, k, v, g_cum, beta, A, do, True)
    assert torch.equal(hm[..., D:], torch.zeros_like(hm[..., D:])), "M"
    assert torch.equal(dhm[..., D:], torch.zeros_like(dhm[..., D:])), "M^T"
    before = gathered_maps(1, 512, 1, "mild", False)
    merged = gdr_cp_merge(torch.cat([before, hm[None]]), 2, 2, False)
    assert torch.equal(merged, hm[..., :D])


def run_chain(slices: list, b: int) -> tuple[torch.Tensor, tuple]:
    """
    The CP chain emulated in one process: W ranks one after another, the all-gather a stack. Returns
    the stitched output and the five gradients (dq, dk summed over each value-head group).
    """
    W = len(slices)
    local = [gdr_cp_fwd_local(k, v, g, beta, r < W - 1) for r, (q, k, v, g, beta, do) in enumerate(slices)]  # fmt: skip
    ag = torch.stack([state[4] for state in local])
    h0s = [gdr_cp_merge(ag, r, r, False) if r > 0 else None for r in range(W)]
    outs = [gdr_cp_fwd_output(s[0], s[1], st[2], st[3], st[0], h0) for s, st, h0 in zip(slices, local, h0s)]  # fmt: skip
    back = [gdr_cp_bwd_local(s[0], s[1], s[2], st[0], s[4], st[1], s[5], r > 0) for r, (s, st) in enumerate(zip(slices, local))]  # fmt: skip
    ag_back = torch.stack([state[3] for state in back])
    grads = []
    for r, (s, st, bk, h0) in enumerate(zip(slices, local, back, h0s)):
        q, k, v, g, beta, do = s
        w, u, dv_local, _ = bk
        h, v_new = gdr_cp_bwd_states(k, w, u, st[0], h0)
        dht = gdr_cp_merge(ag_back, r, W - 1 - r, True) if r < W - 1 else None
        dq, dk, dv, dg, dbeta = gdr_cp_bwd_output(q, k, v, w, v_new, do, dv_local, h, st[0], beta, st[1], u, dht)  # fmt: skip
        T, H = k.shape[1], k.shape[2]
        grads.append((dq.view(1, T, H, 2, D).sum(3), dk.view(1, T, H, 2, D).sum(3), dv, dg, dbeta))
    return torch.cat(outs, 1), tuple(torch.cat(parts, 1) for parts in zip(*grads))


# (W, T per rank, b, gate, resets in the whole sequence)
CHAIN_CASES = [
    (2, 64, 3, "model", ()),
    (2, 1000, 1, "mild", (1000,)),
    (4, 256, 1, "mild", (300, 777)),
    (4, 300, 1, "model", (0, 299, 600)),
    (4, 500, 3, "mild", (250, 1500)),
]


@pytest.mark.parametrize("W,T,b,gate,resets", CHAIN_CASES)
@SM90
def test_chain_against_recurrence(
    W: int, T: int, b: int, gate: str, resets: tuple[int, ...]
) -> None:
    q, k, v, g, beta, do = make_inputs(W * T, b, gate, seed=W * T + b, resets=resets)
    cut = lambda x, r: x[:, r * T : (r + 1) * T].contiguous()
    slices = [tuple(cut(x, r) for x in (q, k, v, g, beta, do)) for r in range(W)]
    o, grads = run_chain(slices, b)
    hv = v.shape[2]
    leaves = [x.detach().float().requires_grad_() for x in (q, k, v, g, beta)]
    o_ref, _ = naive_gdr(per_value_head(leaves[0], hv), per_value_head(leaves[1], hv), *leaves[2:])
    o_ref.backward(do.float())
    assert calc_diff(o, o_ref) < 1e-4
    assert max_abs(o, o_ref) <= 2e-2 * float(o_ref.abs().max())
    for name, got, leaf in zip(GRADS, grads, leaves):
        assert got.shape == leaf.shape, name
        assert torch.isfinite(got).all(), name
        assert calc_diff(got, leaf.grad) < 1e-3, (name, calc_diff(got, leaf.grad))
        assert max_abs(got, leaf.grad) <= 2e-2 * float(leaf.grad.abs().max()), name


@SM90
def test_chain_without_handoff_is_detected() -> None:
    """
    Dropping the forward merge (h0 = 0) or the backward one (dht = 0) fails the comparison above:
    the second rank's output, and with the loss on the second rank only, the first rank's dk and dv.
    """
    W, T = 2, 512
    q, k, v, g, beta, do = make_inputs(W * T, 1, "mild", seed=3)
    do[:, :T] = 0
    leaves = [x.detach().float().requires_grad_() for x in (q, k, v, g, beta)]
    o_ref, _ = naive_gdr(per_value_head(leaves[0], 32), per_value_head(leaves[1], 32), *leaves[2:])
    o_ref.backward(do.float())
    second = tuple(x[:, T:].contiguous() for x in (q, k, v, g, beta, do))
    g_cum, A, w, u, _ = gdr_cp_fwd_local(second[1], second[2], second[3], second[4], False)
    o_alone = gdr_cp_fwd_output(second[0], second[1], w, u, g_cum, None)
    assert calc_diff(o_alone, o_ref[:, T:]) > 1e-4
    first = tuple(x[:, :T].contiguous() for x in (q, k, v, g, beta, do))
    g_cum, A, w, u, _ = gdr_cp_fwd_local(first[1], first[2], first[3], first[4], True)
    w, u, dv_local, _ = gdr_cp_bwd_local(first[0], first[1], first[2], g_cum, first[4], A, first[5], False)  # fmt: skip
    h, v_new = gdr_cp_bwd_states(first[1], w, u, g_cum, None)
    _, dk, dv, _, _ = gdr_cp_bwd_output(first[0], first[1], first[2], w, v_new, first[5], dv_local, h, g_cum, first[4], A, u, None)  # fmt: skip
    assert calc_diff(dk.view(1, T, 16, 2, D).sum(3), leaves[1].grad[:, :T]) > 1e-3
    assert calc_diff(dv, leaves[2].grad[:, :T]) > 1e-3
    run = run_chain([tuple(x[:, r * T : (r + 1) * T].contiguous() for x in (q, k, v, g, beta, do)) for r in range(W)], 1)  # fmt: skip
    assert (
        calc_diff(run[1][1], leaves[1].grad) < 1e-3 and calc_diff(run[1][2], leaves[2].grad) < 1e-3
    )


@SM90
def test_deterministic() -> None:
    slices = [make_inputs(1000, 1, "mild", seed=s) for s in range(3)]
    first = run_chain(slices, 1)
    second = run_chain(slices, 1)
    assert torch.equal(first[0], second[0])
    for name, x, y in zip(GRADS, first[1], second[1]):
        assert torch.equal(x, y), name


CUTE = ("WyRepresentation", "StateRecompute", "TransitionChain", "StateMerge", "ChunkStateOutput", "BwdPrologue", "StateGradient", "ChunkGradients")  # fmt: skip


def launched(fn) -> list[str]:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return sorted(next((c for c in CUTE if c in n), n) for n in names)


@pytest.mark.parametrize("T", [64, 4000])
@SM90
def test_only_cute_kernels(T: int) -> None:
    """
    No Triton or other kernel besides the CuTe ones (and memset / copies of torch's allocations).
    """
    slices = [make_inputs(T, 1, "mild", seed=s) for s in range(3)]
    run_chain(slices, 1)
    names = launched(lambda: run_chain(slices, 1))
    others = [n for n in names if n not in CUTE and "elementwise" not in n.lower() and "copy" not in n.lower() and "fill" not in n.lower() and "reduce" not in n.lower() and "cat" not in n.lower()]  # fmt: skip
    assert not others, others
    assert set(CUTE) <= set(names)


COMPILE_COUNT = """
import sys, torch
import cutlass.cute as cute
sys.path.insert(0, {tests!r})
compiles = []
compile_kernel = cute.compile
cute.compile = lambda kernel, *args, **kwargs: compiles.append(type(kernel).__name__) or compile_kernel(kernel, *args, **kwargs)
from test_gated_delta_rule_cp import make_inputs, run_chain
for W in (2, 3, 4):
    for T in (64, 128, 192, 1000, 1001):
        run_chain([make_inputs(T, 1, "mild", seed=s) for s in range(W)], 1)
torch.cuda.synchronize()
print(" ".join(sorted(compiles)))
"""


@SM90
def test_one_compile_per_kernel_variant() -> None:
    """
    A sweep of lengths, group sizes and ranks compiles each kernel variant once (aligned and ragged,
    with and without the CP flags), nothing per length, rank or group size.
    """
    code = COMPILE_COUNT.format(tests=os.path.dirname(__file__))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env)  # fmt: skip
    expected = (
        # aligned and ragged each: WyRepresentation, ChunkStateOutput with and without h0, BwdPrologue,
        # StateRecompute (final state only; from h0; from zero), TransitionChain forward and reverse,
        # StateGradient (initial gradient only; from dht; from zero), ChunkGradients
        2 * ["WyRepresentation", "BwdPrologue", "ChunkGradients"]
        + 4 * ["ChunkStateOutput"]
        + 6 * ["StateRecompute", "StateGradient"]
        + 4 * ["TransitionChain"]
        + 2 * ["StateMerge"]
    )
    assert out.stdout.split() == sorted(expected), out.stdout


# The SM100 TIRx family.

H, HV, D = 16, 32, 128  # Qwen3.5-35B-A3B: 16 key heads, 32 value heads, head dims 128


def tirx_meta_inputs(S: int = 256, B: int = 1, heads: int = HV, dim: int = D) -> list[torch.Tensor]:
    bf16 = torch.bfloat16
    qkv = [torch.empty(B, S, heads, dim, dtype=bf16, device="meta") for _ in range(3)]
    g = torch.empty(B, S, heads, dtype=torch.float32, device="meta")
    beta = torch.empty(B, S, heads, dtype=bf16, device="meta")
    return [*qkv, g, beta]


SM100 = pytest.mark.skipif(
    not torch.cuda.is_available() or not chunk_gdr_cp_supported(*tirx_meta_inputs()),
    reason="the TIRx CP kernels run on SM100 only",
)


def tirx_make_inputs(
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


def tirx_rank_context(cu: list[int], world: int, rank: int) -> tuple:
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


def tirx_make_plans(cu: list[int], world: int) -> list[dict]:
    plans = []
    for rank in range(world):
        local, first, last, pre, post = tirx_rank_context(cu, world, rank)
        cu_cpu = torch.tensor(local, dtype=torch.int32)
        plans.append(gated_delta_rule_cp_plan(cu_cpu.cuda(), cu_cpu, rank, first, last, pre, post))
    return plans


def tirx_stack(payloads: list[torch.Tensor]) -> torch.Tensor:
    """
    What the all-gather returns: every rank's payload in rank order.
    """
    return torch.stack(payloads)


def tirx_run_cp(q, k, v, g, beta, do, world: int, cu: list[int], gather=tirx_stack) -> dict:
    """
    One emulated CP step over the whole sequence: the shards' results concatenated, and the
    payloads.
    """
    S = q.shape[1] // world
    plans = tirx_make_plans(cu, world)

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


def tirx_calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def tirx_naive_gdr(q, k, v, g, beta, cu: list[int]):
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


def tirx_chunked_gdr(q, k, v, g, beta, cu: list[int], chunk: int = 64):
    """
    The naive recurrence in chunked form: within a chunk entering with state S0 and cumulative gate G (Gamma =
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


def tirx_assert_close_to(got: torch.Tensor, ref: torch.Tensor, name: str, diff: float) -> None:
    assert got.shape == ref.shape, name
    assert torch.isfinite(got).all(), name
    diff_value = tirx_calc_diff(got, ref)
    assert diff_value < diff, f"{name}: calc_diff {diff_value:.2e}"
    err = float((got.float() - ref).abs().max())
    bound = 2e-2 * float(ref.abs().max())
    assert err <= bound, f"{name}: max abs err {err:.3e} > {bound:.3e}"


@SM100
def test_supported_at_the_qwen35_geometry() -> None:
    for S in (1, 100, 1024, 8192):
        assert chunk_gdr_cp_supported(*tirx_meta_inputs(S)), S


@SM100
def test_tirx_not_supported_off_the_geometry() -> None:
    q, k, v, g, beta = tirx_meta_inputs()
    off = {
        "batch 2": tirx_meta_inputs(B=2),
        "16 heads": tirx_meta_inputs(heads=16),
        "64 value heads (folded batch)": tirx_meta_inputs(heads=64),
        "dim 64": tirx_meta_inputs(dim=64),
        "empty": tirx_meta_inputs(S=0),
        "unrepeated key heads": (q[:, :, :H], k[:, :, :H], v, g, beta),
        "value dim 64": (q, k, v[..., :64], g, beta),
        "bf16 gate": (q, k, v, g.to(torch.bfloat16), beta),
        "fp32 q / k / v": (q.float(), k.float(), v.float(), g, beta),
        "fp32 beta": (q, k, v, g, beta.float()),
    }
    for name, args in off.items():
        assert not chunk_gdr_cp_supported(*args), name


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
@SM100
def test_against_naive_recurrence(world: int, S: int, cu, gate: str, resets) -> None:
    T = world * S
    cu = cu or [0, T]
    q, k, v, g, beta, do = tirx_make_inputs(T, gate=gate, seed=T + world, resets=resets)
    assert chunk_gdr_cp_supported(*(t[:, :S] for t in (q, k, v, g, beta)))
    got = tirx_run_cp(q, k, v, g, beta, do, world, cu)

    for o, g_cum, A, initial_state in got["fwd"]:
        assert o.shape == (1, S, HV, D) and o.dtype == torch.bfloat16
        assert g_cum.shape == (1, S, HV) and g_cum.dtype == torch.float32
        assert A.shape == (1, S, HV, 64) and A.dtype == torch.bfloat16
        assert initial_state.shape == (1, HV, D, D) and initial_state.dtype == torch.float32
    for r in range(world):
        _, first, last, _, _ = tirx_rank_context(cu, world, r)
        for name, payload, unread in (("hm", got["hm"][r], last), ("dhm", got["dhm"][r], first)):
            assert payload.shape == (HV, D, 2 * D) and payload.dtype == torch.float32, name
            if unread:
                assert not payload.any(), f"{name} of rank {r} is read by no rank but is not zero"
    for name in ("dq", "dk", "dv", "dbeta"):
        assert got[name].dtype == torch.bfloat16, name
    assert got["dg"].dtype == torch.float32

    leaves = [t.double().requires_grad_() for t in (q, k, v, g, beta)]
    ref = tirx_chunked_gdr(*leaves, cu)
    ref.backward(do.double())

    tirx_assert_close_to(got["o"], ref.detach(), "o", 1e-4)
    for name, leaf in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves):
        tirx_assert_close_to(got[name], leaf.grad, name, 1e-3)


@SM100
def test_reference_is_the_recurrence() -> None:
    T, cu = 300, [0, 100, 164, 300]
    q, k, v, g, beta, do = tirx_make_inputs(T, gate="model", seed=5, resets=(200,))
    leaves = [t.double().requires_grad_() for t in (q, k, v, g, beta)]
    twins = [t.detach().clone().requires_grad_() for t in leaves]
    naive, chunked = tirx_naive_gdr(*leaves, cu), tirx_chunked_gdr(*twins, cu)
    naive.backward(do.double())
    chunked.backward(do.double())
    assert tirx_calc_diff(chunked, naive) < 1e-10
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), twins, leaves):
        assert tirx_calc_diff(a.grad, b.grad) < 1e-9, name


@SM100
def test_comparison_is_sensitive() -> None:
    world, S = 4, 256
    q, k, v, g, beta, _ = tirx_make_inputs(world * S, gate="mild", seed=3)
    cu = [0, world * S]
    ref = tirx_chunked_gdr(q.double(), k.double(), v.double(), g.double(), beta.double(), cu)
    assert tirx_calc_diff(tirx_run_cp(q, k, v, g, beta, None, world, cu)["o"], ref) < 1e-4
    assert tirx_calc_diff(tirx_run_cp(q, k, v, g * 0.9, beta, None, world, cu)["o"], ref) > 1e-4
    # The merge reads every predecessor's payload by its rank: a stack out of rank order fails.
    reverse = tirx_run_cp(q, k, v, g, beta, None, world, cu, gather=lambda p: torch.stack(p[::-1]))
    assert tirx_calc_diff(reverse["o"], ref) > 1e-4
