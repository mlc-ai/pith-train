"""
Verifier of the context-parallel chunked gated delta rule pieces (gated_delta_rule_cp_supported,
gdr_cp_fwd_local, gdr_cp_merge, gdr_cp_fwd_output, gdr_cp_bwd_local, gdr_cp_bwd_states,
gdr_cp_bwd_output): FLA 0.5.2's context-parallel structure on CuTe-DSL kernels, SM90 only (every
test skips where the capability query is False).

A sequence is split into W consecutive rank slices. Rank r's affine state map (h_end = M h_start +
he) comes from its own inputs. The maps are gathered and folded into each rank's initial state h0,
and the backward does the same in reverse for the gradient of each rank's final state.

Checked, against PyTorch and, as test-only oracles, against FLA's own context-parallel kernels
(fla/ops/cp/chunk_delta_h.py) with their autotuners pinned to the configs FLA picks on H200:

- the capability query: True at Qwen3.5's geometry and its batch folds (H 16 b / HV 32 b), False
  off it;
- the exactness ladder (kda-task/docs/r3-preregistration.md), single GPU, synthetic inputs, model
  and mild gates, resets, aligned and ragged slices, HV 32 and 96:
  - the forward map hm, he and M, bit for bit FLA's pre_process_fwd_kernel_merged;
  - the merge in both directions, bit for bit FLA's merge_fwd_bwd_kernel, never reading the slots
    outside its ranks (they are NaN-poisoned);
  - the backward map's M^T half bit for bit FLA's pre_process_bwd_kernel_merged. Its dh half is
    no further from a float64 evaluation of the same chunk algorithm than FLA's is: calc_diff
    within 1.001 x FLA's + 1e-12, max-abs within 1.10 x FLA's;
  - the states recomputed from h0 bit for bit FLA's chunk_gated_delta_rule_fwd_h(h0), and the
    output from h0 within calc_diff 1e-6 of FLA's fwd_h(h0) + chunk_fwd_o;
  - the gradients from h0 and dht no further from the fp32 recurrence (with h0, and with <h_T,
    dht> added to the loss) than FLA's chunk_gated_delta_rule_bwd: 1.10 x FLA's + 1e-7 in
    calc_diff and in max-abs, per gradient;
  - a reset (g = -128) at chunk residue 0, 1 or 63 zeroes M and M^T exactly, and a merge through
    such a rank yields exactly that rank's he;
- the whole chain emulated in one process (W ranks run one after another, the all-gather a stack)
  against the fp32 recurrence over the whole sequence, output and the five gradients, documents
  crossing rank boundaries, aligned and ragged slices, batch 1 and 3 folded;
- determinism (repeated calls bit-identical), only the CuTe kernels launching, and one compile
  per kernel variant over a sweep of lengths, ranks and group sizes.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from pithtrain.kernels import (
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


pytestmark = pytest.mark.skipif(not available(), reason="the CP kernels run on SM90 only")


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def max_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.double() - y.double()).abs().max())


def per_value_head(x: torch.Tensor, hv: int) -> torch.Tensor:
    return x.repeat_interleave(hv // x.shape[2], dim=2).contiguous()


# FLA's autotuned picks on H200 at this geometry (FLA 0.5.2, Triton 3.7.1), pinned so the oracle
# is the same kernel on every run.
RECORDED = {
    "pre_fwd": dict(num_warps=4, num_stages=2),
    "merge": dict(BV=32, num_warps=4, num_stages=2),
    "pre_bwd": dict(num_warps=4, num_stages=3),
}


@pytest.fixture
def fla(monkeypatch: pytest.MonkeyPatch):
    """
    FLA's context-parallel kernels with their autotuners restricted to RECORDED.
    """
    import triton
    from fla.ops.cp import chunk_delta_h

    kernels = {
        "pre_fwd": chunk_delta_h.pre_process_fwd_kernel_merged,
        "merge": chunk_delta_h.merge_fwd_bwd_kernel,
        "pre_bwd": chunk_delta_h.pre_process_bwd_kernel_merged,
    }
    for name, kernel in kernels.items():
        tuner = kernel
        while not isinstance(tuner, triton.runtime.autotuner.Autotuner):
            tuner = tuner.fn
        pick = dict(RECORDED[name])
        launch = {key: pick.pop(key) for key in ("num_warps", "num_stages")}
        monkeypatch.setattr(tuner, "configs", [triton.Config(pick, **launch)])
        monkeypatch.setattr(tuner, "cache", {})
    return kernels


def fla_pre_fwd(kernels, k, w, u, g_cum) -> torch.Tensor:
    T, H, HV = k.shape[1], k.shape[2], u.shape[2]
    hm = torch.zeros(HV, D, 2 * D, device="cuda")
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    kernels["pre_fwd"][(4, HV)](k=k, v=u, w=w, g=g_cum, gk=None, bg=None, u=u, hm=hm, cu_seqlens=cu, T=T, H=H, HV=HV, K=D, V=D, BT=BT, BK1=D, BLOCK_SIZE=64, MULTI_SEQS=False)  # fmt: skip
    return hm


def fla_pre_bwd(kernels, q, k, w, do, dv_local, g_cum) -> torch.Tensor:
    T, H, HV = k.shape[1], k.shape[2], do.shape[2]
    dhm = torch.zeros(HV, D, 2 * D, device="cuda")
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    kernels["pre_bwd"][(4, HV)](q=q, k=k, w=w, g=g_cum, gk=None, do=do, dhm=dhm, dv=dv_local, cu_seqlens=cu, scale=D**-0.5, T=T, H=H, HV=HV, K=D, V=D, BT=BT, BK1=D, BLOCK_SIZE=64, USE_BG=False)  # fmt: skip
    return dhm


def fla_merge(kernels, ag, rank, count, forward) -> torch.Tensor:
    HV = ag.shape[1]
    h = torch.zeros(HV, D, D, device="cuda")
    kernels["merge"][(D // RECORDED["merge"]["BV"], HV)](h=h, ag_hm=ag, pre_or_post_num_ranks=count, rank=rank, seq_offsets=None, init_offsets=None, h0_seq_ids=None, h0=None, HV=HV, K=D, V=D, BK=D, FORWARD=forward, INTRACARD_MODE=False, NUM_SEQ_ENTRIES=0, STATE_V_FIRST=False)  # fmt: skip
    return h


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


def test_supported_at_qwen35_and_folded_geometries() -> None:
    for b in (1, 2, 3):
        for T in (1, 64, 4000):
            assert gated_delta_rule_cp_supported(*make_inputs(T, b)[:5]), (b, T)


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
def test_forward_map_is_fla(fla, T: int, b: int, gate: str, resets: tuple[int, ...]) -> None:
    q, k, v, g, beta, _ = make_inputs(T, b, gate, seed=T + b, resets=resets)
    g_cum, A, w, u, hm = gdr_cp_fwd_local(k, v, g, beta, True)
    assert hm.shape == (32 * b, D, 2 * D) and hm.dtype == torch.float32
    assert torch.equal(hm, fla_pre_fwd(fla, k, w, u, g_cum))


@pytest.mark.parametrize("T,b,gate,resets", TRANSITION_CASES)
def test_backward_map(fla, T: int, b: int, gate: str, resets: tuple[int, ...]) -> None:
    """
    M^T bit for bit FLA's; the dh half no further from the float64 chunk algorithm than FLA's.
    """
    q, k, v, g, beta, do = make_inputs(T, b, gate, seed=T + b + 1, resets=resets)
    g_cum, A, *_ = gdr_cp_fwd_local(k, v, g, beta, False)
    w, u, dv_local, dhm = gdr_cp_bwd_local(q, k, v, g_cum, beta, A, do, True)
    ref = fla_pre_bwd(fla, q, k, w, do, dv_local, g_cum)
    assert torch.equal(dhm[..., D:], ref[..., D:]), "M^T"
    exact = chunk_dh_float64(q, k, w, do, dv_local, g_cum)
    ours, theirs = dhm[..., :D], ref[..., :D]
    assert calc_diff(ours, exact) <= 1.001 * calc_diff(theirs, exact) + 1e-12
    assert max_abs(ours, exact) <= 1.10 * max_abs(theirs, exact)


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
def test_merge_is_fla_and_reads_only_its_ranks(fla, gate: str, backward: bool) -> None:
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
            assert torch.equal(got, fla_merge(fla, ag, rank, count, not backward)), rank


def test_merge_order_and_membership_are_checked(fla) -> None:
    """
    The merge comparison sees a fold in the wrong order and a fold that includes the rank itself.
    """
    ag = gathered_maps(4, 512, 1, "mild", False)
    reference = fla_merge(fla, ag, 3, 3, True)
    assert torch.equal(gdr_cp_merge(ag, 3, 3, False), reference)
    reversed_order = gdr_cp_merge(ag[[2, 1, 0, 3]].contiguous(), 3, 3, False)
    including_self = gdr_cp_merge(ag, 4, 4, False)  # ranks 0 .. 3, the rank 3 itself among them
    assert not torch.equal(reversed_order, reference)
    assert not torch.equal(including_self, reference)


@pytest.mark.parametrize("T,gate", [(64, "model"), (4000, "mild"), (4096, "model")])
def test_states_and_output_from_h0(T: int, gate: str) -> None:
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o

    q, k, v, g, beta, _ = make_inputs(T, 1, gate, seed=T + 7)
    g_cum, A, w, u, _ = gdr_cp_fwd_local(k, v, g, beta, False)
    h0 = torch.randn(32, D, D, device="cuda") * 0.1
    h, v_new = gdr_cp_bwd_states(k, w, u, g_cum, h0)
    k_rep, q_rep = per_value_head(k, 32), per_value_head(q, 32)
    h_ref, v_new_ref, _ = chunk_gated_delta_rule_fwd_h(k=k_rep, w=w, u=u, g=g_cum, initial_state=h0[None])  # fmt: skip
    assert torch.equal(h, h_ref.transpose(-1, -2)), "h"
    assert torch.equal(v_new, v_new_ref), "v_new"
    o = gdr_cp_fwd_output(q, k, w, u, g_cum, h0)
    o_ref = chunk_fwd_o(q=q_rep, k=k_rep, v=v_new_ref, h=h_ref, g=g_cum, scale=D**-0.5)
    assert calc_diff(o, o_ref) < 1e-6


@pytest.mark.parametrize("T,gate", [(256, "model"), (1000, "mild"), (2048, "model")])
def test_gradients_from_h0_and_dht_no_worse_than_fla(T: int, gate: str) -> None:
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd

    q, k, v, g, beta, do = make_inputs(T, 1, gate, seed=T + 9)
    gen = torch.Generator(device="cuda").manual_seed(T)
    h0 = torch.randn(32, D, D, device="cuda", generator=gen) * 0.1
    dht = torch.randn(32, D, D, device="cuda", generator=gen) * 0.1
    g_cum, A, w, u, _ = gdr_cp_fwd_local(k, v, g, beta, False)
    w, u, dv_local, _ = gdr_cp_bwd_local(q, k, v, g_cum, beta, A, do, False)
    h, v_new = gdr_cp_bwd_states(k, w, u, g_cum, h0)
    ours = gdr_cp_bwd_output(q, k, v, w, v_new, do, dv_local, h, g_cum, beta, A, u, dht)
    q_rep, k_rep = per_value_head(q, 32), per_value_head(k, 32)
    dq, dk, dv, db, dg, *_ = chunk_gated_delta_rule_bwd(q=q_rep, k=k_rep, v=v, g=g_cum, beta=beta, A=A, scale=D**-0.5, initial_state=h0[None], do=do, dht=dht[None])  # fmt: skip
    theirs = (dq, dk, dv, dg, db)
    leaves = [x.detach().float().requires_grad_() for x in (q_rep, k_rep, v, g, beta)]
    o_ref, ht_ref = naive_gdr(*leaves, h0=h0)
    ((o_ref * do.float()).sum() + (ht_ref * dht).sum()).backward()
    for name, got, fla_grad, leaf in zip(GRADS, ours, theirs, leaves):
        assert got.shape == fla_grad.shape, name
        assert calc_diff(got, leaf.grad) <= 1.10 * calc_diff(fla_grad, leaf.grad) + 1e-7, name
        assert max_abs(got, leaf.grad) <= 1.10 * max_abs(fla_grad, leaf.grad) + 1e-7, name


@pytest.mark.parametrize("residue", [0, 1, 63])
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
