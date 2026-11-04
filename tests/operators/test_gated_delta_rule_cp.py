"""
The context-parallel gated delta rule through gated_delta_rule(..., cp_group) at Qwen3.5's geometry
(16 key / 32 value heads, dims 128, b samples folded to 16 b / 32 b), on CP2 and CP4 (one process
per GPU, the contiguous layout). The output and all five input gradients (dq, dk, dv, dg, dbeta),
the per-rank shards stitched, are compared with a single-rank fp32 recurrence over the whole
sequence: calc_diff below 1e-4 (output) and 1e-3 (gradients), max abs error within 2e-2 of the
reference's largest magnitude. The reference is the recurrence in chunked form, exact in
arithmetic and with autograd through each chunk, so long sequences fit in memory. On a short
sequence it agrees with the token-by-token loop to calc_diff 1e-6.

Named cases on each group size: rank slices of 64 tokens (batch 3, folded), 2048, 4096, 4000
(ragged) and 8192; packed documents (g = -128 at a document start), with a document boundary
exactly on a rank boundary, a document spanning more than two ranks, a document held entirely by
one rank (the others hold none of its tokens), a rank entirely inside one document, and resets at
chunk residues 0, 1 and 63. Also checked:
- dh0 crosses ranks: a loss on the last rank's output alone gives the first rank's dk, dv, dg and
  dbeta non-zero gradients that match the reference;
- the first rank skips the forward merge, the last rank skips the forward transition (and, in
  the backward, the reverse merge, the first rank the backward transition), and every rank issues
  the same collectives in the same order;
- gathered slots a rank must not read, NaN-poisoned after the collective, change nothing;
- without the forward or the backward merge, or with a reset shifted by one token, the
  comparisons fail.
"""

from dataclasses import dataclass, field

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

import pithtrain.operators.gated_delta_rule as op
from pithtrain.contexts import distributed
from pithtrain.kernels import gated_delta_rule_cp_supported
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.gated_delta_rule import gated_delta_rule
from tests.utilities import launch

D, BT = 128, 64
NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    x = torch.zeros(1, 64, 16, D, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(1, 64, 32, D, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros(1, 64, 32, device="cuda")
    return gated_delta_rule_cp_supported(x, x, v, g, g.bfloat16())


pytestmark = pytest.mark.skipif(not available(), reason="the CP kernels run on SM90 only")


@dataclass
class Case:
    name: str
    T: int  # per rank
    b: int = 1
    gate: str = "model"
    resets: tuple = ()  # global positions, as fractions of a rank slice: (rank, offset)
    last_only: bool = False  # the loss on the last rank's output only


@dataclass
class Request:
    cases: list = field(default_factory=list)


def make_inputs(case: Case, W: int, device) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device=device).manual_seed(case.T * 7 + case.b + len(case.resets))
    S, H, HV = W * case.T, 16 * case.b, 32 * case.b

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=gen)

    q = F.normalize(randn(case.b, S, 16, D).to(torch.bfloat16), dim=-1)
    k = F.normalize(randn(case.b, S, 16, D).to(torch.bfloat16), dim=-1)
    v = randn(case.b, S, 32, D).to(torch.bfloat16)
    if case.gate == "model":
        a_log = torch.log(torch.rand(32, device=device, generator=gen) * 16)
        g = -a_log.exp() * F.softplus(randn(case.b, S, 32) + 1.0)
    else:
        g = -torch.rand(case.b, S, 32, device=device, generator=gen) * 0.02
    for rank, offset in case.resets:
        g[:, rank * case.T + offset] = -128.0
    beta = randn(case.b, S, 32).to(torch.bfloat16).sigmoid()
    do = randn(case.b, S, 32, D).to(torch.bfloat16)
    if case.last_only:
        do[:, : (W - 1) * case.T] = 0
    assert H == 16 * case.b and HV == 32 * case.b
    return q, k, v, g, beta, do


def reference_chunked(q, k, v, g, beta):
    """
    The recurrence S = S e^g; u = beta (v - S^T k); S = S + k u^T; o = scale q^T S, per value head
    (q / k (B, T, HV, D) fp32), evaluated chunk by chunk with the exact intra-chunk solve.
    """
    B, T, HV, _ = v.shape
    S = q.new_zeros(B, HV, D, D)
    out = []
    for start in range(0, T, BT):
        rows = slice(start, min(start + BT, T))
        qc, kc, vc, gc, bc = (
            x[:, rows].transpose(1, 2) for x in (q, k, v, g, beta)
        )  # (B, HV, C, ...)
        G = gc.cumsum(-1)
        decay = (G[..., :, None] - G[..., None, :]).masked_fill(~torch.ones(G.shape[-1], G.shape[-1], dtype=torch.bool, device=G.device).tril(), float("-inf")).exp()  # fmt: skip
        kk = kc @ kc.transpose(-1, -2)
        L = (bc[..., :, None] * kk * decay).tril(-1)
        eye = torch.eye(G.shape[-1], device=G.device)
        rhs = bc[..., None] * (vc - G.exp()[..., None] * (kc @ S))
        U = torch.linalg.solve_triangular(eye + L, rhs, upper=False)
        out.append(
            D**-0.5 * (G.exp()[..., None] * (qc @ S) + ((qc @ kc.transpose(-1, -2)) * decay) @ U)
        )
        S = G[..., -1:].exp()[..., None] * S + (kc * (G[..., -1:] - G).exp()[..., None]).transpose(-1, -2) @ U  # fmt: skip
    return torch.cat(out, 2).transpose(1, 2)


def reference_tokens(q, k, v, g, beta):
    state = q.new_zeros(q.shape[0], v.shape[2], D, D)
    out = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state))
        state = state + k[:, t][..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * D**-0.5, state))
    return torch.stack(out, 1)


def reference(inputs, do, chunked=True):
    leaves = [x.detach().float().requires_grad_() for x in inputs]
    q, k = (x.repeat_interleave(2, dim=2) for x in leaves[:2])
    o = (reference_chunked if chunked else reference_tokens)(q, k, *leaves[2:])
    o.backward(do.float())
    return [o.detach()] + [leaf.grad for leaf in leaves]


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def run_cp(inputs, do, rank, W):
    """
    gated_delta_rule on this rank's slice through autograd: the stitched output and gradients
    (all-gathered along the sequence).
    """
    T = inputs[0].shape[1] // W
    shard = lambda x: x[:, rank * T : (rank + 1) * T].contiguous()
    leaves = [shard(x).detach().requires_grad_() for x in inputs]
    o = gated_delta_rule(*leaves, cp_group=distributed.cp_group)
    o.backward(shard(do))
    local = [o.detach()] + [leaf.grad for leaf in leaves]
    stitched = []
    for x in local:
        parts = [torch.empty_like(x) for _ in range(W)]
        dist.all_gather(parts, x.contiguous(), group=distributed.cp_group)
        stitched.append(torch.cat(parts, 1))
    return stitched


def check(case: Case, got: list, ref: list, rank: int) -> list[str]:
    failures = []
    for name, x, y in zip(NAMES, got, ref):
        bound = 1e-4 if name == "o" else 1e-3
        diff = calc_diff(x, y)
        err, scale = float((x.double() - y.double()).abs().max()), float(y.abs().max())
        if not (torch.isfinite(x).all() and diff < bound and err <= 2e-2 * scale):
            failures.append(f"{case.name} {name}: calc_diff {diff:.2e} max abs {err:.2e} (ref max {scale:.2e})")  # fmt: skip
    return failures


def cases_for(W: int) -> list[Case]:
    T = 2048
    return [
        Case("T64 b3 folded", 64, b=3, resets=((0, 10), (W - 1, 63))),
        Case("T2048", 2048, gate="mild"),
        Case("T4096", 4096),
        Case("T4000 ragged", 4000, gate="mild", resets=((1, 0), (W - 1, 3999))),
        Case("T8192", 8192, gate="mild"),
        Case("reset on a rank boundary", T, gate="mild", resets=((1, 0),)),
        Case(
            "document over more than two ranks",
            T,
            gate="mild",
            resets=((0, T // 2), (W - 1, T // 2)),
        ),  # fmt: skip
        Case("document inside one rank", T, gate="mild", resets=((W - 1, 100), (W - 1, 700))),
        Case(
            "resets at residues 0, 1, 63",
            T,
            gate="mild",
            resets=((0, 5 * BT), (W // 2, 7 * BT + 1), (W - 1, 9 * BT + 63)),
        ),  # fmt: skip
        Case("loss on the last rank only", T, gate="mild", last_only=True),
    ]


class Done:
    """
    A finished collective's handle.
    """

    def wait(self) -> None:
        pass


def worker(req: Request) -> None:
    group = distributed.cp_group
    rank, W = group.rank(), group.size()
    device = torch.device("cuda", torch.cuda.current_device())
    failures = []
    # the chunked reference against the token-by-token loop
    small = Case("reference check", 96, gate="mild", resets=((0, 50),))
    inputs = make_inputs(small, W, device)
    for name, x, y in zip(NAMES, reference(inputs[:5], inputs[5]), reference(inputs[:5], inputs[5], chunked=False)):  # fmt: skip
        if calc_diff(x, y) > 1e-6:  # 1 % of the tightest bound above
            failures.append(f"chunked reference {name}: {calc_diff(x, y):.2e}")
    for case in cases_for(W):
        inputs = make_inputs(case, W, device)
        got, ref = run_cp(inputs[:5], inputs[5], rank, W), reference(inputs[:5], inputs[5])
        failures += check(case, got, ref, rank)
        if case.last_only:
            # q reaches only its own step's output, so dq of the first rank is zero here
            first = [x[:, : case.T] for x in got[2:]]
            if any(float(x.abs().max()) == 0 for x in first):
                failures.append("loss on the last rank only: a first-rank gradient is zero")
    # edge-rank asymmetry, collective order, poisoned slots
    case = Case("roles", 1000, gate="mild", resets=((W - 1, 500),))
    inputs = make_inputs(case, W, device)
    log = []
    original = op.gather_maps

    def logged(local, group_name, async_op):
        gathered, work = original(local, group_name, async_op)
        if work is not None:
            work.wait()
        log.append(tuple(gathered.shape))
        backward = len(log) % 2 == 0
        readable = range(rank + 1, W) if backward else range(rank)
        for s in set(range(W)) - set(readable):
            gathered[s] = float("nan")
        return gathered, Done()

    op.gather_maps = logged
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            got = run_cp(inputs[:5], inputs[5], rank, W)
            torch.cuda.synchronize()
    finally:
        op.gather_maps = original
    failures += check(case, got, reference(inputs[:5], inputs[5]), rank)
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    count = lambda kernel: sum(kernel in n for n in names)
    merges, chains = count("StateMerge"), count("TransitionChain")
    expected_merges = (rank > 0) + (rank < W - 1)
    expected_chains = (rank < W - 1) + (rank > 0)
    if (merges, chains) != (expected_merges, expected_chains):
        failures.append(f"rank {rank} launched {merges} merges / {chains} chains, expected {expected_merges} / {expected_chains}")  # fmt: skip
    logs = [None] * W
    dist.all_gather_object(logs, log, group=group)
    if any(entry != logs[0] for entry in logs) or len(logs[0]) != 2:
        failures.append(f"collectives differ across ranks: {logs}")
    # negative controls: no forward merge, no backward merge, a shifted reset
    case = Case("negative", 1000, gate="mild")
    inputs = make_inputs(case, W, device)
    ref = reference(inputs[:5], inputs[5])
    merge = op.gdr_cp_merge
    for which in (False, True):
        op.gdr_cp_merge = lambda ag, r, c, reverse, which=which: torch.zeros(ag.shape[1], D, D, device=ag.device) if reverse == which else merge(ag, r, c, reverse)  # fmt: skip
        try:
            got = run_cp(inputs[:5], inputs[5], rank, W)
        finally:
            op.gdr_cp_merge = merge
        detected = check(case, got, ref, rank)
        if not detected:
            failures.append(
                f"dropping the {'backward' if which else 'forward'} merge was not detected"
            )
        if which and not any(" dq" in f or " dk" in f for f in detected):
            failures.append("dropping the backward merge left the gradients within bounds")
    shifted = Case("shifted reset", 1000, gate="mild", resets=((1, 0),))
    inputs = make_inputs(shifted, W, device)
    ref = reference(inputs[:5], inputs[5])
    moved = inputs[3].clone()
    moved[:, 1000], moved[:, 1001] = inputs[3][:, 1001], -128.0
    if not check(shifted, run_cp((*inputs[:3], moved, inputs[4]), inputs[5], rank, W), ref, rank):
        failures.append("a reset shifted by one token was not detected")
    if failures:
        raise AssertionError(f"rank {rank}:\n" + "\n".join(failures))


@pytest.mark.parametrize("cp_size", [2, 4])
def test_handoff_against_the_recurrence(cp_size: int) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, worker, Request())
