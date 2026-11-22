"""
Gated DeltaNet at every training shape of the spec (family gated_delta_rule of
tests/kernels/cases.py): Qwen3.5-35B-A3B (16 key / 32 value heads) and Qwen3.5-27B (16 / 48), dims
128, the micro-batch folded into the heads (H = 16 mbs, HV = 32 mbs or 48 mbs, value head j reading key
head j // (HV / H)) as pithtrain/operators/gated_delta_rule.py folds it, at T_local = seq / cp.

- cp == 1: chunk_gdr_fwd_pinned / chunk_gdr_bwd_pinned at the folded geometry, T_local 4k and 8k.
- cp > 1 (multigpu): the operator gated_delta_rule(q, k, v, g, beta, cp_group) on cp ranks, one
  process per GPU, each holding its contiguous T_local slice of a seq = cp * T_local sequence with
  the unfolded (mbs, T_local, heads, 128) inputs training passes, forward and backward through
  autograd; CP 2, 4, 8 and 16 (two nodes).

Inputs are drawn as the model computes them (q, k L2-normalized, gate -exp(A_log) softplus(a +
dt_bias), beta = sigmoid), by make_inputs of tests/kernels/gdn_reference.py.

Reference: the recurrence in fp64, in its chunked form with autograd (chunked_reference of the same
file, itself checked there against the token-by-token loop), run over the whole sequence a block of
value heads at a time (the heads are independent) so that 64k-token sequences fit in memory. It is
never a pithtrain kernel.

Bounds: tests/kernels/gdn_reference.py's BOUNDS (calc_diff per output) and MAX_ABS (largest absolute
error over the reference's largest magnitude), about 3x the largest errors measured on the H200 over
the 24 cases (12 single-GPU, T_local 4k / 8k, HV 32 .. 192; 12 under CP 2 .. 16, every rank):
calc_diff 5.2e-6 (o), 6.8e-6 (dq), 6.9e-6 (dk), 4.6e-6 (dv), 6.4e-6 (dg), 4.7e-6 (dbeta), and
9.2e-3 of the reference's largest magnitude (o at CP 16), with no growth with T, the folded head
count or CP. Each case prints its errors on a MEASURED line (pytest -s).
"""

import pytest
import torch
import torch.distributed as dist

from pithtrain.contexts import distributed
from pithtrain.kernels import (
    chunk_gdr_bwd_pinned,
    chunk_gdr_fwd_pinned,
    gated_delta_rule_pinned_supported,
)
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.gated_delta_rule import gated_delta_rule
from tests.kernels import gdn_reference
from tests.kernels.cases import params, seed_of
from tests.utilities import launch

NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")
TOKEN_HEADS_PER_BLOCK = 1 << 21
"""
Sequence length x value heads the fp64 reference holds at once (its autograd keeps about 2 KB per
token-head): a 131072-token sequence runs 16 value heads at a time.
"""


def reference_by_heads(q, k, v, g, beta, do, keep=slice(None), key_grads=False):
    """
    [o, dq, dk, dv, dg, dbeta] of the fp64 chunked recurrence for B = 1 inputs (q, k with the H
    key heads), computed a block of value heads at a time; dq and dk per value head, or with
    key_grads summed over each key head's group (the operator's key-head gradients). Only tokens
    keep along the sequence are returned, in fp32.
    """
    S, H, HV = q.shape[1], q.shape[2], v.shape[2]
    group = HV // H
    block = max(group, min(HV, TOKEN_HEADS_PER_BLOCK // S) // group * group)
    outs = [[] for _ in NAMES]
    for start in range(0, HV, block):
        values, keys = slice(start, start + block), slice(start // group, (start + block) // group)
        ref, grads = gdn_reference.chunked_reference(q[:, :, keys], k[:, :, keys], *(t[:, :, values] for t in (v, g, beta, do)))  # fmt: skip
        dq, dk = grads["dq"], grads["dk"]
        if key_grads:
            dq, dk = (x.view(1, S, -1, group, x.shape[-1]).sum(3) for x in (dq, dk))
        for out, x in zip(outs, (ref, dq, dk, grads["dv"], grads["dg"], grads["dbeta"])):
            out.append(x[:, keep].float())
        del ref, grads, dq, dk
    return [torch.cat(parts, 2) for parts in outs]


def errors(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, bool]:
    """
    (calc_diff, largest absolute error / the reference's largest magnitude, all finite).
    """
    scale = float(ref.abs().max())
    err = float((got.float() - ref).abs().max())
    return gdn_reference.calc_diff(got, ref), err / scale if scale else err, bool(torch.isfinite(got).all())  # fmt: skip


def failures_of(label: str, got: list, ref: list) -> list[str]:
    failures = []
    for name, x, y in zip(NAMES, got, ref):
        assert x.shape == y.shape, (label, name, x.shape, y.shape)
        diff, rel_max, finite = errors(x, y)
        print(f"MEASURED {label} {name} calc_diff={diff:.3e} max_abs/ref_max={rel_max:.3e}")
        if not (finite and diff < gdn_reference.BOUNDS[name] and rel_max <= gdn_reference.MAX_ABS):
            failures.append(f"{label} {name}: calc_diff {diff:.2e} (bound {gdn_reference.BOUNDS[name]:.0e}), max abs {rel_max:.2e} of ref max (bound {gdn_reference.MAX_ABS:.0e}), finite {finite}")  # fmt: skip
    return failures


@pytest.mark.parametrize("case", params("gated_delta_rule", keep=lambda c: c["axes"]["cp"] == 1))  # fmt: skip
def test_pinned_at_training_shapes(case: dict) -> None:
    axes = case["axes"]
    S, H, HV = axes["T_local"], axes["H"], axes["HV"]
    q, k, v, g, beta, do = gdn_reference.make_inputs(S, H, HV, seed=seed_of(case))
    assert gated_delta_rule_pinned_supported(q, k, v, g, beta), case["id"]
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    dq, dk, dv, dg, dbeta = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    del g_cum, A
    ref = reference_by_heads(q, k, v, g, beta, do)
    failures = failures_of(case["id"], [o, dq, dk, dv, dg, dbeta], ref)
    assert not failures, "\n".join(failures)


def cp_worker(case: dict) -> None:
    group = distributed.cp_group
    rank, W = group.rank(), group.size()
    axes = case["axes"]
    b, T = axes["mbs"], axes["T_local"]
    H, HV, S = axes["H"] // b, axes["HV"] // b, W * T
    # the whole sequence, identical on every rank (same seed), unfolded as training passes it
    q, k, v, g, beta, do = gdn_reference.make_inputs(S, H, HV, B=b, seed=seed_of(case))  # fmt: skip
    mine = slice(rank * T, (rank + 1) * T)
    leaves = [x[:, mine].contiguous().requires_grad_() for x in (q, k, v, g, beta)]
    o = gated_delta_rule(*leaves, cp_group=group)
    o.backward(do[:, mine].contiguous())
    got = [o.detach()] + [leaf.grad for leaf in leaves]
    del o, leaves

    def fold(t: torch.Tensor) -> torch.Tensor:  # [b, S, heads, ...] -> [1, S, b * heads, ...]
        return t.transpose(0, 1).reshape(1, S, -1, *t.shape[3:]).contiguous()

    ref = reference_by_heads(*(fold(t) for t in (q, k, v, g, beta, do)), keep=mine, key_grads=True)
    # [1, T, b * heads, ...] -> [b, T, heads, ...]
    ref = [x.reshape(T, b, -1, *x.shape[3:]).transpose(0, 1) for x in ref]
    failures = failures_of(f"{case['id']} rank {rank}", got, ref)
    gathered = [None] * W
    dist.all_gather_object(gathered, failures, group=group)
    failures = [f for rank_failures in gathered for f in rank_failures]
    assert not failures, "\n".join(failures)


def run_cp(case: dict) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = case["axes"]["cp"]
    launch(cfg, cp_worker, case)


# One function per GPU count, so that -k selects what a run has: CP 2 and 4 fit in 4 GPUs, CP 8 needs
# a node of 8, CP 16 two nodes (see README.md, "Running").
@pytest.mark.parametrize("case", params("gated_delta_rule", keep=lambda c: 1 < c["axes"]["cp"] <= 4, multigpu=True))  # fmt: skip
def test_context_parallel(case: dict) -> None:
    run_cp(case)


@pytest.mark.parametrize("case", params("gated_delta_rule", keep=lambda c: c["axes"]["cp"] == 8, multigpu=True))  # fmt: skip
def test_context_parallel_cp8(case: dict) -> None:
    run_cp(case)


@pytest.mark.parametrize("case", params("gated_delta_rule", keep=lambda c: c["axes"]["cp"] == 16, multigpu=True))  # fmt: skip
def test_context_parallel_cp16(case: dict) -> None:
    run_cp(case)


@pytest.mark.smoke
def test_chunked_reference_is_the_recurrence() -> None:
    """
    The fp64 chunked reference equals the token-by-token recurrence (in fp64) to rounding, a ragged
    last chunk included.
    """
    q, k, v, g, beta, do = gdn_reference.make_inputs(150, 16, 32, seed=9)
    ref, ref_grads = gdn_reference.chunked_reference(q, k, v, g, beta, do)
    leaves = [gdn_reference.per_value_head(q, 32), gdn_reference.per_value_head(k, 32), v, g, beta]
    leaves = [x.detach().double().requires_grad_() for x in leaves]
    naive = gdn_reference.naive_gdr(*leaves)
    naive.backward(do.double())
    torch.testing.assert_close(ref, naive.detach(), rtol=1e-9, atol=1e-9)
    for name, leaf in zip(gdn_reference.GRADS, leaves):
        torch.testing.assert_close(ref_grads[name], leaf.grad, rtol=1e-8, atol=1e-8)
