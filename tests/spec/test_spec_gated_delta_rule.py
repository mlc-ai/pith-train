"""
Gated DeltaNet at every training shape of tools/kernel_cases/kernel_cases.json (family
gated_delta_rule): Qwen3.5-35B-A3B (16 key / 32 value heads) and Qwen3.5-27B (16 / 48), dims 128,
the micro-batch folded into the heads (H = 16 mbs, HV = 32 mbs or 48 mbs, value head j reading key
head j // (HV / H)) as pithtrain/operators/gated_delta_rule.py folds it, at T_local = seq / cp.

- cp == 1: chunk_gdr_fwd_pinned / chunk_gdr_bwd_pinned at the folded geometry, T_local 1k .. 32k.
- cp > 1 (multigpu): the operator gated_delta_rule(q, k, v, g, beta, cp_group) on cp ranks, one
  process per GPU, each holding its contiguous T_local slice of a seq = cp * T_local sequence with
  the unfolded (mbs, T_local, heads, 128) inputs training passes, forward and backward through
  autograd; CP 2, 4 and 8.

Inputs are drawn as the model computes them (q, k L2-normalized, gate -exp(A_log) softplus(a +
dt_bias), beta = sigmoid), by make_inputs of tests/kernels/test_gated_delta_rule_pinned.py.

Reference: the recurrence in fp64, in its chunked form with autograd (chunked_reference of the same
file, itself checked there against the token-by-token loop), run over the whole sequence a block of
value heads at a time (the heads are independent) so that 128k-token sequences fit in memory. It is
never a pithtrain kernel.

Bounds: those of the pinned verifier (BOUNDS: calc_diff per output; MAX_ABS: largest absolute error
within 2e-2 of the reference's largest magnitude), calibrated there at T <= 8192 and checked here up
to T_local = 32k and folded heads up to 4x, and over 128k-token CP sequences. Largest errors
measured on the H200 over the 25 single-GPU cases (T_local 1k .. 32k, HV 32 .. 192): calc_diff
5.3e-6 (o), 5.3e-6 (dq), 5.7e-6 (dk), 4.6e-6 (dv), 8.6e-6 (dg), 4.9e-6 (dbeta), all within the
bounds calibrated at T <= 8192 (2e-5 .. 1e-4), and a largest absolute error of 7.4e-3 of the
reference's largest magnitude (bound 2e-2): no growth with T or with the folded head count. Over
the 70 CP cases (CP 2 / 4 / 8, sequences to 128k tokens), per rank: calc_diff 6.1e-6 (o), 7.7e-6
(dq), 8.3e-6 (dk), 5.0e-6 (dv), 1.1e-5 (dg), 5.5e-6 (dbeta), largest absolute error 8.2e-3 of the
reference's largest magnitude. Each case prints its errors on a MEASURED line (pytest -s).
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
from tests.spec.cases import spec_params
from tests.utilities import launch

NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")
TOKEN_HEADS_PER_BLOCK = 1 << 21
"""
Sequence length x value heads the fp64 reference holds at once (its autograd keeps about 2 KB per
token-head): a 131072-token sequence runs 16 value heads at a time.
"""


def pinned_verifier():
    """
    tests/kernels/test_gated_delta_rule_pinned.py, imported on first use (it probes the GPU at
    import, which a spawned worker must do only after setup_distributed picked its device).
    """
    from tests.kernels import test_gated_delta_rule_pinned

    return test_gated_delta_rule_pinned


def seed_of(case: dict) -> int:
    return int(case["id"].rsplit("/", 1)[1]) * 7919 + case["axes"]["T_local"]


def reference_by_heads(q, k, v, g, beta, do, keep=slice(None), key_grads=False):
    """
    [o, dq, dk, dv, dg, dbeta] of the fp64 chunked recurrence for B = 1 inputs (q, k with the H
    key heads), computed a block of value heads at a time; dq and dk per value head, or with
    key_grads summed over each key head's group (the operator's key-head gradients). Only tokens
    keep along the sequence are returned, in fp32.
    """
    verifier = pinned_verifier()
    S, H, HV = q.shape[1], q.shape[2], v.shape[2]
    group = HV // H
    block = max(group, min(HV, TOKEN_HEADS_PER_BLOCK // S) // group * group)
    outs = [[] for _ in NAMES]
    for start in range(0, HV, block):
        values, keys = slice(start, start + block), slice(start // group, (start + block) // group)
        ref, grads = verifier.chunked_reference(q[:, :, keys], k[:, :, keys], *(t[:, :, values] for t in (v, g, beta, do)))  # fmt: skip
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
    verifier = pinned_verifier()
    scale = float(ref.abs().max())
    err = float((got.float() - ref).abs().max())
    return verifier.calc_diff(got, ref), err / scale if scale else err, bool(torch.isfinite(got).all())  # fmt: skip


def failures_of(label: str, got: list, ref: list) -> list[str]:
    verifier = pinned_verifier()
    failures = []
    for name, x, y in zip(NAMES, got, ref):
        assert x.shape == y.shape, (label, name, x.shape, y.shape)
        diff, rel_max, finite = errors(x, y)
        print(f"MEASURED {label} {name} calc_diff={diff:.3e} max_abs/ref_max={rel_max:.3e}")
        if not (finite and diff < verifier.BOUNDS[name] and rel_max <= verifier.MAX_ABS):
            failures.append(f"{label} {name}: calc_diff {diff:.2e} (bound {verifier.BOUNDS[name]:.0e}), max abs {rel_max:.2e} of ref max (bound {verifier.MAX_ABS:.0e}), finite {finite}")  # fmt: skip
    return failures


@pytest.mark.parametrize("case", spec_params("gated_delta_rule", keep=lambda c: c["axes"]["cp"] == 1))  # fmt: skip
def test_pinned_at_training_shapes(case: dict) -> None:
    axes = case["axes"]
    S, H, HV = axes["T_local"], axes["H"], axes["HV"]
    verifier = pinned_verifier()
    q, k, v, g, beta, do = verifier.make_inputs(S, H=H, gate="model", seed=seed_of(case), hv=HV)
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
    verifier = pinned_verifier()
    # the whole sequence, identical on every rank (same seed), unfolded as training passes it
    q, k, v, g, beta, do = verifier.make_inputs(S, H=H, B=b, gate="model", seed=seed_of(case), hv=HV)  # fmt: skip
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


@pytest.mark.parametrize("case", spec_params("gated_delta_rule", keep=lambda c: c["axes"]["cp"] > 1, marks=[pytest.mark.multigpu]))  # fmt: skip
def test_context_parallel_at_training_shapes(case: dict) -> None:
    if torch.cuda.device_count() < case["axes"]["cp"]:
        pytest.skip(f"needs {case['axes']['cp']} GPUs, have {torch.cuda.device_count()}")
    cfg = DistributedCfg()
    cfg.context_parallel_size = case["axes"]["cp"]
    launch(cfg, cp_worker, case)
