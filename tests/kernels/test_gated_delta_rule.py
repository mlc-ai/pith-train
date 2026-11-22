"""
The pinned Gated DeltaNet kernels at every CP 1 shape of the spec, forward and backward, against the
fp64 chunked recurrence (gdn_reference's BOUNDS and MAX_ABS). The CP cases are in
tests/operators/test_gated_delta_rule_cp.py.
"""

import pytest
import torch

from pithtrain.kernels import (
    chunk_gdr_bwd_pinned,
    chunk_gdr_fwd_pinned,
    gated_delta_rule_pinned_supported,
)
from tests.kernels import gdn_reference
from tests.kernels.cases import params, seed_of


@pytest.mark.parametrize("case", params("gated_delta_rule", keep=lambda c: c["axes"]["cp"] == 1))
def test_pinned_at_training_shapes(case: dict) -> None:
    axes = case["axes"]
    S, H, HV = axes["T_local"], axes["H"], axes["HV"]
    q, k, v, g, beta, do = gdn_reference.make_inputs(S, H, HV, seed=seed_of(case))
    assert gated_delta_rule_pinned_supported(q, k, v, g, beta), case["id"]
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    dq, dk, dv, dg, dbeta = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    del g_cum, A
    ref = gdn_reference.reference_by_heads(q, k, v, g, beta, do)
    failures = gdn_reference.failures_of(case["id"], [o, dq, dk, dv, dg, dbeta], ref)
    assert not failures, "\n".join(failures)


def test_chunked_reference_is_the_recurrence() -> None:
    """
    The fp64 chunked reference equals the token-by-token recurrence, a ragged last chunk included.
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
