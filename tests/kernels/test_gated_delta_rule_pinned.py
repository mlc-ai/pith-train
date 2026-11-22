"""
The pinned Gated DeltaNet kernels at the gated_delta_rule_pinned cases (CP 1) against the fp64
chunked recurrence.
"""

import pytest
import torch

from pithtrain.kernels import (
    chunk_gdr_bwd_pinned,
    chunk_gdr_fwd_pinned,
    gated_delta_rule_pinned_supported,
)
from tests.kernels.cases import params, seed_of
from tests.kernels.common import (
    CHUNK_GDR_GRADS,
    chunk_gdr_failures,
    chunk_gdr_fwd_pinned_inputs,
    chunk_gdr_fwd_pinned_reference,
    chunk_gdr_fwd_pinned_reference_by_heads,
    per_value_head,
)


@pytest.mark.parametrize(
    "case", params("gated_delta_rule_pinned", keep=lambda c: c["axes"]["cp"] == 1)
)
def test_pinned_at_training_shapes(case: dict) -> None:
    axes = case["axes"]
    S, H, HV = axes["T_local"], axes["H"], axes["HV"]
    q, k, v, g, beta, do = chunk_gdr_fwd_pinned_inputs(S, H, HV, seed=seed_of(case))
    assert gated_delta_rule_pinned_supported(q, k, v, g, beta), case["id"]
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    dq, dk, dv, dg, dbeta = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    del g_cum, A
    ref = chunk_gdr_fwd_pinned_reference_by_heads(q, k, v, g, beta, do)
    failures = chunk_gdr_failures(case["id"], [o, dq, dk, dv, dg, dbeta], ref)
    assert not failures, "\n".join(failures)


def test_chunked_reference_is_the_recurrence() -> None:
    """
    The fp64 chunked reference equals the token-by-token recurrence, a ragged last chunk included.
    """
    q, k, v, g, beta, do = chunk_gdr_fwd_pinned_inputs(150, 16, 32, seed=9)
    ref, ref_grads = chunk_gdr_fwd_pinned_reference(q, k, v, g, beta, do)
    leaves = [per_value_head(q, 32), per_value_head(k, 32), v, g, beta]
    leaves = [x.detach().double().requires_grad_() for x in leaves]
    naive = chunk_gdr_fwd_pinned_recurrence(*leaves)
    naive.backward(do.double())
    torch.testing.assert_close(ref, naive.detach(), rtol=1e-9, atol=1e-9)
    for name, leaf in zip(CHUNK_GDR_GRADS, leaves):
        torch.testing.assert_close(ref_grads[name], leaf.grad, rtol=1e-8, atol=1e-8)


def chunk_gdr_fwd_pinned_recurrence(q, k, v, g, beta):
    """
    The token-by-token recurrence, q / k per value head.
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
