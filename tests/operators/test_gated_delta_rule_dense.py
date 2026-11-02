"""
Test the dense gated delta rule custom ops (pithtrain::gated_delta_rule_dense_fwd / _bwd) at the
Qwen3.5 geometry, H = 16 key heads over HV = 32 value heads, where they run the pinned kernels:

- under torch.compile with a symbolic sequence length, against the fp32 token-by-token recurrence,
  the key-head gradients summed over each group by the expand's backward, at aligned lengths and
  lengths that are not a multiple of 64 (the kernels' ragged variants) in alternation;
- no recompilation across sequence lengths, aligned or ragged;
- the fakes against the real ops (shapes, dtypes, strides) with torch.library.opcheck;
- batch 2 and 3 through gated_delta_rule, folded into the heads before the key-head repeat onto the
  pinned kernels (batch 1, H 16 b / HV 32 b), against the same recurrence;
- geometries the capability query rejects, also batched, taking FLA's path against the recurrence.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import gated_delta_rule_pinned_supported
from pithtrain.operators.gated_delta_rule import (
    dense_bwd,
    dense_fwd,
    dense_gated_delta_rule,
    gated_delta_rule,
)

H, HV, D = 16, 32, 128


def make_inputs(S: int, seed: int = 0, B: int = 1) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=gen)

    q = F.normalize(randn(B, S, H, D), dim=-1).to(torch.bfloat16)
    k = F.normalize(randn(B, S, H, D), dim=-1).to(torch.bfloat16)
    v = randn(B, S, HV, D).to(torch.bfloat16)
    a_log = torch.log(torch.rand(HV, device="cuda", generator=gen) * 16)
    g = -a_log.exp() * F.softplus(randn(B, S, HV) + 1.0)
    beta = randn(B, S, HV).to(torch.bfloat16).sigmoid()
    do = randn(B, S, HV, D).to(torch.bfloat16)
    return q, k, v, g, beta, do


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    q, k, v, g, beta, _ = make_inputs(64)
    return gated_delta_rule_pinned_supported(q, k, v, g, beta)


pytestmark = pytest.mark.skipif(not available(), reason="no pinned dense path on this device")


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def naive_gdr(q, k, v, g, beta):
    """
    The fp32 recurrence on key heads repeated to the value heads (q / k (B, T, H, D)), so autograd
    sums the per-value-head gradients of each group into the key heads.
    """
    q, k = (x.repeat_interleave(HV // H, dim=2) for x in (q, k))
    state = q.new_zeros(q.shape[0], HV, D, D)
    out = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state))
        state = state + k[:, t][..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * D**-0.5, state))
    return torch.stack(out, 1)


def test_compiled_against_recurrence_without_recompiles() -> None:
    torch._dynamo.reset()
    compiled = torch.compile(dense_gated_delta_rule, dynamic=True, fullgraph=True)
    for step, S in enumerate((256, 63, 64, 65, 1000, 1024, 512)):
        q, k, v, g, beta, do = make_inputs(S, seed=S)
        leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
        if step >= 1:
            torch.compiler.set_stance("fail_on_recompile")
        try:
            out = compiled(*leaves)
            out.backward(do)
        finally:
            torch.compiler.set_stance("default")
        refs = [x.detach().float().requires_grad_() for x in (q, k, v, g, beta)]
        ref = naive_gdr(*refs)
        ref.backward(do.float())
        assert out.shape == v.shape and out.dtype == torch.bfloat16
        assert calc_diff(out, ref) < 1e-4, f"S={S} out"
        for name, leaf, ref_leaf in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves, refs):
            assert leaf.grad.shape == leaf.shape and leaf.grad.dtype == leaf.dtype, name
            diff = calc_diff(leaf.grad, ref_leaf.grad)
            assert diff < 1e-3, f"S={S} {name}: calc_diff {diff:.2e}"


@pytest.mark.parametrize("S", [256, 1024])
def test_fakes_match_the_ops(S: int) -> None:
    q, k, v, g, beta, do = make_inputs(S)
    qe, ke = (x.unsqueeze(3).expand(1, S, H, HV // H, D) for x in (q, k))
    o, g_cum, A = dense_fwd(qe, ke, v, g, beta)
    checks = ("test_schema", "test_faketensor")
    torch.library.opcheck(
        torch.ops.pithtrain.gated_delta_rule_dense_fwd.default,
        (qe, ke, v, g, beta),
        test_utils=checks,
    )
    torch.library.opcheck(
        torch.ops.pithtrain.gated_delta_rule_dense_bwd.default,
        (qe, ke, v, g_cum, beta, A, do),
        test_utils=checks,
    )
    grads = dense_bwd(qe, ke, v, g_cum, beta, A, do)
    for got, like in zip(grads, (qe, ke, v, g_cum, beta)):
        assert got.shape == like.shape and got.dtype == like.dtype and got.is_contiguous()


def spy_pinned(monkeypatch: pytest.MonkeyPatch) -> list:
    """
    Record the (key heads, value heads) of every pinned forward the operator launches.
    """
    import pithtrain.operators.gated_delta_rule as op

    seen, forward = [], op.chunk_gdr_fwd_pinned

    def recording(q, k, v, g, beta):
        seen.append((q.shape[0], k.shape[2], v.shape[2]))
        return forward(q, k, v, g, beta)

    monkeypatch.setattr(op, "chunk_gdr_fwd_pinned", recording)
    return seen


@pytest.mark.parametrize("B,S", [(2, 192), (3, 64), (3, 1000)])
def test_batch_folds_onto_the_pinned_kernels(
    monkeypatch: pytest.MonkeyPatch, B: int, S: int
) -> None:
    """
    A batch folds into the heads before the key-head repeat: the pinned kernels see batch 1 at H
    16 B / HV 32 B (never q / k repeated to HV), and the output and the five gradients match the
    per-sample fp32 recurrence.
    """
    q, k, v, g, beta, do = make_inputs(S, seed=7, B=B)
    assert not gated_delta_rule_pinned_supported(q, k, v, g, beta)
    seen = spy_pinned(monkeypatch)
    leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
    out = gated_delta_rule(*leaves)
    out.backward(do)
    assert seen == [(1, H * B, HV * B)], seen
    refs = [x.detach().float().requires_grad_() for x in (q, k, v, g, beta)]
    ref = naive_gdr(*refs)
    ref.backward(do.float())
    assert out.shape == v.shape
    assert calc_diff(out, ref) < 1e-4
    for name, leaf, ref_leaf in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves, refs):
        assert leaf.grad.shape == leaf.shape, name
        diff = calc_diff(leaf.grad, ref_leaf.grad)
        assert diff < 1e-3, f"{name}: calc_diff {diff:.2e}"


def test_repeat_before_fold_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The fold-order check has power: q / k repeated to HV before the fold (the old CP order) reach
    the kernels as H == HV, which the assertion above rejects.
    """
    q, k, v, g, beta, _ = make_inputs(64, seed=7, B=3)
    seen = spy_pinned(monkeypatch)
    q, k = (x.repeat_interleave(HV // H, dim=2) for x in (q, k))
    gated_delta_rule(q, k, v, g, beta)
    assert (1, H * 3, HV * 3) not in seen, seen


def general_inputs(
    B: int, S: int, H: int, HV: int, K: int, V: int, seed: int
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=gen)

    q = F.normalize(randn(B, S, H, K), dim=-1).to(torch.bfloat16)
    k = F.normalize(randn(B, S, H, K), dim=-1).to(torch.bfloat16)
    v = randn(B, S, HV, V).to(torch.bfloat16)
    g = -F.softplus(randn(B, S, HV) + 1.0)
    beta = randn(B, S, HV).to(torch.bfloat16).sigmoid()
    return q, k, v, g, beta, randn(B, S, HV, V).to(torch.bfloat16)


def general_recurrence(q, k, v, g, beta):
    """
    The fp32 recurrence at any geometry (key heads repeated to the value heads).
    """
    groups = v.shape[2] // q.shape[2]
    q, k = (x.repeat_interleave(groups, dim=2) for x in (q, k))
    state = q.new_zeros(q.shape[0], v.shape[2], q.shape[-1], v.shape[-1])
    out = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state))
        state = state + k[:, t][..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * q.shape[-1] ** -0.5, state))
    return torch.stack(out, 1)


def to_dtypes(case: str, q, k, v, g, beta):
    if case == "fp32 beta":
        beta = beta.float()
    if case == "fp32 q / k / v":
        q, k, v = q.float(), k.float(), v.float()
    if case == "non-contiguous q / k / v":
        q, k, v = (x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v))
    return q, k, v, g, beta


# (B, S, H, HV, K, V): geometries the predicate rejects, plus a supported one given as strided views.
FALLBACK_CASES = {
    "batch 2, key dim 64": (2, 192, 16, 32, 64, 128),
    "48 value heads": (1, 192, 16, 48, 128, 128),
    "batch 3, 96 value heads over 32 key heads": (3, 64, 32, 96, 128, 128),
    "16 value heads": (1, 192, 16, 16, 128, 128),
    "key dim 64": (1, 192, 16, 32, 64, 128),
    "value dim 64": (1, 192, 16, 32, 128, 64),
    "fp32 beta": (1, 192, 16, 32, 128, 128),
    "fp32 q / k / v": (1, 192, 16, 32, 128, 128),
    "non-contiguous q / k / v": (1, 192, 16, 32, 128, 128),
}


@pytest.mark.parametrize("case", list(FALLBACK_CASES))
def test_rejected_geometries_through_the_operator(case: str) -> None:
    """
    gated_delta_rule on each geometry: off the pinned path (or on it after the operator's own
    contiguous copy, for strided views), output and gradients against the fp32 recurrence.
    """
    q, k, v, g, beta, do = general_inputs(*FALLBACK_CASES[case], seed=11)
    q, k, v, g, beta = to_dtypes(case, q, k, v, g, beta)
    if case != "non-contiguous q / k / v":
        assert not gated_delta_rule_pinned_supported(q, k, v, g, beta), case
    leaves = [x.detach().clone().requires_grad_() for x in (q, k, v, g, beta)]
    out = gated_delta_rule(*leaves)
    out.backward(do.to(out.dtype))
    refs = [x.detach().float().requires_grad_() for x in (q, k, v, g, beta)]
    ref = general_recurrence(*refs)
    ref.backward(do.float())
    assert out.shape == ref.shape
    assert calc_diff(out, ref) < 1e-4, f"{case}: out calc_diff {calc_diff(out, ref):.2e}"
    for name, leaf, ref_leaf in zip(("dq", "dk", "dv", "dg", "dbeta"), leaves, refs):
        assert leaf.grad.shape == leaf.shape and leaf.grad.dtype == leaf.dtype, (case, name)
        diff = calc_diff(leaf.grad, ref_leaf.grad)
        assert diff < 1e-3, f"{case} {name}: calc_diff {diff:.2e}"
