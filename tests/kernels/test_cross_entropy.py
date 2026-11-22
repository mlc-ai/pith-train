"""
cross_entropy_fwd_bwd and the CrossEntropy operator at the cross_entropy cases, against float64.
"""

import pytest
import torch

from pithtrain.kernels import cross_entropy_fwd_bwd
from pithtrain.operators.cross_entropy import cross_entropy
from tests.kernels.cases import params
from tests.kernels.common import (
    GRAD_SLACK,
    IGNORE,
    half_ulp_bf16,
    log_prob_inputs,
    log_prob_reference,
    record,
)

LOSS_REL = 6e-7


def exact_ce(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    total, count = 0.0, 0
    for _, logp, valid, safe in log_prob_reference(logits, target):
        nll = -logp.gather(1, safe[:, None]).squeeze(1)
        total += float(nll[valid].sum())
        count += int(valid.sum())
    return total / max(count, 1), max(count, 1)


def grad_ratio(grad: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, scale: float) -> tuple[float, float]:  # fmt: skip
    """
    Worst gradient error in units of one bf16 rounding, and the off-target gradient's scale bias.
    """
    worst, dot, norm = 0.0, 0.0, 0.0
    for rows, logp, valid, safe in log_prob_reference(logits, target):
        p = logp.exp()
        exact = p.scatter_add(1, safe[:, None], -torch.ones_like(safe[:, None], dtype=p.dtype))
        exact = exact * scale * valid[:, None]
        tol = half_ulp_bf16(exact) + GRAD_SLACK * scale * p
        err = (grad[rows].double() - exact).abs()
        zero_rows = ~valid
        if bool(zero_rows.any()):
            assert bool((grad[rows][zero_rows] == 0).all()), "ignored rows must get zero gradient"
        worst = max(worst, float((err / (tol + 1e-300)).max()))
        off_target = exact.scatter(1, safe[:, None], 0.0)
        dot += float((grad[rows].double() * off_target).sum())
        norm += float((off_target * off_target).sum())
    return worst, dot / norm - 1.0


@pytest.mark.parametrize("case", params("cross_entropy"))
def test_cross_entropy_kernel(case: dict) -> None:
    logits, target = log_prob_inputs(case)
    want, count = exact_ce(logits, target)
    grad_buffer = logits.clone()
    loss, grad = cross_entropy_fwd_bwd(grad_buffer, target, IGNORE)
    assert grad.data_ptr() == grad_buffer.data_ptr(), "row-contiguous logits are overwritten in place"  # fmt: skip
    loss_rel = abs(loss.item() - want) / abs(want)
    ratio, bias = grad_ratio(grad, logits, target, 1.0 / count)
    record(case["id"], "kernel", loss_rel=loss_rel, grad_ratio=ratio, grad_bias=bias)
    assert loss_rel <= LOSS_REL, f"loss {float(loss)} vs {want}: rel {loss_rel:.3g}"
    assert ratio <= 1.0, f"gradient error {ratio:.3g}x one bf16 rounding"


GRAD_OUTPUT = 1.0 / 3.0


@pytest.mark.parametrize("case", params("cross_entropy"))
def test_cross_entropy_op_grad_output(case: dict) -> None:
    logits, target = log_prob_inputs(case)
    want, count = exact_ce(logits, target)
    leaf = logits.clone().requires_grad_(True)
    loss = cross_entropy(leaf.clone(), target, IGNORE)
    (loss * GRAD_OUTPUT).backward()
    loss_rel = abs(loss.item() - want) / abs(want)
    assert loss_rel <= LOSS_REL, f"loss {float(loss)} vs {want}: rel {loss_rel:.3g}"
    ratio, bias = grad_ratio(leaf.grad, logits, target, GRAD_OUTPUT / count)
    record(case["id"], "op_grad_output", loss_rel=loss_rel, grad_ratio=ratio, grad_bias=bias)
    assert ratio <= 1.0, f"gradient error {ratio:.3g}x one bf16 rounding (scale bias {bias:.3g})"
