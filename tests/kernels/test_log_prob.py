"""
log_prob_fwd / _bwd at the log_prob cases, against float64.
"""

import pytest
import torch

from pithtrain.kernels import log_prob_bwd, log_prob_fwd
from tests.kernels.cases import params
from tests.kernels.common import (
    DEV,
    GRAD_SLACK,
    half_ulp_bf16,
    log_prob_inputs,
    log_prob_reference,
    record,
)

STAT_ABS = 6e-6


@pytest.mark.parametrize("case", params("log_prob"))
def test_log_prob(case: dict) -> None:
    logits, target = log_prob_inputs(case)
    rows = logits.shape[0]
    gen = torch.Generator(device=DEV).manual_seed(7)
    grad_logp = torch.randn(rows, device=DEV, generator=gen)
    grad_entropy = torch.randn(rows, device=DEV, generator=gen) * 0.1
    grad = logits.clone()
    logp, entropy, row_max, log_sum = log_prob_fwd(grad, target, 1.0)
    assert torch.equal(grad, logits), "log_prob_fwd must not modify the logits"
    log_prob_bwd(grad, target, row_max, log_sum, entropy, grad_logp, grad_entropy, 1.0)
    worst_stat, worst_grad = 0.0, 0.0
    for rows_, ref_logp, valid, safe in log_prob_reference(logits, target):
        p = ref_logp.exp()
        ref_entropy = -(p * ref_logp).sum(1)
        ref_target_logp = torch.where(valid, ref_logp.gather(1, safe[:, None]).squeeze(1), 0.0)
        worst_stat = max(worst_stat, float((logp[rows_].double() - ref_target_logp).abs().max()))
        worst_stat = max(worst_stat, float((entropy[rows_].double() - ref_entropy).abs().max()))
        # a target outside [0, V) drops the row's grad_logp (log_prob_bwd's contract)
        gl = (grad_logp[rows_].double() * valid)[:, None]
        ge = grad_entropy[rows_].double()[:, None]
        exact = -p * (gl + ge * (ref_logp + ref_entropy[:, None]))
        onehot = torch.zeros_like(exact).scatter_(1, safe[:, None], 1.0) * valid[:, None]
        exact = exact + gl * onehot
        magnitude = p * (gl.abs() + ge.abs() * (ref_logp.abs() + ref_entropy[:, None])) + gl.abs() * onehot  # fmt: skip
        tol = half_ulp_bf16(exact) + GRAD_SLACK * magnitude
        err = (grad[rows_].double() - exact).abs()
        worst_grad = max(worst_grad, float((err / (tol + 1e-300)).max()))
    record(case["id"], "log_prob", stat_abs=worst_stat, grad_ratio=worst_grad)
    assert worst_stat <= STAT_ABS, f"logp / entropy abs error {worst_stat:.3g}"
    assert worst_grad <= 1.0, f"log_prob gradient error {worst_grad:.3g}x one bf16 rounding"
