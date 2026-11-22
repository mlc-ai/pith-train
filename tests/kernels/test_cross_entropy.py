"""
The lm_head loss at every cross_entropy shape against a float64 log-softmax: cross_entropy_fwd_bwd,
the cross_entropy op with a non-power-of-two grad_output, and log_prob_fwd / log_prob_bwd.
Losses to fp32 accuracy, gradients within one bf16 rounding.
"""

import pytest
import torch

from pithtrain.kernels import cross_entropy_fwd_bwd, log_prob_bwd, log_prob_fwd
from pithtrain.operators.cross_entropy import cross_entropy
from tests.kernels.cases import params
from tests.kernels.fp8_reference import half_ulp_bf16
from tests.kernels.measure import record

DEV = "cuda"
ROWS_PER_BLOCK = 512
IGNORE = -100

LOSS_REL = 6e-7
"""
Relative error of the fp32 mean loss.
"""
GRAD_SLACK = 2.0**-16
"""
fp32 softmax error relative to the gradient's term magnitudes, on top of half a bf16 ulp.
"""
STAT_ABS = 6e-6
"""
Absolute error of the fp32 per-row logp and entropy.
"""


def make_case(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    rows, vocab = case["axes"]["rows"], case["axes"]["vocab"]
    gen = torch.Generator(device=DEV).manual_seed(rows + vocab)
    logits = torch.empty(rows, vocab, dtype=torch.bfloat16, device=DEV)
    for start in range(0, rows, 4096):  # draw in slices to bound the fp32 temporary
        chunk = logits[start : start + 4096]
        chunk.copy_(torch.randn(chunk.shape, device=DEV, generator=gen) * 2.0)
    target = torch.randint(0, vocab, (rows,), device=DEV, generator=gen)
    target[torch.rand(rows, device=DEV, generator=gen) < 0.03] = IGNORE
    return logits, target


def reference_blocks(logits: torch.Tensor, target: torch.Tensor):
    """
    Per row block: (rows, float64 log-softmax, valid-row mask, safe target).
    """
    for start in range(0, logits.shape[0], ROWS_PER_BLOCK):
        rows = slice(start, start + ROWS_PER_BLOCK)
        logp = torch.log_softmax(logits[rows].double(), dim=-1)
        valid = target[rows] != IGNORE
        yield rows, logp, valid, torch.where(valid, target[rows], 0)


def exact_ce(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    total, count = 0.0, 0
    for _, logp, valid, safe in reference_blocks(logits, target):
        nll = -logp.gather(1, safe[:, None]).squeeze(1)
        total += float(nll[valid].sum())
        count += int(valid.sum())
    return total / max(count, 1), max(count, 1)


def grad_ratio(grad: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, scale: float) -> tuple[float, float]:  # fmt: skip
    """
    Worst gradient error in units of one bf16 rounding, and the off-target gradient's scale bias.
    """
    worst, dot, norm = 0.0, 0.0, 0.0
    for rows, logp, valid, safe in reference_blocks(logits, target):
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
    logits, target = make_case(case)
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
    logits, target = make_case(case)
    want, count = exact_ce(logits, target)
    leaf = logits.clone().requires_grad_(True)
    loss = cross_entropy(leaf.clone(), target, IGNORE)
    (loss * GRAD_OUTPUT).backward()
    loss_rel = abs(loss.item() - want) / abs(want)
    assert loss_rel <= LOSS_REL, f"loss {float(loss)} vs {want}: rel {loss_rel:.3g}"
    ratio, bias = grad_ratio(leaf.grad, logits, target, GRAD_OUTPUT / count)
    record(case["id"], "op_grad_output", loss_rel=loss_rel, grad_ratio=ratio, grad_bias=bias)
    assert ratio <= 1.0, f"gradient error {ratio:.3g}x one bf16 rounding (scale bias {bias:.3g})"


@pytest.mark.parametrize("case", params("cross_entropy"))
def test_log_prob(case: dict) -> None:
    logits, target = make_case(case)
    rows = logits.shape[0]
    gen = torch.Generator(device=DEV).manual_seed(7)
    grad_logp = torch.randn(rows, device=DEV, generator=gen)
    grad_entropy = torch.randn(rows, device=DEV, generator=gen) * 0.1
    grad = logits.clone()
    logp, entropy, row_max, log_sum = log_prob_fwd(grad, target, 1.0)
    assert torch.equal(grad, logits), "log_prob_fwd must not modify the logits"
    log_prob_bwd(grad, target, row_max, log_sum, entropy, grad_logp, grad_entropy, 1.0)
    worst_stat, worst_grad = 0.0, 0.0
    for rows_, ref_logp, valid, safe in reference_blocks(logits, target):
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
