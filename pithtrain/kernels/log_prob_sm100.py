"""
Per-token log-probability and entropy of a tempered softmax on SM100 (Triton): the forward reduces
each row once, the backward overwrites the row with its gradient.
"""

import torch
import triton
import triton.language as tl

__all__ = ["log_prob_bwd_sm100", "log_prob_fwd_sm100"]


@triton.jit
def log_prob_fwd(
    X_ptr,
    X_stride,
    Y_ptr,
    logp_ptr,
    entropy_ptr,
    max_ptr,
    log_sum_ptr,
    n_cols,
    inv_temperature,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online FP32 reduction per row: z = x / T, d = sum(exp(z - m)),
    s = sum(exp(z - m) * (z - m)). Then log p = (z - m) - log(d), H = log(d) - s / d.
    Keep m and log(d) separate to avoid cancellation at large |m|. Seed m with the first
    logit, clamped to be finite, so the first rescale of s is finite. -inf logits, like the
    padding lanes, add 0 to s (0 log 0 = 0).
    """
    # torch.compile passes a Python float as fp64; keep the arithmetic in fp32.
    inv_temperature = tl.cast(inv_temperature, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * X_stride
    y = tl.load(Y_ptr + row)

    m = tl.maximum(tl.load(X_ptr).to(tl.float32) * inv_temperature, -3.4028234663852886e38)
    d = 0.0
    s = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        z = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        z = z * inv_temperature
        m_new = tl.maximum(m, tl.max(z))
        scale = tl.exp(m - m_new)
        e = tl.exp(z - m_new)
        s = (s + (m - m_new) * d) * scale + tl.sum(e * tl.where(z > float("-inf"), z - m_new, 0.0))
        d = d * scale + tl.sum(e)
        m = m_new

    log_sum = tl.log(d)
    valid = (y >= 0) & (y < n_cols)
    z_y = tl.load(X_ptr + y, mask=valid, other=0.0).to(tl.float32) * inv_temperature
    tl.store(logp_ptr + row, tl.where(valid, (z_y - m) - log_sum, 0.0))
    tl.store(entropy_ptr + row, log_sum - s / d)
    tl.store(max_ptr + row, m)
    tl.store(log_sum_ptr + row, log_sum)


@triton.jit
def log_prob_bwd(
    X_ptr,
    X_stride,
    Y_ptr,
    max_ptr,
    log_sum_ptr,
    entropy_ptr,
    grad_logp_ptr,
    grad_logp_stride,
    grad_entropy_ptr,
    grad_entropy_stride,
    n_cols,
    inv_temperature,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Overwrite each row of logits with its gradient, recomputing log p = (z - m) - log(d):

        dx_i = (g_logp * [i == y] - p_i * (g_logp + g_entropy * (log p_i + H))) / T

    Fold the target term into its owning lane; a later scalar correction would race with
    other warps' stores. Invalid targets have constant log p = 0, so their g_logp is dropped.
    -inf logits have p_i = 0; their log p_i is taken as 0, so the second term is 0, not NaN.
    """
    inv_temperature = tl.cast(inv_temperature, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * X_stride
    y = tl.load(Y_ptr + row)
    m = tl.load(max_ptr + row)
    log_sum = tl.load(log_sum_ptr + row)
    entropy = tl.load(entropy_ptr + row)
    g_logp = tl.load(grad_logp_ptr + row * grad_logp_stride)
    g_logp = tl.where((y >= 0) & (y < n_cols), g_logp, 0.0)
    g_entropy = tl.load(grad_entropy_ptr + row * grad_entropy_stride)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        logp = (x.to(tl.float32) * inv_temperature - m) - log_sum
        p = tl.exp(logp)
        logp = tl.where(logp > float("-inf"), logp, 0.0)
        grad = tl.where(offsets == y, g_logp, 0.0) - p * (g_logp + g_entropy * (logp + entropy))
        tl.store(X_ptr + offsets, (grad * inv_temperature).to(x.dtype), mask=mask)


def log_prob_fwd_sm100(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.stride(-1) != 1:
        target = target.contiguous()
    n_rows, n_cols = inp.shape
    logp = torch.empty(n_rows, dtype=torch.float32, device=inp.device)
    entropy = torch.empty_like(logp)
    row_max = torch.empty_like(logp)
    log_sum = torch.empty_like(logp)
    BLOCK_SIZE = min(65536 // 2, triton.next_power_of_2(n_cols))

    log_prob_fwd[(n_rows,)](
        X_ptr=inp,
        X_stride=inp.stride(-2),
        Y_ptr=target,
        logp_ptr=logp,
        entropy_ptr=entropy,
        max_ptr=row_max,
        log_sum_ptr=log_sum,
        n_cols=n_cols,
        inv_temperature=inv_temperature,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )
    return logp, entropy, row_max, log_sum


def log_prob_bwd_sm100(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    if target.stride(-1) != 1:
        target = target.contiguous()
    n_rows, n_cols = inp.shape
    BLOCK_SIZE = min(65536 // 2, triton.next_power_of_2(n_cols))

    log_prob_bwd[(n_rows,)](
        X_ptr=inp,
        X_stride=inp.stride(-2),
        Y_ptr=target,
        max_ptr=row_max,
        log_sum_ptr=log_sum,
        entropy_ptr=entropy,
        grad_logp_ptr=grad_logp,
        grad_logp_stride=grad_logp.stride(-1),
        grad_entropy_ptr=grad_entropy,
        grad_entropy_stride=grad_entropy.stride(-1),
        n_cols=n_cols,
        inv_temperature=inv_temperature,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    # Triton writes bypass autograd; invalidate other saved references to the logits.
    torch.autograd.graph.increment_version(inp)
