"""
Per-token log-probability and entropy for policy-gradient objectives.

Forward preserves logits; backward replaces them with gradients. Contiguous inputs need only
O(N) extra storage, avoiding the FP32 (N, V) intermediates of a PyTorch log_softmax objective.
"""

import torch
import triton
import triton.language as tl


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
    Keep m and log(d) separate to avoid cancellation at large |m|. Seed m with a logit
    so the first rescale of s is finite.
    """
    # torch.compile passes a Python float as fp64; keep the arithmetic in fp32.
    inv_temperature = tl.cast(inv_temperature, tl.float32)
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * X_stride
    y = tl.load(Y_ptr + row)

    m = tl.load(X_ptr).to(tl.float32) * inv_temperature
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
        s = (s + (m - m_new) * d) * scale + tl.sum(e * tl.where(mask, z - m_new, 0.0))
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
        grad = tl.where(offsets == y, g_logp, 0.0) - p * (g_logp + g_entropy * (logp + entropy))
        tl.store(X_ptr + offsets, (grad * inv_temperature).to(x.dtype), mask=mask)


class LogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, temperature):
        n_rows, n_cols = inp.shape

        if inp.stride(-1) != 1 or inp.stride(-2) != n_cols:
            inp = inp.contiguous()
        if target.stride(-1) != 1:
            target = target.contiguous()

        logp = torch.empty(n_rows, dtype=torch.float32, device=inp.device)
        entropy = torch.empty_like(logp)
        row_max = torch.empty_like(logp)
        log_sum = torch.empty_like(logp)
        inv_temperature = 1.0 / temperature
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

        ctx.save_for_backward(inp.detach(), target, row_max, log_sum, entropy)
        ctx.inv_temperature = inv_temperature
        return logp, entropy

    @staticmethod
    def backward(ctx, grad_logp, grad_entropy):
        inp, target, row_max, log_sum, entropy = ctx.saved_tensors
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
            inv_temperature=ctx.inv_temperature,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        # Triton writes bypass autograd; invalidate other saved references to the logits.
        torch.autograd.graph.increment_version(inp)
        return inp, None, None


def log_prob(
    inp: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused per-token log-probability and entropy of a tempered softmax, with an in-place backward.

    FP32 arithmetic; gradients use ``inp.dtype``. Backward consumes the logits: do not read them
    afterwards or save them in another autograd consumer. Use one call per logits tensor,
    including views; backward invalidates the shared version counter.

    Invalid targets (including -100) give log-probability 0 and no log-probability gradient.
    Entropy is still computed: the objective must mask both outputs to ignore a row.

    Parameters
    ----------
    inp : torch.Tensor
        Logits of shape ``(N, V)`` where N is the number of tokens and V is the vocabulary size.
        Overwritten with its gradient by the backward if row-major contiguous; another layout is
        copied first, which costs one ``(N, V)`` allocation and leaves ``inp`` intact.
    target : torch.Tensor
        Target indices of shape ``(N,)``.
    temperature : float
        Softmax temperature T; the distribution is ``softmax(inp / T)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Log-probability of each target and entropy of each row, both of shape ``(N,)`` in FP32.
    """
    return LogProb.apply(inp, target, temperature)


def reference_log_prob(
    inp: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference; materializes (N, V) intermediates in FP32, or FP64 for FP64 inputs.
    """
    dtype = torch.promote_types(inp.dtype, torch.float32)
    logp = torch.log_softmax(inp.to(dtype) / temperature, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    valid = (target >= 0) & (target < inp.size(-1))
    logp = logp.gather(-1, torch.where(valid, target, 0).unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, logp, 0.0), entropy
