"""
Per-token log-probability and entropy for policy-gradient objectives.

Forward preserves logits; backward replaces them with gradients. Contiguous inputs need only
O(N) extra storage, avoiding the FP32 (N, V) intermediates of a PyTorch log_softmax objective.
"""

import torch

from pithtrain.kernels import log_prob_bwd, log_prob_fwd


class LogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, temperature):
        n_rows, n_cols = inp.shape

        if inp.stride(-1) != 1 or inp.stride(-2) != n_cols:
            inp = inp.contiguous()
        if target.stride(-1) != 1:
            target = target.contiguous()

        inv_temperature = 1.0 / temperature
        logp, entropy, row_max, log_sum = log_prob_fwd(inp, target, inv_temperature)

        ctx.save_for_backward(inp.detach(), target, row_max, log_sum, entropy)
        ctx.inv_temperature = inv_temperature
        return logp, entropy

    @staticmethod
    def backward(ctx, grad_logp, grad_entropy):
        inp, target, row_max, log_sum, entropy = ctx.saved_tensors
        # Overwrites the logits and bumps their version counter, invalidating other saved uses.
        log_prob_bwd(
            inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, ctx.inv_temperature
        )
        return inp, None, None


def log_prob_inplace(
    inp: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused per-token log-probability and entropy of a tempered softmax.

    Forward preserves the logits. Backward overwrites them with their gradients.
    FP32 arithmetic; gradients use ``inp.dtype``. Do not read the logits after backward
    or save them in another autograd consumer. Use one call per logits tensor,
    including views; backward invalidates the shared version counter.

    Invalid targets (including -100) give log-probability 0 and no log-probability gradient.
    Entropy is still computed: the objective must mask both outputs to ignore a row.
    -inf logits, such as masked tokens, add nothing to the entropy (0 log 0 = 0).

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
    entropy = -(logp.exp() * torch.where(logp > float("-inf"), logp, 0.0)).sum(dim=-1)
    valid = (target >= 0) & (target < inp.size(-1))
    logp = logp.gather(-1, torch.where(valid, target, 0).unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, logp, 0.0), entropy
