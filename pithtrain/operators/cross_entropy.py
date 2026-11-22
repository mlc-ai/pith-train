"""
Mean cross-entropy of the lm_head logits through the log_prob kernels.
"""

import torch

from pithtrain.kernels import log_prob_bwd, log_prob_fwd

__all__ = ["cross_entropy"]


class CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, ignore_index):
        num_cols = inp.shape[1]
        if inp.stride(-1) != 1 or inp.stride(-2) != num_cols:
            inp = inp.contiguous()
        valid = target != ignore_index
        # log_prob drops a target outside [0, V); an ignore_index inside it is mapped out.
        if 0 <= ignore_index < num_cols:
            target = torch.where(valid, target, -1)
        elif target.stride(-1) != 1:
            target = target.contiguous()
        logp, entropy, row_max, log_sum = log_prob_fwd(inp, target, 1.0)
        count = valid.sum().clamp_min(1).float()
        ctx.save_for_backward(inp.detach(), target, row_max, log_sum, entropy, count)
        return (0.0 - logp.sum()) / count  # +0 (not -0) when every row is ignored

    @staticmethod
    def backward(ctx, grad_output):
        inp, target, row_max, log_sum, entropy, count = ctx.saved_tensors
        rows = inp.shape[0]
        # fp32, so the logits gradient is rounded once
        grad_logp = (-grad_output.float() / count).expand(rows)
        grad_entropy = grad_logp.new_zeros(()).expand(rows)
        log_prob_bwd(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, 1.0)
        return inp, None, None


def cross_entropy(
    inp: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Mean cross-entropy of inp (N, V); the backward overwrites inp with its gradient.
    """
    return CrossEntropy.apply(inp, target, ignore_index)
