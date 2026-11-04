import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["cross_entropy_fwd_bwd"]


def cross_entropy_fwd_bwd(inp: torch.Tensor, target: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cross-entropy of the logits inp against target, overwriting the logits with the gradient of
    that mean. Returns (loss, grad).

    inp: (N, V) bf16, fp16 or fp32 logits with V >= 1 (SM90 raises ValueError for V == 0; SM100 does
    not check), overwritten in place when row-contiguous (otherwise a contiguous copy is overwritten
    and inp is left alone). target: integer (N,), any stride, each in [0, V) or ignore_index (any
    other value reads out of bounds); not modified.

    loss: fp32 scalar, the sum over non-ignored rows of logsumexp(x) - x[target] divided by n =
    max(non-ignored rows, 1). grad: inp itself (or the copy), (softmax(x) - onehot(target)) / n per
    non-ignored row computed in fp32 and rounded once, zero rows where target == ignore_index.
    Matches F.cross_entropy(inp.float(), target, ignore_index=ignore_index) and its gradient to
    fp32 rounding; deterministic. No (N, V) temporary is allocated for row-contiguous inp.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.cross_entropy_sm90 import cross_entropy_fwd_bwd_sm90
            return cross_entropy_fwd_bwd_sm90(inp, target, ignore_index)
        case 10:
            from pithtrain.kernels.cross_entropy_sm100 import cross_entropy_fwd_bwd_sm100
            return cross_entropy_fwd_bwd_sm100(inp, target, ignore_index)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
