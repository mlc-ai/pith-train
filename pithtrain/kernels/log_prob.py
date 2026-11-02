"""
Picker of log_prob, the fused per-token log-probability and entropy of a tempered softmax and its
in-place backward: the Triton kernels in log_prob_sm90, the TIRx kernels in log_prob_sm100 on SM100.

Both entry points are also called inside torch.compile regions. SM90 runs the Triton kernels there,
which torch.compile traces; SM100 runs the TIRx kernels through torch.library custom ops with fake
implementations, so a compiled region calls them as opaque ops (eager calls run them directly,
without the op's dispatch).
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["log_prob_bwd", "log_prob_fwd"]


@torch.library.custom_op("pithtrain::log_prob_fwd_sm100", mutates_args=())
def log_prob_fwd_op(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The SM100 log_prob_fwd as an op torch.compile does not trace into.
    """
    from pithtrain.kernels.log_prob_sm100 import log_prob_fwd_sm100
    return log_prob_fwd_sm100(inp, target, inv_temperature)


@log_prob_fwd_op.register_fake
def log_prob_fwd_fake(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_rows = inp.shape[0]
    return tuple(inp.new_empty((n_rows,), dtype=torch.float32) for _ in range(4))


@torch.library.custom_op("pithtrain::log_prob_bwd_sm100", mutates_args=("inp",))
def log_prob_bwd_op(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    """
    The SM100 log_prob_bwd as an op torch.compile does not trace into.
    """
    from pithtrain.kernels.log_prob_sm100 import log_prob_bwd_sm100
    log_prob_bwd_sm100(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)


@log_prob_bwd_op.register_fake
def log_prob_bwd_fake(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    return None


def log_prob_fwd(inp: torch.Tensor, target: torch.Tensor, inv_temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per row of softmax(inp / T), with inv_temperature = 1 / T: the log-probability of the target
    and the entropy, plus two row statistics for log_prob_bwd.

    inp: (N, V) float logits with unit inner stride (any row stride; bf16 and fp32 run natively,
    other float dtypes are read as fp32), not modified. target: integer (N,) contiguous; a target
    outside [0, V) (such as -100) gives log-probability 0. Arithmetic is FP32; -inf logits add
    nothing to the entropy (0 log 0 = 0).

    Returns fresh fp32 (N,) tensors (logp, entropy, row_max, log_sum). row_max and log_sum are
    opaque row statistics (natural-log on SM90, base 2 on SM100) that only log_prob_bwd reads.
    Callable inside torch.compile regions.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.log_prob_sm90 import log_prob_fwd_sm90
            return log_prob_fwd_sm90(inp, target, inv_temperature)
        case 10:
            # Only a torch.compile trace needs the opaque op; eager calls skip its dispatch.
            if torch.compiler.is_compiling():
                return log_prob_fwd_op(inp, target, inv_temperature)
            from pithtrain.kernels.log_prob_sm100 import log_prob_fwd_sm100
            return log_prob_fwd_sm100(inp, target, inv_temperature)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def log_prob_bwd(inp: torch.Tensor, target: torch.Tensor, row_max: torch.Tensor, log_sum: torch.Tensor, entropy: torch.Tensor, grad_logp: torch.Tensor, grad_entropy: torch.Tensor, inv_temperature: float) -> None:
    """
    In place: overwrite the logits inp with the gradient of g_logp * logp + g_entropy * entropy,

        dx_i = (g_logp * [i == y] - p_i * (g_logp + g_entropy * (log p_i + H))) / T

    computed in FP32 and rounded once to inp's dtype. inp, target, inv_temperature as given to
    log_prob_fwd, and entropy, row_max, log_sum its outputs. grad_logp, grad_entropy: fp32 (N,),
    any stride (stride 0 included). An invalid target drops its g_logp; -inf logits get the gradient
    0 from the entropy term. Deterministic. Callable inside torch.compile regions.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.log_prob_sm90 import log_prob_bwd_sm90
            log_prob_bwd_sm90(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)
        case 10:
            if torch.compiler.is_compiling():
                log_prob_bwd_op(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)
                return
            from pithtrain.kernels.log_prob_sm100 import log_prob_bwd_sm100
            log_prob_bwd_sm100(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
