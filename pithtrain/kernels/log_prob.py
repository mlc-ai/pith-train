"""
Picker of log_prob, the fused per-token log-probability and entropy of a tempered softmax and its
in-place backward: CuTe-DSL kernels in log_prob_sm90, TIRx kernels in log_prob_sm100.

Both entry points are also called inside torch.compile regions. SM90 launches its kernels through
custom ops defined in log_prob_sm90; SM100 runs the TIRx kernels through the torch.library custom
ops below, with fake implementations, so a compiled region calls them as opaque ops (eager calls
run them directly, without the op's dispatch).
"""

import struct

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["log_prob_bwd", "log_prob_fwd"]


def check_inv_temperature(inv_temperature: float) -> None:
    """
    Raise ValueError unless inv_temperature rounded to fp32, as the kernels launch it, lies in
    (0, inf): a nonpositive or non-finite one makes z = inp * inv_temperature reverse order or lose
    it, which neither architecture's kernel handles (the reduction assumes the largest raw logit is
    the largest z).
    """
    try:
        launched = struct.unpack("f", struct.pack("f", inv_temperature))[0]
    except OverflowError:  # beyond fp32's range, so it rounds to inf
        launched = float("inf")
    if not 0.0 < launched < float("inf"):
        raise ValueError(f"log_prob needs an inv_temperature positive and finite in fp32, got {inv_temperature}")


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
    Per-row statistics of the tempered softmax p = softmax(z), z = inp * inv_temperature, reading
    the logits once and leaving them unchanged. Returns (logp, entropy, row_max, log_sum).

    inp: (N, V) bf16, fp16 or fp32 logits, row-contiguous (stride (V, 1)). target: (N,) integer, any
    stride (int32 or int64 natively; other integer dtypes are converted). inv_temperature: a Python
    float, cast to fp32 inside the kernel, so eager and torch.compile calls agree bit for bit; it
    must be positive and finite in fp32 (1e-50 and 1e100 are not), else ValueError before anything
    launches.

    Outputs, each (N,) fp32: logp = z[target] - row_max - log_sum, and 0 for a target outside [0, V)
    (-100, -1, V, 2^40, ...), which is never dereferenced; entropy = log_sum - sum(e (z - row_max))
    / sum(e), e = exp(z - row_max), where -inf logits add 0 (0 log 0 = 0); row_max = max(z); log_sum
    = log(sum(e)) on SM90, and on SM100 the same statistics in base 2 (max(z) log2(e) and
    log2(sum(e))): only log_prob_bwd reads them. A row with a finite logit has finite outputs however
    many -inf logits it holds; a row of only -inf logits gives NaN logp (for a valid target) and NaN
    entropy, as log_softmax does. fp32 arithmetic; deterministic; O(N) extra memory. N = 0 launches
    nothing.
    """
    check_inv_temperature(inv_temperature)
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
    Overwrite the logits in place with the gradient of grad_logp . logp + grad_entropy . entropy and
    return None. inp, target and inv_temperature as for log_prob_fwd; row_max, log_sum, entropy: the
    (N,) fp32 outputs log_prob_fwd returned for the same inputs; grad_logp, grad_entropy: (N,) fp32,
    any stride (stride 0 included).

    With log p_i = z_i - row_max - log_sum and p_i = exp(log p_i), row i of inp becomes

        (grad_logp [i == target] - p_i (grad_logp + grad_entropy (log p_i + entropy))) * inv_temperature

    computed in fp32 and rounded once to inp's dtype. A target outside [0, V) drops grad_logp. A
    -inf logit gets gradient 0, never NaN, unless it is a valid target, which keeps its grad_logp *
    inv_temperature term. Deterministic; no (N, V) temporary. The version counter of inp is bumped
    once, so autograd rejects any other saved use of the logits.
    """
    check_inv_temperature(inv_temperature)
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.log_prob_sm90 import log_prob_bwd_sm90
            return log_prob_bwd_sm90(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)
        case 10:
            # The op's mutation bumps the version counter in a trace; eager calls bump it here.
            if torch.compiler.is_compiling():
                return log_prob_bwd_op(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)
            from pithtrain.kernels.log_prob_sm100 import log_prob_bwd_sm100
            log_prob_bwd_sm100(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, inv_temperature)
            torch.autograd.graph.increment_version(inp)
            return None
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
