"""
Causal depthwise conv1d of the Qwen3.5 Gated DeltaNet on the channels-last (B, S, C) layout, forward
writing the (B, C, S) storage a sliced nn.Conv1d output has, and backward (dx and dw).

offset is the number of leading rows of x that are history (the K - 1 tokens of conv state prepended
under context parallelism), not outputs.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["causal_conv1d_bwd", "causal_conv1d_empty", "causal_conv1d_fwd"]


def causal_conv1d_empty(x: torch.Tensor, offset: int) -> torch.Tensor:
    """
    Uninitialized output of causal_conv1d_fwd for x (B, offset + S, C), on any device (meta
    included): logical (B, S, C) in x's dtype, a transposed view of a dense (B, C, S) tensor
    (strides (C * S, 1, S)), the layout of nn.Conv1d's sliced output. For fake implementations.
    """
    B, L, C = x.shape
    return x.new_empty((B, C, L - offset)).transpose(1, 2)


def causal_conv1d_fwd(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    """
    Causal depthwise conv y[b, t, c] = sum over k = 0 .. K - 1 of w[c, k] * x[b, t + offset - (K -
    1) + k, c], taps at negative positions skipped.

    x: bf16 (B, offset + S, C) contiguous, its first offset rows history. w: (C, 1, K) or (C, K)
    contiguous with K <= 8 (SM90 raises ValueError beyond; SM100 does not check), any float dtype
    (read as fp32). Neither is modified. Returns a fresh bf16 (B, S, C)
    in the layout of causal_conv1d_empty, bit-identical to ATen's CUDA conv1d(x.transpose(1, 2),
    w, padding=K - 1, groups=C)[..., offset:offset + S].transpose(1, 2).
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.causal_conv1d_sm90 import causal_conv1d_fwd_sm90
            return causal_conv1d_fwd_sm90(x, w, offset)
        case 10:
            from pithtrain.kernels.causal_conv1d_sm100 import causal_conv1d_fwd_sm100
            return causal_conv1d_fwd_sm100(x, w, offset)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")


def causal_conv1d_bwd(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients (dx, dw) of causal_conv1d_fwd for the output gradient g: bf16 (B, S, C), any strides;
    x, w, offset as in the forward. None is modified.

    dx: fresh bf16 (B, offset + S, C) contiguous, history rows included, bit-identical to ATen's
    depthwise conv backward. dw: fresh, w's shape and dtype, a fixed-order fp32 reduction of
    partial sums: deterministic, but equal to ATen's only to fp32 rounding.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.causal_conv1d_sm90 import causal_conv1d_bwd_sm90
            return causal_conv1d_bwd_sm90(g, x, w, offset)
        case 10:
            from pithtrain.kernels.causal_conv1d_sm100 import causal_conv1d_bwd_sm100
            return causal_conv1d_bwd_sm100(g, x, w, offset)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
