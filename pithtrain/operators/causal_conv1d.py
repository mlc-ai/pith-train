"""
Causal depthwise conv1d of the Qwen3.5 Gated DeltaNet on the channels-last (B, S, C) input, as a
custom op with its backward. The forward is bit-identical to nn.functional.conv1d of
x.transpose(1, 2) with padding K - 1, sliced to offset:offset + S along the sequence;
causal_conv1d_fwd states the offset convention.
"""

from __future__ import annotations

import torch

from pithtrain.kernels import causal_conv1d_bwd, causal_conv1d_empty, causal_conv1d_fwd

__all__ = ["causal_conv1d"]


@torch.library.custom_op("pithtrain::qwen35_causal_conv1d", mutates_args=())
def conv_fwd(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    return causal_conv1d_fwd(x, w, offset)


@conv_fwd.register_fake
def _(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    return causal_conv1d_empty(x, offset)


@torch.library.custom_op("pithtrain::qwen35_causal_conv1d_bwd", mutates_args=())
def conv_bwd(
    g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return causal_conv1d_bwd(g, x, w, offset)


@conv_bwd.register_fake
def _(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int):
    return torch.empty_like(x), torch.empty_like(w)


def conv_setup_context(ctx, inputs, output) -> None:
    x, w, offset = inputs
    ctx.save_for_backward(x, w)
    ctx.offset = offset


def conv_backward(ctx, grad_out: torch.Tensor):
    x, w = ctx.saved_tensors
    grad_x, grad_w = conv_bwd(grad_out, x, w, ctx.offset)
    return grad_x, grad_w, None


conv_fwd.register_autograd(conv_backward, setup_context=conv_setup_context)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    Depthwise causal conv of bf16 x (B, offset + S, C) with weight (C, 1, K) or (C, K), no bias.
    Returns (B, S, C) in bf16, stored channels-first (as causal_conv1d_empty allocates it).
    """
    assert x.dtype == torch.bfloat16 and x.dim() == 3
    return conv_fwd(x.contiguous(), weight, offset)
