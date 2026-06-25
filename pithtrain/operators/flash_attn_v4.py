"""
Flash Attention 4 (CuTeDSL).

Wraps FA4's internal _flash_attn_fwd/_flash_attn_bwd with torch.library.custom_op
so that torch.compile can trace through them. Supports both symmetric (GQA/MHA)
and asymmetric (MLA) head dimensions under BSHD layout.
"""

from typing import Tuple

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

# fmt: off
# mypy: ignore-errors

# ---------------------------------------------------------------------------
# MHA / GQA
# ---------------------------------------------------------------------------

@torch.library.custom_op("pithtrain::flash_attn4_mha_fwd", mutates_args=())
def _mha_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    o, lse, *_ = _flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal, return_lse=True)
    return o, lse

@_mha_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool):
    (b, s, h, _), dv = q.shape, v.shape[-1]
    o = torch.empty((b, s, h, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    return o, lse

@torch.library.custom_op("pithtrain::flash_attn4_mha_bwd", mutates_args=())
def _mha_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _flash_attn_bwd(q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=causal)
    return dq, dk, dv

@_mha_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

def _mha_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:
    q, k, v, softmax_scale, causal = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal

def _mha_backward(ctx: torch.autograd.function.FunctionCtx, grad_o: torch.Tensor, grad_lse: torch.Tensor) -> Tuple:
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = _mha_bwd(q, k, v, o, lse, grad_o, ctx.softmax_scale, ctx.causal)
    return dq, dk, dv, None, None

_mha_fwd.register_autograd(_mha_backward, setup_context=_mha_setup_context)

def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = False) -> torch.Tensor:
    o, _ = _mha_fwd(q, k, v, softmax_scale, causal)
    return o
