"""
FlashAttention 4 as custom ops that torch.compile traces as opaque nodes: dense BSHD attention
(flash_attn_func) and packed THD attention (flash_attn_varlen_func), each with MHA, GQA or MLA
head dimensions, an optional sliding window and optional attention sinks.
"""

from typing import Optional, Tuple

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

from pithtrain.kernels import memoized_launch

__all__ = ["flash_attn_func", "flash_attn_varlen_func"]


# FA4's backward returns the sink gradient when given learnable_sink. A custom_op cannot return
# None, so a call without sinks returns an empty tensor in its place.
def grad_sink_or_empty(grad_sink: list, q: torch.Tensor) -> torch.Tensor:
    return grad_sink[0] if grad_sink else q.new_empty(0)


def empty_grad_sink_like(learnable_sink: Optional[torch.Tensor], q: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(learnable_sink) if learnable_sink is not None else q.new_empty(0)


@torch.library.custom_op("pithtrain::flash_attn4_mha_fwd", mutates_args=())
def mha_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    entry = lambda: tuple(_flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal, window_size_left=window_size_left, window_size_right=window_size_right, learnable_sink=learnable_sink, return_lse=True)[:2])  # fmt: skip
    inputs = dict(q=q, k=k, v=v, learnable_sink=learnable_sink)
    o, lse = memoized_launch("fwd", inputs, (softmax_scale, causal, window_size_left, window_size_right), entry)  # fmt: skip
    return o, lse


@mha_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]):  # fmt: skip
    (b, s, h, _), dv = q.shape, v.shape[-1]
    o = torch.empty((b, s, h, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    return o, lse


@torch.library.custom_op("pithtrain::flash_attn4_mha_bwd", mutates_args=())
def mha_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    entry = lambda: tuple(_flash_attn_bwd(q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=causal, window_size_left=window_size_left, window_size_right=window_size_right, learnable_sink=learnable_sink))  # fmt: skip
    inputs = dict(q=q, k=k, v=v, o=o, lse=lse, do=do, learnable_sink=learnable_sink)
    dq, dk, dv, *grad_sink = memoized_launch("bwd", inputs, (softmax_scale, causal, window_size_left, window_size_right), entry)  # fmt: skip
    return dq, dk, dv, grad_sink_or_empty(grad_sink, q)


@mha_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]):  # fmt: skip
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), empty_grad_sink_like(learnable_sink, q)  # fmt: skip


def mha_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:  # fmt: skip
    q, k, v, softmax_scale, causal, window_size_left, window_size_right, learnable_sink = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, learnable_sink)
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right


def mha_backward(ctx: torch.autograd.function.FunctionCtx, grad_o: torch.Tensor, grad_lse: torch.Tensor) -> Tuple:  # fmt: skip
    q, k, v, o, lse, learnable_sink = ctx.saved_tensors
    dq, dk, dv, grad_sink = mha_bwd(q, k, v, o, lse, grad_o, ctx.softmax_scale, ctx.causal, ctx.window_size_left, ctx.window_size_right, learnable_sink)  # fmt: skip
    return dq, dk, dv, None, None, None, None, grad_sink if learnable_sink is not None else None


mha_fwd.register_autograd(mha_backward, setup_context=mha_setup_context)


def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = False, window_size: Tuple[Optional[int], Optional[int]] = (None, None), learnable_sink: Optional[torch.Tensor] = None) -> torch.Tensor:  # fmt: skip
    """
    Attention of q (B, S, H, D) over k and v (B, S, H_kv, D and D_v), returning (B, S, H, D_v).
    window_size is (left, right), None for unbounded; learnable_sink holds one logit per head.
    """
    o, _ = mha_fwd(q, k, v, softmax_scale, causal, window_size[0], window_size[1], learnable_sink)
    return o


@torch.library.custom_op("pithtrain::flash_attn4_varlen_fwd", mutates_args=())
def varlen_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    o, lse, *_ = _flash_attn_fwd(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale, causal=causal, window_size_left=window_size_left, window_size_right=window_size_right, learnable_sink=learnable_sink, return_lse=True)  # fmt: skip
    return o, lse


@varlen_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]):  # fmt: skip
    (t, h, _), dv = q.shape, v.shape[-1]
    o = torch.empty((t, h, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((h, t), dtype=torch.float32, device=q.device)
    return o, lse


@torch.library.custom_op("pithtrain::flash_attn4_varlen_bwd", mutates_args=())
def varlen_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    dq, dk, dv, *grad_sink = _flash_attn_bwd(q, k, v, o, do, lse, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale, causal=causal, window_size_left=window_size_left, window_size_right=window_size_right, learnable_sink=learnable_sink)  # fmt: skip
    return dq, dk, dv, grad_sink_or_empty(grad_sink, q)


@varlen_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]):  # fmt: skip
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), empty_grad_sink_like(learnable_sink, q)  # fmt: skip


def varlen_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:  # fmt: skip
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size_left, window_size_right, learnable_sink = inputs  # fmt: skip
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, learnable_sink)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right


def varlen_backward(ctx: torch.autograd.function.FunctionCtx, grad_o: torch.Tensor, grad_lse: torch.Tensor) -> Tuple:  # fmt: skip
    q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, learnable_sink = ctx.saved_tensors
    dq, dk, dv, grad_sink = varlen_bwd(q, k, v, o, lse, grad_o, cu_seqlens_q, cu_seqlens_k, ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.softmax_scale, ctx.causal, ctx.window_size_left, ctx.window_size_right, learnable_sink)  # fmt: skip
    return dq, dk, dv, None, None, None, None, None, None, None, None, grad_sink if learnable_sink is not None else None  # fmt: skip


varlen_fwd.register_autograd(varlen_backward, setup_context=varlen_setup_context)


def flash_attn_varlen_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, softmax_scale: float, causal: bool = False, window_size: Tuple[Optional[int], Optional[int]] = (None, None), learnable_sink: Optional[torch.Tensor] = None) -> torch.Tensor:  # fmt: skip
    """
    flash_attn_func over packed documents: q (T, H, D), k (T, H_kv, D) and v (T, H_kv, D_v),
    returning (T, H, D_v); cu_seqlens (int32, [0, ..., T]) marks document boundaries and max_seqlen
    bounds the document length.
    """
    o, _ = varlen_fwd(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale, causal, window_size[0], window_size[1], learnable_sink)  # fmt: skip
    return o
