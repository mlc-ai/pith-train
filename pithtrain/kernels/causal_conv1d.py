"""
Causal depthwise conv1d of the Qwen3.5 Gated DeltaNet on the channels-last (B, S, C) layout: one
Triton pass forward, writing the (B, C, S) storage a sliced nn.Conv1d output has, and one pass
backward (dx and per-block partials of dw).

offset is the number of leading rows of x that are history (the K - 1 tokens of conv state prepended
under context parallelism), not outputs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["BLOCK_C", "BLOCK_T", "BWD", "causal_conv1d_bwd", "causal_conv1d_fwd", "causal_conv1d_empty"]


@triton.jit
def conv_fwd_kernel(x_ptr, w_ptr, y_ptr, L, S, C, offset, y_stride_b, y_stride_t, y_stride_c, K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_t, pid_c, batch_idx = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    out_offs, chan_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T), pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    chan_mask = chan_offs < C
    x_base = x_ptr + batch_idx.to(tl.int64) * L * C
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(K):
        w_tap = tl.load(w_ptr + chan_offs * K + k, mask=chan_mask, other=0.0).to(tl.float32)
        src_offs = out_offs + offset - (K - 1) + k
        src_mask = (src_offs >= 0) & (out_offs < S)
        x_tap = tl.load(x_base + src_offs[:, None] * C + chan_offs[None, :], mask=src_mask[:, None] & chan_mask[None, :], other=0.0)
        # Taps in order as fp32 fma, as ATen's depthwise kernel (nvcc contracts its acc += w * x;
        # a skipped padding tap equals fma(w, 0, acc)): bit-identical to conv1d.
        acc = tl.fma(w_tap[None, :], x_tap.to(tl.float32), acc)
    y_ptrs = y_ptr + batch_idx.to(tl.int64) * y_stride_b + out_offs[:, None] * y_stride_t + chan_offs[None, :] * y_stride_c
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=(out_offs < S)[:, None] & chan_mask[None, :])


@triton.jit
def store_dx_block(g_base, w_ptr, dx_base, row_offs, chan_offs, chan_mask, L, S, C, offset, g_stride_t, g_stride_c, K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    """
    dx of the input rows row_offs: dx[s] = sum_k w[k] * g[s - offset + (K - 1) - k], over the
    outputs that exist.
    """
    # Taps k = 0 .. K - 1 in the order of ATen's backward kernel.
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(K):
        w_tap = tl.load(w_ptr + chan_offs * K + k, mask=chan_mask, other=0.0).to(tl.float32)
        out_offs = row_offs - offset + (K - 1) - k
        out_mask = (out_offs >= 0) & (out_offs < S)
        g_tap = tl.load(g_base + out_offs[:, None] * g_stride_t + chan_offs[None, :] * g_stride_c, mask=out_mask[:, None] & chan_mask[None, :], other=0.0)
        acc = tl.fma(w_tap[None, :], g_tap.to(tl.float32), acc)
    tl.store(dx_base + row_offs[:, None] * C + chan_offs[None, :], acc.to(tl.bfloat16), mask=(row_offs < L)[:, None] & chan_mask[None, :])


@triton.jit
def store_dw_partial(g_base, x_base, dw_part_base, row_offs, chan_offs, chan_mask, S, C, offset, g_stride_t, g_stride_c, K: tl.constexpr):
    """
    Partial dw[c, k] = sum over the outputs t in row_offs of g[t] * x[t + offset - (K - 1) + k].
    """
    out_mask = row_offs < S
    g_rows = tl.load(g_base + row_offs[:, None] * g_stride_t + chan_offs[None, :] * g_stride_c, mask=out_mask[:, None] & chan_mask[None, :], other=0.0)
    g_rows = g_rows.to(tl.float32)
    for k in tl.static_range(K):
        src_offs = row_offs + offset - (K - 1) + k
        src_mask = out_mask & (src_offs >= 0)
        x_tap = tl.load(x_base + src_offs[:, None] * C + chan_offs[None, :], mask=src_mask[:, None] & chan_mask[None, :], other=0.0)
        tl.store(dw_part_base + chan_offs * K + k, tl.sum(g_rows * x_tap.to(tl.float32), axis=0), mask=chan_mask)


@triton.jit
def conv_bwd_kernel(g_ptr, x_ptr, w_ptr, dx_ptr, dw_part_ptr, L, S, C, offset, num_row_blocks, g_stride_b, g_stride_t, g_stride_c, K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    """
    dx of input rows [pid_t * BLOCK_T, +BLOCK_T), and the fp32 partial dw of the outputs in the same
    row range (one partial per (batch, row block)).
    """
    pid_t, pid_c, batch_idx = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # input row (dx) / output row (dw)
    chan_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    chan_mask = chan_offs < C
    g_base, x_base = g_ptr + batch_idx.to(tl.int64) * g_stride_b, x_ptr + batch_idx.to(tl.int64) * L * C
    dx_base = dx_ptr + batch_idx.to(tl.int64) * L * C
    store_dx_block(g_base, w_ptr, dx_base, row_offs, chan_offs, chan_mask, L, S, C, offset, g_stride_t, g_stride_c, K, BLOCK_T, BLOCK_C)
    dw_part_base = dw_part_ptr + (batch_idx * num_row_blocks + pid_t).to(tl.int64) * C * K
    store_dw_partial(g_base, x_base, dw_part_base, row_offs, chan_offs, chan_mask, S, C, offset, g_stride_t, g_stride_c, K)


BLOCK_T, BLOCK_C = 64, 128
BWD = dict(BLOCK_T=128, BLOCK_C=64, num_warps=4)  # tuned on H200, (1, 4096, 8192)


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
    contiguous, any float dtype (read as fp32). Neither is modified. Returns a fresh bf16 (B, S, C)
    in the layout of causal_conv1d_empty, bit-identical to ATen's CUDA conv1d(x.transpose(1, 2),
    w, padding=K - 1, groups=C)[..., offset:offset + S].transpose(1, 2).
    """
    B, L, C = x.shape
    K = w.shape[-1]
    S = L - offset
    y = causal_conv1d_empty(x, offset)
    grid = (triton.cdiv(S, BLOCK_T), triton.cdiv(C, BLOCK_C), B)
    conv_fwd_kernel[grid](x, w, y, L, S, C, offset, *y.stride(), K=K, BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C)
    return y


def causal_conv1d_bwd(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients (dx, dw) of causal_conv1d_fwd for the output gradient g: bf16 (B, S, C), any strides;
    x, w, offset as in the forward. None is modified.

    dx: fresh bf16 (B, offset + S, C) contiguous, history rows included, bit-identical to ATen's
    depthwise conv backward. dw: fresh, w's shape and dtype, a fixed-order fp32 reduction of
    per-block partials: deterministic, but equal to ATen's only to fp32 rounding.
    """
    B, L, C = x.shape
    K = w.shape[-1]
    S = L - offset
    num_row_blocks = triton.cdiv(L, BWD["BLOCK_T"])
    dx = torch.empty_like(x)
    dw_part = torch.empty((B * num_row_blocks, C, K), dtype=torch.float32, device=x.device)
    grid = (num_row_blocks, triton.cdiv(C, BWD["BLOCK_C"]), B)
    conv_bwd_kernel[grid](g, x, w, dx, dw_part, L, S, C, offset, num_row_blocks, *g.stride(), K=K, **BWD)
    dw = dw_part.sum(0).to(w.dtype).view(w.shape)
    return dx, dw
