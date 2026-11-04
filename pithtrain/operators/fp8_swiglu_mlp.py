"""
Dense FP8 SwiGLU MLP of DeepSeek-V2 (shared experts and dense layers) as one custom op, so it runs
inside the compiled stage 1. It quantizes x once for gate_proj and up_proj, and its backward
quantizes the SwiGLU gradients straight from dh without writing them in BF16. Every GEMM operand is
bit-identical to the unfused FP8Linear -> silu_mul -> FP8Linear path. Rows must be a multiple of
128; other shapes take the unfused path.
"""

from typing import Tuple

import torch

from pithtrain.kernels import (
    fp8_gemm_nt,
    fp8_quant_act,
    fp8_quant_grad,
    fp8_silu_mul_bwd,
)
from pithtrain.operators.silu_mul import silu_mul

__all__ = ["fp8_swiglu_mlp", "fusable"]


def fusable(x: torch.Tensor) -> bool:
    return x.numel() > 0 and x.flatten(0, -2).shape[0] % 128 == 0


@torch.library.custom_op("pithtrain::fp8_swiglu_mlp_fwd", mutates_args=())
def swiglu_mlp_fwd(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    wg_fp8: torch.Tensor,
    wg_scale: torch.Tensor,
    wg_t: torch.Tensor,
    wg_t_scale: torch.Tensor,
    wu_fp8: torch.Tensor,
    wu_scale: torch.Tensor,
    wu_t: torch.Tensor,
    wu_t_scale: torch.Tensor,
    wd_fp8: torch.Tensor,
    wd_scale: torch.Tensor,
    wd_t: torch.Tensor,
    wd_t_scale: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Returns (y, g, u, x_t, x_t_scale, h_t, h_t_scale): the output, and for the backward the gate and
    up activations and the transposed FP8 x and h. Each projection passes its BF16 weight (only so
    it gets a gradient) and FP8Linear's quantized-weight tuple (fp8, scale, t, t_scale).
    """
    M, inter, hidden = x.shape[0], wg_fp8.shape[0], wd_fp8.shape[0]
    x_fp8, x_scale, x_t, x_t_scale = fp8_quant_act(x)
    g = fp8_gemm_nt(x_fp8, x_scale, wg_fp8, wg_scale, M, inter, x.device, x.dtype)
    u = fp8_gemm_nt(x_fp8, x_scale, wu_fp8, wu_scale, M, inter, x.device, x.dtype)
    del x_fp8, x_scale
    h_fp8, h_scale, h_t, h_t_scale = fp8_quant_act(silu_mul(g, u))
    y = fp8_gemm_nt(h_fp8, h_scale, wd_fp8, wd_scale, M, hidden, x.device, x.dtype)
    return y, g, u, x_t, x_t_scale, h_t, h_t_scale


@swiglu_mlp_fwd.register_fake
def _(x, w_gate, w_up, w_down, wg_fp8, *_):
    M, K = x.shape
    inter, hidden = wg_fp8.shape[0], w_down.shape[0]
    device, fp8_dtype = x.device, torch.float8_e4m3fn
    num_blocks = lambda size: (size + 127) // 128
    return (
        x.new_empty((M, hidden)),
        x.new_empty((M, inter)),
        x.new_empty((M, inter)),
        torch.empty((K, M), dtype=fp8_dtype, device=device),
        torch.empty((num_blocks(K), num_blocks(M)), dtype=torch.float32, device=device),
        torch.empty((inter, M), dtype=fp8_dtype, device=device),
        torch.empty((num_blocks(inter), num_blocks(M)), dtype=torch.float32, device=device),
    )


@torch.library.custom_op("pithtrain::fp8_swiglu_mlp_bwd", mutates_args=())
def swiglu_mlp_bwd(
    dy: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    x_t: torch.Tensor,
    x_t_scale: torch.Tensor,
    h_t: torch.Tensor,
    h_t_scale: torch.Tensor,
    wg_t: torch.Tensor,
    wg_t_scale: torch.Tensor,
    wu_t: torch.Tensor,
    wu_t_scale: torch.Tensor,
    wd_t: torch.Tensor,
    wd_t_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (M, hidden), inter = dy.shape, g.shape[1]
    device, dtype = dy.device, dy.dtype
    dy_fp8, dy_scale, dy_t, dy_t_scale = fp8_quant_grad(dy)
    dh = fp8_gemm_nt(dy_fp8, dy_scale, wd_t, wd_t_scale, M, inter, device, dtype)
    dw_down = fp8_gemm_nt(dy_t, dy_t_scale, h_t, h_t_scale, hidden, inter, device, dtype)
    del dy_fp8, dy_scale, dy_t, dy_t_scale
    grouped_mm_offs = torch.full((1,), M, dtype=torch.int32, device=device)
    dg_quant, du_quant = fp8_silu_mul_bwd(dh, g, u, grouped_mm_offs)
    del dh
    dx = fp8_gemm_nt(dg_quant[0], dg_quant[1], wg_t, wg_t_scale, M, hidden, device, dtype)
    dx += fp8_gemm_nt(du_quant[0], du_quant[1], wu_t, wu_t_scale, M, hidden, device, dtype)
    # One group of all M rows: the K-major per-channel half is the (inter, M) transpose.
    dg_t = dg_quant[2].view(inter, M)
    du_t = du_quant[2].view(inter, M)
    dw_gate = fp8_gemm_nt(dg_t, dg_quant[3], x_t, x_t_scale, inter, hidden, device, dtype)
    dw_up = fp8_gemm_nt(du_t, du_quant[3], x_t, x_t_scale, inter, hidden, device, dtype)
    return dx, dw_gate, dw_up, dw_down


@swiglu_mlp_bwd.register_fake
def _(dy, g, u, x_t, x_t_scale, h_t, h_t_scale, *_):
    (M, hidden), inter = dy.shape, g.shape[1]
    return (
        dy.new_empty((M, hidden)),
        dy.new_empty((inter, hidden)),
        dy.new_empty((inter, hidden)),
        dy.new_empty((hidden, inter)),
    )


def swiglu_mlp_setup_context(ctx, inputs, output):
    quantized = inputs[4:]
    _, g, u, x_t, x_t_scale, h_t, h_t_scale = output
    transposed = (*quantized[2:4], *quantized[6:8], *quantized[10:12])
    ctx.save_for_backward(g, u, x_t, x_t_scale, h_t, h_t_scale, *transposed)


def swiglu_mlp_backward(ctx, dy, *_):
    dx, dw_gate, dw_up, dw_down = swiglu_mlp_bwd(dy, *ctx.saved_tensors)
    return (dx, dw_gate, dw_up, dw_down) + (None,) * 12


swiglu_mlp_fwd.register_autograd(swiglu_mlp_backward, setup_context=swiglu_mlp_setup_context)


def fp8_swiglu_mlp(gate_proj, up_proj, down_proj, x: torch.Tensor) -> torch.Tensor:
    """
    down_proj(silu(gate_proj(x)) * up_proj(x)) over bias-free FP8Linear layers; x must pass fusable.
    """
    x_2d = x.flatten(0, -2)
    weights = (gate_proj.weight, up_proj.weight, down_proj.weight)
    quantized = (
        *gate_proj.get_quantized_weight(),
        *up_proj.get_quantized_weight(),
        *down_proj.get_quantized_weight(),
    )
    y = swiglu_mlp_fwd(x_2d, *weights, *quantized)[0]
    return y.view(*x.shape[:-1], y.shape[-1])
