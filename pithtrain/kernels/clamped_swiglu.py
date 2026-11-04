from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["BLOCK_SIZE", "clamped_swiglu_bwd", "clamped_swiglu_fwd"]


@triton.jit
def clamped_swiglu_fwd_kernel(gate_up_ptr, out_ptr, num_elements, alpha, limit, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    elem_offs = block_start + tl.arange(0, BLOCK_SIZE)
    elem_mask = elem_offs < num_elements

    gate, up = tl.load(gate_up_ptr + 2 * elem_offs, mask=elem_mask), tl.load(gate_up_ptr + 2 * elem_offs + 1, mask=elem_mask)
    gate_f32, up_f32 = gate.to(tl.float32), up.to(tl.float32)
    gate_clamped, up_clamped = tl.minimum(gate_f32, limit), tl.minimum(tl.maximum(up_f32, -limit), limit)

    gate_sigmoid = tl.sigmoid(alpha * gate_clamped)
    activated_gate = gate_clamped * gate_sigmoid
    out = (up_clamped + 1.0) * activated_gate

    tl.store(out_ptr + elem_offs, out.to(gate.dtype), mask=elem_mask)


@triton.jit
def clamped_swiglu_bwd_kernel(grad_out_ptr, gate_up_ptr, grad_gate_up_ptr, num_elements, alpha, limit, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    elem_offs = block_start + tl.arange(0, BLOCK_SIZE)
    elem_mask = elem_offs < num_elements

    grad_out, gate, up = tl.load(grad_out_ptr + elem_offs, mask=elem_mask), tl.load(gate_up_ptr + 2 * elem_offs, mask=elem_mask), tl.load(gate_up_ptr + 2 * elem_offs + 1, mask=elem_mask)
    grad_out_f32, gate_f32, up_f32 = grad_out.to(tl.float32), gate.to(tl.float32), up.to(tl.float32)
    gate_clamped, up_clamped = tl.minimum(gate_f32, limit), tl.minimum(tl.maximum(up_f32, -limit), limit)

    gate_sigmoid = tl.sigmoid(alpha * gate_clamped)
    activated_gate = gate_clamped * gate_sigmoid

    grad_up_clamped, grad_gate_clamped = grad_out_f32 * activated_gate, grad_out_f32 * (up_clamped + 1.0) * gate_sigmoid * (1.0 + alpha * gate_clamped * (1.0 - gate_sigmoid))

    is_gate_unclamped = gate_f32 <= limit
    is_up_unclamped = (up_f32 >= -limit) & (up_f32 <= limit)
    grad_gate, grad_up = tl.where(is_gate_unclamped, grad_gate_clamped, 0.0), tl.where(is_up_unclamped, grad_up_clamped, 0.0)

    tl.store(grad_gate_up_ptr + 2 * elem_offs, grad_gate.to(gate.dtype), mask=elem_mask)
    tl.store(grad_gate_up_ptr + 2 * elem_offs + 1, grad_up.to(up.dtype), mask=elem_mask)


BLOCK_SIZE = 1024


def clamped_swiglu_fwd(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Clamped SwiGLU of an interleaved gate_up: a new contiguous (M, N) tensor in gate_up's dtype,
    (up_clamped + 1) * gate_clamped * sigmoid(alpha * gate_clamped) with gate = gate_up[:, 0::2], up
    = gate_up[:, 1::2], gate_clamped = min(gate, limit), up_clamped = clamp(up, -limit, limit).

    gate_up is contiguous (M, 2N), bf16 (any float dtype). alpha and limit are Python floats
    (gpt-oss: 1.702, 7.0). Computed in fp32 and rounded once to the input dtype. Deterministic.
    """
    assert gate_up.dim() == 2, f"gate_up must be 2-D, got shape {gate_up.shape}"
    M, two_n = gate_up.shape
    assert two_n % 2 == 0, f"gate_up last dim must be even, got {two_n}"
    assert gate_up.is_contiguous(), "gate_up must be contiguous"

    N = two_n // 2
    out = torch.empty((M, N), device=gate_up.device, dtype=gate_up.dtype)

    num_elements = M * N
    if num_elements == 0:
        return out

    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    clamped_swiglu_fwd_kernel[grid](gate_up, out, num_elements, float(alpha), float(limit), BLOCK_SIZE=BLOCK_SIZE)
    return out


def clamped_swiglu_bwd(grad_output: torch.Tensor, gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Gradient of clamped_swiglu_fwd with respect to the interleaved gate_up: a new (M, 2N) tensor
    like gate_up, gate gradients at even and up gradients at odd columns.

    grad_output is contiguous (M, N) in gate_up's dtype; gate_up is the saved contiguous (M, 2N)
    forward input. The gate gradient is zero where gate > limit, the up gradient zero where |up| >
    limit (the boundary itself passes the gradient, as torch.clamp does). fp32 arithmetic, rounded
    once to the dtype. Deterministic.
    """
    assert grad_output.shape[0] == gate_up.shape[0]
    assert grad_output.shape[1] * 2 == gate_up.shape[1]
    assert grad_output.dtype == gate_up.dtype
    assert grad_output.is_contiguous(), "grad_output must be contiguous"

    grad_gate_up = torch.empty_like(gate_up)
    num_elements = grad_output.numel()
    if num_elements == 0:
        return grad_gate_up

    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    clamped_swiglu_bwd_kernel[grid](grad_output, gate_up, grad_gate_up, num_elements, float(alpha), float(limit), BLOCK_SIZE=BLOCK_SIZE)
    return grad_gate_up
