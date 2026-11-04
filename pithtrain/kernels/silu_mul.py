from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["BLOCK_SIZE", "silu_mul_bwd", "silu_mul_fwd"]


@triton.jit
def silu_mul_fwd_kernel(gate_ptr, up_ptr, out_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    elem_offs = block_start + tl.arange(0, BLOCK_SIZE)
    elem_mask = elem_offs < num_elements

    gate, up = tl.load(gate_ptr + elem_offs, mask=elem_mask), tl.load(up_ptr + elem_offs, mask=elem_mask)
    gate_f32, up_f32 = gate.to(tl.float32), up.to(tl.float32)
    gate_silu = gate_f32 * tl.sigmoid(gate_f32)
    out = gate_silu * up_f32

    tl.store(out_ptr + elem_offs, out.to(up.dtype), mask=elem_mask)


@triton.jit
def silu_mul_bwd_kernel(grad_out_ptr, gate_ptr, up_ptr, grad_gate_ptr, grad_up_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    elem_offs = block_start + tl.arange(0, BLOCK_SIZE)
    elem_mask = elem_offs < num_elements

    grad_out, gate, up = tl.load(grad_out_ptr + elem_offs, mask=elem_mask), tl.load(gate_ptr + elem_offs, mask=elem_mask), tl.load(up_ptr + elem_offs, mask=elem_mask)
    grad_out_f32, gate_f32, up_f32 = grad_out.to(tl.float32), gate.to(tl.float32), up.to(tl.float32)

    gate_sigmoid = tl.sigmoid(gate_f32)
    gate_silu = gate_f32 * gate_sigmoid

    grad_up, grad_gate = grad_out_f32 * gate_silu, grad_out_f32 * up_f32 * gate_sigmoid * (1.0 + gate_f32 - gate_silu)

    tl.store(grad_gate_ptr + elem_offs, grad_gate.to(gate.dtype), mask=elem_mask)
    tl.store(grad_up_ptr + elem_offs, grad_up.to(up.dtype), mask=elem_mask)


BLOCK_SIZE = 1024


def silu_mul_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    silu(gate) * up as a new contiguous tensor like gate. gate, up: contiguous, same shape and float
    dtype (bf16 in practice); not modified. Computed in fp32 and rounded once to the dtype; close
    to, not bit-identical with, the PyTorch fp32 formula (Triton's sigmoid). Deterministic;
    fp8_silu_mul_fwd reproduces it bit for bit.
    """
    assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.dtype == up.dtype, f"dtype mismatch: {gate.dtype} vs {up.dtype}"
    assert gate.is_contiguous(), "gate must be contiguous"
    assert up.is_contiguous(), "up must be contiguous"
    out = torch.empty_like(gate)
    num_elements = gate.numel()
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    silu_mul_fwd_kernel[grid](gate, up, out, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def silu_mul_bwd(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients of silu_mul_fwd as new (grad_gate, grad_up). grad_output, gate, up: contiguous, same
    shape and dtype; not modified. Computed in fp32, each rounded once to the dtype. Deterministic;
    fp8_silu_mul_bwd reproduces both bit for bit.
    """
    assert grad_output.shape == gate.shape, f"grad_output shape {grad_output.shape} != gate shape {gate.shape}"
    assert grad_output.dtype == gate.dtype, f"grad_output dtype {grad_output.dtype} != gate dtype {gate.dtype}"
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    grad_gate, grad_up = torch.empty_like(gate), torch.empty_like(up)
    num_elements = gate.numel()
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    silu_mul_bwd_kernel[grid](grad_output, gate, up, grad_gate, grad_up, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    return grad_gate, grad_up
