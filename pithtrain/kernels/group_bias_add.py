import torch
import triton
import triton.language as tl

__all__ = ["group_bias_add"]


@triton.jit
def group_bias_add_kernel(y_ptr, bias_ptr, group_idx_ptr, H, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    row_offs, h_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M), tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs[None, :] < H
    group_idx = tl.load(group_idx_ptr + row_offs).to(tl.int64)
    y_offs = row_offs.to(tl.int64)[:, None] * H + h_offs[None, :]
    y, bias = tl.load(y_ptr + y_offs, mask=h_mask).to(tl.float32), tl.load(bias_ptr + group_idx[:, None] * H + h_offs[None, :], mask=h_mask).to(tl.float32)
    tl.store(y_ptr + y_offs, (y + bias).to(y_ptr.dtype.element_ty), mask=h_mask)


def group_bias_add(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> None:
    """
    In place y[r] += bias[group_indices[r]] per row, without materializing bias[group_indices].

    y is (M, H) of any float dtype, bias (G, H), group_indices integer (M,) with values in [0, G).
    Each sum is computed in fp32 and rounded once to y's dtype, bit-identical to the PyTorch
    fallback y.copy_(y + bias[group_indices]). Deterministic.
    """
    M, H = y.shape
    if y.dtype != bias.dtype or M % 128 != 0 or not y.is_contiguous() or not bias.is_contiguous():
        y.copy_(y + bias[group_indices])
        return
    group_bias_add_kernel[(M // 32, triton.cdiv(H, 256))](y, bias, group_indices, H, BLOCK_M=32, BLOCK_H=256)
