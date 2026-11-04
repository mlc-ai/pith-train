# Adapted from NVIDIA TransformerEngine (Apache 2.0)
# https://github.com/NVIDIA/TransformerEngine
#   transformer_engine/common/triton/cross_entropy.py  (Triton kernels)
#   transformer_engine/pytorch/triton/cross_entropy.py  (autograd wrapper)
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modifications: fused online-softmax + cross-entropy into a single kernel,
# removed tensor-parallelism / label-smoothing / distributed support.

import torch
import triton
import triton.language as tl

__all__ = ["cross_entropy_fwd_bwd"]


@triton.jit
def cross_entropy_fwd(logits_ptr, logits_row_stride, target_ptr, target_stride, loss_ptr, loss_stride, num_cols, num_non_ignored_ptr, ignore_index, BLOCK_SIZE: tl.constexpr):
    """
    One program per row: an online softmax over the vocabulary, then the row is overwritten with
    (softmax(x) - onehot(target)) / num_non_ignored and its loss stored. An ignored row is zeroed
    with loss 0.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    logits_ptr += row_idx * logits_row_stride
    target_idx = tl.load(target_ptr + row_idx * target_stride)

    if target_idx == ignore_index:
        for col_start in range(0, num_cols, BLOCK_SIZE):
            col_offs = col_start + tl.arange(0, BLOCK_SIZE)
            tl.store(logits_ptr + col_offs, 0.0, mask=col_offs < num_cols)
        tl.store(loss_ptr + row_idx * loss_stride, 0.0)
        return

    num_non_ignored: tl.float32 = tl.load(num_non_ignored_ptr).to(tl.float32)

    row_max: tl.float32 = float("-inf")
    exp_sum: tl.float32 = 0.0
    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_offs = col_start + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + col_offs, mask=col_offs < num_cols, other=float("-inf")).to(tl.float32)
        block_max = tl.max(logits)
        new_row_max = tl.maximum(row_max, block_max)
        exp_sum = exp_sum * tl.exp(row_max - new_row_max) + tl.sum(tl.exp(logits - new_row_max))
        row_max = new_row_max

    target_logit = tl.load(logits_ptr + target_idx).to(tl.float32)

    for col_start in range(0, num_cols, BLOCK_SIZE):
        col_offs = col_start + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + col_offs, mask=col_offs < num_cols, other=float("-inf"))
        grad_dtype = logits.dtype
        grad_logits = (tl.exp(logits.to(tl.float32) - row_max) / exp_sum) / num_non_ignored
        grad_logits = tl.where(col_offs == target_idx, grad_logits - 1.0 / num_non_ignored, grad_logits)
        tl.store(logits_ptr + col_offs, grad_logits.to(grad_dtype), mask=col_offs < num_cols)

    loss = -(target_logit - row_max - tl.log(exp_sum))
    tl.store(loss_ptr + row_idx * loss_stride, loss)


def cross_entropy_fwd_bwd(inp: torch.Tensor, target: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cross-entropy of the logits inp against target, overwriting the logits with the gradient of
    that mean. Returns (loss, grad).

    inp: (N, V) bf16 or fp32 logits, overwritten in place when row-contiguous (otherwise a
    contiguous copy is overwritten and inp is left alone). target: integer (N,), each in [0, V) or
    ignore_index (any other value reads out of bounds); not modified.

    loss: fp32 scalar, the sum over non-ignored rows of logsumexp(x) - x[target] divided by n =
    max(non-ignored rows, 1). grad: inp itself (or the copy), (softmax(x) - onehot(target)) / n per
    non-ignored row computed in fp32 and rounded once, zero rows where target == ignore_index.
    Matches F.cross_entropy(inp.float(), target, ignore_index=ignore_index) and its gradient to
    fp32 rounding; deterministic.
    """
    num_rows, num_cols = inp.shape

    if inp.stride(-1) != 1 or inp.stride(-2) != num_cols:
        inp = inp.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    num_non_ignored = (target != ignore_index).sum(dtype=torch.int64).view(1)
    num_non_ignored.clamp_(min=1)

    row_loss = torch.zeros(num_rows, dtype=torch.float32, device=inp.device)
    BLOCK_SIZE = min(65536 // 2, triton.next_power_of_2(num_cols))

    cross_entropy_fwd[(num_rows,)](logits_ptr=inp, logits_row_stride=inp.stride(-2), target_ptr=target, target_stride=target.stride(-1), loss_ptr=row_loss, loss_stride=row_loss.stride(-1), num_cols=num_cols, num_non_ignored_ptr=num_non_ignored, ignore_index=ignore_index, BLOCK_SIZE=BLOCK_SIZE, num_warps=32)

    loss = row_loss.sum() / num_non_ignored.float().squeeze()
    return loss, inp
