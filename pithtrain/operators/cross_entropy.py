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
from triton.language.extra.cuda import libdevice


@triton.jit
def cross_entropy_fwd(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore_ptr,
    ignore_idx,
    BLOCK_SIZE: tl.constexpr,
    REDUCTION: tl.constexpr,
):
    """
    Fused cross-entropy forward: computes per-row loss and overwrites
    the logit tensor in-place with gradients for the requested reduction.

    Each program processes one row (token position). The kernel performs
    two passes over the vocabulary dimension:
      1. Online softmax: numerically stable max (m) and sum-of-exp (d).
      2. Gradient write: softmax(x_i) minus the target indicator, divided
         by the valid-token count only for mean reduction.

    The per-row loss is stored separately for later summation.
    """
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * X_stride
    y = tl.load(Y_ptr + row * Y_stride)

    if y == ignore_idx:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        tl.store(loss_ptr + row * loss_stride, 0.0)
        return

    n_non_ignore: tl.float32 = tl.load(n_non_ignore_ptr).to(tl.float32)

    m: tl.float32 = float("-inf")
    d: tl.float32 = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).to(
            tl.float32
        )
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    ori_X_y = tl.load(X_ptr + y).to(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"))
        grad_dtype = X_block.dtype
        probability = tl.exp(X_block.to(tl.float32) - m) / d
        # Correct the target in its owning lane before the only store. A
        # separate scalar read/modify/write races with the vector stores from
        # other warps, which can lose the correction or read the old logit.
        if REDUCTION == "mean":
            # Preserve the default mean API's existing BF16 rounding.
            rounded = (probability / n_non_ignore).to(grad_dtype)
            corrected = rounded.to(tl.float32) - 1.0 / n_non_ignore
            gradient = tl.where(X_offsets == y, corrected, rounded.to(tl.float32))
        else:
            # Sum in FP32 before the single cast to the activation dtype.
            # A BF16 mean followed by multiplication by the token count cannot
            # recover the summed gradient that was rounded away.
            gradient = probability - tl.where(X_offsets == y, 1.0, 0.0)
        tl.store(X_ptr + X_offsets, gradient.to(grad_dtype), mask=X_offsets < n_cols)

    loss = -(ori_X_y - m - tl.log(d))
    tl.store(loss_ptr + row * loss_stride, loss)


# Match PyTorch SoftMax.cu register accumulation for summed small-codebook CE.
# One fused kernel still stores gradients in the original activation dtype;
# no temporary vocabulary tensor or deterministic-policy switch is introduced.
@triton.jit
def cross_entropy_register_sum(
    X, STRIDE, Y, LOSS, IGNORE: tl.constexpr, V: tl.constexpr, REGS: tl.constexpr
):
    row = tl.program_id(0)
    thread = tl.arange(0, 1024)
    maximum = tl.full((1024,), float("-inf"), tl.float32)
    for register in tl.static_range(REGS):
        col = thread + register * 1024
        value = tl.load(X + row * STRIDE + col, col < V, float("-inf")).to(tl.float32)
        maximum = tl.maximum(maximum, value)
    maximum = tl.max(maximum, 0)
    partial = tl.full((1024,), 0.0, tl.float32)
    for register in tl.static_range(REGS):
        col = thread + register * 1024
        value = tl.load(X + row * STRIDE + col, col < V, float("-inf")).to(tl.float32)
        partial = partial + libdevice.exp(value - maximum)
    normalizer = libdevice.log(tl.sum(partial, 0))
    target = tl.load(Y + row)
    target_value = tl.load(X + row * STRIDE + target, target != IGNORE, 0.0).to(tl.float32)
    loss = -(target_value - maximum - normalizer)
    tl.store(LOSS + row, tl.where(target == IGNORE, 0.0, loss))
    # Complete the scalar target read before any warp overwrites its logit.
    tl.debug_barrier()
    for register in tl.static_range(REGS):
        col = thread + register * 1024
        value = tl.load(X + row * STRIDE + col, col < V, float("-inf")).to(tl.float32)
        log_probability = (value - maximum) - normalizer
        gradient = libdevice.exp(log_probability) - tl.where(col == target, 1.0, 0.0)
        tl.store(X + row * STRIDE + col, tl.where(target == IGNORE, 0.0, gradient), col < V)


class CrossEntropy(torch.autograd.Function):
    """
    Fused cross-entropy that overwrites the logit tensor with gradients
    during the forward pass so no extra activation memory is needed.
    """

    @staticmethod
    def forward(ctx, inp, target, ignore_index, reduction):
        n_rows, n_cols = inp.shape

        if inp.stride(-1) != 1 or inp.stride(-2) != n_cols:
            inp = inp.contiguous()
        if target.stride(-1) != 1:
            target = target.contiguous()

        n_non_ignore = (target != ignore_index).sum(dtype=torch.int64).view(1)
        n_non_ignore.clamp_(min=1)

        if reduction == "sum" and torch.are_deterministic_algorithms_enabled():
            # Strict training uses PyTorch's log-softmax arithmetic. The fused
            # exp/sum path is mathematically equivalent, but a BF16 rounding
            # boundary can change a near-zero downstream gradient's sign.
            # Keep the destructive API and saved-gradient dtype unchanged;
            # this opt-in path needs additional transient FP32 vocabulary memory.
            with torch.enable_grad():
                logits = inp.detach().float().requires_grad_()
                loss = torch.nn.functional.cross_entropy(
                    logits, target, ignore_index=ignore_index, reduction="sum"
                )
                gradient = torch.autograd.grad(loss, logits)[0]
            inp.copy_(gradient)
            ctx.save_for_backward(inp.detach())
            return loss.detach()

        loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=inp.device)
        BLOCK_SIZE = min(65536 // 2, triton.next_power_of_2(n_cols))

        if reduction == "sum" and 2048 < n_cols <= 9216:
            # This is the register-based FP32 log-softmax range in PyTorch.
            # A generic vector reduction sums a thread's registers in another
            # order and can move a BF16 gradient across a rounding boundary.
            cross_entropy_register_sum[(n_rows,)](
                X=inp,
                STRIDE=inp.stride(-2),
                Y=target,
                LOSS=loss_1d,
                IGNORE=ignore_index,
                V=n_cols,
                REGS=triton.cdiv(n_cols, 1024),
                num_warps=32,
                enable_fp_fusion=False,
            )
        else:
            cross_entropy_fwd[(n_rows,)](
                X_ptr=inp,
                X_stride=inp.stride(-2),
                Y_ptr=target,
                Y_stride=target.stride(-1),
                loss_ptr=loss_1d,
                loss_stride=loss_1d.stride(-1),
                n_cols=n_cols,
                n_non_ignore_ptr=n_non_ignore,
                ignore_idx=ignore_index,
                BLOCK_SIZE=BLOCK_SIZE,
                REDUCTION=reduction,
                num_warps=32,
            )

        loss = loss_1d.sum()
        if reduction == "mean":
            loss = loss / n_non_ignore.float().squeeze()
        ctx.save_for_backward(inp.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        inp.mul_(grad_output.to(inp.dtype))
        return inp, None, None, None


def cross_entropy(
    inp: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    In-place fused cross-entropy loss with mean (default) or sum reduction.

    Overwrites ``inp`` with pre-computed gradients during the forward pass,
    eliminating the need for a separate activation tensor.  All arithmetic
    is performed in FP32; gradients are stored in the original dtype of
    ``inp``. With PyTorch deterministic algorithms enabled, sum reduction
    uses PyTorch FP32 cross-entropy arithmetic before the same gradient store.
    This strict compatibility path allocates temporary FP32 vocabulary tensors;
    ordinary training and the legacy mean-reduction path retain the fused kernel.

    Parameters
    ----------
    inp : torch.Tensor
        Logits of shape ``(N, V)`` where N is the number of tokens and V is
        the vocabulary size.  Modified in-place.
    target : torch.Tensor
        Target indices of shape ``(N,)`` with values in ``[0, V-1]``.
    ignore_index : int
        Target value that is ignored in loss and gradient computation.
    reduction : {"mean", "sum"}
        Use "sum" for token-weighted training that normalizes accumulated FP32
        gradients once after the data-parallel reduction.

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss (float32).
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    return CrossEntropy.apply(inp, target, ignore_index, reduction)
