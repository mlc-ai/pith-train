"""
Correctness tests for the IndexedBiasAdd autograd Function.

The Function computes the same output as ``x + bias[group_ids]``; the
custom backward swaps PyTorch's bf16 atomic-add bias-grad path for a
Triton segmented-sum.  Tokens are required to be sorted by group, which
is always the case post prepare_dispatch.

Each test compares against an fp32 ground truth rather than the bf16
autograd reference, because the bf16 atomic-add path is itself lossy
(see test_indexed_bias_add_more_accurate_than_bf16_reference).
"""

import pytest
import torch

from pithtrain.operators.indexed_bias_add import indexed_bias_add


def _build_sorted_inputs(
    M: int,
    H: int,
    E: int,
    dtype: torch.dtype,
    device: str,
    ks=None,
):
    """
    Build sorted-by-group inputs that match the layout post-dispatch.

    Returns (x, bias, group_ids, offs) all on device.  ``ks``, when given,
    fixes the per-group token counts (must sum to M).
    """
    torch.manual_seed(0)
    if ks is None:
        # Random load distribution that sums to M; allow zero-sized groups.
        weights = torch.rand(E, device=device) + 0.1
        ks = (weights * (M / weights.sum())).round().to(torch.int64)
        ks[-1] = M - ks[:-1].sum()
        assert ks.sum().item() == M
    else:
        ks = torch.tensor(ks, device=device, dtype=torch.int64)
    offs = torch.cumsum(ks, dim=0).to(torch.int32)
    group_ids = torch.searchsorted(
        offs.to(torch.int64),
        torch.arange(M, device=device, dtype=torch.int64),
        right=True,
    ).clamp_(max=E - 1)
    x = torch.randn(M, H, device=device, dtype=dtype)
    bias = torch.randn(E, H, device=device, dtype=dtype)
    return x, bias, group_ids, offs


def _fp32_ground_truth(x, bias, group_ids, grad_out):
    """
    Compute (output, x.grad, bias.grad) under pure fp32 autograd.  The
    bf16 atomic-add backward used by PyTorch's default index_put_ is
    itself lossy, so we measure both paths against this fp32 oracle.
    """
    xf = x.float().detach().requires_grad_()
    bf = bias.float().detach().requires_grad_()
    out = xf + bf[group_ids]
    out.backward(grad_out.float())
    return out.detach(), xf.grad, bf.grad


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "M,H,E",
    [
        (128, 256, 4),
        (4096, 2880, 8),
        (4096, 5760, 8),
        (1024, 768, 32),
        (37, 65, 3),
    ],
)
def test_indexed_bias_add_forward_backward(M, H, E, dtype):
    """
    Forward output and both gradients agree with autograd within bf16 tol.

    x.grad should be exactly grad_out (identity-add); bias.grad goes
    through the Triton segmented sum.
    """
    device = "cuda"
    x, bias, group_ids, offs = _build_sorted_inputs(M, H, E, dtype, device)
    grad_out = torch.randn(M, H, device=device, dtype=dtype)

    # Reference autograd path with the same dtype (the slow bf16/fp16 atomic).
    x_ref = x.detach().clone().requires_grad_()
    b_ref = bias.detach().clone().requires_grad_()
    (x_ref + b_ref[group_ids]).backward(grad_out)

    # Triton-backed Function under test.
    x_cu = x.detach().clone().requires_grad_()
    b_cu = bias.detach().clone().requires_grad_()
    out_cu = indexed_bias_add(x_cu, b_cu, group_ids, offs)
    out_cu.backward(grad_out)

    # Forward equality is exact; the underlying gather + add is identical.
    out_ref = x_ref + b_ref[group_ids]
    torch.testing.assert_close(out_cu, out_ref, atol=0.0, rtol=0.0)

    # x.grad is identity-pass; should match exactly.
    torch.testing.assert_close(x_cu.grad, x_ref.grad, atol=0.0, rtol=0.0)

    # bias.grad: both paths go through the same atomic noise budget against
    # the fp32 oracle.  Loosen tolerance to cover bf16 atomic round-off.
    _, _, bias_grad_gt = _fp32_ground_truth(x, bias, group_ids, grad_out)
    atol, rtol = (5e-1, 5e-2) if dtype == torch.bfloat16 else (1e-1, 1e-2)
    torch.testing.assert_close(b_cu.grad.float(), bias_grad_gt, atol=atol, rtol=rtol)


def test_indexed_bias_add_more_accurate_than_bf16_reference():
    """
    The Triton path uses fp32 accumulation; the autograd default uses
    bf16 atomic adds.  Against a fp32 ground truth, the Triton path
    should be at least an order of magnitude more accurate.
    """
    device = "cuda"
    M, H, E = 4096, 5760, 8
    x, bias, group_ids, offs = _build_sorted_inputs(M, H, E, torch.bfloat16, device)
    grad_out = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    _, _, bias_grad_gt = _fp32_ground_truth(x, bias, group_ids, grad_out)

    b_ref = bias.detach().clone().requires_grad_()
    (x.detach().requires_grad_() + b_ref[group_ids]).backward(grad_out)
    err_ref = (b_ref.grad.float() - bias_grad_gt).abs().max().item()

    b_cu = bias.detach().clone().requires_grad_()
    indexed_bias_add(x.detach().requires_grad_(), b_cu, group_ids, offs).backward(grad_out)
    err_cu = (b_cu.grad.float() - bias_grad_gt).abs().max().item()

    assert err_cu < err_ref / 5, (
        f"Triton path (max-abs err {err_cu:.3e}) should be much closer to fp32 GT "
        f"than the bf16 atomic-add reference (max-abs err {err_ref:.3e})."
    )


def test_indexed_bias_add_handles_empty_groups():
    """
    A group with zero tokens should produce a zero gradient row, not a NaN.
    This stresses the Triton kernel's loop-bounds: start == end.
    """
    device = "cuda"
    H, E = 256, 6
    # Group 1 and group 4 are empty; group 0 takes 100 tokens, etc.
    ks = [100, 0, 50, 200, 0, 150]
    M = sum(ks)
    x, bias, group_ids, offs = _build_sorted_inputs(M, H, E, torch.bfloat16, device, ks=ks)
    grad_out = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    b_cu = bias.detach().clone().requires_grad_()
    indexed_bias_add(x.detach().requires_grad_(), b_cu, group_ids, offs).backward(grad_out)

    assert torch.isfinite(b_cu.grad).all()
    # Empty-group rows must be exactly zero.
    assert (b_cu.grad[1] == 0).all()
    assert (b_cu.grad[4] == 0).all()
    # Non-empty rows must be non-trivial.
    assert b_cu.grad[0].abs().sum() > 0
    assert b_cu.grad[3].abs().sum() > 0


def test_indexed_bias_add_realistic_gpt_oss_shape():
    """
    Realistic gpt-oss-20b expert dims: experts_per_rank=8, gate_up_proj
    width 2 * intermediate_size = 5760, ~512 tokens per expert on average.
    """
    device = "cuda"
    H, E = 5760, 8
    ks = [512, 768, 320, 256, 1024, 384, 480, 352]
    M = sum(ks)
    x, bias, group_ids, offs = _build_sorted_inputs(M, H, E, torch.bfloat16, device, ks=ks)
    grad_out = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    _, _, bias_grad_gt = _fp32_ground_truth(x, bias, group_ids, grad_out)

    x_cu = x.detach().clone().requires_grad_()
    b_cu = bias.detach().clone().requires_grad_()
    out_cu = indexed_bias_add(x_cu, b_cu, group_ids, offs)
    out_cu.backward(grad_out)

    # Forward is exact.
    torch.testing.assert_close(out_cu, x + bias[group_ids], atol=0.0, rtol=0.0)
    # bias.grad within bf16-atomic tolerance of the fp32 GT.
    torch.testing.assert_close(b_cu.grad.float(), bias_grad_gt, atol=5e-1, rtol=5e-2)
