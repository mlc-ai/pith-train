"""
Tests of indexed_bias_add against PyTorch: the output equals x + bias[group_ids] exactly, and the
bias gradient (a segmented sum over tokens sorted by group) is compared against an fp32 ground
truth, since PyTorch's bf16 atomic-add backward is itself lossy.
"""

import pytest
import torch

from pithtrain.operators.indexed_bias_add import indexed_bias_add


def build_sorted_inputs(
    M: int,
    H: int,
    E: int,
    dtype: torch.dtype,
    device: str,
    ks=None,
):
    """
    Tokens are sorted by group, as after dispatch. ks, when given, fixes the per-group token counts
    (summing to M).
    """
    torch.manual_seed(0)
    if ks is None:
        # Groups may be empty.
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


def fp32_ground_truth(x, bias, group_ids, grad_out):
    x32 = x.float().detach().requires_grad_()
    bias32 = bias.float().detach().requires_grad_()
    out = x32 + bias32[group_ids]
    out.backward(grad_out.float())
    return out.detach(), x32.grad, bias32.grad


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
    device = "cuda"
    x, bias, group_ids, offs = build_sorted_inputs(M, H, E, dtype, device)
    grad_out = torch.randn(M, H, device=device, dtype=dtype)

    x_ref = x.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()
    (x_ref + bias_ref[group_ids]).backward(grad_out)

    x_cu = x.detach().clone().requires_grad_()
    bias_cu = bias.detach().clone().requires_grad_()
    out_cu = indexed_bias_add(x_cu, bias_cu, group_ids, offs)
    out_cu.backward(grad_out)

    out_ref = x_ref + bias_ref[group_ids]
    torch.testing.assert_close(out_cu, out_ref, atol=0.0, rtol=0.0)

    torch.testing.assert_close(x_cu.grad, x_ref.grad, atol=0.0, rtol=0.0)

    # The tolerance covers rounding the fp32 segment sum once to the bias dtype.
    _, _, bias_grad_gt = fp32_ground_truth(x, bias, group_ids, grad_out)
    atol, rtol = (5e-1, 5e-2) if dtype == torch.bfloat16 else (1e-1, 1e-2)
    torch.testing.assert_close(bias_cu.grad.float(), bias_grad_gt, atol=atol, rtol=rtol)


def test_indexed_bias_add_more_accurate_than_bf16_reference():
    """
    The kernel accumulates in fp32 while autograd uses bf16 atomic adds.
    """
    device = "cuda"
    M, H, E = 4096, 5760, 8
    x, bias, group_ids, offs = build_sorted_inputs(M, H, E, torch.bfloat16, device)
    grad_out = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    _, _, bias_grad_gt = fp32_ground_truth(x, bias, group_ids, grad_out)

    bias_ref = bias.detach().clone().requires_grad_()
    (x.detach().requires_grad_() + bias_ref[group_ids]).backward(grad_out)
    err_ref = (bias_ref.grad.float() - bias_grad_gt).abs().max().item()

    bias_cu = bias.detach().clone().requires_grad_()
    indexed_bias_add(x.detach().requires_grad_(), bias_cu, group_ids, offs).backward(grad_out)
    err_cu = (bias_cu.grad.float() - bias_grad_gt).abs().max().item()

    assert err_cu < err_ref / 5, (
        f"Triton path (max-abs err {err_cu:.3e}) should be much closer to fp32 GT "
        f"than the bf16 atomic-add reference (max-abs err {err_ref:.3e})."
    )


def test_indexed_bias_add_handles_empty_groups():
    """
    A group with zero tokens gets a zero gradient row, not NaN (loop bounds with start == end).
    """
    device = "cuda"
    H, E = 256, 6
    ks = [100, 0, 50, 200, 0, 150]
    M = sum(ks)
    x, bias, group_ids, offs = build_sorted_inputs(M, H, E, torch.bfloat16, device, ks=ks)
    grad_out = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    bias_cu = bias.detach().clone().requires_grad_()
    indexed_bias_add(x.detach().requires_grad_(), bias_cu, group_ids, offs).backward(grad_out)

    assert torch.isfinite(bias_cu.grad).all()
    assert (bias_cu.grad[1] == 0).all()
    assert (bias_cu.grad[4] == 0).all()
    assert bias_cu.grad[0].abs().sum() > 0
    assert bias_cu.grad[3].abs().sum() > 0


def test_indexed_bias_add_realistic_gpt_oss_shape():
    """
    gpt-oss-20b: E is the experts per rank, H the gate_up_proj width.
    """
    device = "cuda"
    H, E = 5760, 8
    ks = [512, 768, 320, 256, 1024, 384, 480, 352]
    M = sum(ks)
    x, bias, group_ids, offs = build_sorted_inputs(M, H, E, torch.bfloat16, device, ks=ks)
    grad_out = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    _, _, bias_grad_gt = fp32_ground_truth(x, bias, group_ids, grad_out)

    x_cu = x.detach().clone().requires_grad_()
    bias_cu = bias.detach().clone().requires_grad_()
    out_cu = indexed_bias_add(x_cu, bias_cu, group_ids, offs)
    out_cu.backward(grad_out)

    torch.testing.assert_close(out_cu, x + bias[group_ids], atol=0.0, rtol=0.0)
    torch.testing.assert_close(bias_cu.grad.float(), bias_grad_gt, atol=5e-1, rtol=5e-2)


@pytest.mark.parametrize("offs_len", [0, 8])
def test_indexed_bias_add_no_rows(offs_len):
    """
    An EP rank that receives no tokens (M == 0, offsets empty or all zero) gets a bias-shaped zero
    gradient, not a (0, H) one.
    """
    device = "cuda"
    E, H = 8, 2880
    x = torch.zeros(0, H, dtype=torch.bfloat16, device=device, requires_grad=True)
    bias = torch.randn(E, H, dtype=torch.bfloat16, device=device, requires_grad=True)
    group_ids = torch.zeros(0, dtype=torch.int64, device=device)
    offs = torch.zeros(offs_len, dtype=torch.int32, device=device)
    out = indexed_bias_add(x, bias, group_ids, offs)
    out.sum().backward()
    assert bias.grad is not None and bias.grad.shape == bias.shape
    assert torch.count_nonzero(bias.grad) == 0
    assert x.grad.shape == x.shape
