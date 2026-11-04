"""
Verifies causal_conv1d_fwd, causal_conv1d_bwd and causal_conv1d_empty, the channels-last causal
depthwise conv of the Qwen3.5 Gated DeltaNet, against PyTorch.

Contract: x (B, offset + S, C) contiguous, weight (C, 1, K); output t reads
x[t + offset - (K - 1) + k] for k = 0 .. K - 1, skipping negative indices, i.e.
conv1d(x.transpose(1, 2), weight, groups=C, padding=K - 1)[..., offset:offset + S]. The forward
and the input gradient are bit-identical to ATen's depthwise conv (taps accumulated in fp32 in
order, rounded once). The weight gradient is an fp32 reduction in another order, so it matches the
exact sum to fp32 rounding, then one bf16 rounding. The output is (B, S, C) stored channels-first,
the layout causal_conv1d_empty returns.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import causal_conv1d_bwd, causal_conv1d_empty, causal_conv1d_fwd


def reference(x, weight, offset, grad_out):
    K, S = weight.shape[-1], x.shape[1] - offset
    x_leaf, weight_leaf = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()  # fmt: skip
    y = F.conv1d(x_leaf.transpose(1, 2), weight_leaf, groups=weight.shape[0], padding=K - 1)
    y = y[..., offset : offset + S].transpose(1, 2)
    y.backward(grad_out)
    return y.detach(), x_leaf.grad, weight_leaf.grad


def exact_dw(x, weight, offset, grad_out):
    """
    The float64 weight gradient and the sum of its terms' magnitudes, for the fp32 rounding bound.
    """
    B, L, C = x.shape
    K, S = weight.shape[-1], L - offset
    x_padded = F.pad(x.double(), (0, 0, K - 1, 0))  # K - 1 zero rows of history
    dw = torch.zeros(C, K, dtype=torch.float64, device=x.device)
    abs_sum = torch.zeros_like(dw)
    for k in range(K):
        x_tap = x_padded[:, offset + k : offset + k + S]  # x[t + offset - (K - 1) + k]
        prod = grad_out.double() * x_tap
        dw[:, k] = prod.sum((0, 1))
        abs_sum[:, k] = prod.abs().sum((0, 1))
    return dw.view(weight.shape), abs_sum.view(weight.shape)


def make(B, S, C, offset, K=4, seed=0, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(B, S + offset, C, generator=gen, device="cuda") * scale).to(torch.bfloat16)
    weight = (torch.randn(C, 1, K, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    grad_out = torch.randn(B, S, C, generator=gen, device="cuda").to(torch.bfloat16)
    return x, weight, grad_out


def check(x, weight, grad_out, offset):
    B, L, C = x.shape
    S = L - offset
    y_ref, dx_ref, dw_ref = reference(x, weight, offset, grad_out)

    y = causal_conv1d_fwd(x, weight, offset)
    like = causal_conv1d_empty(x, offset)
    assert y.shape == like.shape == (B, S, C)
    assert y.stride() == like.stride() == (C * S, 1, S)  # (B, C, S) dense, transposed
    assert y.dtype == like.dtype == x.dtype
    assert torch.equal(y, y_ref)

    dx, dw = causal_conv1d_bwd(grad_out, x, weight, offset)
    assert dx.shape == x.shape and dx.dtype == x.dtype
    assert dw.shape == weight.shape and dw.dtype == weight.dtype
    assert torch.equal(dx, dx_ref)
    dw_exact, abs_sum = exact_dw(x, weight, offset, grad_out)
    # fp32 accumulation error, then one bf16 rounding.
    bound = 4 * torch.finfo(torch.float32).eps * abs_sum + torch.finfo(weight.dtype).eps * dw_exact.abs()  # fmt: skip
    assert ((dw.double() - dw_exact).abs() <= bound + 1e-30).all()
    # The gradient in the output's own (channels-first) layout gives the same result.
    grad_out_like = causal_conv1d_empty(x, offset).copy_(grad_out)
    dx2, dw2 = causal_conv1d_bwd(grad_out_like, x, weight, offset)
    assert torch.equal(dx2, dx) and torch.equal(dw2, dw)
    dx3, dw3 = causal_conv1d_bwd(grad_out, x, weight, offset)
    assert torch.equal(dx3, dx) and torch.equal(dw3, dw)


@pytest.mark.parametrize(
    "B,S,C,offset",
    [
        (1, 4096, 8192, 0),  # qwen3.5-35b-a3b: C = 2 * key_dim + value_dim = 8192
        (2, 1000, 520, 0),
        (1, 2051, 640, 3),  # with K - 1 rows of conv state prepended (context parallel)
        (2, 777, 8192, 3),
        (3, 100, 100, 1),  # C not a multiple of the channel block, offset < K - 1
        (1, 300, 130, 7),  # offset > K - 1
    ],
)
def test_matches_conv1d(B, S, C, offset):
    x, weight, grad_out = make(B, S, C, offset)
    check(x, weight, grad_out, offset)


@pytest.mark.parametrize("S,offset", [(1, 0), (2, 0), (3, 3), (1, 3)])
def test_short_sequences(S, offset):
    """
    Fewer outputs than taps.
    """
    x, weight, grad_out = make(2, S, 256, offset, seed=1)
    check(x, weight, grad_out, offset)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_kernel_widths(K):
    x, weight, grad_out = make(1, 513, 384, K - 1, K=K, seed=2)
    check(x, weight, grad_out, K - 1)


def test_large_negative_and_zero():
    x, weight, grad_out = make(1, 1024, 512, 3, seed=3, scale=1e3)
    x[:, :100] = 0
    weight[:16] = 0
    weight[16:32] = -weight[16:32] * 64
    check(x, weight, grad_out, 3)


def test_history_only_changes_the_first_outputs():
    """
    offset rows of history feed exactly the first K - 1 outputs.
    """
    x, weight, _ = make(1, 512, 256, 3, seed=4)
    y = causal_conv1d_fwd(x, weight, 3)
    y_cold = causal_conv1d_fwd(x[:, 3:].contiguous(), weight, 0)
    assert torch.equal(y[:, 3:], y_cold[:, 3:])
    assert not torch.equal(y[:, :3], y_cold[:, :3])
