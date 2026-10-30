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


def reference_any_weight(x, weight, offset, grad_out):
    """
    reference() for any weight dtype: another weight dtype is read as fp32 and convolves x in fp32
    (ATen's fp32 kernel, the same fp32 fma order), y and dx then rounded once to bf16.
    """
    if weight.dtype == x.dtype:
        return reference(x, weight, offset, grad_out)
    y, dx, dw = reference(x.float(), weight.float(), offset, grad_out.float())
    return y.to(x.dtype), dx.to(x.dtype), dw


def assert_dw_in_bound(dw, x, weight, offset, grad_out):
    dw_exact, abs_sum = exact_dw(x, weight, offset, grad_out)
    # fp32 accumulation error, then one rounding to the weight's dtype.
    bound = 4 * torch.finfo(torch.float32).eps * abs_sum + torch.finfo(weight.dtype).eps * dw_exact.abs()  # fmt: skip
    assert ((dw.double() - dw_exact).abs() <= bound + 1e-30).all()


def check(x, weight, grad_out, offset):
    B, L, C = x.shape
    S = L - offset
    y_ref, dx_ref, dw_ref = reference_any_weight(x, weight, offset, grad_out)

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
    assert_dw_in_bound(dw, x, weight, offset, grad_out)
    # The gradient in the output's own (channels-first) layout: the same dx; dw may be reduced in
    # another order (another number of partial sums), so it is held to the same bound, not bitwise.
    grad_out_like = causal_conv1d_empty(x, offset).copy_(grad_out)
    dx2, dw2 = causal_conv1d_bwd(grad_out_like, x, weight, offset)
    assert torch.equal(dx2, dx) and dw2.dtype == dw.dtype
    assert_dw_in_bound(dw2, x, weight, offset, grad_out)
    dx3, dw3 = causal_conv1d_bwd(grad_out, x, weight, offset)
    assert torch.equal(dx3, dx) and torch.equal(dw3, dw)


@pytest.mark.parametrize(
    "B,S,C,offset",
    [
        (1, 4096, 8192, 0),  # qwen3.5-35b-a3b: C = 2 * key_dim + value_dim = 8192
        (1, 2048, 8192, 3),  # qwen3.5 under CP2: 2048 tokens per rank + 3 rows of conv state
        (1, 1000, 512, 0),  # one sequence, length not a multiple of the row blocks
        (1, 100, 256, 3),  # one short sequence
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


@pytest.mark.parametrize("S", [513, 512])
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
def test_kernel_widths(K, S):
    """
    Every width up to 5, at a length that is a multiple of 8 and one that is not.
    """
    x, weight, grad_out = make(1, S, 384, K - 1, K=K, seed=2)
    check(x, weight, grad_out, K - 1)


def test_kernel_width_limit():
    x, weight, grad_out = make(1, 64, 128, 0, K=9, seed=2)
    with pytest.raises(ValueError):
        causal_conv1d_fwd(x, weight, 0)
    with pytest.raises(ValueError):
        causal_conv1d_bwd(grad_out, x, weight, 0)


def test_weight_without_channel_axis():
    """
    The (C, K) weight form gives the same results as (C, 1, K); dw comes back as (C, K).
    """
    for S, C, offset in ((512, 512, 3), (300, 130, 0)):
        x, weight, grad_out = make(1, S, C, offset, seed=10)
        y_ref, dx_ref, dw_ref = reference(x, weight, offset, grad_out)
        weight_2d = weight.view(C, -1)
        assert torch.equal(causal_conv1d_fwd(x, weight_2d, offset), y_ref)
        dx, dw = causal_conv1d_bwd(grad_out, x, weight_2d, offset)
        assert torch.equal(dx, dx_ref) and dw.shape == weight_2d.shape
        assert_dw_in_bound(dw.view(weight.shape), x, weight, offset, grad_out)


def misaligned_copy(t):
    """
    A contiguous copy of t whose data starts 2 bytes past a 16-byte boundary.
    """
    storage = torch.empty(t.numel() + 8, dtype=t.dtype, device=t.device)
    out = storage[1 : 1 + t.numel()].view(t.shape)
    out.copy_(t)
    assert out.data_ptr() % 16 != 0 and out.is_contiguous()
    return out


def test_misaligned_inputs():
    """
    Contiguous x and channels-last gradient that are not 16-byte aligned.
    """
    x, weight, grad_out = make(1, 512, 512, 3, seed=11)
    check(misaligned_copy(x), weight, grad_out, 3)
    y_ref, dx_ref, dw_ref = reference(x, weight, 3, grad_out)
    dx, dw = causal_conv1d_bwd(misaligned_copy(grad_out), x, weight, 3)
    assert torch.equal(dx, dx_ref)
    assert_dw_in_bound(dw, x, weight, 3, grad_out)


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


@pytest.mark.parametrize("B,S", [(1, 0), (0, 5)])
def test_empty(B, S):
    x, weight, grad_out = make(max(B, 1), S, 256, 3)
    x, grad_out = x[:B], grad_out[:B]
    y = causal_conv1d_fwd(x, weight, 3)
    assert y.shape == (B, S, 256) and y.stride() == causal_conv1d_empty(x, 3).stride()
    dx, dw = causal_conv1d_bwd(grad_out, x, weight, 3)
    assert dx.shape == x.shape and dw.shape == weight.shape
    assert not dx.any() and not dw.any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("B,S,C,offset", [(1, 777, 8192, 3), (1, 776, 8192, 3), (2, 1000, 520, 0)])
def test_float_weight(B, S, C, offset, dtype):
    """
    w of another float dtype is read as fp32; dw comes back in w's dtype.
    """
    x, _, grad_out = make(B, S, C, offset, seed=6)
    weight = (torch.randn(C, 1, 4, device="cuda") * 0.5).to(dtype)
    check(x, weight, grad_out, offset)


def test_strided_grad():
    """
    A gradient that is neither channels-last nor channels-first (a strided view).
    """
    x, weight, _ = make(1, 513, 384, 3, seed=7)
    grad_out = torch.randn(1, 513, 768, device="cuda").to(torch.bfloat16)[..., ::2]
    y_ref, dx_ref, dw_ref = reference(x, weight, 3, grad_out)
    dx, dw = causal_conv1d_bwd(grad_out, x, weight, 3)
    assert torch.equal(dx, dx_ref)
    assert_dw_in_bound(dw, x, weight, 3, grad_out)


@pytest.mark.parametrize(
    "B,S,C,offset",
    [
        (1, 4096, 8192, 0),  # one sequence of the qwen3.5 size: one partial sum per channel tile
        (2, 8192, 64, 3),  # one channel tile, two sequences: the tokens split into many partials
    ],
)
def test_backward_deterministic(B, S, C, offset):
    """
    The weight gradient is a fixed-order reduction: ten backward calls are bitwise equal, with the
    gradient channels-last and channels-first.
    """
    x, weight, grad_out = make(B, S, C, offset, seed=8)
    for grad in (grad_out, causal_conv1d_empty(x, offset).copy_(grad_out)):
        dx, dw = causal_conv1d_bwd(grad, x, weight, offset)
        for _ in range(9):
            dx2, dw2 = causal_conv1d_bwd(grad, x, weight, offset)
            assert torch.equal(dx2, dx) and torch.equal(dw2, dw)


def cancelling(S, C, offset, K=4, seed=9, scale=1e3):
    """
    Data whose weight-gradient sums nearly cancel: the second half of the gradient's rows is the
    negation of the first and x repeats with the same period, so every g[t] * x[t + offset - (K - 1)
    + k] has a negated partner except the K - 1 terms that read the zero padding. The weight is fp32,
    so dw comes back in fp32 and the bound is the fp32 accumulation error alone.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    half = S // 2
    base = (torch.randn(1, half + offset, C, generator=gen, device="cuda") * scale).to(
        torch.bfloat16
    )
    x = torch.cat([base, base[:, offset:]], dim=1)
    grad_half = (torch.randn(1, half, C, generator=gen, device="cuda") * scale).to(torch.bfloat16)
    grad_out = torch.cat([grad_half, -grad_half], dim=1)
    weight = torch.randn(C, 1, K, generator=gen, device="cuda") * 0.5
    return x, weight, grad_out


def test_dw_one_vs_many_partials():
    """
    The same nearly cancelling data reduced with one partial sum per channel tile (channels-last
    gradient, one sequence: one block walks the whole sequence) and with many (the gradient
    channels-first, and the sequence duplicated as a batch of two: the tokens split over blocks):
    every dw within the fp32 bound of the exact sum, though each sum is far smaller than its terms.
    """
    x, weight, grad_out = cancelling(4096, 8192, 3)
    dw_exact, abs_sum = exact_dw(x, weight, 3, grad_out)
    assert (dw_exact.abs() / abs_sum).max() < 0.02
    dx, dw = causal_conv1d_bwd(grad_out, x, weight, 3)
    assert_dw_in_bound(dw, x, weight, 3, grad_out)
    grad_out_like = causal_conv1d_empty(x, 3).copy_(grad_out)
    dx_cf, dw_cf = causal_conv1d_bwd(grad_out_like, x, weight, 3)
    assert torch.equal(dx_cf, dx)
    assert_dw_in_bound(dw_cf, x, weight, 3, grad_out)
    x2, grad_out2 = x.expand(2, -1, -1).contiguous(), grad_out.expand(2, -1, -1).contiguous()
    dx2, dw2 = causal_conv1d_bwd(grad_out2, x2, weight, 3)
    assert torch.equal(dx2[0], dx[0]) and torch.equal(dx2[1], dx[0])
    assert_dw_in_bound(dw2, x2, weight, 3, grad_out2)


@pytest.mark.parametrize("B,S,C", [(1, 512, 512), (2, 300, 130)])
def test_side_stream(B, S, C):
    """
    Called under torch.cuda.stream(s), the kernels run on s, after the work already queued there: a
    long sleep, then the writes of x and the gradient; only s is synchronized before the check.
    """
    x, weight, grad_out = make(B, S, C, 3, seed=12)
    y_ref, dx_ref, _ = reference(x, weight, 3, grad_out)
    x_side, grad_side = torch.zeros_like(x), torch.zeros_like(grad_out)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(100_000_000)
        x_side.copy_(x)
        grad_side.copy_(grad_out)
        y = causal_conv1d_fwd(x_side, weight, 3)
        dx, dw = causal_conv1d_bwd(grad_side, x_side, weight, 3)
    side.synchronize()
    assert torch.equal(y, y_ref) and torch.equal(dx, dx_ref)
    assert_dw_in_bound(dw, x, weight, 3, grad_out)
