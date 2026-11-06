"""Loss and every logit gradient, including targets across vocabulary tiles."""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.operators.cross_entropy import cross_entropy


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [32, 32768, 65536, 152064])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_target_gradient(dtype, vocab, reduction):
    generator = torch.Generator(device="cuda").manual_seed(37)
    logits = torch.randn((128, vocab), generator=generator, device="cuda", dtype=dtype) * 0.5
    labels = torch.randint(vocab, (128,), generator=generator, device="cuda")
    # Exercise the first/last lane and both sides of the 32768-element tile boundary.
    for index, target in enumerate((0, 31, 32767, 32768, 65535, 65536, vocab - 1)):
        if target < vocab:
            labels[index] = target
    labels[9::11] = -100
    reference = logits.float().detach().requires_grad_()
    expected_loss = F.cross_entropy(reference, labels, reduction=reduction)
    expected_loss.backward()
    expected_gradient = reference.grad.to(dtype)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-4)
    for repeat in range(8):
        # The fused forward mutates logits. Each repetition needs fresh storage.
        # Repeating identical inputs exposes the cross-warp target-store race.
        inputs = logits.detach().clone().requires_grad_()
        loss = cross_entropy(inputs, labels, reduction=reduction)
        loss.backward()
        torch.testing.assert_close(loss, expected_loss.detach(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            inputs.grad,
            expected_gradient,
            rtol=rtol,
            atol=atol,
            msg=lambda msg: f"{dtype=}, {vocab=}, {repeat=}: {msg}",
        )
        assert torch.count_nonzero(inputs.grad[labels == -100]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_all_ignored(dtype, reduction):
    inputs = torch.randn((4, 152064), device="cuda", dtype=dtype, requires_grad=True)
    labels = torch.full((4,), -100, device="cuda", dtype=torch.int64)
    loss = cross_entropy(inputs, labels, reduction=reduction)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(inputs.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sum_gradient_is_independent_of_valid_token_count(dtype):
    # An identical row's sum gradient cannot depend on how many other rows are
    # labeled. BF16 mean storage followed by count multiplication violates this.
    values = torch.linspace(-2, 2, 257, device="cuda", dtype=dtype).repeat(257, 1)
    first = None
    for count in (1, 3, 62, 257):
        labels = torch.full((257,), -100, device="cuda", dtype=torch.int64)
        labels[:count] = 127
        inputs = values.clone().requires_grad_()
        loss = cross_entropy(inputs, labels, reduction="sum")
        loss.backward()
        if first is None:
            first = inputs.grad[0].clone()
        torch.testing.assert_close(inputs.grad[0], first, rtol=0, atol=0)
        assert inputs.grad[count:].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_default_mean_matches_explicit_mean(dtype):
    values = torch.linspace(-2, 2, 257, device="cuda", dtype=dtype).repeat(62, 1)
    labels = torch.arange(62, device="cuda") % 257
    default, explicit = values.clone().requires_grad_(), values.clone().requires_grad_()
    actual, expected = (
        cross_entropy(default, labels),
        cross_entropy(explicit, labels, reduction="mean"),
    )
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(default.grad, explicit.grad, rtol=0, atol=0)
