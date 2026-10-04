import pytest
import torch

from pithtrain.operators.log_prob import log_prob, reference_log_prob


def forward_backward(fn, logits, target, temperature, grad_logp, grad_entropy):
    inp = logits.detach().clone().requires_grad_()
    logp, entropy = fn(inp, target, temperature)
    assert torch.equal(inp, logits)
    torch.autograd.backward((logp, entropy), (grad_logp, grad_entropy))
    return logp.detach(), entropy.detach(), inp.grad


@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.parametrize("vocab", [151936, 8192, 1000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_rows", [1, 67])
def test_log_prob(n_rows, dtype, vocab, temperature):
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((n_rows, vocab), generator=generator, device="cuda") * 3
    logits = logits.to(dtype)
    target = torch.randint(vocab, (n_rows,), generator=generator, device="cuda")
    # The first and last lane, and both sides of the 32768-wide tile boundary.
    for row, index in enumerate((0, 32767, 32768, 65535, 65536, vocab - 1)):
        if row < n_rows and index < vocab:
            target[row] = index
    grad_logp, grad_entropy = torch.randn((2, n_rows), generator=generator, device="cuda")
    # Mask rows without losing the tile-boundary cases above.
    grad_logp[6::3] = grad_entropy[6::3] = 0.0

    args = (logits, target, temperature, grad_logp, grad_entropy)
    logp, entropy, grad = forward_backward(log_prob, *args)
    expected_logp, expected_entropy, expected_grad = forward_backward(reference_log_prob, *args)
    torch.testing.assert_close(logp, expected_logp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=1e-5)
    # Both round an FP32 gradient once to the logits dtype, so bf16 may differ by one ulp.
    rtol, atol = (1e-4, 1e-6) if dtype == torch.float32 else (1e-2, 1e-6)
    torch.testing.assert_close(grad, expected_grad, rtol=rtol, atol=atol)
    assert not grad[6::3].any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_log_prob_extreme_logits(dtype):
    vocab = 152064
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((6, vocab), generator=generator, device="cuda")
    target = torch.randint(vocab, (6,), generator=generator, device="cuda")
    logits[0] = 7.0  # uniform: entropy log(V)
    logits[1, target[1]] = 80.0  # the target takes all the mass
    logits[2, 0] = 80.0  # another token takes all the mass
    logits[3] *= 1000.0  # a few tokens take all the mass
    logits[4] = logits[4] * 30.0 - 1e4  # large offset
    logits[5] *= 1e-3  # nearly uniform
    logits = logits.to(dtype)
    grad_logp, grad_entropy = torch.randn((2, 6), generator=generator, device="cuda")

    # Against FP64. T = 0.5 keeps x / T exact in FP32: otherwise rounding it, as PyTorch's FP32
    # path does and the kernels may, costs about 1e-3 in log p at |x| = 1e4.
    for temperature in (1.0, 0.5):
        logp, entropy, grad = forward_backward(
            log_prob, logits, target, temperature, grad_logp, grad_entropy
        )
        grads = (grad_logp.double(), grad_entropy.double())
        truth = forward_backward(reference_log_prob, logits.double(), target, temperature, *grads)
        torch.testing.assert_close(logp.double(), truth[0], rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(entropy.double(), truth[1], rtol=1e-6, atol=1e-5)
        rtol, atol = (1e-4, 1e-6) if dtype == torch.float32 else (1e-2, 1e-6)
        torch.testing.assert_close(grad, truth[2].to(dtype), rtol=rtol, atol=atol)


def test_log_prob_out_of_range_targets():
    """The 2^40 target faults if its logit load is unmasked; entropy must remain unaffected."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((6, 8192), generator=generator, device="cuda", dtype=torch.bfloat16)
    target = torch.tensor([-100, 0, -1, 8191, 8192, 2**40], device="cuda")
    grad_logp, grad_entropy = torch.randn((2, 6), generator=generator, device="cuda")
    args = (logits, target, 0.6, grad_logp, grad_entropy)
    logp, entropy, grad = forward_backward(log_prob, *args)
    expected_logp, expected_entropy, expected_grad = forward_backward(reference_log_prob, *args)
    torch.testing.assert_close(logp, expected_logp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-2, atol=1e-6)


def test_log_prob_sum_of_outputs():
    """Exercise stride-0 upstream gradients and noncontiguous logits and targets."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((8192, 33), generator=generator, device="cuda", dtype=torch.bfloat16).t()
    target = torch.randint(8192, (33, 2), generator=generator, device="cuda")[:, 0]
    grads = []
    for fn in (log_prob, reference_log_prob):
        inp = logits.detach().clone().requires_grad_()
        logp, entropy = fn(inp, target, 0.6)
        (logp.sum() - 0.01 * entropy.sum()).backward()
        grads.append(inp.grad)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-2, atol=1e-7)


def test_log_prob_under_torch_compile():
    """torch.compile passes temperature as FP64; kernels must still match eager bitwise."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((16, 8192), generator=generator, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(8192, (16,), generator=generator, device="cuda")
    grad_logp, grad_entropy = torch.randn((2, 16), generator=generator, device="cuda")
    args = (logits, target, 0.6, grad_logp, grad_entropy)
    compiled = forward_backward(torch.compile(log_prob, fullgraph=True), *args)
    eager = forward_backward(log_prob, *args)
    assert all(torch.equal(c, e) for c, e in zip(compiled, eager))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_log_prob_backward_is_deterministic(dtype):
    vocab = 152064
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((128, vocab), generator=generator, device="cuda").to(dtype)
    target = torch.randint(vocab, (128,), generator=generator, device="cuda")
    target[:6] = torch.tensor([0, 32767, 32768, 65535, 65536, vocab - 1])
    grad_logp, grad_entropy = torch.randn((2, 128), generator=generator, device="cuda")
    first = forward_backward(log_prob, logits, target, 1.0, grad_logp, grad_entropy)[2]
    for _ in range(8):
        again = forward_backward(log_prob, logits, target, 1.0, grad_logp, grad_entropy)[2]
        assert torch.equal(again, first)


def test_log_prob_memory():
    """The tail crosses 2^31 elements, requiring 64-bit offsets; gradients reuse logits storage."""
    n_rows, vocab, n_tail = 16384, 151936, 4
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn((n_rows, vocab), generator=generator, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(vocab, (n_rows,), generator=generator, device="cuda")
    grad_logp, grad_entropy = torch.zeros((2, n_rows), device="cuda")
    grad_logp[-n_tail:], grad_entropy[-n_tail:] = torch.randn(
        (2, n_tail), generator=generator, device="cuda"
    )
    tail = logits[-n_tail:].clone()
    logits.requires_grad_()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    logp, entropy = log_prob(logits, target)
    (grad,) = torch.autograd.grad((logp, entropy), logits, (grad_logp, grad_entropy))
    peak = torch.cuda.max_memory_allocated() - base
    assert grad.data_ptr() == logits.data_ptr()
    assert peak < n_rows * vocab, f"{peak=} bytes, one byte per logit is {n_rows * vocab}"

    args = (tail, target[-n_tail:], 1.0, grad_logp[-n_tail:], grad_entropy[-n_tail:])
    expected_logp, expected_entropy, expected_grad = forward_backward(reference_log_prob, *args)
    torch.testing.assert_close(logp[-n_tail:], expected_logp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy[-n_tail:], expected_entropy, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad[-n_tail:], expected_grad, rtol=1e-2, atol=1e-6)
    assert not grad[:-n_tail].any()


def test_log_prob_rejects_other_saved_uses_of_the_logits():
    logits = torch.randn((4, 1000), device="cuda", requires_grad=True)
    target = torch.randint(1000, (4,), device="cuda")
    lse = torch.logsumexp(logits, dim=-1)  # saves logits; its backward runs after log_prob's
    logp, _ = log_prob(logits, target)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        (logp + lse).sum().backward()
    first, _ = log_prob(logits[:2], target[:2])
    second, _ = log_prob(logits[2:], target[2:])
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        (first + second).sum().backward()
