"""
Contract: the fp32 mean cross-entropy of the (N, V) logits against target over the rows whose
target is not ignore_index (dividing by at least one, so all-ignored gives 0). The logits are
overwritten with the gradient of that mean, (softmax(x) - onehot(y)) / n on the n counted rows and
zero on ignored rows, rounded once to the logits' dtype. grad is inp itself when inp is
row-contiguous, otherwise a contiguous copy.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import cross_entropy_fwd_bwd

IGNORE = -100


def reference(logits, target, ignore_index):
    x = logits.float()
    keep = target != ignore_index
    num_counted = max(int(keep.sum()), 1)
    loss = F.cross_entropy(x, target, ignore_index=ignore_index, reduction="sum") / num_counted
    grad = torch.softmax(x, dim=-1) / num_counted
    rows = torch.nonzero(keep).squeeze(1)
    grad[rows, target[rows]] -= 1.0 / num_counted
    grad[~keep] = 0
    return loss, grad, num_counted


def check(logits, target, ignore_index=IGNORE):
    want_loss, want_grad, num_counted = reference(logits, target, ignore_index)
    inp = logits.clone()
    loss, grad = cross_entropy_fwd_bwd(inp, target, ignore_index)
    assert loss.dtype == torch.float32 and loss.dim() == 0
    assert grad.shape == logits.shape and grad.dtype == logits.dtype
    assert grad.data_ptr() == inp.data_ptr()  # in place on row-contiguous logits
    torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
    rtol = 1e-5 if logits.dtype == torch.float32 else torch.finfo(logits.dtype).eps
    # (p_y - 1) / n cancels when p_y ~ 1: fp32 noise is absolute there, of order eps32 / n.
    # fp16 gradients of large vocabularies are subnormal: their spacing is absolute.
    atol = 1e-6 / num_counted + torch.finfo(logits.dtype).smallest_normal * torch.finfo(logits.dtype).eps  # fmt: skip
    torch.testing.assert_close(grad.float(), want_grad, rtol=rtol, atol=atol)
    # The target entry must hold (p_y - 1) / n, never a stale logit, and ignored rows are zero.
    keep = target != ignore_index
    rows = torch.nonzero(keep).squeeze(1)
    grad_at_target = grad.float()[rows, target[rows]]
    torch.testing.assert_close(grad_at_target, want_grad[rows, target[rows]], rtol=rtol, atol=atol)
    assert torch.all(grad_at_target <= 0)
    assert torch.all(grad[~keep] == 0)
    return loss, grad


# vocab sizes: small, not a power of two, and the four models' vocabularies
# (deepseek-v2-lite 102400, qwen3-30b-a3b 151936, gpt-oss-20b 201088, qwen3.5-35b-a3b 248320).
@pytest.mark.parametrize("vocab", [1000, 32000, 102400, 151936, 201088, 248320])
def test_loss_and_grad(vocab):
    gen = torch.Generator(device="cuda").manual_seed(0)
    tokens = 4096 if vocab <= 32000 else 1024
    logits = (torch.randn(tokens, vocab, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    target = torch.randint(0, vocab, (tokens,), generator=gen, device="cuda")
    target[::7] = IGNORE
    check(logits, target)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtypes(dtype):
    gen = torch.Generator(device="cuda").manual_seed(1)
    logits = (torch.randn(777, 5003, generator=gen, device="cuda") * 3).to(dtype)
    target = torch.randint(0, 5003, (777,), generator=gen, device="cuda")
    target[5::11] = IGNORE
    check(logits, target)


def test_large_logits_stable():
    gen = torch.Generator(device="cuda").manual_seed(2)
    logits = (torch.randn(512, 151936, generator=gen, device="cuda") * 50).to(torch.bfloat16)
    logits[:8, :1000] = -1e4
    logits[8:16, 7] = 3e4  # one dominant class: p ~ 1, gradient cancels to ~0
    target = torch.randint(0, 151936, (512,), generator=gen, device="cuda")
    target[8:16] = 7
    loss, grad = check(logits, target)
    assert torch.isfinite(loss) and torch.isfinite(grad).all()


def test_target_at_edges_and_one_column():
    gen = torch.Generator(device="cuda").manual_seed(3)
    V = 70000  # more than one 32768-column block, with a ragged tail
    logits = torch.randn(64, V, generator=gen, device="cuda").to(torch.bfloat16)
    target = torch.tensor([0, V - 1, 32767, 32768, 65535, 65536] * 10 + [1, 2, 3, 4], device="cuda")
    check(logits, target)
    one_column = torch.randn(16, 1, device="cuda")
    loss, grad = check(one_column, torch.zeros(16, dtype=torch.long, device="cuda"))
    assert loss.item() == 0 and torch.all(grad == 0)


def test_all_ignored_gives_zero():
    logits = torch.randn(32, 1000, device="cuda").to(torch.bfloat16)
    target = torch.full((32,), IGNORE, device="cuda")
    loss, grad = cross_entropy_fwd_bwd(logits.clone(), target, IGNORE)
    assert loss.item() == 0.0
    assert torch.all(grad == 0)


def test_other_ignore_index():
    gen = torch.Generator(device="cuda").manual_seed(4)
    logits = torch.randn(1000, 4096, generator=gen, device="cuda").to(torch.bfloat16)
    target = torch.randint(0, 4096, (1000,), generator=gen, device="cuda")
    target[::3] = 5  # 5 is a real class elsewhere, here the ignore index
    check(logits, target, ignore_index=5)


def test_single_row_and_mean_over_counted_rows():
    logits = torch.randn(1, 151936, device="cuda").to(torch.bfloat16)
    check(logits, torch.tensor([123], device="cuda"))
    logits = torch.randn(100, 1000, device="cuda").to(torch.bfloat16)
    target = torch.full((100,), IGNORE, device="cuda")
    target[42] = 9
    loss, _ = check(logits, target)
    torch.testing.assert_close(loss, F.cross_entropy(logits[42:43].float(), target[42:43]))


def test_non_contiguous_logits_return_a_copy():
    gen = torch.Generator(device="cuda").manual_seed(5)
    wide = torch.randn(300, 5000, generator=gen, device="cuda").to(torch.bfloat16)
    logits = wide[:, :4000]
    target = torch.randint(0, 4000, (300,), generator=gen, device="cuda")
    target[::4] = IGNORE
    want_loss, want_grad, num_counted = reference(logits, target, IGNORE)
    loss, grad = cross_entropy_fwd_bwd(logits, target, IGNORE)
    assert grad.is_contiguous() and grad.data_ptr() != wide.data_ptr()
    torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad.float(), want_grad, rtol=2**-8, atol=1e-6 / num_counted)
    strided_target = torch.stack([target, torch.zeros_like(target)], 1).reshape(-1)[::2]
    assert not strided_target.is_contiguous() or strided_target.stride(-1) != 1
    loss2, _ = cross_entropy_fwd_bwd(logits.contiguous(), strided_target, IGNORE)
    torch.testing.assert_close(loss2, want_loss, rtol=1e-5, atol=1e-5)
