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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [1, 2, 7, 8, 9, 15, 16, 17, 5003, 70001])
def test_ragged_vocab_and_row_alignment(dtype, vocab):
    # An odd row length shifts every row's 16-byte alignment, so rows start at each offset in turn.
    gen = torch.Generator(device="cuda").manual_seed(vocab)
    tokens = 64 if vocab > 5003 else 257
    logits = (torch.randn(tokens, vocab, generator=gen, device="cuda") * 3).to(dtype)
    target = torch.randint(0, vocab, (tokens,), generator=gen, device="cuda")
    target[:4] = torch.tensor([0, vocab - 1, vocab // 2, max(vocab - 2, 0)])
    target[5::9] = IGNORE
    check(logits, target)


@pytest.mark.parametrize("vocab", [102400, 151936, 201088, 248320])
def test_target_on_every_partition_boundary(vocab):
    # Columns around every split of the row into 2, 4 and 8 slices and their halves, the first and
    # last columns, on rows that start at different 16-byte offsets.
    columns = {0, 1, vocab - 2, vocab - 1}
    units = vocab * 2 // 16
    for parts in (2, 4, 8, 16):
        slice_units = (units + parts - 1) // parts
        for k in range(1, parts):
            for edge in (k * slice_units * 8, k * slice_units * 8 + slice_units * 4):
                columns.update(c for c in (edge - 9, edge - 8, edge - 1, edge, edge + 1, edge + 7, edge + 8) if 0 <= c < vocab)
    columns = sorted(columns)
    gen = torch.Generator(device="cuda").manual_seed(6)
    for offset in (0, 3):
        storage = torch.randn(len(columns) * vocab + offset, generator=gen, device="cuda").to(torch.bfloat16)
        logits = storage[offset:].view(len(columns), vocab)
        target = torch.tensor(columns, device="cuda")
        logits[torch.arange(len(columns)), target] += 12  # the target dominates: p_y - 1 is small
        check(logits, target)
        want_loss, want_grad, num_counted = reference(logits, target, IGNORE)
        loss, grad = cross_entropy_fwd_bwd(logits, target, IGNORE)
        assert grad.data_ptr() == storage.data_ptr() + offset * 2  # in place despite the offset
        torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad.float(), want_grad, rtol=torch.finfo(torch.bfloat16).eps, atol=1e-6 / num_counted)


def check_narrow_target(logits, target, ignore_index=IGNORE):
    # The reference indexes with and F.cross_entropy takes int64 targets; the kernel gets target as is.
    want_loss, want_grad, num_counted = reference(logits, target.long(), ignore_index)
    before = target.clone()
    loss, grad = cross_entropy_fwd_bwd(logits.clone(), target, ignore_index)
    assert torch.equal(target, before)  # target is never modified
    torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad.float(), want_grad, rtol=torch.finfo(logits.dtype).eps, atol=1e-6 / num_counted)


@pytest.mark.parametrize("target_dtype", [torch.int32, torch.int64, torch.int16, torch.uint8])
def test_target_dtypes_and_strides(target_dtype):
    gen = torch.Generator(device="cuda").manual_seed(7)
    vocab = 200 if target_dtype == torch.uint8 else 4096
    ignore_index = 255 if target_dtype == torch.uint8 else IGNORE
    logits = torch.randn(300, vocab, generator=gen, device="cuda").to(torch.bfloat16)
    target = torch.randint(0, vocab, (300,), generator=gen, device="cuda").to(target_dtype)
    target[::6] = ignore_index
    check_narrow_target(logits, target, ignore_index)
    strided = torch.stack([target, torch.zeros_like(target)], 1).reshape(-1)[::2]
    assert strided.stride(-1) == 2
    check_narrow_target(logits, strided, ignore_index)


def test_ignore_index_outside_int32():
    gen = torch.Generator(device="cuda").manual_seed(8)
    logits = torch.randn(200, 3000, generator=gen, device="cuda").to(torch.bfloat16)
    target = torch.randint(0, 3000, (200,), generator=gen, device="cuda")
    target[::3] = 2**40 + 7
    target[1::3] = 7  # equals ignore_index in its low 32 bits: must count
    check(logits, target, ignore_index=2**40 + 7)
    # An int32 target can never equal it: every row counts.
    check_narrow_target(logits, target.clamp(max=2999).to(torch.int32), ignore_index=2**40 + 7)


def test_storage_offset_is_overwritten_in_place():
    gen = torch.Generator(device="cuda").manual_seed(9)
    for dtype, offset in ((torch.bfloat16, 1), (torch.float16, 5), (torch.float32, 1)):
        storage = (torch.randn(100 * 3001 + offset, generator=gen, device="cuda") * 2).to(dtype)
        logits = storage[offset:].view(100, 3001)
        target = torch.randint(0, 3001, (100,), generator=gen, device="cuda")
        want_loss, want_grad, num_counted = reference(logits, target, IGNORE)
        loss, grad = cross_entropy_fwd_bwd(logits, target, IGNORE)
        assert grad.data_ptr() == logits.data_ptr()
        torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
        rtol = 1e-5 if dtype == torch.float32 else torch.finfo(dtype).eps
        atol = 1e-6 / num_counted + torch.finfo(dtype).smallest_normal * torch.finfo(dtype).eps
        torch.testing.assert_close(grad.float(), want_grad, rtol=rtol, atol=atol)


def test_non_contiguous_logits_leave_the_source_alone():
    gen = torch.Generator(device="cuda").manual_seed(10)
    wide = torch.randn(64, 151936 * 2, generator=gen, device="cuda").to(torch.bfloat16)
    logits = wide[:, ::2]
    before = wide.clone()
    target = torch.randint(0, 151936, (64,), generator=gen, device="cuda")
    want_loss, want_grad, num_counted = reference(logits, target, IGNORE)
    loss, grad = cross_entropy_fwd_bwd(logits, target, IGNORE)
    assert grad.is_contiguous() and torch.equal(wide, before)
    torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad.float(), want_grad, rtol=2**-8, atol=1e-6 / num_counted)


def test_empty_and_tiny_batches():
    loss, grad = cross_entropy_fwd_bwd(torch.empty(0, 151936, device="cuda", dtype=torch.bfloat16), torch.empty(0, dtype=torch.long, device="cuda"), IGNORE)
    assert loss.dtype == torch.float32 and loss.dim() == 0 and loss.item() == 0 and grad.shape == (0, 151936)
    gen = torch.Generator(device="cuda").manual_seed(11)
    for tokens in (1, 3):  # fewer rows than clusters in flight
        logits = torch.randn(tokens, 248320, generator=gen, device="cuda").to(torch.bfloat16)
        check(logits, torch.randint(0, 248320, (tokens,), generator=gen, device="cuda"))


def test_more_rows_than_grid_y_allows():
    gen = torch.Generator(device="cuda").manual_seed(12)
    logits = torch.randn(70000, 64, generator=gen, device="cuda").to(torch.bfloat16)
    target = torch.randint(0, 64, (70000,), generator=gen, device="cuda")
    target[::13] = IGNORE
    check(logits, target)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fp32_fp16_at_contract_vocabularies(dtype):
    gen = torch.Generator(device="cuda").manual_seed(13)
    for vocab in (102400, 151936, 201088, 248320):
        logits = (torch.randn(96, vocab, generator=gen, device="cuda") * 2).to(dtype)
        target = torch.randint(0, vocab, (96,), generator=gen, device="cuda")
        target[::7] = IGNORE
        check(logits, target)


def test_all_ignored_large():
    logits = torch.randn(4096, 151936, device="cuda").to(torch.bfloat16)
    loss, grad = cross_entropy_fwd_bwd(logits, torch.full((4096,), IGNORE, device="cuda"), IGNORE)
    assert loss.item() == 0.0 and torch.all(grad == 0)


def test_64bit_offsets():
    # 8192 x 248320 bf16 is 4.07 GB: element offsets of the last rows pass 2^31.
    gen = torch.Generator(device="cuda").manual_seed(14)
    logits = (torch.randn(8192, 248320, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    target = torch.randint(0, 248320, (8192,), generator=gen, device="cuda")
    target[::7] = IGNORE
    last = logits[-64:].clone()
    num_counted = int((target != IGNORE).sum())
    want_loss = sum(F.cross_entropy(logits[i : i + 1024].float(), target[i : i + 1024], ignore_index=IGNORE, reduction="sum") for i in range(0, 8192, 1024)) / num_counted
    loss, grad = cross_entropy_fwd_bwd(logits, target, IGNORE)
    torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
    want_grad = torch.softmax(last.float(), dim=-1) / num_counted
    rows = torch.nonzero(target[-64:] != IGNORE).squeeze(1)
    want_grad[rows, target[-64:][rows]] -= 1.0 / num_counted
    want_grad[target[-64:] == IGNORE] = 0
    torch.testing.assert_close(grad[-64:].float(), want_grad, rtol=torch.finfo(torch.bfloat16).eps, atol=1e-6 / num_counted)


def test_rows_too_large_for_shared_memory():
    gen = torch.Generator(device="cuda").manual_seed(15)
    logits = (torch.randn(16, 2_000_000, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    target = torch.randint(0, 2_000_000, (16,), generator=gen, device="cuda")
    target[3] = IGNORE
    logits[torch.arange(8), target[:8]] = 40  # dominant targets: a stale x[target] would show
    check(logits, target)


@pytest.mark.parametrize("vocab", [102400, 151936, 201088, 248320, 2_000_000])
def test_deterministic(vocab):
    gen = torch.Generator(device="cuda").manual_seed(16)
    tokens = 16 if vocab > 248320 else 512
    logits = (torch.randn(tokens, vocab, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    target = torch.randint(0, vocab, (tokens,), generator=gen, device="cuda")
    target[::5] = IGNORE
    first_loss, first_grad = cross_entropy_fwd_bwd(logits.clone(), target, IGNORE)
    for _ in range(9):
        loss, grad = cross_entropy_fwd_bwd(logits.clone(), target, IGNORE)
        assert torch.equal(loss.view(torch.int32), first_loss.view(torch.int32))
        assert torch.equal(grad.view(torch.int16), first_grad.view(torch.int16))


def test_extra_memory_is_linear_in_rows():
    logits = torch.randn(2048, 151936, device="cuda").to(torch.bfloat16)
    target = torch.randint(0, 151936, (2048,), device="cuda")
    cross_entropy_fwd_bwd(logits, target, IGNORE)  # compile outside the measurement
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    live = torch.cuda.memory_allocated()
    cross_entropy_fwd_bwd(logits, target, IGNORE)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - live <= 64 * 2048 + 2**20


def test_does_not_load_the_triton_kernel():
    import subprocess
    import sys

    code = "import sys, torch\nfrom pithtrain.kernels import cross_entropy_fwd_bwd\nfrom pithtrain.kernels.arch import ARCH_MAJOR\nbefore = 'triton' in sys.modules\ncross_entropy_fwd_bwd(torch.randn(8, 1000, device='cuda').to(torch.bfloat16), torch.zeros(8, dtype=torch.long, device='cuda'), -100)\nif ARCH_MAJOR == 9:\n    assert 'pithtrain.kernels.cross_entropy_sm100' not in sys.modules\n    assert before or 'triton' not in sys.modules\n"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_non_finite_logits(dtype):
    gen = torch.Generator(device="cuda").manual_seed(17)
    logits = (torch.randn(8, 5003, generator=gen, device="cuda") * 2).to(dtype)
    target = torch.randint(0, 5003, (8,), generator=gen, device="cuda")
    # -inf entries in rows with a finite max are exact: probability and gradient 0.
    finite_max = logits.clone()
    finite_max[:, ::3] = float("-inf")
    target_finite = target.clone()
    target_finite[target_finite % 3 == 0] += 1
    check(finite_max, target_finite)
    # A NaN, a +inf or a row of -inf makes that row's loss and gradient NaN, as in PyTorch.
    for row_fill in ("nan", "inf", "all_neg_inf"):
        bad = logits.clone()
        if row_fill == "all_neg_inf":
            bad[2] = float("-inf")
        else:
            bad[2, 17] = float(row_fill)
        want_loss, want_grad, _ = reference(bad, target, IGNORE)
        loss, grad = cross_entropy_fwd_bwd(bad.clone(), target, IGNORE)
        assert torch.isnan(loss) and torch.isnan(want_loss)
        assert torch.isnan(grad[2].float()).all() and torch.isnan(want_grad[2]).all()
        # Ignored rows are never read: non-finite logits there leave a zero gradient and a finite mean.
        ignored = target.clone()
        ignored[2] = IGNORE
        want_loss, want_grad, num_counted = reference(torch.where(ignored[:, None] == IGNORE, 0, bad), ignored, IGNORE)
        loss, grad = cross_entropy_fwd_bwd(bad.clone(), ignored, IGNORE)
        assert torch.all(grad[2] == 0)
        torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)


def test_side_streams_and_temporaries():
    # Producer, the three kernels and the consumer on a side stream ordered only by events; then
    # alternating streams with the caching allocator reusing the row-loss / count temporaries.
    gen = torch.Generator(device="cuda").manual_seed(18)
    base = (torch.randn(256, 151936, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    target = torch.randint(0, 151936, (256,), generator=gen, device="cuda")
    target[::9] = IGNORE
    want_loss, want_grad = cross_entropy_fwd_bwd(base.clone(), target, IGNORE)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    produced, consumed = torch.cuda.Event(), torch.cuda.Event()
    with torch.cuda.stream(streams[0]):
        logits = base * 1  # produced on the side stream
        produced.record()
    streams[1].wait_event(produced)
    with torch.cuda.stream(streams[1]):
        loss, grad = cross_entropy_fwd_bwd(logits, target, IGNORE)
        loss_copy, grad_copy = loss.clone(), grad.clone()
        consumed.record()
    consumed.synchronize()
    assert torch.equal(loss_copy, want_loss) and torch.equal(grad_copy, want_grad)
    for i in range(6):
        stream = streams[i % 2]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            inp = base.clone()
            loss, grad = cross_entropy_fwd_bwd(inp, target, IGNORE)
            ok = torch.equal(loss, want_loss) & torch.equal(grad, want_grad)
        torch.cuda.current_stream().wait_stream(stream)
        assert bool(ok)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_target_cancellation(dtype):
    # p_y -> 1: (p_y - 1) / n cancels to (almost) 0 and the target entry must stay <= 0 and exact.
    gen = torch.Generator(device="cuda").manual_seed(19)
    logits = torch.randn(64, 32000, generator=gen, device="cuda").to(dtype)
    target = torch.randint(0, 32000, (64,), generator=gen, device="cuda")
    margins = torch.tensor([4.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 60.0], device="cuda").repeat(8)
    logits[torch.arange(64), target] = (logits.float().amax(dim=1) + margins).to(dtype)
    loss, grad = check(logits, target)
    assert loss.item() >= 0


def test_zero_classes_rejected():
    with pytest.raises(ValueError):
        cross_entropy_fwd_bwd(torch.empty(4, 0, device="cuda", dtype=torch.bfloat16), torch.zeros(4, dtype=torch.long, device="cuda"), IGNORE)


@pytest.mark.parametrize("vocab", [3, 7, 11, 13, 97, 1000])
def test_correctly_rounded_scale(vocab):
    # Equal logits make every exp exactly 1 and the row sum exactly vocab, so the fp32 gradient is
    # (1 * rcp(vocab)) * rcp(n), minus rcp(n) at the target: each step rounded once. Compared bit for
    # bit with torch's correctly rounded reciprocals over n = 1..256, where an approximate reciprocal
    # (rcp.approx, which equals Triton's div.full.f32) differs for some n at every one of these vocabs.
    rcp = lambda value: torch.reciprocal(torch.tensor(float(value), device="cuda"))
    for num_counted in range(1, 257):
        logits = torch.full((num_counted, vocab), 0.5, dtype=torch.float32, device="cuda")
        target = torch.zeros(num_counted, dtype=torch.long, device="cuda")
        _, grad = cross_entropy_fwd_bwd(logits, target, IGNORE)
        want = torch.empty_like(grad)
        want[:, 1:] = rcp(vocab) * rcp(num_counted)
        want[:, 0] = rcp(vocab) * rcp(num_counted) - rcp(num_counted)
        assert torch.equal(grad, want), f"vocab {vocab}, n {num_counted}: gradient not correctly rounded"
