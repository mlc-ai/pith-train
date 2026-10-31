"""
Contract: log_prob_fwd(inp, target, inv_temperature) returns the (N,) fp32 logp, entropy, row_max and
log_sum of softmax(inp * inv_temperature) and leaves the logits alone; log_prob_bwd overwrites the
logits in place with the gradient of grad_logp . logp + grad_entropy . entropy, computed in fp32 and
rounded once, bumps their version counter once and returns None. Targets outside [0, V) give logp
0 and drop grad_logp; -inf logits add 0 to the entropy and get gradient 0. Checked against an FP64
PyTorch reference written here.
"""

import pytest
import torch

from pithtrain.kernels import log_prob_bwd, log_prob_fwd

INT64_MAX, INT64_MIN = 2**63 - 1, -(2**63)
# The four models' vocabularies: deepseek-v2-lite, qwen3-30b-a3b, gpt-oss-20b, qwen3.5-35b-a3b.
MODEL_VOCABS = [102400, 151936, 201088, 248320]


def reference(logits, target, temperature, grad_logp, grad_entropy, rows_per_chunk=512):
    """
    FP64 logp, entropy, row_max and log_sum, and the FP64 gradient rounded once to the logits' dtype,
    a few rows at a time.
    """
    vocab = logits.shape[1]
    outputs = []
    for start in range(0, max(logits.shape[0], 1), rows_per_chunk):
        x = logits[start : start + rows_per_chunk].double() / temperature
        y = target[start : start + rows_per_chunk].long()
        g_logp = grad_logp[start : start + rows_per_chunk]
        g_ent = grad_entropy[start : start + rows_per_chunk].double()
        log_p = torch.log_softmax(x, dim=-1)
        p = log_p.exp()
        log_p_finite = torch.where(log_p > float("-inf"), log_p, 0.0)  # 0 log 0 = 0
        entropy = -(p * log_p_finite).sum(dim=-1)
        valid = (y >= 0) & (y < vocab)
        safe = torch.where(valid, y, 0)
        logp = torch.where(valid, log_p.gather(-1, safe.unsqueeze(-1)).squeeze(-1), 0.0)
        row_max = x.max(dim=-1).values
        log_sum = torch.logsumexp(x - row_max.unsqueeze(-1), dim=-1)
        g_logp = torch.where(valid, g_logp.double(), 0.0)
        grad = -p * (
            g_logp.unsqueeze(-1) + g_ent.unsqueeze(-1) * (log_p_finite + entropy.unsqueeze(-1))
        )
        rows = torch.nonzero(valid).squeeze(-1)
        grad[rows, safe[rows]] += g_logp[rows]
        outputs.append((logp, entropy, row_max, log_sum, (grad / temperature).to(logits.dtype)))
    return [torch.cat(parts) for parts in zip(*outputs)]


def run_kernels(logits, target, temperature, grad_logp, grad_entropy):
    inp = logits.clone()
    logp, entropy, row_max, log_sum = log_prob_fwd(inp, target, 1.0 / temperature)
    assert torch.equal(inp, logits)  # the forward leaves the logits unchanged
    for out in (logp, entropy, row_max, log_sum):
        assert out.shape == (logits.shape[0],) and out.dtype == torch.float32
    data_ptr, version = inp.data_ptr(), inp._version
    assert log_prob_bwd(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, 1.0 / temperature) is None  # fmt: skip
    assert inp.data_ptr() == data_ptr and inp._version == version + 1  # in place, bumped once
    return logp, entropy, row_max, log_sum, inp


def check(logits, target, temperature=1.0, grad_logp=None, grad_entropy=None, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_rows = logits.shape[0]
    if grad_logp is None:
        grad_logp, grad_entropy = torch.randn((2, num_rows), generator=generator, device="cuda")
    got = run_kernels(logits, target, temperature, grad_logp, grad_entropy)
    want = reference(logits, target, temperature, grad_logp, grad_entropy)
    logp, entropy, row_max, log_sum, grad = got
    torch.testing.assert_close(logp.double(), want[0], rtol=1e-5, atol=1e-5, equal_nan=True)
    torch.testing.assert_close(entropy.double(), want[1], rtol=1e-5, atol=1e-5, equal_nan=True)
    # row_max and log_sum are internal to the pair but defined for rows with a finite logit.
    finite = torch.isfinite(want[2])
    torch.testing.assert_close(row_max.double()[finite], want[2][finite], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(log_sum.double()[finite], want[3][finite], rtol=1e-5, atol=1e-5)
    # Both round an fp32 gradient once to the logits' dtype, so 16-bit types may differ by one ulp.
    rtol, atol = (1e-4, 1e-6) if logits.dtype == torch.float32 else (1e-2, 1e-6)
    torch.testing.assert_close(grad, want[4], rtol=rtol, atol=atol, equal_nan=True)
    return got


def random_problem(num_rows, vocab, dtype, invalid_every=7, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn((num_rows, vocab), generator=generator, device="cuda") * 3).to(dtype)
    target = torch.randint(vocab, (num_rows,), generator=generator, device="cuda")
    target[::invalid_every] = -100
    return logits, target


@pytest.mark.parametrize("num_rows", [1, 3, 4096, 8192])
@pytest.mark.parametrize("vocab", MODEL_VOCABS)
def test_model_vocabularies(num_rows, vocab):
    logits, target = random_problem(num_rows, vocab, torch.bfloat16)
    check(logits, target, temperature=1.0 if num_rows % 2 == 0 else 0.6)


@pytest.mark.parametrize("temperature", [1.0, 0.6, 0.5])
@pytest.mark.parametrize(
    "dtype, vocab",
    [
        (torch.bfloat16, 5003),
        (torch.float16, 5003),
        (torch.float32, 33),
        (torch.float32, 151936),
        (torch.float16, 151936),
        (torch.bfloat16, 1),
        (torch.float32, 1),
    ],
)
def test_dtypes_and_temperatures(dtype, vocab, temperature):
    logits, target = random_problem(67, vocab, dtype)
    check(logits, target, temperature)


@pytest.mark.parametrize(
    "inv_temperature", [-1.0, -0.0, 0.0, float("inf"), float("nan"), 1e-50, 1e100, 3.5e38]
)
def test_rejects_nonpositive_or_nonfinite_temperature(inv_temperature):
    """
    Where the entry points check the temperature (SM90), both raise ValueError before launching
    anything, also for values a Python float holds but fp32 rounds to 0 or inf; elsewhere the result
    is unspecified and the case skips.
    """
    logits, target = random_problem(3, 5003, torch.bfloat16)
    stats = torch.zeros((3, 3), device="cuda")
    try:
        log_prob_fwd(logits, target, inv_temperature)
    except ValueError:
        pass
    else:
        pytest.skip("this device leaves a nonpositive or non-finite inv_temperature unspecified")
    with pytest.raises(ValueError):
        log_prob_bwd(logits.clone(), target, *stats, *stats[:2], inv_temperature)


# The SM90 layout, restated so targets sit exactly on its CTA slice and chunk edges (elsewhere these
# are ordinary columns). Forward: above 64 rows 1 CTA per row up to 80 KB, else a cluster of 2; up to
# 64 rows the smallest of 1, 2, 4, 8 CTAs whose slices fit 40 KB; slices stream in chunks of about
# 16 KB, except up to 8 rows, where a slice lands whole. Backward: ceil(row bytes / 12 KB) slices, or
# 2 KB up to 16 rows. Slices and chunks are whole numbers of 128-byte lines.
TINY_FWD_ROWS, SMALL_FWD_ROWS, TINY_BWD_ROWS = 8, 64, 16


def line_units(num_units, num_parts):
    return (-(-num_units // num_parts) + 7) // 8 * 8


def fwd_layout(vocab, element_bytes, num_rows):
    """
    (cluster size, chunks per slice) of the forward.
    """
    row_bytes = vocab * element_bytes
    if num_rows <= SMALL_FWD_ROWS:
        ctas = next((ctas for ctas in (1, 2, 4, 8) if row_bytes <= ctas * 40 * 1024), 8)
    else:
        ctas = 1 if row_bytes <= 80 * 1024 else 2
    chunks = 1 if num_rows <= TINY_FWD_ROWS else max(1, -(-row_bytes // (ctas * 16 * 1024)))
    return ctas, chunks


def slice_counts(vocab, element_bytes, num_rows):
    bwd_bytes = 2 * 1024 if num_rows <= TINY_BWD_ROWS else 12 * 1024
    return fwd_layout(vocab, element_bytes, num_rows)[0], max(
        1, -(-vocab * element_bytes // bwd_bytes)
    )


def slice_boundaries(vocab, element_bytes, num_rows):
    """
    First column of every forward and backward CTA slice but the first, and of every forward chunk,
    for rows that start on a 16-byte boundary (V * element_bytes a multiple of 16).
    """
    num_units, unit_elements = vocab * element_bytes // 16, 16 // element_bytes
    fwd, bwd = slice_counts(vocab, element_bytes, num_rows)
    starts = set()
    for num_slices in (fwd, bwd):
        slice_units = line_units(num_units, num_slices)
        starts.update(k * slice_units for k in range(1, num_slices) if k * slice_units < num_units)
    fwd_units = line_units(num_units, fwd)
    chunk_units = line_units(fwd_units, fwd_layout(vocab, element_bytes, num_rows)[1])
    for rank in range(fwd):
        for k in range(1, -(-fwd_units // chunk_units)):
            if rank * fwd_units + k * chunk_units < min((rank + 1) * fwd_units, num_units):
                starts.add(rank * fwd_units + k * chunk_units)
    return sorted(unit * unit_elements for unit in starts)


def test_slice_boundaries_follow_the_layout():
    assert slice_counts(151936, 2, 4096) == (2, 25) and slice_counts(102400, 2, 4096) == (2, 17)
    assert slice_counts(151936, 4, 4096) == (2, 50) and slice_counts(30000, 2, 4096) == (1, 5)
    assert slice_counts(151936, 2, 3) == (8, 149) and slice_counts(102400, 2, 16) == (8, 100)
    assert fwd_layout(151936, 2, 8) == (8, 1) and fwd_layout(151936, 2, 64) == (8, 3)
    # bf16 151936: 1 cluster edge, 24 backward slice edges, 2 x 9 forward chunk edges, one of which
    # (unit 11400) is also a backward edge.
    assert len(slice_boundaries(151936, 2, 4096)) == 1 + 24 + 18 - 1


@pytest.mark.parametrize("dtype, vocab", [(torch.bfloat16, 5003), (torch.float32, 33)])
def test_targets_on_every_column(dtype, vocab):
    # One row per target column: every head, tail and slice edge of a ragged row, misaligned rows too.
    logits, _ = random_problem(vocab, vocab, dtype)
    check(logits, torch.arange(vocab, device="cuda"))


@pytest.mark.parametrize(
    "rows_per_call", [TINY_FWD_ROWS, TINY_BWD_ROWS, SMALL_FWD_ROWS, SMALL_FWD_ROWS + 1]
)
@pytest.mark.parametrize(
    "dtype, vocab",
    [(torch.bfloat16, v) for v in MODEL_VOCABS]
    + [(torch.float32, 151936), (torch.float16, 151936)],
)
def test_targets_on_every_slice_boundary(dtype, vocab, rows_per_call):
    # One row per target: both sides of every forward and backward slice edge (the last element of
    # one CTA's last unit, the first of the next CTA's), plus the row's first and last columns, in
    # calls of rows_per_call rows, so each row-count regime's layout is checked at its own edges.
    element_bytes = torch.finfo(dtype).bits // 8
    unit_elements = 16 // element_bytes
    columns = {0, 1, vocab - 2, vocab - 1}
    for start in slice_boundaries(vocab, element_bytes, rows_per_call):
        columns.update((start - unit_elements, start - 1, start, start + unit_elements - 1))
    columns = sorted(columns)
    columns += [0] * (-len(columns) % rows_per_call)  # whole calls only
    logits, _ = random_problem(len(columns), vocab, dtype)
    for start in range(0, len(columns), rows_per_call):
        rows = slice(start, start + rows_per_call)
        check(logits[rows], torch.tensor(columns[rows], device="cuda"))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_invalid_targets(dtype):
    vocab = 8192
    element_bytes = torch.finfo(dtype).bits // 8
    invalid = [-100, -1, vocab, 2**40, INT64_MAX, INT64_MIN, 2**62, -(2**62)]
    # Targets whose unguarded byte offset target * element_bytes wraps into or near the row.
    invalid += [2**63 // element_bytes + k for k in (0, 1, 5)]
    invalid += [
        2**64 // element_bytes + k for k in (0, 1, 5) if 2**64 // element_bytes + k <= INT64_MAX
    ]
    target = torch.tensor(invalid + [0, vocab - 1], device="cuda")
    logits, _ = random_problem(len(target), vocab, dtype)
    logp = check(logits, target, temperature=0.6)[0]
    assert not logp[: len(invalid)].any()


def test_all_targets_invalid():
    logits, _ = random_problem(64, 151936, torch.bfloat16)
    logp = check(logits, torch.full((64,), -100, device="cuda"))[0]
    assert not logp.any()


@pytest.mark.parametrize("target_dtype", [torch.int32, torch.int64, torch.int16])
@pytest.mark.parametrize("is_strided", [False, True])
def test_target_dtypes_and_strides(target_dtype, is_strided):
    vocab = 30000 if target_dtype == torch.int16 else 151936
    logits, target = random_problem(67, vocab, torch.bfloat16)
    target = target.to(target_dtype)
    if is_strided:
        target = torch.stack((target, torch.zeros_like(target)), dim=1)[:, 0]
        assert target.stride(0) == 2
    check(logits, target)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_negative_infinity(dtype):
    vocab, num_rows = 151936, 8
    logits, target = random_problem(num_rows, vocab, dtype)
    neg_inf = float("-inf")
    logits[0, ::2] = neg_inf  # every other logit, from column 0
    target[0] = 4  # a -inf target: logp -inf, gradient 0 there
    logits[1, :-1] = neg_inf  # only the last logit is finite
    logits[2, : vocab * 7 // 8] = neg_inf  # leading slices and cluster ranks entirely -inf
    logits[3, vocab // 2 :] = neg_inf  # trailing ones
    logits[4, 1::2] = neg_inf
    logits[5] = neg_inf  # every logit -inf: NaN entropy and gradient, as log_softmax gives
    target[5] = 7
    logits[6] = neg_inf
    target[6] = -100  # ... and logp 0 for an invalid target
    logp, entropy, _, _, grad = check(logits, target)
    assert torch.isfinite(entropy[:5]).all() and torch.isfinite(grad[:5]).all()
    is_masked = logits[:5] == neg_inf
    is_masked[torch.arange(5, device="cuda"), target[:5]] = (
        False  # d logp / dx at the target is 1 / T
    )
    assert (grad[:5][is_masked] == 0).all()
    assert entropy[5:7].isnan().all() and logp[5].isnan() and logp[6] == 0


def test_empty():
    logits = torch.empty((0, 151936), dtype=torch.bfloat16, device="cuda")
    target = torch.empty((0,), dtype=torch.int64, device="cuda")
    outputs = log_prob_fwd(logits, target, 1.0)
    assert all(out.shape == (0,) and out.dtype == torch.float32 for out in outputs)
    empty = torch.empty((0,), device="cuda")
    assert log_prob_bwd(logits, target, *outputs[2:], outputs[1], empty, empty, 1.0) is None


def test_stride_zero_upstream_gradients():
    logits, target = random_problem(33, 151936, torch.bfloat16)
    ones = torch.ones((), device="cuda")
    check(logits, target, 0.6, ones.expand(33), (-0.01 * ones).expand(33))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_deterministic(dtype):
    logits, target = random_problem(256, 151936, dtype)
    generator = torch.Generator(device="cuda").manual_seed(1)
    grads = torch.randn((2, 256), generator=generator, device="cuda")
    first = run_kernels(logits, target, 0.6, *grads)
    for _ in range(10):
        again = run_kernels(logits, target, 0.6, *grads)
        assert all(torch.equal(a, b) for a, b in zip(again, first))


def test_custom_ops():
    """
    Where the kernels launch through custom ops (needed for torch.compile), they pass opcheck.
    """
    logits, target = random_problem(5, 1000, torch.bfloat16)
    log_prob_fwd(logits, target, 1.0)  # registers the ops where they exist
    namespace = torch.ops.pithtrain
    if not (hasattr(namespace, "log_prob_fwd") and hasattr(namespace, "log_prob_bwd")):
        pytest.skip("this device launches the kernels without custom ops")
    torch.library.opcheck(namespace.log_prob_fwd, (logits, target, 1.25))
    stats = namespace.log_prob_fwd(logits, target, 1.25)
    grads = torch.randn((2, 5), device="cuda")
    bwd_args = (logits.clone(), target, stats[2], stats[3], stats[1], grads[0], grads[1], 1.25)
    torch.library.opcheck(namespace.log_prob_bwd, bwd_args)


def test_under_torch_compile():
    """
    Compiled (fullgraph) forward + backward match eager bitwise and update the logits in place; the
    version counter advances once in eager.
    """

    def forward_backward(inp, target, grad_logp, grad_entropy):
        logp, entropy, row_max, log_sum = log_prob_fwd(inp, target, 1.25)
        log_prob_bwd(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, 1.25)
        return logp, entropy

    logits, target = random_problem(16, 8192, torch.bfloat16)
    grads = torch.randn((2, 16), device="cuda")
    compiled = torch.compile(forward_backward, fullgraph=True)
    results = []
    for fn in (compiled, forward_backward):
        inp = logits.clone()
        data_ptr, version = inp.data_ptr(), inp._version
        outputs = fn(inp, target, *grads)
        # Eager: the backward's own single bump. Compiled: that bump, where the graph runs the kernel
        # in place on inp, plus at most Inductor's input-mutation epilogue (a copy_ of inp onto
        # itself), never a second bump from the kernel.
        bumps = inp._version - version
        assert inp.data_ptr() == data_ptr and bumps in ((1, 2) if fn is compiled else (1,))
        results.append((*outputs, inp))
    assert all(torch.equal(a, b) for a, b in zip(*results))
