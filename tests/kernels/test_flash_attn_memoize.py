"""
Verifies memoized_launch against calling entry() directly: on the first call of a signature, on
the recording call and on every replayed call, with fresh inputs, it returns what entry() returns.
With FA4's entry, as the flash_attn_v4 operator calls it, the outputs equal FA4's own (forward bit
for bit, backward up to FA4's atomic accumulation order) and match plain attention in PyTorch.
"""

import importlib.metadata
import itertools

import pytest
import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

from pithtrain.kernels import memoized_launch


def fa4_memoized():
    """
    Whether the installed FA4 is the version the memoized path is validated on; other versions
    keep calling FA4's entry, which must still give the same outputs.
    """
    try:
        return importlib.metadata.version("flash-attn-4") == "4.0.0b28"
    except importlib.metadata.PackageNotFoundError:
        return False


SEEDS = itertools.count(1000)

# (batch, seq, q heads, kv heads, qk dim, v dim, causal, window, sink): the attention shapes of
# the four models, at a shorter sequence.
CASES = {
    "qwen3-gqa": (1, 1024, 32, 4, 128, 128, True, (None, None), False),
    "dsv2-mla": (1, 1024, 16, 16, 192, 128, True, (None, None), False),
    "gptoss-sliding-sink": (1, 1024, 64, 8, 64, 64, True, (127, 0), True),
    "gptoss-full-sink": (1, 1024, 64, 8, 64, 64, True, (None, None), True),
    "qwen35-full": (1, 1024, 16, 2, 256, 256, True, (None, None), False),
}


def make(case, seed):
    B, seq, hq, hkv, d, dv, causal, window, sink = CASES[case]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, seq, hq, d, generator=gen, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, seq, hkv, d, generator=gen, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, seq, hkv, dv, generator=gen, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, seq, hq, dv, generator=gen, device="cuda", dtype=torch.bfloat16)
    learnable_sink = torch.randn(hq, generator=gen, device="cuda", dtype=torch.float32) if sink else None  # fmt: skip
    return q, k, v, do, learnable_sink, d**-0.5, causal, window


class Counted:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self):
        self.calls += 1
        return self.fn()


def fwd_entry(q, k, v, learnable_sink, scale, causal, window):
    return lambda: tuple(
        _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=causal,
            window_size_left=window[0],
            window_size_right=window[1],
            learnable_sink=learnable_sink,
            return_lse=True,
        )[:2]
    )


def bwd_entry(q, k, v, o, lse, do, learnable_sink, scale, causal, window):
    return lambda: tuple(
        _flash_attn_bwd(
            q,
            k,
            v,
            o,
            do,
            lse,
            softmax_scale=scale,
            causal=causal,
            window_size_left=window[0],
            window_size_right=window[1],
            learnable_sink=learnable_sink,
        )
    )


def memo_fwd(q, k, v, learnable_sink, scale, causal, window):
    entry = Counted(fwd_entry(q, k, v, learnable_sink, scale, causal, window))
    inputs = dict(q=q, k=k, v=v, learnable_sink=learnable_sink)
    outs = memoized_launch("fwd", inputs, (scale, causal, window[0], window[1]), entry)
    return outs, entry.calls


def memo_bwd(q, k, v, o, lse, do, learnable_sink, scale, causal, window):
    entry = Counted(bwd_entry(q, k, v, o, lse, do, learnable_sink, scale, causal, window))
    inputs = dict(q=q, k=k, v=v, o=o, lse=lse, do=do, learnable_sink=learnable_sink)
    outs = memoized_launch("bwd", inputs, (scale, causal, window[0], window[1]), entry)
    return outs, entry.calls


def attention_reference(q, k, v, learnable_sink, scale, causal, window):
    """
    Plain fp32 attention on (B, S, H, D) tensors with GQA, a sliding window and a learnable sink
    (an extra per-head logit in the softmax denominator). Returns (o, lse), differentiable.
    """
    hq, hkv = q.shape[2], k.shape[2]
    qf, kf, vf = (t.transpose(1, 2) for t in (q, k, v))
    kf = kf.repeat_interleave(hq // hkv, dim=1)
    vf = vf.repeat_interleave(hq // hkv, dim=1)
    scores = (qf @ kf.transpose(-1, -2)) * scale
    seq = q.shape[1]
    i = torch.arange(seq, device=q.device)[:, None]
    j = torch.arange(seq, device=q.device)[None, :]
    allowed = torch.ones(seq, seq, dtype=torch.bool, device=q.device)
    if causal:
        allowed &= j <= i
    if window[0] is not None:
        allowed &= j >= i - window[0]
    if window[1] is not None:
        allowed &= j <= i + window[1]
    scores = scores.masked_fill(~allowed, float("-inf"))
    if learnable_sink is not None:
        sink = learnable_sink.view(1, -1, 1, 1).expand(*scores.shape[:3], 1)
        scores = torch.cat([scores, sink], dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)[..., :seq]
    return (p @ vf).transpose(1, 2), lse


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def test_generic_entry_every_call():
    for i in range(5):
        q = torch.randn(4, 333, device="cuda")
        k = torch.randn(4, 333, device="cuda")
        entry = Counted(lambda: (q * 2 + k, (q @ k.T).softmax(-1)))
        out = memoized_launch("test-generic", dict(q=q, k=k), (1.5, True), entry)
        want = (q * 2 + k, (q @ k.T).softmax(-1))
        assert entry.calls == 1  # no FA4 kernel in it: never replayed, entry runs every time
        assert len(out) == 2 and all(torch.equal(a, b) for a, b in zip(out, want))
    # An entry returning its own input (an alias, not a fresh allocation).
    for i in range(4):
        q = torch.randn(8, 8, device="cuda")
        out = memoized_launch("test-alias", dict(q=q), (), lambda: (q,))
        assert out[0] is q


@pytest.mark.parametrize("case", list(CASES))
def test_forward(case):
    calls = []
    for _ in range(5):
        q, k, v, do, learnable_sink, scale, causal, window = make(case, next(SEEDS))
        (o, lse), num_entry_calls = memo_fwd(q, k, v, learnable_sink, scale, causal, window)
        calls.append(num_entry_calls)
        ref_o, ref_lse = fwd_entry(q, k, v, learnable_sink, scale, causal, window)()
        assert o.shape == ref_o.shape and o.dtype == ref_o.dtype
        assert lse.shape == ref_lse.shape and lse.dtype == ref_lse.dtype
        assert torch.equal(o, ref_o) and torch.equal(lse, ref_lse)
    if fa4_memoized():
        assert calls[-2:] == [0, 0], f"no replay: entry calls per call {calls}"
    want_o, want_lse = attention_reference(
        q.float(), k.float(), v.float(), learnable_sink, scale, causal, window
    )
    assert rel(o, want_o) < 1e-2
    torch.testing.assert_close(lse, want_lse, atol=2e-3, rtol=1e-3)
    if learnable_sink is None:
        sdpa = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            is_causal=causal,
            scale=scale,
            enable_gqa=True,
        ).transpose(1, 2)
        if window == (None, None):
            assert rel(o, sdpa) < 1e-2


@pytest.mark.parametrize("case", list(CASES))
def test_backward(case):
    calls = []
    for _ in range(5):
        q, k, v, do, learnable_sink, scale, causal, window = make(case, next(SEEDS))
        o, lse = fwd_entry(q, k, v, learnable_sink, scale, causal, window)()
        grads, num_entry_calls = memo_bwd(q, k, v, o, lse, do, learnable_sink, scale, causal, window)  # fmt: skip
        calls.append(num_entry_calls)
        ref = bwd_entry(q, k, v, o, lse, do, learnable_sink, scale, causal, window)()
        assert len(grads) == len(ref) == (3 if learnable_sink is None else 4)
        # FA4 accumulates dQ (and dK/dV under GQA) with atomics, so its backward is not
        # bit-reproducible run to run: the gradients differ from FA4's entry only by
        # accumulation-order rounding, a few bf16 ulps, as two runs of the entry do.
        for x, r in zip(grads, ref):
            assert x.shape == r.shape and x.dtype == r.dtype
            ulp = torch.finfo(torch.bfloat16).eps * torch.maximum(x.float().abs(), r.float().abs())
            assert ((x.float() - r.float()).abs() <= 4 * ulp + 1e-6).all()
    if fa4_memoized():
        assert calls[-2:] == [0, 0], f"no replay: entry calls per call {calls}"
    leaves = [t.float().requires_grad_() for t in (q, k, v)]
    sink = learnable_sink.clone().requires_grad_() if learnable_sink is not None else None
    want_o, _ = attention_reference(*leaves, sink, scale, causal, window)
    want_o.backward(do.float())
    for x, leaf in zip(grads[:3], leaves):
        assert rel(x, leaf.grad) < 2e-2
    if sink is not None:
        assert rel(grads[3], sink.grad) < 2e-2


def test_back_to_back_calls_own_their_buffers():
    q, k, v, do, learnable_sink, scale, causal, window = make("qwen3-gqa", 0)
    o, lse = fwd_entry(q, k, v, learnable_sink, scale, causal, window)()
    for _ in range(3):
        memo_bwd(q, k, v, o, lse, do, learnable_sink, scale, causal, window)
        memo_fwd(q, k, v, learnable_sink, scale, causal, window)
    (first, _), (second, _) = (
        memo_bwd(q, k, v, o, lse, do, learnable_sink, scale, causal, window) for _ in range(2)
    )
    assert not {t.data_ptr() for t in first} & {t.data_ptr() for t in second}
    (f1, _), (f2, _) = (memo_fwd(q, k, v, learnable_sink, scale, causal, window) for _ in range(2))
    kept = [t.clone() for t in f1]
    q2, k2, v2, *_ = make("qwen3-gqa", 1)
    (f3, _) = memo_fwd(q2, k2, v2, learnable_sink, scale, causal, window)
    assert not {t.data_ptr() for t in f1} & {t.data_ptr() for t in f2 + f3}
    assert all(torch.equal(a, b) for a, b in zip(f1, kept))
    assert not torch.equal(f3[0], f1[0])


def test_unseen_signature_calls_entry():
    q, k, v, do, learnable_sink, scale, causal, window = make("qwen3-gqa", 2)
    q2, k2, v2 = (t[:, :640].contiguous() for t in (q, k, v))
    (o, lse), num_entry_calls = memo_fwd(q2, k2, v2, None, scale, causal, (None, None))
    assert num_entry_calls == 1
    assert torch.equal(o, fwd_entry(q2, k2, v2, None, scale, causal, (None, None))()[0])
    (o, lse), num_entry_calls = memo_fwd(q2, k2, v2, None, scale * 0.5, causal, (None, None))
    assert num_entry_calls == 1
    assert torch.equal(o, fwd_entry(q2, k2, v2, None, scale * 0.5, causal, (None, None))()[0])
