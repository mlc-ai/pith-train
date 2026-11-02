"""
Verifies log_prob_fwd and log_prob_bwd, the fused per-token log-probability and entropy of a
tempered softmax and its in-place backward, against PyTorch.

Contract: for logits x (N, V) and inv_temperature = 1 / T, p = softmax(x / T) per row; logp is
log p[target] (0 for a target outside [0, V)), entropy is -sum p log p with 0 log 0 = 0, both fp32.
The backward overwrites x with the gradient of g_logp * logp + g_entropy * entropy, computed in fp32
and rounded once to x's dtype (an invalid target drops its g_logp). row_max and log_sum are opaque
statistics passed from the forward to the backward; only their shape and dtype are checked.
"""

import pytest
import torch

from pithtrain.kernels import log_prob_bwd, log_prob_fwd

# Gradient tolerances: both sides round an fp32 gradient once to the logits' dtype, so the result
# may differ by one ulp.
GRAD_TOL = {torch.float32: (1e-4, 1e-6), torch.bfloat16: (1e-2, 1e-6), torch.float16: (2e-3, 1e-6)}


def reference(logits, target, temperature, grad_logp, grad_entropy):
    """
    logp, entropy and the logits' gradient, through log_softmax in fp32 (fp64 for fp64 logits).
    """
    dtype = torch.promote_types(logits.dtype, torch.float32)
    x = logits.detach().to(dtype).requires_grad_()
    logp_all = torch.log_softmax(x / temperature, dim=-1)
    entropy = -(logp_all.exp() * torch.where(logp_all > float("-inf"), logp_all, 0.0)).sum(dim=-1)
    valid = (target >= 0) & (target < logits.shape[-1])
    logp = logp_all.gather(-1, torch.where(valid, target, 0).long().unsqueeze(-1)).squeeze(-1)
    logp = torch.where(valid, logp, 0.0)
    torch.autograd.backward((logp, entropy), (grad_logp.to(dtype), grad_entropy.to(dtype)))
    return logp.detach(), entropy.detach(), x.grad.to(logits.dtype)


def run(inp, target, temperature, grad_logp, grad_entropy):
    """
    log_prob_fwd then log_prob_bwd on inp (overwritten with the gradient); returns logp, entropy.
    """
    N = inp.shape[0]
    before = inp.clone()
    logp, entropy, row_max, log_sum = log_prob_fwd(inp, target, 1.0 / temperature)
    assert torch.equal(inp, before)  # the forward leaves the logits intact
    for t in (logp, entropy, row_max, log_sum):
        assert t.shape == (N,) and t.dtype == torch.float32
    log_prob_bwd(inp, target, row_max, log_sum, entropy, grad_logp, grad_entropy, 1.0 / temperature)
    return logp, entropy


def check(logits, target, temperature, grad_logp, grad_entropy, inp=None):
    expected_logp, expected_entropy, expected_grad = reference(logits, target, temperature, grad_logp, grad_entropy)  # fmt: skip
    inp = logits.clone() if inp is None else inp
    logp, entropy = run(inp, target, temperature, grad_logp, grad_entropy)
    torch.testing.assert_close(logp, expected_logp.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, expected_entropy.float(), rtol=1e-5, atol=1e-5)
    rtol, atol = GRAD_TOL[logits.dtype]
    torch.testing.assert_close(inp, expected_grad, rtol=rtol, atol=atol)
    return inp


def make(N, V, dtype, seed=0, scale=3.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn((N, V), generator=gen, device="cuda") * scale).to(dtype)
    target = torch.randint(V, (N,), generator=gen, device="cuda")
    grad_logp, grad_entropy = torch.randn((2, N), generator=gen, device="cuda")
    return logits, target, grad_logp, grad_entropy


@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "N,V",
    [
        (3, 151936),  # qwen3-30b-a3b
        (3, 248320),  # qwen3.5-35b-a3b
        (3, 102400),  # deepseek-v2-lite
        (3, 201088),  # gpt-oss
        (67, 8192),
        (5, 1000),  # V not a multiple of 128
        (4, 1),
        (2, 7),  # V below one vector
    ],
)
def test_matches_log_softmax(N, V, dtype, temperature):
    logits, target, grad_logp, grad_entropy = make(N, V, dtype)
    # the first and last column, and both sides of power-of-two tile boundaries
    for row, index in enumerate((0, V - 1, 32767, 32768, 16384)):
        if row < N and index < V:
            target[row] = index
    check(logits, target, temperature, grad_logp, grad_entropy)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_masked_logits(dtype):
    """
    -inf logits add nothing to the entropy and get no gradient; row 0 keeps only its last column.
    """
    logits, target, grad_logp, grad_entropy = make(9, 151936, dtype, seed=1)
    logits[:, ::2] = float("-inf")
    logits[0, :-1] = float("-inf")
    target[0] = 151935
    target[1:] = (
        2 * torch.arange(1, 9, device="cuda") + 1
    )  # every other row's target on an unmasked odd column
    grad = check(logits, target, 0.6, grad_logp, grad_entropy)
    assert not grad[:, ::2].any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_extreme_logits(dtype):
    """
    Spike, uniform and offset rows against an fp64 reference.
    """
    V = 152064
    logits, target, grad_logp, grad_entropy = make(6, V, torch.float32, seed=2, scale=1.0)
    logits[0] = 7.0  # uniform: entropy log(V)
    logits[1, target[1]] = 80.0  # the target takes all the mass
    logits[2, 0] = 80.0  # another token takes all the mass
    logits[3] *= 1000.0  # a few tokens take all the mass
    logits[4] = logits[4] * 30.0 - 1e4  # large offset
    logits[5] *= 1e-3  # nearly uniform
    logits = logits.to(dtype)
    # T = 0.5 keeps x / T exact in fp32.
    for temperature in (1.0, 0.5):
        truth = reference(logits.double(), target, temperature, grad_logp.double(), grad_entropy.double())  # fmt: skip
        inp = logits.clone()
        logp, entropy = run(inp, target, temperature, grad_logp, grad_entropy)
        torch.testing.assert_close(logp.double(), truth[0], rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(entropy.double(), truth[1], rtol=1e-6, atol=1e-5)
        rtol, atol = GRAD_TOL[dtype]
        torch.testing.assert_close(inp, truth[2].to(dtype), rtol=rtol, atol=atol)


def test_out_of_range_targets():
    """
    Invalid targets give logp 0 and drop g_logp; int32 targets work as int64 ones do.
    """
    logits, _, grad_logp, grad_entropy = make(6, 8192, torch.bfloat16, seed=3)
    target = torch.tensor([-100, 0, -1, 8191, 8192, 2**40], device="cuda")
    check(logits, target, 0.6, grad_logp, grad_entropy)
    target32 = torch.tensor([-100, 0, -1, 8191, 8192, 2**30], device="cuda", dtype=torch.int32)
    check(logits, target32, 0.6, grad_logp, grad_entropy)


def test_other_float_dtypes():
    """
    fp16 and fp64 logits are read as fp32; the gradient is rounded once to their dtype.
    """
    logits, target, grad_logp, grad_entropy = make(5, 4099, torch.float32, seed=4)
    check(logits.half(), target, 1.0, grad_logp, grad_entropy)
    logits64 = logits.double()
    expected = reference(logits, target, 1.0, grad_logp, grad_entropy)  # fp32 arithmetic
    inp = logits64.clone()
    logp, entropy = run(inp, target, 1.0, grad_logp, grad_entropy)
    torch.testing.assert_close(logp, expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, expected[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(inp, expected[2].double(), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("pad", [0, 8, 3])
def test_row_stride_and_gradient_strides(pad):
    """
    Logits with a row stride above V (aligned or not) are read and overwritten in place, padding
    untouched; stride-0 and strided upstream gradients.
    """
    N, V = 33, 8192
    logits, target, _, _ = make(N, V, torch.bfloat16, seed=5)
    storage = torch.full((N, V + pad), 7.0, device="cuda", dtype=torch.bfloat16)
    inp = storage[:, :V]
    inp.copy_(logits)
    grad_logp = torch.full((1,), 1.0, device="cuda").expand(N)  # stride 0, as from logp.sum()
    grad_entropy = torch.randn(2 * N, device="cuda")[::2]  # stride 2
    check(logits, target, 0.6, grad_logp, grad_entropy, inp=inp)
    assert (storage[:, V:] == 7.0).all()


def test_masked_rows_get_zero_gradient():
    logits, target, grad_logp, grad_entropy = make(12, 151936, torch.bfloat16, seed=6)
    grad_logp[::3] = grad_entropy[::3] = 0.0
    grad = check(logits, target, 1.0, grad_logp, grad_entropy)
    assert not grad[::3].any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_backward_is_deterministic(dtype):
    logits, target, grad_logp, grad_entropy = make(64, 152064, dtype, seed=7)
    first = logits.clone()
    run(first, target, 1.0, grad_logp, grad_entropy)
    for _ in range(4):
        again = logits.clone()
        run(again, target, 1.0, grad_logp, grad_entropy)
        assert torch.equal(again, first)


def test_empty():
    inp = torch.empty((0, 1000), device="cuda", dtype=torch.bfloat16)
    target = torch.empty((0,), device="cuda", dtype=torch.int64)
    grads = torch.empty((2, 0), device="cuda")
    logp, entropy = run(inp, target, 1.0, grads[0], grads[1])
    assert logp.shape == entropy.shape == (0,)
