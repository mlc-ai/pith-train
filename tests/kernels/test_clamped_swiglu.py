import pytest
import torch

from pithtrain.kernels import clamped_swiglu_bwd, clamped_swiglu_fwd

ALPHA = 1.702
LIMIT = 7.0
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
ALL_DTYPES = DTYPES + [torch.float64]  # float64 is computed in fp32 like float32

# (M, 2N): tiny, odd N, sizes off the 1024-element block, and gpt-oss-20b's intermediate 2880.
SHAPES = [(1, 2), (128, 768), (1023, 778), (37, 2050), (4096, 2 * 2880), (8192, 4096)]
# Tails: N = 1 with M > 1, M * N below and off a 16-byte vector of outputs, odd M * N.
TAIL_SHAPES = [(5, 2), (3, 10), (1, 6), (7, 18), (37, 2 * 1025), (129, 2 * 131)]


def ref_fwd(gate_up, alpha, limit):
    gate, up = gate_up.float()[:, ::2], gate_up.float()[:, 1::2]
    gc = gate.clamp(max=limit)
    uc = up.clamp(min=-limit, max=limit)
    return (uc + 1.0) * (gc * torch.sigmoid(alpha * gc))


def ref_bwd(grad_out, gate_up, alpha, limit):
    grad_out_f32 = grad_out.float()
    gate, up = gate_up.float()[:, ::2], gate_up.float()[:, 1::2]
    gc = gate.clamp(max=limit)
    uc = up.clamp(min=-limit, max=limit)
    sig = torch.sigmoid(alpha * gc)
    grad_up = grad_out_f32 * (gc * sig)
    grad_gate = grad_out_f32 * (uc + 1.0) * sig * (1.0 + alpha * gc * (1.0 - sig))
    grad_gate = torch.where(gate <= limit, grad_gate, 0.0)
    grad_up = torch.where((up >= -limit) & (up <= limit), grad_up, 0.0)
    out = torch.empty(gate_up.shape, device=gate_up.device, dtype=torch.float32)
    out[:, ::2], out[:, 1::2] = grad_gate, grad_up
    return out


def assert_rounded(got, want32):
    scale = want32.abs().max().item() if want32.numel() else 1.0
    if got.dtype in (torch.float32, torch.float64):
        rtol, atol = 1e-5, 1e-6 * scale
    else:
        rtol, atol = torch.finfo(got.dtype).eps, 1e-6 * scale
    torch.testing.assert_close(got.float(), want32, rtol=rtol, atol=atol)


def run(gate_up, grad_out, alpha=ALPHA, limit=LIMIT):
    M, two_n = gate_up.shape
    out = clamped_swiglu_fwd(gate_up, alpha, limit)
    assert out.shape == (M, two_n // 2) and out.dtype == gate_up.dtype
    assert_rounded(out, ref_fwd(gate_up, alpha, limit))
    grad = clamped_swiglu_bwd(grad_out, gate_up, alpha, limit)
    assert grad.shape == gate_up.shape and grad.dtype == gate_up.dtype
    assert_rounded(grad, ref_bwd(grad_out, gate_up, alpha, limit))
    return out, grad


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_backward(shape, dtype):
    gen = torch.Generator(device="cuda").manual_seed(0)
    M, two_n = shape
    # A mix of in-range and saturating values on both signs of up and the top of gate.
    gate_up = (torch.randn(M, two_n, generator=gen, device="cuda") * 4).to(dtype)
    grad_out = torch.randn(M, two_n // 2, generator=gen, device="cuda").to(dtype)
    run(gate_up, grad_out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_matches_autograd(dtype):
    gen = torch.Generator(device="cuda").manual_seed(1)
    gate_up = (torch.randn(256, 2 * 2880, generator=gen, device="cuda") * 4).to(dtype)
    grad_out = torch.randn(256, 2880, generator=gen, device="cuda").to(dtype)
    x = gate_up.double().requires_grad_()
    gate, up = x[:, ::2], x[:, 1::2]
    gc, uc = gate.clamp(max=LIMIT), up.clamp(min=-LIMIT, max=LIMIT)
    ((uc + 1) * gc * torch.sigmoid(ALPHA * gc)).backward(grad_out.double())
    grad = clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT)
    scale = x.grad.abs().max().item()
    eps = torch.finfo(dtype).eps if dtype != torch.float32 else 1e-5
    torch.testing.assert_close(grad.double(), x.grad, rtol=eps, atol=1e-5 * scale)


def test_saturated_gradient_is_zero():
    M, N = 8, 16
    gate = torch.linspace(-LIMIT - 4.0, LIMIT + 4.0, M * N, device="cuda").view(M, N)
    up = torch.linspace(LIMIT + 3.0, LIMIT + 5.0, M * N, device="cuda").view(M, N)
    up[::2] = -up[::2]
    for dtype in DTYPES:
        gate_up = torch.empty(M, 2 * N, device="cuda", dtype=dtype)
        gate_up[:, ::2], gate_up[:, 1::2] = gate.to(dtype), up.to(dtype)
        grad_out = torch.randn(M, N, device="cuda").to(dtype)
        _, grad = run(gate_up, grad_out)
        gate_f32 = gate_up[:, ::2].float()
        assert torch.all(grad[:, ::2][gate_f32 > LIMIT] == 0)
        assert torch.all(grad[:, 1::2] == 0)  # every |up| > LIMIT
        clamped = gate_up.clone()
        clamped[:, ::2] = gate_up[:, ::2].clamp(max=LIMIT)
        clamped[:, 1::2] = gate_up[:, 1::2].clamp(min=-LIMIT, max=LIMIT)
        assert torch.equal(
            clamped_swiglu_fwd(gate_up, ALPHA, LIMIT), clamped_swiglu_fwd(clamped, ALPHA, LIMIT)
        )


def test_limit_boundary_is_active():
    gate_up = torch.tensor([[LIMIT, LIMIT, LIMIT, -LIMIT]], device="cuda", dtype=torch.float32)
    grad = clamped_swiglu_bwd(torch.ones(1, 2, device="cuda"), gate_up, ALPHA, LIMIT)
    assert torch.all(grad != 0)
    assert_rounded(grad, ref_bwd(torch.ones(1, 2, device="cuda"), gate_up, ALPHA, LIMIT))


@pytest.mark.parametrize("alpha,limit", [(1.0, 1.0), (1.702, 7.0), (2.5, 100.0)])
def test_other_alpha_limit(alpha, limit):
    gen = torch.Generator(device="cuda").manual_seed(2)
    gate_up = (torch.randn(333, 2 * 1000, generator=gen, device="cuda") * 5).to(torch.bfloat16)
    grad_out = torch.randn(333, 1000, generator=gen, device="cuda").to(torch.bfloat16)
    run(gate_up, grad_out, alpha, limit)


def test_large_negative_and_zero():
    gen = torch.Generator(device="cuda").manual_seed(3)
    gate_up = (torch.randn(1024, 2 * 2880, generator=gen, device="cuda") * 1000).to(torch.bfloat16)
    gate_up[:16] = 0
    gate_up[16:32, ::2] = -1e30
    grad_out = (torch.randn(1024, 2880, generator=gen, device="cuda") * 100).to(torch.bfloat16)
    out, grad = run(gate_up, grad_out)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert torch.all(out[:16] == 0)  # gc = 0


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_empty(dtype):
    gate_up = torch.empty(0, 2 * 2880, device="cuda", dtype=dtype)
    out = clamped_swiglu_fwd(gate_up, ALPHA, LIMIT)
    assert out.shape == (0, 2880) and out.dtype == dtype
    grad = clamped_swiglu_bwd(out, gate_up, ALPHA, LIMIT)
    assert grad.shape == (0, 2 * 2880) and grad.dtype == dtype


def test_deterministic():
    gen = torch.Generator(device="cuda").manual_seed(4)
    gate_up = (torch.randn(4096, 2 * 2880, generator=gen, device="cuda") * 4).to(torch.bfloat16)
    grad_out = torch.randn(4096, 2880, generator=gen, device="cuda").to(torch.bfloat16)
    assert torch.equal(
        clamped_swiglu_fwd(gate_up, ALPHA, LIMIT), clamped_swiglu_fwd(gate_up, ALPHA, LIMIT)
    )
    assert torch.equal(
        clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT),
        clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT),
    )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", TAIL_SHAPES)
def test_tails(shape, dtype):
    gen = torch.Generator(device="cuda").manual_seed(5)
    M, two_n = shape
    gate_up = (torch.randn(M, two_n, generator=gen, device="cuda") * 4).to(dtype)
    grad_out = torch.randn(M, two_n // 2, generator=gen, device="cuda").to(dtype)
    run(gate_up, grad_out)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", [(64, 200), (37, 2050), (3, 10)])
def test_misaligned_storage(shape, dtype):
    """
    Contiguous gate_up and grad_output whose data starts one element into their storage (not 16-byte
    aligned) give the bytes of the aligned call.
    """
    gen = torch.Generator(device="cuda").manual_seed(7)
    M, two_n = shape
    N = two_n // 2
    gate_up = (torch.randn(M, two_n, generator=gen, device="cuda") * 4).to(dtype)
    grad_out = torch.randn(M, N, generator=gen, device="cuda").to(dtype)
    gate_up_off = torch.empty(1 + M * two_n, device="cuda", dtype=dtype)[1:].view(M, two_n)
    grad_out_off = torch.empty(1 + M * N, device="cuda", dtype=dtype)[1:].view(M, N)
    gate_up_off.copy_(gate_up)
    grad_out_off.copy_(grad_out)
    assert gate_up_off.is_contiguous() and gate_up_off.data_ptr() % 16 != 0
    out, grad = run(gate_up_off, grad_out_off)
    assert torch.equal(
        out.view(-1).view(torch.uint8),
        clamped_swiglu_fwd(gate_up, ALPHA, LIMIT).view(-1).view(torch.uint8),
    )
    assert torch.equal(
        grad.view(-1).view(torch.uint8),
        clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT).view(-1).view(torch.uint8),
    )
    # Only grad_output misaligned.
    assert torch.equal(clamped_swiglu_bwd(grad_out_off, gate_up, ALPHA, LIMIT), grad)


def limit_neighbours(dtype):
    """
    LIMIT and its neighbours one ulp above and below, in dtype; float64 takes float32's neighbours,
    since it is rounded to float32 before the clamp.
    """
    work = torch.float32 if dtype == torch.float64 else dtype
    limit = torch.tensor(LIMIT, device="cuda", dtype=work)
    bits = limit.view({2: torch.int16, 4: torch.int32}[limit.element_size()])
    return limit.to(dtype), (bits + 1).view(work).to(dtype), (bits - 1).view(work).to(dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_limit_boundary_every_dtype(dtype):
    """
    At gate == limit and up == +/-limit both gradients pass, one ulp inside too; one ulp outside
    they are exactly zero.
    """
    limit, above, below = limit_neighbours(dtype)
    gates = torch.stack([limit, limit, above, below, limit, limit])
    ups = torch.stack([limit, -limit, limit, below, above, -above])
    gate_up = torch.stack([gates, ups], dim=-1).view(1, -1)
    grad_out = torch.ones(1, gates.numel(), device="cuda", dtype=dtype)
    grad = clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT)
    assert_rounded(grad, ref_bwd(grad_out, gate_up, ALPHA, LIMIT))
    grad_gate, grad_up = grad[0, ::2], grad[0, 1::2]
    assert torch.all(grad_gate[[0, 1, 3, 4, 5]] != 0) and grad_gate[2] == 0
    assert torch.all(grad_up[[0, 1, 2, 3]] != 0) and grad_up[4] == 0 and grad_up[5] == 0


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_limit_boundary_vectors(dtype):
    """
    The boundary cases repeated across full and partial vectors of outputs: zero gradient exactly
    where gate > limit or |up| > limit, nonzero at and inside the boundary.
    """
    limit, above, below = limit_neighbours(dtype)
    gates = torch.stack([limit, limit, above, below, limit, limit, -limit, below]).repeat(3)[:21]
    ups = torch.stack([limit, -limit, limit, below, above, -above, -limit, -below]).repeat(3)[:21]
    gate_up = torch.stack([gates, ups], dim=-1).view(1, -1)
    grad_out = torch.ones(1, gates.numel(), device="cuda", dtype=dtype)
    grad = clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT)
    want_gate = gates.double() <= LIMIT
    want_up = (ups.double() >= -LIMIT) & (ups.double() <= LIMIT)
    assert torch.equal(grad[0, ::2] != 0, want_gate)
    assert torch.equal(grad[0, 1::2] != 0, want_up)


def test_fresh_outputs():
    gen = torch.Generator(device="cuda").manual_seed(8)
    gate_up = (torch.randn(64, 2 * 96, generator=gen, device="cuda") * 4).to(torch.bfloat16)
    grad_out = torch.randn(64, 96, generator=gen, device="cuda").to(torch.bfloat16)
    snapshot = (gate_up.clone(), grad_out.clone())
    outs = [
        clamped_swiglu_fwd(gate_up, ALPHA, LIMIT),
        clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT),
    ]
    outs += [
        clamped_swiglu_fwd(gate_up, ALPHA, LIMIT),
        clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT),
    ]
    ptrs = {t.untyped_storage().data_ptr() for t in outs + [gate_up, grad_out]}
    assert len(ptrs) == 6
    assert torch.equal(gate_up, snapshot[0]) and torch.equal(grad_out, snapshot[1])
    assert outs[0].is_contiguous() and outs[1].is_contiguous()
    assert torch.equal(outs[0].view(torch.int16), outs[2].view(torch.int16))
    assert torch.equal(outs[1].view(torch.int16), outs[3].view(torch.int16))


def test_rejects_bad_inputs():
    gate_up = torch.randn(8, 32, device="cuda").to(torch.bfloat16)
    grad_out = torch.randn(8, 16, device="cuda").to(torch.bfloat16)
    with pytest.raises(AssertionError):
        clamped_swiglu_fwd(gate_up.t(), ALPHA, LIMIT)  # not contiguous
    with pytest.raises(AssertionError):
        clamped_swiglu_fwd(gate_up.view(-1), ALPHA, LIMIT)  # 1-D
    with pytest.raises(AssertionError):
        clamped_swiglu_fwd(gate_up[:, :31].contiguous(), ALPHA, LIMIT)  # odd width
    with pytest.raises(AssertionError):
        clamped_swiglu_bwd(torch.randn(16, 8, device="cuda").to(torch.bfloat16).t(), gate_up, ALPHA, LIMIT)  # fmt: skip
    with pytest.raises(AssertionError):
        clamped_swiglu_bwd(grad_out.float(), gate_up, ALPHA, LIMIT)  # dtype mismatch


def golden_inputs():
    """
    Every bf16 pattern (inf and NaN included) as gate, a fixed shuffle of them as up, and every bf16
    pattern, shuffled, as grad_output: gate_up (256, 512), grad_output (256, 256), built on the CPU.
    """
    gen = torch.Generator().manual_seed(1234)
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    gate, up = (
        bits.view(torch.bfloat16),
        bits[torch.randperm(65536, generator=gen)].view(torch.bfloat16),
    )
    gate_up = torch.stack([gate, up], dim=-1).view(256, 512)
    grad_out = bits[torch.randperm(65536, generator=gen)].view(torch.bfloat16).view(256, 256)
    return gate_up.cuda(), grad_out.cuda()


# SHA-256 of the bf16 output bytes of kda/sm90-base's clamped_swiglu kernels (b723719) on
# golden_inputs(): the bf16 results are pinned bit for bit to that recipe, NaN payloads included.
GOLDEN_SHA256 = {
    "fwd": "824f70e2d5130fc63bc03889cfcd04ceeac67d60e8bbf9c896f2b0c9951f3fb1",
    "bwd": "3b0c7d52f3d1c38578a8d474edd71470fb1a56a1d44bfba0898bb21835dd4a9f",
}


def test_bf16_golden_bytes():
    import hashlib

    gate_up, grad_out = golden_inputs()
    outs = {
        "fwd": clamped_swiglu_fwd(gate_up, ALPHA, LIMIT),
        "bwd": clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT),
    }
    for name, out in outs.items():
        digest = hashlib.sha256(
            out.contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        assert digest == GOLDEN_SHA256[name], name


def test_byte_offsets_past_2_31():
    """
    gate_up with more than 2^30 bf16 elements, so its byte offsets (and grad_gate_up's) pass 2^31: the last
    rows must match the plain-PyTorch reference, so a narrowing of the offsets to 32 bits fails here.
    """
    M, two_n = (1 << 30) // 2048 + 192, 2048  # 2^30 + 393216 elements, ~2.1 GB in bf16
    gate_up = torch.empty(M, two_n, device="cuda", dtype=torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(9)
    tail = 256
    gate_up[:-tail].zero_()
    gate_up[-tail:] = (torch.randn(tail, two_n, generator=gen, device="cuda") * 4).to(
        torch.bfloat16
    )
    grad_out = torch.zeros(M, two_n // 2, device="cuda", dtype=torch.bfloat16)
    grad_out[-tail:] = torch.randn(tail, two_n // 2, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    assert gate_up.numel() * gate_up.element_size() > 2**31
    out = clamped_swiglu_fwd(gate_up, ALPHA, LIMIT)
    assert_rounded(out[-tail:], ref_fwd(gate_up[-tail:], ALPHA, LIMIT))
    assert not bool(out[:-tail].any())
    del out
    grad = clamped_swiglu_bwd(grad_out, gate_up, ALPHA, LIMIT)
    assert_rounded(grad[-tail:], ref_bwd(grad_out[-tail:], gate_up[-tail:], ALPHA, LIMIT))
    assert not bool(grad[:-tail].any())
