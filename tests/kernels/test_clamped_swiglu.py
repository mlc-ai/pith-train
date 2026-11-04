import pytest
import torch

from pithtrain.kernels import clamped_swiglu_bwd, clamped_swiglu_fwd

ALPHA = 1.702
LIMIT = 7.0
DTYPES = [torch.bfloat16, torch.float16, torch.float32]

# (M, 2N): tiny, odd N, sizes off the 1024-element block, and gpt-oss-20b's intermediate 2880.
SHAPES = [(1, 2), (128, 768), (1023, 778), (37, 2050), (4096, 2 * 2880), (8192, 4096)]


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
    if got.dtype == torch.float32:
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


@pytest.mark.parametrize("dtype", DTYPES)
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


def test_empty():
    gate_up = torch.empty(0, 2 * 2880, device="cuda", dtype=torch.bfloat16)
    out = clamped_swiglu_fwd(gate_up, ALPHA, LIMIT)
    assert out.shape == (0, 2880)
    grad = clamped_swiglu_bwd(out, gate_up, ALPHA, LIMIT)
    assert grad.shape == (0, 2 * 2880)


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
