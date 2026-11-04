"""
The kernels compute in fp32 and round once to the input dtype. Their sigmoid may differ from torch's
by fp32 ulps, so outputs must lie within one output-dtype ulp of the fp32 reference.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import silu_mul_bwd, silu_mul_fwd

DTYPES = [torch.bfloat16, torch.float16, torch.float32]

# (rows, cols): tiny, not a multiple of the 1024-element block, and the expert / dense MLP
# intermediate sizes of deepseek-v2-lite (1408, dense 10944), qwen3-30b-a3b (768),
# qwen3.5-35b-a3b (512) and gpt-oss-20b (2880).
SHAPES = [
    (1, 1),
    (1023, 777),
    (128, 768),
    (4096, 1408),
    (512, 10944),
    (4096, 768),
    (4096, 512),
    (2048, 2880),
    (4096, 2048),
]


def ref_fwd(gate, up):
    g, u = gate.float(), up.float()
    return g * torch.sigmoid(g) * u


def ref_bwd(grad_out, gate, up):
    do, g, u = grad_out.float(), gate.float(), up.float()
    sig = torch.sigmoid(g)
    silu = g * sig
    return do * u * sig * (1.0 + g - silu), do * silu


def assert_rounded(got, want32, scale):
    """
    Check that got (in its dtype) is within one ulp of the fp32 reference, plus fp32 noise.
    """
    eps = torch.finfo(got.dtype).eps
    atol = 1e-5 * scale if got.dtype == torch.float32 else 1e-6 * scale + 1e-30
    rtol = 1e-5 if got.dtype == torch.float32 else eps
    torch.testing.assert_close(got.float(), want32, rtol=rtol, atol=atol)


def run(gate, up, grad_out):
    out = silu_mul_fwd(gate, up)
    assert out.shape == gate.shape and out.dtype == gate.dtype
    want = ref_fwd(gate, up)
    assert_rounded(out, want, want.abs().max().item() if want.numel() else 1.0)
    dg, du = silu_mul_bwd(grad_out, gate, up)
    assert dg.shape == gate.shape and du.shape == up.shape
    assert dg.dtype == gate.dtype and du.dtype == up.dtype
    want_g, want_u = ref_bwd(grad_out, gate, up)
    # The gate term (1 + g - silu(g)) cancels near g = -1.28, so absolute fp32 noise there
    # scales with the size of the factors, not of the result.
    scale_g = (grad_out.float() * up.float()).abs().max().item() if dg.numel() else 1.0
    assert_rounded(dg, want_g, max(scale_g, 1e-30))
    assert_rounded(du, want_u, want_u.abs().max().item() if du.numel() else 1.0)
    if gate.numel() and gate.dtype != torch.float16:
        g64 = gate.double().requires_grad_()
        u64 = up.double().requires_grad_()
        (F.silu(g64) * u64).backward(grad_out.double())
        torch.testing.assert_close(want_g.double(), g64.grad, rtol=1e-4, atol=1e-4 * scale_g)
        torch.testing.assert_close(want_u.double(), u64.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_backward(shape, dtype):
    gen = torch.Generator(device="cuda").manual_seed(0)
    gate = torch.randn(shape, generator=gen, device="cuda").mul(3).to(dtype)
    up = torch.randn(shape, generator=gen, device="cuda").to(dtype)
    grad_out = torch.randn(shape, generator=gen, device="cuda").to(dtype)
    run(gate, up, grad_out)


@pytest.mark.parametrize("dtype", DTYPES)
def test_three_dims(dtype):
    """
    Any contiguous shape works: the kernels are flat elementwise.
    """
    gen = torch.Generator(device="cuda").manual_seed(1)
    shape = (2, 3, 1000)
    gate, up, do = (torch.randn(shape, generator=gen, device="cuda").to(dtype) for _ in range(3))
    run(gate, up, do)


@pytest.mark.parametrize("dtype", DTYPES)
def test_zeros(dtype):
    zeros = torch.zeros(300, 129, device="cuda", dtype=dtype)
    out = silu_mul_fwd(zeros, zeros)
    assert torch.equal(out, zeros)
    dg, du = silu_mul_bwd(torch.ones_like(zeros), zeros, zeros)
    assert torch.equal(dg, zeros) and torch.equal(du, zeros)  # silu(0) = 0, d/dg = u * 1/2 = 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_and_negative(dtype):
    """
    Sigmoid saturates at both ends: silu(g) -> g for g >> 0, and -> 0 (signed) for g << 0.
    """
    gen = torch.Generator(device="cuda").manual_seed(2)
    gate = torch.cat(
        [
            torch.linspace(-120, 120, 4096, device="cuda"),
            torch.randn(4096, generator=gen, device="cuda") * 40,
            torch.full((128,), -88.0, device="cuda"),
            torch.full((128,), 88.0, device="cuda"),
        ]
    ).to(dtype)
    up = (torch.randn(gate.shape, generator=gen, device="cuda") * 1000).to(dtype)
    do = (torch.randn(gate.shape, generator=gen, device="cuda") * 100).to(dtype)
    run(gate, up, do)
    out = silu_mul_fwd(gate, up)
    big = gate.float() > 30
    torch.testing.assert_close(
        out.float()[big], (gate.float() * up.float())[big].to(dtype).float(), rtol=1e-2, atol=0
    )
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_float16_moderate_range(dtype):
    gen = torch.Generator(device="cuda").manual_seed(3)
    gate = (torch.randn(1000, 1000, generator=gen, device="cuda") * 8).to(dtype)
    up = (torch.randn(1000, 1000, generator=gen, device="cuda") * 50).to(dtype)
    do = (torch.randn(1000, 1000, generator=gen, device="cuda")).to(dtype)
    run(gate, up, do)


def test_empty():
    empty = torch.empty(0, 768, device="cuda", dtype=torch.bfloat16)
    assert silu_mul_fwd(empty, empty).shape == (0, 768)
    dg, du = silu_mul_bwd(empty, empty, empty)
    assert dg.shape == (0, 768) and du.shape == (0, 768)


def test_deterministic():
    gen = torch.Generator(device="cuda").manual_seed(4)
    gate, up, do = (
        torch.randn(4096, 1408, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)
    )
    assert torch.equal(silu_mul_fwd(gate, up), silu_mul_fwd(gate, up))
    first, second = silu_mul_bwd(do, gate, up), silu_mul_bwd(do, gate, up)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
