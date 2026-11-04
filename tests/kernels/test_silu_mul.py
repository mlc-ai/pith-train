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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(7,), (1, 13), (129, 9), (3, 5, 11)])
def test_odd_numel(shape, dtype):
    """
    numel not a multiple of the 16-byte vector: the last elements take the per-element tail.
    """
    gen = torch.Generator(device="cuda").manual_seed(5)
    gate, up, do = (
        torch.randn(shape, generator=gen, device="cuda").mul(3).to(dtype) for _ in range(3)
    )
    run(gate, up, do)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [1, 3])
def test_misaligned_views(dtype, offset):
    """
    Contiguous views whose data does not start on a 16-byte boundary give the same bytes as
    aligned copies of the same values.
    """
    gen = torch.Generator(device="cuda").manual_seed(6)
    rows, cols = 1023, 777
    storage = [
        torch.randn(rows * cols + offset, generator=gen, device="cuda").mul(3).to(dtype)
        for _ in range(3)
    ]
    gate, up, do = (t[offset:].view(rows, cols) for t in storage)
    assert gate.is_contiguous() and gate.data_ptr() % 16 != 0
    run(gate, up, do)
    assert torch.equal(silu_mul_fwd(gate, up), silu_mul_fwd(gate.clone(), up.clone()))
    got, want = silu_mul_bwd(do, gate, up), silu_mul_bwd(do.clone(), gate.clone(), up.clone())
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_side_stream():
    """
    The kernels run on the current stream: a launch on a side stream, read after wait_stream, sees
    inputs written on the default stream before it.
    """
    gen = torch.Generator(device="cuda").manual_seed(7)
    gate, up, do = (
        torch.randn(4096, 1408, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)
    )
    want_out, (want_g, want_u) = silu_mul_fwd(gate, up), silu_mul_bwd(do, gate, up)
    gate2, up2, do2 = torch.empty_like(gate), torch.empty_like(up), torch.empty_like(do)
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    for dst, src in ((gate2, gate), (up2, up), (do2, do)):
        dst.copy_(src)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        out = silu_mul_fwd(gate2, up2)
        dg, du = silu_mul_bwd(do2, gate2, up2)
    main.wait_stream(side)
    assert torch.equal(out, want_out) and torch.equal(dg, want_g) and torch.equal(du, want_u)


def test_empty_all_dtypes():
    for dtype in DTYPES:
        empty = torch.empty(0, device="cuda", dtype=dtype)
        assert silu_mul_fwd(empty, empty).shape == (0,)
        dg, du = silu_mul_bwd(empty, empty, empty)
        assert dg.shape == (0,) and du.shape == (0,)


def test_rejects_bad_inputs():
    """
    Non-contiguous inputs are rejected, not copied, and shapes and dtypes must match (the checks
    every architecture makes: the forward checks both inputs, the backward checks grad_output
    against gate).
    """
    gen = torch.Generator(device="cuda").manual_seed(8)
    gate, up, do = (
        torch.randn(64, 128, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)
    )
    gate_nc, up_nc, do_nc = (t.t().contiguous().t() for t in (gate, up, do))
    assert not gate_nc.is_contiguous() and torch.equal(gate_nc, gate)
    with pytest.raises(AssertionError):
        silu_mul_fwd(gate_nc, up)
    with pytest.raises(AssertionError):
        silu_mul_fwd(gate, up_nc)
    with pytest.raises(AssertionError):
        silu_mul_fwd(gate[:, ::2], up[:, ::2])
    with pytest.raises(AssertionError):
        silu_mul_bwd(do_nc, gate, up)
    with pytest.raises(AssertionError):
        silu_mul_fwd(gate, up[:32])
    with pytest.raises(AssertionError):
        silu_mul_bwd(do[:32], gate, up)
    with pytest.raises(AssertionError):
        silu_mul_fwd(gate, up.float())
    with pytest.raises(AssertionError):
        silu_mul_bwd(do.float(), gate, up)


@pytest.mark.parametrize("cpp_wrapper", [False, True])
def test_compiled_matches_eager(cpp_wrapper):
    """
    Callable inside a fullgraph torch.compile region (as the compiled MLP stages call it). The
    compiled code may compute the same fp32 formulas differently from the eager kernel (SM90 runs
    Inductor's code there), so each output is checked within one output ulp of the fp32 reference
    and within one ulp of the eager result; the largest difference is reported.
    """
    torch._dynamo.reset()
    gen = torch.Generator(device="cuda").manual_seed(9)
    gate, up, do = (
        torch.randn(1000, 1408, generator=gen, device="cuda").mul(s).to(torch.bfloat16)
        for s in (3, 1, 1)
    )

    def body(gate, up, do):
        return silu_mul_fwd(gate, up), *silu_mul_bwd(do, gate, up)

    compiled = torch.compile(body, fullgraph=True, options={"cpp_wrapper": cpp_wrapper})
    got, want = compiled(gate, up, do), body(gate, up, do)
    want_out, (want_g, want_u) = ref_fwd(gate, up), ref_bwd(do, gate, up)
    scale_g = (do.float() * up.float()).abs().max().item()
    for name, compiled_out, eager_out, ref, scale in zip(
        ("out", "grad_gate", "grad_up"),
        got,
        want,
        (want_out, want_g, want_u),
        (want_out.abs().max().item(), scale_g, want_u.abs().max().item()),
    ):
        assert compiled_out.dtype == eager_out.dtype and compiled_out.shape == eager_out.shape
        assert_rounded(compiled_out, ref, scale)
        # one ulp per element, exactly: adjacent bf16 values of one sign differ by 1 in their bit patterns
        bits_c, bits_e = compiled_out.view(torch.int16).int(), eager_out.view(torch.int16).int()
        same_sign = (bits_c < 0) == (bits_e < 0)
        ulps = torch.where(same_sign, (bits_c - bits_e).abs(), bits_c.abs() + bits_e.abs())
        print(f"{name}: max |compiled - eager| = {ulps.max().item()} ulp")
        assert ulps.max().item() <= 1, name


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_non_current_device():
    """
    Inputs on a device other than the current one run there, on that device's current stream, with
    the same bytes as a launch from that device; inputs on two devices are rejected.
    """
    gen = torch.Generator(device="cuda:1").manual_seed(10)
    gate, up, do = (
        torch.randn(1000, 1408, generator=gen, device="cuda:1").to(torch.bfloat16) for _ in range(3)
    )
    with torch.cuda.device(0):
        out, (dg, du) = silu_mul_fwd(gate, up), silu_mul_bwd(do, gate, up)
    with torch.cuda.device(1):
        want_out, (want_g, want_u) = silu_mul_fwd(gate, up), silu_mul_bwd(do, gate, up)
    assert out.device == gate.device and torch.equal(out, want_out)
    assert torch.equal(dg, want_g) and torch.equal(du, want_u)
    with pytest.raises((AssertionError, RuntimeError)):
        silu_mul_fwd(gate, up.to("cuda:0"))
