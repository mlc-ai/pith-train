"""Correctness tests for the fused clamped-SwiGLU operator used in gpt-oss."""

import pytest
import torch

from pithtrain.operators.clamped_swiglu import clamped_swiglu

ALPHA = 1.702
LIMIT = 7.0


def _ref_forward(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    gate = gate_up[:, ::2]
    up = gate_up[:, 1::2]
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    glu = gate_c * torch.sigmoid(alpha * gate_c)
    return (up_c + 1) * glu


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 2),
        (128, 768),
        (4096, 2048),
        (8192, 4096),
        (1023, 778),  # N odd, not a power of two
        (37, 2050),  # M arbitrary, 2N not aligned to block
    ],
)
def test_clamped_swiglu_forward_backward(shape, dtype):
    torch.manual_seed(0)
    device = "cuda"

    M, two_n = shape
    assert two_n % 2 == 0, "test shapes must have even last dim"

    # Sample a mix of in-range and saturating values so the clamp branch is
    # exercised on both signs of ``up`` and the upper bound of ``gate``.
    gate_up_ref = (torch.randn(M, two_n, device=device, dtype=dtype) * 4.0).requires_grad_()
    gate_up = gate_up_ref.detach().clone().requires_grad_()

    out_ref = _ref_forward(gate_up_ref, ALPHA, LIMIT)
    out = clamped_swiglu(gate_up, ALPHA, LIMIT)

    # bf16/fp16 store each intermediate rounded; our fused kernel keeps
    # everything in fp32 until the final store, so it is strictly *more*
    # accurate than the reference - loosen tolerance for the reference's noise.
    atol, rtol = (5e-2, 1e-2) if dtype == torch.float16 else (3e-1, 5e-2)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(gate_up.grad, gate_up_ref.grad, atol=atol, rtol=rtol)


def test_clamped_swiglu_saturated_gradient_is_zero():
    """Values strictly outside [-limit, limit] must produce zero gradient.

    Clamp's chain-rule is zero once we leave the feasible region, so any
    implementation that drops the mask (e.g. treating ``clamp`` as identity
    in backward) will fail this test.
    """
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    M, N = 8, 16
    two_n = 2 * N
    gate_up = torch.empty(M, two_n, device=device, dtype=dtype)

    # Gate even columns, up odd columns.
    gate_vals = torch.linspace(-LIMIT - 4.0, LIMIT + 4.0, M * N, device=device).view(M, N)
    up_vals = torch.linspace(LIMIT + 3.0, LIMIT + 5.0, M * N, device=device).view(M, N)
    gate_up[:, ::2] = gate_vals
    gate_up[:, 1::2] = up_vals
    gate_up.requires_grad_()

    out = clamped_swiglu(gate_up, ALPHA, LIMIT)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    grad_gate = gate_up.grad[:, ::2]
    grad_up = gate_up.grad[:, 1::2]

    # Gate: one-sided saturation. g > LIMIT => grad_g == 0.
    saturated_gate = gate_vals > LIMIT
    assert torch.all(grad_gate[saturated_gate] == 0)

    # Up: every up_val > LIMIT here, so every grad_up cell must be zero.
    assert torch.all(grad_up == 0)


def test_clamped_swiglu_matches_hf_reference_shape():
    """Smoke test using gpt-oss-20b-like dims: 32 experts x 768-width tile."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # Realistic per-expert token count and intermediate_size.
    M, intermediate = 4096, 2880
    gate_up_ref = torch.randn(M, 2 * intermediate, device=device, dtype=dtype).requires_grad_()
    gate_up = gate_up_ref.detach().clone().requires_grad_()

    out_ref = _ref_forward(gate_up_ref, ALPHA, LIMIT)
    out = clamped_swiglu(gate_up, ALPHA, LIMIT)
    torch.testing.assert_close(out, out_ref, atol=3e-1, rtol=5e-2)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)
    torch.testing.assert_close(gate_up.grad, gate_up_ref.grad, atol=3e-1, rtol=5e-2)
