"""Correctness tests for the fused ``silu(gate) * up`` operator."""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.operators.silu_mul import silu_mul


def _ref_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (128, 768),
        (4096, 2048),
        (8192, 4096),
        (1023, 777),  # non-multiple of BLOCK_SIZE
    ],
)
def test_silu_mul_forward_backward(shape, dtype):
    torch.manual_seed(0)
    device = "cuda"

    gate_ref = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    up_ref = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    gate = gate_ref.detach().clone().requires_grad_()
    up = up_ref.detach().clone().requires_grad_()

    out_ref = _ref_forward(gate_ref, up_ref)
    out = silu_mul(gate, up)

    atol = 1e-2
    rtol = 1e-2
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(gate.grad, gate_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(up.grad, up_ref.grad, atol=atol, rtol=rtol)
