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


def _bf16_round(t: torch.Tensor) -> torch.Tensor:
    """
    t rounded to bf16 and back, with the identity as its gradient.
    """
    return t + (t.bfloat16().float() - t).detach()


def _expert_reference(x, offs, w_gu, b_gu, w_down, b_down, limit):
    """
    fp32 grouped GEMM + bias + clamped SwiGLU + grouped GEMM + bias, one expert at a time, rounding
    to bf16 where the BF16 module stores (so both see the same values at the clamp).
    """
    outs, start = [], 0
    for e, end in enumerate(offs.tolist()):
        xe = x[start:end]
        gate_up = _bf16_round(_bf16_round(xe @ w_gu[e].T) + b_gu[e])
        gate, up = gate_up[:, ::2], gate_up[:, 1::2]
        if limit is not None:
            gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        h = _bf16_round((up + 1) * (gate * torch.sigmoid(ALPHA * gate)))
        outs.append(h @ w_down[e].T + b_down[e])
        start = end
    return torch.cat(outs)


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.double() - want.double()).norm() / want.double().norm()).item()


def test_gpt_oss_experts_unsupported_path(monkeypatch):
    """
    GptOssExperts.forward in BF16 with more local experts than the fused BF16 experts take (so
    uses_bf16_cute() is false and stage 3 runs this forward), whose activation is the clamped_swiglu
    operator, against fp32 PyTorch; the same comparison without the clamp fails, so the check sees
    the activation.
    """
    from types import SimpleNamespace

    from pithtrain.contexts import training
    from pithtrain.models.gpt_oss import GptOssExperts

    monkeypatch.setattr(training, "fp8", False, raising=False)
    num_experts, hidden, inter = 65, 128, 200
    config = SimpleNamespace(hidden_size=hidden, intermediate_size=inter, swiglu_limit=LIMIT)
    gen = torch.Generator().manual_seed(0)
    sizes = torch.randint(0, 9, (num_experts,), generator=gen)
    sizes[3] = 0
    offs = sizes.cumsum(0).to(torch.int32).cuda()
    M = int(sizes.sum())

    experts = GptOssExperts(config, num_experts).cuda()
    with torch.no_grad():
        experts.gate_up_proj.copy_(torch.randn(experts.gate_up_proj.shape, generator=gen) * 0.5)
        experts.gate_up_proj_bias.copy_(torch.randn(experts.gate_up_proj_bias.shape, generator=gen))  # fmt: skip
        experts.down_proj.copy_(torch.randn(experts.down_proj.shape, generator=gen) * 0.05)
        experts.down_proj_bias.copy_(torch.randn(experts.down_proj_bias.shape, generator=gen))
    experts = experts.to(torch.bfloat16)
    assert not experts.uses_bf16_cute()

    x = torch.randn(M, hidden, generator=gen).to(torch.bfloat16).cuda().requires_grad_()
    out = experts(x, offs)
    grad_out = torch.randn(out.shape, generator=gen).to(torch.bfloat16).cuda()
    out.backward(grad_out)

    params = [experts.gate_up_proj, experts.gate_up_proj_bias, experts.down_proj, experts.down_proj_bias]  # fmt: skip
    ref_params = [p.detach().float().requires_grad_() for p in params]
    x_ref = x.detach().float().requires_grad_()
    ref = _expert_reference(x_ref, offs, *ref_params, LIMIT)
    ref.backward(grad_out.float())
    # gate_up ~ N(0, 6^2): the clamp saturates a large share of gates and ups.
    gate_up = x_ref @ ref_params[0][0].T
    assert (gate_up.abs() > LIMIT).float().mean() > 0.1

    tol = 2e-2
    assert _rel_err(out, ref) < tol
    for got, want in zip([x.grad] + [p.grad for p in params], [x_ref.grad] + [p.grad for p in ref_params]):  # fmt: skip
        assert _rel_err(got, want) < tol
    unclamped = _expert_reference(x_ref.detach(), offs, *[p.detach() for p in ref_params], None)
    assert _rel_err(out, unclamped) > 10 * tol
