"""
Verifier of the pinned dense chunked gated delta rule (gated_delta_rule_pinned_supported,
chunk_gdr_fwd_pinned, chunk_gdr_bwd_pinned), the H200 path of Qwen3.5's Gated DeltaNet.

The recurrence, per value head with state S of shape (K, V) and scale = K ** -0.5:

    S = S * exp(g_t);  u = beta_t * (v_t - k_t @ S);  S = S + outer(k_t, u);  o_t = scale * q_t @ S

q and k are L2-normalized per head, H key heads serve groups of HV // H consecutive value heads,
g is the raw (natural-log) gate in fp32 and beta is bf16.

Checked:

- the capability query: True at the Qwen3.5 geometry on the H200 (batch 1, 32 value heads, 16 or
  32 key heads, dim 128, bf16 q / k / v / beta, fp32 gate), False off it;
- the forward and backward (dq, dk per value head, dv, dg for the raw gate, dbeta) against a naive
  token-by-token fp32 recurrence and its autograd. The kernels keep some chunk intermediates in
  bf16, so the check is calc_diff below 1e-4 (output) and 1e-3 (gradients), plus a max absolute
  error within 2e-2 of the reference's largest magnitude;
- bit-exactness against FLA's chunked gated delta rule on q / k repeated to the value heads, with
  FLA's autotuners restricted to the recorded launch configs.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import (
    chunk_gdr_bwd_pinned,
    chunk_gdr_fwd_pinned,
    gated_delta_rule_pinned_supported,
)

HV, D = 32, 128  # Qwen3.5-35B-A3B: 32 value heads, 16 key heads, head dims 128


def make_inputs(
    S: int,
    H: int = 16,
    B: int = 1,
    gate: str = "model",
    seed: int = 0,
    resets: tuple[int, ...] = (),
) -> tuple[torch.Tensor, ...]:
    """
    Return q / k (B, S, H, D) L2-normalized bf16, v (B, S, HV, D) bf16, g (B, S, HV) fp32, beta
    (B, S, HV) bf16 and do like v. gate="model" draws the gate as the model computes it,
    -exp(A_log) * softplus(a + dt_bias) with A_log = log U(0, 16); "mild" keeps a long memory.
    resets are packed-document starts, where g = -128 as the model sets it.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    device, bf16 = "cuda", torch.bfloat16

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=gen)

    q = F.normalize(randn(B, S, H, D).to(bf16), dim=-1)
    k = F.normalize(randn(B, S, H, D).to(bf16), dim=-1)
    v = randn(B, S, HV, D).to(bf16)
    if gate == "model":
        a_log = torch.log(torch.rand(HV, device=device, generator=gen) * 16)
        g = -a_log.exp() * F.softplus(randn(B, S, HV) + 1.0)
    else:
        g = -torch.rand(B, S, HV, device=device, generator=gen) * 0.1
    if resets:
        g[:, list(resets)] = -128.0
    beta = randn(B, S, HV).to(bf16).sigmoid()
    do = randn(B, S, HV, D).to(bf16)
    return q, k, v, g, beta, do


def h200_path_available() -> bool:
    q, k, v, g, beta, _ = make_inputs(64)
    return gated_delta_rule_pinned_supported(q, k, v, g, beta)


pytestmark = pytest.mark.skipif(
    not h200_path_available(), reason="the pinned launches run on the H200 path only"
)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def naive_gdr(q, k, v, g, beta):
    """
    q / k are per value head, (B, T, HV, D).
    """
    B, T, num_heads, K = q.shape
    scale = K**-0.5
    state = q.new_zeros(B, num_heads, K, v.shape[-1])
    out = []
    for t in range(T):
        state = state * g[:, t].exp()[..., None, None]
        k_t = k[:, t]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k_t, state))
        state = state + k_t[..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state))
    return torch.stack(out, 1)


def per_value_head(x: torch.Tensor) -> torch.Tensor:
    return x.repeat_interleave(HV // x.shape[2], dim=2)


def assert_close_to(got: torch.Tensor, ref: torch.Tensor, name: str, diff: float) -> None:
    assert got.shape == ref.shape, name
    assert torch.isfinite(got).all(), name
    diff_value = calc_diff(got, ref)
    assert diff_value < diff, f"{name}: calc_diff {diff_value:.2e}"
    err = float((got.float() - ref).abs().max())
    bound = 2e-2 * float(ref.abs().max())
    assert err <= bound, f"{name}: max abs err {err:.3e} > {bound:.3e}"


def test_supported_at_the_qwen35_geometry() -> None:
    for H in (16, 32):
        for S in (64, 1000, 4096):
            q, k, v, g, beta, _ = make_inputs(S, H=H)
            assert gated_delta_rule_pinned_supported(q, k, v, g, beta), (H, S)


def test_not_supported_off_the_geometry() -> None:
    q, k, v, g, beta, _ = make_inputs(256)
    bf16 = torch.bfloat16
    off = {
        "batch 2": make_inputs(256, B=2)[:5],
        "16 value heads": (q, k, v[:, :, :16], g[:, :, :16], beta[:, :, :16]),
        "key heads not dividing value heads": (q[:, :, :12], k[:, :, :12], v, g, beta),
        "key dim 64": (q[..., :64].contiguous(), k[..., :64].contiguous(), v, g, beta),
        "value dim 64": (q, k, v[..., :64].contiguous(), g, beta),
        "bf16 gate": (q, k, v, g.to(bf16), beta),
        "fp32 q / k / v": (q.float(), k.float(), v.float(), g, beta),
        "fp32 beta": (q, k, v, g, beta.float()),
        "q and k differ": (q[:, :, :8], k, v, g, beta),
    }
    for name, args in off.items():
        assert not gated_delta_rule_pinned_supported(*args), name


CASES = [
    (64, 16, "model", ()),
    (100, 16, "mild", ()),
    (256, 32, "model", ()),
    (300, 16, "mild", (0, 77, 128, 250)),
    (512, 16, "model", (0, 200)),
    (512, 32, "mild", ()),
]


@pytest.mark.parametrize("S,H,gate,resets", CASES)
def test_against_naive_recurrence(S: int, H: int, gate: str, resets: tuple[int, ...]) -> None:
    q, k, v, g, beta, do = make_inputs(S, H=H, gate=gate, seed=S + H, resets=resets)
    assert gated_delta_rule_pinned_supported(q, k, v, g, beta)

    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    assert o.shape == v.shape and o.dtype == torch.bfloat16
    assert g_cum.shape == g.shape and g_cum.dtype == torch.float32
    assert A.shape == (1, S, HV, 64)
    dq, dk, dv, dg, dbeta = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    assert dq.shape == dk.shape == (1, S, HV, D)
    assert dv.shape == v.shape and dg.shape == g.shape and dbeta.shape == beta.shape
    assert dg.dtype == torch.float32 and dbeta.dtype == dv.dtype == torch.bfloat16

    leaves = [per_value_head(q).float(), per_value_head(k).float(), v.float(), g, beta.float()]
    leaves = [leaf_value.detach().clone().requires_grad_() for leaf_value in leaves]
    ref = naive_gdr(*leaves)
    ref.backward(do.float())

    assert_close_to(o, ref.detach(), "o", 1e-4)
    for name, got, leaf in zip(("dq", "dk", "dv", "dg", "dbeta"), (dq, dk, dv, dg, dbeta), leaves):
        assert_close_to(got, leaf.grad, name, 1e-3)


def test_comparison_is_sensitive() -> None:
    q, k, v, g, beta, _ = make_inputs(256, gate="model", seed=3)
    o, _, _ = chunk_gdr_fwd_pinned(q, k, v, g * 0.9, beta)
    ref = naive_gdr(
        per_value_head(q).float(), per_value_head(k).float(), v.float(), g, beta.float()
    )
    assert calc_diff(o, ref) > 1e-4


# The launch configs FLA's autotuner picks on H200 at this geometry (FLA 0.5.2, Triton 3.7.1), which
# the pinned launches use. FLA is restricted to them so both sides run the same kernels.
RECORDED = {
    "cumsum": dict(num_warps=1, num_stages=3),
    "kkt_solve": dict(BK=64, num_warps=1, num_stages=3),
    "w_u": dict(num_warps=4, num_stages=4),
    "fwd_h": dict(BV=32, num_warps=4, num_stages=3),
    "fwd_o": dict(BK=64, BV=64, num_warps=4, num_stages=3),
    "dv_local": dict(num_warps=2, num_stages=3),
    "bwd_dhu": dict(BV=32, num_warps=4, num_stages=3),
    "dqkwg": dict(num_warps=4, num_stages=3),
    "wy_bwd": dict(num_warps=4, num_stages=4),
}
LAUNCH = ("num_warps", "num_stages")


def fla_kernels() -> dict:
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from fla.ops.common.chunk_o import (
        chunk_bwd_kernel_dqkwg,
        chunk_bwd_kernel_dv_local,
        chunk_fwd_kernel_o,
    )
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.gated_delta_rule.wy_fast import (
        prepare_wy_repr_bwd_kernel,
        recompute_w_u_fwd_kernel,
    )
    from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel

    return {
        "cumsum": chunk_local_cumsum_scalar_kernel,
        "kkt_solve": chunk_gated_delta_rule_fwd_kkt_solve_kernel,
        "w_u": recompute_w_u_fwd_kernel,
        "fwd_h": chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        "fwd_o": chunk_fwd_kernel_o,
        "dv_local": chunk_bwd_kernel_dv_local,
        "bwd_dhu": chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        "dqkwg": chunk_bwd_kernel_dqkwg,
        "wy_bwd": prepare_wy_repr_bwd_kernel,
    }


def pin_fla(monkeypatch: pytest.MonkeyPatch, configs: dict) -> None:
    import triton

    for name, kernel in fla_kernels().items():
        tuner, pick = kernel.fn, configs[name]
        meta = {key: val for key, val in pick.items() if key not in LAUNCH}
        launch = {key: pick[key] for key in LAUNCH}
        monkeypatch.setattr(tuner, "configs", [triton.Config(meta, **launch)])
        monkeypatch.setattr(tuner, "cache", {})


def fla_reference(q, k, v, g, beta, do):
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_fwd,
    )

    q, k = per_value_head(q).contiguous(), per_value_head(k).contiguous()
    scale = D**-0.5
    g_cum, o, A, *_ = chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=None, output_final_state=False
    )
    dq, dk, dv, dbeta, dg, *_ = chunk_gated_delta_rule_bwd(
        q=q, k=k, v=v, g=g_cum, beta=beta, A=A, scale=scale, initial_state=None, do=do, dht=None
    )
    return o, g_cum, A, dq, dk, dv, dg, dbeta


def pinned_path(q, k, v, g, beta, do):
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    dq, dk, dv, dg, dbeta = chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do)
    return o, g_cum, A, dq, dk, dv, dg, dbeta


NAMES = ("o", "g_cum", "A", "dq", "dk", "dv", "dg", "dbeta")


@pytest.mark.parametrize("H", [32, 16])
@pytest.mark.parametrize(
    "S,resets", [(4096, ()), (1000, ()), (4096, (0, 300, 1777, 2048, 4091)), (8192, ())]
)
def test_bit_exact_against_fla(monkeypatch: pytest.MonkeyPatch, S: int, resets, H: int) -> None:
    """
    H = 16 runs grouped value heads on unrepeated q / k; FLA gets them repeated.
    """
    pin_fla(monkeypatch, RECORDED)
    q, k, v, g, beta, do = make_inputs(S, H=H, gate="mild", seed=S, resets=resets)
    got, ref = pinned_path(q, k, v, g, beta, do), fla_reference(q, k, v, g, beta, do)
    for name, got_part, ref_part in zip(NAMES, got, ref):
        assert got_part.dtype == ref_part.dtype and got_part.shape == ref_part.shape, name
        assert torch.equal(got_part, ref_part), name


def test_fla_config_drift_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Another launch config on FLA's side (dqkwg's num_warps) changes the gate gradient, so the exact
    comparison detects config drift.
    """
    configs = dict(RECORDED, dqkwg=dict(RECORDED["dqkwg"], num_warps=2))
    pin_fla(monkeypatch, configs)
    q, k, v, g, beta, do = make_inputs(4096, H=32, gate="mild")
    got, ref = pinned_path(q, k, v, g, beta, do), fla_reference(q, k, v, g, beta, do)
    assert not torch.equal(got[6], ref[6])
