"""
The Gated DeltaNet recurrence in plain PyTorch, for the spec-driven GDN tests. Per value head with
state S of shape (K, V) and scale = K ** -0.5:

    S = S * exp(g_t);  u = beta_t * (v_t - k_t @ S);  S = S + outer(k_t, u);  o_t = scale * q_t @ S

q and k are L2-normalized per head, H key heads serve groups of HV // H consecutive value heads, g
is the raw (natural-log) gate in fp32 and beta is bf16.

BOUNDS (calc_diff per output) and MAX_ABS (largest absolute error over the reference's largest
magnitude) are those the pinned kernels were calibrated with at T <= 8192 (about 2.5 to 3 times the
largest errors measured on the H200: calc_diff 8.1e-6 (o), 8.1e-6 (dq), 1.1e-5 (dk), 6.1e-6 (dv),
3.6e-5 (dg, at T = 2), 7.7e-6 (dbeta), and 9.3e-3 of the largest magnitude). The kernels keep some
chunk intermediates in bf16, so these are measured, not analytic.
"""

import torch
import torch.nn.functional as F

D = 128  # Qwen3.5's key and value head dims
BT = 64  # the kernels' chunk size
GRADS = ("dq", "dk", "dv", "dg", "dbeta")
BOUNDS = {"o": 2e-5, "dq": 2e-5, "dk": 3e-5, "dv": 2e-5, "dg": 1e-4, "dbeta": 2e-5}
MAX_ABS = 2e-2


def make_inputs(
    S: int, H: int, hv: int, B: int = 1, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """
    q / k (B, S, H, D) L2-normalized bf16, v (B, S, hv, D) bf16, g (B, S, hv) fp32, beta (B, S, hv)
    bf16 and do like v, the gate drawn as the model computes it: -exp(A_log) * softplus(a + dt_bias)
    with A_log = log U(0, 16).
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    device, bf16 = "cuda", torch.bfloat16

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=gen)

    q = F.normalize(randn(B, S, H, D).to(bf16), dim=-1)
    k = F.normalize(randn(B, S, H, D).to(bf16), dim=-1)
    v = randn(B, S, hv, D).to(bf16)
    a_log = torch.log(torch.rand(hv, device=device, generator=gen) * 16)
    g = -a_log.exp() * F.softplus(randn(B, S, hv) + 1.0)
    beta = randn(B, S, hv).to(bf16).sigmoid()
    do = randn(B, S, hv, D).to(bf16)
    return q, k, v, g, beta, do


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def per_value_head(x: torch.Tensor, hv: int) -> torch.Tensor:
    return x.repeat_interleave(hv // x.shape[2], dim=2)


def naive_gdr(q, k, v, g, beta):
    """
    The token-by-token recurrence; q / k are per value head, (B, T, HV, D).
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


def chunked_reference(q, k, v, g, beta, do, chunk: int = BT) -> tuple[torch.Tensor, dict]:
    """
    The recurrence's output and autograd gradients (dq, dk per value head) in fp64, in its chunked
    form, whose autograd stays small at long lengths: within a chunk entering with state S0 and
    cumulative gate G, U solves (I + A) U = beta * (V - exp(G) * (K S0)) with A[t, i] = beta_t
    exp(G_t - G_i) k_t . k_i for i < t; o = scale * (exp(G) * (Q S0) + (exp(G_t - G_i) q_t .
    k_i)_{i <= t} U); the state leaving is exp(G_L) S0 + sum_i exp(G_L - G_i) outer(k_i, u_i).
    """
    hv = v.shape[2]
    leaves = [per_value_head(q, hv), per_value_head(k, hv), v, g, beta]
    leaves = [x.detach().double().requires_grad_() for x in leaves]
    qd, kd, vd, gd, bd = leaves
    scale = qd.shape[-1] ** -0.5
    state = qd.new_zeros(hv, qd.shape[-1], vd.shape[-1])
    out = []
    for c0 in range(0, qd.shape[1], chunk):
        span = slice(c0, min(c0 + chunk, qd.shape[1]))
        qc, kc, vc = (t[0, span].transpose(0, 1) for t in (qd, kd, vd))
        gc, bc = gd[0, span].transpose(0, 1).cumsum(-1), bd[0, span].transpose(0, 1)
        L = gc.shape[-1]
        lower = torch.ones(L, L, dtype=torch.bool, device=qd.device).tril()
        decay = torch.where(lower, gc[:, :, None] - gc[:, None, :], float("-inf")).exp()
        strict = decay * (~torch.eye(L, dtype=torch.bool, device=qd.device))
        A = bc[:, :, None] * strict * (kc @ kc.transpose(-1, -2))
        rhs = bc[..., None] * (vc - gc.exp()[..., None] * (kc @ state))
        eye = torch.eye(L, dtype=qd.dtype, device=qd.device)
        u = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
        o = scale * (gc.exp()[..., None] * (qc @ state) + (decay * (qc @ kc.transpose(-1, -2))) @ u)  # fmt: skip
        out.append(o.transpose(0, 1))
        k_decayed = kc * (gc[:, -1:] - gc).exp()[..., None]
        state = gc[:, -1].exp()[:, None, None] * state + k_decayed.transpose(-1, -2) @ u
    ref = torch.cat(out, 0).unsqueeze(0)
    ref.backward(do.double())
    return ref.detach(), {name: leaf.grad for name, leaf in zip(GRADS, leaves)}
