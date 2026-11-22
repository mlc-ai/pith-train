"""
The Gated DeltaNet recurrence in plain PyTorch. BOUNDS (calc_diff) and MAX_ABS (max abs error over
the reference's max magnitude) are about 3x measured on the H200.
"""

import torch
import torch.nn.functional as F

D = 128  # Qwen3.5's key and value head dims
BT = 64  # the kernels' chunk size
GRADS = ("dq", "dk", "dv", "dg", "dbeta")
BOUNDS = {"o": 1.6e-5, "dq": 2e-5, "dk": 2e-5, "dv": 1.4e-5, "dg": 2e-5, "dbeta": 1.4e-5}
MAX_ABS = 2.8e-2


def make_inputs(S: int, H: int, hv: int, B: int = 1, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """
    Random q, k, v, g, beta and do, drawn as the model computes them.
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
    The token-by-token recurrence, q / k per value head.
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
    The recurrence's output and autograd gradients in fp64, in chunked form.
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


NAMES = ("o", "dq", "dk", "dv", "dg", "dbeta")
TOKEN_HEADS_PER_BLOCK = 1 << 21  # caps the fp64 reference's memory


def reference_by_heads(q, k, v, g, beta, do, keep=slice(None), key_grads=False):
    """
    The fp64 reference's [o, dq, dk, dv, dg, dbeta] for B = 1, a block of value heads at a time.
    """
    S, H, HV = q.shape[1], q.shape[2], v.shape[2]
    group = HV // H
    block = max(group, min(HV, TOKEN_HEADS_PER_BLOCK // S) // group * group)
    outs = [[] for _ in NAMES]
    for start in range(0, HV, block):
        values, keys = slice(start, start + block), slice(start // group, (start + block) // group)
        ref, grads = chunked_reference(q[:, :, keys], k[:, :, keys], *(t[:, :, values] for t in (v, g, beta, do)))  # fmt: skip
        dq, dk = grads["dq"], grads["dk"]
        if key_grads:
            dq, dk = (x.view(1, S, -1, group, x.shape[-1]).sum(3) for x in (dq, dk))
        for out, x in zip(outs, (ref, dq, dk, grads["dv"], grads["dg"], grads["dbeta"])):
            out.append(x[:, keep].float())
        del ref, grads, dq, dk
    return [torch.cat(parts, 2) for parts in outs]


def errors(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, bool]:
    """
    (calc_diff, max abs error / the reference's max magnitude, all finite).
    """
    scale = float(ref.abs().max())
    err = float((got.float() - ref).abs().max())
    return calc_diff(got, ref), err / scale if scale else err, bool(torch.isfinite(got).all())  # fmt: skip


def failures_of(label: str, got: list, ref: list) -> list[str]:
    failures = []
    for name, x, y in zip(NAMES, got, ref):
        assert x.shape == y.shape, (label, name, x.shape, y.shape)
        diff, rel_max, finite = errors(x, y)
        print(f"MEASURED {label} {name} calc_diff={diff:.3e} max_abs/ref_max={rel_max:.3e}")
        if not (finite and diff < BOUNDS[name] and rel_max <= MAX_ABS):
            failures.append(f"{label} {name}: calc_diff {diff:.2e} (bound {BOUNDS[name]:.0e}), max abs {rel_max:.2e} of ref max (bound {MAX_ABS:.0e}), finite {finite}")  # fmt: skip
    return failures
