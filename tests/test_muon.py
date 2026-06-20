"""CPU tests for the Muon optimizer (``pithtrain.modules.optimizer``) and its
checkpoint wiring. The FSDP gather/reshard path is in the multi-GPU tests."""

import torch
import torch.nn as nn
from torch.optim import AdamW

from pithtrain.modules.checkpoint import to_canonical_optim, to_localized_optim
from pithtrain.modules.optimizer import Muon, muon_scale_factor, zeropower_via_newtonschulz5
from pithtrain.modules.training import is_muon_param

# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------


def _well_conditioned(n: int, m: int):
    """Random (n, m) with singular values in [0.5, 1.5] and its true UV^T.

    NS only drives the spectrum toward ~1 on well-conditioned inputs (a raw
    Gaussian is near-singular), so we control it to test convergence."""
    r = min(n, m)
    u, _ = torch.linalg.qr(torch.randn(n, r))
    v, _ = torch.linalg.qr(torch.randn(m, r))
    s = torch.empty(r).uniform_(0.5, 1.5)
    return (u * s) @ v.mT, u @ v.mT


@torch.no_grad()
def test_newton_schulz_approximates_orthogonalization():
    torch.manual_seed(0)
    # Square, tall, and wide: NS must handle the transpose-for-tall trick.
    for shape in [(64, 64), (96, 48), (48, 96)]:
        g, true = _well_conditioned(*shape)
        o = zeropower_via_newtonschulz5(g, steps=5).float()

        # Newton-Schulz drives the singular values toward ~1.
        svals = torch.linalg.svdvals(o)
        assert svals.min() > 0.5, (shape, svals.min().item())
        assert svals.max() < 1.5, (shape, svals.max().item())

        # ... and points the same direction as the true UV^T.
        cos = torch.nn.functional.cosine_similarity(o.flatten(), true.flatten(), dim=0)
        assert cos > 0.97, (shape, cos.item())


@torch.no_grad()
def test_newton_schulz_batched_experts():
    """A stacked [E, out, in] expert weight is orthogonalized per-expert."""
    torch.manual_seed(0)
    mats = [_well_conditioned(32, 48) for _ in range(4)]
    g = torch.stack([m[0] for m in mats])
    o = zeropower_via_newtonschulz5(g, steps=5).float()
    assert o.shape == g.shape
    for e, (_, true) in enumerate(mats):
        cos = torch.nn.functional.cosine_similarity(o[e].flatten(), true.flatten(), dim=0)
        assert cos > 0.97, (e, cos.item())


def test_muon_scale_factor():
    assert muon_scale_factor(8, 8) == 0.2 * 8**0.5
    assert muon_scale_factor(100, 4) == 0.2 * 100**0.5  # uses max dim
    assert muon_scale_factor(4, 100) == 0.2 * 100**0.5


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------


def test_is_muon_param_classification():
    # (name, ndim, expected) covering every param kind across the 3 models.
    cases = [
        # 2D attention / MLP / MLA hidden weights -> Muon
        ("model.layers.0.self_attn.q_proj.weight", 2, True),
        ("model.layers.0.self_attn.k_proj.weight", 2, True),
        ("model.layers.0.self_attn.o_proj.weight", 2, True),
        ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", 2, True),
        ("model.layers.0.self_attn.kv_b_proj.weight", 2, True),
        ("model.layers.0.mlp.gate_proj.weight", 2, True),
        ("model.layers.0.mlp.up_proj.weight", 2, True),
        ("model.layers.0.mlp.down_proj.weight", 2, True),
        # 3D stacked expert weights (incl. fused gate_up) -> Muon
        ("model.layers.0.mlp.experts.gate_proj.weight", 3, True),
        ("model.layers.0.mlp.experts.down_proj.weight", 3, True),
        ("model.layers.0.mlp.experts.gate_up_proj.weight", 3, True),
        # biases (1D and 2D-stacked) -> adamw
        ("model.layers.0.self_attn.q_proj.bias", 1, False),
        ("model.layers.0.mlp.experts.gate_up_proj_bias", 2, False),
        ("model.layers.0.mlp.experts.down_proj_bias", 2, False),
        # norms / sinks (1D) -> adamw
        ("model.layers.0.input_layernorm.weight", 1, False),
        ("model.layers.0.self_attn.q_norm.weight", 1, False),
        ("model.layers.0.self_attn.kv_a_layernorm.weight", 1, False),
        ("model.layers.0.self_attn.sinks", 1, False),
        # router / gate (2D but a classifier) -> adamw
        ("model.layers.0.mlp.gate.weight", 2, False),
        ("model.layers.0.mlp.router.weight", 2, False),
        ("model.layers.0.mlp.router.bias", 1, False),
        # embeddings / head (2D) -> adamw
        ("model.embed_tokens.weight", 2, False),
        ("model.lm_head.weight", 2, False),
    ]
    for name, ndim, expected in cases:
        param = torch.empty([4] * ndim)
        assert is_muon_param(name, param) is expected, name


class _TinyModel(nn.Module):
    """Minimal module whose named parameters span both optimizer groups."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=True)  # weight -> Muon, bias -> adamw
        self.o_proj = nn.Linear(8, 8, bias=False)  # weight -> Muon
        self.norm = nn.LayerNorm(8)  # weight + bias (1D) -> adamw
        self.embed_tokens = nn.Embedding(16, 8)  # 2D, name-excluded -> adamw


# ---------------------------------------------------------------------------
# Optimizer step math (plain tensors)
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_muon_step_matches_manual_update():
    torch.manual_seed(0)
    lr, beta, wd = 0.02, 0.9, 0.1
    p = nn.Parameter(torch.randn(16, 24))
    p.grad = torch.randn(16, 24)
    p0, grad = p.detach().clone(), p.grad.clone()

    Muon([p], lr=lr, momentum=beta, weight_decay=wd).step()

    buf = (1 - beta) * grad
    update = grad.lerp(buf, beta)
    orth = zeropower_via_newtonschulz5(update, steps=5).to(p.dtype)
    orth = orth * muon_scale_factor(update.size(-2), update.size(-1))
    expected = p0 * (1 - lr * wd) - lr * orth
    torch.testing.assert_close(p.detach(), expected, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_step_accepts_bf16_grad():
    """bf16 grads must not dtype-clash with Muon's fp32 state."""
    torch.manual_seed(0)
    # bf16 params/grads (as under FSDP param_dtype=bf16) vs the fp32 momentum
    # buffer, composed as in training: Muon for the weight, AdamW for the bias.
    w = nn.Parameter(torch.randn(16, 24, dtype=torch.bfloat16))  # 2D -> Muon
    b = nn.Parameter(torch.randn(24, dtype=torch.bfloat16))  # 1D -> adamw
    w.grad = torch.randn(16, 24, dtype=torch.bfloat16)
    b.grad = torch.randn(24, dtype=torch.bfloat16)

    Muon([w], lr=0.02).step()
    AdamW([b], lr=3e-4).step()
    assert torch.isfinite(w).all() and torch.isfinite(b).all()


# ---------------------------------------------------------------------------
# Checkpoint param-group round-trip (no DCP, no experts)
# ---------------------------------------------------------------------------


def test_checkpoint_param_groups_preserve_membership():
    """A composed [Muon, AdamW] state dict round-trips: each group keeps its own
    membership and hyperparameters (no leakage), as DCP does for optimizers."""
    model = _TinyModel()
    muon_params, adamw_params = [], []
    for n, p in model.named_parameters():
        (muon_params if is_muon_param(n, p) else adamw_params).append(p)
    optimizers = (Muon(muon_params, lr=0.01), AdamW(adamw_params, lr=0.01, weight_decay=0.0))

    name_of = {p: n for n, p in model.named_parameters()}
    muon_fqns = [name_of[p] for p in muon_params]
    adamw_fqns = [name_of[p] for p in adamw_params]

    # Combined (Muon, AdamW) state dict as DCP returns it: FQN-keyed state,
    # param_groups in optimizer order.
    optim_state = {
        "state": {n: {"momentum_buffer": torch.zeros(1)} for n in muon_fqns}
        | {
            n: {"exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1), "step": 1}
            for n in adamw_fqns
        },
        "param_groups": [
            {"params": list(muon_fqns), "lr": 0.01, "momentum": 0.95, "weight_decay": 0.0},
            {
                "params": list(adamw_fqns),
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
        ],
    }

    localized = to_localized_optim(to_canonical_optim(optim_state, model), model, optimizers)
    muon_group, adamw_group = localized["param_groups"]

    assert set(muon_group["params"]) == set(muon_fqns)
    assert set(adamw_group["params"]) == set(adamw_fqns)
    assert set(muon_group["params"]).isdisjoint(adamw_group["params"])
    # Per-group hyperparameters stay with the right group (no leakage).
    assert "momentum" in muon_group and "betas" not in muon_group
    assert "betas" in adamw_group and "momentum" not in adamw_group
