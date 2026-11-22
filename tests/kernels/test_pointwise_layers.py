"""
The torch-compiled per-token layers (RMSNorm, RoPE, the MoE router) at every pointwise_layers
shape, compiled as the model stages compile them, forward and backward against float64.
"""

import copy
import functools
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig

from pithtrain.kernels import stage_compile_options
from tests.kernels.cases import params
from tests.kernels.common import half_ulp_bf16, record

DEV = "cuda"
F64 = torch.float64
BF16 = dict(device=DEV, dtype=torch.bfloat16)
CONFIG_ROOT = Path(__file__).resolve().parents[2] / "examples" / "pretrain_lm"

SLACK = 2.0**-18  # fp32 error that may flip one bf16 rounding
POINTWISE = 1.0
ROW_REL = 1.2e-2
ROUTE_TIE = 1.7e-2  # near-tied experts may rank either way against float64
GEMM_COEF = 3.3e-2
OWN_TIE = 2.0**-20
WEIGHT_ABS = 8e-7


@functools.cache
def model_config(name: str):
    config = AutoConfig.from_pretrained(CONFIG_ROOT / name)
    if config.model_type in ("qwen3_5", "qwen3_5_moe"):
        config = config.text_config
    return config


def family(config) -> str:
    return {"deepseek_v2": "deepseek_v2", "qwen3_moe": "qwen3_moe", "gpt_oss": "gpt_oss"}.get(
        config.model_type, "qwen35"
    )


def compiled(fn):
    options = {"cpp_wrapper": True, **stage_compile_options()}
    return torch.compile(fn, fullgraph=True, dynamic=False, options=options)


@pytest.fixture(autouse=True)
def stage_setup():
    """
    A fresh Dynamo cache per case and training's TF32 matmul setting.
    """
    torch._dynamo.reset()
    tf32, precision = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    yield
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.set_float32_matmul_precision(precision)
    torch._dynamo.reset()


def rounding_ratio(got: torch.Tensor, exact: torch.Tensor, magnitude: torch.Tensor) -> float:
    err = (got.double() - exact).abs()
    return float((err / (half_ulp_bf16(exact) + SLACK * magnitude + 1e-300)).max())


def gemm_coef(got: torch.Tensor, exact: torch.Tensor, magnitude: torch.Tensor) -> float:
    """
    max |got - exact| / sum |terms| of a reduction (a GEMM or a sum over rows).
    """
    return float(((got.double() - exact).abs() / magnitude.clamp_min(1e-300)).max())


def row_rel(got: torch.Tensor, exact: torch.Tensor) -> float:
    """
    Worst over rows (all leading dims) of max |got - exact| / max |exact|.
    """
    err = (got.double() - exact).abs().flatten(0, -2).amax(-1)
    scale = exact.abs().flatten(0, -2).amax(-1).clamp_min(1e-300)
    return float((err / scale).max())


def make_norm(config, dim: int) -> tuple[nn.Module, bool]:
    """
    The model's RMSNorm of width dim in bf16, and whether it scales by (1 + w).
    """
    eps = config.rms_norm_eps
    if family(config) == "qwen35":
        from pithtrain.models.qwen35_moe import Qwen35MoeRMSNorm

        norm, plus_one = Qwen35MoeRMSNorm(dim, eps=eps), True
    else:
        norm, plus_one = nn.RMSNorm(dim, eps=eps), False
    return norm.to(**BF16), plus_one


def check_norm(case_id: str, name: str, config, x: torch.Tensor, gen: torch.Generator) -> None:
    dim = x.shape[-1]
    norm, plus_one = make_norm(config, dim)
    with torch.no_grad():
        noise = torch.randn(dim, device=DEV, generator=gen) * 0.1
        norm.weight.copy_(noise if plus_one else 1 + noise)
    grad = torch.randn(x.shape, device=DEV, generator=gen).to(torch.bfloat16)
    leaf = x.clone().requires_grad_(True)
    out = compiled(norm)(leaf)
    out.backward(grad)

    xd, gd = x.double().requires_grad_(True), grad.double()
    wd = norm.weight.detach().double().requires_grad_(True)
    scale = (1 + wd) if plus_one else wd
    r = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
    ref = xd * r * scale
    ref.backward(gd)
    with torch.no_grad():
        metrics = dict(
            out=rounding_ratio(out, ref, ref.abs()),
            dx=row_rel(leaf.grad, xd.grad),
            dw=row_rel(norm.weight.grad[None], wd.grad[None]),
        )
    record(case_id, f"rmsnorm[{name}]", **metrics)
    assert metrics["out"] <= POINTWISE, f"{name} out: {metrics['out']:.3g}x one bf16 rounding"
    for key in ("dx", "dw"):
        assert metrics[key] <= ROW_REL, f"{name} {key}: row-relative error {metrics[key]:.3g}"


def scaled_rows(shape: tuple, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(shape, device=DEV, generator=gen)
    scale = torch.exp2(torch.rand(shape[:-1] + (1,), device=DEV, generator=gen) * 4 - 2)
    return (x * scale).to(torch.bfloat16)


@pytest.mark.parametrize("case", params("pointwise_layers"))
def test_rmsnorm(case: dict) -> None:
    axes = case["axes"]
    config = model_config(case["model"])
    T, B, S, hidden = axes["T"], axes["B"], axes["S_local"], axes["hidden"]
    gen = torch.Generator(device=DEV).manual_seed(T + hidden)
    check_norm(case["id"], "hidden", config, scaled_rows((B, S, hidden), gen), gen)
    if family(config) in ("qwen3_moe", "qwen35"):
        D = config.head_dim
        check_norm(case["id"], "q_norm", config, scaled_rows((B, S, axes["Hq"], D), gen), gen)
        check_norm(case["id"], "k_norm", config, scaled_rows((B, S, axes["Hkv"], D), gen), gen)
    elif family(config) == "deepseek_v2":
        check_norm(case["id"], "kv_a", config, scaled_rows((B, S, config.kv_lora_rank), gen), gen)


def rope_parts(config, Hq: int, Hkv: int):
    """
    (apply_rotary_posemb, rotary embedding module, q heads x dims, k heads x dims) of the model.
    """
    match family(config):
        case "deepseek_v2":
            from pithtrain.models.deepseek_v2 import DeepSeekV2Attention as Attention
            from pithtrain.models.deepseek_v2 import DeepSeekV2RotaryEmbedding as Rotary

            dim = config.qk_rope_head_dim
            return Attention.apply_rotary_posemb, Rotary(config), (Hq, dim), (1, dim)
        case "qwen3_moe":
            from pithtrain.models.qwen3_moe import Qwen3MoeAttention as Attention
            from pithtrain.models.qwen3_moe import Qwen3MoeRotaryEmbedding as Rotary
        case "gpt_oss":
            from pithtrain.models.gpt_oss import GptOssAttention as Attention
            from pithtrain.models.gpt_oss import GptOssRotaryEmbedding as Rotary
        case _:
            from pithtrain.models.qwen35_moe import Qwen35MoeAttention as Attention
            from pithtrain.models.qwen35_moe import Qwen35MoeRotaryEmbedding as Rotary
    D = config.head_dim
    return Attention.apply_rotary_posemb, Rotary(config), (Hq, D), (Hkv, D)


@pytest.mark.parametrize("case", params("pointwise_layers"))
def test_rope(case: dict) -> None:
    axes = case["axes"]
    config = model_config(case["model"])
    B, S = axes["B"], axes["S_local"]
    config = copy.copy(config)
    config.max_position_embeddings = S  # as setup_model sets it to the sequence length
    apply, rotary, q_shape, k_shape = rope_parts(config, axes["Hq"], axes["Hkv"])
    cos, sin = (t[:S].to(DEV)[None] for t in rotary(S))
    gen = torch.Generator(device=DEV).manual_seed(axes["T"] + q_shape[0])
    q = torch.randn(B, S, *q_shape, device=DEV, generator=gen).to(torch.bfloat16)
    k = torch.randn(B, S, *k_shape, device=DEV, generator=gen).to(torch.bfloat16)
    gq = torch.randn(q.shape, device=DEV, generator=gen).to(torch.bfloat16)
    gk = torch.randn(k.shape, device=DEV, generator=gen).to(torch.bfloat16)
    ql, kl = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
    q_out, k_out = compiled(lambda a, b, c, s: apply(a, b, (c, s)))(ql, kl, cos, sin)
    torch.autograd.backward((q_out, k_out), (gq, gk))

    qd, kd = q.double().requires_grad_(True), k.double().requires_grad_(True)
    q_ref, k_ref = apply(qd, kd, (cos.double(), sin.double()))
    torch.autograd.backward((q_ref, k_ref), (gq.double(), gk.double()))
    # each output element is a sum of two products of an input and a table entry
    table = max(float(cos.abs().max()), float(sin.abs().max()))
    with torch.no_grad():

        def terms(t: torch.Tensor) -> torch.Tensor:
            return 2 * table * t.double().abs().amax(-1, keepdim=True).expand(t.shape)

        metrics = dict(
            q=rounding_ratio(q_out, q_ref, terms(q)),
            k=rounding_ratio(k_out, k_ref, terms(k)),
            dq=rounding_ratio(ql.grad, qd.grad, terms(gq)),
            dk=rounding_ratio(kl.grad, kd.grad, terms(gk)),
        )
    record(case["id"], "rope", **metrics)
    for key, value in metrics.items():
        assert value <= POINTWISE, f"{key}: {value:.3g}x one bf16 rounding"


def make_gate(config) -> nn.Module:
    match family(config):
        case "deepseek_v2":
            from pithtrain.models.deepseek_v2 import DeepSeekV2MoEGate as Gate
        case "qwen3_moe":
            from pithtrain.models.qwen3_moe import Qwen3MoeGate as Gate
        case "gpt_oss":
            from pithtrain.models.gpt_oss import GptOssTopKRouter as Gate
        case _:
            from pithtrain.models.qwen35_moe import Qwen35MoeTopKRouter as Gate
    return Gate(config).to(**BF16)


def model_logits(config, gate: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    The gate's logits as its forward computes them, eagerly.
    """
    if family(config) == "deepseek_v2":
        return F.linear(x.float(), gate.weight.float())
    return F.linear(x, gate.weight, getattr(gate, "bias", None))


def weights_of(config, logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    The weights of the experts idx (T, k) from logits (T, E), as the model's gate computes them.
    """
    match family(config):
        case "gpt_oss":
            return logits.gather(-1, idx).softmax(-1)
        case "deepseek_v2":
            assert config.topk_method == "greedy"
            return logits.softmax(-1).gather(-1, idx) * config.routed_scaling_factor
        case name:
            weights = logits.softmax(-1).gather(-1, idx)
            if name == "qwen35" or config.norm_topk_prob:
                weights = weights / weights.sum(-1, keepdim=True)
            return weights


@pytest.mark.parametrize("case", params("pointwise_layers", keep=lambda c: "E" in c["axes"]))
def test_router(case: dict) -> None:
    """
    The compiled gate's choice, weights and gradients against float64.
    """
    axes = case["axes"]
    config = model_config(case["model"])
    T, hidden, E, top_k = axes["T"], axes["hidden"], axes["E"], axes["top_k"]
    gate = make_gate(config)
    gen = torch.Generator(device=DEV).manual_seed(sum(map(ord, case["id"])))
    with torch.no_grad():
        gate.weight.copy_(torch.randn(E, hidden, device=DEV, generator=gen) * hidden**-0.5)
        if getattr(gate, "bias", None) is not None:
            gate.bias.copy_(torch.randn(E, device=DEV, generator=gen) * 0.1)
    x = scaled_rows((T, hidden), gen)
    grad = torch.randn(T, top_k, device=DEV, generator=gen)
    leaf = x.clone().requires_grad_(True)
    # gpt-oss compiles its router itself
    run = gate if family(config) == "gpt_oss" else compiled(lambda h: gate(h))
    idx, weights, _ = run(leaf)
    weights.backward(grad)

    bias = getattr(gate, "bias", None)
    with torch.no_grad():
        xd, wd = x.double(), gate.weight.double()
        exact = xd @ wd.T + (bias.double() if bias is not None else 0)
        given = model_logits(config, gate, x).double()
    given.requires_grad_(True)
    ref_weights = weights_of(config, given, idx)
    ref_weights.backward(grad.double())
    with torch.no_grad():
        dlogits = given.grad

        def gap(logits):
            kth = logits.topk(top_k, dim=-1).values[:, -1]
            return float(((kth - logits.gather(-1, idx).amin(-1)) / logits.abs().amax(-1)).max())

        unique = bool((idx.sort(-1).values.diff(dim=-1) > 0).all())
        exact_idx = exact.topk(top_k, -1).indices.sort(-1).values
        metrics = dict(
            route_gap=max(gap(exact), 0.0),
            own_gap=max(gap(given), 0.0),
            mismatched_tokens=int((exact_idx != idx.sort(-1).values).any(-1).sum()),
            weight=float((weights.double() - ref_weights).abs().max()),
            dx=gemm_coef(leaf.grad, dlogits @ wd, dlogits.abs() @ wd.abs()),
            dw=gemm_coef(gate.weight.grad, dlogits.T @ xd, dlogits.abs().T @ xd.abs()),
        )
        if bias is not None:
            metrics["db"] = gemm_coef(bias.grad, dlogits.sum(0), dlogits.abs().sum(0))
    record(case["id"], "router", **metrics)
    assert idx.shape == (T, top_k) and unique, "the selection must be top_k distinct experts"
    assert metrics["route_gap"] <= ROUTE_TIE, f"an expert chosen {metrics['route_gap']:.3g} below the exact k-th logit"  # fmt: skip
    assert metrics["own_gap"] <= OWN_TIE, f"an expert chosen {metrics['own_gap']:.3g} below the gate's own k-th logit"  # fmt: skip
    assert metrics["weight"] <= WEIGHT_ABS, f"router weights off by {metrics['weight']:.3g}"
    for key in ("dx", "dw", "db"):
        if key in metrics:
            assert metrics[key] <= GEMM_COEF, f"{key}: error {metrics[key]:.3g} x sum |terms|"
