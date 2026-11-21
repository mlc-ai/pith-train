"""
Kernel-facing shapes of the supported models, read from examples/pretrain_lm/*/config.json.

One extractor per model_type, written against the model files in pithtrain/models/: a config whose
model_type has no extractor fails the generator (and tests/test_kernel_cases.py), so a model added
to examples/pretrain_lm/ cannot land without kernel-case coverage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "examples" / "pretrain_lm"


@dataclass(frozen=True)
class Projection:
    """
    One training.Linear of a layer: y (M, n) = x (M, k) @ W.T, run layers times per micro-batch.
    """

    name: str
    n: int
    k: int
    layers: int


@dataclass(frozen=True)
class Attention:
    """
    Softmax attention as FlashAttention-4 sees it (after MLA decompression for DeepSeek-V2).
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    layers: int
    window: int | None = None
    sinks: bool = False
    mla_kv_lora_rank: int | None = None
    mla_rope_dim: int | None = None
    supports_cp: bool = True


@dataclass(frozen=True)
class Experts:
    """
    Routed experts: num_experts of hidden -> 2 * inter -> hidden, top_k per token.
    """

    num_experts: int
    top_k: int
    hidden: int
    inter: int
    layers: int
    act: str
    bias: bool
    gate_up_layout: str
    """
    separate (two (E, I, H) weights), halves (one (E, 2I, H) weight [gate; up]) or interleaved (one
    weight, gate / up rows alternating, gpt-oss): it decides the gate_up GEMM width and kernels.
    """
    swiglu_alpha: float | None = None
    swiglu_limit: float | None = None
    """
    The clamped SwiGLU's alpha (1.702 in pithtrain/models/gpt_oss.py) and limit (config).
    """


@dataclass(frozen=True)
class LinearAttention:
    """
    Gated DeltaNet: key heads of key_dim, value heads of value_dim, and its causal conv1d over the
    mixed q / k / v channels.
    """

    num_key_heads: int
    num_value_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int
    layers: int

    @property
    def conv_channels(self) -> int:
        return 2 * self.num_key_heads * self.key_dim + self.num_value_heads * self.value_dim


@dataclass(frozen=True)
class Model:
    """
    Everything the case generator needs from one config.json.
    """

    name: str
    model_type: str
    config_path: str
    config_sha256: str
    hidden: int
    vocab: int
    num_layers: int
    projections: tuple[Projection, ...]
    attention: Attention
    experts: Experts | None
    linear_attention: LinearAttention | None
    swiglu_widths: tuple[tuple[str, int, int], ...] = field(default=())
    """
    (name, inter, layers) of the dense / shared-expert SwiGLU MLPs (silu_mul at M = tokens).
    """


def count(layer_types: list[str], kind: str) -> int:
    return sum(1 for layer_type in layer_types if layer_type == kind)


def deepseek_v2(cfg: dict) -> dict:
    H, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    nope, rope, vdim, lora = (
        cfg["qk_nope_head_dim"],
        cfg["qk_rope_head_dim"],
        cfg["v_head_dim"],
        cfg["kv_lora_rank"],
    )
    L, dense = cfg["num_hidden_layers"], cfg["first_k_dense_replace"]
    shared = cfg["n_shared_experts"] * cfg["moe_intermediate_size"]
    assert cfg.get("q_lora_rank") is None, "q_lora_rank adds q_a / q_b projections"
    projections = (
        Projection("q_proj", heads * (nope + rope), H, L),
        Projection("kv_a_proj_with_mqa", lora + rope, H, L),
        Projection("kv_b_proj", heads * (nope + vdim), lora, L),
        Projection("o_proj", H, heads * vdim, L),
        Projection("dense_gate_up_proj", cfg["intermediate_size"], H, 2 * dense),
        Projection("dense_down_proj", H, cfg["intermediate_size"], dense),
        Projection("shared_gate_up_proj", shared, H, 2 * (L - dense)),
        Projection("shared_down_proj", H, shared, L - dense),
    )
    attention = Attention(
        heads, heads, nope + rope, vdim, L, mla_kv_lora_rank=lora, mla_rope_dim=rope
    )
    experts = Experts(
        cfg["n_routed_experts"],
        cfg["num_experts_per_tok"],
        H,
        cfg["moe_intermediate_size"],
        L - dense,
        "swiglu",
        False,
        "separate",
    )
    swiglu = (("dense_mlp", cfg["intermediate_size"], dense), ("shared_experts", shared, L - dense))
    return dict(
        projections=projections,
        attention=attention,
        experts=experts,
        linear_attention=None,
        swiglu_widths=swiglu,
    )


def qwen3_moe(cfg: dict) -> dict:
    H, heads, kv, D = (
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    L = cfg["num_hidden_layers"]
    assert not cfg.get("mlp_only_layers") and cfg.get("decoder_sparse_step", 1) == 1
    projections = (
        Projection("q_proj", heads * D, H, L),
        Projection("k_proj", kv * D, H, L),
        Projection("v_proj", kv * D, H, L),
        Projection("o_proj", H, heads * D, L),
    )
    attention = Attention(heads, kv, D, D, L)
    experts = Experts(
        cfg["num_experts"],
        cfg["num_experts_per_tok"],
        H,
        cfg["moe_intermediate_size"],
        L,
        "swiglu",
        False,
        "separate",
    )
    return dict(
        projections=projections, attention=attention, experts=experts, linear_attention=None
    )


def gpt_oss(cfg: dict) -> dict:
    H, heads, kv, D = (
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    L = cfg["num_hidden_layers"]
    projections = (
        Projection("q_proj", heads * D, H, L),
        Projection("k_proj", kv * D, H, L),
        Projection("v_proj", kv * D, H, L),
        Projection("o_proj", H, heads * D, L),
    )
    # Alternating sliding (window sliding_window, FA4 left window sliding_window - 1) and full
    # layers, both with learnable sinks. Ring attention has neither, so gpt-oss runs CP 1 only.
    attention = Attention(
        heads, kv, D, D, L, window=cfg["sliding_window"], sinks=True, supports_cp=False
    )
    experts = Experts(
        cfg["num_local_experts"],
        cfg["num_experts_per_tok"],
        H,
        cfg["intermediate_size"],
        L,
        "clamped_swiglu",
        True,
        "interleaved",
        1.702,
        float(cfg["swiglu_limit"]),
    )
    return dict(
        projections=projections, attention=attention, experts=experts, linear_attention=None
    )


def qwen3_5(cfg: dict, moe: bool) -> dict:
    H, heads, kv, D = (
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    layer_types = cfg["layer_types"]
    full, linear = count(layer_types, "full_attention"), count(layer_types, "linear_attention")
    gdn = LinearAttention(
        cfg["linear_num_key_heads"],
        cfg["linear_num_value_heads"],
        cfg["linear_key_head_dim"],
        cfg["linear_value_head_dim"],
        cfg["linear_conv_kernel_dim"],
        linear,
    )
    key_dim, value_dim = gdn.num_key_heads * gdn.key_dim, gdn.num_value_heads * gdn.value_dim
    L = cfg["num_hidden_layers"]
    projections = [
        Projection("q_proj", heads * D * 2, H, full),  # query and its output gate
        Projection("k_proj", kv * D, H, full),
        Projection("v_proj", kv * D, H, full),
        Projection("o_proj", H, heads * D, full),
        Projection("in_proj_qkv", 2 * key_dim + value_dim, H, linear),
        Projection("in_proj_z", value_dim, H, linear),
        Projection("out_proj", H, value_dim, linear),
    ]
    if moe:
        shared = cfg["shared_expert_intermediate_size"]
        projections += [
            Projection("shared_gate_up_proj", shared, H, 2 * L),
            Projection("shared_down_proj", H, shared, L),
        ]
        experts = Experts(
            cfg["num_experts"],
            cfg["num_experts_per_tok"],
            H,
            cfg["moe_intermediate_size"],
            L,
            "swiglu",
            False,
            "halves",
        )
        swiglu = (("shared_expert", shared, L),)
    else:
        inter = cfg["intermediate_size"]
        projections += [
            Projection("gate_up_proj", inter, H, 2 * L),
            Projection("down_proj", H, inter, L),
        ]
        experts = None
        swiglu = (("mlp", inter, L),)
    attention = Attention(heads, kv, D, D, full)
    return dict(
        projections=tuple(projections),
        attention=attention,
        experts=experts,
        linear_attention=gdn,
        swiglu_widths=swiglu,
    )


EXTRACTORS = {
    "deepseek_v2": deepseek_v2,
    "qwen3_moe": qwen3_moe,
    "gpt_oss": gpt_oss,
    "qwen3_5_moe_text": lambda cfg: qwen3_5(cfg, moe=True),
    "qwen3_5_text": lambda cfg: qwen3_5(cfg, moe=False),
}
"""
model_type -> extractor; the model_types setup_model builds (pithtrain/modules/training.py).
"""


def load_model(config_path: Path) -> Model:
    raw = config_path.read_bytes()
    cfg = json.loads(raw)
    model_type = cfg.get("model_type")
    if model_type not in EXTRACTORS:
        raise ValueError(
            f"{config_path}: model_type {model_type!r} has no kernel-case extractor in tools/kernel_cases/models.py"
        )
    fields = EXTRACTORS[model_type](cfg)
    fields.setdefault("swiglu_widths", ())
    return Model(
        name=config_path.parent.name,
        model_type=model_type,
        config_path=str(config_path.relative_to(REPO_ROOT)),
        config_sha256=hashlib.sha256(raw).hexdigest(),
        hidden=cfg["hidden_size"],
        vocab=cfg["vocab_size"],
        num_layers=cfg["num_hidden_layers"],
        **fields,
    )


def load_models(root: Path = CONFIG_ROOT) -> list[Model]:
    """
    Every examples/pretrain_lm/<model>/config.json, in name order.
    """
    return [load_model(path) for path in sorted(root.glob("*/config.json"))]
