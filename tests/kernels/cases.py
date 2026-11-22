"""
The kernel-shape spec: every kernel launch shape training produces, generated at collection.
- models: examples/pretrain_lm/*/config.json, one extractor per model_type
- contexts: 4k / 8k at CP 1, 16k at CP 2 / 4, 32k at CP 4 / 8, 64k at CP 8 / 16
- micro-batch 1 / 2 / 4 at CP 1, 1 under CP; EP 1 .. 16 dividing the expert count
- gpt-oss: CP 1 only (no ring attention); dense models: EP 1
- routing: balanced, log-normal, skewed, stress, BF16 tuning-bucket boundaries
- pytest tests/kernels runs every case on one GPU; CP > 1 runs in tests/operators (CP 16: one
  pytest per node with WORLD_SIZE=16 LOCAL_WORLD_SIZE=8 NODE_RANK MASTER_ADDR)
- bounds: analytic, else about 3x the worst measured (H200); growing one is a finding
- kernel PRs post pytest tests/kernels (+ tests/operators for CP code) and
  python -m benchmarks.kernels --base main; add cases, never remove them
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "examples" / "pretrain_lm"

CONTEXTS = {4096: (1,), 8192: (1,), 16384: (2, 4), 32768: (4, 8), 65536: (8, 16)}  # context: CP
MIN_LOCAL = 4096
MICRO_BATCH_SIZES = (1, 2, 4)
EXPERT_PARALLEL = (1, 2, 4, 8, 16)
SIGMA = 0.2  # log-normal routing spread, measured 0.212 on Qwen3-30B-A3B
HOT_FACTOR = 4  # skewed routing's hot expert, x the mean
FP8_GROUP_ALIGN = 128
BF16_FUSED_MAX_GROUPS = 64  # bf16_experts_supported's limit
TUNING_BUCKETS = (3, 17)  # tuning.py's MIN_BUCKET, MAX_BUCKET


# Models ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    """
    One training.Linear: y (M, n) = x (M, k) @ W.T.
    """

    name: str
    n: int
    k: int
    layers: int


@dataclass(frozen=True)
class Attention:
    """
    Attention as FlashAttention-4 sees it (after MLA decompression).
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
    num_experts: int
    top_k: int
    hidden: int
    inter: int
    layers: int
    act: str
    bias: bool
    gate_up_layout: str  # separate, halves or interleaved (gpt-oss)
    swiglu_alpha: float | None = None
    swiglu_limit: float | None = None


@dataclass(frozen=True)
class LinearAttention:
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
    name: str
    model_type: str
    hidden: int
    vocab: int
    num_layers: int
    projections: tuple[Projection, ...]
    attention: Attention
    experts: Experts | None
    linear_attention: LinearAttention | None
    swiglu_widths: tuple[tuple[str, int, int], ...] = field(
        default=()
    )  # (name, inter, layers) of the dense / shared SwiGLU MLPs


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
    # ring attention has no window or sinks, so CP 1 only
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


def load_model(config_path: Path) -> Model:
    cfg = json.loads(config_path.read_text())
    model_type = cfg.get("model_type")
    if model_type not in EXTRACTORS:
        raise ValueError(f"{config_path}: model_type {model_type!r} has no extractor in tests/kernels/cases.py")  # fmt: skip
    fields = EXTRACTORS[model_type](cfg)
    fields.setdefault("swiglu_widths", ())
    return Model(
        name=config_path.parent.name,
        model_type=model_type,
        hidden=cfg["hidden_size"],
        vocab=cfg["vocab_size"],
        num_layers=cfg["num_hidden_layers"],
        **fields,
    )


@cache
def load_models() -> tuple[Model, ...]:
    return tuple(load_model(path) for path in sorted(CONFIG_ROOT.glob("*/config.json")))


# Meshes ------------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Mesh:
    model: str
    context: int
    cp: int
    mbs: int
    ep: int

    @property
    def s_local(self) -> int:
        return self.context // self.cp

    @property
    def tokens_per_gpu(self) -> int:
        return self.mbs * self.s_local

    @property
    def label(self) -> str:
        return f"ctx{k(self.context)}-cp{self.cp}-mbs{self.mbs}-ep{self.ep}"


def k(n: int) -> str:
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def allowed_ep(model: Model) -> list[int]:
    if model.experts is None:
        return [1]
    return [ep for ep in EXPERT_PARALLEL if model.experts.num_experts % ep == 0]


def mesh_rows(model: Model) -> list[tuple[int, int, int]]:
    rows = []
    for context, cps in CONTEXTS.items():
        for cp in cps:
            if cp > 1 and not model.attention.supports_cp:
                continue
            assert cp == 1 or context // cp >= MIN_LOCAL, (context, cp)
            for mbs in MICRO_BATCH_SIZES if cp == 1 else (1,):
                rows.append((context, cp, mbs))
    return rows


def meshes_of(model: Model) -> list[Mesh]:
    return [Mesh(model.name, c, cp, mbs, ep) for c, cp, mbs in mesh_rows(model) for ep in allowed_ep(model)]  # fmt: skip


# Families ----------------------------------------------------------------------------------------

FAMILIES = OrderedDict(
    moe_experts=dict(
        doc="Routed experts of one EP rank: the gate_up and down grouped GEMMs (forward, dgrad, wgrad) in FP8 and BF16, the fused activation (+ quantization) between them, and the token movement around them.",
        kernels=[
            "fp8_quant_grouped",
            "fp8_quant_weight_grouped",
            "fp8_grouped_gemm",
            "fp8_group_indices",
            "k_grouped_fp8_gemm",
            "fp8_grouped_wgrad_empty",
            "fp8_silu_mul_fwd",
            "fp8_silu_mul_bwd",
            "fp8_silu_mul_gu_fwd",
            "fp8_silu_mul_gu_bwd",
            "fp8_clamped_swiglu_rowwise",
            "fp8_clamped_swiglu_kmajor",
            "fp8_clamped_swiglu_bwd",
            "clamped_swiglu_fwd",
            "clamped_swiglu_bwd",
            "scatter_quant",
            "group_bias_add",
            "segment_sum",
            "scatter_by_expert",
            "ep_dispatch_send_idx",
            "ep_dispatch_recv_idx",
            "bf16_expert_gate_up_fwd",
            "bf16_expert_down_fwd",
            "bf16_expert_down_dgrad",
            "bf16_expert_gate_up_dgrad",
            "bf16_expert_gate_up_wgrad",
            "bf16_expert_down_wgrad",
            "bf16_expert_group_slots",
            "bf16_expert_gather_rows",
            "bf16_expert_gather_sum",
            "bf16_expert_segment_sum",
        ],  # fmt: skip
    ),
    dense_gemm=dict(
        doc="FP8 dense linears (training.Linear with fp8=True): y = x W^T, dgrad and wgrad with the activation / gradient quantizations, at M = tokens per GPU.",
        kernels=[
            "fp8_quant_act",
            "fp8_quant_grad",
            "fp8_quant_weight",
            "fp8_gemm_fwd",
            "fp8_gemm_bwd",
        ],  # fmt: skip
    ),
    weight_quant=dict(
        doc="FP8 weight quantization and the FP8 parameter all-gather kernels, per rank's weights.",
        kernels=[
            "fp8_quant_weight",
            "fp8_quant_weight_grouped",
            "fp8_pack_transpose",
            "fp8_scales",
            "fp8_scales_multi",
            "fp8_shard_amax",
            "fp8_shard_amax_multi",
            "fp8_shard_cast",
        ],  # fmt: skip
    ),
    attention=dict(
        doc="FlashAttention-4 launches as training makes them: dense causal at CP 1, the three zigzag ring-step launches at CP > 1 (diag: causal over the local 2 blocks; front: local q over the remote front block, a strided k / v view; back: the local back-block q view over the remote 2 blocks), and packed varlen at CP 1.",
        kernels=["memoized_launch"],
    ),
    ring_attention=dict(
        doc="The CP ring operators end to end on cp GPUs (CP 16 spans two nodes): ring_attention_func / mla_ring_attention_func at the production S_local, forward and backward (FlashAttention-4 called through its own interface, no pithtrain kernel).",
        kernels=[],
    ),
    gated_delta_rule=dict(
        doc="Gated DeltaNet with the batch folded into the heads: the pinned kernels at CP 1, the CP operator (gdr_cp_* pieces around its all-gather) on cp GPUs.",
        kernels=[
            "chunk_gdr_fwd_pinned",
            "chunk_gdr_bwd_pinned",
            "gdr_cp_fwd_local",
            "gdr_cp_fwd_output",
            "gdr_cp_merge",
            "gdr_cp_bwd_local",
            "gdr_cp_bwd_output",
            "gdr_cp_bwd_states",
            "gated_delta_rule_cp_plan",
        ],  # fmt: skip
    ),
    causal_conv1d=dict(
        doc="Qwen3.5's channels-last causal depthwise conv over the mixed q / k / v channels.",
        kernels=["causal_conv1d_fwd", "causal_conv1d_bwd", "causal_conv1d_empty"],
    ),
    cross_entropy=dict(
        doc="The lm_head loss on the last pipeline stage: rows = tokens per GPU, the full vocab.",
        kernels=["cross_entropy_fwd_bwd", "log_prob_fwd", "log_prob_bwd"],
    ),
    swiglu_dense=dict(
        doc="SiLU-mul of the dense and shared-expert MLPs at M = tokens per GPU, and for DeepSeek-V2 the fused FP8 MLP (pithtrain.operators.fp8_swiglu_mlp).",
        kernels=[
            "silu_mul_fwd",
            "silu_mul_bwd",
            "fp8_gemm_nt",
            "act_scale_like",
            "fp8_channels_as_rows",
        ],
    ),
    pointwise_layers=dict(
        doc="The torch-compiled per-token layers around attention and the experts (no pithtrain kernel; Inductor-generated from each model's own modules): RMSNorm, q / k norm, RoPE, router top-k.",
        kernels=[],
    ),
)

SHAPE_INDEPENDENT = ["fp8_fused_experts_supported", "bf16_experts_supported", "gated_delta_rule_pinned_supported", "gated_delta_rule_cp_supported", "tune_kernels", "stage_compile_options"]  # fmt: skip

SM100_ONLY = ["chunk_gdr_cp_fwd_pre", "chunk_gdr_cp_fwd_post", "chunk_gdr_cp_bwd_pre", "chunk_gdr_cp_bwd_post", "chunk_gdr_cp_supported"]  # fmt: skip


def stable_seed(*parts) -> int:
    """
    A seed stable across runs and Python versions (unlike hash()).
    """
    text = "/".join(str(p) for p in parts)
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % (2**31)


def round_counts(exact: list[float], total: int, cap: int) -> list[int]:
    """
    Largest-remainder rounding to integers summing to total, each at most cap.
    """
    counts = [min(int(math.floor(x)), cap) for x in exact]
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - math.floor(exact[i])), i))
    remaining = total - sum(counts)
    for i in order * 2:
        if remaining <= 0:
            break
        if counts[i] < cap:
            counts[i] += 1
            remaining -= 1
    return counts


def routing_counts(model: Model, T: int, ep: int, routing: str, seed: int) -> list[int]:
    """
    Rows per local expert on the busiest EP rank under one routing.
    """
    experts = model.experts
    E, top_k, G = experts.num_experts, experts.top_k, experts.num_experts // ep
    group_tokens = T * ep
    slots = group_tokens * top_k
    if routing == "balanced":
        return round_counts([slots / E] * E, slots, group_tokens)[:G]
    if routing == "lognormal":
        rng = random.Random(seed)
        loads = [rng.lognormvariate(0.0, SIGMA) for _ in range(E)]
        counts = round_counts([load / sum(loads) * slots for load in loads], slots, group_tokens)
        return max((counts[r * G : (r + 1) * G] for r in range(ep)), key=sum)
    local = T * top_k
    if routing == "skewed":
        if G == 1:
            return [local]
        hot = min(HOT_FACTOR * local // G, group_tokens, local)
        if G == 2:
            return [hot, local - hot]
        return [hot, 0] + round_counts([(local - hot) / (G - 2)] * (G - 2), local - hot, group_tokens)  # fmt: skip
    if routing == "stress":
        share = group_tokens * (top_k - 1) / (E - 1)
        return [group_tokens] + round_counts([share] * (G - 1), round(share * (G - 1)), group_tokens)  # fmt: skip
    raise ValueError(routing)


def pad(count: int, align: int = FP8_GROUP_ALIGN) -> int:
    return -(-count // align) * align


def tuning_bucket(rows_per_expert: int) -> int:
    """
    The tuning bucket of a launch with this many rows per expert.
    """
    low, high = TUNING_BUCKETS
    return min(max(rows_per_expert.bit_length() - 1, low), high)


class Collector:
    """
    A family's cases by label; a label seen again only adds its mesh.
    """

    def __init__(self):
        self.cases: OrderedDict[str, dict] = OrderedDict()

    def add(self, label: str, mesh: Mesh | None, make) -> dict:
        case = self.cases.get(label)
        if case is None:
            case = self.cases[label] = make()
            case.setdefault("meshes", [])
        if mesh is not None and mesh.label not in case["meshes"]:
            case["meshes"].append(mesh.label)
        return case


def expert_axes(model: Model, T: int, ep: int, routing: str, counts: list[int]) -> dict:
    experts = model.experts
    G, rows = len(counts), sum(counts)
    return dict(
        T=T,
        ep=ep,
        G=G,
        E=experts.num_experts,
        top_k=experts.top_k,
        hidden=experts.hidden,
        inter=experts.inter,
        act=experts.act,
        bias=experts.bias,
        gate_up_layout=experts.gate_up_layout,
        swiglu_alpha=experts.swiglu_alpha,
        swiglu_limit=experts.swiglu_limit,
        routing=routing,
        rows=rows,
        fp8_rows=sum(pad(c) for c in counts),
        avg_rows=rows // G,
        max_rows=max(counts),
        empty_groups=sum(1 for c in counts if c == 0),
        bf16_fused=G <= BF16_FUSED_MAX_GROUPS,
        group_sizes=counts,
    )


def moe_cases(model: Model, meshes: list[Mesh]) -> OrderedDict:
    """
    Each (T, EP) under every routing, plus both edges of each BF16 tuning bucket reached.
    """
    experts = model.experts
    collect = Collector()
    pairs = sorted({(m.tokens_per_gpu, m.ep) for m in meshes})
    largest_ep = {T: max(ep for t, ep in pairs if t == T) for T, _ in pairs}
    flops_per_row = 2 * experts.hidden * 3 * experts.inter  # gate_up (2I) + down (I)
    for T, ep in pairs:
        routings = ["balanced", "lognormal", "skewed"] + (["stress"] if ep == largest_ep[T] > 1 else [])  # fmt: skip
        for routing in routings:
            counts = routing_counts(model, T, ep, routing, stable_seed(model.name, T, ep, routing))

            def make(T=T, ep=ep, routing=routing, counts=counts):
                return dict(
                    axes=expert_axes(model, T, ep, routing, counts),
                    weight=dict(flops_fwd=flops_per_row * sum(counts), layers=experts.layers),
                )

            for mesh in meshes:
                if (mesh.tokens_per_gpu, mesh.ep) == (T, ep):
                    collect.add(f"T{k(T)}-ep{ep}-{routing}", mesh, make)
    realistic = [c["axes"] for c in collect.cases.values() if c["axes"]["routing"] != "stress"]
    for ep in sorted({a["ep"] for a in realistic}):
        G = experts.num_experts // ep
        if G > BF16_FUSED_MAX_GROUPS:
            continue
        for bucket in sorted({tuning_bucket(a["avg_rows"]) for a in realistic if a["ep"] == ep}):
            edges = [2**bucket] + ([2 ** (bucket + 1) - 1] if bucket < TUNING_BUCKETS[1] else [])
            for avg in edges:
                counts = [avg] * G

                def make(ep=ep, counts=counts):
                    rows = sum(counts)
                    return dict(
                        axes=expert_axes(model, rows // experts.top_k, ep, "boundary", counts),
                        weight=dict(flops_fwd=flops_per_row * rows, layers=experts.layers),
                    )

                collect.add(f"ep{ep}-rows{avg}-boundary", None, make)
    return collect.cases


def dense_gemm_cases(model: Model, meshes: list[Mesh]) -> OrderedDict:
    collect = Collector()
    Ts = sorted({m.tokens_per_gpu for m in meshes})
    for proj in model.projections:
        if proj.layers == 0:
            continue
        for T in Ts:

            def make(T=T, proj=proj):
                return dict(
                    axes=dict(projection=proj.name, M=T, N=proj.n, K=proj.k),
                    weight=dict(flops_fwd=2 * T * proj.n * proj.k, layers=proj.layers),
                )

            for mesh in meshes:
                if mesh.tokens_per_gpu == T:
                    collect.add(f"{proj.name}-M{k(T)}", mesh, make)
    return collect.cases


def weight_quant_cases(model: Model, meshes: list[Mesh]) -> OrderedDict:
    collect = Collector()
    for proj in model.projections:
        if proj.layers:
            make = lambda proj=proj: dict(axes=dict(weight_name=proj.name, G=0, N=proj.n, K=proj.k))  # fmt: skip
            for mesh in meshes:
                collect.add(proj.name, mesh, make)
    if model.experts is not None:
        experts = model.experts
        for ep in sorted({m.ep for m in meshes}):
            G = experts.num_experts // ep
            for name, N, K in (("gate_up", 2 * experts.inter, experts.hidden), ("down", experts.hidden, experts.inter)):  # fmt: skip
                make = lambda G=G, name=name, N=N, K=K, ep=ep: dict(axes=dict(weight_name=name, G=G, ep=ep, N=N, K=K))  # fmt: skip
                for mesh in meshes:
                    if mesh.ep == ep:
                        collect.add(f"experts-{name}-ep{ep}", mesh, make)
    return collect.cases


def attention_heads(model: Model) -> dict:
    att = model.attention
    return dict(Hq=att.num_heads, Hkv=att.num_kv_heads, D=att.head_dim, Dv=att.v_head_dim, mla=att.mla_kv_lora_rank is not None)  # fmt: skip


def attention_cases(model: Model, meshes: list[Mesh]) -> OrderedDict:
    att = model.attention
    collect = Collector()
    heads = attention_heads(model)
    variants = [(None, att.sinks)] if att.window is None else [(None, att.sinks), (att.window - 1, att.sinks)]  # fmt: skip
    layers = (
        att.layers if att.window is None else att.layers // 2
    )  # gpt-oss alternates full / sliding
    for mesh in meshes:
        B, S = mesh.mbs, mesh.s_local
        if mesh.cp == 1:
            launches = [("dense", S, S, True)]
        else:
            launches = [("ring_diag", S, S, True), ("ring_front", S, S // 2, False), ("ring_back", S // 2, S, False)]  # fmt: skip
        for kind, Sq, Skv, causal in launches:
            for window, sinks in variants:

                def make(kind=kind, Sq=Sq, Skv=Skv, causal=causal, window=window, sinks=sinks, B=B):
                    pairs = Sq * Skv // (2 if causal else 1)
                    flops = 2 * B * att.num_heads * pairs * (att.head_dim + att.v_head_dim)
                    axes = dict(kind=kind, B=B, Sq=Sq, Skv=Skv, causal=causal, window=window, sinks=sinks, **heads)  # fmt: skip
                    return dict(axes=axes, weight=dict(flops_fwd=flops, layers=layers))

                label = f"{kind}-B{B}-q{k(Sq)}-kv{k(Skv)}" + (f"-w{window}" if window is not None else "")  # fmt: skip
                collect.add(label, mesh, make)
    # packed documents (the SFT path), log-normal lengths around context / 6
    for context in sorted({m.context for m in meshes if m.cp == 1}):
        rng = random.Random(stable_seed(model.name, "varlen", context))
        cu, left = [0], context
        while left > 0:
            length = max(128, min(left, int(rng.lognormvariate(math.log(context / 6), 1.0))))
            length = left if left - length < 128 else length
            cu.append(cu[-1] + length)
            left -= length
        for window, sinks in variants:

            def make(context=context, cu=cu, window=window, sinks=sinks):
                pairs = sum((b - a) * (b - a) // 2 for a, b in zip(cu, cu[1:]))
                flops = 2 * att.num_heads * pairs * (att.head_dim + att.v_head_dim)
                axes = dict(kind="varlen", B=1, Sq=context, Skv=context, causal=True, window=window, sinks=sinks, cu_seqlens=cu, **heads)  # fmt: skip
                return dict(axes=axes, weight=dict(flops_fwd=flops, layers=layers))

            collect.add(f"varlen-{k(context)}" + (f"-w{window}" if window is not None else ""), None, make)  # fmt: skip
    return collect.cases


def ring_cases(model: Model, meshes: list[Mesh]) -> OrderedDict:
    collect = Collector()
    heads = attention_heads(model)
    for mesh in meshes:
        if mesh.cp > 1:
            make = lambda mesh=mesh: dict(axes=dict(cp=mesh.cp, B=mesh.mbs, S_local=mesh.s_local, seq=mesh.context, **heads))  # fmt: skip
            collect.add(f"cp{mesh.cp}-B{mesh.mbs}-S{k(mesh.s_local)}", mesh, make)
    return collect.cases


def gdn_cases(model: Model, meshes: list[Mesh]) -> tuple[OrderedDict, OrderedDict]:
    gdn = model.linear_attention
    rule, conv = Collector(), Collector()
    for mesh in meshes:
        B, T_local, cp = mesh.mbs, mesh.s_local, mesh.cp
        make_rule = lambda B=B, T_local=T_local, cp=cp: dict(axes=dict(cp=cp, mbs=B, T_local=T_local, H=gdn.num_key_heads * B, HV=gdn.num_value_heads * B, K=gdn.key_dim, V=gdn.value_dim), weight=dict(layers=gdn.layers))  # fmt: skip
        make_conv = lambda B=B, T_local=T_local, cp=cp: dict(axes=dict(cp=cp, B=B, T_local=T_local, C=gdn.conv_channels, width=gdn.conv_kernel), weight=dict(layers=gdn.layers))  # fmt: skip
        label = f"cp{cp}-mbs{B}-T{k(T_local)}"
        rule.add(label, mesh, make_rule)
        conv.add(label, mesh, make_conv)
    return rule.cases, conv.cases


def cross_entropy_cases(model: Model, meshes: list[Mesh]) -> OrderedDict:
    collect = Collector()
    for mesh in meshes:
        T = mesh.tokens_per_gpu
        make = lambda T=T: dict(axes=dict(rows=T, vocab=model.vocab, hidden=model.hidden))  # fmt: skip
        collect.add(f"rows{k(T)}", mesh, make)
    return collect.cases


def swiglu_cases(model: Model, meshes: list[Mesh]) -> OrderedDict:
    collect = Collector()
    for name, inter, layers in model.swiglu_widths:
        if layers == 0:
            continue
        for mesh in meshes:
            T = mesh.tokens_per_gpu
            # DeepSeekV2MLP runs the fused FP8 operator
            make = lambda T=T, name=name, inter=inter, layers=layers: dict(axes=dict(mlp=name, M=T, inter=inter, hidden=model.hidden, fp8_swiglu_mlp=model.model_type == "deepseek_v2"), weight=dict(layers=layers))  # fmt: skip
            collect.add(f"{name}-M{k(T)}", mesh, make)
    return collect.cases


def pointwise_cases(model: Model, meshes: list[Mesh]) -> OrderedDict:
    collect = Collector()
    att = model.attention
    for mesh in meshes:
        T = mesh.tokens_per_gpu

        def make(T=T, mesh=mesh):
            axes = dict(T=T, B=mesh.mbs, S_local=mesh.s_local, hidden=model.hidden, Hq=att.num_heads, Hkv=att.num_kv_heads, D=att.head_dim)  # fmt: skip
            if model.experts is not None:
                axes.update(E=model.experts.num_experts, top_k=model.experts.top_k)
            return dict(axes=axes)

        collect.add(f"T{k(T)}-B{mesh.mbs}", mesh, make)
    return collect.cases


def families_of(model: Model) -> list[str]:
    families = ["dense_gemm", "weight_quant"]
    families += ["moe_experts"] if model.experts is not None else []
    families += ["attention"] + (["ring_attention"] if model.attention.supports_cp else [])
    families += ["gated_delta_rule", "causal_conv1d"] if model.linear_attention is not None else []
    families += ["cross_entropy"] + (["swiglu_dense"] if model.swiglu_widths else [])
    return families + ["pointwise_layers"]


def model_cases(model: Model) -> list[dict]:
    meshes = meshes_of(model)
    per_family = OrderedDict()
    for family in families_of(model):
        if family == "gated_delta_rule":
            per_family["gated_delta_rule"], per_family["causal_conv1d"] = gdn_cases(model, meshes)
        elif family != "causal_conv1d":
            per_family[family] = GENERATORS[family](model, meshes)
    cases = []
    for family, collected in per_family.items():
        for label, case in collected.items():
            row = dict(id=f"{family}/{model.name}/{label}", family=family, model=model.name, axes=case["axes"], meshes=case["meshes"])  # fmt: skip
            if "weight" in case:
                row["weight"] = case["weight"]
            cases.append(row)
    return cases


GENERATORS = dict(
    moe_experts=moe_cases,
    dense_gemm=dense_gemm_cases,
    weight_quant=weight_quant_cases,
    attention=attention_cases,
    ring_attention=ring_cases,
    cross_entropy=cross_entropy_cases,
    swiglu_dense=swiglu_cases,
    pointwise_layers=pointwise_cases,
)


@cache
def all_cases() -> tuple[dict, ...]:
    return tuple(case for model in load_models() for case in model_cases(model))


def select(family: str, model: str | None = None) -> list[dict]:
    return [c for c in all_cases() if c["family"] == family and model in (None, c["model"])]


def params(family: str, keep=None) -> list:
    """
    The family's cases as pytest params, id = the case id.
    """
    import pytest

    return [pytest.param(c, id=c["id"]) for c in select(family) if keep is None or keep(c)]


def seed_of(case: dict) -> int:
    return stable_seed(case["id"])


def main() -> int:
    """
    Per model its meshes, EP degrees and case counts; then per family.
    """
    for model in load_models():
        meshes = ", ".join(f"{k(c)}/cp{cp}x{mbs}" for c, cp, mbs in mesh_rows(model))
        families = Counter(c["family"] for c in model_cases(model))
        print(f"{model.name}: meshes {meshes}; EP {', '.join(map(str, allowed_ep(model)))}")
        print("  " + ", ".join(f"{family} {n}" for family, n in families.items()))
    counts = Counter(c["family"] for c in all_cases())
    for family in FAMILIES:
        print(f"{family}: {counts[family]}")
    print(f"total: {len(all_cases())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
