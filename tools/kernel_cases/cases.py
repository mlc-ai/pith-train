"""
Derives the per-kernel case lists from the models (models.py) and the meshes (meshes.py).

Each family below names the pithtrain.kernels entry points its cases drive, the axes a case
carries, its reference and its tolerance. A case is one distinct kernel launch shape; every mesh
that produces it is listed in its "meshes", so a case is traceable to the training runs that need
it, and its "weight" (FLOPs or bytes per micro-batch summed over the layers that launch it) says how
much of a step it accounts for.

Tiers:

- smoke: one or two cases per (family, model), small enough for a pre-merge run on one GPU;
- spec: every distinct shape the meshes produce (includes smoke);
- stress: shapes past what balanced training reaches (all of an EP group's tokens on one expert).

Expert routing is materialized here, not in the consumers, so every consumer (tests, benchmarks,
kernel-design agents) runs the same per-expert row counts. random.Random is used because its
streams are stable across Python versions; the counts are written out, so nothing re-draws them.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict

from .meshes import TOKENS_PER_GPU, Mesh, matrix_meshes, shipped_meshes
from .models import Model, load_models

SIGMA = 0.2
"""
Log-normal spread of the per-expert token shares (pithtrain/modules/imbalance.py, measured 0.212 on
Qwen3-30B-A3B).
"""

HOT_FACTOR = 4
"""
The skewed routing's hot expert: HOT_FACTOR x the mean rows per local expert (p99 of the measured
spread is about 2.1x; 4x is the review's stress margin), next to one empty expert.
"""

FP8_GROUP_ALIGN = 128
"""
The FP8 expert path pads every group to a multiple of 128 rows (fp8_quant_grouped's contract).
"""

BF16_FUSED_MAX_GROUPS = 64
"""
bf16_experts_supported(num_groups): the fused BF16 experts take at most 64 local experts; above it
training runs F.grouped_mm.
"""

FAMILIES = OrderedDict(
    moe_experts=dict(
        doc="Routed experts of one EP rank: gate_up and down grouped GEMMs (fwd, dgrad, wgrad) in FP8 and BF16, the fused activation (+ quantization) between them, and the token movement around them.",
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
        ],
        axes="T (tokens per GPU), ep, G = E / ep local experts, top_k, hidden, inter, act, bias, routing, group_sizes (rows per local expert; the FP8 path pads each to a multiple of 128), bf16_fused (G <= 64)",
        reference="per-group float64 (BF16) or fp32 of the exactly dequantized FP8 operands, written in the test",
        tolerance="FP8: calc_diff and a per-element bound against the exact-dequant reference; BF16: half a bf16 ulp + 1e-5 x sum |products| (tests/spec/test_spec_moe_experts.py)",
    ),
    dense_gemm=dict(
        doc="FP8 dense linears (training.Linear with fp8=True): y = x W^T, dgrad and wgrad, with the activation / gradient quantizations, at M = tokens per GPU.",
        kernels=[
            "fp8_quant_act",
            "fp8_quant_grad",
            "fp8_quant_weight",
            "fp8_gemm_fwd",
            "fp8_gemm_bwd",
            "fp8_gemm_nt",
            "act_scale_like",
            "fp8_channels_as_rows",
        ],
        axes="M (= tokens per GPU), N, K, projection name",
        reference="fp32 matmul of the exactly dequantized operands; quantizations bit-exact against the E8M0 recipe",
        tolerance="calc_diff and per-element bound vs exact dequant; bf16-truth calc_diff (tests/spec/test_spec_dense_gemm.py)",
    ),
    weight_quant=dict(
        doc="FP8 weight quantization and the FP8 parameter all-gather, per rank's weights.",
        kernels=[
            "fp8_quant_weight",
            "fp8_quant_weight_grouped",
            "fp8_pack_transpose",
            "fp8_scales",
            "fp8_scales_multi",
            "fp8_shard_amax",
            "fp8_shard_amax_multi",
            "fp8_shard_cast",
        ],
        axes="G (local experts, 0 = dense), N, K",
        reference="bit-exact 128x128 E8M0 recipe",
        tolerance="exact",
    ),
    attention=dict(
        doc="FlashAttention-4 launches as training makes them: dense causal at CP 1 and the three zigzag ring-step launches at CP > 1 (step 0 causal over the local 2 blocks; front: all local q over the remote front block, a strided k / v view; back: the local back-block q view over the remote 2 blocks), plus packed varlen at CP 1.",
        kernels=["memoized_launch"],
        axes="kind (dense | ring_diag | ring_front | ring_back | varlen), B, Sq, Skv, Hq, Hkv, D, Dv, causal, window, sinks, cu_seqlens",
        reference="fp32 attention computed in query blocks, with a hand-written fp32 backward (not FA4), random dout",
        tolerance="relative L2 error per output (tests/spec/test_spec_attention.py)",
    ),
    ring_attention=dict(
        doc="The CP ring operators end to end on cp GPUs: ring_attention_func / mla_ring_attention_func at the production S_local, fwd and bwd.",
        kernels=["memoized_launch"],
        axes="cp, B, S_local, Hq, Hkv, D, Dv, mla",
        reference="fp32 full causal attention of the gathered sequence, in query blocks (not FA4)",
        tolerance="relative L2 error per output (tests/spec/test_spec_ring_attention.py, multigpu)",
    ),
    gated_delta_rule=dict(
        doc="Gated DeltaNet at the folded head counts (batch folded into the heads): the pinned kernels at CP 1, the CP pieces at CP > 1.",
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
            "chunk_gdr_cp_fwd_pre",
            "chunk_gdr_cp_fwd_post",
            "chunk_gdr_cp_bwd_pre",
            "chunk_gdr_cp_bwd_post",
        ],
        axes="cp, mbs, T_local (= seq / cp), H = key heads x mbs, HV = value heads x mbs, K = V = 128",
        reference="the fp64 chunked recurrence (tests/kernels/test_gated_delta_rule_pinned.py)",
        tolerance="calc_diff per output and max-abs vs the reference's largest magnitude",
    ),
    causal_conv1d=dict(
        doc="Qwen3.5's channels-last causal depthwise conv over the mixed q / k / v channels.",
        kernels=["causal_conv1d_fwd", "causal_conv1d_bwd", "causal_conv1d_empty"],
        axes="B (= mbs), T_local, C, width, cp (conv state from the previous rank when cp > 1)",
        reference="fp32 F.conv1d",
        tolerance="half a bf16 ulp of the fp32 result plus fp32 accumulation slack",
    ),
    cross_entropy=dict(
        doc="The lm_head loss on the last pipeline stage: rows = tokens per GPU, the full vocab.",
        kernels=["cross_entropy_fwd_bwd", "log_prob_fwd", "log_prob_bwd"],
        axes="rows, vocab",
        reference="float64 log-softmax, gradient through the CrossEntropy autograd op with a grad_output that is not 1",
        tolerance="loss rel 1e-6; gradient within one bf16 rounding of the exact value",
    ),
    swiglu_dense=dict(
        doc="SiLU-mul of the dense and shared-expert MLPs at M = tokens per GPU (eager silu_mul kernels; FP8 runs the fused fp8_silu_mul with one group).",
        kernels=["silu_mul_fwd", "silu_mul_bwd"],
        axes="M, inter",
        reference="float64 silu(g) * u",
        tolerance="one bf16 rounding",
    ),
    pointwise_layers=dict(
        doc="The torch-compiled per-token layers around attention and the experts (no pithtrain kernel; Inductor-generated): RMSNorm, q/k norm, RoPE (partial for Qwen3.5), router top-k.",
        kernels=[],
        axes="T, hidden, heads x head_dim, rotary_dim, E, top_k",
        reference="float64 eager PyTorch",
        tolerance="per output, see the test",
    ),
)

SHAPE_INDEPENDENT = [
    "fp8_fused_experts_supported",
    "bf16_experts_supported",
    "gated_delta_rule_pinned_supported",
    "gated_delta_rule_cp_supported",
    "chunk_gdr_cp_supported",
    "tune_kernels",
    "stage_compile_options",
]
"""
pithtrain.kernels names with no launch shape: capability queries, the tuner, compile options.
"""


def round_counts(exact: list[float], total: int, cap: int) -> list[int]:
    """
    Largest-remainder rounding of exact to integers summing to total, each at most cap.
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
    Rows per local expert on the busiest EP rank, for T tokens per GPU on each of the ep ranks of
    the EP group, each token picking top_k distinct experts:

    - balanced: every expert the same share;
    - lognormal: per-expert shares log-normal with SIGMA over all E experts (the benchmark router's
      imbalance), the rank whose experts carry the most rows;
    - skewed: one local expert at HOT_FACTOR x the local mean and one empty, the rest even;
    - stress: every token of the EP group routes to local expert 0 (its rows = ep * T), the other
      top_k - 1 picks spread over the remaining experts.

    (The route-boundary cases of moe_cases set their counts directly.)
    """
    experts = model.experts
    E, k, G = experts.num_experts, experts.top_k, experts.num_experts // ep
    group_tokens = T * ep
    slots = group_tokens * k
    if routing == "balanced":
        counts = round_counts([slots / E] * E, slots, group_tokens)
        return counts[:G]
    if routing == "lognormal":
        rng = random.Random(seed)
        loads = [rng.lognormvariate(0.0, SIGMA) for _ in range(E)]
        exact = [load / sum(loads) * slots for load in loads]
        counts = round_counts(exact, slots, group_tokens)
        ranks = [counts[r * G : (r + 1) * G] for r in range(ep)]
        return max(ranks, key=sum)
    local = T * k
    if routing == "skewed":
        if G == 1:
            return [local]
        hot = min(HOT_FACTOR * local // G, group_tokens, local)
        rest = local - hot
        if G == 2:
            return [hot, rest]
        return [hot, 0] + round_counts([rest / (G - 2)] * (G - 2), rest, group_tokens)
    if routing == "stress":
        others = group_tokens * (k - 1)
        share = others / (E - 1)
        return [group_tokens] + round_counts(
            [share] * (G - 1), round(share * (G - 1)), group_tokens
        )
    raise ValueError(routing)


def pad(count: int, align: int = FP8_GROUP_ALIGN) -> int:
    return -(-count // align) * align


class Collector:
    """
    Cases of one family keyed by their shape; a key seen again only adds its mesh.
    """

    def __init__(self, family: str):
        self.family = family
        self.cases: OrderedDict[tuple, dict] = OrderedDict()

    def add(self, key: tuple, mesh: Mesh | None, make) -> dict:
        case = self.cases.get(key)
        if case is None:
            case = make()
            case.setdefault("meshes", [])
            self.cases[key] = case
        if mesh is not None and mesh.label not in case["meshes"]:
            case["meshes"].append(mesh.label)
        return case


def moe_cases(model: Model, meshes: list[Mesh]) -> list[dict]:
    experts = model.experts
    collect = Collector("moe_experts")
    by_T_ep = sorted({(m.tokens_per_gpu, m.ep) for m in meshes})
    largest_ep = {}
    for T, ep in by_T_ep:
        largest_ep[T] = max(largest_ep.get(T, 1), ep)
    for T, ep in by_T_ep:
        G = experts.num_experts // ep
        routings = ["balanced", "lognormal", "skewed"] + (
            ["stress"] if ep == largest_ep[T] and ep > 1 else []
        )
        for routing in routings:
            seed = hash_seed(model.name, T, ep, routing)
            counts = routing_counts(model, T, ep, routing, seed)
            rows, fp8_rows = sum(counts), sum(pad(c) for c in counts)
            tier = "stress" if routing == "stress" else "spec"

            def make(
                T=T,
                ep=ep,
                G=G,
                routing=routing,
                counts=counts,
                rows=rows,
                fp8_rows=fp8_rows,
                tier=tier,
            ):
                flops = 2 * rows * experts.hidden * 3 * experts.inter  # gate_up (2I) + down (I)
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
                    routing=routing,
                    rows=rows,
                    fp8_rows=fp8_rows,
                    avg_rows=rows // G,
                    max_rows=max(counts),
                    empty_groups=sum(1 for c in counts if c == 0),
                    bf16_fused=G <= BF16_FUSED_MAX_GROUPS,
                    group_sizes=counts,
                    tier=tier,
                    weight=dict(flops_fwd=flops, layers=experts.layers),
                )

            for mesh in meshes:
                if (mesh.tokens_per_gpu, mesh.ep) == (T, ep):
                    collect.add((T, ep, routing), mesh, make)
    # Route boundaries at the EP 8 expert count, on both sides of every threshold the SM100 pickers
    # switch on (fp8_grouped_gemm: 512 rows per expert; k_grouped_fp8_gemm: 2048; bf16 experts:
    # swap-AB at <= 448 / 512, materialized gate_up at >= 1024), every expert the same rows. These
    # are synthetic (no mesh), so a route training does not reach today is still verified.
    G = experts.num_experts // max(ep for ep in (1, 2, 4, 8) if experts.num_experts % ep == 0)
    for avg in BOUNDARY_ROWS:

        def make(avg=avg, G=G):
            rows = avg * G
            return dict(
                T=rows // experts.top_k,
                ep=experts.num_experts // G,
                G=G,
                E=experts.num_experts,
                top_k=experts.top_k,
                hidden=experts.hidden,
                inter=experts.inter,
                act=experts.act,
                bias=experts.bias,
                routing="boundary",
                rows=rows,
                fp8_rows=pad(avg) * G,
                avg_rows=avg,
                max_rows=avg,
                empty_groups=0,
                bf16_fused=G <= BF16_FUSED_MAX_GROUPS,
                group_sizes=[avg] * G,
                tier="spec",
            )

        collect.add(("boundary", avg), None, make)
    smoke_T = min(T for T, _ in by_T_ep if T >= 4096)
    smoke_ep = max(ep for T, ep in by_T_ep if T == smoke_T and ep <= 8)
    collect.cases[(smoke_T, smoke_ep, "lognormal")]["tier"] = "smoke"
    return list(collect.cases.values())


BOUNDARY_ROWS = (384, 448, 512, 640, 1024, 1920, 2048, 2176)
"""
Rows per expert of the route-boundary cases, either side of 448, 512, 1024 and 2048 (448 is not
128-aligned, so its FP8 groups pad to 512).
"""


def hash_seed(*parts) -> int:
    """
    A seed from the case's identity, stable across runs (unlike hash()).
    """
    text = "/".join(str(p) for p in parts)
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % (2**31)


def dense_gemm_cases(model: Model, meshes: list[Mesh]) -> list[dict]:
    collect = Collector("dense_gemm")
    Ts = sorted({m.tokens_per_gpu for m in meshes})
    for proj in model.projections:
        if proj.layers == 0:
            continue
        for T in Ts:

            def make(T=T, proj=proj):
                return dict(
                    projection=proj.name,
                    M=T,
                    N=proj.n,
                    K=proj.k,
                    tier="spec",
                    weight=dict(flops_fwd=2 * T * proj.n * proj.k, layers=proj.layers),
                )

            for mesh in meshes:
                if mesh.tokens_per_gpu == T:
                    collect.add((proj.name, T), mesh, make)
    smoke_T = min(T for T in Ts if T >= 4096)
    for (name, T), case in collect.cases.items():
        if T == smoke_T:
            case["tier"] = "smoke"
    return list(collect.cases.values())


def weight_quant_cases(model: Model, meshes: list[Mesh]) -> list[dict]:
    collect = Collector("weight_quant")
    for proj in model.projections:
        if proj.layers:
            collect.add(
                ("dense", proj.name),
                None,
                lambda proj=proj: dict(weight_name=proj.name, G=0, N=proj.n, K=proj.k, tier="spec"),
            )
    if model.experts is not None:
        experts = model.experts
        for ep in sorted({m.ep for m in meshes}):
            G = experts.num_experts // ep
            for name, N, K in (
                ("gate_up", 2 * experts.inter, experts.hidden),
                ("down", experts.hidden, experts.inter),
            ):

                def make(G=G, name=name, N=N, K=K, ep=ep):
                    return dict(weight_name=name, G=G, ep=ep, N=N, K=K, tier="spec")

                for mesh in meshes:
                    if mesh.ep == ep:
                        collect.add(("experts", name, ep), mesh, make)
    first = next(iter(collect.cases.values()))
    first["tier"] = "smoke"
    return list(collect.cases.values())


def attention_heads(model: Model) -> dict:
    att = model.attention
    return dict(
        Hq=att.num_heads,
        Hkv=att.num_kv_heads,
        D=att.head_dim,
        Dv=att.v_head_dim,
        mla=att.mla_kv_lora_rank is not None,
    )


def attention_cases(model: Model, meshes: list[Mesh]) -> list[dict]:
    att = model.attention
    collect = Collector("attention")
    heads = attention_heads(model)
    variants = [(None, att.sinks)]
    if att.window is not None:
        variants = [(None, att.sinks), (att.window - 1, att.sinks)]  # full and sliding layers
    for mesh in meshes:
        B, S = mesh.mbs, mesh.s_local
        launches = []
        if mesh.cp == 1:
            launches.append(("dense", S, S, True))
        else:
            block = S // 2
            launches += [
                ("ring_diag", S, S, True),
                ("ring_front", S, block, False),
                ("ring_back", block, S, False),
            ]
        for kind, Sq, Skv, causal in launches:
            for window, sinks in variants:
                layers = att.layers if att.window is None else att.layers // 2

                def make(
                    kind=kind,
                    Sq=Sq,
                    Skv=Skv,
                    causal=causal,
                    window=window,
                    sinks=sinks,
                    layers=layers,
                    B=B,
                ):
                    pairs = Sq * Skv // (2 if causal else 1)
                    flops = 2 * B * att.num_heads * pairs * (att.head_dim + att.v_head_dim)
                    return dict(
                        kind=kind,
                        B=B,
                        Sq=Sq,
                        Skv=Skv,
                        causal=causal,
                        window=window,
                        sinks=sinks,
                        **heads,
                        tier="spec",
                        weight=dict(flops_fwd=flops, layers=layers),
                    )

                collect.add((kind, B, Sq, Skv, window), mesh, make)
    # packed documents at CP 1 (the SFT path; pretraining runs dense): documents of log-normal
    # length around S / 6, at least 128 tokens
    for T in TOKENS_PER_GPU:
        rng = random.Random(hash_seed(model.name, "varlen", T))
        lengths, left = [], T
        while left > 0:
            length = max(128, min(left, int(rng.lognormvariate(math.log(T / 6), 1.0))))
            if left - length < 128:
                length = left
            lengths.append(length)
            left -= length
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        for window, sinks in variants:

            def make(T=T, cu=cu, window=window, sinks=sinks):
                pairs = sum((b - a) * (b - a) // 2 for a, b in zip(cu, cu[1:]))
                return dict(
                    kind="varlen",
                    B=1,
                    Sq=T,
                    Skv=T,
                    causal=True,
                    window=window,
                    sinks=sinks,
                    cu_seqlens=cu,
                    **heads,
                    tier="spec",
                    weight=dict(
                        flops_fwd=2 * att.num_heads * pairs * (att.head_dim + att.v_head_dim),
                        layers=att.layers,
                    ),
                )

            collect.add(("varlen", T, window), None, make)
    for case in collect.cases.values():
        local = max(case["Sq"], case["Skv"])
        if case["kind"] == "dense" and (case["B"], local) == (1, 4096):
            case["tier"] = "smoke"
        if case["kind"].startswith("ring") and (case["B"], local) == (1, 4096):
            case["tier"] = "smoke"
    return list(collect.cases.values())


def ring_cases(model: Model, meshes: list[Mesh]) -> list[dict]:
    collect = Collector("ring_attention")
    heads = attention_heads(model)
    for mesh in meshes:
        if mesh.cp == 1:
            continue
        make = lambda mesh=mesh: dict(
            cp=mesh.cp, B=mesh.mbs, S_local=mesh.s_local, seq=mesh.seq, **heads, tier="spec"
        )
        collect.add((mesh.cp, mesh.mbs, mesh.s_local), mesh, make)
    for (cp, B, S_local), case in collect.cases.items():
        if B == 1 and S_local == 4096 and cp in (2, 8):
            case["tier"] = "smoke"
    return list(collect.cases.values())


def gdn_cases(model: Model, meshes: list[Mesh]) -> tuple[list[dict], list[dict]]:
    gdn = model.linear_attention
    rule, conv = Collector("gated_delta_rule"), Collector("causal_conv1d")
    for mesh in meshes:
        B, T_local, cp = mesh.mbs, mesh.s_local, mesh.cp

        def make_rule(B=B, T_local=T_local, cp=cp):
            return dict(
                cp=cp,
                mbs=B,
                T_local=T_local,
                H=gdn.num_key_heads * B,
                HV=gdn.num_value_heads * B,
                K=gdn.key_dim,
                V=gdn.value_dim,
                tier="spec",
                weight=dict(layers=gdn.layers),
            )

        def make_conv(B=B, T_local=T_local, cp=cp):
            return dict(
                cp=cp,
                B=B,
                T_local=T_local,
                C=gdn.conv_channels,
                width=gdn.conv_kernel,
                tier="spec",
                weight=dict(layers=gdn.layers),
            )

        rule.add((cp, B, T_local), mesh, make_rule)
        conv.add((cp, B, T_local), mesh, make_conv)
    for collect in (rule, conv):
        for (cp, B, T_local), case in collect.cases.items():
            if B == 1 and T_local == 4096 and cp in (1, 2):
                case["tier"] = "smoke"
    return list(rule.cases.values()), list(conv.cases.values())


def cross_entropy_cases(model: Model, meshes: list[Mesh]) -> list[dict]:
    collect = Collector("cross_entropy")
    for mesh in meshes:
        T = mesh.tokens_per_gpu
        make = lambda T=T: dict(
            rows=T, vocab=model.vocab, hidden=model.hidden, tier="smoke" if T == 4096 else "spec"
        )
        collect.add((T,), mesh, make)
    return list(collect.cases.values())


def swiglu_cases(model: Model, meshes: list[Mesh]) -> list[dict]:
    collect = Collector("swiglu_dense")
    for name, inter, layers in model.swiglu_widths:
        if layers == 0:
            continue
        for mesh in meshes:
            T = mesh.tokens_per_gpu
            make = lambda T=T, name=name, inter=inter, layers=layers: dict(
                mlp=name,
                M=T,
                inter=inter,
                tier="smoke" if T == 4096 else "spec",
                weight=dict(layers=layers),
            )
            collect.add((name, T), mesh, make)
    return list(collect.cases.values())


def pointwise_cases(model: Model, meshes: list[Mesh]) -> list[dict]:
    collect = Collector("pointwise_layers")
    att = model.attention
    for mesh in meshes:
        T = mesh.tokens_per_gpu

        def make(T=T, mesh=mesh):
            case = dict(
                T=T,
                B=mesh.mbs,
                S_local=mesh.s_local,
                hidden=model.hidden,
                Hq=att.num_heads,
                Hkv=att.num_kv_heads,
                D=att.head_dim,
                tier="smoke" if T == 4096 else "spec",
            )
            if model.experts is not None:
                case.update(E=model.experts.num_experts, top_k=model.experts.top_k)
            return case

        collect.add((T, mesh.mbs), mesh, make)
    return list(collect.cases.values())


def model_meshes(model: Model, shipped: list[Mesh]) -> list[Mesh]:
    return matrix_meshes(model) + [mesh for mesh in shipped if mesh.model == model.name]


def generate() -> dict:
    """
    The whole spec: inputs (model config hashes, the mesh matrix, the shipped meshes), the family
    table and every case, each with a stable id.
    """
    from . import meshes as mesh_module

    models = load_models()
    by_name = {model.name: model for model in models}
    shipped = shipped_meshes(by_name)
    cases = []
    for model in models:
        meshes = model_meshes(model, shipped)
        per_family = OrderedDict()
        per_family["dense_gemm"] = dense_gemm_cases(model, meshes)
        per_family["weight_quant"] = weight_quant_cases(model, meshes)
        if model.experts is not None:
            per_family["moe_experts"] = moe_cases(model, meshes)
        per_family["attention"] = attention_cases(model, meshes)
        if model.attention.supports_cp:
            per_family["ring_attention"] = ring_cases(model, meshes)
        if model.linear_attention is not None:
            per_family["gated_delta_rule"], per_family["causal_conv1d"] = gdn_cases(model, meshes)
        per_family["cross_entropy"] = cross_entropy_cases(model, meshes)
        if model.swiglu_widths:
            per_family["swiglu_dense"] = swiglu_cases(model, meshes)
        per_family["pointwise_layers"] = pointwise_cases(model, meshes)
        for family, family_cases in per_family.items():
            for index, case in enumerate(family_cases):
                tier = case.pop("tier")
                meshes_of = case.pop("meshes")
                weight = case.pop("weight", None)
                row = OrderedDict(
                    id=f"{family}/{model.name}/{index:03d}",
                    family=family,
                    model=model.name,
                    tier=tier,
                )
                row["axes"] = case
                row["meshes"] = meshes_of
                if weight is not None:
                    row["weight"] = weight
                cases.append(row)
    return OrderedDict(
        version=1,
        doc="Authoritative kernel test / benchmark cases of PithTrain. Generated by `python -m tools.kernel_cases --write`; do not edit. See docs/kernel-validation.md.",
        inputs=OrderedDict(
            models={
                m.name: OrderedDict(
                    config=m.config_path, sha256=m.config_sha256, model_type=m.model_type
                )
                for m in models
            },
            matrix=OrderedDict(
                tokens_per_gpu=list(mesh_module.TOKENS_PER_GPU),
                micro_batch_size=list(mesh_module.MICRO_BATCH_SIZES),
                context_parallel=list(mesh_module.CONTEXT_PARALLEL),
                expert_parallel=list(mesh_module.EXPERT_PARALLEL),
                max_sequence_length=mesh_module.MAX_SEQUENCE_LENGTH,
            ),
            shipped_meshes=[
                OrderedDict(
                    source=m.source,
                    model=m.model,
                    pp=m.pp,
                    dp=m.dp,
                    cp=m.cp,
                    ep=m.ep,
                    seq=m.seq,
                    mbs=m.mbs,
                    fp8=m.fp8,
                    tokens_per_gpu=m.tokens_per_gpu,
                )
                for m in shipped
            ],
            routing=OrderedDict(
                sigma=SIGMA,
                hot_factor=HOT_FACTOR,
                fp8_group_align=FP8_GROUP_ALIGN,
                bf16_fused_max_groups=BF16_FUSED_MAX_GROUPS,
            ),
        ),
        families=FAMILIES,
        shape_independent=SHAPE_INDEPENDENT,
        cases=cases,
    )
