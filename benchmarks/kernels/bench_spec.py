"""
Times the kernels at the cases of tests/kernels/cases.py, the same cases the kernel
tests verify, and writes machine-readable results; --compare checks them against a stored baseline
with the docs/kernel-styles.md section 9 rule (median within 5% + 3 us; benchmarks/kernels/compare.py).

    python -m benchmarks.kernels.bench_spec --tier smoke --out results.json
    python -m benchmarks.kernels.bench_spec --tier smoke --compare benchmarks/kernels/baselines/h200.json
    python -m benchmarks.kernels.bench_spec --family moe_experts --model qwen3-30b-a3b --tier spec

One case runs several ops (e.g. the FP8 gate_up forward, its dgrad and wgrad); each op is timed
with CUDA events around single launches, the L2 flushed before each, after warmup, and reported as
median / p10 / p90 microseconds and TFLOP/s. Correctness is not checked here (tests/kernels does).
Timings are only comparable on an otherwise idle GPU.
"""

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.kernels.compare import gate
from pithtrain.kernels import (
    bf16_expert_down_fwd,
    bf16_expert_gate_up_fwd,
    bf16_experts_supported,
    causal_conv1d_bwd,
    causal_conv1d_fwd,
    chunk_gdr_bwd_pinned,
    chunk_gdr_fwd_pinned,
    cross_entropy_fwd_bwd,
    fp8_clamped_swiglu_bwd,
    fp8_clamped_swiglu_kmajor,
    fp8_clamped_swiglu_rowwise,
    fp8_gemm_bwd,
    fp8_gemm_fwd,
    fp8_grouped_gemm,
    fp8_grouped_wgrad_empty,
    fp8_pack_transpose,
    fp8_quant_grouped,
    fp8_quant_weight,
    fp8_quant_weight_grouped,
    fp8_silu_mul_bwd,
    fp8_silu_mul_fwd,
    fp8_silu_mul_gu_bwd,
    fp8_silu_mul_gu_fwd,
    gated_delta_rule_pinned_supported,
    k_grouped_fp8_gemm,
)
from tests.kernels.cases import select

DEV = "cuda"
BF16 = dict(device=DEV, dtype=torch.bfloat16)
FAMILIES = [
    "moe_experts",
    "dense_gemm",
    "weight_quant",
    "attention",
    "gated_delta_rule",
    "causal_conv1d",
    "cross_entropy",
]


def randn(*shape: int, scale: float = 1.0, gen: torch.Generator) -> torch.Tensor:
    return (torch.randn(*shape, device=DEV, generator=gen) * scale).to(torch.bfloat16)


class Timer:
    """
    Median / p10 / p90 of single launches with a cold L2.
    """

    def __init__(self, warmup: int, reps: int):
        self.warmup, self.reps = warmup, reps
        self.flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

    def __call__(self, fn: Callable[[], object]) -> dict:
        for _ in range(self.warmup):
            fn()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.reps)]
        for start, end in zip(starts, ends):
            self.flush.zero_()
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        times = sorted(start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends))
        pick = lambda q: times[min(len(times) - 1, int(q * len(times)))]
        return dict(median_us=pick(0.5), p10_us=pick(0.1), p90_us=pick(0.9))


def moe_ops(axes: dict, gen: torch.Generator) -> dict:
    """
    FP8: gate_up and down forward, their dgrad (the grouped GEMM against the transposed weight), and
    the k-grouped wgrad; BF16 fused gate_up and down forward where the fused path runs.
    """
    G, H, inter = axes["G"], axes["hidden"], axes["inter"]
    sizes = [-(-c // 128) * 128 for c in axes["group_sizes"]]
    M = sum(sizes)
    offs = torch.tensor(sizes, dtype=torch.int64).cumsum(0).to(torch.int32).to(DEV)
    ks = torch.tensor(sizes, dtype=torch.int32, device=DEV)
    x, dy = randn(M, H, gen=gen), randn(M, H, gen=gen)
    h = randn(M, inter, gen=gen)
    w_gu, w_dn = (
        randn(G, 2 * inter, H, scale=0.02, gen=gen),
        randn(G, H, inter, scale=0.02, gen=gen),
    )
    x_fp8, x_scale, x_ch, x_ch_scale = fp8_quant_grouped(x, offs)
    h_fp8, h_scale, h_ch, h_ch_scale = fp8_quant_grouped(h, offs)
    dy_fp8, dy_scale, dy_ch, dy_ch_scale = fp8_quant_grouped(dy, offs)
    gu_fp8, gu_scale, gut_fp8, gut_scale = fp8_quant_weight_grouped(w_gu)
    dn_fp8, dn_scale, dnt_fp8, dnt_scale = fp8_quant_weight_grouped(w_dn)
    wgrad = fp8_grouped_wgrad_empty(torch.Size((G, H, inter)), torch.device(DEV))
    gemm = 2 * M * H * inter
    ops = {
        "fp8_quant_grouped[x]": (lambda: fp8_quant_grouped(x, offs), 0),
        "fp8_grouped_gemm[gate_up_fwd]": (
            lambda: fp8_grouped_gemm((x_fp8, x_scale), (gu_fp8, gu_scale), 2 * inter, offs),
            2 * gemm,
        ),
        "fp8_grouped_gemm[down_fwd]": (
            lambda: fp8_grouped_gemm((h_fp8, h_scale), (dn_fp8, dn_scale), H, offs),
            gemm,
        ),
        "fp8_grouped_gemm[down_dgrad]": (
            lambda: fp8_grouped_gemm((dy_fp8, dy_scale), (dnt_fp8, dnt_scale), inter, offs),
            gemm,
        ),
        "k_grouped_fp8_gemm[down_wgrad]": (
            lambda: k_grouped_fp8_gemm((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), wgrad, sizes, ks),
            gemm,
        ),
    }
    ops.update(activation_quant_ops(axes, M, offs, gen))
    if axes["bf16_fused"] and bf16_experts_supported(G):
        raw = axes["group_sizes"]
        M_raw = sum(raw)
        cu = torch.tensor([0] + raw, dtype=torch.int64).cumsum(0).to(torch.int32).to(DEV)
        a_idx = torch.randperm(M_raw, device=DEV, generator=gen).to(torch.int32)
        slot_of = torch.randperm(M_raw, device=DEV, generator=gen).to(torch.int32)
        xb = randn(M_raw, H, gen=gen)
        w_gate, w_up = w_gu[:, :inter], w_gu[:, inter:]
        hb = randn(M_raw, inter, gen=gen)
        clamped = axes["act"] == "clamped_swiglu"  # gpt-oss: biases and the clamped SwiGLU
        gu_bias, dn_bias = (
            (randn(G, 2 * inter, gen=gen), randn(G, H, gen=gen)) if axes["bias"] else (None, None)
        )
        act = (1.702, 7.0) if clamped else ()
        ops["bf16_expert_gate_up_fwd"] = (
            lambda: bf16_expert_gate_up_fwd(xb, w_gate, w_up, cu, a_idx, gu_bias, act),
            4 * M_raw * H * inter,
        )
        ops["bf16_expert_down_fwd"] = (
            lambda: bf16_expert_down_fwd(hb, w_dn, cu, slot_of, dn_bias),
            2 * M_raw * H * inter,
        )
    return ops


def activation_quant_ops(axes: dict, M: int, offs: torch.Tensor, gen: torch.Generator) -> dict:
    """
    The fused activation + FP8 quantization between the expert GEMMs, forward and backward, in the
    model's gate / up layout (separate, the halves of one weight, or gpt-oss's interleaved rows).
    """
    G, inter, layout = axes["G"], axes["inter"], axes["gate_up_layout"]
    gu, dh = randn(M, 2 * inter, gen=gen), randn(M, inter, gen=gen)
    if layout == "interleaved":
        bias = randn(G, 2 * inter, gen=gen)
        alpha, limit = axes["swiglu_alpha"], axes["swiglu_limit"]
        return {
            "fp8_clamped_swiglu_rowwise": (
                lambda: fp8_clamped_swiglu_rowwise(gu, bias, offs, alpha, limit),
                0,
            ),
            "fp8_clamped_swiglu_kmajor": (
                lambda: fp8_clamped_swiglu_kmajor(gu, bias, offs, alpha, limit),
                0,
            ),
            "fp8_clamped_swiglu_bwd": (
                lambda: fp8_clamped_swiglu_bwd(dh, gu, bias, offs, alpha, limit),
                0,
            ),
        }
    if layout == "halves":
        return {
            "fp8_silu_mul_gu_fwd": (lambda: fp8_silu_mul_gu_fwd(gu, offs), 0),
            "fp8_silu_mul_gu_bwd": (lambda: fp8_silu_mul_gu_bwd(dh, gu, offs), 0),
        }
    g, u = gu[:, :inter].contiguous(), gu[:, inter:].contiguous()
    return {
        "fp8_silu_mul_fwd": (lambda: fp8_silu_mul_fwd(g, u, offs), 0),
        "fp8_silu_mul_bwd": (lambda: fp8_silu_mul_bwd(dh, g, u, offs), 0),
    }


def weight_quant_ops(axes: dict, gen: torch.Generator) -> dict:
    """
    The per-step FP8 weight quantization of one weight (dense, or the (G, N, K) experts of a
    rank) and the packing of its FP8 bytes with their transpose for the parameter all-gather.
    """
    G, N, K = axes["G"], axes["N"], axes["K"]
    w = randn(max(G, 1), N, K, scale=0.02, gen=gen)
    quant = (lambda: fp8_quant_weight_grouped(w)) if G else (lambda: fp8_quant_weight(w[0]))
    w_fp8 = fp8_quant_weight_grouped(w)[0]
    packed = torch.empty(2 * w_fp8.numel(), dtype=torch.uint8, device=DEV)
    return {
        "fp8_quant_weight" + ("_grouped" if G else ""): (quant, 0),
        "fp8_pack_transpose": (lambda: fp8_pack_transpose(w_fp8, packed), 0),
    }


def dense_ops(axes: dict, gen: torch.Generator) -> dict:
    M, N, K = axes["M"], axes["N"], axes["K"]
    x, w, dy = randn(M, K, gen=gen), randn(N, K, scale=0.02, gen=gen), randn(M, N, gen=gen)
    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(w)
    _, xt_fp8, xt_scale = fp8_gemm_fwd(x, w_fp8, w_scale)
    return {
        "fp8_gemm_fwd": (lambda: fp8_gemm_fwd(x, w_fp8, w_scale), 2 * M * N * K),
        "fp8_gemm_bwd": (
            lambda: fp8_gemm_bwd(dy, wt_fp8, wt_scale, xt_fp8, xt_scale, K),
            4 * M * N * K,
        ),
    }


def attention_ops(axes: dict, gen: torch.Generator) -> dict:
    """
    FA4 forward and backward through the interface ring_attention.py and flash_attn_v4.py call, with
    the strided views of the ring steps.
    """
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    kind, B, Sq, Skv = axes["kind"], axes["B"], axes["Sq"], axes["Skv"]
    Hq, Hkv, D, Dv = axes["Hq"], axes["Hkv"], axes["D"], axes["Dv"]
    scale = D**-0.5
    window = (axes["window"], 0) if axes["window"] is not None else (None, None)
    sink = torch.zeros(Hq, device=DEV, dtype=torch.bfloat16) if axes["sinks"] else None
    extra = dict(
        causal=axes["causal"],
        window_size_left=window[0],
        window_size_right=window[1],
        learnable_sink=sink,
    )
    if kind == "varlen":
        cu = torch.tensor(axes["cu_seqlens"], dtype=torch.int32, device=DEV)
        q, k, v = (
            randn(Sq, Hq, D, gen=gen),
            randn(Skv, Hkv, D, gen=gen),
            randn(Skv, Hkv, Dv, gen=gen),
        )
        extra.update(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=Sq, max_seqlen_k=Skv)
    elif kind == "ring_front":  # q (B, S_local), k / v the front half of (B, S_local) buffers
        q = randn(B, Sq, Hq, D, gen=gen)
        k, v = (
            randn(B, 2 * Skv, Hkv, D, gen=gen)[:, :Skv],
            randn(B, 2 * Skv, Hkv, Dv, gen=gen)[:, :Skv],
        )
    elif kind == "ring_back":  # q the back half of a (B, S_local) buffer
        q = randn(B, 2 * Sq, Hq, D, gen=gen)[:, Sq:]
        k, v = randn(B, Skv, Hkv, D, gen=gen), randn(B, Skv, Hkv, Dv, gen=gen)
    else:
        q, k, v = (
            randn(B, Sq, Hq, D, gen=gen),
            randn(B, Skv, Hkv, D, gen=gen),
            randn(B, Skv, Hkv, Dv, gen=gen),
        )
    out, lse, *_ = _flash_attn_fwd(q, k, v, softmax_scale=scale, return_lse=True, **extra)
    dout = torch.randn_like(out)
    flops = axes_flops(axes)
    return {
        "fa4_fwd": (
            lambda: _flash_attn_fwd(q, k, v, softmax_scale=scale, return_lse=True, **extra),
            flops,
        ),
        "fa4_bwd": (
            lambda: _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale=scale, **extra),
            int(2.5 * flops),
        ),
    }


def axes_flops(axes: dict) -> int:
    if axes["kind"] == "varlen":
        cu = axes["cu_seqlens"]
        pairs = sum((b - a) * (b - a) // 2 for a, b in zip(cu, cu[1:]))
    else:
        pairs = axes["B"] * axes["Sq"] * axes["Skv"] // (2 if axes["causal"] else 1)
    if axes["window"] is not None:
        pairs = min(pairs, axes["B"] * axes["Sq"] * (axes["window"] + 1))
    return 2 * axes["Hq"] * pairs * (axes["D"] + axes["Dv"])


def gdn_ops(axes: dict, gen: torch.Generator) -> dict:
    if axes["cp"] != 1:
        return {}
    T, H, HV, K = axes["T_local"], axes["H"], axes["HV"], axes["K"]
    q = F.normalize(randn(1, T, H, K, gen=gen).float(), dim=-1).to(torch.bfloat16)
    k = F.normalize(randn(1, T, H, K, gen=gen).float(), dim=-1).to(torch.bfloat16)
    v = randn(1, T, HV, K, gen=gen)
    g = -torch.rand(1, T, HV, device=DEV, generator=gen) * 0.5
    beta = torch.rand(1, T, HV, device=DEV, generator=gen).to(torch.bfloat16)
    if not gated_delta_rule_pinned_supported(q, k, v, g, beta):
        return {}
    o, g_cum, A = chunk_gdr_fwd_pinned(q, k, v, g, beta)
    do = torch.randn_like(o)
    flops = 4 * T * HV * K * K * 2  # chunked: ~state update + output per token, order of magnitude
    return {
        "chunk_gdr_fwd_pinned": (lambda: chunk_gdr_fwd_pinned(q, k, v, g, beta), flops),
        "chunk_gdr_bwd_pinned": (
            lambda: chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do),
            2 * flops,
        ),
    }


def conv_ops(axes: dict, gen: torch.Generator) -> dict:
    B, T, C, W = axes["B"], axes["T_local"], axes["C"], axes["width"]
    offset = W - 1 if axes["cp"] > 1 else 0
    x = randn(B, offset + T, C, gen=gen)
    w = torch.randn(C, W, device=DEV, generator=gen)
    gy = randn(B, T, C, gen=gen)
    return {
        "causal_conv1d_fwd": (lambda: causal_conv1d_fwd(x, w, offset), 2 * B * T * C * W),
        "causal_conv1d_bwd": (lambda: causal_conv1d_bwd(gy, x, w, offset), 4 * B * T * C * W),
    }


def ce_ops(axes: dict, gen: torch.Generator) -> dict:
    rows, vocab = axes["rows"], axes["vocab"]
    logits = randn(rows, vocab, gen=gen)
    target = torch.randint(0, vocab, (rows,), device=DEV, generator=gen)
    work = logits.clone()

    def run():
        work.copy_(logits)  # the kernel overwrites its input with the gradient
        cross_entropy_fwd_bwd(work, target, -100)

    return {"cross_entropy_fwd_bwd(+copy)": (run, 0)}


OPS = dict(
    moe_experts=moe_ops,
    dense_gemm=dense_ops,
    weight_quant=weight_quant_ops,
    attention=attention_ops,
    gated_delta_rule=gdn_ops,
    causal_conv1d=conv_ops,
    cross_entropy=ce_ops,
)


def environment() -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    props = torch.cuda.get_device_properties(0)
    versions = {}
    for package in (
        "deep-gemm",
        "flash-attn-4",
        "quack-kernels",
        "nvidia-cutlass-dsl",
        "apache-tvm-tirx",
        "tirx-kernels",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(
        gpu=props.name,
        sm_count=props.multi_processor_count,
        capability=f"{props.major}.{props.minor}",
        torch=torch.__version__,
        versions=versions,
        git_sha=sha,
        date=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def run(args) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    timer = Timer(args.warmup, args.reps)
    results = []
    for family in args.family or FAMILIES:
        for case in select(family, args.tier, model=args.model):
            if args.tier != "stress" and case["tier"] == "stress":
                continue
            gen = torch.Generator(device=DEV).manual_seed(0)
            try:
                ops = OPS[family](case["axes"], gen)
                for op, (fn, flops) in ops.items():
                    timing = timer(fn)
                    tflops = flops / timing["median_us"] / 1e6 if flops else None
                    results.append(dict(case=case["id"], op=op, **timing, tflops=tflops))
                    print(
                        f"{case['id']:<44} {op:<34} {timing['median_us']:10.1f} us"
                        + (f" {tflops:7.1f} TFLOP/s" if tflops else ""),
                        flush=True,
                    )
            except torch.OutOfMemoryError as error:
                results.append(dict(case=case["id"], op="*", error=f"OOM: {error}"[:200]))
            finally:
                ops = None
                torch.cuda.empty_cache()
    return dict(
        meta=dict(environment(), tier=args.tier, warmup=args.warmup, reps=args.reps),
        results=results,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--model")
    parser.add_argument("--tier", default="smoke", choices=["smoke", "spec", "stress"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", type=Path, help="baseline JSON; exit 1 on a regression")
    args = parser.parse_args()
    result = run(args)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    if args.compare:
        return gate(result, json.loads(args.compare.read_text()))
    return 0 if all(math.isfinite(r.get("median_us", 0.0)) for r in result["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
