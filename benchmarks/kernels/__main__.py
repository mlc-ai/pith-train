"""
Times the kernels at the cases of tests/kernels/cases.py, for this tree alone or against a base git
ref (--base), each op over cold-L2 single launches and weighted by its work times its layers.

    python -m benchmarks.kernels --base upstream/main --family cross_entropy --out ab.json
    python -m benchmarks.kernels --base upstream/main --family moe_experts --op 'quant|silu'
    python -m benchmarks.kernels --base-dir . --family weight_quant      (the noise floor)
    python -m benchmarks.kernels --out results.json                       (this tree alone)
    python -m benchmarks.kernels --shard 3/8 --out shard3.json            (one per GPU)
    python -m benchmarks.kernels --merge shard*.json --out results.json

--base exits 1 when an op that ran at the base fails here, or an op name >5% slower work-weighted.
"""

import argparse
import importlib.metadata
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from tests.kernels.cases import all_cases, select

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"
TOLERANCE = 0.05
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


class Op:
    """
    One timed launch: fn and its work (FLOPs, or elements read when memory_bound).
    """

    def __init__(self, fn: Callable[[], object], work: int, memory_bound: bool = False):
        self.fn, self.work, self.memory_bound = fn, work, memory_bound


def mem(fn: Callable[[], object], *tensors: torch.Tensor) -> Op:
    return Op(fn, sum(t.numel() for t in tensors), memory_bound=True)


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
    The FP8 expert GEMMs and quantizations, and the fused BF16 expert GEMMs where supported.
    """
    from pithtrain.kernels import (
        bf16_expert_down_fwd,
        bf16_expert_gate_up_fwd,
        bf16_experts_supported,
        fp8_grouped_gemm,
        fp8_grouped_wgrad_empty,
        fp8_quant_grouped,
        fp8_quant_weight_grouped,
        k_grouped_fp8_gemm,
    )

    G, H, inter = axes["G"], axes["hidden"], axes["inter"]
    ops = {}
    w_gu, w_dn = (
        randn(G, 2 * inter, H, scale=0.02, gen=gen),
        randn(G, H, inter, scale=0.02, gen=gen),
    )
    if axes["routing"] != "boundary":
        sizes = [-(-c // 128) * 128 for c in axes["group_sizes"]]
        M = sum(sizes)
        offs = torch.tensor(sizes, dtype=torch.int64).cumsum(0).to(torch.int32).to(DEV)
        ks = torch.tensor(sizes, dtype=torch.int32, device=DEV)
        x, dy, h = randn(M, H, gen=gen), randn(M, H, gen=gen), randn(M, inter, gen=gen)
        ops["fp8_quant_grouped[x]"] = mem(lambda: fp8_quant_grouped(x, offs), x)
        x_fp8, x_scale, _, _ = fp8_quant_grouped(x, offs)
        h_fp8, h_scale, h_ch, h_ch_scale = fp8_quant_grouped(h, offs)
        dy_fp8, dy_scale, dy_ch, dy_ch_scale = fp8_quant_grouped(dy, offs)
        gu_fp8, gu_scale, _, _ = fp8_quant_weight_grouped(w_gu)
        dn_fp8, dn_scale, dnt_fp8, dnt_scale = fp8_quant_weight_grouped(w_dn)
        wgrad = fp8_grouped_wgrad_empty(torch.Size((G, H, inter)), torch.device(DEV))
        gemm = 2 * M * H * inter
        ops["fp8_grouped_gemm[gate_up_fwd]"] = Op(lambda: fp8_grouped_gemm((x_fp8, x_scale), (gu_fp8, gu_scale), 2 * inter, offs), 2 * gemm)  # fmt: skip
        ops["fp8_grouped_gemm[down_fwd]"] = Op(lambda: fp8_grouped_gemm((h_fp8, h_scale), (dn_fp8, dn_scale), H, offs), gemm)  # fmt: skip
        ops["fp8_grouped_gemm[down_dgrad]"] = Op(lambda: fp8_grouped_gemm((dy_fp8, dy_scale), (dnt_fp8, dnt_scale), inter, offs), gemm)  # fmt: skip
        ops["k_grouped_fp8_gemm[down_wgrad]"] = Op(lambda: k_grouped_fp8_gemm((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), wgrad, sizes, ks), gemm)  # fmt: skip
        ops.update(activation_quant_ops(axes, M, offs, gen))
    if axes["bf16_fused"] and bf16_experts_supported(G):
        raw = axes["group_sizes"]
        M_raw = sum(raw)
        cu = torch.tensor([0] + raw, dtype=torch.int64).cumsum(0).to(torch.int32).to(DEV)
        a_idx = torch.randperm(M_raw, device=DEV, generator=gen).to(torch.int32)
        slot_of = torch.randperm(M_raw, device=DEV, generator=gen).to(torch.int32)
        xb, hb = randn(M_raw, H, gen=gen), randn(M_raw, inter, gen=gen)
        w_gate, w_up = w_gu[:, :inter], w_gu[:, inter:]
        gu_bias, dn_bias = (randn(G, 2 * inter, gen=gen), randn(G, H, gen=gen)) if axes["bias"] else (None, None)  # fmt: skip
        act = (axes["swiglu_alpha"], axes["swiglu_limit"]) if axes["act"] == "clamped_swiglu" else ()  # fmt: skip
        ops["bf16_expert_gate_up_fwd"] = Op(lambda: bf16_expert_gate_up_fwd(xb, w_gate, w_up, cu, a_idx, gu_bias, act), 4 * M_raw * H * inter)  # fmt: skip
        ops["bf16_expert_down_fwd"] = Op(lambda: bf16_expert_down_fwd(hb, w_dn, cu, slot_of, dn_bias), 2 * M_raw * H * inter)  # fmt: skip
    return ops


def activation_quant_ops(axes: dict, M: int, offs: torch.Tensor, gen: torch.Generator) -> dict:
    """
    The fused activation + FP8 quantization between the expert GEMMs, in the model's layout.
    """
    from pithtrain.kernels import (
        fp8_clamped_swiglu_bwd,
        fp8_clamped_swiglu_kmajor,
        fp8_clamped_swiglu_rowwise,
        fp8_silu_mul_bwd,
        fp8_silu_mul_fwd,
        fp8_silu_mul_gu_bwd,
        fp8_silu_mul_gu_fwd,
    )

    G, inter, layout = axes["G"], axes["inter"], axes["gate_up_layout"]
    gu, dh = randn(M, 2 * inter, gen=gen), randn(M, inter, gen=gen)
    if layout == "interleaved":
        bias = randn(G, 2 * inter, gen=gen)
        alpha, limit = axes["swiglu_alpha"], axes["swiglu_limit"]
        return {
            "fp8_clamped_swiglu_rowwise": mem(
                lambda: fp8_clamped_swiglu_rowwise(gu, bias, offs, alpha, limit), gu
            ),  # fmt: skip
            "fp8_clamped_swiglu_kmajor": mem(
                lambda: fp8_clamped_swiglu_kmajor(gu, bias, offs, alpha, limit), gu
            ),  # fmt: skip
            "fp8_clamped_swiglu_bwd": mem(
                lambda: fp8_clamped_swiglu_bwd(dh, gu, bias, offs, alpha, limit), dh, gu
            ),  # fmt: skip
        }
    if layout == "halves":
        return {
            "fp8_silu_mul_gu_fwd": mem(lambda: fp8_silu_mul_gu_fwd(gu, offs), gu),
            "fp8_silu_mul_gu_bwd": mem(lambda: fp8_silu_mul_gu_bwd(dh, gu, offs), dh, gu),
        }
    g, u = gu[:, :inter].contiguous(), gu[:, inter:].contiguous()
    return {
        "fp8_silu_mul_fwd": mem(lambda: fp8_silu_mul_fwd(g, u, offs), g, u),
        "fp8_silu_mul_bwd": mem(lambda: fp8_silu_mul_bwd(dh, g, u, offs), dh, g, u),
    }


def weight_quant_ops(axes: dict, gen: torch.Generator) -> dict:
    """
    The FP8 quantization of one weight and the packing of its bytes for the parameter all-gather.
    """
    from pithtrain.kernels import fp8_pack_transpose, fp8_quant_weight, fp8_quant_weight_grouped

    G, N, K = axes["G"], axes["N"], axes["K"]
    w = randn(max(G, 1), N, K, scale=0.02, gen=gen)
    quant = (lambda: fp8_quant_weight_grouped(w)) if G else (lambda: fp8_quant_weight(w[0]))
    w_fp8 = fp8_quant_weight_grouped(w)[0]
    packed = torch.empty(2 * w_fp8.numel(), dtype=torch.uint8, device=DEV)
    return {
        "fp8_quant_weight" + ("_grouped" if G else ""): mem(quant, w),
        "fp8_pack_transpose": mem(lambda: fp8_pack_transpose(w_fp8, packed), w_fp8),
    }


def dense_ops(axes: dict, gen: torch.Generator) -> dict:
    from pithtrain.kernels import fp8_gemm_bwd, fp8_gemm_fwd, fp8_quant_weight

    M, N, K = axes["M"], axes["N"], axes["K"]
    x, w, dy = randn(M, K, gen=gen), randn(N, K, scale=0.02, gen=gen), randn(M, N, gen=gen)
    w_fp8, w_scale, wt_fp8, wt_scale = fp8_quant_weight(w)
    _, xt_fp8, xt_scale = fp8_gemm_fwd(x, w_fp8, w_scale)
    return {
        "fp8_gemm_fwd": Op(lambda: fp8_gemm_fwd(x, w_fp8, w_scale), 2 * M * N * K),
        "fp8_gemm_bwd": Op(
            lambda: fp8_gemm_bwd(dy, wt_fp8, wt_scale, xt_fp8, xt_scale, K), 4 * M * N * K
        ),  # fmt: skip
    }


def attention_ops(axes: dict, gen: torch.Generator) -> dict:
    """
    FA4 forward and backward, with the strided views of the ring steps.
    """
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    kind, B, Sq, Skv = axes["kind"], axes["B"], axes["Sq"], axes["Skv"]
    Hq, Hkv, D, Dv = axes["Hq"], axes["Hkv"], axes["D"], axes["Dv"]
    scale = D**-0.5
    window = (axes["window"], 0) if axes["window"] is not None else (None, None)
    sink = torch.zeros(Hq, device=DEV, dtype=torch.bfloat16) if axes["sinks"] else None
    extra = dict(causal=axes["causal"], window_size_left=window[0], window_size_right=window[1], learnable_sink=sink)  # fmt: skip
    if kind == "varlen":
        cu = torch.tensor(axes["cu_seqlens"], dtype=torch.int32, device=DEV)
        q, k, v = randn(Sq, Hq, D, gen=gen), randn(Skv, Hkv, D, gen=gen), randn(Skv, Hkv, Dv, gen=gen)  # fmt: skip
        extra.update(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=Sq, max_seqlen_k=Skv)
    elif kind == "ring_front":  # q (B, S_local), k / v the front half of (B, S_local) buffers
        q = randn(B, Sq, Hq, D, gen=gen)
        k, v = randn(B, 2 * Skv, Hkv, D, gen=gen)[:, :Skv], randn(B, 2 * Skv, Hkv, Dv, gen=gen)[:, :Skv]  # fmt: skip
    elif kind == "ring_back":  # q the back half of a (B, S_local) buffer
        q = randn(B, 2 * Sq, Hq, D, gen=gen)[:, Sq:]
        k, v = randn(B, Skv, Hkv, D, gen=gen), randn(B, Skv, Hkv, Dv, gen=gen)
    else:
        q, k, v = randn(B, Sq, Hq, D, gen=gen), randn(B, Skv, Hkv, D, gen=gen), randn(B, Skv, Hkv, Dv, gen=gen)  # fmt: skip
    out, lse, *_ = _flash_attn_fwd(q, k, v, softmax_scale=scale, return_lse=True, **extra)
    dout = torch.randn_like(out)
    flops = attention_flops(axes)
    return {
        "fa4_fwd": Op(
            lambda: _flash_attn_fwd(q, k, v, softmax_scale=scale, return_lse=True, **extra), flops
        ),  # fmt: skip
        "fa4_bwd": Op(
            lambda: _flash_attn_bwd(q, k, v, out, dout, lse, softmax_scale=scale, **extra),
            int(2.5 * flops),
        ),  # fmt: skip
    }


def attention_flops(axes: dict) -> int:
    if axes["kind"] == "varlen":
        cu = axes["cu_seqlens"]
        pairs = sum((b - a) * (b - a) // 2 for a, b in zip(cu, cu[1:]))
    else:
        pairs = axes["B"] * axes["Sq"] * axes["Skv"] // (2 if axes["causal"] else 1)
    if axes["window"] is not None:
        pairs = min(pairs, axes["B"] * axes["Sq"] * (axes["window"] + 1))
    return 2 * axes["Hq"] * pairs * (axes["D"] + axes["Dv"])


def gdn_ops(axes: dict, gen: torch.Generator) -> dict:
    from pithtrain.kernels import (
        chunk_gdr_bwd_pinned,
        chunk_gdr_fwd_pinned,
        gated_delta_rule_pinned_supported,
    )

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
    flops = 4 * T * HV * K * K * 2  # order of magnitude
    return {
        "chunk_gdr_fwd_pinned": Op(lambda: chunk_gdr_fwd_pinned(q, k, v, g, beta), flops),
        "chunk_gdr_bwd_pinned": Op(
            lambda: chunk_gdr_bwd_pinned(q, k, v, g_cum, beta, A, do), 2 * flops
        ),  # fmt: skip
    }


def conv_ops(axes: dict, gen: torch.Generator) -> dict:
    from pithtrain.kernels import causal_conv1d_bwd, causal_conv1d_fwd

    B, T, C, W = axes["B"], axes["T_local"], axes["C"], axes["width"]
    offset = W - 1 if axes["cp"] > 1 else 0
    x = randn(B, offset + T, C, gen=gen)
    w = torch.randn(C, W, device=DEV, generator=gen)
    gy = randn(B, T, C, gen=gen)
    return {
        "causal_conv1d_fwd": Op(lambda: causal_conv1d_fwd(x, w, offset), 2 * B * T * C * W),
        "causal_conv1d_bwd": Op(lambda: causal_conv1d_bwd(gy, x, w, offset), 4 * B * T * C * W),
    }


def ce_ops(axes: dict, gen: torch.Generator) -> dict:
    """
    The cross_entropy op forward and backward and the fused kernel, each timed with a logits copy.
    """
    from pithtrain.kernels import cross_entropy_fwd_bwd
    from pithtrain.operators.cross_entropy import cross_entropy

    rows, vocab = axes["rows"], axes["vocab"]
    logits = randn(rows, vocab, gen=gen)
    target = torch.randint(0, vocab, (rows,), device=DEV, generator=gen)
    work = torch.empty_like(logits)
    grad = torch.full((), 1.0 / 3.0, device=DEV)

    def op():
        with torch.no_grad():
            work.copy_(logits)
        leaf = work.detach().requires_grad_()
        cross_entropy(leaf, target, -100).backward(grad)

    def kernel():
        work.copy_(logits)
        cross_entropy_fwd_bwd(work, target, -100)

    return {"cross_entropy[op fwd+bwd](+copy)": mem(op, logits), "cross_entropy_fwd_bwd(+copy)": mem(kernel, logits)}  # fmt: skip


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
    for package in ("deep-gemm", "flash-attn-4", "quack-kernels", "nvidia-cutlass-dsl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(gpu=props.name, sm_count=props.multi_processor_count, capability=f"{props.major}.{props.minor}", torch=torch.__version__, versions=versions, git_sha=sha, date=time.strftime("%Y-%m-%dT%H:%M:%S"))  # fmt: skip


def cases_of(families: list[str], model: str | None, shard: str | None) -> list[dict]:
    """
    The cases of families, split by shard "I/N" into every N-th case starting at I.
    """
    cases = [c for family in families for c in select(family, model=model)]
    if shard is not None:
        index, count = (int(part) for part in shard.split("/"))
        cases = cases[index::count]
    return cases


def build(case: dict) -> dict[str, Op]:
    """
    The ops of a case on freshly drawn inputs (seed 0).
    """
    return OPS[case["family"]](case["axes"], torch.Generator(device=DEV).manual_seed(0))


def error_of(error: BaseException) -> dict:
    return dict(error=f"{type(error).__name__}: {error}"[:300])


def time_case(case: dict, timer: Timer, pattern: re.Pattern | None) -> dict:
    """
    {op: timing and work, or {"error": ...}} of the case's ops matching pattern ("*": build failed).
    """
    reply, ops = {}, {}
    try:
        ops = build(case)
        for op, entry in ops.items():
            if pattern is None or pattern.search(op):
                try:
                    reply[op] = dict(timer(entry.fn), work=entry.work, memory_bound=entry.memory_bound)  # fmt: skip
                except Exception as error:
                    reply[op] = error_of(error)
    except Exception as error:
        reply["*"] = error_of(error)
    finally:
        ops = None
        torch.cuda.empty_cache()
    return reply


def layers_of(case: dict) -> int:
    return case.get("weight", {}).get("layers", 1)


def run(args) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    timer, pattern = Timer(args.warmup, args.reps), re.compile(args.op) if args.op else None
    results = []
    for case in cases_of(args.family or FAMILIES, args.model, args.shard):
        for op, timing in time_case(case, timer, pattern).items():
            row = dict(case=case["id"], op=op, layers=layers_of(case), **timing)
            if "median_us" in row and not row["memory_bound"] and row["work"]:
                row["tflops"] = row["work"] / row["median_us"] / 1e6
            results.append(row)
            print(f"{case['id']:<52} {op:<36} " + (f"{row['median_us']:10.1f} us" if "median_us" in row else row["error"]), flush=True)  # fmt: skip
    return dict(meta=dict(environment(), warmup=args.warmup, reps=args.reps), results=results)  # fmt: skip


def merge(paths: list[Path]) -> dict:
    """
    One result of several shards of the same run (the first shard's meta).
    """
    runs = [json.loads(path.read_text()) for path in paths]
    return dict(meta=dict(runs[0]["meta"], shards=len(runs)), results=[r for run in runs for r in run["results"]])  # fmt: skip


def serve(tree: str) -> None:
    """
    An A/B worker importing pithtrain from tree: one JSON request line in, one time_case line out.
    """
    sys.path.insert(0, tree)
    import pithtrain

    assert Path(pithtrain.__file__).resolve().is_relative_to(Path(tree).resolve()), pithtrain.__file__  # fmt: skip
    sys.path.remove(tree)
    torch.backends.cuda.matmul.allow_tf32 = False
    by_id, timer = {case["id"]: case for case in all_cases()}, None
    print(json.dumps(dict(ready=pithtrain.__file__)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        timer = timer or Timer(request["warmup"], request["reps"])
        print(json.dumps(time_case(by_id[request["case"]], timer, request["op"] and re.compile(request["op"]))), flush=True)  # fmt: skip


class Worker:
    def __init__(self, tree: Path):
        cmd = [sys.executable, "-m", "benchmarks.kernels", "--serve", str(tree)]
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
        self.proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)  # fmt: skip
        self.pithtrain = self.read()["ready"]

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited with {self.proc.wait()}")
        return json.loads(line)

    def time(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        return self.read()

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def base_tree(ref: str) -> Path:
    sha = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    tree = REPO_ROOT / "workspace" / "ab" / sha[:12]
    if not (tree / "pithtrain").is_dir():
        subprocess.run(["git", "worktree", "add", "--detach", str(tree), sha], cwd=REPO_ROOT, check=True)  # fmt: skip
    return tree


def ab_row(case: dict, op: str, timings: dict) -> dict:
    """
    One op of a case: each side's median over the rounds (or its error) and the speedup.
    """
    row = dict(case=case["id"], op=op, layers=layers_of(case))
    for side, rounds in timings.items():
        times = [t["median_us"] for t in rounds if "median_us" in t]
        if times:
            row[f"{side}_us"], row["work"] = statistics.median(times), rounds[0]["work"]
        else:
            row[f"{side}_error"] = next((t["error"] for t in rounds if "error" in t), "not timed")
    if "base_us" in row and "head_us" in row:
        row["speedup"] = row["base_us"] / row["head_us"]
    return row


def summarize(rows: list[dict]) -> dict:
    """
    Per op name and over every op ("*"): the work-weighted speedup, the op count, min and max.
    """
    groups = defaultdict(list)
    for row in rows:
        if "speedup" in row:
            groups[row["op"]].append(row)
            groups["*"].append(row)
    out = {}
    for name, members in groups.items():
        w = [r["work"] * r["layers"] for r in members]
        weighted = sum(w) / sum(wi / r["speedup"] for wi, r in zip(w, members)) if sum(w) else None
        out[name] = dict(ops=len(members), weighted_speedup=weighted, min=min(r["speedup"] for r in members), max=max(r["speedup"] for r in members))  # fmt: skip
    return out


def regressions(rows: list[dict], summary: dict) -> list[str]:
    problems = [f"{r['case']} {r['op']}: {r['head_error']}" for r in rows if "head_error" in r and "base_us" in r]  # fmt: skip
    for name, s in sorted(summary.items()):
        if name != "*" and s["weighted_speedup"] and s["weighted_speedup"] < 1 / (1 + TOLERANCE):
            problems.append(f"{name}: work-weighted speedup x{s['weighted_speedup']:.3f}")
    return problems


def ab(args) -> int:
    sides = dict(base=Worker(args.base_dir.resolve() if args.base_dir else base_tree(args.base)), head=Worker(REPO_ROOT))  # fmt: skip
    rows = []
    for case in cases_of(args.family or FAMILIES, args.model, args.shard):
        request = dict(case=case["id"], op=args.op, warmup=args.warmup, reps=args.reps)
        timings = dict(base=defaultdict(list), head=defaultdict(list))
        for r in range(args.rounds):  # base, head, head, base, ...
            for side in ("base", "head") if r % 2 == 0 else ("head", "base"):
                for op, timing in sides[side].time(request).items():
                    timings[side][op].append(timing)
        for op in dict.fromkeys([*timings["base"], *timings["head"]]):
            rows.append(ab_row(case, op, {side: timings[side][op] for side in timings}))
            row = rows[-1]
            shown = f"{row['base_us']:10.1f} {row['head_us']:10.1f} us  x{row['speedup']:.3f}" if "speedup" in row else f"base: {row.get('base_error', 'ok')} | head: {row.get('head_error', 'ok')}"  # fmt: skip
            print(f"{case['id']:<52} {op:<36} {shown}", flush=True)
    for worker in sides.values():
        worker.close()
    summary = summarize(rows)
    for name, s in sorted(summary.items()):
        weighted = f"{s['weighted_speedup']:.3f}" if s["weighted_speedup"] else "-"
        print(f"{name:<40} {s['ops']:5d} ops  work-weighted speedup x{weighted}  (min {s['min']:.3f}, max {s['max']:.3f})")  # fmt: skip
    problems = regressions(rows, summary)
    print("".join(f"REGRESSION {line}\n" for line in problems) + f"{len(problems)} regressions")
    if args.out:
        meta = dict(base=args.base or str(args.base_dir), base_pithtrain=sides["base"].pithtrain, head_pithtrain=sides["head"].pithtrain, rounds=args.rounds)  # fmt: skip
        args.out.write_text(json.dumps(dict(meta=meta, summary=summary, results=rows, regressions=problems), indent=1) + "\n")  # fmt: skip
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--model")
    parser.add_argument("--op", help="time only the ops whose name matches this regex")
    parser.add_argument("--shard", help="I/N: every N-th case starting at I")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--base", help="A/B: a git ref to time this tree against")
    parser.add_argument("--base-dir", type=Path, help="A/B: a checkout to time this tree against")
    parser.add_argument("--rounds", type=int, default=3, help="A/B: timings per op and side")
    parser.add_argument("--merge", nargs="+", type=Path, metavar="SHARD", help="merge shard results into --out instead of running")  # fmt: skip
    parser.add_argument("--serve", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
        return 0
    if args.base or args.base_dir:
        if args.base and args.base_dir:
            parser.error("give one of --base and --base-dir")
        return ab(args)
    result = merge(args.merge) if args.merge else run(args)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0 if all(math.isfinite(r.get("median_us", math.nan)) for r in result["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
