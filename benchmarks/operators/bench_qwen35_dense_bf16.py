"""Profile Qwen3.5-27B BF16 operators at their original widths on one GPU."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from pithtrain.contexts import distributed, training
from pithtrain.models.qwen35 import Qwen35DecoderLayer
from pithtrain.models.qwen35_moe import Qwen35MoeRMSNorm, Qwen35MoeRotaryEmbedding
from pithtrain.operators.cross_entropy import cross_entropy
from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.gated_delta_rule import gated_delta_rule
from pithtrain.operators.linear import BF16Linear
from pithtrain.operators.silu_mul import silu_mul

PEAK = 989e12
CONFIG = Path(__file__).resolve().parents[2] / "examples/pretrain_lm/qwen3.5-27b/config.json"


def causal_attention_flops(seq, heads, dim):
    return 6 * (seq + 1) * heads * dim


def gated_delta_flops(heads, key_dim, value_dim):
    # Useful recurrent arithmetic for K @ state, outer update, scaling state and Q @ state.
    return 3 * 8 * heads * key_dim * value_dim


def random_tensor(*shape):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)


def configure_main_grad(module, enabled):
    if enabled:
        for child in module.modules():
            if isinstance(child, BF16Linear):
                child.main_grad = torch.zeros_like(child.weight, dtype=torch.float32)


def get_flash_attention_3():
    try:
        from flash_attn_interface import flash_attn_func as fa3
    except ImportError:
        from kernels import get_kernel

        fa3 = get_kernel("kernels-community/flash-attn3", version=1).flash_attn_func
    return fa3


def measure(name, forward, leaves, tokens, flops_per_token, args, backward=True):
    def iteration():
        for leaf in leaves:
            leaf.grad = None
        out = forward()
        if backward:
            if out.ndim == 0:
                out.backward()
            else:
                out.backward(upstream)
        return out

    sample = forward()
    upstream = torch.randn_like(sample) if sample.ndim and backward else None
    del sample
    for _ in range(args.warmup):
        out = iteration()
    torch.cuda.synchronize()
    sanity = float(out) if out.ndim == 0 else float(out.float().square().mean())
    del out
    if not math.isfinite(sanity) or any(
        not torch.isfinite(leaf.grad).all().item() for leaf in leaves if leaf.grad is not None
    ):
        raise RuntimeError(f"{name} produced a non-finite value or gradient")

    def timed(run, mode):
        durations = []
        gpu_durations = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin = time.perf_counter()
            start.record()
            for _ in range(args.inner):
                run()
            end.record()
            end.synchronize()
            durations.append((time.perf_counter() - begin) * 1e3 / args.inner)
            gpu_durations.append(start.elapsed_time(end) / args.inner)
        elapsed = statistics.median(durations)
        rate = tokens * 1e3 / elapsed
        print(
            json.dumps(
                {
                    "case": name,
                    "mode": mode,
                    "tokens": tokens,
                    "seq": args.seq,
                    "wall_ms": round(elapsed, 4),
                    "event_ms": round(statistics.median(gpu_durations), 4),
                    "tokens_per_s": round(rate),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
                    "mfu_pct": round(100 * rate * flops_per_token / PEAK, 3)
                    if flops_per_token
                    else None,
                    "sanity": sanity,
                    "flops_per_token": flops_per_token,
                    "main_grad": args.main_grad,
                }
            ),
            flush=True,
        )

    timed(iteration, "direct")
    if args.graph:
        for leaf in leaves:
            leaf.grad = None
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                captured_loss = iteration()
            timed(graph.replay, "graph")
            del captured_loss
        except RuntimeError as exc:
            print(json.dumps({"case": name, "mode": "graph", "error": str(exc)}), flush=True)
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(name)
        iteration()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


def cases(args, config):
    b, s, h = args.batch, args.seq, config.hidden_size
    i, d, qh, kh = (
        config.intermediate_size,
        config.head_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
    )
    gk, gv = config.linear_num_key_heads, config.linear_num_value_heads
    kd, vd = config.linear_key_head_dim, config.linear_value_head_dim
    vocab = config.vocab_size
    gdn_projection = 6 * h * (2 * gk * kd + 3 * gv * vd + 2 * gv)
    gdn_core = gated_delta_flops(gv, kd, vd)
    attn_projection = 6 * h * (3 * qh + 2 * kh) * d
    attn_core = causal_attention_flops(s, qh, d)
    mlp_flops = 18 * h * i

    def run(name, fn, leaves, flops=0, backward=True):
        if name in args.cases or "all" in args.cases:
            measure(name, fn, tuple(leaves), b * s, flops, args, backward)

    for idx, name, mix_flops in [
        (
            0,
            "delta",
            gdn_projection + gdn_core + 6 * config.linear_conv_kernel_dim * (2 * gk * kd + gv * vd),
        ),
        (3, "attention", attn_projection + attn_core),
    ]:
        variants = {name, f"{name}-eager", f"{name}-compiled", f"{name}-mix"}
        if not (variants.intersection(args.cases) or "all" in args.cases):
            continue
        layer = Qwen35DecoderLayer(config, idx).cuda().bfloat16()
        configure_main_grad(layer, args.main_grad)
        x = random_tensor(b, s, h)
        pos = tuple(t.unsqueeze(0) for t in Qwen35MoeRotaryEmbedding(config).cuda()(s))
        leaves = (x, *layer.parameters())

        def stages():
            normalized, residual, _ = layer.forward_stage1(x, pos)
            return layer.forward_stage5(layer.forward_stage3(normalized), None, None, residual)

        run(name, stages, leaves, mix_flops + mlp_flops)
        run(f"{name}-eager", lambda: layer.reference_forward(x, pos), leaves, mix_flops + mlp_flops)
        if f"{name}-compiled" in args.cases or "all" in args.cases:
            compiled = torch.compile(layer.reference_forward, fullgraph=True)
            run(f"{name}-compiled", lambda: compiled(x, pos), leaves, mix_flops + mlp_flops)
        run(f"{name}-mix", lambda: layer.forward_stage1(x, pos)[0], leaves, mix_flops)

    if {"all", "delta-core", "fla"}.intersection(args.cases):
        q, k = [random_tensor(b, s, gk, kd) for _ in range(2)]
        v = random_tensor(b, s, gv, vd)
        g = -torch.rand(b, s, gv, device="cuda")
        beta = torch.rand(b, s, gv, device="cuda", dtype=torch.bfloat16)
        leaves = (q, k, v, g.requires_grad_(), beta.requires_grad_())
        run("delta-core", lambda: gated_delta_rule(*leaves), leaves, gdn_core)
        if "fla" in args.cases or "all" in args.cases:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            except ImportError as exc:
                print(json.dumps({"case": "fla", "error": str(exc)}), flush=True)
            else:
                run("fla", lambda: chunk_gated_delta_rule(*leaves, scale=1.0)[0], leaves, gdn_core)

    if {"all", "fa4", "fa3", "sdpa"}.intersection(args.cases):
        q, k, v = random_tensor(b, s, qh, d), random_tensor(b, s, kh, d), random_tensor(b, s, kh, d)
        leaves = (q, k, v)

        def sdpa():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = F.scaled_dot_product_attention(
                    *(t.transpose(1, 2) for t in leaves), is_causal=True, enable_gqa=True
                )
            return output.transpose(1, 2)

        run(
            "fa4",
            lambda: flash_attn_func(*leaves, softmax_scale=d**-0.5, causal=True),
            leaves,
            attn_core,
        )
        run("sdpa", sdpa, leaves, attn_core)
        if "fa3" in args.cases or "all" in args.cases:
            try:
                fa3 = get_flash_attention_3()
            except (ImportError, AttributeError, RuntimeError, ValueError, OSError) as exc:
                print(json.dumps({"case": "fa3", "error": str(exc)}), flush=True)
            else:
                run(
                    "fa3",
                    lambda: fa3(*leaves, softmax_scale=d**-0.5, causal=True),
                    leaves,
                    attn_core,
                )

    if {"all", "mlp", "silu", "silu-native", "norm", "norm-compiled"}.intersection(args.cases):
        layer = Qwen35DecoderLayer(config, 3).mlp.cuda().bfloat16()
        configure_main_grad(layer, args.main_grad)
        x = random_tensor(b, s, h)
        run("mlp", lambda: layer(x), (x, *layer.parameters()), mlp_flops)
        gate, up = random_tensor(b * s, i), random_tensor(b * s, i)
        run("silu", lambda: silu_mul(gate, up), (gate, up))
        run("silu-native", lambda: F.silu(gate) * up, (gate, up))
        norm = Qwen35MoeRMSNorm(h).cuda().bfloat16()
        run("norm", lambda: norm(x), (x, *norm.parameters()))
        norm_compiled = torch.compile(norm, fullgraph=True)
        run("norm-compiled", lambda: norm_compiled(x), (x, *norm.parameters()))

    if {"all", "embedding", "head", "head-chunk", "head-native"}.intersection(args.cases):
        ids = torch.randint(vocab, (b, s), device="cuda")
        labels = ids.flatten().roll(-1)
        labels[::17] = -100
        if "embedding" in args.cases or "all" in args.cases:
            embed = nn.Embedding(vocab, h, device="cuda", dtype=torch.bfloat16)
            run("embedding", lambda: embed(ids), embed.parameters())
            del embed
        norm = Qwen35MoeRMSNorm(h).cuda().bfloat16()
        head = BF16Linear(h, vocab, bias=False, device="cuda", dtype=torch.bfloat16)
        configure_main_grad(head, args.main_grad)
        x = random_tensor(b, s, h)
        leaves = (x, *head.parameters(), *norm.parameters())
        run(
            "head",
            lambda: cross_entropy(head(norm(x)).flatten(0, 1), labels),
            leaves,
            6 * h * vocab,
        )
        run(
            "head-native",
            lambda: F.cross_entropy(head(norm(x)).flatten(0, 1).float(), labels),
            leaves,
            6 * h * vocab,
        )

        def chunked():
            normalized = norm(x).flatten(0, 1)
            weights = (labels != -100).sum().clamp_min(1)
            losses = [
                cross_entropy(head(part), target) * (target != -100).sum()
                for part, target in zip(normalized.split(args.chunk), labels.split(args.chunk))
            ]
            return sum(losses) / weights

        run("head-chunk", chunked, leaves, 6 * h * vocab)

    for out_features, in_features in dict.fromkeys(
        [
            (i, h),
            (h, i),
            (2 * gk * kd + gv * vd, h),
            (gv * vd, h),
            (h, gv * vd),
            (gv, h),
            (2 * qh * d, h),
            (kh * d, h),
            (vocab, h),
        ]
    ):
        prefix = f"gemm-{out_features}x{in_features}"
        selected = {f"{prefix}-{leg}" for leg in ("fwd", "dgrad", "wgrad", "main-grad")}
        if not ("gemms" in args.cases or selected.intersection(args.cases)):
            continue
        x = random_tensor(b * s, in_features).detach()
        weight = random_tensor(out_features, in_features).detach()
        dout = random_tensor(b * s, out_features).detach()
        calls = {
            "fwd": lambda: F.linear(x, weight),
            "dgrad": lambda: dout @ weight,
            "wgrad": lambda: dout.t() @ x,
        }
        for leg, operation in calls.items():
            if "gemms" in args.cases or f"{prefix}-{leg}" in args.cases:
                measure(
                    f"{prefix}-{leg}",
                    operation,
                    (),
                    b * s,
                    2 * in_features * out_features,
                    args,
                    False,
                )
        if "gemms" in args.cases or f"{prefix}-main-grad" in args.cases:
            main_grad = torch.zeros_like(weight, dtype=torch.float32)

            def accumulate():
                torch.addmm(main_grad, dout.t(), x, out_dtype=torch.float32, out=main_grad)
                return main_grad

            measure(
                f"{prefix}-main-grad",
                accumulate,
                (),
                b * s,
                2 * in_features * out_features,
                args,
                False,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--inner", type=int, default=8)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--main-grad", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    distributed.cp_size = 1
    training.fp8 = False
    training.Linear = BF16Linear
    config = Qwen3_5TextConfig(**json.loads(CONFIG.read_text()))
    config.max_position_embeddings = args.seq
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "batch": args.batch,
                "args": args.cases,
            }
        ),
        flush=True,
    )
    cases(args, config)


if __name__ == "__main__":
    main()
