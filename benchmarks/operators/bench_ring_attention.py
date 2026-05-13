"""Benchmark forward and backward latency of ring attention."""

import argparse
import sys
from types import SimpleNamespace

import torch

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx, distributed_context
from pithtrain.operators.ring_attention import ring_attention_func


def run(ctx: DistributedCtx, args: argparse.Namespace) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_size = cp_group.size()
    device = torch.cuda.current_device()
    softmax_scale = args.D**-0.5
    S_local = args.S // cp_size

    torch.manual_seed(42)
    kwargs = dict(device=device, dtype=torch.bfloat16)
    q = torch.randn(args.B, S_local, args.HQ, args.D, requires_grad=True, **kwargs)
    k = torch.randn(args.B, S_local, args.HK, args.D, requires_grad=True, **kwargs)
    v = torch.randn(args.B, S_local, args.HK, args.D, requires_grad=True, **kwargs)
    grad_out = torch.randn(args.B, S_local, args.HQ, args.D, **kwargs)

    def once() -> None:
        q.grad, k.grad, v.grad = None, None, None
        out = ring_attention_func(q, k, v, softmax_scale, cp_group)
        out.backward(grad_out)

    # Warmup
    for _ in range(args.warmup):
        once()
    torch.cuda.synchronize()

    # Timed forward/backward, separated by CUDA events.
    fwd_total_ms = 0.0
    bwd_total_ms = 0.0
    for _ in range(args.niters):
        q.grad, k.grad, v.grad = None, None, None
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record()
        out = ring_attention_func(q, k, v, softmax_scale, cp_group)
        fwd_end.record()
        out.backward(grad_out)
        bwd_end.record()
        torch.cuda.synchronize()
        fwd_total_ms += fwd_start.elapsed_time(fwd_end)
        bwd_total_ms += fwd_end.elapsed_time(bwd_end)

    fwd_avg = fwd_total_ms / args.niters
    bwd_avg = bwd_total_ms / args.niters

    if ctx.rank == 0:
        print(f"B={args.B}, S={args.S}, HQ={args.HQ}, HK={args.HK}, D={args.D}, CP={args.cp_size}")
        print(f"fwd: {fwd_avg:7.3f} ms")
        print(f"bwd: {bwd_avg:7.3f} ms")
        sys.stdout.flush()

    # Profile capture: one iteration between cudaProfilerStart and cudaProfilerStop.
    # nsys with --capture-range=cudaProfilerApi records only this region.
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    once()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, required=True, help="Batch size")
    p.add_argument("--S", type=int, required=True, help="Sequence length")
    p.add_argument("--HQ", type=int, required=True, help="Number of query heads")
    p.add_argument("--HK", type=int, required=True, help="Number of key/value heads")
    p.add_argument("--D", type=int, required=True, help="Head dimension")
    p.add_argument("--cp-size", type=int, required=True, help="Context parallel size")
    p.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    p.add_argument("--niters", type=int, default=100, help="Timed iterations")
    args = p.parse_args()

    cfg = DistributedCfg()
    cfg.context_parallel_size = args.cp_size
    parent_cfg = SimpleNamespace(distributed=cfg)
    parent_ctx = SimpleNamespace(distributed=DistributedCtx())
    with distributed_context(parent_cfg, parent_ctx) as ctx:
        run(ctx, args)


if __name__ == "__main__":
    main()
