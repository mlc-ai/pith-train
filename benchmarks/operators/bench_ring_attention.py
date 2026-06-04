"""Benchmark forward and backward latency of ring attention."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx, distributed_context
from pithtrain.operators.ring_attention import ring_attention_func


def parse_scenario(scenario: str) -> tuple[dict, int, int]:
    m = re.match(r"^(.+)-cp(\d+)-s(\d+)k$", scenario)
    if not m:
        raise ValueError(f"invalid scenario '{scenario}', expected <model>-cp<N>-s<N>k")
    model = m.group(1)
    with open(Path(f"examples/pretrain_lm/{model}/config.json")) as f:
        config = json.load(f)
    return config, int(m.group(2)), int(m.group(3)) * 1024


def run(ctx: DistributedCtx, scenario: str, config: dict, cp_size: int, S: int) -> None:
    B = 1
    WARMUP, NITERS = 25, 100
    HQ, HK = config["num_attention_heads"], config["num_key_value_heads"]
    D = config["head_dim"]

    cp_group = ctx.device_mesh.get_group("cp")
    device = torch.cuda.current_device()
    softmax_scale = D**-0.5
    S_local = S // cp_size

    torch.manual_seed(42)
    kwargs = dict(device=device, dtype=torch.bfloat16)
    q = torch.randn(B, S_local, HQ, D, requires_grad=True, **kwargs)
    k = torch.randn(B, S_local, HK, D, requires_grad=True, **kwargs)
    v = torch.randn(B, S_local, HK, D, requires_grad=True, **kwargs)
    grad_out = torch.randn(B, S_local, HQ, D, **kwargs)

    def run_once() -> None:
        q.grad, k.grad, v.grad = None, None, None
        out = ring_attention_func(q, k, v, softmax_scale, cp_group)
        out.backward(grad_out)

    for _ in range(WARMUP):
        run_once()
    torch.cuda.synchronize()

    # Timed forward/backward, separated by CUDA events.
    fwd_total_ms = 0.0
    bwd_total_ms = 0.0
    for _ in range(NITERS):
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

    fwd_avg = fwd_total_ms / NITERS
    bwd_avg = bwd_total_ms / NITERS

    if ctx.rank == 0:
        print(f"{scenario} | fwd: {fwd_avg:7.3f} ms , bwd: {bwd_avg:7.3f} ms", flush=True)
    torch.distributed.barrier()

    # Nsys profile capture with one iteration.
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    run_once()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    scenario = sys.argv[1]
    config, cp_size, S = parse_scenario(scenario)

    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    parent_cfg = SimpleNamespace(distributed=cfg)
    parent_ctx = SimpleNamespace(distributed=DistributedCtx())
    with distributed_context(parent_cfg, parent_ctx) as ctx:
        run(ctx, scenario, config, cp_size, S)
