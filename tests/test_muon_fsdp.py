"""
Multi-GPU correctness for Muon's distributed Newton-Schulz.

Run (pure DP)::

    torchrun --nproc-per-node=2 tests/test_muon_fsdp.py
    torchrun --nproc-per-node=8 tests/test_muon_fsdp.py

Run (expert-parallel; experts and dense weights land on different meshes)::

    torchrun --nproc-per-node=8 tests/test_muon_fsdp.py --ep-size 2

Mirrors ``apply_fsdp``: experts shard on the ``(dp, cp)`` mesh, everything else
on the flattened ``(dp, cp, ep)`` mesh. Injects identical grads into (a) an
unsharded reference holding this EP rank's slice and (b) the FSDP-sharded model,
steps both with ``[Muon, AdamW]``, and asserts the sharded update matches the
reference. fp32 params, so any mismatch is plumbing, not bf16 NS noise (the
gathered matrix -- hence the NS -- is identical on both sides).
"""

import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate
from torch.optim import AdamW

from pithtrain.layers.group_linear import GroupLinear
from pithtrain.modules.distributed import DistributedCfg, DistributedCtx, distributed_context
from pithtrain.modules.optimizer import Muon
from pithtrain.modules.training import is_muon_param

NUM_EXPERTS = 8  # divisible by the EP sizes we test (1, 2, 4)


def step_composed(model, lr):
    """Step the composed optimizers as training does: Muon for hidden weights,
    AdamW for the rest (separable, so it works element-wise on each shard)."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        (muon_params if is_muon_param(name, p) else adamw_params).append(p)
    Muon(muon_params, lr=lr).step()
    if adamw_params:
        AdamW(adamw_params, lr=lr, weight_decay=0.0).step()


class _Model(nn.Module):
    """Spans both optimizer groups and both mesh shapes (2D weight, 3D experts,
    1D/2D Adam)."""

    def __init__(self, num_experts: int):
        super().__init__()
        self.q_proj = nn.Linear(64, 48, bias=False)  # 2D -> Muon
        self.experts = GroupLinear(num_experts, 64, 32)  # 3D experts -> Muon
        self.norm = nn.LayerNorm(64)  # 1D weight + bias -> Adam
        self.embed_tokens = nn.Embedding(100, 64)  # 2D, name-excluded -> Adam
        # GroupLinear uses torch.empty (the framework fills it via init_weights
        # before FSDP); init it here so the optimizer steps real values, not
        # uninitialized memory (differs per rank, would break the comparison).
        nn.init.normal_(self.experts.weight, std=0.02)


def main(ctx: DistributedCtx):
    mesh = ctx.device_mesh
    ep_size, ep_rank = ctx.ep_size, ctx.ep_rank
    # Same split as training.apply_fsdp: experts on (dp, cp), the rest on the
    # flattened (dp, cp, ep) mesh. At ep=cp=1 both collapse to pure dp.
    moe_mesh = mesh["dp", "cp"]._flatten()
    other_mesh = mesh["dp", "cp", "ep"]._flatten()
    device = torch.device("cuda", ctx.local_rank)
    lr = 0.1

    # Full weights + grads, identical on every rank (same seed).
    torch.manual_seed(0)
    full = _Model(NUM_EXPERTS).to(device=device, dtype=torch.float32)
    full_state = {n: p.detach().clone() for n, p in full.named_parameters()}
    torch.manual_seed(123)
    full_grads = {n: torch.randn_like(p) for n, p in full.named_parameters()}

    # This EP rank owns experts[lo:hi]; dense weights are replicated across EP.
    k = NUM_EXPERTS // ep_size
    lo, hi = ep_rank * k, (ep_rank + 1) * k

    def ep_slice(name, t):
        return t[lo:hi] if name == "experts.weight" else t

    # Reference: unsharded model with this EP rank's slice, stepped with Muon.
    ref = _Model(k).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            p.copy_(ep_slice(n, full_state[n]))
    for n, p in ref.named_parameters():
        p.grad = ep_slice(n, full_grads[n]).clone()
    step_composed(ref, lr)
    ref_after = {n: p.detach().clone() for n, p in ref.named_parameters()}

    # FSDP-sharded model: same slice + grads, sharded on the two meshes.
    shd = _Model(k).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for n, p in shd.named_parameters():
            p.copy_(ep_slice(n, full_state[n]))
    fully_shard(shd.experts, mesh=moe_mesh)
    fully_shard(shd.q_proj, mesh=other_mesh)
    fully_shard(shd.norm, mesh=other_mesh)
    fully_shard(shd.embed_tokens, mesh=other_mesh)
    fully_shard(shd, mesh=other_mesh)

    for n, p in shd.named_parameters():
        g = ep_slice(n, full_grads[n])
        g_full = DTensor.from_local(
            g, p.device_mesh, [Replicate()] * p.device_mesh.ndim, run_check=False
        )
        p.grad = g_full.redistribute(placements=p.placements)
    step_composed(shd, lr)

    max_diff, worst = 0.0, None
    for n, p in shd.named_parameters():
        full_p = p.full_tensor() if isinstance(p, DTensor) else p
        diff = (full_p - ref_after[n]).abs().max().item()
        if diff > max_diff:
            max_diff, worst = diff, n
        assert torch.allclose(full_p, ref_after[n], atol=1e-3, rtol=1e-3), (n, diff)

    if ctx.rank == 0:
        print(
            f"[INFO] Muon FSDP step matches single-process reference "
            f"(dp={ctx.dp_size} cp={ctx.cp_size} ep={ep_size}). "
            f"max|diff|={max_diff:.2e} (worst param: {worst})",
            flush=True,
        )
        print("[PASS] test_muon_fsdp", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parsed = parser.parse_args()

    cfg, ctx = SimpleNamespace(), SimpleNamespace()
    cfg.distributed = DistributedCfg()
    cfg.distributed.pipeline_parallel_size = 1
    cfg.distributed.expert_parallel_size = parsed.ep_size
    cfg.distributed.context_parallel_size = parsed.cp_size
    ctx.distributed = DistributedCtx()
    with distributed_context(cfg, ctx):
        main(ctx.distributed)
