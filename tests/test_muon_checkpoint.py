"""
Multi-GPU checkpoint round-trip for the composed Muon + AdamW optimizer.

Run (needs >=2 GPUs; ep-size must divide the world)::

    torchrun --nproc-per-node=8 tests/test_muon_checkpoint.py
    torchrun --nproc-per-node=8 tests/test_muon_checkpoint.py --model gpt-oss-20b

Builds a real model (``--model``: deepseek-v2-lite, qwen3-30b-a3b, gpt-oss-20b;
reduced to a few layers) with FSDP2 + DualPipeV, steps the composed
``[Muon, AdamW]`` on synthetic grads to populate state, then
save_checkpoint -> fresh optimizers -> load_checkpoint and asserts the optimizer
state round-trips exactly. Exercises the resharding (``to_canonical_optim`` /
``to_localized_optim``) for a combined multi-optimizer state dict over stacked
experts (expanded to per-expert FQNs on disk) and DualPipeV ``module.N.``
prefixes. No forward pass, so no dataset and no attention kernel needed.
"""

import argparse
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

from pithtrain.modules.distributed import distributed_context
from pithtrain.modules.logging import logging_context
from pithtrain.modules.training import setup_model, setup_optimizer, setup_scheduler
from pithtrain.tasks.pretrain_lm import (
    PretrainLMCfg,
    PretrainLMCtx,
    load_checkpoint,
    save_checkpoint,
)

NUM_LAYERS = 4  # a few MoE layers; small enough to build quickly

MODELS = {
    "deepseek-v2-lite": "examples/pretrain_lm/deepseek-v2-lite/config.json",  # MLA, GroupLinear experts
    "qwen3-30b-a3b": "examples/pretrain_lm/qwen3-30b-a3b/config.json",  # GQA + q/k_norm
    "gpt-oss-20b": "examples/pretrain_lm/gpt-oss-20b/config.json",  # sinks, fused gate_up, expert biases
}


def opt_state_snapshot(optimizers, model):
    """fqn -> {state_key: cpu fp32 tensor (or scalar)} across all optimizers."""
    param_to_fqn = {p: n for n, p in model.named_parameters()}
    snap = {}
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                entry = {}
                for k, v in opt.state.get(p, {}).items():
                    if isinstance(v, torch.Tensor):
                        full = v.full_tensor() if isinstance(v, DTensor) else v
                        entry[k] = full.detach().float().cpu().clone()
                    else:
                        entry[k] = v
                snap[param_to_fqn[p]] = entry
    return snap


def main(cfg: PretrainLMCfg, ctx: PretrainLMCtx):
    rank = torch.distributed.get_rank()

    def rprint(*a):
        if rank == 0:
            print(*a, flush=True)

    model = ctx.training.model
    optimizers = ctx.training.optimizers
    n_muon = sum(len(g["params"]) for g in optimizers[0].param_groups)
    n_aux = sum(len(g["params"]) for g in optimizers[1].param_groups)
    rprint(f"[INFO] composed optimizers: Muon over {n_muon} params, AdamW over {n_aux} params")

    # Synthetic grads -> step to populate optimizer state (no forward needed).
    torch.manual_seed(1234 + rank)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p).mul_(0.01)
    for opt in optimizers:
        opt.step()
    # Clear grads so the load-side _init_optim_state materializes the state
    # template (it skips if any grad is set) -- as in the real resume flow.
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    bad = [
        n
        for n, p in model.named_parameters()
        if not torch.isfinite(p.to_local() if isinstance(p, DTensor) else p).all()
    ]
    assert not bad, ("non-finite params after step", bad[:3])

    # Snapshot, save, rebuild fresh optimizers, load, compare.
    ctx.training.step = 1
    before = opt_state_snapshot(optimizers, model)
    save_checkpoint(cfg, ctx)

    setup_optimizer(cfg.training, ctx.training)  # fresh, empty-state optimizers
    setup_scheduler(cfg.training, ctx.training)
    load_checkpoint(cfg, ctx)
    after = opt_state_snapshot(ctx.training.optimizers, model)

    assert set(before) == set(after), "param FQN set changed across the round-trip"
    max_diff, worst = 0.0, None
    for n in before:
        assert set(before[n]) == set(after[n]), (n, set(before[n]), set(after[n]))
        for k in before[n]:
            a, b = before[n][k], after[n][k]
            if isinstance(a, torch.Tensor):
                diff = (a - b).abs().max().item()
                if diff > max_diff:
                    max_diff, worst = diff, (n, k)
            else:
                assert a == b, (n, k, a, b)

    muon_fqns = [n for n in before if "momentum_buffer" in before[n]]
    aux_fqns = [n for n in before if "exp_avg" in before[n]]
    assert muon_fqns and aux_fqns, "expected both Muon and aux optimizer state"
    assert max_diff < 1e-3, f"optimizer state mismatch after round-trip: {max_diff}"
    if rank == 0:
        print(
            f"[INFO] optimizer state round-trips: {len(muon_fqns)} Muon (momentum_buffer), "
            f"{len(aux_fqns)} aux (exp_avg/exp_avg_sq). max|diff|={max_diff:.2e} (worst: {worst})",
            flush=True,
        )
        print("[PASS] test_muon_checkpoint", flush=True)


def _entry():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--model", choices=list(MODELS), default="deepseek-v2-lite")
    parsed = parser.parse_args()

    cfg = PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size = 1
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.expert_parallel_size = parsed.ep_size
    t = cfg.training
    t.optimizer = "Muon"
    t.scheduler = "CosineAnnealing"
    t.max_lr = 4.2e-4
    t.min_lr = 1.0e-5
    t.warmup_steps = 8
    t.max_steps = 40
    t.micro_batch_size = 1
    t.global_batch_size = 64
    t.sequence_length = 2048
    t.moe_load_balance_type = "sequence"
    t.moe_load_balance_coef = 3e-3
    t.fp8_training = "disabled"

    ctx = PretrainLMCtx()
    with ExitStack() as stack:
        stack.enter_context(logging_context(cfg, ctx))
        stack.enter_context(distributed_context(cfg, ctx))

        # Reduced config + checkpoint dir on local scratch (rank 0 writes).
        scratch = Path(tempfile.gettempdir(), "pithtrain_test_muon_checkpoint")
        if torch.distributed.get_rank() == 0:
            print(
                f"[INFO] model={parsed.model}, ep={parsed.ep_size}, layers={NUM_LAYERS}", flush=True
            )
            shutil.rmtree(scratch, ignore_errors=True)
            scratch.mkdir(parents=True)
            src = Path(__file__).resolve().parent.parent / MODELS[parsed.model]
            config = json.loads(src.read_text())
            config["num_hidden_layers"] = NUM_LAYERS
            if "layer_types" in config:  # gpt-oss alternates sliding/full attention per layer
                config["layer_types"] = config["layer_types"][:NUM_LAYERS]
            (scratch / "config.json").write_text(json.dumps(config))
        torch.distributed.barrier()
        t.model = scratch / "config.json"
        t.dataset = scratch  # unused; set to satisfy the config
        t.save_location = scratch / "checkpoint"

        # Build model + optimizers + schedulers directly (no dataset needed).
        ctx.training.step = 0
        torch.manual_seed(0)
        setup_model(cfg.training, ctx.training, cfg.distributed, ctx.distributed)
        setup_optimizer(cfg.training, ctx.training)
        setup_scheduler(cfg.training, ctx.training)
        main(cfg, ctx)


if __name__ == "__main__":
    _entry()
