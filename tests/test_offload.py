"""
Multi-GPU round trip for offload_training_state and reload_training_state.

Run (needs >=2 GPUs; ep-size must divide the ranks of a pipeline stage)::

    torchrun --nproc-per-node=8 tests/test_offload.py
    torchrun --nproc-per-node=8 tests/test_offload.py --model gpt-oss-20b --pp-size 2
    torchrun --nproc-per-node=6 tests/test_offload.py  # uneven shards, so FSDP pads them

Uses a reduced FSDP2/DualPipeV model with Muon and AdamW state from synthetic gradients.
Checks pinning failure, memory release, entry-point guards, bitwise reload and pinned-buffer reuse.
No forward pass or attention kernel is needed.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from pithtrain.contexts import training
from pithtrain.modules.checkpoint import iter_canonical_parameters, load_checkpoint, save_checkpoint
from pithtrain.modules.distributed import setup_distributed
from pithtrain.modules.logging import setup_logging
from pithtrain.modules.training import (
    make_constant_scheduler,
    make_muon_optimizer,
    offload_buffer_pool,
    offload_training_state,
    reload_training_state,
    setup_model,
    take_pinned_buffer,
)
from pithtrain.tasks.pretrain_lm import PretrainLMCfg

NUM_LAYERS, NUM_EXPERTS, VOCAB_SIZE = 4, 16, 8192
CYCLES = 3

MODELS = {
    "deepseek-v2-lite": "examples/pretrain_lm/deepseek-v2-lite/config.json",
    "qwen3-30b-a3b": "examples/pretrain_lm/qwen3-30b-a3b/config.json",
    "gpt-oss-20b": "examples/pretrain_lm/gpt-oss-20b/config.json",  # expert biases go to AdamW
    "qwen3.5-35b-a3b": "examples/pretrain_lm/qwen3.5-35b-a3b/config.json",
}


def local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def state_tensors():
    yield from training.model.named_parameters()
    names = {p: n for n, p in training.model.named_parameters()}
    for optimizer in training.optimizers:
        for p, state in optimizer.state.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    yield f"{names[p]}:{key}", value


def snapshot() -> dict[str, torch.Tensor]:
    return {n: local(t).detach().cpu().clone() for n, t in state_tensors()}


def assert_bitwise(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], what: str) -> None:
    assert a.keys() == b.keys(), what
    bad = [n for n in a if not torch.equal(a[n], b[n])]
    assert not bad, (what, bad[:3])


def requested() -> int:
    """
    Exclude allocator rounding, which can differ after reloading the same tensors.
    """
    return torch.cuda.memory_stats()["requested_bytes.all.current"]


def raises(fn, text: str) -> bool:
    try:
        fn()
    except Exception as e:
        return text in str(e)
    return False


def pooled() -> list[int]:
    return sorted(buf.data_ptr() for bufs in offload_buffer_pool.values() for buf in bufs)


def main(scratch: Path):
    rank = torch.distributed.get_rank()

    def rprint(*a):
        if rank == 0:
            print(*a, flush=True)

    torch.manual_seed(1)
    for p in training.model.parameters():
        p.grad = torch.randn_like(p).mul_(0.01)
    for optimizer in training.optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    snap = snapshot()
    full = {n: p.full_tensor().cpu() for n, p in training.model.named_parameters()}
    storages = {}
    for _, t in state_tensors():
        if local(t).is_cuda:
            storages.setdefault(local(t).untyped_storage().data_ptr(), local(t).untyped_storage())
    state_bytes = sum(s.nbytes() for s in storages.values())
    kinds = {n.rsplit(":", 1)[-1] for n in snap if ":" in n}
    assert {"momentum_buffer", "exp_avg", "exp_avg_sq"} <= kinds, kinds

    reload_training_state()
    assert_bitwise(snapshot(), snap, "reload with nothing offloaded")

    # Fail after one queued copy; retry must offload the entire state.
    fail_second = dict(
        wraps=take_pinned_buffer, side_effect=[mock.DEFAULT, RuntimeError("no memory")]
    )
    with mock.patch("pithtrain.modules.training.take_pinned_buffer", **fail_second):
        assert raises(offload_training_state, "no memory")
    assert not training.offloaded
    assert_bitwise(snapshot(), snap, "failed offload")

    fsdp_modules = [m for m in training.model.modules() if isinstance(m, FSDPModule)]
    for cycle in range(CYCLES):
        before = requested()
        if cycle == 0:  # offload must also free what a caller left unsharded
            for module in fsdp_modules:
                module.unshard()
        reserved = torch.cuda.memory_reserved()
        offload_training_state()
        after = requested()
        assert before - after >= state_bytes, (cycle, before - after, state_bytes)
        assert torch.cuda.memory_reserved() < reserved, (cycle, reserved)
        for n, t in state_tensors():
            assert not local(t).is_cuda or local(t).untyped_storage().size() == 0, n

        offload_training_state()
        assert training.offloaded and requested() == after
        assert raises(lambda: training.model.step([], None), "offloaded")
        assert raises(lambda: next(iter_canonical_parameters()), "offloaded")
        assert raises(lambda: save_checkpoint(scratch / "checkpoint", 1), "offloaded")
        assert raises(lambda: load_checkpoint(scratch / "checkpoint", 1), "offloaded")

        reload_training_state()
        assert not training.offloaded and requested() == before
        assert_bitwise(snapshot(), snap, f"reload {cycle}")

        if cycle == 0:
            pool = pooled()
            pool_bytes = sum(b.numel() for bufs in offload_buffer_pool.values() for b in bufs)
            assert pool_bytes == state_bytes, (pool_bytes, state_bytes)
        assert pooled() == pool, f"cycle {cycle} allocated new pinned buffers"
        rprint(
            f"[INFO] cycle {cycle}: freed {(before - after) / 2**20:.1f} MiB "
            f"(state {state_bytes / 2**20:.1f} MiB, {len(storages)} storages)"
        )

    for module in fsdp_modules:
        module.unshard()
    for n, p in training.model.named_parameters():
        assert torch.equal(p, full[n].to(p.device, p.dtype)), n
    for module in fsdp_modules:
        module.reshard()
    rprint(f"[INFO] {len(snap)} tensors round-trip bitwise over {CYCLES} cycles")
    rprint("[PASS] test_offload")


def _entry():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--model", choices=list(MODELS), default="qwen3-30b-a3b")
    parsed = parser.parse_args()

    cfg = PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size = parsed.pp_size
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.expert_parallel_size = parsed.ep_size
    t = cfg.training
    t.optimizer, t.scheduler = make_muon_optimizer, make_constant_scheduler
    t.lr, t.sequence_length = 4.2e-4, 2048

    setup_logging(cfg)
    setup_distributed(cfg)

    scratch = Path(tempfile.gettempdir(), "pithtrain_test_offload")
    if torch.distributed.get_rank() == 0:
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        src = Path(__file__).resolve().parent.parent / MODELS[parsed.model]
        config = json.loads(src.read_text())
        config["num_hidden_layers"], config["vocab_size"] = NUM_LAYERS, VOCAB_SIZE
        if "layer_types" in config:
            config["layer_types"] = config["layer_types"][:NUM_LAYERS]
        for key in ("num_experts", "n_routed_experts", "num_local_experts"):
            if key in config:
                config[key] = NUM_EXPERTS
        (scratch / "config.json").write_text(json.dumps(config))
        print(f"[INFO] model={parsed.model} pp={parsed.pp_size} ep={parsed.ep_size}", flush=True)
    torch.distributed.barrier()
    t.model = scratch

    torch.manual_seed(0)
    setup_model(cfg.training, cfg.distributed)
    training.optimizers = cfg.training.optimizer(cfg.training)
    training.schedulers = cfg.training.scheduler(cfg.training)
    main(scratch)


if __name__ == "__main__":
    _entry()
