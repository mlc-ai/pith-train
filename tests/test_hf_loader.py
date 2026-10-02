"""
Bit-exact test for load_hf_into_model.

Loads the same HF safetensors checkpoint two ways into an FSDP-sharded model:
  A. Reference: rank 0 runs hf2dcp into a scratch DCP directory, then every rank loads it via
     the existing dcp.load path.
  B. Under test: every rank streams straight from the HF shards via load_hf_into_model.

Then asserts every local DTensor shard matches per parameter. Skips when the HF checkpoint is not
present so the test is safe to include in a pytest sweep on a fresh workspace.
"""

import argparse
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from pithtrain.contexts import distributed, training
from pithtrain.modules.checkpoint import load_checkpoint
from pithtrain.modules.distributed import DistributedCfg, setup_distributed
from pithtrain.modules.hf_loader import load_hf_into_model
from pithtrain.modules.training import TrainingCfg, setup_model
from pithtrain.tasks.convert_checkpoint._core import ConvertCheckpointCfg, hf2dcp


def _hf_present(hf_ckpt: Path) -> bool:
    return (
        Path(hf_ckpt, "model.safetensors.index.json").is_file()
        or Path(hf_ckpt, "model.safetensors").is_file()
    )


def _broadcast_present(present: bool) -> bool:
    payload = [present]
    torch.distributed.broadcast_object_list(payload, src=0)
    return bool(payload[0])


def _snapshot_locals(model) -> dict:
    return {n: p._local_tensor.detach().clone() for n, p in model.named_parameters()}


def _build_model(cfg) -> None:
    # Match setup_training's RNG reset so both paths produce identical constructor init for
    # params the loader disowns (Qwen3.5 GDN A_log uses uniform_(0, 16) at __init__ time).
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    setup_model(cfg.training, cfg.distributed)
    # load_checkpoint reads training.optimizers/schedulers; the HF paths are model-only so empty
    # tuples suffice.
    training.optimizers = ()
    training.schedulers = ()


def main(model_config: str, hf_ckpt: Path) -> None:
    if distributed.rank == 0:
        print("[INFO] test_hf_loader model=%s hf_ckpt=%s" % (model_config, hf_ckpt), flush=True)
        print(
            "[INFO] pp=%d ep=%d cp=%d"
            % (distributed.pp_size, distributed.ep_size, distributed.cp_size),
            flush=True,
        )

    present = _broadcast_present(distributed.rank == 0 and _hf_present(hf_ckpt))
    if not present:
        if distributed.rank == 0:
            print("[SKIP] HF checkpoint not present at %s" % hf_ckpt, flush=True)
        return

    cfg = SimpleNamespace()
    cfg.distributed = DistributedCfg()
    cfg.distributed.pipeline_parallel_size = distributed.pp_size
    cfg.distributed.expert_parallel_size = distributed.ep_size
    cfg.distributed.context_parallel_size = distributed.cp_size

    cfg.training = TrainingCfg()
    cfg.training.model = Path(model_config).parent
    cfg.training.sequence_length = 128

    # Path A: reference. Rank 0 runs the offline converter into a scratch DCP dir; every rank
    # loads it through the existing dcp.load path so the reference exercises the same code the
    # production resume path does.
    tmp_root = None
    if distributed.rank == 0:
        tmp_root = Path(tempfile.mkdtemp(prefix="hf_loader_ref_"))
        dcp_dir = tmp_root / "torch-dcp" / "00000000"
        dcp_dir.mkdir(parents=True)
        convert_cfg = ConvertCheckpointCfg()
        convert_cfg.operation = "hf2dcp"
        convert_cfg.load_path = hf_ckpt
        convert_cfg.save_path = dcp_dir
        import logging as py_logging

        hf2dcp(convert_cfg, py_logging.getLogger("test_hf_loader"))
    payload = [tmp_root]
    torch.distributed.broadcast_object_list(payload, src=0)
    tmp_root = Path(payload[0])
    torch.distributed.barrier()

    _build_model(cfg)
    load_checkpoint(tmp_root, 0)
    ref = _snapshot_locals(training.model)
    torch.distributed.barrier()

    # Path B: under test. Rebuild a fresh model to guarantee no state leaks, then stream weights
    # in through the new loader.
    _build_model(cfg)
    load_hf_into_model(hf_ckpt, training.model)
    got = _snapshot_locals(training.model)
    torch.distributed.barrier()

    names = sorted(set(ref.keys()) & set(got.keys()))
    missing_ref = sorted(set(got.keys()) - set(ref.keys()))
    missing_got = sorted(set(ref.keys()) - set(got.keys()))
    if missing_ref or missing_got:
        raise RuntimeError(
            "param name mismatch: only-in-loader=%s only-in-reference=%s"
            % (missing_ref, missing_got)
        )

    failed = []
    largest_diff = 0.0
    largest_diff_name = None
    for n in names:
        r, g = ref[n], got[n]
        if r.shape != g.shape:
            failed.append((n, "shape %s vs %s" % (tuple(r.shape), tuple(g.shape))))
            continue
        if not torch.equal(r, g):
            diff = (r.to(torch.float32) - g.to(torch.float32)).abs().max().item()
            if diff > largest_diff:
                largest_diff, largest_diff_name = diff, n
            failed.append((n, "max_abs_diff=%.3e" % diff))

    for rank in range(distributed.world_size):
        if rank == distributed.rank:
            if failed:
                for n, why in failed[:10]:
                    print("[ERROR] rank-%d %s: %s" % (distributed.rank, n, why), flush=True)
                print(
                    "[ERROR] rank-%d %d/%d params mismatched, largest diff %.3e at %s"
                    % (distributed.rank, len(failed), len(names), largest_diff, largest_diff_name),
                    flush=True,
                )
            else:
                print(
                    "[INFO] rank-%d %d params matched bit-exact" % (distributed.rank, len(names)),
                    flush=True,
                )
        torch.distributed.barrier()

    ok = torch.tensor(0 if failed else 1, device="cuda")
    torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
    assert ok.item() == 1, "bit-exact check failed on at least one rank"

    if distributed.rank == 0:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print("[INFO] OK: load_hf_into_model matches hf2dcp -> load_checkpoint", flush=True)


@record
def _entry() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hf-ckpt", type=str, required=True)
    parsed = parser.parse_args()

    cfg = SimpleNamespace()
    cfg.distributed = DistributedCfg()
    cfg.distributed.pipeline_parallel_size = parsed.pp_size
    cfg.distributed.expert_parallel_size = parsed.ep_size
    cfg.distributed.context_parallel_size = parsed.cp_size

    setup_distributed(cfg)
    main(parsed.model, Path(parsed.hf_ckpt))


if __name__ == "__main__":
    _entry()
