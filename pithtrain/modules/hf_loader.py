"""
Stream a HuggingFace safetensors checkpoint into a live FSDP-sharded model, skipping hf2dcp.

Each rank reads only the rows of its own shards from the safetensors files. Per-model quirks live
in optional hooks on the matching converter in ``convert_checkpoint._registry``.
"""

import contextlib
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open

from pithtrain.contexts import logging
from pithtrain.tasks.convert_checkpoint._registry import CONVERTERS, ModelConverter
from pithtrain.tasks.convert_checkpoint._streaming import CopyOp, generic_plan_param

__all__ = ["load_hf_into_model"]


def _plan_copies(
    model: nn.Module, weight_map: Dict[str, str], converter: Optional[ModelConverter]
) -> Tuple[List[CopyOp], set]:
    named_modules = dict(model.named_modules())
    plan: List[CopyOp] = []
    consumed: set = set()
    plan_param = getattr(converter, "plan_param", None) or generic_plan_param

    for name, param in model.named_parameters():
        ops = plan_param(name, param, weight_map, named_modules)
        plan.extend(ops)
        for op in ops:
            consumed.update(op.hf_keys)

    return plan, consumed


def load_hf_into_model(hf_root: Path, model: nn.Module) -> None:
    """
    Copy the HuggingFace safetensors weights under hf_root into model's live DTensor parameters.
    Only Shard(0) params are supported.
    """
    stdout = logging.stdout
    hf_root = Path(hf_root)
    stdout.info("Load HF checkpoint: %s" % hf_root)
    t0 = time.monotonic()

    converter = next((c for c in CONVERTERS if c.detect_hf(hf_root)), None)
    stdout.info("Selected HF loader: %s" % (converter.name if converter else "generic"))

    with open(Path(hf_root, "model.safetensors.index.json")) as f:
        raw = json.load(f)["weight_map"]
    # Planners look up keys without the "model." prefix; reads use the name stored in the file.
    file_key = {k.removeprefix("model."): k for k in raw}
    weight_map = {k: raw[v] for k, v in file_key.items()}
    plan, consumed = _plan_copies(model, weight_map, converter)

    by_shard: Dict[str, List[CopyOp]] = {}
    for op in plan:
        by_shard.setdefault(op.shard_files[0], []).append(op)

    all_shards = sorted({s for op in plan for s in op.shard_files})
    filled = 0
    with contextlib.ExitStack() as stack:
        handles: Dict[str, "safe_open"] = {
            shard: stack.enter_context(
                safe_open(str(Path(hf_root, shard)), framework="pt", device="cpu")
            )  # fmt: skip
            for shard in all_shards
        }
        for shard_name in sorted(by_shard):
            entries = by_shard[shard_name]
            stdout.info("Load HF shard: %s (%d ops)" % (shard_name, len(entries)))
            for op in entries:
                srcs = [
                    handles[shard].get_slice(file_key[key])[op.row_slice]
                    for key, shard in zip(op.hf_keys, op.shard_files)
                ]
                src = op.transform(*srcs, dtype=op.dst.dtype) if op.transform else srcs[0]
                if tuple(src.shape) != tuple(op.dst.shape):
                    raise RuntimeError(
                        "%s: shape mismatch, HF %s slice %s vs local %s"
                        % (op.local_fqn, op.hf_keys, tuple(src.shape), tuple(op.dst.shape))
                    )
                op.dst.copy_(src.to(dtype=op.dst.dtype))
                filled += 1

    # Each rank loads only its own shards, so a key is unmapped only if no rank consumed it.
    gathered: List[Optional[set]] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, consumed)
    unmapped = set(weight_map.keys()).difference(*gathered)
    # An unloaded lm_head.weight means the model ties embeddings: expected, not a warning.
    expected = {"lm_head.weight"} & unmapped
    if hasattr(converter, "expected_unmapped"):
        expected |= converter.expected_unmapped(unmapped)
    for hf_key in sorted(expected):
        if hf_key == "lm_head.weight":
            stdout.info("HF has lm_head.weight; model uses tied embeddings, skipping")
        else:
            stdout.info("Expected unmapped HF key: %s" % hf_key)
    if torch.distributed.get_rank() == 0:
        for hf_key in sorted(unmapped - expected):
            stdout.warning("Unmapped HF key: %s" % hf_key)

    del plan, by_shard
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.barrier()

    dt = torch.tensor(time.monotonic() - t0, device="cuda")
    dt_min, dt_max = dt.clone(), dt.clone()
    torch.distributed.all_reduce(dt_min, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(dt_max, op=torch.distributed.ReduceOp.MAX)
    stdout.info(
        "Load HF checkpoint: %d ops, elapsed min=%.1fs, max=%.1fs"
        % (filled, dt_min.item(), dt_max.item())
    )
