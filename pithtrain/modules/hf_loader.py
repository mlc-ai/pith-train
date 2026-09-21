"""
Stream a HuggingFace safetensors checkpoint into a live FSDP-sharded model.

Skips the offline hf2dcp conversion: each rank memmaps only the byte ranges belonging to its own
DTensor shards and copies them straight into _local_tensor. No GPU all-gather. No full-model
materialization on any rank. Multi-rank reads of the same shard files are handled by the OS page
cache via safetensors' mmap.

Generic mapping only (Qwen3, DeepSeek-V2): HF keys have their model. prefix stripped, lm_head.
stays as-is. Quantized (GPT-OSS MXFP4) and fused-expert (Qwen3.5-MoE) checkpoints are not covered
by this loader and must still go through hf2dcp.
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from pithtrain.contexts import logging
from pithtrain.modules.checkpoint import expand_localized_fqn, find_moe, local_shard_range

__all__ = ["load_hf_into_model"]


PlanEntry = Tuple[str, str, torch.Tensor, str, slice, slice]


def _read_weight_map(hf_root: Path) -> Tuple[Dict[str, str], Path]:
    """
    Return (weight_map, shards_dir) from a sharded (index.json) or single-file HF checkpoint.
    """
    index = Path(hf_root, "model.safetensors.index.json")
    if index.is_file():
        with open(index) as f:
            return json.load(f)["weight_map"], hf_root
    single = Path(hf_root, "model.safetensors")
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}, hf_root
    raise FileNotFoundError("No safetensors index or model.safetensors under %s" % hf_root)


def _plan_copies(model: nn.Module, weight_map: Dict[str, str]) -> Tuple[List[PlanEntry], List[str]]:
    """
    Build the per-local-parameter copy plan and the sorted list of unmapped HF keys.

    Each entry is (localized_fqn, hf_key, param_local, shard_file, row_slice, expert_row):
      - Non-expert Shard(0) DTensor: row_slice is this rank's FSDP slice, expert_row is slice(None)
        so the whole local tensor is filled in one copy.
      - Stacked-expert DTensor: one entry per local expert row, row_slice is slice(None) (the whole
        HF per-expert tensor) and expert_row is slice(i, i + 1).
    """
    named_modules = dict(model.named_modules())
    plan: List[PlanEntry] = []
    consumed: set = set()

    for name, param in model.named_parameters():
        if not isinstance(param, DTensor):
            raise RuntimeError(
                "%s is not a DTensor; call load_hf_into_model after apply_fsdp" % name
            )
        if (
            not param.placements
            or not isinstance(param.placements[0], Shard)
            or param.placements[0].dim != 0
        ):
            raise RuntimeError("%s: only Shard(0) placements are supported" % name)

        local = param._local_tensor
        canon_keys = expand_localized_fqn(name, named_modules)
        moe = find_moe(name, named_modules)
        dp_rank = param.device_mesh.get_local_rank()
        dp_size = param.device_mesh.size()

        if moe is None:
            (hf_key,) = canon_keys
            if hf_key not in weight_map:
                raise KeyError("HF checkpoint missing %s for local param %s" % (hf_key, name))
            start, length = local_shard_range(param.shape[0], dp_rank, dp_size)
            assert length == local.shape[0], "%s: computed length %d != local shape %d" % (
                name,
                length,
                local.shape[0],
            )
            plan.append(
                (name, hf_key, local, weight_map[hf_key], slice(start, start + length), slice(None))
            )
            consumed.add(hf_key)
        else:
            dp_offset, dp_len = local_shard_range(moe.experts_per_rank, dp_rank, dp_size)
            assert dp_len == local.shape[0], "%s: expert dp_len %d != local shape %d" % (
                name,
                dp_len,
                local.shape[0],
            )
            for i, hf_key in enumerate(canon_keys[dp_offset : dp_offset + dp_len]):
                if hf_key not in weight_map:
                    raise KeyError(
                        "HF checkpoint missing %s for local param %s expert %d" % (hf_key, name, i)
                    )
                plan.append((name, hf_key, local, weight_map[hf_key], slice(None), slice(i, i + 1)))
                consumed.add(hf_key)

    return plan, sorted(set(weight_map.keys()) - consumed)


def load_hf_into_model(hf_root: Path, model: nn.Module) -> None:
    """
    Copy the HuggingFace safetensors weights under hf_root into model's live DTensor parameters.

    Only Shard(0) DTensor params are supported: FSDP2 attention shards and stacked expert shards
    both fit. Every local parameter is filled exactly once. Unmapped HF keys are logged as
    warnings, tolerating the tied-embeddings case where HF ships a bare lm_head.weight the live
    model does not carry as a separate parameter.
    """
    stdout = logging.stdout
    hf_root = Path(hf_root)
    stdout.info("Load HF checkpoint: %s" % hf_root)
    t0 = time.monotonic()

    weight_map, shards_dir = _read_weight_map(hf_root)
    plan, unmapped = _plan_copies(model, weight_map)

    by_shard: Dict[str, List[Tuple[str, str, torch.Tensor, slice, slice]]] = {}
    for name, hf_key, local, shard, row, expert_row in plan:
        by_shard.setdefault(shard, []).append((name, hf_key, local, row, expert_row))

    filled = 0
    for shard_name in sorted(by_shard):
        entries = by_shard[shard_name]
        stdout.info("Load HF shard: %s (%d tensors)" % (shard_name, len(entries)))
        with safe_open(str(Path(shards_dir, shard_name)), framework="pt", device="cpu") as f:
            for name, hf_key, local, row, expert_row in entries:
                slab = f.get_slice(hf_key)
                src = slab[row] if row != slice(None) else slab[:]
                dst = local[expert_row]
                if tuple(src.shape) != tuple(dst.shape):
                    raise RuntimeError(
                        "%s: shape mismatch, HF %s slice %s vs local %s"
                        % (name, hf_key, tuple(src.shape), tuple(dst.shape))
                    )
                dst.copy_(src.to(dtype=local.dtype))
                filled += 1

    for hf_key in unmapped:
        if hf_key == "lm_head.weight":
            stdout.info("HF has lm_head.weight; model uses tied embeddings, skipping")
        else:
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
        "Load HF checkpoint: %d local tensors filled, elapsed min=%.1fs, max=%.1fs"
        % (filled, dt_min.item(), dt_max.item())
    )
