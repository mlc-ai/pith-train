"""
Stream a HuggingFace safetensors checkpoint into a live FSDP-sharded model.

Skips the offline hf2dcp conversion: each rank memmaps only the byte ranges belonging to its own
DTensor shards and copies them straight into _local_tensor. No GPU all-gather. No full-model
materialization on any rank. Multi-rank reads of the same shard files are handled by the OS page
cache via safetensors' mmap.

Per-model quirks (MXFP4 dequant for GPT-OSS, fused expert layout and vision/mtp drops for
Qwen3.5-MoE) are delegated to per-model loaders selected by HF ``config.json``. The generic loader
covers Qwen3 and DeepSeek-V2 verbatim and acts as the fallback for unknown checkpoints.
"""

import contextlib
import gc
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from pithtrain.contexts import logging
from pithtrain.modules.checkpoint import expand_localized_fqn, find_moe, local_shard_range

__all__ = ["load_hf_into_model"]


class CopyOp(NamedTuple):
    """One contiguous write into a local param.

    Non-expert Shard(0): dst is the whole ``param._local_tensor``, ``row_slice`` is this rank's
    FSDP window on the source, ``expert_row`` == ``slice(None)`` (unused), ``transform`` is None.
    Stacked-expert: dst is ``param._local_tensor[i:i+1]``, ``row_slice`` selects one HF expert row
    (or slice(None) when the HF key is already per-expert), and ``transform`` runs MXFP4 dequant
    on the paired ``(blocks, scales)`` slabs when needed.
    """

    local_fqn: str
    dst: torch.Tensor
    hf_keys: Tuple[str, ...]
    shard_files: Tuple[str, ...]
    row_slice: slice
    transform: Optional[Callable[[List[torch.Tensor], torch.dtype], torch.Tensor]]


class ModelHfLoader(Protocol):
    """Per-model plan builder for load_hf_into_model."""

    name: str

    def detect_hf(self, hf_root: Path) -> bool: ...

    def owns_local(self, local_fqn: str) -> bool: ...

    def plan_param(
        self,
        local_fqn: str,
        param: DTensor,
        weight_map: Dict[str, str],
        named_modules: Dict[str, nn.Module],
    ) -> List[CopyOp]: ...

    def expected_unmapped(self, hf_keys: set) -> set: ...


def _read_weight_map(hf_root: Path) -> Tuple[Dict[str, str], Path]:
    """
    Return ``(weight_map, shards_dir)`` from a sharded (index.json) or single-file HF checkpoint,
    with the ``model.`` prefix stripped from every key so downstream lookups share the same
    canonical namespace as ``expand_localized_fqn``. ``lm_head.weight`` (already top-level) is
    unaffected; ``model.language_model.foo`` becomes ``language_model.foo``.
    """
    index = Path(hf_root, "model.safetensors.index.json")
    if index.is_file():
        with open(index) as f:
            raw = json.load(f)["weight_map"]
    else:
        single = Path(hf_root, "model.safetensors")
        if not single.is_file():
            raise FileNotFoundError("No safetensors index or model.safetensors under %s" % hf_root)
        with safe_open(str(single), framework="pt", device="cpu") as f:
            raw = {k: "model.safetensors" for k in f.keys()}
    return {k.removeprefix("model."): v for k, v in raw.items()}, hf_root


def _assert_shardable(local_fqn: str, param: DTensor) -> None:
    if not isinstance(param, DTensor):
        raise RuntimeError(
            "%s is not a DTensor; call load_hf_into_model after apply_fsdp" % local_fqn
        )
    if len(param.placements) > 1:
        raise RuntimeError(
            "%s: HSDP (hsdp_replica > 1) is not supported by load_hf_into_model; "
            "use hf2dcp offline conversion instead" % local_fqn
        )
    if (
        not param.placements
        or not isinstance(param.placements[0], Shard)
        or param.placements[0].dim != 0
    ):
        raise RuntimeError("%s: only Shard(0) placements are supported" % local_fqn)


def _generic_plan_param(
    local_fqn: str,
    param: DTensor,
    weight_map: Dict[str, str],
    named_modules: Dict[str, nn.Module],
    remap: Callable[[str], str] = lambda k: k,
) -> List[CopyOp]:
    """
    Plan a param the way the generic path handles it, with an optional canonical->HF key remap
    for prefix-nested checkpoints (e.g. Qwen3.5's ``language_model.``).
    """
    local = param._local_tensor
    canon_keys = expand_localized_fqn(local_fqn, named_modules)
    moe = find_moe(local_fqn, named_modules)
    dp_rank = param.device_mesh.get_local_rank()
    dp_size = param.device_mesh.size()

    if moe is None:
        (canon,) = canon_keys
        hf_key = remap(canon)
        if hf_key not in weight_map:
            raise KeyError("HF checkpoint missing %s for local param %s" % (hf_key, local_fqn))
        start, length = local_shard_range(param.shape[0], dp_rank, dp_size)
        if length != local.shape[0]:
            raise RuntimeError(
                "%s: computed length %d != local shape %d" % (local_fqn, length, local.shape[0])
            )
        return [
            CopyOp(
                local_fqn=local_fqn,
                dst=local,
                hf_keys=(hf_key,),
                shard_files=(weight_map[hf_key],),
                row_slice=slice(start, start + length),
                transform=None,
            )
        ]

    dp_offset, dp_len = local_shard_range(moe.experts_per_rank, dp_rank, dp_size)
    if dp_len != local.shape[0]:
        raise RuntimeError(
            "%s: expert dp_len %d != local shape %d" % (local_fqn, dp_len, local.shape[0])
        )
    ops: List[CopyOp] = []
    for i, canon in enumerate(canon_keys[dp_offset : dp_offset + dp_len]):
        hf_key = remap(canon)
        if hf_key not in weight_map:
            raise KeyError(
                "HF checkpoint missing %s for local param %s expert %d" % (hf_key, local_fqn, i)
            )
        ops.append(
            CopyOp(
                local_fqn=local_fqn,
                dst=local[i : i + 1],
                hf_keys=(hf_key,),
                shard_files=(weight_map[hf_key],),
                row_slice=slice(None),
                transform=None,
            )
        )
    return ops


class GenericHfLoader:
    """Fallback loader: bare ``model.`` prefix strip, per-expert HF keys."""

    name = "generic"

    def detect_hf(self, hf_root: Path) -> bool:
        return True

    def owns_local(self, local_fqn: str) -> bool:
        return True

    def plan_param(
        self,
        local_fqn: str,
        param: DTensor,
        weight_map: Dict[str, str],
        named_modules: Dict[str, nn.Module],
    ) -> List[CopyOp]:
        return _generic_plan_param(local_fqn, param, weight_map, named_modules)

    def expected_unmapped(self, hf_keys: set) -> set:
        # HF may ship a bare lm_head.weight the live model does not carry as a separate parameter
        # (tied embeddings) — treat as expected, not a warning.
        return {"lm_head.weight"} & hf_keys


def _plan_copies(
    model: nn.Module, weight_map: Dict[str, str], loader: ModelHfLoader
) -> Tuple[List[CopyOp], set, List[str]]:
    named_modules = dict(model.named_modules())
    plan: List[CopyOp] = []
    consumed: set = set()
    skipped: List[str] = []

    for name, param in model.named_parameters():
        _assert_shardable(name, param)
        if not loader.owns_local(name):
            skipped.append(name)
            continue
        ops = loader.plan_param(name, param, weight_map, named_modules)
        if not ops:
            raise RuntimeError(
                "loader %s claims to own %s but returned no CopyOps" % (loader.name, name)
            )
        plan.extend(ops)
        for op in ops:
            consumed.update(op.hf_keys)

    return plan, consumed, skipped


def load_hf_into_model(hf_root: Path, model: nn.Module) -> None:
    """
    Copy the HuggingFace safetensors weights under hf_root into model's live DTensor parameters.

    Only Shard(0) DTensor params are supported. Every local parameter is either filled exactly
    once from HF or, when the selected loader disowns it (e.g. Qwen3.5's Gated DeltaNet
    ``A_log``/``dt_bias``, which HF does not ship), left at its module-constructor init.
    """
    stdout = logging.stdout
    hf_root = Path(hf_root)
    stdout.info("Load HF checkpoint: %s" % hf_root)
    t0 = time.monotonic()

    loader = _select(hf_root)
    stdout.info("Selected HF loader: %s" % loader.name)

    weight_map, shards_dir = _read_weight_map(hf_root)
    plan, consumed, skipped = _plan_copies(model, weight_map, loader)
    for name in skipped:
        stdout.info("Leaving %s at constructor init (not in HF)" % name)

    by_shard: Dict[str, List[CopyOp]] = {}
    for op in plan:
        by_shard.setdefault(op.shard_files[0], []).append(op)

    all_shards = sorted({s for op in plan for s in op.shard_files})
    filled = 0
    with contextlib.ExitStack() as stack:
        handles: Dict[str, "safe_open"] = {
            shard: stack.enter_context(
                safe_open(str(Path(shards_dir, shard)), framework="pt", device="cpu")
            )  # fmt: skip
            for shard in all_shards
        }
        for shard_name in sorted(by_shard):
            entries = by_shard[shard_name]
            stdout.info("Load HF shard: %s (%d ops)" % (shard_name, len(entries)))
            for op in entries:
                srcs = [
                    handles[shard].get_slice(key)[op.row_slice]
                    for key, shard in zip(op.hf_keys, op.shard_files)
                ]
                src = op.transform(srcs, op.dst.dtype) if op.transform is not None else srcs[0]
                if tuple(src.shape) != tuple(op.dst.shape):
                    raise RuntimeError(
                        "%s: shape mismatch, HF %s slice %s vs local %s"
                        % (op.local_fqn, op.hf_keys, tuple(src.shape), tuple(op.dst.shape))
                    )
                op.dst.copy_(src.to(dtype=op.dst.dtype))
                filled += 1

    unmapped = set(weight_map.keys()) - consumed
    expected = loader.expected_unmapped(unmapped)
    for hf_key in sorted(expected):
        if hf_key == "lm_head.weight":
            stdout.info("HF has lm_head.weight; model uses tied embeddings, skipping")
        else:
            stdout.info("Expected unmapped HF key: %s" % hf_key)
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
        "Load HF checkpoint: %d ops, %d skipped, elapsed min=%.1fs, max=%.1fs"
        % (filled, len(skipped), dt_min.item(), dt_max.item())
    )


# ─── Per-model loaders ────────────────────────────────────────────────────────
# Imported after all core definitions to allow model files to import CopyOp /
# _generic_plan_param from this module without hitting a circular import.

from pithtrain.models.gpt_oss import GptOssHfLoader  # noqa: E402
from pithtrain.models.qwen35_moe import Qwen35MoeHfLoader  # noqa: E402

_LOADERS: List[ModelHfLoader] = [GptOssHfLoader(), Qwen35MoeHfLoader(), GenericHfLoader()]


def _select(hf_root: Path) -> ModelHfLoader:
    for loader in _LOADERS:
        if loader.detect_hf(hf_root):
            return loader
    raise RuntimeError("No HF loader claimed %s" % hf_root)
