"""GPT-OSS checkpoint converter: MXFP4 dequant + stacked-expert transpose."""

import json
import re
from logging import Logger
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from safetensors import safe_open
from torch.distributed.tensor import DTensor

from pithtrain.modules.checkpoint import (
    CopyOp,
    _generic_plan_param,
    expand_localized_fqn,
    find_moe,
    local_shard_range,
)
from pithtrain.operators.mxfp4 import dequantize_mxfp4

_EXPERT_BIAS_LEAVES = {"gate_up_proj_bias", "down_proj_bias"}
_EXPERT_MXFP4_LEAVES = {"gate_up_proj", "down_proj"}


def _parse_expert_key(canon_key: str) -> Tuple[str, int]:
    """
    Split ``layers.L.mlp.experts.42.gate_up_proj`` into the fused HF base key
    (``layers.L.mlp.experts.gate_up_proj``) and the global expert index (42). HF stores
    every expert projection as a single fused ``[E, ...]`` key with no numeric index.
    """
    prefix, _, tail = canon_key.rpartition(".experts.")
    idx_str, _, suffix = tail.partition(".")
    if not idx_str.isdigit():
        raise RuntimeError("expected indexed expert key, got %s" % canon_key)
    return "%s.experts.%s" % (prefix, suffix), int(idx_str)


def _mxfp4_transform(srcs: List[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    # srcs are [1, out, G, 16] blocks + [1, out, G] scales; dequant yields [1, out, G*B*2].
    return dequantize_mxfp4(srcs[0], srcs[1], dtype=dtype)


def _plan_expert(
    local_fqn: str,
    param: DTensor,
    weight_map: Dict[str, str],
    moe: nn.Module,
    canon_keys: List[str],
    *,
    mxfp4: bool,
) -> List[CopyOp]:
    local = param._local_tensor
    dp_rank = param.device_mesh.get_local_rank()
    dp_size = param.device_mesh.size()
    dp_offset, dp_len = local_shard_range(moe.experts_per_rank, dp_rank, dp_size)
    if dp_len != local.shape[0]:
        raise RuntimeError(
            "%s: expert dp_len %d != local shape %d" % (local_fqn, dp_len, local.shape[0])
        )
    base_key, ep_start = _parse_expert_key(canon_keys[0])
    if mxfp4:
        blocks_key = base_key + "_blocks"
        scales_key = base_key + "_scales"
        for k in (blocks_key, scales_key):
            if k not in weight_map:
                raise KeyError("HF checkpoint missing %s for local param %s" % (k, local_fqn))
        hf_keys: Tuple[str, ...] = (blocks_key, scales_key)
        shard_files: Tuple[str, ...] = (weight_map[blocks_key], weight_map[scales_key])
        transform = _mxfp4_transform
    else:
        if base_key not in weight_map:
            raise KeyError("HF checkpoint missing %s for local param %s" % (base_key, local_fqn))
        hf_keys = (base_key,)
        shard_files = (weight_map[base_key],)
        transform = None
    ops = []
    for i in range(dp_len):
        global_idx = ep_start + dp_offset + i
        ops.append(
            CopyOp(
                local_fqn=local_fqn,
                dst=local[i : i + 1],
                hf_keys=hf_keys,
                shard_files=shard_files,
                row_slice=slice(global_idx, global_idx + 1),
                transform=transform,
            )
        )
    return ops


class GptOssConverter:
    name: str = "gpt_oss"

    def detect_hf(self, load_path: Path) -> bool:
        config_path = Path(load_path, "config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            return config.get("model_type") == "gpt_oss"
        return False

    def detect_dcp(self, metadata) -> bool:
        # Gate on the per-expert bias, which is unique to GPT-OSS among our
        # fused-expert models (Qwen3.5-MoE also has gate_up_proj but no bias).
        return any("gate_up_proj_bias" in k for k in metadata.state_dict_metadata.keys())

    def hf2dcp(self, load_path: Path, save_path: Path, stdout: Logger) -> None:
        with open(Path(load_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        shard_files = set(weight_map.values())
        stdout.info(
            "Converting GPT-OSS HF checkpoint from %s (%d shards)" % (load_path, len(shard_files))
        )

        raw: Dict[str, torch.Tensor] = dict()
        for i, shard_file in enumerate(sorted(shard_files), start=1):
            stdout.info("Reading shard %d/%d: %s" % (i, len(shard_files), shard_file))
            with safe_open(str(Path(load_path, shard_file)), framework="pt", device="cpu") as f:
                for key in f.keys():
                    raw[key] = f.get_tensor(key)

        # MXFP4 on-disk is [E, out, in] with quant axis along in (32 FP4/block).
        # Dequant drops the last two dims:
        #   gate_up_proj_blocks [E, 2*inter, G, 16] -> [E, 2*inter, hidden]
        #   down_proj_blocks    [E, hidden,  G, 16] -> [E, hidden, inter]
        dequantized: Dict[str, torch.Tensor] = dict()
        seen_blocks = set()
        for key in sorted(raw.keys()):
            if key.endswith("_blocks"):
                base = key.removesuffix("_blocks")
                scales_key = base + "_scales"
                if scales_key in raw:
                    stdout.info("Dequantizing MXFP4: %s" % base)
                    flat = dequantize_mxfp4(raw[key], raw[scales_key])
                    dequantized[base] = flat.contiguous()
                    seen_blocks.add(key)
                    seen_blocks.add(scales_key)

        for key, tensor in raw.items():
            if key not in seen_blocks and key not in dequantized:
                dequantized[key] = tensor

        model_state_dict: Dict[str, torch.Tensor] = dict()
        for key, tensor in dequantized.items():
            canon = key.removeprefix("model.")

            if canon.endswith(
                (
                    ".mlp.experts.gate_up_proj",
                    ".mlp.experts.gate_up_proj_bias",
                    ".mlp.experts.down_proj",
                    ".mlp.experts.down_proj_bias",
                )
            ):
                for idx in range(tensor.shape[0]):
                    expert_key = canon.replace(".experts.", ".experts.%d." % idx)
                    model_state_dict[expert_key] = tensor[idx].contiguous()
            else:
                model_state_dict[canon] = tensor

        save_path.mkdir(parents=True, exist_ok=True)
        dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
        stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(model_state_dict)))

    def postprocess_canonical(
        self, canonical: Dict[str, torch.Tensor], stdout: Logger
    ) -> Dict[str, torch.Tensor]:
        # DCP stores per-expert [out, in]; HF's live stacked Parameter wants
        # [E, in, out]. The transpose lives here so the model and hf2dcp
        # never see it. 1-D biases pass through unchanged.
        _WEIGHT_KEYS = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")

        indexed = re.compile(r"(.*\.mlp\.experts)\.(\d+)\.(.*)")
        to_stack: Dict[str, Dict[int, torch.Tensor]] = {}
        plain: Dict[str, torch.Tensor] = {}

        for canon, tensor in canonical.items():
            m = indexed.match(canon)
            if m:
                prefix, idx_str, suffix = m.group(1), m.group(2), m.group(3)
                stacked_canon = "%s.%s" % (prefix, suffix)
                to_stack.setdefault(stacked_canon, {})[int(idx_str)] = tensor
            else:
                plain[canon] = tensor

        result = dict(plain)
        for stacked_canon, by_idx in to_stack.items():
            items = sorted(by_idx.items())
            stacked = torch.stack([t for _, t in items])
            if stacked_canon.endswith(_WEIGHT_KEYS):
                stacked = stacked.transpose(-2, -1).contiguous()
            result[stacked_canon] = stacked
        stdout.info(
            "Stacked %d expert tensors into %d grouped keys"
            % (sum(len(v) for v in to_stack.values()), len(to_stack))
        )
        return result

    def plan_param(
        self,
        local_fqn: str,
        param: DTensor,
        weight_map: Dict[str, str],
        named_modules: Dict[str, nn.Module],
    ) -> List[CopyOp]:
        """Streaming load: fused expert biases + MXFP4-quantized expert weights."""
        moe = find_moe(local_fqn, named_modules)
        if moe is None:
            return _generic_plan_param(local_fqn, param, weight_map, named_modules)
        canon_keys = expand_localized_fqn(local_fqn, named_modules)
        leaf = canon_keys[0].rsplit(".", 1)[-1]
        if leaf in _EXPERT_BIAS_LEAVES or leaf in _EXPERT_MXFP4_LEAVES:
            return _plan_expert(
                local_fqn,
                param,
                weight_map,
                moe,
                canon_keys,
                mxfp4=leaf in _EXPERT_MXFP4_LEAVES,
            )
        return _generic_plan_param(local_fqn, param, weight_map, named_modules)
