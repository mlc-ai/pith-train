"""GPT-OSS checkpoint converter: MXFP4 dequant + stacked-expert transpose."""

import json
import math
import re
from logging import Logger
from pathlib import Path
from typing import Dict

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open


def _dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Dequantize MXFP4 blocks (low nibble first, scales biased by 127)."""
    # Adapted from Megatron-Bridge gpt_oss_bridge._dequantize_mxfp4.
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
    FP4_VALUES = [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)
        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]
        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


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
                    flat = _dequantize_mxfp4(raw[key], raw[scales_key])
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
