"""GPT-OSS checkpoint converter: MXFP4 dequant + stacked-expert transpose."""

import json
import math
import re
from logging import Logger
from pathlib import Path
from typing import Dict

import torch

from ._streaming import HfCheckpoint, save_dcp


def _dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768,
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

    def hf2dcp(
        self,
        load_path: Path,
        save_path: Path,
        stdout: Logger,
        *,
        max_chunk_size: int,
        max_shard_size: int,
    ) -> None:
        stdout.info("Converting GPT-OSS HF checkpoint from %s" % load_path)
        checkpoint = HfCheckpoint(load_path, stdout)
        quantized = {
            key.removesuffix("_blocks")
            for key in checkpoint.tensors
            if key.endswith("_blocks")
            and key.removesuffix("_blocks") + "_scales" in checkpoint.tensors
        }
        dequantized = dict(checkpoint.tensors)
        for key in sorted(quantized):
            blocks = dequantized.pop(key + "_blocks")
            scales = dequantized.pop(key + "_scales")
            assert blocks.shape[:-1] == scales.shape, (
                f"{blocks.shape=} does not match {scales.shape=}"
            )
            shape = (*blocks.shape[:-2], blocks.shape[-2] * blocks.shape[-1] * 2)
            dequantized[key] = torch.empty(shape, dtype=torch.bfloat16, device="meta")

        tensors, sources = {}, {}
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
                    tensors[expert_key] = tensor[idx]
                    sources[expert_key] = key, idx
            else:
                tensors[canon] = tensor
                sources[canon] = key, None

        def load_tensor(canon, slices):
            key, idx = sources[canon]
            if key in quantized:
                width = checkpoint.tensors[key + "_blocks"].shape[-1] * 2
                start, stop = slices[-1].start, slices[-1].stop
                groups = slice(start // width, (stop + width - 1) // width)
                prefix = slices[:-1]
                tensor = _dequantize_mxfp4(
                    checkpoint.load(key + "_blocks", idx, (*prefix, groups, slice(None))),
                    checkpoint.load(key + "_scales", idx, (*prefix, groups)),
                )
                offset = start % width
                return tensor[..., offset : offset + stop - start].contiguous()
            return checkpoint.load(key, idx, slices)

        save_dcp(
            tensors,
            load_tensor,
            save_path,
            stdout,
            max_chunk_size=max_chunk_size,
            max_shard_size=max_shard_size,
        )

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
