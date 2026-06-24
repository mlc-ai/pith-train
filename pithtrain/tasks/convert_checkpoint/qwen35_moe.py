"""Qwen3.5-MoE checkpoint converter.

Qwen3.5-35B-A3B ships as a vision-language checkpoint. Only the text tower is
converted: ``model.language_model.*`` (+ top-level ``lm_head.weight``) map to
the canonical PithTrain keys; the vision tower (``model.visual.*``) and the
multi-token-prediction head (``mtp.*``) are dropped.

Experts are already stored fused and in the training-framework ``[E, out, in]``
layout (``gate_up_proj`` is ``[E, 2*inter, hidden]``, ``down_proj`` is
``[E, hidden, inter]``), so hf2dcp only splits along the expert axis (adding a
``.weight`` suffix for the runtime ``GroupLinear`` submodules) and dcp2hf
re-stacks and strips it — there is no layout transpose anywhere.
"""

import json
import re
from logging import Logger
from pathlib import Path
from typing import Dict

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

# HF keys that nest the text tower; stripped to canonical PithTrain keys.
_TEXT_PREFIX = "model.language_model."
# HF keys dropped entirely (no PithTrain counterpart).
_DROP_PREFIXES = ("model.visual.", "mtp.")
# Fused expert weights split per-expert in hf2dcp / re-stacked in dcp2hf.
_EXPERT_SUFFIXES = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")


class Qwen35MoeConverter:
    name: str = "qwen35_moe"

    def detect_hf(self, load_path: Path) -> bool:
        config_path = Path(load_path, "config.json")
        if not config_path.exists():
            return False
        with open(config_path) as f:
            config = json.load(f)
        if config.get("model_type") == "qwen3_5_moe_text":
            return True
        text = config.get("text_config", {})
        return isinstance(text, dict) and text.get("model_type") == "qwen3_5_moe_text"

    def detect_dcp(self, metadata) -> bool:
        keys = metadata.state_dict_metadata.keys()
        return any(".linear_attn." in k for k in keys) and any("gate_up_proj" in k for k in keys)

    def _canonical_key(self, hf_key: str) -> str | None:
        """Map an HF key to its canonical PithTrain key, or None to drop it."""
        if hf_key.startswith(_DROP_PREFIXES):
            return None
        if hf_key == "lm_head.weight":
            return hf_key
        if hf_key.startswith(_TEXT_PREFIX):
            return hf_key.removeprefix(_TEXT_PREFIX)
        return None

    def hf2dcp(self, load_path: Path, save_path: Path, stdout: Logger) -> None:
        with open(Path(load_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        shard_files = sorted(set(weight_map.values()))
        stdout.info(
            "Converting Qwen3.5-MoE HF checkpoint from %s (%d shards)"
            % (load_path, len(shard_files))
        )

        model_state_dict: Dict[str, torch.Tensor] = dict()
        dropped = 0
        for i, shard_file in enumerate(shard_files, start=1):
            stdout.info("Reading shard %d/%d: %s" % (i, len(shard_files), shard_file))
            with safe_open(str(Path(load_path, shard_file)), framework="pt", device="cpu") as f:
                for key in f.keys():
                    canon = self._canonical_key(key)
                    if canon is None:
                        dropped += 1
                        continue
                    tensor = f.get_tensor(key)
                    if canon.endswith(_EXPERT_SUFFIXES):
                        # Fused [E, out, in] -> per-expert [out, in] (no transpose).
                        # Experts are GroupLinear submodules at runtime, so the
                        # canonical key gains a ".weight" suffix vs HF's bare key.
                        for idx in range(tensor.shape[0]):
                            expert_key = (
                                canon.replace(".experts.", ".experts.%d." % idx) + ".weight"
                            )
                            model_state_dict[expert_key] = tensor[idx].contiguous()
                    else:
                        model_state_dict[canon] = tensor

        stdout.info("Dropped %d non-text keys (vision / mtp)" % dropped)
        save_path.mkdir(parents=True, exist_ok=True)
        dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
        stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(model_state_dict)))

    def postprocess_canonical(
        self, canonical: Dict[str, torch.Tensor], stdout: Logger
    ) -> Dict[str, torch.Tensor]:
        # Re-stack per-expert [out, in] into fused [E, out, in] (no transpose:
        # the canonical layout already matches HF's live expert layout), then
        # re-nest the text tower under ``language_model.`` so the generic
        # dcp2hf "model." prefix yields ``model.language_model.*``. lm_head
        # stays top-level.
        indexed = re.compile(r"(.*\.mlp\.experts)\.(\d+)\.(.*)")
        to_stack: Dict[str, Dict[int, torch.Tensor]] = {}
        plain: Dict[str, torch.Tensor] = {}

        for canon, tensor in canonical.items():
            m = indexed.match(canon)
            if m:
                # Strip the runtime GroupLinear ".weight" suffix to recover HF's
                # bare fused-expert key (inverse of the hf2dcp append).
                prefix, idx_str, suffix = m.group(1), m.group(2), m.group(3).removesuffix(".weight")
                stacked_canon = "%s.%s" % (prefix, suffix)
                to_stack.setdefault(stacked_canon, {})[int(idx_str)] = tensor
            else:
                plain[canon] = tensor

        result: Dict[str, torch.Tensor] = {}
        for canon, tensor in plain.items():
            result[self._renest(canon)] = tensor
        for stacked_canon, by_idx in to_stack.items():
            stacked = torch.stack([t for _, t in sorted(by_idx.items())])
            result[self._renest(stacked_canon)] = stacked

        stdout.info(
            "Stacked %d expert tensors into %d grouped keys"
            % (sum(len(v) for v in to_stack.values()), len(to_stack))
        )
        return result

    @staticmethod
    def _renest(canon: str) -> str:
        """Inverse of ``_canonical_key`` (minus the generic ``model.`` prefix)."""
        if canon == "lm_head.weight":
            return canon
        return "language_model." + canon
