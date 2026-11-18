"""Qwen3.5-MoE checkpoint converter.

Qwen3.5-35B-A3B ships as a vision-language checkpoint. Only the text tower is
converted: ``model.language_model.*`` (+ top-level ``lm_head.weight``) map to
the canonical PithTrain keys; the vision tower (``model.visual.*``) and the
multi-token-prediction head (``mtp.*``) are dropped.

Experts are already stored fused and in the training-framework ``[E, out, in]``
layout (``gate_up_proj`` is ``[E, 2*inter, hidden]``, ``down_proj`` is
``[E, hidden, inter]``), so the import only slices along the expert axis and
dcp2hf re-stacks and strips the runtime ``GroupedLinear`` ``.weight`` suffix;
there is no layout transpose anywhere.
"""

import json
import re
from logging import Logger
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from pithtrain.modules.checkpoint import find_moe

from ._streaming import CopyOp, fused_expert_ops, generic_plan_param

# HF keys dropped entirely (no PithTrain counterpart).
_DROP_PREFIXES = ("model.visual.", "mtp.")


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
                # Strip the runtime GroupedLinear ".weight" suffix to recover HF's
                # bare fused-expert key.
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
        """Map a canonical key to its HF key (minus the generic ``model.`` prefix)."""
        if canon == "lm_head.weight":
            return canon
        return "language_model." + canon

    def plan_param(
        self,
        local_fqn: str,
        param: DTensor,
        weight_map: Dict[str, str],
        named_modules: Dict[str, nn.Module],
    ) -> List[CopyOp]:
        """Streaming load: text tower nested under ``language_model.``, fused experts."""
        args = (local_fqn, param, weight_map, named_modules)
        if find_moe(local_fqn, named_modules) is not None:
            return fused_expert_ops(*args, lambda k: (self._renest(k),))
        return generic_plan_param(*args, remap=self._renest)

    def expected_unmapped(self, hf_keys: set) -> set:
        """Streaming load: the vision tower and MTP head have no PithTrain counterpart."""
        # hf_loader strips the ``model.`` prefix from every HF key.
        drop = tuple(p.removeprefix("model.") for p in _DROP_PREFIXES)
        return {k for k in hf_keys if k.startswith(drop)}
