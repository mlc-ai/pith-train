"""Dense Qwen3.5 text-tower conversion; vision and MTP weights are intentionally excluded."""

import json
from logging import Logger
from pathlib import Path

import torch

from ._streaming import generic_plan_param


class Qwen35Converter:
    name = "qwen35"

    def detect_hf(self, load_path: Path) -> bool:
        config_path = Path(load_path, "config.json")
        if not config_path.exists():
            return False
        config = json.loads(config_path.read_text())
        text = config.get("text_config", config)
        return isinstance(text, dict) and text.get("model_type") == "qwen3_5_text"

    def detect_dcp(self, metadata) -> bool:
        keys = metadata.state_dict_metadata
        return any(".linear_attn." in k for k in keys) and any(".mlp.gate_proj." in k for k in keys)

    @staticmethod
    def _renest(canon: str) -> str:
        return canon if canon == "lm_head.weight" else "language_model." + canon

    def postprocess_canonical(self, canonical: dict[str, torch.Tensor], stdout: Logger):
        # Match the public multimodal checkpoint's text-tower namespace. As with the MoE
        # converter, conversion produces weight shards; the caller supplies the HF config.
        return {self._renest(key): value for key, value in canonical.items()}

    def plan_param(self, local_fqn, param, weight_map, named_modules):
        # hf_loader strips the leading model. prefix. Accept both the public multimodal
        # checkpoint and a standalone Qwen3_5ForCausalLM text checkpoint.
        def remap(canon):
            nested = self._renest(canon)
            return nested if nested in weight_map else canon

        return generic_plan_param(local_fqn, param, weight_map, named_modules, remap=remap)

    def expected_unmapped(self, hf_keys: set) -> set:
        return {key for key in hf_keys if key.startswith(("visual.", "mtp."))}
