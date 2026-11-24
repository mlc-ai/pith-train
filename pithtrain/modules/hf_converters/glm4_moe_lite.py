"""GLM-4.7-Flash checkpoint converter.

The decoder keys are per-expert and untransposed, so the generic import and export handle them,
the multi-token-prediction layer's too (``model.layers.<num_hidden_layers>``). That layer's copies
of the embedding and of the head (``embed_tokens.weight``, ``shared_head.head.weight``) are not
loaded, as the layer shares the trunk's: they are expected to stay unmapped on import, and an
export writes them from the trunk's, as the released checkpoint has them.
"""

import json
import re
from logging import Logger
from pathlib import Path
from typing import Dict

import torch

# hf_loader strips the ``model.`` prefix from every HF key; the canonical keys have none either.
_SHARED_COPY = re.compile(r"^layers\.\d+\.(embed_tokens|shared_head\.head)\.weight$")
_MTP_LAYER = re.compile(r"layers\.(\d+)\.eh_proj\.weight")


class Glm4MoeLiteConverter:
    name: str = "glm4_moe_lite"

    def detect_hf(self, load_path: Path) -> bool:
        config_path = Path(load_path, "config.json")
        if not config_path.exists():
            return False
        with open(config_path) as f:
            config = json.load(f)
        return config.get("model_type") == "glm4_moe_lite"

    def expected_unmapped(self, unmapped: set) -> set:
        return {hf_key for hf_key in unmapped if _SHARED_COPY.match(hf_key)}

    def detect_dcp(self, metadata) -> bool:
        keys = metadata.state_dict_metadata
        return any(".eh_proj.weight" in k for k in keys) and any(".e_score_correction_bias" in k for k in keys)  # fmt: skip

    def postprocess_canonical(
        self, canonical: Dict[str, torch.Tensor], stdout: Logger
    ) -> Dict[str, torch.Tensor]:
        # The MTP layer's copies of the embedding and of the head (safetensors refuses shared
        # storage, so clones).
        for key in list(canonical):
            match = _MTP_LAYER.fullmatch(key)
            if match:
                layer = match.group(1)
                canonical[f"layers.{layer}.embed_tokens.weight"] = canonical["embed_tokens.weight"].clone()  # fmt: skip
                canonical[f"layers.{layer}.shared_head.head.weight"] = canonical["lm_head.weight"].clone()  # fmt: skip
        return canonical
