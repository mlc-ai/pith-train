"""GLM-4.7-Flash checkpoint converter.

The decoder keys are per-expert and untransposed, so the generic import and export handle them,
the multi-token-prediction layer's too (``model.layers.<num_hidden_layers>``). That layer's copies
of the embedding and of the head (``embed_tokens.weight``, ``shared_head.head.weight``) are not
loaded, as the layer shares the trunk's: they are expected to stay unmapped on import, and an
export does not write them.
"""

import json
import re
from logging import Logger
from pathlib import Path
from typing import Dict

import torch

# hf_loader strips the ``model.`` prefix from every HF key.
_SHARED_COPY = re.compile(r"^layers\.\d+\.(embed_tokens|shared_head\.head)\.weight$")


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
        # Nothing to postprocess: the canonical keys are the HF keys of the decoder.
        return False

    def postprocess_canonical(
        self, canonical: Dict[str, torch.Tensor], stdout: Logger
    ) -> Dict[str, torch.Tensor]:
        return canonical
