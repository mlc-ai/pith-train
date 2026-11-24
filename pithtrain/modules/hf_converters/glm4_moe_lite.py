"""GLM-4.7-Flash checkpoint converter.

The decoder keys are DeepSeek-V2's, per-expert and untransposed, so the generic import and export
handle them. The checkpoint also holds a multi-token-prediction layer under
``model.layers.<num_hidden_layers>`` (its ``eh_proj``, ``enorm``, ``hnorm``, ``shared_head``, a
copy of the embedding and of the head, and one more decoder layer), which the model does not
instantiate: those keys are expected to stay unmapped on import.
"""

import json
import re
from logging import Logger
from pathlib import Path
from typing import Dict

import torch

# hf_loader strips the ``model.`` prefix from every HF key.
_LAYER_KEY = re.compile(r"^layers\.(\d+)\.")


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
        # The MTP layers are the ones with an eh_proj; every key of such a layer is expected.
        mtp_layers = set()
        for hf_key in unmapped:
            match = _LAYER_KEY.match(hf_key)
            if match and hf_key.endswith(".eh_proj.weight"):
                mtp_layers.add(match.group(1))
        expected = set()
        for hf_key in unmapped:
            match = _LAYER_KEY.match(hf_key)
            if match and match.group(1) in mtp_layers:
                expected.add(hf_key)
        return expected

    def detect_dcp(self, metadata) -> bool:
        # Nothing to postprocess: the canonical keys are the HF keys of the decoder.
        return False

    def postprocess_canonical(
        self, canonical: Dict[str, torch.Tensor], stdout: Logger
    ) -> Dict[str, torch.Tensor]:
        return canonical
