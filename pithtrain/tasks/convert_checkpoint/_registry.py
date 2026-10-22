"""Model-specific checkpoint converters.

Each converter handles one model family when the generic path is
insufficient (quantized weights, stacked experts with HF-side
transposes). Converters are tried in order; the first whose detect_*
returns True wins.

``pithtrain.modules.hf_loader`` (streaming HF -> FSDP load) also dispatches
over CONVERTERS. A converter may define any of these optional hooks; a
missing hook, or a checkpoint no converter claims, gets the generic
behavior:

- ``plan_param(local_fqn, param, weight_map, named_modules) -> List[CopyOp]``
  (default ``_streaming.generic_plan_param``)
- ``expected_unmapped(hf_keys) -> set``: HF keys beyond ``lm_head.weight``
  that are fine to leave unloaded (default none)
"""

from logging import Logger
from pathlib import Path
from typing import Dict, List, Protocol

import torch

from .gpt_oss import GptOssConverter
from .qwen35_moe import Qwen35MoeConverter


class ModelConverter(Protocol):
    """Per-model HF<->DCP conversion hooks."""

    name: str

    def detect_hf(self, load_path: Path) -> bool:
        """True if this converter should handle the HF checkpoint at load_path."""

    def detect_dcp(self, metadata) -> bool:
        """True if this converter should postprocess the given DCP metadata."""

    def hf2dcp(self, load_path: Path, save_path: Path, stdout: Logger) -> None:
        """Full HF->DCP conversion (replaces the generic path)."""

    def postprocess_canonical(
        self, canonical: Dict[str, torch.Tensor], stdout: Logger
    ) -> Dict[str, torch.Tensor]:
        """Transform the canonical DCP state_dict (no 'model.' prefix yet)
        into the layout HF expects before sharding."""


CONVERTERS: List[ModelConverter] = [GptOssConverter(), Qwen35MoeConverter()]
