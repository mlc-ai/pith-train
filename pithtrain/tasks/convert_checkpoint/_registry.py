"""Model-specific checkpoint converters.

Each converter handles one model family when the generic path is
insufficient (quantized weights, stacked experts with HF-side
transposes). Converters are tried in order; the first whose detect_*
returns True wins.
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
