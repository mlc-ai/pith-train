"""Generic HF<->DCP conversion, dispatcher, and launch entry-point.

Model-specific converters (quantized weights, stacked experts, layout
transposes) live in sibling modules and are registered in
``_registry.CONVERTERS``. The generic path here handles un-quantized,
un-transposed HF checkpoints (Qwen3, DeepSeek-V2).
"""

import json
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader

from pithtrain.config import SlottedDefault
from pithtrain.contexts import logging
from pithtrain.modules.logging import LoggingCfg, setup_logging

from ._registry import CONVERTERS


@dataclass(init=False, slots=True)
class ConvertCheckpointCfg(SlottedDefault):
    """Configuration for checkpoint conversion."""

    operation: Literal["hf2dcp", "dcp2hf"]
    """Conversion operation."""

    load_path: Path
    """Source checkpoint directory."""

    save_path: Path
    """Destination checkpoint directory."""

    max_shard_size: int = 8 * 1024**3
    """Maximum dcp2hf shard size in bytes (default 8GB)."""

    logging: LoggingCfg = field(default_factory=LoggingCfg)
    """Logging configuration."""


def hf2dcp(cfg: ConvertCheckpointCfg, stdout: Logger) -> None:
    """Convert HuggingFace checkpoint to DCP format."""
    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)

    for converter in CONVERTERS:
        if converter.detect_hf(load_path):
            stdout.info("Dispatching hf2dcp to %s converter" % converter.name)
            converter.hf2dcp(load_path, save_path, stdout)
            return

    with open(Path(load_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = set(weight_map.values())
    stdout.info("Converting HF checkpoint from %s (%d shards)" % (load_path, len(shard_files)))

    model_state_dict: Dict[str, torch.Tensor] = dict()
    for i, shard_file in enumerate(sorted(shard_files), start=1):
        stdout.info("Reading shard %d/%d: %s" % (i, len(shard_files), shard_file))
        with safe_open(str(Path(load_path, shard_file)), framework="pt", device="cpu") as f:
            for key in f.keys():
                model_state_dict[key.removeprefix("model.")] = f.get_tensor(key)

    save_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
    stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(model_state_dict)))


def dcp2hf(cfg: ConvertCheckpointCfg, stdout: Logger) -> None:
    """Convert DCP checkpoint to HuggingFace format."""
    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)
    max_shard_size = cfg.max_shard_size
    stdout.info("Converting DCP checkpoint from %s" % load_path)

    model_prefix = "app.model."
    state_dict, metadata = dict(), FileSystemReader(load_path).read_metadata()
    for key, tensor_meta in metadata.state_dict_metadata.items():
        if key.startswith(model_prefix):
            state_dict[key] = torch.empty(tensor_meta.size, dtype=tensor_meta.properties.dtype)
    dcp.load(state_dict, checkpoint_id=load_path, no_dist=True)
    stdout.info("Loaded %d model weights from DCP" % len(state_dict))

    canonical = {k.removeprefix(model_prefix): v for k, v in state_dict.items()}

    for converter in CONVERTERS:
        if converter.detect_dcp(metadata):
            stdout.info("Dispatching dcp2hf postprocess to %s converter" % converter.name)
            canonical = converter.postprocess_canonical(canonical, stdout)
            break

    hf_state_dict = dict()
    for canon, tensor in canonical.items():
        hf_state_dict[canon if canon.startswith("lm_head.") else "model." + canon] = tensor

    shards: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    current_shard: Dict[str, torch.Tensor] = dict()
    current_size, shard_idx = 0, 0

    for key, tensor in hf_state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(("model-%05d.safetensors" % shard_idx, current_shard))
            current_shard, current_size, shard_idx = dict(), 0, shard_idx + 1
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(("model-%05d.safetensors" % shard_idx, current_shard))

    weight_map, total_size = dict(), 0
    save_path.mkdir(parents=True, exist_ok=True)
    for i, (_, shard_tensors) in enumerate(shards):
        shard_name = "model-%05d-of-%05d.safetensors" % (i, len(shards))
        stdout.info("Writing shard %d/%d: %s" % (i + 1, len(shards), shard_name))
        save_file(shard_tensors, str(Path(save_path, shard_name)))
        for key in shard_tensors:
            weight_map[key] = shard_name
        total_size += sum(t.numel() * t.element_size() for t in shard_tensors.values())

    with open(Path(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
    stdout.info(
        "Saved HF checkpoint to %s (%d weights, %d shards)"
        % (save_path, len(weight_map), len(shards))
    )


def launch(cfg: ConvertCheckpointCfg) -> None:
    """Launch checkpoint conversion."""
    setup_logging(cfg)
    logging.stdout.info("launch(cfg=%s)" % cfg)
    match cfg.operation:
        case "hf2dcp":
            hf2dcp(cfg, logging.stdout)
        case "dcp2hf":
            dcp2hf(cfg, logging.stdout)
