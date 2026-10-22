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
from .qwen3_omni_moe import (
    SIDECAR,
    ConversionError,
    Qwen3OmniTextConverter,
    canonical_to_hf,
    export_config,
    load_canonical,
    read_json,
    require_empty_destination,
)


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

    omni_text_config: Path | None = None
    """Omni text config for native DCP export without a conversion sidecar."""

    omni_prefix: Literal["thinker.", ""] = "thinker."
    """Exact HF text namespace, for Omni import and export."""

    omni_expert_layout: Literal["split", "fused"] = "fused"
    """Omni export layout: released indexed/split or Transformers 5.17 fused."""

    omni_text_only: bool = False
    """Explicitly exclude known media/Talker weights on Omni import; report every key."""

    omni_multimodal: bool = False
    """Explicitly select the complete Thinker text/vision/audio contract."""

    omni_thinker_config: Path | None = None
    """Complete Thinker config for native DCP export without a sidecar."""

    omni_thinker_only: bool = False
    """Explicitly exclude and report Talker/code2wav on complete Thinker import."""

    omni_components: Literal["thinker", "talker", "code2wav", "all"] | None = None
    """Explicit complete-component selection, separate from legacy text/Thinker modes."""

    omni_component_config: Path | None = None
    """Exact component/full config for model-only native DCP export."""

    omni_component_namespace: Literal["component", "standalone"] = "component"
    """HF key namespace; all requires component roots. Native canonical keys are fixed."""

    omni_restore_resume: bool = False
    """Rebuild original strict shards from an HF export with resume companion."""

    logging: LoggingCfg = field(default_factory=LoggingCfg)
    """Logging configuration."""


def _component_mode(cfg, load_path):
    selected = cfg.omni_components is not None
    if selected:
        if (
            cfg.omni_multimodal
            or cfg.omni_text_only
            or cfg.omni_text_config
            or cfg.omni_thinker_config
            or cfg.omni_thinker_only
            or cfg.omni_prefix != "thinker."
        ):
            raise ConversionError("Component selection cannot be combined with legacy Omni options")
        return True
    if (
        cfg.omni_component_config
        or cfg.omni_component_namespace != "component"
        or (load_path / "qwen3_omni_components.json").exists()
    ):
        raise ConversionError("Component checkpoint/options require explicit omni_components")
    return False


def hf2dcp(cfg: ConvertCheckpointCfg, stdout: Logger) -> None:
    """Convert HuggingFace checkpoint to DCP format."""
    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)
    if _component_mode(cfg, load_path):
        from . import qwen3_omni_full

        qwen3_omni_full.hf2dcp(cfg, stdout)
        return
    if cfg.omni_multimodal:
        from . import qwen3_omni_thinker

        if cfg.omni_text_only or cfg.omni_text_config:
            raise ConversionError("Text-only options cannot be used in multimodal mode")
        qwen3_omni_thinker.hf2dcp(cfg, stdout)
        return
    if cfg.omni_thinker_config or cfg.omni_thinker_only:
        raise ConversionError("Complete Thinker options require omni_multimodal=True")
    if (load_path / "qwen3_omni_thinker.json").exists():
        raise ConversionError("Complete Thinker checkpoint requires omni_multimodal=True")

    if cfg.omni_restore_resume:
        from .qwen3_omni_moe import hf_to_canonical, read_hf_text, text_config
        from .strict_omni import restore_from_hf

        require_empty_destination(save_path)
        raw = read_json(load_path / "config.json")
        if raw.get("quantization_config"):
            raise ConversionError("Quantized Omni checkpoints are unsupported")
        config = text_config(raw)
        state, excluded, _ = read_hf_text(
            load_path, config, prefix=cfg.omni_prefix, text_only=cfg.omni_text_only
        )
        canonical, _ = hf_to_canonical(state, config, prefix=cfg.omni_prefix)
        restore_from_hf(load_path, save_path, config, canonical)
        stdout.info(
            "Rebuilt original strict resume container at %s; excluded non-text keys (%d): %s",
            save_path,
            len(excluded),
            excluded,
        )
        return

    if (load_path / "config.json").exists():
        read_json(load_path / "config.json")
    for converter in CONVERTERS:
        if converter.detect_hf(load_path):
            stdout.info("Dispatching hf2dcp to %s converter" % converter.name)
            if isinstance(converter, Qwen3OmniTextConverter):
                converter.hf2dcp(
                    load_path,
                    save_path,
                    stdout,
                    prefix=cfg.omni_prefix,
                    text_only=cfg.omni_text_only,
                )
            else:
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
    if _component_mode(cfg, load_path):
        from . import qwen3_omni_full

        qwen3_omni_full.dcp2hf(cfg, stdout)
        return
    if cfg.omni_multimodal:
        from . import qwen3_omni_thinker

        if cfg.omni_text_only or cfg.omni_text_config:
            raise ConversionError("Text-only options cannot be used in multimodal mode")
        qwen3_omni_thinker.dcp2hf(cfg, stdout)
        return
    if cfg.omni_thinker_config or cfg.omni_thinker_only:
        raise ConversionError("Complete Thinker options require omni_multimodal=True")
    if (load_path / "qwen3_omni_thinker.json").exists():
        raise ConversionError("Complete Thinker checkpoint requires omni_multimodal=True")
    if cfg.max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive")
    from .strict_omni import MANIFEST, assemble, read_payloads, same_config, write_companion

    if cfg.omni_restore_resume:
        raise ConversionError("omni_restore_resume only applies to hf2dcp")
    strict = (load_path / MANIFEST).exists()
    if not strict and any(load_path.glob("rank-*.pt")):
        raise ConversionError("Incomplete strict container: missing completion manifest")
    if strict:
        _, payloads, config, pieces = read_payloads(load_path)
        explicit = export_config(load_path, cfg.omni_text_config)
        if explicit is not None:
            same_config(config, explicit)
    else:
        config = export_config(load_path, cfg.omni_text_config)
    if config is not None:
        require_empty_destination(save_path)
        canonical = assemble(config, pieces) if strict else load_canonical(load_path, config)
        hf_state = canonical_to_hf(
            canonical, config, prefix=cfg.omni_prefix, layout=cfg.omni_expert_layout
        )
        del canonical  # release unstacked source experts before writing fused copies
        _write_hf(hf_state, save_path, cfg.max_shard_size, stdout)
        (save_path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        manifest = {
            "format_version": 1,
            "scope": "thinker_text_only",
            "text_config": config.to_dict(),
            "source_prefix": cfg.omni_prefix,
            "source_expert_layout": cfg.omni_expert_layout,
        }
        (save_path / SIDECAR).write_text(json.dumps(manifest, indent=2) + "\n")
        if strict:
            write_companion(save_path, payloads, payloads[0]["header"]["step"])
        return
    stdout.info("Converting DCP checkpoint from %s" % load_path)

    model_prefix = "app.model."
    state_dict, metadata = dict(), FileSystemReader(load_path).read_metadata()
    if any(
        ".experts." in key and key.endswith(".gate_up_proj") for key in metadata.state_dict_metadata
    ) and not any(converter.detect_dcp(metadata) for converter in CONVERTERS):
        raise ConversionError(
            "Unidentified fused expert DCP: Omni export requires omni_text_config or its sidecar"
        )
    for key, tensor_meta in metadata.state_dict_metadata.items():
        if key.startswith(model_prefix):
            state_dict[key] = torch.empty(
                tensor_meta.size, dtype=tensor_meta.properties.dtype, device="cpu"
            )
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

    _write_hf(hf_state_dict, save_path, cfg.max_shard_size, stdout)


def _write_hf(hf_state_dict, save_path, max_shard_size, stdout):
    """Shared shard writer; family-specific validation precedes this function."""
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
