"""Explicit, strict HF <-> canonical conversion for the complete Omni Thinker.

The text-only converter remains the default. This module includes both encoders,
including DeepStack projection weights, and never silently drops Talker weights.
Schema construction uses HF modules on the meta device, not released allocations.
"""

import copy
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

from . import qwen3_omni_moe as text

SIDECAR = "qwen3_omni_thinker.json"
SCOPE = "thinker_multimodal"
_EXCLUDED = ("talker.", "code2wav.")


def thinker_config(raw):
    """Reject text-only and incomplete composite configs before reading weights."""
    if raw.get("quantization_config"):
        raise text.ConversionError("Quantized Omni checkpoints are unsupported")
    if raw.get("model_type") == "qwen3_omni_moe":
        raw = raw["thinker_config"]
    if raw.get("quantization_config"):
        raise text.ConversionError("Quantized Thinker checkpoints are unsupported")
    if raw.get("model_type") != "qwen3_omni_moe_thinker":
        raise text.ConversionError("Multimodal conversion requires a full Thinker config")
    for kind, fields in (
        (
            "vision_config",
            (
                "depth",
                "hidden_size",
                "intermediate_size",
                "num_heads",
                "out_hidden_size",
                "patch_size",
                "temporal_patch_size",
                "spatial_merge_size",
                "in_channels",
            ),
        ),
        (
            "audio_config",
            (
                "num_mel_bins",
                "encoder_layers",
                "encoder_attention_heads",
                "encoder_ffn_dim",
                "d_model",
                "output_dim",
                "downsample_hidden_size",
                "max_source_positions",
                "n_window",
                "n_window_infer",
                "conv_chunksize",
            ),
        ),
    ):
        cfg = raw.get(kind)
        if not isinstance(cfg, dict):
            raise text.ConversionError(f"Missing {kind}")
        if cfg.get("quantization_config"):
            raise text.ConversionError("Quantized encoders are unsupported")
        for field in fields:
            if type(cfg.get(field)) is not int or cfg[field] <= 0:
                raise text.ConversionError(f"Invalid/missing {kind}.{field}")
    if "deepstack_visual_indexes" not in raw["vision_config"]:
        raise text.ConversionError("Missing vision_config.deepstack_visual_indexes")
    config = Qwen3OmniMoeThinkerConfig(**copy.deepcopy(raw))
    text_cfg = text.text_config(raw)
    # The released 26291f7 config omits this field. HF's config supplies its
    # documented architecture default; explicit checkpoint values still win.
    if (
        type(config.vision_config.num_position_embeddings) is not int
        or config.vision_config.num_position_embeddings <= 0
    ):
        raise text.ConversionError("Invalid vision_config.num_position_embeddings")
    if (
        config.vision_config.out_hidden_size != text_cfg.hidden_size
        or config.audio_config.output_dim != text_cfg.hidden_size
    ):
        raise text.ConversionError("Encoder projection dimensions must match text hidden_size")
    if config.vision_config.hidden_size % config.vision_config.num_heads:
        raise text.ConversionError("Vision hidden_size must divide num_heads")
    if config.audio_config.d_model % config.audio_config.encoder_attention_heads:
        raise text.ConversionError("Audio d_model must divide encoder_attention_heads")
    indexes = config.vision_config.deepstack_visual_indexes
    if (
        not isinstance(indexes, list)
        or any(type(i) is not int or not 0 <= i < config.vision_config.depth for i in indexes)
        or len(set(indexes)) != len(indexes)
        or len(indexes) > text_cfg.num_hidden_layers
    ):
        raise text.ConversionError("Invalid DeepStack layer indexes")
    return config


def parameter_shapes(config, layout="canonical", prefix="thinker."):
    # These optional HF modeling imports are deliberately local: the default
    # text-only converter does not require media processor/model dependencies.
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeVisionEncoder,
    )

    result = text.parameter_shapes(config.text_config, layout, prefix)
    with torch.device("meta"):
        encoders = (
            ("visual", Qwen3OmniMoeVisionEncoder(copy.deepcopy(config.vision_config))),
            ("audio_tower", Qwen3OmniMoeAudioEncoder(copy.deepcopy(config.audio_config))),
        )
    for name, model in encoders:
        for key, value in model.state_dict().items():
            root = "" if layout == "canonical" else prefix
            result[f"{root}{name}.{key}"] = tuple(value.shape)
    return result


def _selected(keys, thinker_only):
    excluded = sorted(key for key in keys if key.startswith(_EXCLUDED))
    if excluded and not thinker_only:
        raise text.ConversionError(
            f"Talker/code2wav require explicit omni_thinker_only=True: {excluded}"
        )
    return excluded


def hf_to_canonical(state, config, *, prefix="thinker.", thinker_only=False):
    excluded = _selected(state, thinker_only)
    state = {key: value for key, value in state.items() if key not in set(excluded)}
    layout = "fused" if any(key.endswith(".experts.gate_up_proj") for key in state) else "split"
    text.validate_tensors(state, parameter_shapes(config, layout, prefix), "HF complete Thinker")
    text_keys = text.parameter_shapes(config.text_config, layout, prefix)
    canonical, _ = text.hf_to_canonical(
        {key: state[key] for key in text_keys}, config.text_config, prefix=prefix
    )
    for key, value in state.items():
        if key not in text_keys:
            canonical[key.removeprefix(prefix)] = value
    text.validate_tensors(canonical, parameter_shapes(config), "Canonical complete Thinker")
    return canonical, excluded


def canonical_to_hf(state, config, *, prefix="thinker.", layout="fused"):
    text.validate_tensors(state, parameter_shapes(config), "Canonical complete Thinker")
    text_keys = text.parameter_shapes(config.text_config)
    result = text.canonical_to_hf(
        {key: state[key] for key in text_keys}, config.text_config, prefix=prefix, layout=layout
    )
    result.update({prefix + key: value for key, value in state.items() if key not in text_keys})
    text.validate_tensors(result, parameter_shapes(config, layout, prefix), "HF complete Thinker")
    return result


def read_hf(load_path, config, *, prefix="thinker.", thinker_only=False):
    """Validate every shard header/index before allocating any weight tensors."""
    load_path = Path(load_path)
    index = load_path / "model.safetensors.index.json"
    weight_map = text.read_json(index).get("weight_map") if index.exists() else None
    if index.exists() and (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(isinstance(v, str) and v for v in weight_map.values())
    ):
        raise text.ConversionError("HF weight_map must be a nonempty tensor-to-shard mapping")
    filenames = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    shapes, owners, dtypes = {}, {}, {}
    for filename in filenames:
        if Path(filename).name != filename:
            raise text.ConversionError(f"Shard must be a filename within source: {filename!r}")
        text._validate_header_json(load_path / filename)
        with safe_open(load_path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in owners:
                    raise text.ConversionError(f"Duplicate tensor across shards: {key}")
                value = handle.get_slice(key)
                shapes[key], owners[key] = tuple(value.get_shape()), filename
                dtypes[key] = text._SAFE_DTYPES.get(value.get_dtype(), value.get_dtype())
    if weight_map is not None and owners != weight_map:
        raise text.ConversionError("HF index/shard mismatch")
    excluded = _selected(shapes, thinker_only)
    selected = {key: shape for key, shape in shapes.items() if key not in set(excluded)}
    layout = "fused" if any(key.endswith(".experts.gate_up_proj") for key in selected) else "split"
    text.validate_shapes(selected, parameter_shapes(config, layout, prefix), "HF Thinker headers")
    text.validate_dtypes({key: dtypes[key] for key in selected}, "HF Thinker headers")
    state = {}
    for filename in filenames:
        with safe_open(load_path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in selected:
                    state[key] = handle.get_tensor(key)
    return state, excluded, layout


def load_canonical(load_path, config):
    metadata = FileSystemReader(load_path).read_metadata()
    prefix = "app.model."
    entries = {
        key.removeprefix(prefix): value
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith(prefix)
    }
    if any(not hasattr(value, "size") for value in entries.values()):
        raise text.ConversionError("Non-tensor model metadata")
    text.validate_shapes(
        {key: tuple(value.size) for key, value in entries.items()},
        parameter_shapes(config),
        "DCP Thinker metadata",
    )
    text.validate_dtypes(
        {key: value.properties.dtype for key, value in entries.items()}, "DCP Thinker metadata"
    )
    state = {
        prefix + key: torch.empty(value.size, dtype=value.properties.dtype, device="cpu")
        for key, value in entries.items()
    }
    dcp.load(state, checkpoint_id=load_path, no_dist=True)
    return {key.removeprefix(prefix): value for key, value in state.items()}


def export_config(load_path, config_path=None):
    path = Path(load_path) / SIDECAR
    saved = None
    if path.exists():
        manifest = text.read_json(path)
        if manifest.get("format_version") != 1 or manifest.get("scope") != SCOPE:
            raise text.ConversionError("Invalid complete Thinker sidecar format/scope")
        saved = thinker_config(manifest["thinker_config"])
    explicit = thinker_config(text.read_json(config_path)) if config_path else None
    if saved and explicit:
        from .strict_omni import same_config

        same_config(saved, explicit)
    return explicit or saved


def _write_sidecar(path, config, prefix, layout, excluded=()):
    manifest = dict(
        format_version=1,
        scope=SCOPE,
        thinker_config=config.to_dict(),
        source_prefix=prefix,
        source_expert_layout=layout,
        excluded_keys=list(excluded),
    )
    (Path(path) / SIDECAR).write_text(json.dumps(manifest, indent=2) + "\n")


def hf2dcp(cfg, stdout):
    from .strict_omni import restore_from_hf

    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)
    text.require_empty_destination(save_path)
    config = thinker_config(text.read_json(load_path / "config.json"))
    state, excluded, layout = read_hf(
        load_path, config, prefix=cfg.omni_prefix, thinker_only=cfg.omni_thinker_only
    )
    canonical, _ = hf_to_canonical(state, config, prefix=cfg.omni_prefix)
    if cfg.omni_restore_resume:
        restore_from_hf(load_path, save_path, config, canonical, multimodal=True)
    else:
        save_path.mkdir(parents=True, exist_ok=True)
        dcp.save({"app": {"model": canonical}}, checkpoint_id=save_path, no_dist=True)
        _write_sidecar(save_path, config, cfg.omni_prefix, layout, excluded)
    stdout.info("Saved complete Thinker: %d weights; excluded keys: %s", len(canonical), excluded)


def dcp2hf(cfg, stdout):
    from ._core import _write_hf
    from .strict_omni import MANIFEST, assemble, read_payloads, same_config, write_companion

    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)
    text.require_empty_destination(save_path)
    if cfg.max_shard_size <= 0:
        raise text.ConversionError("max_shard_size must be positive")
    if cfg.omni_restore_resume:
        raise text.ConversionError("omni_restore_resume only applies to hf2dcp")
    config = export_config(load_path, cfg.omni_thinker_config)
    strict = (load_path / MANIFEST).exists()
    if not strict and any(load_path.glob("rank-*.pt")):
        raise text.ConversionError("Incomplete strict container: missing completion manifest")
    if strict:
        _, payloads, saved, pieces = read_payloads(load_path, multimodal=True)
        if config is not None:
            same_config(config, saved)
        config = saved
        canonical = assemble(config, pieces)
    else:
        if config is None:
            raise text.ConversionError("Complete Thinker export requires config or sidecar")
        canonical = load_canonical(load_path, config)
    state = canonical_to_hf(
        canonical, config, prefix=cfg.omni_prefix, layout=cfg.omni_expert_layout
    )
    _write_hf(state, save_path, cfg.max_shard_size, stdout)
    (save_path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    _write_sidecar(save_path, config, cfg.omni_prefix, cfg.omni_expert_layout)
    if strict:
        write_companion(save_path, payloads, payloads[0]["header"]["step"])
