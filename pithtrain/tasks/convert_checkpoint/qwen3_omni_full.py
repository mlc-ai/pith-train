"""Strict, explicit component conversion for the complete Qwen3-Omni architecture.

``omni_components=all`` includes Thinker, Talker (including its residual-code
predictor), and code2wav. Single-component selection reports every excluded key.
This is a model-initialization checkpoint format, not an optimizer-resume format.
Legacy text/Thinker commands retain their separate contracts and sidecars.

HF namespaces are explicit: ``component`` preserves released component roots;
``standalone`` is available for a single component. The HF 5.17 Code2Wav class
requires its concrete Code2WavConfig when loading from disk; for a namespaced
export it additionally requires ``key_mapping={r"^code2wav\\.": ""}`` in
``from_pretrained``. Standalone export needs no key mapping. Full Omni export
loads through the ordinary composite class with all keys present.
"""

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader

from . import qwen3_omni_moe as text
from . import qwen3_omni_thinker as thinker

SIDECAR = "qwen3_omni_components.json"
COMPONENTS = ("thinker", "talker", "code2wav")
_INDEXED = re.compile(r"(.*\.experts)\.(\d+)\.(gate_up_proj|down_proj)$")


def _positive(raw, fields, label):
    for field in fields:
        if type(raw.get(field)) is not int or raw[field] <= 0:
            raise text.ConversionError(f"Invalid/missing {label}.{field}")


def _unquantized(raw):
    if isinstance(raw, dict):
        if raw.get("quantization_config"):
            raise text.ConversionError("Quantized Omni components are unsupported")
        for value in raw.values():
            _unquantized(value)


def _speech_config(raw, name):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeCode2WavConfig,
        Qwen3OmniMoeTalkerConfig,
    )

    common = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    )
    if name == "talker":
        # Like Code2Wav, HF 5.17 TalkerConfig has no class model_type;
        # its own to_dict/save_pretrained emits an empty identity. Explicit
        # selection plus complete sub-configs and schema supply the identity.
        if raw.get("model_type", "") not in ("", "qwen3_omni_moe_talker"):
            raise text.ConversionError("Talker conversion requires the complete Talker config")
        _positive(raw, ("num_code_groups", "thinker_hidden_size", "spatial_merge_size"), name)
        for child in ("text_config", "code_predictor_config"):
            if not isinstance(raw.get(child), dict):
                raise text.ConversionError(f"Missing talker.{child}")
            _positive(raw[child], (*common, "vocab_size", "head_dim"), f"talker.{child}")
        _positive(
            raw["text_config"],
            (
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            ),
            "talker.text_config",
        )
        # HF serializes num_experts through its num_local_experts attribute_map.
        expert_config = raw["text_config"]
        for field in ("num_experts", "num_local_experts"):
            if field in expert_config:
                _positive(expert_config, (field,), "talker.text_config")
        experts = expert_config.get("num_experts", expert_config.get("num_local_experts"))
        if type(experts) is not int or experts <= 0:
            raise text.ConversionError("Invalid/missing talker expert count")
        if (
            "num_experts" in expert_config
            and "num_local_experts" in expert_config
            and expert_config["num_experts"] != expert_config["num_local_experts"]
        ):
            raise text.ConversionError("Conflicting Talker expert count aliases")
        _positive(raw["code_predictor_config"], ("num_code_groups",), "code_predictor")
        config = Qwen3OmniMoeTalkerConfig(**copy.deepcopy(raw))
        if config.text_config.num_experts != experts:
            raise text.ConversionError("Resolved Talker expert count disagrees with input")
        if config.num_code_groups < 2 or (
            config.num_code_groups != config.code_predictor_config.num_code_groups
            or config.text_config.hidden_size != config.code_predictor_config.hidden_size
        ):
            raise text.ConversionError("Talker and code predictor dimensions/groups disagree")
        if not 1 <= config.text_config.num_experts_per_tok <= config.text_config.num_experts:
            raise text.ConversionError("Talker top-k must be within the expert count")
        for cfg in (config, config.text_config, config.code_predictor_config):
            if getattr(cfg, "tie_word_embeddings", False):
                raise text.ConversionError("Tied Talker parameters are unsupported")
        configs = (config.text_config, config.code_predictor_config)
    else:
        # HF 5.17's Code2WavConfig and the released nested config have model_type="".
        # The explicit selection supplies the identity; no generic model inference.
        if raw.get("model_type", "") not in ("", "qwen3_omni_moe_code2wav"):
            raise text.ConversionError("Expected an explicit Code2Wav config")
        _positive(raw, (*common, "codebook_size", "num_quantizers", "decoder_dim"), name)
        for field in ("upsample_rates", "upsampling_ratios"):
            values = raw.get(field)
            if (
                not isinstance(values, (list, tuple))
                or not values
                or any(type(value) is not int or value <= 0 for value in values)
            ):
                raise text.ConversionError(f"Invalid/missing code2wav.{field}")
        if raw["decoder_dim"] % 2 ** len(raw["upsample_rates"]):
            raise text.ConversionError("Code2Wav decoder_dim must divide all decoder stages")
        config = Qwen3OmniMoeCode2WavConfig(**copy.deepcopy(raw))
        configs = (config,)
    for cfg in configs:
        if cfg.num_attention_heads % cfg.num_key_value_heads:
            raise text.ConversionError(f"Invalid {name} attention head grouping")
        if not getattr(cfg, "head_dim", None) and cfg.hidden_size % cfg.num_attention_heads:
            raise text.ConversionError(f"Invalid {name} attention head dimension")
    return config


@dataclass
class ComponentConfig:
    """Exact source JSON and resolved local-HF schemas, never tensor payloads."""

    raw: dict
    selection: str
    configs: dict
    fused: dict

    @property
    def names(self):
        return COMPONENTS if self.selection == "all" else (self.selection,)

    def export_raw(self):
        if self.selection == "all" or self.raw.get("model_type") != "qwen3_omni_moe":
            return copy.deepcopy(self.raw)
        return copy.deepcopy(self.raw[self.selection + "_config"])


def configuration(raw, selection):
    """Require explicit architecture dimensions before any tensor allocation."""
    if selection not in (*COMPONENTS, "all"):
        raise text.ConversionError(f"Unknown Omni component selection: {selection!r}")
    if not isinstance(raw, dict):
        raise text.ConversionError("Omni configuration must be a JSON object")
    _unquantized(raw)
    composite = raw.get("model_type") == "qwen3_omni_moe"
    if selection == "all" and (not composite or raw.get("enable_audio_output") is not True):
        raise text.ConversionError(
            "All components require full Omni config and enable_audio_output"
        )
    names = COMPONENTS if selection == "all" else (selection,)
    configs, fused = {}, {}
    for name in names:
        part = raw.get(name + "_config") if composite else raw
        if not isinstance(part, dict):
            raise text.ConversionError(f"Missing {name}_config")
        cfg = thinker.thinker_config(part) if name == "thinker" else _speech_config(part, name)
        configs[name] = cfg
        if name == "thinker":
            fused[name] = thinker.parameter_shapes(cfg, "fused", prefix="")
        else:
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                Qwen3OmniMoeCode2Wav,
                Qwen3OmniMoeTalkerForConditionalGeneration,
            )

            model_cls = (
                Qwen3OmniMoeTalkerForConditionalGeneration
                if name == "talker"
                else Qwen3OmniMoeCode2Wav
            )
            model_cfg = copy.deepcopy(cfg)
            model_cfg._attn_implementation = "eager"
            if name == "talker":
                model_cfg.text_config._experts_implementation = "eager"
            with torch.device("meta"):
                model = model_cls(model_cfg)
            fused[name] = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    if selection == "all":
        talker, codec = configs["talker"], configs["code2wav"]
        if talker.thinker_hidden_size != configs["thinker"].text_config.hidden_size:
            raise text.ConversionError("Thinker hidden_size does not match Talker projections")
        if talker.num_code_groups != codec.num_quantizers:
            raise text.ConversionError("Talker groups do not match Code2Wav quantizers")
        if talker.code_predictor_config.vocab_size != codec.codebook_size:
            raise text.ConversionError("Residual codec vocabulary does not match Code2Wav codebook")
    return ComponentConfig(copy.deepcopy(raw), selection, configs, fused)


def _prefix(config, name, namespace):
    if namespace not in ("component", "standalone"):
        raise text.ConversionError("Component namespace must be 'component' or 'standalone'")
    if namespace == "standalone" and config.selection == "all":
        raise text.ConversionError("All components require their component namespaces")
    return name + "." if namespace == "component" else ""


def _component_shapes(config, name, layout):
    if layout not in ("canonical", "split", "fused"):
        raise text.ConversionError(f"Unknown layout: {layout!r}")
    if name == "thinker" and layout == "canonical":
        return thinker.parameter_shapes(config.configs[name])
    result = {}
    for key, shape in config.fused[name].items():
        if layout != "fused" and key.endswith(".experts.gate_up_proj"):
            root = key.removesuffix("gate_up_proj")
            for i in range(shape[0]):
                if layout == "split":
                    for proj in ("gate_proj", "up_proj"):
                        result[f"{root}{i}.{proj}.weight"] = (shape[1] // 2, shape[2])
                else:
                    result[f"{root}{i}.gate_up_proj"] = shape[1:]
        elif layout != "fused" and key.endswith(".experts.down_proj"):
            root = key.removesuffix("down_proj")
            for i in range(shape[0]):
                suffix = ".weight" if layout == "split" else ""
                result[f"{root}{i}.down_proj{suffix}"] = shape[1:]
        else:
            result[key] = shape
    return result


def parameter_shapes(config, layout="canonical", *, namespace="component"):
    """Full schema, also usable with only public released safetensors headers."""
    result = {}
    for name in config.names:
        chosen = layout[name] if isinstance(layout, dict) else layout
        prefix = (
            (name + "." if config.selection == "all" else "")
            if chosen == "canonical"
            else _prefix(config, name, namespace)
        )
        result.update({prefix + k: v for k, v in _component_shapes(config, name, chosen).items()})
    return result


def _selection(keys, config, namespace):
    prefixes = tuple(_prefix(config, name, namespace) for name in config.names)
    selected = [key for key in keys if key.startswith(prefixes)]
    excluded = sorted(set(keys) - set(selected))
    if excluded and (
        namespace == "standalone"
        or any(not key.startswith(tuple(name + "." for name in COMPONENTS)) for key in excluded)
    ):
        raise text.ConversionError(f"Unrecognized component keys: {excluded}")
    return selected, excluded


def _layouts(keys, config, namespace):
    return {
        name: "fused"
        if any(
            key.startswith(_prefix(config, name, namespace))
            and key.endswith(".experts.gate_up_proj")
            for key in keys
        )
        else "split"
        for name in config.names
    }


def hf_to_canonical(state, config, *, namespace="component"):
    selected, excluded = _selection(state, config, namespace)
    state = {key: state[key] for key in selected}
    layouts = _layouts(state, config, namespace)
    text.validate_tensors(
        state, parameter_shapes(config, layouts, namespace=namespace), "HF components"
    )
    result = {}
    for name in config.names:
        prefix = _prefix(config, name, namespace)
        part = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
        if name == "thinker":
            canonical, _ = thinker.hf_to_canonical(part, config.configs[name], prefix="")
        else:
            canonical = {}
            for key in _component_shapes(config, name, "canonical"):
                match = _INDEXED.fullmatch(key)
                if not match:
                    canonical[key] = part[key]
                else:
                    root, idx, proj = match.groups()
                    if layouts[name] == "fused":
                        canonical[key] = part[f"{root}.{proj}"][int(idx)].contiguous()
                    elif proj == "down_proj":
                        canonical[key] = part[key + ".weight"]
                    else:
                        canonical[key] = torch.cat(
                            (
                                part[f"{root}.{idx}.gate_proj.weight"],
                                part[f"{root}.{idx}.up_proj.weight"],
                            ),
                            dim=0,
                        )
        out_prefix = name + "." if config.selection == "all" else ""
        result.update({out_prefix + k: v for k, v in canonical.items()})
    text.validate_tensors(result, parameter_shapes(config), "Canonical components")
    return result, excluded, layouts


def canonical_to_hf(state, config, *, namespace="component", layout="fused"):
    if layout not in ("fused", "split"):
        raise text.ConversionError("HF export requires split or fused layout")
    text.validate_tensors(state, parameter_shapes(config), "Canonical components")
    result = {}
    for name in config.names:
        source_prefix = name + "." if config.selection == "all" else ""
        part = {
            k.removeprefix(source_prefix): v
            for k, v in state.items()
            if k.startswith(source_prefix)
        }
        prefix = _prefix(config, name, namespace)
        if name == "thinker":
            output = thinker.canonical_to_hf(part, config.configs[name], prefix="", layout=layout)
        else:
            output = {}
            for key in _component_shapes(config, name, layout):
                if layout == "fused" and key.endswith(
                    (".experts.gate_up_proj", ".experts.down_proj")
                ):
                    root, proj = key.rsplit(".", 1)
                    output[key] = torch.stack(
                        [part[f"{root}.{i}.{proj}"] for i in range(config.fused[name][key][0])]
                    )
                elif layout == "split" and ".experts." in key:
                    root, proj, _ = key.rsplit(".", 2)
                    if proj == "down_proj":
                        output[key] = part[root + ".down_proj"]
                    else:
                        value = part[root + ".gate_up_proj"]
                        offset = 0 if proj == "gate_proj" else value.shape[0] // 2
                        output[key] = value[offset : offset + value.shape[0] // 2].contiguous()
                else:
                    output[key] = part[key]
        result.update({prefix + k: v for k, v in output.items()})
    text.validate_tensors(
        result, parameter_shapes(config, layout, namespace=namespace), "HF output"
    )
    return result


def native_key_map(config):
    """Exact full native -> HF fused mapping (no prefix guessing).

    Native Thinker owns text layers/embed_tokens/norm directly; HF places these
    under ``model``. Talker and Code2Wav retain their HF module attributes. The
    full container must expose exactly the thinker/talker/code2wav roots.
    """
    result = {}
    for name in config.names:
        native_prefix = name + "." if config.selection == "all" else ""
        text_keys = (
            text.parameter_shapes(config.configs[name].text_config, "fused", prefix="")
            if name == "thinker"
            else {}
        )
        for key in config.fused[name]:
            local = key.removeprefix("model.") if key in text_keys else key
            result[native_prefix + local] = name + "." + key
    if len(result) != sum(len(config.fused[name]) for name in config.names):
        raise text.ConversionError("Native/HF module mapping has a key collision")
    return result


def native_to_canonical(state, config):
    """Convert exact, unsharded native state; distributed localization is separate."""
    mapping = native_key_map(config)
    expected = parameter_shapes(config, "fused")
    text.validate_tensors(
        state, {key: expected[value] for key, value in mapping.items()}, "Native components"
    )
    result, excluded, _ = hf_to_canonical({mapping[k]: v for k, v in state.items()}, config)
    assert not excluded
    return result


def canonical_to_native(state, config):
    fused = canonical_to_hf(state, config)
    return {key: fused[value] for key, value in native_key_map(config).items()}


def read_hf_headers(path, config, *, namespace="component"):
    """Audit all HF keys/shapes/dtypes without materializing any payload."""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    weight_map = text.read_json(index).get("weight_map") if index.exists() else None
    if index.exists() and (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in weight_map.items())
    ):
        raise text.ConversionError("HF weight_map must be a nonempty tensor-to-shard mapping")
    filenames = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    shapes, owners, dtypes = {}, {}, {}
    for filename in filenames:
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise text.ConversionError(f"Shard must be a filename within the source: {filename!r}")
        text._validate_header_json(path / filename)
        with safe_open(path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in owners:
                    raise text.ConversionError(f"Duplicate tensor across shards: {key}")
                view = handle.get_slice(key)
                shapes[key], owners[key] = tuple(view.get_shape()), filename
                dtypes[key] = text._SAFE_DTYPES.get(view.get_dtype(), view.get_dtype())
    if weight_map is not None and owners != weight_map:
        raise text.ConversionError("HF index/shard key ownership mismatch")
    selected, excluded = _selection(shapes, config, namespace)
    layouts = _layouts(selected, config, namespace)
    text.validate_shapes(
        {k: shapes[k] for k in selected},
        parameter_shapes(config, layouts, namespace=namespace),
        "HF headers",
    )
    text.validate_dtypes({k: dtypes[k] for k in selected}, "HF headers")
    return shapes, owners, dtypes, excluded, layouts


def read_hf(path, config, *, namespace="component"):
    """Audit every index/header before loading any selected component's payload."""
    path = Path(path)
    _, owners, _, excluded, layouts = read_hf_headers(path, config, namespace=namespace)
    selected = set(owners) - set(excluded)
    filenames = sorted(set(owners.values()))
    state = {}
    for filename in filenames:
        with safe_open(path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in selected:
                    state[key] = handle.get_tensor(key)
    return state, excluded, layouts


def load_canonical(path, config):
    metadata = FileSystemReader(path).read_metadata()
    prefix = "app.model."
    entries = {
        k.removeprefix(prefix): v
        for k, v in metadata.state_dict_metadata.items()
        if k.startswith(prefix)
    }
    if any(not hasattr(value, "size") for value in entries.values()):
        raise text.ConversionError("Non-tensor model metadata")
    text.validate_shapes(
        {k: tuple(v.size) for k, v in entries.items()}, parameter_shapes(config), "DCP components"
    )
    text.validate_dtypes({k: v.properties.dtype for k, v in entries.items()}, "DCP components")
    state = {
        prefix + k: torch.empty(v.size, dtype=v.properties.dtype, device="cpu")
        for k, v in entries.items()
    }
    dcp.load(state, checkpoint_id=path, no_dist=True)
    excluded = sorted(k for k in metadata.state_dict_metadata if not k.startswith(prefix))
    return {k.removeprefix(prefix): v for k, v in state.items()}, excluded


def _manifest(config, namespace, layouts, excluded, excluded_dcp=()):
    return {
        "format_version": 1,
        "scope": "qwen3_omni_components",
        "checkpoint_kind": "model_initialization_only",
        "selection": config.selection,
        "source_config": config.raw,
        "export_config": config.export_raw(),
        "namespace": namespace,
        "source_expert_layouts": layouts,
        "excluded_keys": list(excluded),
        "excluded_non_model_dcp_entries": list(excluded_dcp),
    }


def _read_manifest(path, selection):
    manifest = text.read_json(path)
    if (
        manifest.get("format_version") != 1
        or manifest.get("scope") != "qwen3_omni_components"
        or manifest.get("checkpoint_kind") != "model_initialization_only"
        or manifest.get("selection") != selection
    ):
        raise text.ConversionError("Omni component sidecar format/scope/selection mismatch")
    return manifest


def _reject_resume(cfg):
    from .strict_omni import MANIFEST, RESUME_DIR

    path = Path(cfg.load_path)
    if (
        cfg.omni_restore_resume
        or (path / MANIFEST).exists()
        or (path / RESUME_DIR).exists()
        or any(path.glob("rank-*.pt"))
    ):
        raise text.ConversionError(
            "Component conversion is model initialization only; complete-model strict resume "
            "requires a separate optimizer/topology contract. Use legacy Thinker mode for its bridge."
        )


def hf2dcp(cfg, stdout):
    _reject_resume(cfg)
    src, dst = Path(cfg.load_path), Path(cfg.save_path)
    text.require_empty_destination(dst)
    raw = text.read_json(src / "config.json")
    prior = None
    if (src / SIDECAR).exists():
        prior = _read_manifest(src / SIDECAR, cfg.omni_components)
        if raw != prior["export_config"]:
            raise text.ConversionError("HF config disagrees with component sidecar")
        raw = prior["source_config"]
    config = configuration(raw, cfg.omni_components)
    if cfg.omni_component_config is not None:
        if text.read_json(cfg.omni_component_config) != raw:
            raise text.ConversionError("Explicit config disagrees with source config")
    state, excluded, layouts = read_hf(src, config, namespace=cfg.omni_component_namespace)
    canonical, _, _ = hf_to_canonical(state, config, namespace=cfg.omni_component_namespace)
    del state
    dst.mkdir(parents=True, exist_ok=True)
    dcp.save({"app": {"model": canonical}}, checkpoint_id=dst, no_dist=True)
    manifest = _manifest(config, cfg.omni_component_namespace, layouts, excluded)
    if prior:
        manifest["original_excluded_keys"] = prior.get(
            "original_excluded_keys", prior["excluded_keys"]
        )
    (dst / SIDECAR).write_text(json.dumps(manifest, indent=2) + "\n")
    stdout.info(
        "Saved %s model weights; excluded component keys (%d): %s",
        cfg.omni_components,
        len(excluded),
        excluded,
    )


def dcp2hf(cfg, stdout):
    from ._core import _write_hf

    _reject_resume(cfg)
    src, dst = Path(cfg.load_path), Path(cfg.save_path)
    text.require_empty_destination(dst)
    if cfg.max_shard_size <= 0:
        raise text.ConversionError("max_shard_size must be positive")
    manifest = (
        _read_manifest(src / SIDECAR, cfg.omni_components) if (src / SIDECAR).exists() else None
    )
    raw = text.read_json(cfg.omni_component_config) if cfg.omni_component_config else None
    if manifest:
        if raw is not None and raw != manifest["source_config"]:
            raise text.ConversionError("Explicit config disagrees with component sidecar")
        raw = manifest["source_config"]
    if raw is None:
        raise text.ConversionError("Native component DCP export requires omni_component_config")
    config = configuration(raw, cfg.omni_components)
    state, excluded_dcp = load_canonical(src, config)
    output = canonical_to_hf(
        state, config, namespace=cfg.omni_component_namespace, layout=cfg.omni_expert_layout
    )
    del state
    _write_hf(output, dst, cfg.max_shard_size, stdout)
    (dst / "config.json").write_text(json.dumps(config.export_raw(), indent=2) + "\n")
    sidecar = _manifest(
        config,
        cfg.omni_component_namespace,
        {name: cfg.omni_expert_layout for name in config.names},
        manifest.get("excluded_keys", []) if manifest else [],
        excluded_dcp,
    )
    if manifest and "original_excluded_keys" in manifest:
        sidecar["original_excluded_keys"] = manifest["original_excluded_keys"]
    (dst / SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n")
    stdout.info(
        "Exported %s model initialization; excluded non-model DCP entries: %s",
        cfg.omni_components,
        excluded_dcp,
    )
