"""Strict CPU conversion of the Qwen3-Omni Thinker *text* weights.

Released HF experts are indexed split Linear weights. Transformers 5.17 and
PithTrain use fused [experts, out, in] Parameters (no transpose, no .weight).
DCP stores those Parameters as indexed [out, in] slices, as checkpoint.py does.
Media towers/Talker are excluded only by explicit opt-in, with every key reported.
"""

import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig

SIDECAR = "qwen3_omni_text.json"
TEXT_TYPE = "qwen3_omni_moe_text"
_NON_TEXT = ("thinker.audio_tower.", "thinker.visual.", "talker.", "code2wav.")
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_SAFE_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_EXPERT_DTYPE = re.compile(
    r"(.+\.experts)(?:\.\d+)?\.(gate_proj\.weight|up_proj\.weight|gate_up_proj|down_proj(?:\.weight)?)$"
)
_INDEXED = re.compile(r"(layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_up_proj|down_proj)$")


class ConversionError(ValueError):
    """A checkpoint violates the complete text parameter contract."""


def _decode_json(data, label):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ConversionError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(data, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ConversionError(f"{label}: invalid JSON") from error
    if not isinstance(result, dict):
        raise ConversionError(f"{label}: expected a JSON object")
    return result


def read_json(path):
    """Do not silently replace duplicate config, sidecar or index entries."""
    return _decode_json(Path(path).read_bytes(), str(path))


def _validate_header_json(path):
    # safetensors' parser accepts duplicate names with last-value-wins. Check
    # raw JSON first; reading only its bounded header never materializes weights.
    with Path(path).open("rb") as handle:
        size = handle.read(8)
        length = int.from_bytes(size, "little")
        if len(size) != 8 or not 0 < length <= min(100_000_000, Path(path).stat().st_size - 8):
            raise ConversionError(f"{path}: invalid safetensors header length")
        _decode_json(handle.read(length), str(path))


def require_empty_destination(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ConversionError(f"Destination must be empty: {path}")


def validate_dtypes(dtypes, label):
    invalid = {key: str(dtype) for key, dtype in dtypes.items() if dtype not in _FLOAT_DTYPES}
    if invalid:
        raise ConversionError(f"{label}: unsupported non-floating weights: {invalid}")
    groups = {}
    for key, dtype in dtypes.items():
        match = _EXPERT_DTYPE.fullmatch(key)
        if match:
            root, proj = match.groups()
            proj = (
                "gate_up_proj"
                if proj in ("gate_proj.weight", "up_proj.weight", "gate_up_proj")
                else "down_proj"
            )
            groups.setdefault((root, proj), set()).add(dtype)
    for (root, proj), values in groups.items():
        if len(values) > 1:
            raise ConversionError(f"{label}: Mixed expert dtypes: {root}.{proj}: {values}")


def text_config(raw: dict) -> Qwen3OmniMoeTextConfig:
    """Accept exactly full Omni, Thinker, or text configs; never infer dimensions."""
    kind = raw.get("model_type")
    if kind == "qwen3_omni_moe":
        raw = raw["thinker_config"]["text_config"]
    elif kind == "qwen3_omni_moe_thinker":
        raw = raw["text_config"]
    elif kind != TEXT_TYPE:
        raise ConversionError(f"Expected Omni/Thinker/text config, got model_type={kind!r}")
    required = (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "moe_intermediate_size",
        "num_experts",
        "num_experts_per_tok",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ConversionError(f"Missing config dimensions: {missing}")
    for key in required:
        value = raw[key]
        if type(value) is not int or value < (0 if key == "num_experts" else 1):
            raise ConversionError(f"Invalid config dimension {key}={value!r}")
    if raw.get("quantization_config"):
        raise ConversionError("Quantized text checkpoints are unsupported")
    # HF normalizes nested legacy RoPE dictionaries in place; preserve source JSON.
    config = Qwen3OmniMoeTextConfig(**copy.deepcopy(raw))
    if config.tie_word_embeddings or config.attention_dropout or config.hidden_act != "silu":
        raise ConversionError("Native Thinker requires untied embeddings, zero dropout and SiLU")
    if config.decoder_sparse_step < 1:
        raise ConversionError("decoder_sparse_step must be positive")
    if config.num_experts and not 1 <= config.num_experts_per_tok <= config.num_experts:
        raise ConversionError("num_experts_per_tok must be within the expert count")
    if any(
        type(i) is not int or not 0 <= i < config.num_hidden_layers for i in config.mlp_only_layers
    ):
        raise ConversionError("mlp_only_layers must contain valid layer indices")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ConversionError("num_attention_heads must be divisible by num_key_value_heads")
    rope = config.rope_parameters
    if (
        rope["rope_type"] != "default"
        or len(rope["mrope_section"]) != 3
        or sum(rope["mrope_section"]) != config.head_dim // 2
    ):
        raise ConversionError("Unsupported native Thinker RoPE configuration")
    return config


def parameter_shapes(config, layout="canonical", prefix="thinker."):
    """Complete text+head schema; usable without materializing any model weights."""
    if layout not in ("canonical", "split", "fused"):
        raise ConversionError(f"Unknown expert layout: {layout!r}")
    if prefix not in ("thinker.", ""):
        raise ConversionError("prefix must be exactly 'thinker.' or '' (standalone text+head)")
    h, m, e = config.hidden_size, config.moe_intermediate_size, config.num_experts
    shapes = {
        "embed_tokens.weight": (config.vocab_size, h),
        "norm.weight": (h,),
        "lm_head.weight": (config.vocab_size, h),
    }
    for i in range(config.num_hidden_layers):
        root = f"layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[root + norm + ".weight"] = (h,)
        attn = root + "self_attn."
        for proj, out, inp in (
            ("q_proj", config.num_attention_heads * config.head_dim, h),
            ("k_proj", config.num_key_value_heads * config.head_dim, h),
            ("v_proj", config.num_key_value_heads * config.head_dim, h),
            ("o_proj", h, config.num_attention_heads * config.head_dim),
        ):
            shapes[attn + proj + ".weight"] = (out, inp)
            if config.attention_bias:
                shapes[attn + proj + ".bias"] = (out,)
        for norm in ("q_norm", "k_norm"):
            shapes[attn + norm + ".weight"] = (config.head_dim,)
        mlp = root + "mlp."
        is_moe = (
            i not in config.mlp_only_layers and e > 0 and (i + 1) % config.decoder_sparse_step == 0
        )
        if not is_moe:
            for proj, shape in (
                ("gate_proj", (config.intermediate_size, h)),
                ("up_proj", (config.intermediate_size, h)),
                ("down_proj", (h, config.intermediate_size)),
            ):
                shapes[mlp + proj + ".weight"] = shape
            continue
        shapes[mlp + "gate.weight"] = (e, h)
        if layout == "fused":
            shapes[mlp + "experts.gate_up_proj"] = (e, 2 * m, h)
            shapes[mlp + "experts.down_proj"] = (e, h, m)
        else:
            for j in range(e):
                expert = mlp + f"experts.{j}."
                if layout == "split":
                    shapes[expert + "gate_proj.weight"] = (m, h)
                    shapes[expert + "up_proj.weight"] = (m, h)
                    shapes[expert + "down_proj.weight"] = (h, m)
                else:
                    shapes[expert + "gate_up_proj"] = (2 * m, h)
                    shapes[expert + "down_proj"] = (h, m)
    if layout == "canonical":
        return shapes
    return {
        prefix + (key if key == "lm_head.weight" else "model." + key): shape
        for key, shape in shapes.items()
    }


def validate_shapes(actual: Mapping, expected: Mapping, label: str):
    """Report all missing, unexpected and wrong-shape keys together."""
    missing, unexpected = (
        sorted(expected.keys() - actual.keys()),
        sorted(actual.keys() - expected.keys()),
    )
    wrong = {
        key: {"expected": tuple(expected[key]), "actual": tuple(actual[key])}
        for key in expected.keys() & actual.keys()
        if tuple(actual[key]) != tuple(expected[key])
    }
    if missing or unexpected or wrong:
        raise ConversionError(
            f"{label}: missing={missing}; unexpected={unexpected}; shape_mismatches={wrong}"
        )


def validate_tensors(state, expected, label):
    invalid = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if invalid:
        raise ConversionError(f"{label}: unsupported non-floating/non-tensor weights: {invalid}")
    validate_shapes({key: tuple(value.shape) for key, value in state.items()}, expected, label)
    validate_dtypes({key: value.dtype for key, value in state.items()}, label)


def _scope(keys, prefix, text_only):
    excluded = sorted(
        key for key in keys if key.startswith(_NON_TEXT if prefix else ("audio_tower.", "visual."))
    )
    if excluded and not text_only:
        raise ConversionError(f"Non-text keys require explicit omni_text_only=True: {excluded}")
    # Unknown prefixes are deliberately left for strict schema validation.
    return excluded


def hf_to_canonical(state, config, *, prefix="thinker.", text_only=False):
    """Return canonical text tensors and an auditable list of excluded keys."""
    excluded = _scope(state, prefix, text_only)
    excluded_set = set(excluded)
    state = {key: value for key, value in state.items() if key not in excluded_set}
    layout = "fused" if any(key.endswith(".experts.gate_up_proj") for key in state) else "split"
    validate_tensors(state, parameter_shapes(config, layout, prefix), f"HF {layout}")
    raw = {key.removeprefix(prefix).removeprefix("model."): value for key, value in state.items()}
    result = {}
    for key in parameter_shapes(config):
        match = _INDEXED.fullmatch(key)
        if not match:
            result[key] = raw[key]
            continue
        root, idx, proj = match.groups()
        if layout == "fused":
            result[key] = raw[f"{root}.{proj}"][int(idx)].contiguous()
        elif proj == "down_proj":
            result[key] = raw[key + ".weight"]
        else:
            gate = raw[f"{root}.{idx}.gate_proj.weight"]
            up = raw[f"{root}.{idx}.up_proj.weight"]
            result[key] = torch.cat((gate, up), dim=0)
    validate_tensors(result, parameter_shapes(config), "Canonical output")
    return result, excluded


def canonical_to_hf(state, config, *, prefix="thinker.", layout="fused"):
    if layout not in ("split", "fused"):
        raise ConversionError(f"HF export requires split or fused layout, got {layout!r}")
    validate_tensors(state, parameter_shapes(config), "DCP canonical")
    expected = parameter_shapes(config, layout, prefix)
    result = {}
    for key in expected:
        canon = key.removeprefix(prefix).removeprefix("model.")
        if layout == "fused" and canon.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            root, proj = canon.rsplit(".", 1)
            values = [state[f"{root}.{i}.{proj}"] for i in range(config.num_experts)]
            result[key] = torch.stack(values)
        elif layout == "split" and ".experts." in canon:
            root, proj, _ = canon.rsplit(".", 2)
            if proj == "down_proj":
                result[key] = state[root + ".down_proj"]
            else:
                offset = 0 if proj == "gate_proj" else config.moe_intermediate_size
                result[key] = state[root + ".gate_up_proj"][
                    offset : offset + config.moe_intermediate_size
                ].contiguous()
        else:
            result[key] = state[canon]
    validate_tensors(result, expected, "HF output")
    return result


def read_hf_text(load_path, config, *, prefix="thinker.", text_only=False):
    """Validate index and every header before reading text tensors; never load media."""
    load_path = Path(load_path)
    index = load_path / "model.safetensors.index.json"
    weight_map = None
    if index.exists():
        weight_map = read_json(index).get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or not all(isinstance(value, str) and value for value in weight_map.values())
        ):
            raise ConversionError("HF weight_map must be a nonempty tensor-to-shard mapping")
    filenames = (
        sorted(set(weight_map.values())) if weight_map is not None else ["model.safetensors"]
    )
    shapes, owners, dtypes = {}, {}, {}
    for filename in filenames:
        if Path(filename).name != filename:
            raise ConversionError(f"Shard must be a filename within the source: {filename!r}")
        _validate_header_json(load_path / filename)
        with safe_open(load_path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in shapes:
                    raise ConversionError(f"Duplicate tensor across shards: {key}")
                view = handle.get_slice(key)
                shapes[key], owners[key] = tuple(view.get_shape()), filename
                dtypes[key] = _SAFE_DTYPES.get(view.get_dtype(), view.get_dtype())
    if weight_map is not None and owners != weight_map:
        raise ConversionError(
            f"HF index/shard mismatch: missing={sorted(weight_map.keys() - owners.keys())}; unexpected={sorted(owners.keys() - weight_map.keys())}; wrong_shard={sorted(key for key in owners.keys() & weight_map.keys() if owners[key] != weight_map[key])}"
        )
    excluded = _scope(shapes, prefix, text_only)
    excluded_set = set(excluded)
    selected = {key: shape for key, shape in shapes.items() if key not in excluded_set}
    layout = "fused" if any(key.endswith(".experts.gate_up_proj") for key in selected) else "split"
    validate_shapes(selected, parameter_shapes(config, layout, prefix), f"HF {layout} headers")
    validate_dtypes({key: dtypes[key] for key in selected}, "HF headers")
    state = {}
    for filename in filenames:
        with safe_open(load_path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in selected:
                    state[key] = handle.get_tensor(key)
    return state, excluded, layout


def load_canonical(load_path, config):
    """Strictly validate all model metadata *before* allocating CPU tensors."""
    metadata = FileSystemReader(load_path).read_metadata()
    prefix = "app.model."
    entries = {
        key.removeprefix(prefix): meta
        for key, meta in metadata.state_dict_metadata.items()
        if key.startswith(prefix)
    }
    invalid = [key for key, meta in entries.items() if not hasattr(meta, "size")]
    if invalid:
        raise ConversionError(f"Non-tensor model metadata: {invalid}")
    validate_shapes(
        {key: tuple(meta.size) for key, meta in entries.items()},
        parameter_shapes(config),
        "DCP metadata",
    )
    validate_dtypes({key: meta.properties.dtype for key, meta in entries.items()}, "DCP metadata")
    state = {
        prefix + key: torch.empty(meta.size, dtype=meta.properties.dtype, device="cpu")
        for key, meta in entries.items()
    }
    dcp.load(state, checkpoint_id=load_path, no_dist=True)
    return {key.removeprefix(prefix): value for key, value in state.items()}


class Qwen3OmniTextConverter:
    name = "qwen3_omni_text"

    def detect_hf(self, load_path):
        path = Path(load_path) / "config.json"
        return path.exists() and read_json(path).get("model_type") in (
            TEXT_TYPE,
            "qwen3_omni_moe",
            "qwen3_omni_moe_thinker",
        )

    def detect_dcp(self, metadata):
        # Text-only canonical keys are not a unique family identifier. Export
        # uses the sidecar or an explicit config instead of guessing by shape.
        return False

    def hf2dcp(self, load_path, save_path, stdout, *, prefix="thinker.", text_only=False):
        save_path = Path(save_path)
        require_empty_destination(save_path)
        raw = read_json(Path(load_path) / "config.json")
        if raw.get("quantization_config"):
            raise ConversionError("Quantized Omni checkpoints are unsupported")
        config = text_config(raw)
        state, excluded, layout = read_hf_text(
            load_path, config, prefix=prefix, text_only=text_only
        )
        canonical, _ = hf_to_canonical(state, config, prefix=prefix)
        del state  # release split source gate/up storage before the DCP writer runs
        save_path.mkdir(parents=True, exist_ok=True)
        dcp.save({"app": {"model": canonical}}, checkpoint_id=save_path, no_dist=True)
        manifest = {
            "format_version": 1,
            "scope": "thinker_text_only",
            "text_config": config.to_dict(),
            "source_prefix": prefix,
            "source_expert_layout": layout,
            "excluded_keys": excluded,
        }
        (save_path / SIDECAR).write_text(json.dumps(manifest, indent=2) + "\n")
        stdout.info(
            "Saved %d text weights; excluded non-text keys (%d): %s",
            len(canonical),
            len(excluded),
            excluded,
        )

    def postprocess_canonical(self, canonical, stdout):
        raise ConversionError(
            "Omni export requires an explicit text config or its conversion sidecar"
        )


def export_config(load_path, config_path=None):
    """Recover shape/semantic contract, also for native DCPs without a sidecar."""
    path = Path(load_path) / SIDECAR
    sidecar = None
    if path.exists():
        manifest = read_json(path)
        if manifest.get("format_version") != 1 or manifest.get("scope") != "thinker_text_only":
            raise ConversionError("Unsupported Omni text sidecar format/scope")
        sidecar = text_config(manifest["text_config"])
    explicit = text_config(read_json(config_path)) if config_path else None
    if explicit and sidecar:
        # In particular a RoPE/router config change must not silently alter predictions.
        left, right = explicit.to_dict(), sidecar.to_dict()
        differences = [
            key
            for key in left.keys() | right.keys()
            if key not in ("transformers_version", "_name_or_path")
            and left.get(key) != right.get(key)
        ]
        if differences:
            raise ConversionError(
                f"Explicit config disagrees with Omni sidecar: {sorted(differences)}"
            )
    return explicit or sidecar
