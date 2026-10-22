"""CPU real-tensor/DCP and released-header contracts for all Omni components."""

import copy
import hashlib
import json
import logging
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
)

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, dcp2hf, hf2dcp
from pithtrain.tasks.convert_checkpoint import qwen3_omni_full as full
from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import ConversionError

LOG = logging.getLogger(__name__)


def tiny_raw():
    text = json.loads((Path(__file__).parent / "configs/qwen3_omni_text/tiny.json").read_text())
    text.update(
        vocab_size=37,
        hidden_size=24,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=10,
        num_experts=3,
        num_experts_per_tok=2,
    )
    text["rope_parameters"]["mrope_section"] = [2, 1, 1]
    return {
        "model_type": "qwen3_omni_moe",
        "enable_audio_output": True,
        "tts_bos_token_id": 3,
        "tts_eos_token_id": 4,
        "tts_pad_token_id": 5,
        "thinker_config": {
            "model_type": "qwen3_omni_moe_thinker",
            "text_config": text,
            "vision_config": {
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_heads": 4,
                "out_hidden_size": 24,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "in_channels": 3,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [0, 1],
            },
            "audio_config": {
                "num_mel_bins": 16,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 24,
                "d_model": 16,
                "output_dim": 24,
                "downsample_hidden_size": 4,
                "max_source_positions": 13,
                "n_window": 50,
                "n_window_infer": 200,
                "conv_chunksize": 2,
            },
            "image_token_id": 10,
            "vision_start_token_id": 11,
            "vision_end_token_id": 12,
            "audio_token_id": 13,
            "audio_start_token_id": 14,
            "audio_end_token_id": 15,
            "video_token_id": 16,
        },
        "talker_config": {
            "model_type": "qwen3_omni_moe_talker",
            "num_code_groups": 3,
            "thinker_hidden_size": 24,
            "accept_hidden_layer": 1,
            "spatial_merge_size": 2,
            "seconds_per_chunk": 2,
            "text_config": {
                "hidden_size": 16,
                "intermediate_size": 20,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "moe_intermediate_size": 10,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "shared_expert_intermediate_size": 12,
                "vocab_size": 37,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "mrope_section": [2, 1, 1],
                    "interleaved": True,
                },
            },
            "code_predictor_config": {
                "hidden_size": 16,
                "intermediate_size": 28,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 17,
                "num_code_groups": 3,
            },
        },
        "code2wav_config": {
            "model_type": "",
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "codebook_size": 17,
            "num_quantizers": 3,
            "decoder_dim": 32,
            "upsample_rates": [2, 2],
            "upsampling_ratios": [2],
            "sliding_window": 8,
        },
    }


def reference(raw, dtype=torch.float32):
    cfg = Qwen3OmniMoeConfig(**copy.deepcopy(raw))
    cfg._attn_implementation = "eager"
    cfg.thinker_config.text_config._experts_implementation = "eager"
    cfg.talker_config.text_config._experts_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(701)
        model = Qwen3OmniMoeForConditionalGeneration(cfg).to(dtype=dtype).eval()
        for param in model.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)
    return model


@pytest.fixture(scope="module")
def source():
    raw = tiny_raw()
    model = reference(raw)
    return raw, {k: v.clone() for k, v in model.state_dict().items()}


def split_fixture(state, layout):
    """Independent released Linear-key fixture; never use converter inverse."""
    result = {}
    for key, value in state.items():
        component = key.split(".")[0]
        selected = layout[component] if isinstance(layout, dict) else layout
        if selected == "split" and key.endswith(".experts.gate_up_proj"):
            root = key.removesuffix("gate_up_proj")
            for i in range(value.shape[0]):
                gate, up = value[i].chunk(2, dim=0)
                result[f"{root}{i}.gate_proj.weight"] = gate.clone()
                result[f"{root}{i}.up_proj.weight"] = up.clone()
        elif selected == "split" and key.endswith(".experts.down_proj"):
            root = key.removesuffix("down_proj")
            for i, expert in enumerate(value):
                result[f"{root}{i}.down_proj.weight"] = expert.clone()
        else:
            result[key] = value.clone()
    return result


def write_hf(path, raw, state):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(raw))
    # Separate gate/up into different shards to exercise actual complete assembly.
    weight_map = {}
    for i in range(2):
        part = dict(list(state.items())[i::2])
        name = f"part-{i}.safetensors"
        save_file(part, path / name)
        weight_map.update({key: name for key in part})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def read_hf(path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    return {
        key: value
        for name in sorted(set(index["weight_map"].values()))
        for key, value in load_file(path / name).items()
    }


def options(**kwargs):
    cfg = ConvertCheckpointCfg()
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def exact(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].dtype == expected[key].dtype, key
        assert torch.equal(actual[key], expected[key]), key


@pytest.mark.parametrize("selection", ["thinker", "talker", "code2wav", "all"])
@pytest.mark.parametrize("layout", ["split", "fused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_hf_dcp_hf_roundtrip(source, tmp_path, selection, layout, dtype):
    raw, fused = source
    original = split_fixture({k: v.to(dtype) for k, v in fused.items()}, layout)
    expected = {
        k: v for k, v in original.items() if selection == "all" or k.startswith(selection + ".")
    }
    src, disk, out = (tmp_path / name for name in ("source", "dcp", "export"))
    write_hf(src, raw, original)
    cfg = options(
        load_path=src,
        save_path=disk,
        omni_components=selection,
        omni_expert_layout=layout,
        max_shard_size=3000,
    )
    hf2dcp(cfg, LOG)
    sidecar = json.loads((disk / full.SIDECAR).read_text())
    assert sidecar["excluded_keys"] == sorted(original.keys() - expected.keys())
    assert sidecar["source_config"] == raw
    assert sidecar["checkpoint_kind"] == "model_initialization_only"
    cfg.load_path, cfg.save_path = disk, out
    dcp2hf(cfg, LOG)
    actual = read_hf(out)
    exact(actual, expected)
    assert json.loads((out / "config.json").read_text()) == (
        raw if selection == "all" else raw[selection + "_config"]
    )
    config = full.configuration(raw, selection)
    canonical, excluded, _ = full.hf_to_canonical(actual, config)
    assert not excluded
    reloaded = full.canonical_to_hf(canonical, config)
    model = reference(raw, dtype)
    if selection == "all":
        model.load_state_dict(reloaded, strict=True)
        exact(model.state_dict(), {k: v.to(dtype) for k, v in fused.items()})
    else:
        component = getattr(model, selection)
        component.load_state_dict(
            {k.removeprefix(selection + "."): v for k, v in reloaded.items()}, strict=True
        )
        exact(
            component.state_dict(),
            {
                k.removeprefix(selection + "."): v.to(dtype)
                for k, v in fused.items()
                if k.startswith(selection + ".")
            },
        )
    cfg.load_path, cfg.save_path = out, tmp_path / "second-dcp"
    hf2dcp(cfg, LOG)
    twice, _ = full.load_canonical(cfg.save_path, config)
    exact(twice, canonical)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("selection", ["thinker", "talker", "code2wav", "all"])
def test_explicit_native_mapping_and_checkpoint_stack(source, selection):
    raw, fused = source
    cfg = full.configuration(raw, selection)
    state, _, _ = full.hf_to_canonical(fused, cfg)
    native = full.canonical_to_native(state, cfg)
    exact(full.native_to_canonical(native, cfg), state)
    if selection == "all":
        assert "thinker.embed_tokens.weight" in native
        assert "thinker.model.embed_tokens.weight" not in native
        for key in (
            "talker.model.codec_embedding.weight",
            "talker.code_predictor.lm_head.1.weight",
            "talker.model.layers.0.mlp.shared_expert_gate.weight",
            "code2wav.decoder.1.block.2.conv1.conv.weight",
        ):
            assert key in native


@pytest.mark.parametrize("selection", ["thinker", "talker", "code2wav"])
def test_standalone_component_file_roundtrip(source, tmp_path, selection):
    raw, fused = source
    state = {
        k.removeprefix(selection + "."): v
        for k, v in fused.items()
        if k.startswith(selection + ".")
    }
    src = tmp_path / "src"
    write_hf(src, raw[selection + "_config"], state)
    cfg = options(
        load_path=src,
        save_path=tmp_path / "dcp",
        omni_components=selection,
        omni_component_namespace="standalone",
    )
    hf2dcp(cfg, LOG)
    cfg.load_path, cfg.save_path = cfg.save_path, tmp_path / "out"
    dcp2hf(cfg, LOG)
    exact(read_hf(cfg.save_path), state)


def test_mixed_component_expert_layouts_are_detected_separately(source):
    raw, fused = source
    layouts = {"thinker": "split", "talker": "fused", "code2wav": "split"}
    state = split_fixture(fused, layouts)
    cfg = full.configuration(raw, "all")
    canonical, excluded, detected = full.hf_to_canonical(state, cfg)
    assert detected == layouts and not excluded
    exact(full.canonical_to_hf(canonical, cfg), fused)


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "shape", "dtype", "expert-dtype", "mixed-layout", "unknown-root"]
)
def test_bad_weight_contracts_fail_before_destination(source, tmp_path, fault):
    raw, fused = source
    state = {k: v.clone() for k, v in fused.items()}
    key = "talker.model.layers.0.mlp.shared_expert_gate.weight"
    if fault == "missing":
        del state[key]
    elif fault == "extra":
        state["code2wav.code_offset"] = torch.ones(3)
    elif fault == "shape":
        state[key] = torch.ones(1)
    elif fault == "dtype":
        state[key] = state[key].int()
    elif fault == "expert-dtype":
        state = split_fixture(state, "split")
        key = "talker.model.layers.0.mlp.experts.0.up_proj.weight"
        state[key] = state[key].double()
    elif fault == "mixed-layout":
        state["talker.model.layers.0.mlp.experts.0.gate_proj.weight"] = torch.ones(10, 16)
    else:
        state["unknown.weight"] = torch.ones(1)
    src = tmp_path / "source"
    write_hf(src, raw, state)
    cfg = options(load_path=src, save_path=tmp_path / "dest", omni_components="all")
    with pytest.raises(ConversionError):
        hf2dcp(cfg, LOG)
    assert not cfg.save_path.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "missing-talker",
        "missing-predictor",
        "dimension",
        "groups",
        "vocabulary",
        "disabled",
        "quantized",
        "partial-all",
    ],
)
def test_invalid_full_configs_fail_closed(source, fault):
    raw = copy.deepcopy(source[0])
    if fault == "missing-talker":
        del raw["talker_config"]
    elif fault == "missing-predictor":
        del raw["talker_config"]["code_predictor_config"]
    elif fault == "dimension":
        del raw["code2wav_config"]["hidden_size"]
    elif fault == "groups":
        raw["code2wav_config"]["num_quantizers"] += 1
    elif fault == "vocabulary":
        raw["talker_config"]["code_predictor_config"]["vocab_size"] += 1
    elif fault == "disabled":
        raw["enable_audio_output"] = False
    elif fault == "quantized":
        raw["talker_config"]["quantization_config"] = {"method": "fp8"}
    else:
        raw = raw["talker_config"]
    with pytest.raises(ConversionError):
        full.configuration(raw, "all")


@pytest.mark.parametrize("fault", ["index-owner", "duplicate-shard", "path-traversal"])
def test_index_integrity(source, tmp_path, fault):
    raw, state = source
    src = tmp_path / "source"
    write_hf(src, raw, state)
    path = src / "model.safetensors.index.json"
    data = json.loads(path.read_text())
    key = next(iter(data["weight_map"]))
    if fault == "index-owner":
        data["weight_map"][key] = "part-1.safetensors"
    elif fault == "path-traversal":
        data["weight_map"][key] = "../outside.safetensors"
    else:
        other = load_file(src / "part-1.safetensors")
        other[key] = state[key]
        save_file(other, src / "part-1.safetensors")
    path.write_text(json.dumps(data))
    with pytest.raises(ConversionError):
        full.read_hf(src, full.configuration(raw, "all"))


@pytest.mark.parametrize(
    "fault",
    [
        "legacy-multimodal",
        "legacy-text",
        "legacy-thinker-only",
        "omitted-selection",
        "wrong-selection",
        "strict-resume",
    ],
)
def test_mode_confusion_fails_without_output(tmp_path, fault):
    src = tmp_path / "source"
    src.mkdir()
    cfg = options(load_path=src, save_path=tmp_path / "out", omni_components="all")
    if fault == "legacy-multimodal":
        cfg.omni_multimodal = True
    elif fault == "legacy-text":
        cfg.omni_text_only = True
    elif fault == "legacy-thinker-only":
        cfg.omni_thinker_only = True
    elif fault == "strict-resume":
        cfg.omni_restore_resume = True
    else:
        (src / full.SIDECAR).write_text(json.dumps({"selection": "all"}))
        cfg.omni_components = None if fault == "omitted-selection" else "talker"
    with pytest.raises((ConversionError, FileNotFoundError)):
        dcp2hf(cfg, LOG)
    assert not cfg.save_path.exists()


def test_native_dcp_config_semantics_and_non_model_reporting(source, tmp_path):
    raw, state = source
    config = full.configuration(raw, "all")
    canonical, _, _ = full.hf_to_canonical(state, config)
    disk = tmp_path / "native"
    dcp.save(
        {"app": {"model": canonical, "optimizer": {"step": torch.tensor(7)}}},
        checkpoint_id=disk,
        no_dist=True,
    )
    cfg = options(load_path=disk, save_path=tmp_path / "out", omni_components="all")
    with pytest.raises(ConversionError, match="requires omni_component_config"):
        dcp2hf(cfg, LOG)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    cfg.omni_component_config = path
    dcp2hf(cfg, LOG)
    sidecar = json.loads((cfg.save_path / full.SIDECAR).read_text())
    assert sidecar["excluded_non_model_dcp_entries"] == ["app.optimizer.step"]
    exact(read_hf(cfg.save_path), state)
    cfg.load_path, cfg.save_path = cfg.save_path, tmp_path / "changed"
    changed = copy.deepcopy(raw)
    changed["talker_config"]["text_config"]["norm_topk_prob"] = False
    path.write_text(json.dumps(changed))
    with pytest.raises(ConversionError, match="disagrees"):
        hf2dcp(cfg, LOG)


def test_roundtrip_reproduces_talker_predictor_and_waveform(source):
    raw, state = source
    cfg = full.configuration(raw, "all")
    canonical, _, _ = full.hf_to_canonical(split_fixture(state, "split"), cfg)
    before, after = reference(raw), reference(raw)
    after.load_state_dict(full.canonical_to_hf(canonical, cfg), strict=True)
    values = torch.linspace(-1, 1, 3 * 24).reshape(1, 3, 24)
    codes = torch.tensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]]])

    def predict(model):
        projected = model.talker.text_projection(values) + model.talker.hidden_projection(values)
        hidden = model.talker.model(inputs_embeds=projected, use_cache=False).last_hidden_state
        outputs = [model.talker.codec_head(hidden)]
        for group, head in enumerate(model.talker.code_predictor.lm_head):
            embedded = model.talker.code_predictor.model.codec_embedding[group](codes[:, group + 1])
            residual = model.talker.code_predictor.model(
                inputs_embeds=embedded + hidden, use_cache=False
            ).last_hidden_state
            outputs.append(head(residual))
        outputs.append(model.code2wav(codes))
        return outputs

    with torch.no_grad():
        left, right = predict(before), predict(after)
    assert len(left) == 4
    for actual, expected in zip(left, right, strict=True):
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected)


def test_released_configuration_preserves_raw_json(tmp_path):
    path = Path(__file__).parent / "configs/qwen3_omni_full/released-config.json"
    raw = json.loads(path.read_text())
    original = copy.deepcopy(raw)
    config = full.configuration(raw, "all")
    assert raw == original
    assert config.raw == original
    manifest = full._manifest(config, "component", {name: "split" for name in full.COMPONENTS}, [])
    sidecar = tmp_path / full.SIDECAR
    sidecar.write_text(json.dumps(manifest))
    restored = full._read_manifest(sidecar, "all")
    assert restored["source_config"] == restored["export_config"] == original
    exported = tmp_path / "config.json"
    exported.write_text(json.dumps(config.export_raw()))
    again = full.configuration(json.loads(exported.read_text()), "all")
    assert again.raw == original
    assert all(
        config.configs[name].to_dict() == again.configs[name].to_dict() for name in full.COMPONENTS
    )
    assert full.parameter_shapes(config) == full.parameter_shapes(again)
    assert len(full.parameter_shapes(config)) == 19306


def test_released_26291_complete_header_schema():
    path = Path(__file__).parent / "configs/qwen3_omni_full/released-config.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"
    )
    config = full.configuration(json.loads(path.read_text()), "all")
    shapes = full.parameter_shapes(config, "split")
    assert {name: sum(k.startswith(name + ".") for k in shapes) for name in full.COMPONENTS} == {
        "thinker": 19743,
        "talker": 8037,
        "code2wav": 230,
    }
    # Digest of the independent 15 public safetensors headers at revision
    # 26291f793822fb6be9555850f06dfe95f2d7e695; no released payload was loaded.
    digest = hashlib.sha256(
        json.dumps(shapes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == "ad1cc335d7ff1654d7b506a9c4e30ede3675d2b51c465346f3f5f995f1b100cf"
    assert len(config.fused["talker"]) == 397
    assert len(config.fused["code2wav"]) == 230
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("selection", ["thinker", "talker", "code2wav", "all"])
@pytest.mark.parametrize("namespace", ["component", "standalone"])
def test_offline_hf_standard_loader(source, tmp_path, selection, namespace):
    if selection == "all" and namespace == "standalone":
        with pytest.raises(ConversionError, match="namespaces"):
            full.parameter_shapes(
                full.configuration(source[0], selection), "fused", namespace=namespace
            )
        return
    raw, state = source
    config = full.configuration(raw, selection)
    canonical, _, _ = full.hf_to_canonical(state, config)
    disk = tmp_path / "dcp"
    dcp.save({"app": {"model": canonical}}, checkpoint_id=disk, no_dist=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))
    cfg = options(
        load_path=disk,
        save_path=tmp_path / "hf",
        omni_components=selection,
        omni_component_config=config_path,
        omni_component_namespace=namespace,
    )
    dcp2hf(cfg, LOG)
    original = reference(raw)
    component = original if selection == "all" else getattr(original, selection)
    # HF Code2Wav inherits a composite config_class; explicitly pass the correct
    # component config instead of inferring an empty released model_type.
    from_disk, info = type(component).from_pretrained(
        cfg.save_path,
        config=copy.deepcopy(component.config),
        local_files_only=True,
        output_loading_info=True,
        attn_implementation="eager",
        dtype=torch.float32,
        # The official standalone Code2Wav class has no code2wav base prefix.
        # Its public key_mapping API is needed for a composite-namespace export;
        # standalone exports need no mapping. Strict loading checks stay exact.
        key_mapping={r"^code2wav\.": ""}
        if selection == "code2wav" and namespace == "component"
        else None,
    )
    assert all(
        not info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), info
    exact(from_disk.state_dict(), component.state_dict())


def test_existing_checkpoint_stack_unstack_all_namespaces(source, monkeypatch):
    from pithtrain.contexts import distributed
    from pithtrain.modules.checkpoint import to_canonical_model, to_localized_model

    raw, state = source
    config = full.configuration(raw, "all")
    expected, _, _ = full.hf_to_canonical(state, config)
    model = reference(raw)
    # Real HF parameters with precisely the existing native Thinker namespace,
    # then exercise the production stack/unstack implementation (not its imitation).
    thinker = torch.nn.ModuleDict(
        {
            "embed_tokens": model.thinker.model.embed_tokens,
            "layers": model.thinker.model.layers,
            "norm": model.thinker.model.norm,
            "lm_head": model.thinker.lm_head,
            "visual": model.thinker.visual,
            "audio_tower": model.thinker.audio_tower,
        }
    )
    for layers, count in ((thinker["layers"], 3), (model.talker.model.layers, 4)):
        for layer in layers:
            layer.mlp.experts_per_rank = count
    body = torch.nn.ModuleDict(
        {"thinker": thinker, "talker": model.talker, "code2wav": model.code2wav}
    )
    runtime = torch.nn.ModuleDict({"module": torch.nn.ModuleList([body])})
    monkeypatch.setattr(distributed, "ep_rank", 0, raising=False)
    monkeypatch.setattr(distributed, "ep_size", 1, raising=False)
    actual = to_canonical_model(runtime.state_dict(), runtime)
    exact(actual, expected)
    runtime.load_state_dict(to_localized_model(expected, runtime), strict=True)
    exact(to_canonical_model(runtime.state_dict(), runtime), expected)
    assert not torch.cuda.is_initialized()


def test_released_split_full_export_loads_with_hf_conversion_mapping(source, tmp_path):
    raw, state = source
    config = full.configuration(raw, "all")
    canonical, _, _ = full.hf_to_canonical(state, config)
    disk = tmp_path / "dcp"
    dcp.save({"app": {"model": canonical}}, checkpoint_id=disk, no_dist=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))
    cfg = options(
        load_path=disk,
        save_path=tmp_path / "hf",
        omni_components="all",
        omni_component_config=config_path,
        omni_expert_layout="split",
    )
    dcp2hf(cfg, LOG)
    from_disk, info = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        cfg.save_path,
        local_files_only=True,
        output_loading_info=True,
        attn_implementation="eager",
        dtype=torch.float32,
    )
    assert all(
        not info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), info
    exact(from_disk.state_dict(), state)


@pytest.mark.parametrize("missing", ["talker", "code2wav"])
def test_all_cannot_be_a_partial_checkpoint(source, tmp_path, missing):
    raw, state = source
    src = tmp_path / "source"
    write_hf(src, raw, {k: v for k, v in state.items() if not k.startswith(missing + ".")})
    cfg = options(load_path=src, save_path=tmp_path / "dest", omni_components="all")
    with pytest.raises(ConversionError, match="missing="):
        hf2dcp(cfg, LOG)
    assert not cfg.save_path.exists()


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "integer"])
def test_bad_canonical_dcp_is_rejected_before_hf_output(source, tmp_path, fault):
    raw, fused = source
    config = full.configuration(raw, "all")
    state, _, _ = full.hf_to_canonical(fused, config)
    key = "talker.code_predictor.model.codec_embedding.0.weight"
    if fault == "missing":
        del state[key]
    elif fault == "extra":
        state["code2wav.code_offset"] = torch.ones(3)
    elif fault == "shape":
        state[key] = torch.ones(1)
    else:
        state[key] = state[key].int()
    disk = tmp_path / "dcp"
    dcp.save({"app": {"model": state}}, checkpoint_id=disk, no_dist=True)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))
    cfg = options(
        load_path=disk,
        save_path=tmp_path / "out",
        omni_components="all",
        omni_component_config=config_path,
    )
    with pytest.raises(ConversionError):
        dcp2hf(cfg, LOG)
    assert not cfg.save_path.exists()


@pytest.mark.parametrize("kind", ["manifest", "incomplete-rank", "companion"])
def test_full_strict_resume_is_not_misrepresented_as_model_initialization(tmp_path, kind):
    from pithtrain.tasks.convert_checkpoint.strict_omni import MANIFEST, RESUME_DIR

    src = tmp_path / "source"
    src.mkdir()
    if kind == "manifest":
        (src / MANIFEST).write_text("{}")
    elif kind == "incomplete-rank":
        (src / "rank-00000.pt").write_bytes(b"incomplete")
    else:
        (src / RESUME_DIR).mkdir()
    cfg = options(load_path=src, save_path=tmp_path / "out", omni_components="all")
    with pytest.raises(ConversionError, match="optimizer/topology contract"):
        dcp2hf(cfg, LOG)
    assert not cfg.save_path.exists()


def test_real_cli_roundtrip_requires_explicit_selection(source, tmp_path):
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    script = repo / "examples/convert_checkpoint/qwen3-omni-full/script.py"
    raw, state = source
    src, disk, out = (tmp_path / name for name in ("source", "dcp", "hf"))
    write_hf(src, raw, state)
    env = dict(
        os.environ,
        PYTHONPATH=str(repo),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
    )
    prefix = [sys.executable, "-B", str(script)]
    rejected = subprocess.run(
        prefix + ["hf2dcp", str(src), str(disk)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode == 2 and "--components" in rejected.stderr
    assert not disk.exists()
    for operation, load, save in (("hf2dcp", src, disk), ("dcp2hf", disk, out)):
        result = subprocess.run(
            prefix + [operation, str(load), str(save), "--components", "all"],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    exact(read_hf(out), state)


def test_hf_own_serialized_full_and_standalone_talker_configs(source, tmp_path):
    raw, state = source
    model = reference(raw)
    serialized = model.config.to_dict()
    assert serialized["talker_config"]["model_type"] == ""
    assert serialized["talker_config"]["text_config"]["num_local_experts"] == 4
    config = full.configuration(serialized, "all")
    canonical, _, _ = full.hf_to_canonical(state, config)
    exact(full.canonical_to_hf(canonical, config), state)
    src = tmp_path / "hf-saved-talker"
    model.talker.save_pretrained(src, safe_serialization=True)
    saved = json.loads((src / "config.json").read_text())
    assert "model_type" not in saved  # save_pretrained diff omits the empty default.
    cfg = options(
        load_path=src,
        save_path=tmp_path / "dcp",
        omni_components="talker",
        omni_component_namespace="standalone",
    )
    hf2dcp(cfg, LOG)
    cfg.load_path, cfg.save_path = cfg.save_path, tmp_path / "out"
    dcp2hf(cfg, LOG)
    exact(read_hf(cfg.save_path), model.talker.state_dict())
    assert json.loads((cfg.save_path / "config.json").read_text()) == saved


@pytest.mark.parametrize("alias", [5, True, 0])
def test_talker_expert_alias_conflict_rejected(source, alias):
    raw = copy.deepcopy(source[0])
    raw["talker_config"]["text_config"]["num_local_experts"] = alias
    with pytest.raises(ConversionError, match="expert"):
        full.configuration(raw, "all")
