"""CPU contracts only: official-size shapes use meta, numerical faults use small tensors."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVideoEncoder, Qwen3OmniMoeVisionEncoder
from tests.acceptance import omni_full_media as gate
from tests.acceptance import omni_full_size as text_gate
from tests.acceptance.omni_topology_contract import BY_NAME
from tests.test_omni_topology_harness import native_constructor_namespace

ROOT = Path(__file__).resolve().parents[1]


def full_config():
    cfg = Qwen3OmniMoeThinkerConfig(
        text_config=json.loads((ROOT / "tests/configs/qwen3_omni_text/full.json").read_text()),
        vision_config=dict(
            depth=27,
            hidden_size=1152,
            num_heads=16,
            intermediate_size=4304,
            out_hidden_size=2048,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[8, 16, 24],
        ),
        audio_config=dict(
            encoder_layers=32,
            encoder_attention_heads=20,
            encoder_ffn_dim=5120,
            d_model=1280,
            output_dim=2048,
            downsample_hidden_size=480,
            num_mel_bins=128,
            n_window=50,
            n_window_infer=800,
            max_source_positions=1500,
            conv_chunksize=500,
        ),
    )
    for component in (cfg.text_config, cfg.vision_config, cfg.audio_config):
        component._attn_implementation = "eager"
    return cfg


def native_full_constructor(modality):
    namespace = native_constructor_namespace(ROOT, BY_NAME["dense-single"], 0)
    namespace["distributed"].cp_size = 1
    namespace.update(
        Qwen3OmniMoeVisionEncoder=Qwen3OmniMoeVisionEncoder,
        Qwen3OmniMoeVideoEncoder=Qwen3OmniMoeVideoEncoder,
        Qwen3OmniMoeAudioEncoder=Qwen3OmniMoeAudioEncoder,
    )
    files = (
        ["qwen3_omni_image", "qwen3_omni_video"]
        if modality == "video"
        else ["qwen3_omni_audio_model" if modality == "audio" else "qwen3_omni_image"]
    )
    for filename in files:
        tree = ast.parse((ROOT / f"pithtrain/models/{filename}.py").read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        for node in classes:
            for fn in [child for child in ast.walk(node) if isinstance(child, ast.FunctionDef)]:
                fn.decorator_list = [
                    d
                    for d in fn.decorator_list
                    if not (
                        isinstance(d, ast.Call)
                        and isinstance(d.func, ast.Attribute)
                        and ast.unparse(d.func) == "torch.compile"
                    )
                ]
        module = ast.Module(body=classes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[f"Qwen3OmniMoeThinker{modality.title()}Model"]


@pytest.mark.parametrize("modality", gate.MODALITIES)
def test_official_full_native_and_hf_parameter_shapes_are_identical(modality):
    config = full_config()
    shapes = gate.full_inventory(config, modality)
    with torch.device("meta"):
        native = native_full_constructor(modality)(config, -1)
    text_gate.check_inventory(gate.parameter_map(native, True), shapes)
    assert len(shapes) == (1056 if modality == "audio" else 882)
    assert all(parameter.is_meta for parameter in native.parameters())
    assert config.text_config.hidden_size == 2048 and config.text_config.num_hidden_layers == 48


@pytest.mark.parametrize("modality", gate.MODALITIES)
def test_full_media_never_silently_accepts_a_tiny_encoder_or_decoder(modality):
    config = full_config()
    config.text_config.num_hidden_layers = 2
    with pytest.raises(AssertionError, match="inventory"):
        gate.full_inventory(config, modality)
    config = full_config()
    if modality == "audio":
        config.audio_config.encoder_layers = 2
    else:
        config.vision_config.depth = 3
        config.vision_config.deepstack_visual_indexes = [0, 1, 2]
    with pytest.raises(AssertionError, match="inventory"):
        gate.full_inventory(config, modality)


@pytest.mark.parametrize("modality", gate.MODALITIES)
def test_meta_to_empty_rebuilds_every_hf_buffer_with_finite_expected_values(modality):
    cfg = full_config()
    cfg.text_config.num_hidden_layers = 1
    cfg.text_config.hidden_size = 64
    cfg.text_config.vocab_size = 64
    cfg.text_config.moe_intermediate_size = 32
    cfg.text_config.num_experts = 4
    cfg.text_config.num_experts_per_tok = 2
    cfg.text_config.num_attention_heads = 4
    cfg.text_config.num_key_value_heads = 2
    cfg.audio_config.encoder_layers = 1
    cfg.audio_config.d_model = 32
    cfg.audio_config.encoder_attention_heads = 4
    cfg.audio_config.encoder_ffn_dim = 64
    cfg.audio_config.downsample_hidden_size = 4
    cfg.audio_config.output_dim = 64
    cfg.vision_config.depth = 1
    cfg.vision_config.hidden_size = 64
    cfg.vision_config.num_heads = 4
    cfg.vision_config.intermediate_size = 96
    cfg.vision_config.out_hidden_size = 64
    cfg.vision_config.deepstack_visual_indexes = [0]
    model = gate.build_model(cfg, modality, False, torch.device("cpu"))
    assert all(not value.is_meta and torch.isfinite(value).all() for value in model.buffers())
    snapshot = {name: value.clone() for name, value in model.named_buffers()}
    assert snapshot
    for value in model.buffers():
        value.fill_(float("nan"))
    gate.restore_reference_buffers(model, cfg, modality, torch.device("cpu"))
    assert snapshot.keys() == dict(model.named_buffers()).keys()
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)


def parameters():
    values = {
        "model.layers.0.mlp.gate.weight": [1e-8, -2e-8],
        "model.norm.weight": [1.0, 2.0],
        "lm_head.weight": [1.0, 2.0],
        "audio_tower.layers.0.self_attn.k_proj.bias": [0.0, 0.0],
        "audio_tower.layers.0.self_attn.q_proj.weight": [3.0, 4.0],
    }
    return {
        name: SimpleNamespace(shape=(2,), grad=torch.tensor(value))
        for name, value in values.items()
    }


def test_diagnostics_keep_small_and_zero_gradients_and_all_parameter_rows():
    actual = parameters()
    shapes = {name: parameter.shape for name, parameter in actual.items()}
    expected = {name: parameter.grad.clone() for name, parameter in actual.items()}
    rows = gate.compare_gradients(actual, shapes, expected)
    assert len(rows) == len(actual)
    assert all(row["status"] == "PASS" and row["compared"] for row in rows)
    assert all(row["threshold"] == 1e-2 and row["difference"] == 0 for row in rows)
    audio = next(row for row in rows if row["name"].endswith("q_proj.weight"))
    assert audio["actual_l2"] == 5 and audio["actual_max_abs"] == 4
    assert audio["reference_l2"] == 5 and audio["difference_max_abs"] == 0


@pytest.mark.parametrize(
    "fault", ["missing", "nonfinite", "wrong_gradient", "tiny_wrong_gradient", "all_component_zero"]
)
def test_numeric_failure_collects_every_later_parameter_before_failing(fault):
    actual = parameters()
    shapes = {name: parameter.shape for name, parameter in actual.items()}
    expected = {name: parameter.grad.clone() for name, parameter in actual.items()}
    first = min(actual)
    if fault == "missing":
        actual[first].grad = None
    elif fault == "nonfinite":
        actual[first].grad.fill_(float("nan"))
    elif fault == "wrong_gradient":
        actual[first].grad.fill_(1)
    elif fault == "tiny_wrong_gradient":
        actual["model.layers.0.mlp.gate.weight"].grad.neg_()
    else:
        for name, parameter in actual.items():
            if name.startswith("audio_tower."):
                parameter.grad.zero_()
    rows = gate.compare_gradients(actual, shapes, expected)
    assert len(rows) == len(actual)
    assert any(row["status"] == "FAIL" for row in rows)
    assert rows[-1]["actual_l2"] > 0
    json.dumps(rows, allow_nan=False)


def test_nse_boundary_is_not_relaxed(monkeypatch):
    actual = parameters()
    expected = {name: parameter.grad.clone() for name, parameter in actual.items()}
    monkeypatch.setattr(text_gate, "metric", lambda *_: 1e-2)
    rows = gate.compare_gradients(actual, {n: p.shape for n, p in actual.items()}, expected)
    assert all(row["status"] == "FAIL" for row in rows)


def test_diagnostics_match_direct_fp64_norm_and_difference_across_chunk_boundary(monkeypatch):
    monkeypatch.setattr(text_gate, "METRIC_CHUNK", 3)
    actual = torch.tensor([0.0, 1e-7, 4.0, -2.0, 1.0, 0.0, 7.0])
    expected = actual * 0.99
    result = text_gate.gradient_diagnostics(actual, expected)
    assert result["actual_l2"] == pytest.approx(actual.double().norm().item())
    assert result["reference_l2"] == pytest.approx(expected.double().norm().item())
    assert result["difference_max_abs"] == pytest.approx(
        (actual.double() - expected.double()).abs().max().item()
    )
    assert result["difference"] == pytest.approx(
        (
            (actual.double() - expected.double()).square().sum()
            / (actual.double().square().sum() + expected.double().square().sum())
        ).item()
    )


@pytest.mark.parametrize("modality", gate.MODALITIES)
def test_memory_plan_is_full_bf16_only_and_requires_measured_headroom(modality):
    plan = gate.estimate(modality)
    assert plan["bf16_weights_and_gradients_bytes"] == 4 * plan["parameters"]
    assert plan["minimum_free_gpu_bytes"] == 4 * plan["parameters"] + 12 * 2**30
    assert plan["recommended_host_memory_bytes"] == 256 * 2**30
    assert plan["outer_timeout_seconds"] == 600
    assert plan["optimizer_update"] == "NOT_RUN"
    assert plan["released_weights"] == "NOT_RUN"
    assert plan["estimate_only"] is True


@pytest.mark.parametrize("modality", gate.MODALITIES)
def test_fixture_is_fixed_first_sample_even_when_second_is_shorter(tmp_path, monkeypatch, modality):
    batches = [
        SimpleNamespace(
            model_inputs={"input_ids": torch.ones(1, length, dtype=torch.long)},
            labels=torch.tensor([[1]]),
        )
        for length in (386, 14)
    ]
    training = SimpleNamespace(
        source_inputs=lambda _: (
            {"recipe": {"revision": text_gate.REVISION}},
            [{"id": "first-fixed"}, {"id": "shorter-second"}],
            batches,
            None,
        )
    )
    validator = lambda *args: torch.arange(386).reshape(1, 1, -1).expand(3, 1, -1)
    validation = SimpleNamespace(**{f"validate_{modality}_inputs": validator})
    monkeypatch.setattr(
        gate.importlib,
        "import_module",
        lambda name: training if name.endswith("_training") else validation,
    )
    (tmp_path / "checksums.sha256").write_text("fixture checksum")
    batch, report = gate.select_fixture(tmp_path, modality, object())
    assert batch is batches[0] and report["sample_id"] == "first-fixed"
    assert report["input_tokens"] == 386
    batches[0].model_inputs["input_ids"] = torch.ones(1, 513, dtype=torch.long)
    with pytest.raises(ValueError, match="fixed first"):
        gate.select_fixture(tmp_path, modality, object())


def test_reference_nan_is_reported_without_losing_later_parameter_diagnostics():
    actual = parameters()
    expected = {name: parameter.grad.clone() for name, parameter in actual.items()}
    expected[min(expected)].fill_(float("nan"))
    rows = gate.compare_gradients(actual, {n: p.shape for n, p in actual.items()}, expected)
    assert len(rows) == len(actual) and rows[0]["status"] == "FAIL"
    assert rows[0]["reference_finite"] is False and rows[0]["nse"] is None
    assert rows[-1]["status"] == "PASS"
    json.dumps(rows, allow_nan=False)
