"""CPU-only full-size contract checks; full parameters are constructed only on meta."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel as HFModel,
)

from tests.acceptance import omni_full_size as gate
from tests.acceptance.omni_topology_contract import BY_NAME
from tests.test_omni_topology_harness import native_constructor_namespace

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "tests/configs/qwen3_omni_text/full.json").read_text())


def test_official_full_shape_inventory_agrees_with_both_actual_meta_constructors():
    cfg = config()
    expected = gate.full_shapes(cfg)
    assert len(expected) == 531
    assert (
        sum(
            parameter.numel()
            for parameter in (torch.empty(shape, device="meta") for shape in expected.values())
        )
        == 30_532_646_912
    )
    with torch.device("meta"):
        hf = torch.nn.ModuleDict(
            {
                "model": HFModel(Qwen3OmniMoeTextConfig(**cfg)),
                "lm_head": torch.nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False),
            }
        )
    gate.check_inventory(gate.parameter_map(hf, False), expected)
    namespace = native_constructor_namespace(ROOT, BY_NAME["dense-single"], 0)
    cfg["pad_token_id"] = None
    with torch.device("meta"):
        native = namespace["Qwen3OmniMoeThinkerTextModel"](SimpleNamespace(**cfg), -1)
    gate.check_inventory(gate.parameter_map(native, True), expected)


def test_full_memory_lower_bounds_do_not_claim_fp32_main_grad_fits_one_h200():
    estimate = gate.resource_estimate()
    assert estimate["bf16_weights_and_gradients_bytes"] == 122_130_587_648
    assert estimate["bf16_weights_and_fp32_gradients_bytes"] > 141 * 2**30
    assert estimate["estimate_only"] is True
    assert estimate["sequence_length"] == 32 and estimate["batch"] == 1


def official_fixture(tmp_path, monkeypatch):
    cfg = config()
    official = copy.deepcopy(cfg)
    rope = official.pop("rope_parameters")
    official["rope_theta"] = rope.pop("rope_theta")
    official["rope_scaling"] = dict(rope, type="default")
    payload = json.dumps({"thinker_config": {"text_config": official}}).encode()
    path = tmp_path / "official.json"
    path.write_bytes(payload)
    monkeypatch.setattr(gate, "OFFICIAL_CONFIG_SHA256", hashlib.sha256(payload).hexdigest())
    return path


def test_official_config_verifies_shape_and_rope_fields(tmp_path, monkeypatch):
    path = official_fixture(tmp_path, monkeypatch)
    value, shapes = gate.load_config(path, ROOT)
    assert value == config() and len(shapes) == 531
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="pinned revision"):
        gate.load_config(path, ROOT)


@pytest.mark.parametrize("field", ["num_hidden_layers", "num_experts", "head_dim", "vocab_size"])
def test_smaller_or_modified_config_cannot_claim_full_size(tmp_path, monkeypatch, field):
    path = official_fixture(tmp_path, monkeypatch)
    value = json.loads(path.read_text())
    value["thinker_config"]["text_config"][field] //= 2
    payload = json.dumps(value).encode()
    path.write_bytes(payload)
    monkeypatch.setattr(gate, "OFFICIAL_CONFIG_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match=field):
        gate.load_config(path, ROOT)


def small_parameters():
    return {
        "model.layers.0.mlp.gate.weight": torch.nn.Parameter(
            torch.empty(4, 8, dtype=torch.bfloat16)
        ),
        "model.norm.weight": torch.nn.Parameter(torch.empty(8, dtype=torch.bfloat16)),
        "lm_head.weight": torch.nn.Parameter(torch.empty(32, 8, dtype=torch.bfloat16)),
    }


def test_name_seeded_weights_have_identical_bits_despite_order_and_new_rng_state():
    first, second = small_parameters(), dict(reversed(list(small_parameters().items())))
    first_receipt = gate.initialize(first, 0.02)
    torch.manual_seed(89213)
    second_receipt = gate.initialize(second, 0.02)
    assert first_receipt == second_receipt
    for name in first:
        assert torch.equal(first[name], second[name])
    assert torch.equal(first["model.norm.weight"], torch.ones(8))
    third_receipt = gate.initialize(second, 0.02, seed=1)
    assert first_receipt != third_receipt
    assert not torch.equal(first["lm_head.weight"], second["lm_head.weight"])


def test_fingerprints_are_from_actual_initialized_parameter_bytes():
    parameters = small_parameters()
    receipt = gate.initialize(parameters, 0.02)
    for name, parameter in parameters.items():
        digest = hashlib.sha256(parameter.detach().view(torch.uint8).numpy().tobytes()).hexdigest()
        assert digest == receipt[name]["sha256"]


@pytest.mark.parametrize("fault", ["missing", "shape", "nan", "zero", "wrong_gradient"])
def test_every_full_size_gate_gradient_fault_is_rejected(fault):
    name = "model.norm.weight"
    parameter = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    parameter.grad = torch.ones_like(parameter) * 0.01
    reference = {name: parameter.grad.clone()}
    shapes = {name: (8,)}
    if fault == "missing":
        parameter.grad = None
    elif fault == "shape":
        shapes[name] = (4, 2)
    elif fault == "nan":
        parameter.grad[0] = float("nan")
    elif fault == "zero":
        parameter.grad.zero_()
    elif fault == "wrong_gradient":
        parameter.grad.neg_()
    if fault == "shape":
        with pytest.raises(AssertionError, match="shape"):
            gate.gradient_rows({name: parameter}, shapes, reference)
    else:
        rows = gate.gradient_rows({name: parameter}, shapes, reference)
        assert rows[0]["status"] == "FAIL"


def test_successful_comparison_keeps_every_parameter_and_original_threshold(monkeypatch):
    parameters = small_parameters()
    shapes = {name: tuple(parameter.shape) for name, parameter in parameters.items()}
    for parameter in parameters.values():
        parameter.grad = torch.full_like(parameter, 0.001)
    reference = {name: parameter.grad.clone() for name, parameter in parameters.items()}
    rows = gate.gradient_rows(parameters, shapes, reference)
    assert len(rows) == len(parameters)
    assert all(row["status"] == "PASS" and row["difference"] == 0 for row in rows)
    assert all(row["threshold"] == 1e-2 for row in rows)
    monkeypatch.setattr(gate, "metric", lambda *_: 1e-2)
    assert all(row["status"] == "FAIL" for row in gate.gradient_rows(parameters, shapes, reference))


def test_streaming_metric_agrees_with_full_fp64_expression():
    generator = torch.Generator().manual_seed(17)
    a = torch.randn((1 << 20) + 31, generator=generator)
    b = a + torch.randn(a.shape, generator=generator) * 0.03
    expected = (
        (a.double() - b.double()).square().sum()
        / (a.double().square().sum() + b.double().square().sum())
    ).item()
    assert gate.metric(a, b) == pytest.approx(expected, rel=1e-12)
    assert gate.metric(torch.zeros(7), torch.zeros(7)) == 0
