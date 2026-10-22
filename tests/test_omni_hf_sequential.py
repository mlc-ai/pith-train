"""Actual HF CPU parity/restore and explicit serial-residency admission."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pithtrain.tasks.omni_hf_rollout import (
    LocalHFRolloutCfg,
    _canonical,
    _config,
    _Engine,
    _new_hf,
    weight_memory_plan,
    worker_contract,
)
from pithtrain.tasks.omni_rl import _cpu_tree
from tests.test_omni_hf_rollout import raw_config
from tests.test_omni_rl import tree_equal


def actual_inputs(kind):
    torch.manual_seed(42)
    if kind == "text":
        raw = raw_config()
        prompt = {"input_ids": torch.tensor([[2, 3, 4]])}
        cfg = LocalHFRolloutCfg(max_sequence_length=10, max_new_tokens=2)
    else:
        from tests.test_qwen3_omni_multimodal import context

        config, prompt = context(kind, frames=8)
        raw = config.to_dict()
        raw["text_config"] = raw_config()
        raw["text_config"].update(hidden_size=24, vocab_size=128)
        cfg = LocalHFRolloutCfg(scope="thinker", max_sequence_length=64, max_new_tokens=1)
    model, config = _new_hf(raw, cfg.scope)
    reference = _canonical(_cpu_tree(model.state_dict()), config, cfg.scope)
    current = {k: v.clone() for k, v in reference.items()}
    heads = [k for k in current if k.endswith("lm_head.weight")]
    assert len(heads) == 1
    current[heads[0]].mul_(1.1)
    return raw, prompt, cfg, reference, current


def observe_engine(engine, monkeypatch):
    """Observe real CPU .to/forward order, not a claim of measured CUDA residency."""
    events = []
    for role in ("policy", "reference"):
        model = getattr(engine, role)
        actual = model.to

        def move(*args, _actual=actual, _role=role, **kwargs):
            events.append(("move", _role, str(args[0])))
            return _actual(*args, **kwargs)

        monkeypatch.setattr(model, "to", move)
    actual_logits = engine._logits

    def logits(model, *args):
        events.append(("logits", "policy" if model is engine.policy else "reference"))
        return actual_logits(model, *args)

    monkeypatch.setattr(engine, "_logits", logits)
    return events


@pytest.mark.parametrize("kind", ["text", "audio", "all"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_serial_real_hf_outputs_rng_weights_and_phase_order(kind, dtype, monkeypatch):
    torch.set_num_threads(1)
    raw, prompt, cfg, reference, current = actual_inputs(kind)
    cfg = replace(cfg, compute_dtype=dtype)
    resident = _Engine(raw, cfg, reference)
    serial = _Engine(raw, replace(cfg, sequential_models=True), reference)
    assert (
        resident.synchronize(current)["installed_weights_sha256"]
        == serial.synchronize(current)["installed_weights_sha256"]
    )
    before_prompt = _cpu_tree(prompt)
    before_rng = torch.random.get_rng_state().clone()
    rng = torch.Generator().manual_seed(42).get_state()
    expected = resident.generate(prompt, rng, 1)
    events = observe_engine(serial, monkeypatch)
    actual = serial.generate(prompt, rng, 1)
    tree_equal(expected, actual)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert torch.equal(rng, torch.Generator().manual_seed(42).get_state())
    tree_equal(prompt, before_prompt)
    tree_equal(_canonical(serial.policy.state_dict(), serial.config, cfg.scope), current)
    tree_equal(_canonical(serial.reference.state_dict(), serial.config, cfg.scope), reference)
    assert not torch.equal(actual["old_logprobs"], actual["reference_logprobs"])
    moves = [i for i, e in enumerate(events) if e[0] == "move"]
    assert [events[i] for i in moves] == [
        ("move", "policy", "cpu"),
        ("move", "reference", "cpu"),
        ("move", "reference", "cpu"),
        ("move", "policy", "cpu"),
    ]
    assert all(e == ("logits", "policy") for e in events[: moves[0]])
    assert events[moves[1] + 1 : moves[2]] == [("logits", "reference")] * cfg.group_size
    assert all(
        p.dtype == torch.float32 for m in (serial.policy, serial.reference) for p in m.parameters()
    )
    assert not torch.cuda.is_initialized()


def test_serial_reference_failure_restores_policy_placement_without_weight_changes(monkeypatch):
    raw, prompt, cfg, reference, current = actual_inputs("text")
    engine = _Engine(raw, replace(cfg, sequential_models=True), reference)
    engine.synchronize(current)
    events = observe_engine(engine, monkeypatch)
    actual = engine._logits

    def fail_reference(model, *args):
        if model is engine.reference:
            raise ValueError("reference scoring failed")
        return actual(model, *args)

    monkeypatch.setattr(engine, "_logits", fail_reference)
    with pytest.raises(ValueError, match="reference scoring failed"):
        engine.generate(prompt, torch.Generator().manual_seed(42).get_state(), 1)
    assert events[-2:] == [("move", "reference", "cpu"), ("move", "policy", "cpu")]
    tree_equal(_canonical(engine.policy.state_dict(), engine.config, "text"), current)
    tree_equal(_canonical(engine.reference.state_dict(), engine.config, "text"), reference)


def test_released_serial_admission_keeps_host_limit_and_binds_worker_contract():
    raw = json.loads(
        (Path(__file__).parent / "configs/qwen3_omni_full/released-config.json").read_text()
    )
    config = _config(raw, "thinker")
    weight = 126876821952
    cfg = LocalHFRolloutCfg(
        scope="thinker",
        device="cuda:0",
        sequential_models=True,
        max_weight_bytes=weight,
        max_host_weight_bytes=5 * weight,
        max_device_weight_bytes=weight,
    )
    plan = weight_memory_plan(config, cfg)
    assert plan["device_resident_weight_bytes"] == weight and plan["device_models"] == 1
    assert plan["host_peak_weight_bytes"] == 5 * weight and plan["worker_models"] == 2
    with pytest.raises(ValueError, match="device weight budget"):
        weight_memory_plan(config, replace(cfg, max_device_weight_bytes=weight - 1))
    with pytest.raises(ValueError, match="host weight residency"):
        weight_memory_plan(config, replace(cfg, max_host_weight_bytes=5 * weight - 1))
    with pytest.raises(ValueError, match="device weight budget"):
        weight_memory_plan(config, replace(cfg, sequential_models=False))
    with pytest.raises(ValueError, match="boolean"):
        replace(cfg, sequential_models="true").validate()
    assert worker_contract(raw, cfg)["sequential_models"] is True
    assert worker_contract(raw, replace(cfg, sequential_models=False))["sequential_models"] is False
    assert not torch.cuda.is_initialized()
