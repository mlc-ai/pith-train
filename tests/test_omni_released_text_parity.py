"""CPU checks for the released-weight text parity gate, on small fake public shards."""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextExperts,
)

from tests.acceptance import omni_full_size as gate
from tests.acceptance import omni_released_continuation as continuation
from tests.acceptance import omni_released_text_parity as parity

ROOT = Path(__file__).resolve().parents[1]
EXPERTS = "thinker.model.layers.0.mlp.experts."
TINY = {
    "model.layers.0.mlp.experts.gate_up_proj": (3, 4, 5),
    "model.layers.0.mlp.experts.down_proj": (3, 5, 2),
    "model.norm.weight": (5,),
}


def full_config():
    return json.loads((ROOT / "tests/configs/qwen3_omni_text/full.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def byte_sha(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def header_row(shape, dtype="BF16"):
    return dict(
        file="model-00001-of-00001.safetensors",
        offset=0,
        bytes=2 * math.prod(shape),
        dtype=dtype,
        shape=tuple(shape),
    )


def released_text_keys(cfg):
    """The public checkpoint's Thinker text keys and shapes, written out independently."""
    h, v, d = cfg["hidden_size"], cfg["vocab_size"], cfg["head_dim"]
    e, m = cfg["num_experts"], cfg["moe_intermediate_size"]
    q, kv = cfg["num_attention_heads"] * d, cfg["num_key_value_heads"] * d
    keys = {
        "thinker.model.embed_tokens.weight": (v, h),
        "thinker.model.norm.weight": (h,),
        "thinker.lm_head.weight": (v, h),
    }
    for i in range(cfg["num_hidden_layers"]):
        layer = f"thinker.model.layers.{i}."
        keys.update(
            {
                layer + "input_layernorm.weight": (h,),
                layer + "post_attention_layernorm.weight": (h,),
                layer + "self_attn.q_proj.weight": (q, h),
                layer + "self_attn.k_proj.weight": (kv, h),
                layer + "self_attn.v_proj.weight": (kv, h),
                layer + "self_attn.o_proj.weight": (h, q),
                layer + "self_attn.q_norm.weight": (d,),
                layer + "self_attn.k_norm.weight": (d,),
                layer + "mlp.gate.weight": (e, h),
            }
        )
        for x in range(e):
            keys[layer + f"mlp.experts.{x}.gate_proj.weight"] = (m, h)
            keys[layer + f"mlp.experts.{x}.up_proj.weight"] = (m, h)
            keys[layer + f"mlp.experts.{x}.down_proj.weight"] = (h, m)
    return keys


def raw_tensors(seed=0):
    generator = torch.Generator().manual_seed(seed)
    tensors = {"thinker.model.norm.weight": torch.randn(5, generator=generator)}
    for x in range(3):
        tensors[f"{EXPERTS}{x}.gate_proj.weight"] = torch.randn(2, 5, generator=generator)
        tensors[f"{EXPERTS}{x}.up_proj.weight"] = torch.randn(2, 5, generator=generator)
        tensors[f"{EXPERTS}{x}.down_proj.weight"] = torch.randn(5, 2, generator=generator)
    # Another component shares the checkpoint; the text plan must leave it alone.
    tensors["thinker.audio_tower.proj.weight"] = torch.randn(2, 2, generator=generator)
    return {key: value.to(torch.bfloat16) for key, value in tensors.items()}


def write_download(source, shards):
    rows = [
        dict(name=name, bytes=(source / name).stat().st_size, observed_sha256=sha(source / name))
        for name in shards
    ]
    download = dict(
        status=parity.DOWNLOAD_STATUS,
        model=gate.MODEL_ID,
        revision=gate.REVISION,
        application_retries=0,
        private_uploads=0,
        files=rows,
    )
    (source / "download.json").write_text(json.dumps(download))
    return sha(source / "download.json")


def fake_release(tmp_path, tensors):
    """Two public-format shards, their pinned metadata, manifest and download receipt."""
    metadata, source = tmp_path / "metadata", tmp_path / "source"
    metadata.mkdir()
    source.mkdir()
    names = sorted(tensors)
    shards = {
        "model-00001-of-00002.safetensors": names[::2],
        "model-00002-of-00002.safetensors": names[1::2],
    }
    index = {}
    for shard, keys in shards.items():
        save_file({key: tensors[key] for key in keys}, source / shard)
        payload = (source / shard).read_bytes()
        size = int.from_bytes(payload[:8], "little")
        (metadata / f"{shard}.header.json").write_bytes(payload[8 : 8 + size])
        index.update(dict.fromkeys(keys, shard))
    for directory in (metadata, source):
        (directory / "config.json").write_text('{"tiny": true}')
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    files = [
        dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(metadata.iterdir())
    ]
    manifest = dict(model=gate.MODEL_ID, revision=gate.REVISION, files=files)
    (metadata / "manifest.json").write_text(json.dumps(manifest))
    receipt = write_download(source, sorted(shards))
    return metadata, source, sha(metadata / "manifest.json"), receipt


def tiny_parameters():
    return {
        name: torch.nn.Parameter(torch.empty(shape, dtype=torch.bfloat16))
        for name, shape in TINY.items()
    }


def test_released_pins_match_the_released_continuation_gate():
    assert parity.METADATA_SHA256 == continuation.METADATA_SHA
    assert parity.DOWNLOAD_STATUS in Path(continuation.__file__).read_text()
    assert gate.OFFICIAL_CONFIG_SHA256 == continuation.original.CONFIG_SHA
    assert gate.REVISION == continuation.original.REVISION


def test_plan_maps_every_full_size_parameter_to_the_public_split_keys():
    cfg = full_config()
    headers = {key: header_row(shape) for key, shape in released_text_keys(cfg).items()}
    headers["thinker.audio_tower.conv2d1.weight"] = header_row((480, 1, 3, 3), "F32")
    headers["talker.model.norm.weight"] = header_row((1024,))
    plan = parity.release_plan(gate.full_shapes(cfg), headers)
    assert len(plan) == 531
    assert sum(len(keys) for keys in plan.values()) == parity.TEXT_RAW_KEYS == 18_867
    assert plan["model.layers.3.mlp.experts.gate_up_proj"][:4] == [
        "thinker.model.layers.3.mlp.experts.0.gate_proj.weight",
        "thinker.model.layers.3.mlp.experts.0.up_proj.weight",
        "thinker.model.layers.3.mlp.experts.1.gate_proj.weight",
        "thinker.model.layers.3.mlp.experts.1.up_proj.weight",
    ]
    assert plan["model.layers.3.mlp.experts.down_proj"][-1] == (
        "thinker.model.layers.3.mlp.experts.127.down_proj.weight"
    )
    assert plan["model.embed_tokens.weight"] == ["thinker.model.embed_tokens.weight"]
    assert plan["lm_head.weight"] == ["thinker.lm_head.weight"]


@pytest.mark.parametrize(
    ("fault", "match"),
    [
        ("missing", "lacks"),
        ("dtype", "dtype or shape"),
        ("shape", "dtype or shape"),
        ("unmapped", "unmapped"),
    ],
)
def test_plan_rejects_any_header_fault_before_a_write(fault, match):
    cfg = full_config()
    headers = {key: header_row(shape) for key, shape in released_text_keys(cfg).items()}
    key = "thinker.model.layers.7.mlp.experts.5.up_proj.weight"
    if fault == "missing":
        del headers[key]
    elif fault == "dtype":
        headers[key]["dtype"] = "F16"
    elif fault == "shape":
        headers[key] = header_row((768, 2047))
    else:
        headers["thinker.model.layers.7.mlp.shared_expert.up_proj.weight"] = header_row((768, 2048))
    with pytest.raises(ValueError, match=match):
        parity.release_plan(gate.full_shapes(cfg), headers)


def test_loader_assembles_experts_like_hf_and_both_models_get_identical_bytes(tmp_path):
    tensors = raw_tensors()
    metadata, source, manifest, receipt = fake_release(tmp_path, tensors)
    headers = parity.pinned_headers(metadata, manifest)
    identity = parity.verify_download(metadata, source, receipt, headers)
    assert [row["name"] for row in identity["shards"]] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert identity["tensors"] == len(tensors) and identity["download_receipt_sha256"] == receipt
    plan = parity.release_plan(TINY, headers)
    reference, native = tiny_parameters(), tiny_parameters()
    reference_map = parity.load_released(reference, plan, headers, source, parts=True)
    native_map = parity.load_released(native, plan, headers, source)
    gate_up = reference["model.layers.0.mlp.experts.gate_up_proj"]
    down = reference["model.layers.0.mlp.experts.down_proj"]
    for x in range(3):
        assert torch.equal(gate_up[x, :2], tensors[f"{EXPERTS}{x}.gate_proj.weight"])
        assert torch.equal(gate_up[x, 2:], tensors[f"{EXPERTS}{x}.up_proj.weight"])
        assert torch.equal(down[x], tensors[f"{EXPERTS}{x}.down_proj.weight"])
    assert torch.equal(reference["model.norm.weight"], tensors["thinker.model.norm.weight"])
    for name, parameter in reference.items():
        assert torch.equal(parameter, native[name])
        assert reference_map[name]["sha256"] == native_map[name]["sha256"] == byte_sha(parameter)
    parts = reference_map["model.layers.0.mlp.experts.gate_up_proj"]["parts"]
    assert [part["key"] for part in parts[:2]] == [
        f"{EXPERTS}0.gate_proj.weight",
        f"{EXPERTS}0.up_proj.weight",
    ]
    assert all(part["sha256"] == byte_sha(tensors[part["key"]]) for part in parts)
    assert "parts" not in native_map["model.norm.weight"]
    assert parity.unchanged(identity, source)


def test_hf_experts_use_gate_proj_as_the_first_half_of_gate_up_proj(tmp_path):
    tensors = raw_tensors(seed=1)
    metadata, source, manifest, _ = fake_release(tmp_path, tensors)
    headers = parity.pinned_headers(metadata, manifest)
    config = Qwen3OmniMoeTextConfig(
        hidden_size=5, moe_intermediate_size=2, num_experts=3, hidden_act="silu"
    )
    config._experts_implementation = "eager"
    experts = Qwen3OmniMoeThinkerTextExperts(config).to(torch.bfloat16)
    parameters = {
        "model.layers.0.mlp.experts.gate_up_proj": experts.gate_up_proj,
        "model.layers.0.mlp.experts.down_proj": experts.down_proj,
        "model.norm.weight": torch.nn.Parameter(torch.empty(5, dtype=torch.bfloat16)),
    }
    parity.load_released(parameters, parity.release_plan(TINY, headers), headers, source)
    experts.float()
    hidden = torch.randn(4, 5, generator=torch.Generator().manual_seed(2))
    index = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1]])
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.9, 0.1]])
    with torch.no_grad():
        actual = experts(hidden, index, weights)
    expected = torch.zeros_like(hidden)
    for token in range(4):
        for slot in range(2):
            x = index[token, slot].item()
            gate_w, up_w, down_w = (
                tensors[f"{EXPERTS}{x}.{part}_proj.weight"].float()
                for part in ("gate", "up", "down")
            )
            value = down_w @ (F.silu(gate_w @ hidden[token]) * (up_w @ hidden[token]))
            expected[token] += weights[token, slot] * value
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fault", ["payload", "receipt", "header", "metadata"])
def test_download_identity_rejects_any_changed_byte(tmp_path, fault):
    tensors = raw_tensors()
    metadata, source, manifest, receipt = fake_release(tmp_path, tensors)
    shard = source / "model-00002-of-00002.safetensors"
    if fault == "metadata":
        (metadata / "config.json").write_text('{"tiny": false}')
        with pytest.raises(ValueError, match="manifest"):
            parity.pinned_headers(metadata, manifest)
        return
    headers = parity.pinned_headers(metadata, manifest)
    if fault == "payload":
        payload = bytearray(shard.read_bytes())
        payload[-1] ^= 1
        shard.write_bytes(bytes(payload))
        match = "SHA-256"
    elif fault == "receipt":
        receipt, match = "0" * 64, "receipt"
    else:
        # Same tensors and a consistent receipt, but header bytes that differ from the pin.
        index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        keys = sorted(key for key, name in index.items() if name == shard.name)
        save_file({key: tensors[key] for key in keys}, shard, metadata={"note": "changed"})
        receipt = write_download(source, sorted({row["file"] for row in headers.values()}))
        match = "header"
    with pytest.raises(ValueError, match=match):
        parity.verify_download(metadata, source, receipt, headers)


def test_pinned_counts_reject_a_release_other_than_the_pinned_one(tmp_path):
    metadata, _, manifest, _ = fake_release(tmp_path, raw_tensors())
    headers = parity.pinned_headers(metadata, manifest)
    with pytest.raises(ValueError, match="pinned shards"):
        parity.pinned_text_plan(TINY, headers)
    with pytest.raises(ValueError, match="pinned manifest"):
        parity.pinned_headers(metadata)


def test_unchanged_notices_a_shard_modified_after_hashing(tmp_path):
    metadata, source, manifest, receipt = fake_release(tmp_path, raw_tensors())
    identity = parity.verify_download(
        metadata, source, receipt, parity.pinned_headers(metadata, manifest)
    )
    assert parity.unchanged(identity, source)
    shard = source / "model-00001-of-00002.safetensors"
    status = shard.stat()
    os.utime(shard, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
    assert not parity.unchanged(identity, source)


def phase_result(cross_entropy, flips, failing):
    return {
        "cross_entropy": cross_entropy,
        "logits_error": 2e-3,
        "routes": {
            "token_layers": 64,
            "flipped_token_layers": sum(flips),
            "flipped_tokens_per_layer": flips,
        },
        "gradient_parameters": [
            {"name": name, "status": "FAIL" if name in failing else "PASS"}
            for name in ("a", "b", "c")
        ],
    }


def test_free_running_outcome_is_recorded_without_deciding_the_verdict():
    result = phase_result(12.5, [1, 2], {"b"})
    loss = torch.tensor(12.5)
    summary = parity.free_running_summary(result, loss, loss)
    assert summary["original_criterion"] == "FAIL" and summary["verdict_effect"] == "NONE"
    assert (summary["gradient_pass"], summary["gradient_fail"], summary["route_flips"]) == (2, 1, 3)
    result["logits_error"] = 1e-4
    result["gradient_parameters"][1]["status"] = "PASS"
    assert parity.free_running_summary(result, loss, loss)["original_criterion"] == "PASS"
    assert parity.free_running_summary(result, loss, loss + 1)["original_criterion"] == "FAIL"


def test_self_control_agreement_compares_routes_failures_and_losses():
    control, native = phase_result(12.5, [1, 2], {"b"}), phase_result(12.5, [1, 2], {"b"})
    agreement = parity.self_control_agreement(control, native)
    assert agreement.pop("available") and agreement.pop("verdict_effect") == "NONE"
    assert all(agreement.values())
    native["routes"]["flipped_tokens_per_layer"] = [2, 1]
    native["gradient_parameters"][0]["status"] = "FAIL"
    agreement = parity.self_control_agreement(control, native)
    assert not agreement["route_flips_equal"] and not agreement["failing_gradients_equal"]
    assert parity.self_control_agreement({"status": "ERROR"}, native) == {
        "available": False,
        "verdict_effect": "NONE",
    }


def test_released_estimate_drops_only_the_released_weight_exclusion():
    estimate, original = parity.resource_estimate(), gate.resource_estimate()
    assert set(original["does_not_cover"]) - set(estimate["does_not_cover"]) == {"released weights"}
    assert estimate["weight_source"] == "released"
    assert (
        estimate["bf16_weights_and_gradients_bytes"] == original["bf16_weights_and_gradients_bytes"]
    )
