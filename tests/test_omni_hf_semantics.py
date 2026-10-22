"""Native semantic-contract fixtures plus real meta audio/vision modules.

The text pipeline imports CUDA-only operators, so its recorded module tree here
is explicitly a contract fixture, not a CPU execution of DualPipeV. HF and native
encoder metadata are instantiated from the real implementations on meta.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pithtrain.modules.omni_rollout_data import file_sha
from pithtrain.modules.resume_semantics import implementation_contract, model_contract, seal
from pithtrain.tasks.omni_hf_rollout import (
    LocalHFRolloutCfg,
    _config,
    _Engine,
    _hf_module_name,
    _new_hf,
    validate_native_runtime,
    validate_policy_semantics,
    worker_contract,
)
from pithtrain.tasks.omni_rl import PithTrainPolicyAdapter, PolicyIdentity, PolicySnapshot
from tests.test_omni_hf_rollout import raw_config, trainer

ROOT = Path(__file__).resolve().parents[1]


def thinker_config():
    from tests.test_qwen3_omni_multimodal import context

    raw = context("all", frames=8)[0].to_dict()
    raw["text_config"] = raw_config()
    raw["text_config"].update(hidden_size=24, vocab_size=128)
    return raw


def native_contract(tmp_path, raw, scope):
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with torch.device("meta"):
        hf, config = _new_hf(raw, scope)
    hf_modules = model_contract(hf)["identity"]["modules"]

    def entry(identifier, attributes=None):
        source = ROOT / (identifier.rsplit(".", 1)[0].replace(".", "/") + ".py")
        return dict(
            implementation=identifier,
            implementation_sha256=file_sha(source),
            numerical_attributes=attributes or {},
            config=None,
        )

    modules = {"": entry("pithtrain.pipeline.dualpipev.DualPipeV")}
    for chunk, global_id in ((0, 0), (1, 1)):
        for suffix, cls in (("input_layernorm", "RMSNorm"), ("mlp.gate", "TopKRouter")):
            name = f"layers.{global_id}.{suffix}"
            modules[f"module.{chunk}.{name}"] = entry(
                "pithtrain.models.qwen3_omni_moe.Qwen3OmniMoeThinkerText" + cls,
                copy.deepcopy(hf_modules["model." + name]["numerical_attributes"]),
            )
        text = config if scope == "text" else config.text_config
        modules[f"module.{chunk}.rotary_emb"] = entry(
            "pithtrain.models.qwen3_omni_moe.Qwen3OmniMoeThinkerTextRotaryEmbedding",
            dict(
                base=text.rope_parameters["rope_theta"],
                head_dim=text.head_dim,
                mrope_section=text.rope_parameters["mrope_section"],
            ),
        )
    if scope == "thinker":
        from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
        from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVideoEncoder

        with torch.device("meta"):
            encoders = dict(
                visual=Qwen3OmniMoeVideoEncoder(config.vision_config),
                audio_tower=Qwen3OmniMoeAudioEncoder(config.audio_config),
            )
        for prefix, encoder in encoders.items():
            for name, value in model_contract(encoder)["identity"]["modules"].items():
                modules[f"module.0.{prefix}" + ("." + name if name else "")] = value
    rank = dict(
        schema=1,
        model=seal(dict(schema=3, modules=modules)),
        forward=dict(
            adapter=implementation_contract(PithTrainPolicyAdapter),
            step=dict(
                name="pithtrain.pipeline.dualpipev.DualPipeV.step",
                source_sha256=file_sha(ROOT / "pithtrain/pipeline/dualpipev.py"),
            ),
        ),
        training=dict(
            model=str(tmp_path),
            fp8=False,
            benchmark=False,
            moe_load_balance_coef=0.0,
            native_multimodal=scope == "thinker",
        ),
    )
    return rank


def ranks(value):
    value["model"] = seal(value["model"]["identity"])
    return seal(dict(schema=1, kind="native_policy_ranks", ranks=[seal(value)]))


@pytest.mark.parametrize("scope", ["text", "thinker"])
def test_native_contract_positive_and_actual_encoder_meta_mapping(tmp_path, scope):
    raw = raw_config()
    if scope == "thinker":
        raw = thinker_config()
    cfg = LocalHFRolloutCfg(scope=scope, compute_dtype="bfloat16")
    rank = native_contract(tmp_path, raw, scope)
    assert validate_policy_semantics(ranks(rank), raw, cfg) == worker_contract(raw, cfg)
    assert _hf_module_name("module.1.layers.17.mlp.gate") == "model.layers.17.mlp.gate"
    assert _hf_module_name("module.0.visual.blocks.1.norm1") == "visual.blocks.1.norm1"
    assert (
        _hf_module_name("module.0.audio_tower.layers.1.self_attn")
        == "audio_tower.layers.1.self_attn"
    )


@pytest.mark.parametrize(
    "change",
    [
        "rms",
        "rope",
        "router",
        "top_k",
        "num_experts_per_tok",
        "mrope_section",
        "head_dim",
        "forward",
        "stored_config",
    ],
)
def test_native_contract_drift_rejected_before_worker_install(tmp_path, change):
    task = trainer(tmp_path / "data")
    task.rollout.cfg = replace(task.rollout.cfg, compute_dtype="bfloat16")
    rank = native_contract(tmp_path, raw_config(), "text")
    modules = rank["model"]["identity"]["modules"]
    if change == "rms":
        modules["module.1.layers.1.input_layernorm"]["numerical_attributes"][
            "variance_epsilon"
        ] *= 2
    elif change == "rope":
        modules["module.0.rotary_emb"]["numerical_attributes"]["base"] *= 2
    elif change == "router":
        modules["module.1.layers.1.mlp.gate"]["numerical_attributes"]["norm_topk_prob"] = False
    elif change in {"top_k", "num_experts_per_tok"}:
        modules["module.1.layers.1.mlp.gate"]["numerical_attributes"][change] = 1
    elif change in {"mrope_section", "head_dim"}:
        modules["module.0.rotary_emb"]["numerical_attributes"][change] = (
            [1, 2, 1] if change == "mrope_section" else 16
        )
    elif change == "forward":
        rank["forward"]["step"]["source_sha256"] = "0" * 64
    else:
        from pithtrain.modules.resume_semantics import _canonical_model_value

        config = _config(raw_config(), "text")
        config.rms_norm_eps *= 2
        modules["module.0.rotary_emb"]["config"] = _canonical_model_value(config)
    original = task.backend.snapshot(0, task.contract)
    semantics = ranks(rank)
    identity = PolicyIdentity(
        0, original.identity.weights_sha256, task.contract, semantics["sha256"]
    )
    with pytest.raises(ValueError):
        task.rollout.synchronize(PolicySnapshot(identity, original.payload, semantics))
    assert task.rollout.worker is None


@pytest.mark.parametrize("prefix", ["visual", "audio_tower"])
def test_actual_encoder_runtime_eps_rejected(tmp_path, prefix):
    raw = thinker_config()
    rank = native_contract(tmp_path, raw, "thinker")
    modules = rank["model"]["identity"]["modules"]
    name = next(
        n
        for n, v in modules.items()
        if n.startswith("module.0." + prefix) and "eps" in v["numerical_attributes"]
    )
    modules[name]["numerical_attributes"]["eps"] *= 2
    with pytest.raises(ValueError, match="numerical attributes"):
        validate_policy_semantics(
            ranks(rank), raw, LocalHFRolloutCfg(scope="thinker", compute_dtype="bfloat16")
        )


def test_live_stored_image_config_drift_is_not_hidden_by_config_file(tmp_path, monkeypatch):
    # Pure metadata contract fixture: actual live .image_config is read even
    # though generic model_contract intentionally only records .config.
    import pithtrain.tasks.omni_hf_rollout as provider
    from pithtrain.contexts import training

    raw = thinker_config()
    model = torch.nn.Module()
    model.image_config = _config(raw, "thinker")
    rank = native_contract(tmp_path, raw, "thinker")
    monkeypatch.setattr(provider, "model_contract", lambda model: rank["model"])
    monkeypatch.setattr(training, "fp8", False, raising=False)
    monkeypatch.setattr(training, "PARAM_DTYPE", torch.bfloat16)
    cfg = LocalHFRolloutCfg(scope="thinker", compute_dtype="bfloat16")
    validate_native_runtime(model, raw, cfg)
    model.image_config.text_config.rope_parameters["rope_theta"] *= 2
    with pytest.raises(ValueError, match="stored image_config"):
        validate_native_runtime(model, raw, cfg)


def test_worker_and_distributed_provider_reject_wrong_transformers_before_allocation(monkeypatch):
    import pithtrain.tasks.omni_hf_rollout as provider
    from pithtrain.modules.omni_hf_transport import DistributedHFThinkerRollout

    monkeypatch.setattr(provider.importlib.metadata, "version", lambda name: "5.16.0")
    with pytest.raises(ValueError, match="5.17.0"):
        _Engine({}, LocalHFRolloutCfg(), {})
    with pytest.raises(ValueError, match="5.17.0"):
        DistributedHFThinkerRollout(
            LocalHFRolloutCfg(),
            group=None,
            dp_rank=0,
            pp_rank=0,
            pp_size=1,
            model_config={},
            prompts=None,
            reference=None,
            eos_token_id=1,
            reward_config={},
        )
