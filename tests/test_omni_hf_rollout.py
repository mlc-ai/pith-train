"""Actual tiny HF CPU rollout, reward, update, installation and strict recovery."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from pithtrain.modules.omni_rollout_data import Prompt, PromptSet, prepare_prompts
from pithtrain.tasks.omni_hf_rollout import (
    ExactAnswerReward,
    HFCanonicalPolicyAdapter,
    LocalHFRolloutCfg,
    LocalHFThinkerRollout,
    _canonical,
    _config,
    _Engine,
    _new_hf,
)
from pithtrain.tasks.omni_rl import (
    OmniRLCfg,
    OmniRLTrainer,
    RolloutRequest,
    _cpu_tree,
    _weights_digest,
)
from tests.test_omni_rl import tree_equal


def raw_config():
    raw = json.loads((Path(__file__).parent / "configs/qwen3_omni_text/tiny.json").read_text())
    raw.update(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        moe_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=2,
    )
    raw["rope_parameters"]["mrope_section"] = [2, 1, 1]
    return raw


def tokenizer(path):
    vocab = {
        "[UNK]": 0,
        "[EOS]": 1,
        "What": 2,
        "is": 3,
        "one": 4,
        "plus": 5,
        "two": 6,
        "?": 7,
        "red": 8,
        "blue": 9,
        "Return": 10,
        "the": 11,
        "word": 12,
        "green": 13,
        "yes": 14,
        "no": 15,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]")
    result.save_pretrained(path)
    return result


def inputs(tmp_path):
    token_path = tmp_path / "tokenizer"
    tokenizer(token_path)
    path = tmp_path / "prompts.jsonl"
    rows = [
        dict(id=f"prompt-{i}", text="Return the word red", media=[], answer="red") for i in range(4)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    data, tokens = prepare_prompts(
        path, token_path, _config(raw_config(), "text"), scope="text", max_length=8
    )
    return data, tokens


def trainer(tmp_path, *, initial=None):
    prompts, tokens = inputs(tmp_path)
    torch.manual_seed(42)
    policy, config = _new_hf(raw_config(), "text")
    policy.requires_grad_(True)
    reference = _canonical(_cpu_tree(policy.state_dict()), config, "text")
    if initial is not None:
        policy.load_state_dict(initial)
    else:
        # A fixed reference independent of the current starting policy gives a
        # real KL gradient even when this tiny random model earns equal rewards.
        with torch.no_grad():
            policy.lm_head.weight.mul_(1.1)
    reward = ExactAnswerReward(prompts, tokens)
    rollout = LocalHFThinkerRollout(
        LocalHFRolloutCfg(max_sequence_length=10, max_new_tokens=2),
        model_config=raw_config(),
        prompts=prompts,
        reference=reference,
        eos_token_id=tokens.eos_token_id,
        reward_config=reward.config,
    )
    cfg = OmniRLCfg()
    cfg.max_requests = cfg.max_updates = 4
    cfg.max_samples = 2
    cfg.max_sequence_length = 10
    backend = HFCanonicalPolicyAdapter(
        policy, torch.optim.AdamW(policy.parameters(), lr=0.002, weight_decay=0)
    )
    return OmniRLTrainer(
        cfg, contract=rollout.contract, backend=backend, rollout=rollout, reward=reward
    )


def test_real_hf_sampling_update_resynchronize_and_fresh_resume(tmp_path):
    torch.set_num_threads(1)
    first = trainer(tmp_path / "first")
    restored = None
    try:
        initial = _cpu_tree(first.backend.model.state_dict())
        result = first.train_step()
        assert result["status"] == "UPDATED"
        first_digest = first.rollout.installation_receipt["installed_weights_sha256"]
        first.backend.save(tmp_path / "checkpoint", first)
        expected = first.train_step()
        second_digest = first.rollout.installation_receipt["installed_weights_sha256"]
        assert second_digest != first_digest
        assert (
            first.rollout.installation_receipt["reference_weights_sha256"]
            == first.contract.reference_policy_sha256
        )
        expected_weights = _cpu_tree(first.backend.model.state_dict())
        expected_optimizer = _cpu_tree(first.backend.optimizer.state_dict())
        # Same content in another directory: identity binds contents, not machine paths.
        restored = trainer(tmp_path / "restored", initial=initial)
        restored.backend.load(tmp_path / "checkpoint", restored)
        assert restored.train_step() == expected
        tree_equal(restored.backend.model.state_dict(), expected_weights)
        tree_equal(restored.backend.optimizer.state_dict(), expected_optimizer)
        tree_equal(restored.state_dict(), first.state_dict())
        assert not torch.cuda.is_initialized()
    finally:
        first.rollout.close()
        if restored is not None:
            restored.rollout.close()


def test_engine_selected_logprobs_match_independent_actual_forward(tmp_path):
    prompts, _ = inputs(tmp_path)
    cfg = LocalHFRolloutCfg(max_sequence_length=10, max_new_tokens=2)
    policy, config = _new_hf(raw_config(), "text")
    reference = _canonical(_cpu_tree(policy.state_dict()), config, "text")
    engine = _Engine(raw_config(), cfg, reference)
    receipt = engine.synchronize(reference)
    assert receipt["installed_weights_sha256"] == _weights_digest(reference)
    result = engine.generate(prompts.payload(0), torch.Generator().manual_seed(19).get_state(), 1)
    for index in range(cfg.group_size):
        logits = engine.policy(result["input_ids"][index : index + 1]).float()
        selected = (
            logits.log_softmax(-1)
            .gather(-1, result["sampled_token_ids"][index : index + 1, :, None])
            .squeeze(-1)
        )
        torch.testing.assert_close(
            result["old_logprobs"][index : index + 1], selected, rtol=0, atol=0
        )
    assert not result["token_mask"][:, :4].any()
    assert result["model_context"] is None


@pytest.mark.parametrize("change", ["data", "rng", "cursor", "reward", "dtype", "reference"])
def test_resume_rejects_drift_before_any_training_parameter_apply(tmp_path, change):
    task = trainer(tmp_path)
    state = task.rollout.state_dict()
    before = _cpu_tree(task.backend.model.state_dict())
    if change == "data":
        task.rollout.prompts.prompts[0].inputs["input_ids"][0, 0] = 3
    elif change == "rng":
        state["rng"] = torch.zeros(2, dtype=torch.uint8)
    elif change == "cursor":
        state["cursor"] = 3
    elif change == "reward":
        state["semantic"]["contract"]["reward_config_sha256"] = "0" * 64
    elif change == "dtype":
        state["semantic"]["cfg"]["compute_dtype"] = "bfloat16"
    else:
        state["semantic"]["contract"]["reference_policy_sha256"] = "0" * 64
    with pytest.raises((ValueError, RuntimeError)):
        task.rollout.validate_state_dict(state)
    tree_equal(before, task.backend.model.state_dict())
    assert task.rollout.worker is None


def test_declared_media_is_not_dropped_by_text_policy(tmp_path):
    path = tmp_path / "prompts.jsonl"
    token_path = tmp_path / "tokenizer"
    tokenizer(token_path)
    (tmp_path / "clip.wav").write_bytes(b"not decoded")
    path.write_text(
        json.dumps(
            dict(
                id="a",
                text="What is this?",
                answer="sound",
                media=[dict(type="audio", path="clip.wav")],
            )
        )
    )
    with pytest.raises(ValueError, match="cannot silently drop"):
        prepare_prompts(path, token_path, _config(raw_config(), "text"), scope="text", max_length=8)


def test_publish_digest_tampering_and_nonleader_are_rejected_without_worker(tmp_path):
    task = trainer(tmp_path)
    snapshot = task.backend.snapshot(0, task.contract)
    bad = replace(snapshot, identity=replace(snapshot.identity, weights_sha256="0" * 64))
    with pytest.raises(ValueError, match="digest"):
        task.rollout.synchronize(bad)
    assert task.rollout.worker is None
    request = RolloutRequest(snapshot.identity, 0, 0, 1234, 2, 2, 10, 0, 1, False)
    with pytest.raises(ValueError, match="distributed"):
        task.rollout.generate(request)


def test_weight_and_prompt_limits_before_allocation(tmp_path):
    task = trainer(tmp_path)
    with pytest.raises(ValueError, match="byte limit"):
        LocalHFThinkerRollout(
            replace(task.rollout.cfg, max_weight_bytes=4),
            model_config=raw_config(),
            prompts=task.rollout.prompts,
            reference=task.rollout.reference,
            eos_token_id=1,
            reward_config=task.reward.config,
        )
    with pytest.raises(ValueError, match="byte limit"):
        PromptSet(
            [Prompt("x", {"input_ids": torch.ones((1, 2), dtype=torch.long)}, "")],
            provenance={},
            max_tensor_bytes=1,
        )


def test_source_file_changes_rejected(tmp_path):
    prompts, _ = inputs(tmp_path)
    (tmp_path / "prompts.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        prompts.payload(0)


def test_unequal_or_nonlocal_layout_is_not_inferred(tmp_path):
    task = trainer(tmp_path)
    snapshot = task.backend.snapshot(0, task.contract)
    task.rollout.identity = snapshot.identity
    with pytest.raises(ValueError, match="bounds"):
        task.rollout.generate(RolloutRequest(snapshot.identity, 0, 0, 1234, 4, 4, 10, 0, 0, True))
    assert task.rollout.worker is None


@pytest.mark.parametrize("kind", ["text", "image", "audio", "video", "av", "all"])
def test_actual_complete_hf_media_sampling_keeps_every_payload(kind):
    from tests.test_qwen3_omni_multimodal import context

    cfg, prompt = context(kind, frames=8)
    raw = cfg.to_dict()
    raw["text_config"] = raw_config()
    raw["text_config"].update(hidden_size=24, vocab_size=128)
    policy, config = _new_hf(raw, "thinker")
    canonical = _canonical(_cpu_tree(policy.state_dict()), config, "thinker")
    options = LocalHFRolloutCfg(scope="thinker", max_sequence_length=64, max_new_tokens=1)
    engine = _Engine(raw, options, canonical)
    assert engine.synchronize(canonical)["installed_weights_sha256"] == _weights_digest(canonical)
    result = engine.generate(prompt, torch.Generator().manual_seed(42).get_state(), 1)
    media_keys = {"pixel_values", "pixel_values_videos", "input_features"}
    for index in range(options.group_size):
        actual = {**(result["media_inputs"][index] or {}), **result["model_context"][index]}
        for name, value in prompt.items():
            if name not in {"input_ids", "attention_mask"}:
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        assert set(result["media_inputs"][index] or {}) == set(prompt) & media_keys
    assert result["token_mask"].sum().item() == 2


def _wait_forever():
    import time

    time.sleep(60)


def test_owned_worker_deadline_terminates_real_process_without_retry():
    import multiprocessing as mp
    import tempfile
    import time

    from pithtrain.tasks.omni_hf_rollout import _WorkerProcess

    process = _WorkerProcess.__new__(_WorkerProcess)
    process.cfg = LocalHFRolloutCfg(timeout_seconds=0.05)
    process.directory = tempfile.TemporaryDirectory()
    context = mp.get_context("spawn")
    process.connection, child = context.Pipe()
    process.process = context.Process(target=_wait_forever, daemon=True)
    process.process.start()
    with pytest.raises(TimeoutError, match="no retry"):
        process._wait("ready", time.monotonic() + process.cfg.timeout_seconds)
    assert process.process is None
    assert process.connection is None
    assert not Path(process.directory.name).exists()
    child.close()


def test_fixed_reference_and_reward_decoder_mutation_are_rejected(tmp_path):
    task = trainer(tmp_path)
    state = task.rollout.state_dict()
    next(iter(task.rollout.reference.values())).add_(1)
    with pytest.raises(ValueError, match="Fixed reference"):
        task.rollout.validate_state_dict(state)
    task.reward.tokenizer.add_tokens(["changed"])
    with pytest.raises(ValueError, match="tokenizer changed"):
        task.reward.validate_state_dict(task.reward.state_dict())


@pytest.mark.parametrize("change", ["config_eps", "runtime_eps", "rope", "forward"])
def test_same_shape_policy_semantic_drift_rejected_before_install(tmp_path, change):
    task = trainer(tmp_path)
    if change == "config_eps":
        task.backend.model.config.rms_norm_eps *= 2
    elif change == "runtime_eps":
        task.backend.model.model.layers[0].input_layernorm.variance_epsilon *= 2
    elif change == "rope":
        task.backend.model.config.rope_parameters["rope_theta"] *= 2
    else:
        task.backend.forward = lambda values, context: task.backend.model(values[0]) * 2
    before = _weights_digest(_cpu_tree(task.backend.model.state_dict()))
    with pytest.raises(ValueError):
        task.rollout.synchronize(task.backend.snapshot(0, task.contract))
    assert task.rollout.worker is None
    assert before == _weights_digest(_cpu_tree(task.backend.model.state_dict()))


def test_worker_deadline_includes_serialization_and_result_loading(tmp_path, monkeypatch):
    """Clock/IPC control simulation; real disk IO, no HF model or worker execution."""
    import tempfile
    from types import SimpleNamespace

    from pithtrain.tasks import omni_hf_rollout as provider

    clock = [0.0]
    monkeypatch.setattr(provider, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    save, load, temporary = torch.save, torch.load, tempfile.TemporaryDirectory

    def slow_save(value, path):
        save(value, path)
        clock[0] += 4.0

    def forbidden_spawn(*args):
        raise AssertionError("Expired serialization must not start a worker")

    monkeypatch.setattr(provider.torch, "save", slow_save)
    monkeypatch.setattr(provider.mp, "get_context", forbidden_spawn)
    monkeypatch.setattr(
        provider.tempfile, "TemporaryDirectory", lambda **kw: temporary(dir=tmp_path, **kw)
    )
    cfg = LocalHFRolloutCfg(timeout_seconds=3.0)
    with pytest.raises(TimeoutError, match="no retry"):
        provider._WorkerProcess({}, cfg, {"weight": torch.ones(1)})
    assert list(tmp_path.iterdir()) == []

    sent, polls, closed = [], [], []
    process = provider._WorkerProcess.__new__(provider._WorkerProcess)
    process.cfg = cfg
    process.directory = SimpleNamespace(name=str(tmp_path))
    process.process = SimpleNamespace(is_alive=lambda: True)
    process.close = lambda: closed.append(True)

    def poll(remaining):
        polls.append(remaining)
        clock[0] += 0.5
        return True

    process.connection = SimpleNamespace(send=sent.append, poll=poll, recv=lambda: ("ok", None))
    with pytest.raises(TimeoutError, match="no retry"):
        process.call("synchronize", {})
    assert not sent and not polls and closed
    closed.clear()

    def serialize(value, path):
        save(value, path)
        clock[0] += 2.0

    def deserialize(*args, **kwargs):
        result = load(*args, **kwargs)
        clock[0] += 1.0
        return result

    save({"receipt": True}, tmp_path / "result.pt")
    monkeypatch.setattr(provider.torch, "save", serialize)
    monkeypatch.setattr(provider.torch, "load", deserialize)
    with pytest.raises(TimeoutError, match="no retry"):
        process.call("synchronize", {})
    assert sent == ["synchronize"] and polls == [1.0] and closed
    assert not (tmp_path / "result.pt").exists()
