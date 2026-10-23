"""Actual tiny native Talker CPU policy tests; no released/GPU/quality claim."""

import copy
from dataclasses import replace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

from pithtrain.tasks import omni_codec_rl as codec
from pithtrain.tasks import omni_rl
from tests.acceptance.omni_speech_chain import exact
from tests.test_omni_speech_integration import pair


@pytest.fixture(autouse=True)
def cpu_state():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


def policy_and_prompt(frames=2):
    native, hf = pair()
    config = Qwen3OmniMoeConfig(talker_config=copy.deepcopy(native.config))
    config.im_start_token_id, config.user_token_id, config.assistant_token_id = 100, 101, 102
    config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id = 200, 201, 202
    policy = codec.CodecPolicy(native, config, max_frames=frames)
    torch.manual_seed(71)
    ids = torch.tensor([[100, 101, 198, 40, 100, 102, 198, 41, 42]])
    prompt = dict(
        input_ids=ids,
        thinker_embeddings=torch.randn(1, len(ids[0]), 20),
        thinker_hidden=torch.randn(1, len(ids[0]), 20),
        special_embeddings=torch.randn(1, 3, 20),
        prompt_lengths=torch.tensor([7]),
    )
    return policy, hf, prompt


def make_contract(provider, reward):
    return omni_rl.RolloutContract(
        chat_template_sha256=omni_rl.fingerprint({"format": "explicit frozen tensor fixture"}),
        processor_sha256=omni_rl.fingerprint({"source": "CPU features, not raw media"}),
        position_config_sha256=omni_rl.fingerprint(provider.actor.config.to_dict()),
        reference_policy_sha256=provider.reference_sha256,
        reward_config_sha256=omni_rl.fingerprint(reward.config),
    )


def request(identity, *, seed=1234, index=0):
    return omni_rl.RolloutRequest(identity, index, index * 2, seed + index, 2, 2, 33, 0, 0, True)


def trainer_fixture():
    policy, _, values = policy_and_prompt()
    dummy = codec.CodecPrompt(
        "preview", values, "a" * 64, torch.zeros(2, 16, dtype=torch.long), False
    )
    preview = codec.LocalCodecRollout(copy.deepcopy(policy), copy.deepcopy(policy), [dummy])
    reward = codec.ExactCodecReward([dummy], policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=0)
    backend = codec.CodecPolicyAdapter(policy, optimizer)
    snapshot = backend.snapshot(0, make_contract(preview, reward))
    preview.synchronize(snapshot)
    batch = preview.generate(request(snapshot.identity))
    assert not torch.equal(batch.sampled_token_ids[0], batch.sampled_token_ids[1])
    length = int(batch.model_context[0]["lengths"][0])
    eos = bool(batch.model_context[0]["reached_eos"][0])
    target = batch.sampled_token_ids[0, : length * 16].reshape(length, 16)
    # One bounded initial group defines a fixed exact-match fixture. Never adapt
    # its target after training; this is protocol correctness, not quality gain.
    prompts = [codec.CodecPrompt(f"prompt-{i}", values, "a" * 64, target, eos) for i in range(3)]
    provider = codec.LocalCodecRollout(copy.deepcopy(policy), copy.deepcopy(policy), prompts)
    reward = codec.ExactCodecReward(prompts, policy)
    cfg = omni_rl.OmniRLCfg()
    cfg.max_updates = cfg.max_requests = 3
    cfg.group_size = cfg.max_samples = 2
    cfg.max_sequence_length = 33
    cfg.micro_batch_size = 1
    cfg.seed = 1234
    contract = make_contract(provider, reward)
    trainer = codec.create_trainer(
        cfg, contract=contract, backend=backend, rollout=provider, reward=reward
    )
    return trainer


@pytest.mark.parametrize("length,eos", [(0, True), (1, True), (2, False)])
def test_action_layout_eos_cap_and_all_sixteen_groups(length, eos):
    codes = torch.arange(length * 16).reshape(1, length, 16) % 23
    actions, mask = codec.action_layout(
        codes, torch.tensor([length]), torch.tensor([eos]), max_frames=2, vocab=23
    )
    assert actions.shape == mask.shape == (1, 33)
    assert int(mask.sum()) == length * 16 + eos
    assert torch.equal(actions[:, : length * 16], codes.flatten(1))
    if eos:
        assert int(actions[0, length * 16]) == 23
    assert (actions[~mask] == 0).all()


@pytest.mark.parametrize("damage", ["code", "padding", "length", "fake_eos", "no_eos_short"])
def test_layout_rejects_invalid_or_fictitious_actions(damage):
    codes, lengths, eos = (
        torch.zeros(1, 2, 16, dtype=torch.long),
        torch.tensor([1]),
        torch.tensor([True]),
    )
    if damage == "code":
        codes[0, 0, 15] = 23
    if damage == "padding":
        codes[0, 1, 0] = 1
    if damage == "length":
        lengths[0] = 3
    if damage == "fake_eos":
        lengths[0] = 2
    if damage == "no_eos_short":
        eos[0] = False
    with pytest.raises(ValueError):
        codec.action_layout(codes, lengths, eos, max_frames=2, vocab=23)


def test_actual_autoregressive_old_logprob_and_fixed_reference():
    trainer = trainer_fixture()
    provider, backend = trainer.rollout, trainer.backend
    snap = backend.snapshot(0, trainer.contract)
    provider.synchronize(snap)
    batch = provider.generate(request(snap.identity))
    for i, context in enumerate(batch.model_context):
        logits = backend.model(batch.input_ids[i : i + 1], **context)
        selected = (
            logits.log_softmax(-1)
            .gather(-1, batch.sampled_token_ids[i : i + 1, None].transpose(1, 2))
            .squeeze(-1)
        )
        mask = batch.token_mask[i : i + 1]
        torch.testing.assert_close(
            selected[mask], batch.old_logprobs[i : i + 1][mask], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            selected[mask], batch.reference_logprobs[i : i + 1][mask], atol=1e-6, rtol=1e-5
        )
    torch.testing.assert_close(
        trainer.reward.score(request(snap.identity), batch), torch.tensor([1.0, 0.0])
    )
    with torch.no_grad():
        next(backend.model.parameters()).add_(0.1)
    newer = backend.snapshot(1, trainer.contract)
    assert newer.identity.weights_sha256 != snap.identity.weights_sha256
    provider.synchronize(newer)
    assert codec._tensor_identity(provider.reference.state_dict()) == provider.reference_sha256


def test_actual_codec_grpo_update_checkpoint_and_fresh_resume(tmp_path):
    trainer = trainer_fixture()
    before = codec._tensor_identity(trainer.backend.model.state_dict())
    report = trainer.train_step()
    assert report["status"] == "UPDATED" and report["gradient_norm"] > 0
    assert codec._tensor_identity(trainer.backend.model.state_dict()) != before
    assert trainer.policy_version == 1
    for prefix in ("talker.model.", "talker.code_predictor.", "talker.text_projection."):
        assert any(
            name.startswith(prefix) and p in trainer.backend.optimizer.state
            for name, p in trainer.backend.model.named_parameters()
        )
    trainer.backend.save(tmp_path / "checkpoint", trainer)
    expected_report = trainer.train_step()
    expected = copy.deepcopy(trainer.backend.model.state_dict())
    expected_optimizer = copy.deepcopy(trainer.backend.optimizer.state_dict())
    expected_task = copy.deepcopy(trainer.state_dict())
    fresh = trainer_fixture()
    fresh.backend.load(tmp_path / "checkpoint", fresh)
    assert fresh.policy_version == fresh.requests == fresh.updates == 1
    actual_report = fresh.train_step()
    exact(actual_report, expected_report)
    exact(fresh.backend.model.state_dict(), expected)
    exact(fresh.backend.optimizer.state_dict(), expected_optimizer)
    exact(fresh.state_dict(), expected_task)


@pytest.mark.parametrize("damage", ["weights", "reference", "prompt", "mask", "identity"])
def test_provider_rejects_stale_or_mutated_semantics(damage):
    trainer = trainer_fixture()
    backend, provider = trainer.backend, trainer.rollout
    snapshot = backend.snapshot(0, trainer.contract)
    if damage == "weights":
        snapshot.payload[next(iter(snapshot.payload))].add_(1)
        with pytest.raises(ValueError, match="identity"):
            provider.synchronize(snapshot)
        return
    provider.synchronize(snapshot)
    if damage == "reference":
        with torch.no_grad():
            next(provider.reference.parameters()).add_(1)
    elif damage == "prompt":
        provider.prompts[0].payload["thinker_hidden"].add_(1)
    elif damage == "identity":
        snapshot = replace(snapshot, identity=replace(snapshot.identity, version=1))
    if damage != "mask":
        with pytest.raises(ValueError):
            provider.generate(request(snapshot.identity))
    else:
        batch = provider.generate(request(snapshot.identity))
        batch.token_mask[0, 0] = False
        with pytest.raises(ValueError, match="mask"):
            backend.validate_batch_layout(batch, trainer.cfg, torch.ones(2))


def test_resume_rejects_condition_change_before_model_mutation(tmp_path):
    trainer = trainer_fixture()
    trainer.train_step()
    trainer.backend.save(tmp_path / "checkpoint", trainer)
    fresh = trainer_fixture()
    before = codec._tensor_identity(fresh.backend.model.state_dict())
    fresh.rollout.prompts[0].payload["special_embeddings"].add_(0.01)
    with pytest.raises(ValueError, match="preflight"):
        fresh.backend.load(tmp_path / "checkpoint", fresh)
    assert codec._tensor_identity(fresh.backend.model.state_dict()) == before
    assert fresh.requests == fresh.updates == 0


def test_all_eos_policy_keeps_conditional_adam_state_honest(tmp_path):
    policy, _, prompt = policy_and_prompt()
    actions, mask = codec.action_layout(
        torch.empty(1, 0, 16, dtype=torch.long),
        torch.tensor([0]),
        torch.tensor([True]),
        max_frames=2,
        vocab=23,
    )
    scores = policy(actions, lengths=torch.tensor([0]), reached_eos=torch.tensor([True]), **prompt)
    (-scores.log_softmax(-1)[mask, 23]).sum().backward()
    assert all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for p in policy.talker.code_predictor.parameters()
    )
    assert any(p.grad is None for p in policy.talker.code_predictor.parameters())
    assert any(p.grad is not None for p in policy.talker.model.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("length,eos", [(0, True), (1, True), (2, False)])
def test_codec_selected_logprob_and_vjp_match_independent_hf(length, eos):
    policy, hf, prompt = policy_and_prompt()
    codes = (torch.arange(length * 16).reshape(1, length, 16) * 7) % 23
    lengths, reached = torch.tensor([length]), torch.tensor([eos])
    actions, mask = codec.action_layout(codes, lengths, reached, max_frames=2, vocab=23)
    actual = policy(actions, lengths=lengths, reached_eos=reached, **prompt)
    # Build the HF graph from independent trunk/predictor public modules. The
    # common condition builder has its own HF parity regression; here both
    # sets of projections remain in their respective autograd graphs.
    condition = codec.prepare_talker_condition(
        hf,
        prompt["thinker_embeddings"],
        prompt["thinker_hidden"],
        prompt["input_ids"],
        config=policy.config,
        text_embedding=lambda _: prompt["special_embeddings"],
        prompt_lengths=prompt["prompt_lengths"],
        speaker="ethan",
    )
    embedded = [hf.get_input_embeddings()(codes[..., 0])]
    embedded += [
        table(codes[..., i + 1]) for i, table in enumerate(hf.code_predictor.model.codec_embedding)
    ]
    tail = torch.cat(
        (condition.trailing_text_hidden, condition.tts_pad_embed.expand(-1, length, -1)), 1
    )[:, :length]
    inputs = torch.cat((condition.inputs_embeds, torch.stack(embedded).sum(0) + tail), 1)
    appended = condition.position_ids.amax() + torch.arange(1, length + 1)
    positions = torch.cat((condition.position_ids, appended[None, None].expand(3, 1, -1)), -1)
    attention = torch.cat((condition.attention_mask, torch.ones(1, length, dtype=torch.long)), 1)
    hidden = hf.model(
        inputs_embeds=inputs, attention_mask=attention, position_ids=positions, use_cache=False
    ).last_hidden_state
    begin = condition.inputs_embeds.shape[1] - 1
    first = hf.codec_head(hidden[:, begin:]).float()[..., [*range(23), 31]].log_softmax(-1)
    rows = []
    for frame in range(length):
        rows.append(first[:, frame])
        for group in range(15):
            prefix = torch.stack(
                [hidden[:, begin + frame], *[v[:, frame] for v in embedded[: group + 1]]], 1
            )
            states = hf.code_predictor.model(
                inputs_embeds=prefix, use_cache=False
            ).last_hidden_state
            logp = hf.code_predictor.lm_head[group](states[:, -1]).float().log_softmax(-1)
            rows.append(torch.nn.functional.pad(logp, (0, 1), value=-1e9))
    if eos:
        rows.append(first[:, length])
    expected = torch.stack(rows, 1)
    selected = actual.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)[mask]
    reference = (
        expected.log_softmax(-1)
        .gather(-1, actions[:, : expected.shape[1], None])
        .squeeze(-1)
        .flatten()
    )
    torch.testing.assert_close(selected, reference, atol=2e-6, rtol=1e-5)
    (-selected.sum()).backward()
    (-reference.sum()).backward()
    compared = set()
    for name, parameter in policy.talker.named_parameters():
        other = dict(hf.named_parameters())[name]
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is None and other.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=1e-4, msg=name)
            compared.add(name)
    assert any("text_projection" in name for name in compared)
    if length:
        assert any("code_predictor" in name for name in compared)


@pytest.mark.parametrize("damage", ["installed_actor", "provider_cursor", "reward_cursor"])
def test_codec_strict_identity_and_transaction_guards(damage):
    trainer = trainer_fixture()
    if damage == "installed_actor":
        snapshot = trainer.backend.snapshot(0, trainer.contract)
        trainer.rollout.synchronize(snapshot)
        with torch.no_grad():
            next(trainer.rollout.actor.parameters()).add_(0.01)
        with pytest.raises(ValueError, match="weights changed"):
            trainer.rollout.generate(request(snapshot.identity))
    else:
        state = codec._CheckpointData(trainer).state_dict()
        key = "rollout" if damage == "provider_cursor" else "reward"
        state["rl_state"][key]["calls"] = 1
        with pytest.raises(ValueError, match="cursors differ"):
            codec._CheckpointData(trainer).validate_state_dict(state)
        assert (
            trainer.requests
            == trainer.updates
            == trainer.rollout.calls
            == trainer.reward.calls
            == 0
        )


def test_local_recipe_real_subprocess_resume(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    trainer = trainer_fixture()
    policy = trainer.backend.model
    bundle = dict(
        config=policy.config.to_dict(),
        talker_state=policy.talker.state_dict(),
        reference_state=trainer.rollout.reference.talker.state_dict(),
        prompts=[
            dict(
                identity=p.identity,
                payload=p.payload,
                source_sha256=p.source_sha256,
                target_codes=p.target_codes,
                target_eos=p.target_eos,
            )
            for p in trainer.rollout.prompts
        ],
        upstream_contract=dict(
            max_frames=2,
            max_prompt_tokens=1024,
            speaker="ethan",
            chat_template_sha256=trainer.contract.chat_template_sha256,
            processor_sha256=trainer.contract.processor_sha256,
            position_config_sha256=trainer.contract.position_config_sha256,
        ),
    )
    source = tmp_path / "inputs.pt"
    torch.save(bundle, source)
    root = Path(__file__).resolve().parents[1]
    script = root / "examples/omni_rl/codec-policy/script.py"
    environment = dict(
        os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"
    )

    def run(name, stop, resume=None):
        output = tmp_path / name
        command = [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output",
            str(output),
            "--max-updates",
            "3",
            "--stop-after-requests",
            str(stop),
            "--device",
            "cpu",
        ]
        if resume is not None:
            command += ["--resume", str(resume)]
        result = subprocess.run(
            command,
            env=environment,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return output, json.loads(result.stdout.splitlines()[-1])

    first, report = run("first.pt", 1)
    assert report["updates"] == 1 and report["reports"][0]["gradient_norm"] > 0
    resumed, final = run("resumed.pt", 2, first)
    uninterrupted, direct = run("direct.pt", 2)
    assert final["updates"] == direct["updates"] == 2
    left = torch.load(resumed, weights_only=True)
    right = torch.load(uninterrupted, weights_only=True)
    for key in ("model", "optimizer", "scheduler", "data", "task", "rng", "modes"):
        exact(left[key], right[key])


@pytest.mark.parametrize("damage", ["nonfinite_gradient", "missing_used_gradient"])
def test_codec_rejects_invalid_gradient_before_update(damage):
    policy, _, _ = policy_and_prompt()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    backend = codec.CodecPolicyAdapter(policy, optimizer)
    backend._residual_used = True
    before = codec._tensor_identity(policy.state_dict())
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    parameter = next(policy.parameters())
    if damage == "nonfinite_gradient":
        parameter.grad.flatten()[0] = float("nan")
    else:
        parameter.grad = None
    with pytest.raises(ValueError):
        backend.update(torch.tensor(33), 1.0)
    assert codec._tensor_identity(policy.state_dict()) == before
    assert not optimizer.state


def test_resume_rejects_changed_objective_implementation(tmp_path, monkeypatch):
    trainer = trainer_fixture()
    trainer.train_step()
    trainer.backend.save(tmp_path / "checkpoint", trainer)
    fresh = trainer_fixture()
    before = codec._tensor_identity(fresh.backend.model.state_dict())
    monkeypatch.setattr(omni_rl, "grpo_loss_sum", lambda *args: None)
    with pytest.raises(ValueError, match="preflight"):
        fresh.backend.load(tmp_path / "checkpoint", fresh)
    assert codec._tensor_identity(fresh.backend.model.state_dict()) == before
    assert fresh.requests == fresh.updates == 0


@pytest.mark.parametrize("role", ["actor", "reference"])
def test_training_storage_must_not_alias_rollout_or_reference(role):
    trainer = trainer_fixture()
    actor = trainer.backend.model if role == "actor" else copy.deepcopy(trainer.backend.model)
    reference = (
        trainer.backend.model if role == "reference" else copy.deepcopy(trainer.backend.model)
    )
    rollout = codec.LocalCodecRollout(actor, reference, trainer.rollout.prompts)
    with pytest.raises(ValueError, match="independent parameter storage"):
        codec.create_trainer(
            trainer.cfg,
            contract=trainer.contract,
            backend=trainer.backend,
            rollout=rollout,
            reward=trainer.reward,
        )
    assert not trainer.backend.optimizer.state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_frozen_feature_dtype_matches_explicit_projection_cast(dtype):
    policy, _, original = policy_and_prompt()
    prompt = dict(original)
    keys = ("thinker_embeddings", "thinker_hidden", "special_embeddings")
    for key in keys:
        prompt[key] = prompt[key].to(dtype)
    before = codec._tensor_identity(prompt)
    explicit = {key: value.float() if key in keys else value for key, value in prompt.items()}
    reference = copy.deepcopy(policy)
    actions, mask = codec.action_layout(
        torch.arange(32).reshape(1, 2, 16) % 23,
        torch.tensor([2]),
        torch.tensor([False]),
        max_frames=2,
        vocab=23,
    )
    context = dict(lengths=torch.tensor([2]), reached_eos=torch.tensor([False]))
    actual = policy(actions, **context, **prompt)
    expected = reference(actions, **context, **explicit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for result in (actual, expected):
        (-result.gather(-1, actions[..., None]).squeeze(-1)[mask].sum()).backward()
    for name, parameter in policy.named_parameters():
        other = dict(reference.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)
    assert codec._tensor_identity(prompt) == before
    assert all(prompt[key].dtype == dtype for key in keys)


def test_codec_rejects_stateless_reward_protocol_before_training():
    class StatelessReward:
        def score(self, request, batch):
            return torch.zeros(batch.input_ids.shape[0])

        def state_dict(self):
            return {}

        def validate_state_dict(self, state):
            assert state == {}

        def load_state_dict(self, state):
            self.validate_state_dict(state)

    trainer = trainer_fixture()
    before = codec._tensor_identity(trainer.backend.model.state_dict())
    for entry in (codec.create_trainer, codec.launch):
        with pytest.raises(TypeError, match="ExactCodecReward"):
            entry(
                trainer.cfg,
                contract=trainer.contract,
                backend=trainer.backend,
                rollout=trainer.rollout,
                reward=StatelessReward(),
            )
    assert not trainer.backend.optimizer.state
    assert codec._tensor_identity(trainer.backend.model.state_dict()) == before
