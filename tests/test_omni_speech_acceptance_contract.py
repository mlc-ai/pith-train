"""CPU contracts for the full speech driver; none establish CUDA acceptance."""

import ast
import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.acceptance import omni_speech_chain as gate


def test_named_gradients_include_exact_zeros_and_tiny_nonzero_failures():
    left = {name: nn.Parameter(torch.ones(2)) for name in ("zero", "tiny", "missing")}
    right = {name: nn.Parameter(torch.ones(2)) for name in left}
    left["zero"].grad = right["zero"].grad = torch.zeros(2)
    left["tiny"].grad, right["tiny"].grad = torch.full((2,), 1e-12), torch.full((2,), -1e-12)
    report, bad = gate.compare_named_gradients(left, right)
    assert set(report) == set(left)
    assert report["zero"]["nse"] == 0
    assert bad == ["tiny", "missing"]
    assert report["tiny"]["nse"] == 2


def test_missing_gradient_identity_fails_instead_of_intersection():
    with pytest.raises(AssertionError, match="identities"):
        gate.compare_named_gradients({"a": nn.Parameter(torch.ones(1))}, {})


@pytest.mark.parametrize("change", ("value", "dtype", "shape", "keys", "type"))
def test_exact_checkpoint_comparison_rejects_each_difference(change):
    a = {"model": torch.tensor([1.0]), "task": [2, 2]}
    b = copy.deepcopy(a)
    if change == "value":
        b["model"] += 1e-6
    elif change == "dtype":
        b["model"] = b["model"].double()
    elif change == "shape":
        b["model"] = b["model"].view(1, 1)
    elif change == "keys":
        b["other"] = None
    else:
        b["task"] = tuple(b["task"])
    with pytest.raises(AssertionError):
        gate.exact(a, b)


def test_parameter_counter_requires_every_active_parameter():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (1, 2, 3, 4):
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert set(gate.parameter_steps(model, optimizer, step).values()) == {step}
    optimizer.state[model.bias]["step"] -= 1
    with pytest.raises(AssertionError, match="bias"):
        gate.parameter_steps(model, optimizer, 4)


def test_data_restore_replays_exact_input_hashes_and_all_cursors():
    samples = [{"x": torch.tensor(0.25), "codes": torch.arange(16).view(1, 1, 16)}]
    data = gate.FixedData(samples, "sha")
    for step in range(2):
        data.get_batch(step, torch.device("cpu"))
        data.commit_update(scan_index=step, update_index=step)
    state = data.state_dict()
    next_batch = data.get_batch(2, torch.device("cpu"))
    fresh = gate.FixedData(samples, "sha")
    fresh.load_state_dict(state)
    gate.exact(
        gate.tensor_tree(next_batch), gate.tensor_tree(fresh.get_batch(2, torch.device("cpu")))
    )
    assert fresh.observed[2] == data.observed[2]
    for field, value in (
        ("completed_updates", 1),
        ("fingerprint", "other"),
        ("scanned_batches", True),
    ):
        wrong = dict(state, **{field: value})
        with pytest.raises(ValueError):
            fresh.validate_state_dict(wrong)


class PrecisionFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_omni_moe")
        self.thinker = nn.Linear(3, 3)
        self.talker = nn.Linear(3, 3)
        self.code2wav = nn.Linear(3, 1)
        self.register_buffer("rotary", torch.tensor(1.234567, dtype=torch.float32))

    def forward(self, x):
        assert self.rotary.dtype == torch.float32
        assert self.thinker.weight.dtype == torch.bfloat16
        return self.talker(self.thinker(x))


def test_compute_cast_keeps_real_fp32_optimizer_leaves_buffers_and_gradients():
    original = PrecisionFixture()
    wrapped = gate.ComputeModel(original)
    value = torch.ones(1, 3, dtype=torch.bfloat16, requires_grad=True)
    out = wrapped(value)
    wave = wrapped.decode_training_codes(torch.ones(1, 3, dtype=torch.bfloat16))
    (out.float().square().sum() + wave.float().square().sum()).backward()
    assert all(p.dtype == p.grad.dtype == torch.float32 for p in wrapped.parameters())
    assert wrapped.model.rotary.dtype == torch.float32
    assert value.grad is not None and value.grad.dtype == torch.bfloat16
    assert {id(p) for p in wrapped.parameters()} == {id(p) for p in original.parameters()}


def test_fixture_adds_exactly_one_official_header_and_no_target_shift_twice():
    config = SimpleNamespace(im_start_token_id=100, user_token_id=101, assistant_token_id=102)
    batch = SimpleNamespace(
        model_inputs={
            "input_ids": torch.tensor([[3, 4]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
    )
    sample = gate.make_sample(batch, config, 0)
    assert sample["input_ids"].tolist() == [[100, 101, 198, 3, 4, 100, 102, 198, 40, 41, 42, 43]]
    assert sample["prompt_lengths"] == [8]
    assert sample["labels"].tolist() == [[-100] * 7 + [40, 41, 42, 43, -100]]
    assert sample["codes"].shape == (1, 4, 16)
    assert torch.equal(sample["codes"], gate.make_sample(batch, config, 0)["codes"])
    batch.model_inputs["input_ids"][0, 0] = 100
    with pytest.raises(gate.PreconditionError, match="nested"):
        gate.make_sample(batch, config, 0)


def test_contract_source_retains_all_gates_and_no_gpu_execution_in_cpu_tests():
    source = gate.Path(gate.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = {item.name: item for item in module.body if isinstance(item, ast.FunctionDef)}
    checkpoint = ast.unparse(functions["checkpoint_gate"])
    assert "producer['pid'] == os.getpid()" in checkpoint
    assert "task.load_checkpoint" in checkpoint and "task.save_checkpoint" in checkpoint
    assert "task.launch" in checkpoint and "range(start, 5)" in checkpoint
    assert "parameter_steps(native, optimizer, 2)" in checkpoint
    assert gate.GRADIENT_NSE == 1e-2 and gate.OUTPUT_NSE == 1e-3 and gate.STEPS == 4
    assert gate.MEDIA == ("text", "image", "audio", "video", "mixed")
    assert not torch.cuda.is_initialized()


def test_prepared_identity_rejects_revision_before_loading_tensor_payload(tmp_path):
    gate.write_json(
        tmp_path / "fixture.json",
        {
            "schema": gate.SCHEMA,
            "source": {"commit": "a" * 40},
            "released_config_revision": "wrong",
            "released_config_sha256": gate.RELEASED_CONFIG_SHA256,
        },
    )
    with pytest.raises(gate.PreconditionError, match="released"):
        gate.load_fixture(tmp_path, "a" * 40)


def test_report_json_is_strict_utf8_and_roundtrips(tmp_path):
    value = {"status": "NOT_RUN_PRECONDITION", "reason": "源码前提", "nse": 0.0}
    path = tmp_path / "result.json"
    gate.write_json(path, value)
    assert json.loads(path.read_bytes().decode("utf-8", errors="strict")) == value
    with pytest.raises(ValueError):
        gate.write_json(path, {"nse": float("nan")})


def test_dependency_lock_rejects_package_or_runtime_drift(tmp_path):
    observed = dict(
        torch="2.13.0+cu130",
        cuda="13.0",
        transformers="5.17.0",
        python="3.12.1",
        packages=["torch==2.13.0+cu130", "transformers==5.17.0"],
    )
    path = tmp_path / "runtime-lock.json"
    gate.write_json(path, observed)
    assert gate.verify_runtime_lock(path, observed) == gate.sha(path)
    for field, value in (("transformers", "5.18.0"), ("packages", []), ("torch", "other")):
        with pytest.raises(gate.PreconditionError, match="differ"):
            gate.verify_runtime_lock(path, dict(observed, **{field: value}))
    with pytest.raises(gate.PreconditionError, match="exact"):
        gate.verify_runtime_lock(path, {"torch": observed["torch"]})


@pytest.mark.parametrize("phase", ("source_precondition", "numerical", "checkpoint_training"))
def test_failure_report_preserves_execution_phase_without_claiming_pass(phase):
    report = gate.failure_report(phase, AssertionError("original failure"))
    assert report["failure_phase"] == phase and report["error"] == "original failure"
    assert report["status"] == (
        "NOT_RUN_PRECONDITION" if phase == "source_precondition" else "FAIL"
    )
    assert report["gpu_acceptance"] == (
        "NOT_RUN" if phase == "source_precondition" else "FAIL_THIS_GATE"
    )


def test_canonical_map_preserves_complete_component_boundaries():
    assert (
        gate.canonical("thinker.layers.0.mlp.experts.gate_up_proj")
        == "thinker.model.layers.0.mlp.experts.gate_up_proj"
    )
    for name in (
        "thinker.visual.x",
        "thinker.audio_tower.x",
        "thinker.lm_head.weight",
        "talker.model.codec_embedding.weight",
        "talker.code_predictor.lm_head.0.weight",
        "code2wav.decoder.0.x",
    ):
        assert gate.canonical(name) == name


def test_actual_hf_reference_cpu_forward_backward_and_sixteen_heads():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    # Exercise the raw released schema used by GPU preparation, including its
    # legacy rope_scaling fields; a default config hides migration conflicts.
    released_path = gate.REPO / "tests/configs/qwen3_omni_full/released-config.json"
    assert gate.sha(released_path) == gate.RELEASED_CONFIG_SHA256
    released = json.loads(released_path.read_text())
    cfg = Qwen3OmniMoeConfig(**copy.deepcopy(released))
    thinker = cfg.thinker_config
    thinker.vision_start_token_id = 210
    thinker.position_id_per_seconds = 13
    text = thinker.text_config
    text.vocab_size, text.hidden_size, text.intermediate_size = 256, 16, 24
    text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads = 2, 2, 1
    text.head_dim, text.moe_intermediate_size, text.num_experts, text.num_experts_per_tok = (
        8,
        12,
        4,
        2,
    )
    text.rope_parameters.update(mrope_section=[2, 1, 1])
    text.layer_types = ["full_attention"] * 2
    vision = thinker.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size, vision.num_heads = 2, 16, 24, 2
    vision.out_hidden_size, vision.deepstack_visual_indexes = 16, [0, 1]
    audio = thinker.audio_config
    audio.encoder_layers, audio.d_model, audio.encoder_attention_heads, audio.encoder_ffn_dim = (
        1,
        16,
        2,
        24,
    )
    audio.downsample_hidden_size, audio.output_dim = 8, 16
    cfg.im_start_token_id, cfg.user_token_id, cfg.assistant_token_id = 100, 101, 102
    cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id = 200, 201, 202
    for name in (
        "im_start_token_id",
        "user_token_id",
        "assistant_token_id",
        "tts_bos_token_id",
        "tts_eos_token_id",
        "tts_pad_token_id",
    ):
        released[name] = getattr(cfg, name)
    actual = gate.tiny_config(thinker, released)
    # Preparation serializes and the GPU process reloads this exact config.
    actual = Qwen3OmniMoeConfig(**json.loads(json.dumps(actual.to_dict())))
    assert actual.talker_config.text_config.rope_parameters["mrope_section"] == [4, 2, 2]
    from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration

    Qwen3OmniMoeTalkerForConditionalGeneration(actual.talker_config)
    model = gate.HFChain(actual)
    sample = gate.make_sample(
        SimpleNamespace(
            model_inputs={
                "input_ids": torch.tensor([[3, 4]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }
        ),
        actual,
        0,
    )
    positional, kwargs = gate.speech_args(sample)
    output = model(*positional, **kwargs)
    assert output.first_logits.shape == (1, 5, 41)
    assert output.residual_logits.shape == (1, 4, 15, 23)
    codec, _, text_loss, _ = gate.loss_terms(output, sample["labels"])
    waveform = model.code2wav(sample["codes"].transpose(1, 2))
    (codec + text_loss + gate.waveform_loss(waveform, sample).loss_sum).backward()
    for head in model.talker.code_predictor.lm_head:
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert all(p.grad is not None for p in model.code2wav.parameters())
    assert not torch.cuda.is_initialized()


def cpu_restore_worker(root, mode):
    """Actual separate process around the production checkpoint API, CPU toy only."""
    from tests.test_omni_speech_task import ToySpeech, sample

    root = gate.Path(root)
    args = SimpleNamespace(
        mode="restore",
        training_mode=mode,
        output=root / "restore",
        producer=root / "produce",
        prepared=root / "prepared",
        source={"commit": "a" * 40},
        device=torch.device("cpu"),
    )
    args.output.mkdir()
    result = gate.checkpoint_gate(args, [sample()], ToySpeech(), {"samples_sha256": "cpu-protocol"})
    gate.write_json(
        args.output / "cpu-result.json",
        dict(
            result,
            pid=os.getpid(),
            numerical_scope="CPU_TOY_TASK_PROTOCOL_ONLY",
            cuda_initialized=torch.cuda.is_initialized(),
        ),
    )


@pytest.mark.parametrize("mode", ("text_and_codec", "waveform"))
def test_actual_driver_production_checkpoint_fresh_process_cpu_protocol(mode, tmp_path):
    from tests.test_omni_speech_task import ToySpeech, sample

    prepared, produce = tmp_path / "prepared", tmp_path / "produce"
    prepared.mkdir()
    produce.mkdir()
    gate.write_json(
        prepared / "fixture.json", {"kind": "CPU protocol fixture; no native CUDA claim"}
    )
    args = SimpleNamespace(
        mode="produce",
        training_mode=mode,
        output=produce,
        prepared=prepared,
        source={"commit": "a" * 40},
        device=torch.device("cpu"),
    )
    result = gate.checkpoint_gate(args, [sample()], ToySpeech(), {"samples_sha256": "cpu-protocol"})
    assert [r["step"] for r in result["steps"]] == [1, 2, 3, 4]
    gate.write_json(produce / "result.json", dict(result, pid=os.getpid(), source=args.source))
    command = [
        sys.executable,
        "-c",
        "from tests.test_omni_speech_acceptance_contract import cpu_restore_worker; import sys; cpu_restore_worker(sys.argv[1],sys.argv[2])",
        str(tmp_path),
        mode,
    ]
    completed = subprocess.run(
        command,
        cwd=gate.REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    restored = json.loads((tmp_path / "restore" / "cpu-result.json").read_text())
    assert restored["pid"] != os.getpid()
    assert restored["status"] == "PASS" and restored["restored_boundary"] == "EXACT"
    assert not restored["cuda_initialized"]


def test_update_delta_gate_catches_opposite_tiny_updates_hidden_by_large_weights():
    before = {"p": torch.tensor([100.0], dtype=torch.float64), "zero": torch.zeros(1)}
    actual = {"p": before["p"] + 1e-6, "zero": torch.zeros(1)}
    expected = {"p": before["p"] - 1e-6, "zero": torch.zeros(1)}
    assert gate.nse(actual["p"], expected["p"]) < 1e-12
    rows, failures = gate.compare_update_deltas(actual, expected, before, before)
    assert rows["p"]["nse"] == 2
    assert failures == ["p"]
    assert rows["zero"]["status"] == "PASS"


def test_update_delta_gate_keeps_all_rows_when_one_parameter_is_nonfinite():
    before = {name: torch.zeros(1) for name in ("bad", "good")}
    actual = {"bad": torch.tensor([float("nan")]), "good": torch.ones(1)}
    expected = {name: torch.ones(1) for name in before}
    rows, failures = gate.compare_update_deltas(actual, expected, before, before)
    assert set(rows) == set(before) and failures == ["bad"]
    assert rows["good"]["status"] == "PASS"


def test_model_setup_failure_is_attempted_failure_not_unrun_precondition():
    report = gate.failure_report("model_setup", RuntimeError("out of memory"))
    assert report["status"] == "FAIL" and report["gpu_acceptance"] == "FAIL_THIS_GATE"


def test_hf_chain_audio_vjp_uses_official_single_codebook_reduction(request):
    """CPU arithmetic regression, not native/CUDA numerical acceptance.

    A second forward replaces only the teacher-forced codec prefix with the
    official concatenation/single-reduction expression, independently using
    HF's tables and assistant-conditioning helper. All audio gradients must
    remain exact, including the near-null key-projection biases.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration as HFComposite,
    )

    from tests.test_omni_speech_hidden_states import config as text_config
    from tests.test_qwen3_omni_multimodal import context

    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_rng_state(rng))
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    torch.set_num_threads(1)
    torch.manual_seed(gate.SEED)
    released = json.loads(
        (gate.REPO / "tests/configs/qwen3_omni_full/released-config.json").read_text()
    )
    thinker, inputs = context("audio")
    thinker.text_config = text_config()
    thinker.text_config.vocab_size = 256
    released.update(
        im_start_token_id=100,
        user_token_id=101,
        assistant_token_id=102,
        tts_bos_token_id=200,
        tts_eos_token_id=201,
        tts_pad_token_id=202,
    )
    cfg = gate.tiny_config(thinker, released)
    model = gate.ComputeModel(gate.HFChain(cfg)).eval()
    sample = gate.make_sample(SimpleNamespace(model_inputs=inputs), cfg, 2)
    results = []
    for replace_with_official in (False, True):
        model.zero_grad(set_to_none=True)
        observed = {}

        def audio_hook(module, args, output):
            output.last_hidden_state.retain_grad()
            observed["audio"] = output.last_hidden_state

        def thinker_hook(module, args, output):
            observed["thinker"] = output

        def trunk_hook(module, args, kwargs):
            # Compute the official per-frame expression separately from the
            # HFChain implementation, under the same BF16 functional weights.
            codes = sample["codes"]
            batch, frames, groups = codes.shape
            flat = codes.reshape(batch * frames, groups)
            talker = model.talker
            embedded = [talker.get_input_embeddings()(flat[:, :1])]
            embedded += [
                table(flat[:, i + 1 : i + 2])
                for i, table in enumerate(talker.code_predictor.get_input_embeddings())
            ]
            previous = torch.cat(embedded, dim=1).sum(1).reshape(batch, frames, -1)
            specials = sample["input_ids"].new_tensor(
                [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]]
            )
            bos, eos, pad = talker.text_projection(
                model.thinker.get_input_embeddings()(specials)
            ).chunk(3, 1)
            _, _, tail = HFComposite._get_talker_assistant_parts(
                model.model,
                sample["prompt_lengths"][0] - 3,
                sample["input_ids"].shape[1],
                cfg.talker_config.speaker_id["ethan"],
                observed["thinker"].hidden_states[0],
                pad,
                bos,
                eos,
            )
            expected = previous + torch.cat((tail, pad.expand(-1, frames, -1)), 1)[:, :frames]
            actual = kwargs["inputs_embeds"][:, -frames:]
            observed["prefix_matches"] = torch.equal(actual, expected)
            if replace_with_official:
                kwargs = dict(
                    kwargs,
                    inputs_embeds=torch.cat((kwargs["inputs_embeds"][:, :-frames], expected), 1),
                )
                return args, kwargs

        hooks = [
            model.thinker.audio_tower.register_forward_hook(audio_hook),
            model.thinker.register_forward_hook(thinker_hook),
            model.talker.model.register_forward_pre_hook(trunk_hook, with_kwargs=True),
        ]
        try:
            batch = gate.move(sample, torch.device("cpu"), gradients=True)
            positional, kwargs = gate.speech_args(batch)
            output = model(*positional, **kwargs)
            value, count, text_value, text_count = gate.loss_terms(output, batch["labels"])
            (value / count + text_value / text_count).backward()
            assert observed["prefix_matches"], (
                "HFChain codebook prefix differs from official BF16 reduction"
            )
            results.append(
                dict(
                    outputs=(output.first_logits.detach(), output.residual_logits.detach()),
                    cotangent=observed["audio"].grad.clone(),
                    gradients={
                        name: p.grad.clone()
                        for name, p in model.thinker.audio_tower.named_parameters()
                    },
                )
            )
        finally:
            for hook in hooks:
                hook.remove()
    gate.exact(results[0], results[1])
    assert all(torch.isfinite(value).all() for value in results[0]["gradients"].values())
    assert all(f"layers.{i}.self_attn.k_proj.bias" in results[0]["gradients"] for i in range(2))
    assert not torch.cuda.is_initialized()
