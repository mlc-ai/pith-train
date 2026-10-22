"""Generation-gate contracts and real CPU HF codec controls, not native GPU proof."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.acceptance import omni_speech_generation as gate


class MethodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("rotary", torch.tensor([1.0, 0.1]))

    def generate(self, value):
        assert self.linear.weight.dtype == torch.bfloat16
        assert self.rotary.dtype == torch.float32
        return self.linear(value)


def test_functional_generate_dispatch_preserves_master_and_rotary_buffers():
    model = MethodModel()
    before = gate.state_hash(model)
    ids = [id(p) for p in model.parameters()]
    result = gate.compute(
        model, lambda module, value: module.generate(value), torch.ones(1, 2, dtype=torch.bfloat16)
    )
    assert result.dtype == torch.bfloat16
    assert gate.state_hash(model) == before
    assert [id(p) for p in model.parameters()] == ids
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_control_restores_rng_modes_and_mutable_cache_even_after_exception():
    from pithtrain.tasks.omni_speech import _rng_state

    model = MethodModel()
    model.linear.eval()
    model.rope_deltas = torch.tensor([3])
    rng, modes = _rng_state(), [m.training for m in model.modules()]
    with pytest.raises(RuntimeError, match="control stopped"):
        with gate.preserve_generation_state(model) as proof:
            model.rope_deltas = torch.tensor([9])
            torch.rand(13)
            raise RuntimeError("control stopped")
    gate.chain.exact(_rng_state(), rng)
    assert [m.training for m in model.modules()] == modes
    assert model.rope_deltas.tolist() == [3]
    assert all(proof.values())


def result():
    speech = SimpleNamespace(
        codes=torch.zeros(1, 2, 16, dtype=torch.long),
        code_lengths=torch.tensor([2]),
        reached_eos=torch.tensor([False]),
        waveforms=[torch.ones(1, 10)],
    )
    return SimpleNamespace(
        sequences=torch.tensor([[1, 2, 3, 4]]),
        text_reached_eos=False,
        speech=speech,
        logits=[torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])],
    )


@pytest.mark.parametrize(
    "damage", ["text", "code", "length", "eos", "wave", "nan", "logits", "missing"]
)
def test_comparison_rejects_each_real_output_axis(damage):
    actual, expected = result(), result()
    rows = [value.clone() for value in expected.logits]
    if damage == "text":
        actual.sequences[0, -1] = 7
    elif damage == "code":
        actual.speech.codes[0, 0, 15] = 1
    elif damage == "length":
        actual.speech.code_lengths[0] = 1
    elif damage == "eos":
        actual.speech.reached_eos[0] = True
    elif damage == "wave":
        actual.speech.waveforms[0].zero_()
    elif damage == "nan":
        actual.speech.waveforms[0][0, 0] = float("nan")
    elif damage == "logits":
        rows[0].neg_()
    elif damage == "missing":
        actual.speech = None
    with pytest.raises(AssertionError):
        gate.compare_generation(actual, rows, expected, 2)


def codec_pair():
    from tests.test_omni_speech_integration import condition, pair
    from tests.test_qwen3_omni_code2wav import paired, tiny_config

    talker, hf_talker = pair()
    codec, hf_codec = paired(tiny_config(num_quantizers=16, codebook_size=23))

    class Components(nn.Module):
        def __init__(self, talker, codec):
            super().__init__()
            self.talker, self.code2wav = talker, codec
            self.config = SimpleNamespace(talker_config=talker.config)

    torch.manual_seed(71)
    prepared = condition(talker, 1)
    values = dict(
        inputs_embeds=prepared.inputs_embeds,
        talker_input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        trailing_text_hidden=prepared.trailing_text_hidden,
        tts_pad_embed=prepared.tts_pad_embed,
    )
    values = {
        name: value.detach().to(torch.bfloat16) if value.is_floating_point() else value
        for name, value in values.items()
    }
    return Components(talker, codec), Components(hf_talker, hf_codec), values


def test_actual_hf_cached_ragged_codec_and_waveform_control_cpu():
    torch.set_num_threads(1)
    native, hf, prepared = codec_pair()
    assert "forward" not in native.talker.__dict__
    with torch.no_grad(), gate.preserve_generation_state(native, hf) as preserved:
        report = gate.ragged_control(native, hf, prepared, {"position_kwargs": {}})
    assert "forward" not in native.talker.__dict__
    assert report["scope"] == "controlled_codec_eos_boundary"
    assert report["code_lengths"] == [0, 2]
    assert report["reached_eos"] == [True, True]
    assert report["waveforms"][0]["samples"] == 0
    assert report["waveforms"][1]["samples"] > 0
    assert all(wave["nse"] < 1e-3 for wave in report["waveforms"])
    assert all(preserved.values())


def test_unforced_generation_and_control_paths_are_separate_and_bounded():
    path = Path(gate.__file__)
    tree = ast.parse(path.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    native_source = ast.unparse(functions["native_generate"])
    assert "model.generate" in native_source and "original(*args, **kwargs)" in native_source
    assert "logits_processor" not in native_source and "controlled" not in native_source
    hf_source = ast.unparse(functions["hf_codec"])
    assert "model.talker.generate" in hf_source and "model.code2wav.chunked_decode" in hf_source
    validate = ast.unparse(functions["validate"])
    assert "for sample in samples" in validate and "set(cases) != set(MEDIA)" in validate
    assert "generated[1]" in validate
    official = ast.unparse(functions["hf_generate"])
    assert "model.generate(input_ids=" in official and "HFComposite" in official
    assert "hf_features" not in functions and "hf_condition" not in functions
    assert "require_nonempty_case" in validate and "multi_turn" in validate
    assert (gate.TEXT_TOKENS, gate.CODEC_FRAMES, gate.OUTPUT_NSE) == (4, 8, 1e-3)
    assert gate.MEDIA == ("text", "image", "audio", "video", "mixed")
    assert "source_identity(args.expected_commit)" in ast.unparse(functions["main"])
    assert "runtime_precondition(args)" in ast.unparse(functions["main"])


def test_summary_is_complete_and_compact_without_raw_tensor_payloads():
    actual = result()
    report = gate.compare_generation(actual, actual.logits, copy.deepcopy(actual), 2)
    assert report["status"] == "PASS" and report["code_groups"] == 16
    assert report["generated_tokens"] == [[3, 4]]
    import json

    encoded = json.dumps(report, allow_nan=False).encode()
    assert len(encoded) < 4096
    assert report["thinker_logits_nse"] == [0.0, 0.0]


@pytest.mark.parametrize("existing", [False, True])
def test_hf_position_metadata_restored_on_exception(existing):
    owner = SimpleNamespace()
    prior = SimpleNamespace(original=True)
    if existing:
        owner.audio_config = prior
    hf = SimpleNamespace(talker=SimpleNamespace(config=owner))
    config = SimpleNamespace(thinker_config=SimpleNamespace(audio_config={"official": 7}))
    with pytest.raises(RuntimeError, match="stop"):
        with gate.hf_position_metadata(hf, config):
            assert owner.audio_config == {"official": 7}
            assert owner.audio_config is not config.thinker_config.audio_config
            raise RuntimeError("stop")
    assert hasattr(owner, "audio_config") is existing
    if existing:
        assert owner.audio_config is prior


@pytest.mark.parametrize("fail", [False, True])
def test_native_logit_observer_restores_method_without_instance_override(fail):
    class Thinker(nn.Module):
        def forward_epilog(self, value):
            return value

    class ProtocolModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.thinker = Thinker()
            self.value = torch.ones(1, 2, 3)

        def generate(self, *args, **kwargs):
            observed = self.thinker.forward_epilog(self.value)
            assert observed is self.value
            if fail:
                raise RuntimeError("generation stopped")
            return observed

    model = ProtocolModel()
    batch = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "model_context": {},
        "position_kwargs": {},
    }
    if fail:
        with pytest.raises(RuntimeError, match="generation stopped"):
            gate.native_generate(model, batch)
    else:
        output, rows = gate.native_generate(model, batch)
        assert output is model.value
        assert torch.equal(rows[0], model.value[:, -1])
    assert "forward_epilog" not in model.thinker.__dict__


def test_preservation_removes_generation_cache_created_during_control():
    model = MethodModel()
    assert not hasattr(model, "rope_deltas")
    with gate.preserve_generation_state(model) as proof:
        model.rope_deltas = torch.tensor([3])
    assert not hasattr(model, "rope_deltas")
    assert all(proof.values())


@pytest.fixture(scope="module")
def official_full_cpu():
    """Actual TF5.17 composite, official revision, reduced dimensions only."""
    import json

    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTextConfig,
    )

    torch.set_num_threads(1)
    top = AutoConfig.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", revision=gate.chain.REVISION, local_files_only=True
    )
    thinker = copy.deepcopy(top.thinker_config)
    path = Path(__file__).parent / "configs/qwen3_omni_text/tiny.json"
    thinker.text_config = Qwen3OmniMoeTextConfig(**json.loads(path.read_text()))
    v, a = thinker.vision_config, thinker.audio_config
    v.depth, v.hidden_size, v.num_heads, v.intermediate_size = 3, 64, 4, 96
    v.out_hidden_size, v.deepstack_visual_indexes = thinker.text_config.hidden_size, [0, 2]
    a.encoder_layers, a.d_model, a.encoder_attention_heads, a.encoder_ffn_dim = 2, 64, 4, 96
    a.downsample_hidden_size, a.output_dim = 8, thinker.text_config.hidden_size
    config = gate.chain.tiny_config(thinker, top.to_dict())
    for cfg in (
        config.thinker_config,
        config.talker_config,
        config.code2wav_config,
        config.thinker_config.text_config,
        config.talker_config.text_config,
        config.talker_config.code_predictor_config,
    ):
        cfg._attn_implementation = "eager"
        cfg._experts_implementation = "eager"
    torch.manual_seed(gate.chain.SEED)
    # Use the same fully initialized official components as the GPU fixture.
    # TF5.17 composite.post_init leaves raw Talker expert storage uninitialized;
    # allocator contents can otherwise make isolated runs pass and full suites NaN.
    model = gate.official_composite(gate.chain.HFChain(config).float()).eval()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    ids = torch.tensor(
        [
            [
                config.im_start_token_id,
                config.user_token_id,
                198,
                40,
                41,
                config.im_end_token_id,
                198,
                config.im_start_token_id,
                config.assistant_token_id,
                198,
            ]
        ]
    )
    batch = dict(
        input_ids=ids,
        model_context=dict(input_ids=ids, attention_mask=torch.ones_like(ids)),
        prompt_lengths=[ids.shape[1]],
        position_kwargs={},
    )
    return model, batch


def cpu_generation_pair(official_full_cpu, monkeypatch):
    """Real generation/Talker/decoder; HF Thinker adapter isolates CPU control flow.

    This deliberately does not claim native CUDA Thinker numerical acceptance.
    """
    import pithtrain.models.qwen3_omni_speech as speech
    from pithtrain.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
    from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration

    hf, batch = copy.deepcopy(official_full_cpu)

    class CPUThinker(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.reference = model

        @property
        def embed_tokens(self):
            return self.reference.get_input_embeddings()

        def forward_epilog(self, logits):
            return logits

    def features(thinker, ids, context, layer):
        output = thinker.reference(
            **dict(context, input_ids=ids, output_hidden_states=True, use_cache=False)
        )
        return speech.ThinkerSpeechOutput(
            thinker.forward_epilog(output.logits),
            output.hidden_states[0],
            output.hidden_states[layer],
        )

    monkeypatch.setattr(speech, "thinker_speech_forward", features)
    talker = Qwen3OmniMoeTalkerForConditionalGeneration(
        copy.deepcopy(hf.config.talker_config)
    ).float()
    decoder = Qwen3OmniMoeCode2Wav(copy.deepcopy(hf.config.code2wav_config)).float()
    talker.load_state_dict(hf.talker.state_dict(), strict=True)
    decoder.load_state_dict(hf.code2wav.state_dict(), strict=True)
    native = speech.Qwen3OmniMoeSpeechModel(
        hf.config, CPUThinker(copy.deepcopy(hf.thinker)), talker, decoder
    )
    return native, hf, batch


@pytest.mark.parametrize("variant", ["cap", "multiturn", "delayed_eos"])
def test_full_official_composite_matches_native_generation_protocol_cpu(
    official_full_cpu, monkeypatch, variant
):
    native, hf, batch = cpu_generation_pair(official_full_cpu, monkeypatch)
    options = dict(text_tokens=4, codec_frames=3)
    if variant == "multiturn":
        batch = gate.multi_turn(batch, hf.config)
    with (
        torch.no_grad(),
        gate.preserve_generation_state(native, hf) as preserved,
        gate.hf_position_metadata(hf, hf.config),
    ):
        if variant == "delayed_eos":
            preliminary = gate.compute(hf, gate.hf_generate, batch, **options)
            tokens = preliminary.sequences[0, batch["input_ids"].shape[1] :]
            assert tokens[0] != tokens[1]
            options["eos_id"] = int(tokens[1])
        expected = gate.compute(hf, gate.hf_generate, batch, **options)
        actual, rows = gate.compute(native, gate.native_generate, batch, **options)
        report = gate.compare_generation(actual, rows, expected, batch["input_ids"].shape[1])
        gate.require_nonempty_case(report, variant)
        assert (
            expected.reference_entrypoint
            == "transformers.Qwen3OmniMoeForConditionalGeneration.generate"
        )
        assert len(rows) == len(expected.logits)
        if variant == "cap":
            assert len(report["generated_tokens"][0]) == 4 and not report["text_reached_eos"]
        if variant == "delayed_eos":
            assert report["text_reached_eos"] and len(report["generated_tokens"][0]) == 2
    assert all(preserved.values())
    assert "generate" not in hf.thinker.__dict__ and "generate" not in hf.talker.__dict__
    assert "chunked_decode" not in hf.code2wav.__dict__


def test_every_media_requires_nonempty_waveform():
    empty = {"waveforms": []}
    for media in gate.MEDIA:
        with pytest.raises(AssertionError, match="every natural media"):
            gate.require_nonempty_case(empty, media)


def test_actual_code2wav_crosses_official_300_frame_chunk_cpu():
    native, hf, _ = codec_pair()
    speech = SimpleNamespace(
        codes=torch.arange(48).reshape(1, 3, 16) % 23, code_lengths=torch.tensor([3])
    )
    with torch.no_grad(), gate.preserve_generation_state(native, hf) as preserved:
        report = gate.chunk_boundary(native, hf, speech)
    assert report["frames"] == 301 and report["chunk_size"] == 300 and report["chunks"] == 2
    assert report["code_groups"] == 16 and report["finite"] and report["nse"] < 1e-3
    assert report["scope"] == "decoder_component_chunk_boundary"
    assert "not_natural_301_frame_generation" in report["source"]
    assert all(preserved.values())


@pytest.mark.parametrize("failure", [False, True])
def test_official_observer_returns_original_and_restores_on_error(failure):
    model = MethodModel()
    seen = []
    original = model.generate

    def observer(args, kwargs, result):
        seen.append(result)
        if failure:
            raise RuntimeError("observer stopped")

    with gate.observe_method(model, "generate", observer):
        if failure:
            with pytest.raises(RuntimeError, match="observer stopped"):
                gate.compute(
                    model, lambda module: module.generate(torch.ones(1, 2, dtype=torch.bfloat16))
                )
        else:
            result = gate.compute(
                model, lambda module: module.generate(torch.ones(1, 2, dtype=torch.bfloat16))
            )
            assert result is seen[0]
    assert model.generate.__func__ is original.__func__ and "generate" not in model.__dict__


def test_bf16_codec_reduction_and_all_embedding_gradients_match_official_expression():
    from pithtrain.models.qwen3_omni_speech import codec_embeddings
    from tests.test_omni_speech_integration import pair

    native, hf = pair()
    codes = torch.arange(2 * 4 * 16).reshape(2, 4, 16) % 23

    def official_reduction(model, values):
        # HF prepare_inputs_for_generation concatenates all16 embedding rows
        # and sums once; use its actual independently paired embedding tables.
        batch, frames, groups = values.shape
        flat = values.reshape(batch * frames, groups)
        embedded = [model.get_input_embeddings()(flat[:, :1])]
        embedded += [
            table(flat[:, index + 1 : index + 2])
            for index, table in enumerate(model.code_predictor.get_input_embeddings())
        ]
        return torch.cat(embedded, 1).sum(1).reshape(batch, frames, -1)

    actual = gate.compute(native, codec_embeddings, codes)
    expected = gate.compute(hf, official_reduction, codes)
    assert torch.equal(actual, expected)
    for value in (actual, expected):
        value.float().square().sum().backward()
    used = []
    for name, parameter in native.named_parameters():
        reference = hf.get_parameter(name)
        if parameter.grad is None:
            assert reference.grad is None, name
        else:
            assert torch.equal(parameter.grad, reference.grad), name
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            used.append(name)
    assert len(used) == 16


def test_bf16_teacher_force_recurrence_matches_hf_all_logits_and_gradients():
    from pithtrain.models.qwen3_omni_speech import speech_teacher_force
    from tests.test_omni_speech_integration import condition, independent_hf, pair

    native, hf = pair()
    torch.manual_seed(33)
    codes = torch.randint(0, 23, (2, 4, 16))
    lengths = torch.tensor([4, 2])
    codes[1, 2:] = -100

    def actual_forward(model):
        torch.manual_seed(33)
        return speech_teacher_force(model, condition(model), codes, lengths)

    def reference_forward(model):
        torch.manual_seed(33)
        return independent_hf(model, condition(model), codes, lengths)

    # Build conditioning from the same FP32 random tensors; the public projection
    # inputs must match the BF16 parameter compute policy used by functional_call.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = gate.compute(native, actual_forward)
        first, residual = gate.compute(hf, reference_forward)
    for a, b in ((actual.first_logits, first), (actual.residual_logits, residual)):
        assert gate.chain.nse(a, b) < 1e-3
    for first_logits, residual_logits in (
        (actual.first_logits, actual.residual_logits),
        (first, residual),
    ):
        loss = torch.nn.functional.cross_entropy(
            first_logits.float().flatten(0, 1), actual.first_labels.flatten(), reduction="sum"
        )
        loss += torch.nn.functional.cross_entropy(
            residual_logits.float().flatten(0, 2), actual.residual_labels.flatten(), reduction="sum"
        )
        loss.backward()
    for name, parameter in native.named_parameters():
        reference = hf.get_parameter(name)
        assert parameter.grad is not None and reference.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and torch.isfinite(reference.grad).all(), name
        assert gate.chain.nse(parameter.grad, reference.grad) < 1e-2, name


def test_official_full_cpu_fixture_initializes_poisoned_expert_storage():
    policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    try:
        # Allocator-independent negative control: torch.empty returns NaN until
        # explicitly initialized, exposing the original four raw expert tensors.
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        model, _ = official_full_cpu.__wrapped__()
        experts = {
            name: parameter
            for name, parameter in model.named_parameters()
            if name.startswith("talker.") and ".mlp.experts." in name
        }
        assert len(experts) == 4
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in experts.values()
        )
        assert all(torch.isfinite(value).all() for value in model.parameters())
        # Repeat the same fixture builder with the allocator's poison disabled;
        # raw uninitialized memory contents must not affect any saved state.
        torch.utils.deterministic.fill_uninitialized_memory = False
        independently_initialized, _ = official_full_cpu.__wrapped__()
        assert gate.state_hash(model) == gate.state_hash(independently_initialized)
    finally:
        torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])
        torch.utils.deterministic.fill_uninitialized_memory = policy[2]
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)
