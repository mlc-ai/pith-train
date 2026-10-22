"""Native speech recurrence compared with independent HF model expressions."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration as HFComposite,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration as HFTalker,
)

from pithtrain.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
from pithtrain.models.qwen3_omni_speech import (
    TalkerCondition,
    generate_speech_from_condition,
    prepare_talker_condition,
    speech_teacher_force,
)
from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration


def config():
    text = dict(
        vocab_size=41,
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        attention_dropout=0.0,
        max_position_embeddings=256,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000,
            "mrope_section": [2, 1, 1],
            "interleaved": True,
        },
    )
    predictor = dict(
        vocab_size=23,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_code_groups=16,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000},
    )
    cfg = Qwen3OmniMoeTalkerConfig(
        text_config=text,
        code_predictor_config=predictor,
        thinker_hidden_size=20,
        num_code_groups=16,
        spatial_merge_size=2,
        codec_eos_token_id=31,
        codec_nothink_id=32,
        codec_think_bos_id=33,
        codec_think_eos_id=34,
        codec_pad_id=35,
        codec_bos_id=36,
        speaker_id={"ethan": 37},
    )
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.text_config._experts_implementation = "eager"
    cfg.code_predictor_config._attn_implementation = "eager"
    return cfg


def pair():
    torch.manual_seed(18)
    torch.set_num_threads(1)
    cfg = config()
    hf = HFTalker(copy.deepcopy(cfg)).float().eval()
    with torch.no_grad():
        for name, parameter in hf.named_parameters():
            if ".mlp.experts." in name:
                parameter.normal_(std=0.02)
    native = Qwen3OmniMoeTalkerForConditionalGeneration(copy.deepcopy(cfg)).float().eval()
    native.load_state_dict(hf.state_dict(), strict=True)
    return native, hf


def condition(model, batch=2):
    # Both projections are exercised, independently of the prompt parser.
    source = torch.randn(batch, 5, 20)
    embeddings = model.text_projection(source) + model.hidden_projection(source * 0.7)
    tail = model.text_projection(torch.randn(batch, 2, 20))
    pad = model.text_projection(torch.randn(batch, 1, 20))
    ids = torch.zeros(batch, 5, dtype=torch.long)
    return TalkerCondition(
        embeddings,
        ids,
        torch.ones_like(ids),
        torch.arange(5).view(1, 1, -1).expand(3, batch, -1),
        tail,
        pad,
    )


def independent_hf(model, prepared, codes, lengths):
    """Construct the HF autoregressive graph without the native recurrence helper."""
    batch, frames, groups = codes.shape
    valid = torch.arange(frames)[None] < lengths[:, None]
    safe = codes.masked_fill(~valid[..., None], 0)
    embedded = [model.get_input_embeddings()(safe[..., 0])]
    embedded += [
        table(safe[..., index + 1])
        for index, table in enumerate(model.code_predictor.model.codec_embedding)
    ]
    tail = torch.cat(
        (prepared.trailing_text_hidden, prepared.tts_pad_embed.expand(-1, frames, -1)), 1
    )[:, :frames]
    inputs = torch.cat((prepared.inputs_embeds, torch.stack(embedded).sum(0) + tail), 1)
    positions = torch.arange(inputs.shape[1]).view(1, 1, -1).expand(3, batch, -1)
    attention = torch.cat((prepared.attention_mask, valid.long()), 1)
    hidden = model.model(
        inputs_embeds=inputs, attention_mask=attention, position_ids=positions, use_cache=False
    ).last_hidden_state
    begin = prepared.inputs_embeds.shape[1] - 1
    first = model.codec_head(hidden[:, begin:])
    residual = []
    # Evaluate each group from its independent prefix, as generation would.
    for group in range(groups - 1):
        sequence = torch.stack(
            [hidden[:, begin : begin + frames], embedded[0], *embedded[1 : group + 1]], 2
        )
        flat = sequence.reshape(batch * frames, group + 2, -1)
        values = model.code_predictor.model(inputs_embeds=flat, use_cache=False).last_hidden_state
        residual.append(
            model.code_predictor.lm_head[group](values[:, -1]).reshape(batch, frames, -1)
        )
    return first, torch.stack(residual, 2)


def test_teacher_forcing_all_logits_gradients_and_masked_lengths():
    native, hf = pair()
    torch.manual_seed(33)
    actual_condition = condition(native)
    torch.manual_seed(33)
    expected_condition = condition(hf)
    codes = torch.randint(0, 23, (2, 4, 16))
    lengths = torch.tensor([4, 2])
    codes[1, 2:] = -100  # Poisoned padding must never index a codebook.
    actual = speech_teacher_force(native, actual_condition, codes, lengths)
    first, residual = independent_hf(hf, expected_condition, codes, lengths)
    torch.testing.assert_close(actual.first_logits, first, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual.residual_logits, residual, rtol=1e-5, atol=1e-7)
    assert actual.first_labels.tolist()[1][2:] == [31, -100, -100]
    assert (actual.residual_labels[1, 2:] == -100).all()
    for first_logits, residual_logits in (
        (actual.first_logits, actual.residual_logits),
        (first, residual),
    ):
        loss = F.cross_entropy(
            first_logits.flatten(0, 1), actual.first_labels.flatten(), reduction="sum"
        )
        loss += F.cross_entropy(
            residual_logits.flatten(0, 2), actual.residual_labels.flatten(), reduction="sum"
        )
        loss.backward()
    for name, parameter in native.named_parameters():
        other = hf.get_parameter(name)
        assert parameter.grad is not None and other.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        numerator = (parameter.grad.double() - other.grad.double()).square().sum()
        denominator = parameter.grad.double().square().sum() + other.grad.double().square().sum()
        assert numerator / denominator.clamp_min(1e-30) < 1e-8, name


def test_current_first_code_does_not_leak_and_residual_order_is_causal():
    native, _ = pair()
    prepared = condition(native, batch=1)
    codes = torch.randint(0, 23, (1, 4, 16))
    baseline = speech_teacher_force(native, prepared, codes, torch.tensor([4]))
    changed = codes.clone()
    changed[:, 2] = (changed[:, 2] + 1) % 23
    after = speech_teacher_force(native, prepared, changed, torch.tensor([4]))
    torch.testing.assert_close(
        baseline.first_logits[:, :3], after.first_logits[:, :3], rtol=0, atol=0
    )
    changed = codes.clone()
    changed[:, 1, 10] = (changed[:, 1, 10] + 1) % 23
    after = speech_teacher_force(native, prepared, changed, torch.tensor([4]))
    torch.testing.assert_close(
        baseline.residual_logits[:, 1, :10], after.residual_logits[:, 1, :10], rtol=0, atol=0
    )


def test_conditioning_matches_official_helpers_and_preserves_input_gradients():
    native, hf = pair()
    top = SimpleNamespace(
        talker_config=native.config,
        thinker_config=SimpleNamespace(image_token_id=50, audio_token_id=51, video_token_id=52),
        im_start_token_id=10,
        user_token_id=11,
        assistant_token_id=15,
        tts_bos_token_id=40,
        tts_eos_token_id=41,
        tts_pad_token_id=42,
    )
    ids = torch.tensor([[10, 11, 12, 13, 14, 12, 10, 15, 12, 16, 17]])
    embedding = nn.Embedding(64, 20)
    features = embedding(ids)
    hidden = torch.randn_like(features, requires_grad=True)
    actual = prepare_talker_condition(
        native, features, hidden, ids, config=top, text_embedding=embedding, prompt_lengths=[9]
    )
    special = hf.text_projection(embedding(torch.tensor([[40, 41, 42]])))
    bos, eos, pad = special.chunk(3, 1)
    proxy = SimpleNamespace(config=top, talker=hf)
    user = HFComposite._get_talker_user_parts(
        proxy, 0, 6, torch.zeros_like(ids, dtype=torch.bool), hidden, features
    )
    assistant, assistant_ids, tail = HFComposite._get_talker_assistant_parts(
        proxy, 6, ids.shape[1], 37, features, pad, bos, eos
    )
    torch.testing.assert_close(actual.inputs_embeds, torch.cat((user, assistant), 1))
    torch.testing.assert_close(actual.trailing_text_hidden, tail)
    torch.testing.assert_close(actual.input_ids, torch.cat((ids[:, :6], assistant_ids), 1))
    actual.inputs_embeds.sum().backward()
    assert embedding.weight.grad is not None
    assert native.text_projection.linear_fc1.weight.grad.abs().sum() > 0
    assert hidden.grad is not None  # Zero for text-only; graph remains collective-compatible.


def test_individual_eos_truncation_and_training_mode_restoration():
    native, _ = pair()
    codec = Qwen3OmniMoeCode2Wav(
        Qwen3OmniMoeCode2WavConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=24,
            num_hidden_layers=1,
            num_quantizers=16,
            codebook_size=23,
            decoder_dim=16,
            upsample_rates=[2, 2],
            upsampling_ratios=[2, 2],
            sliding_window=3,
        )
    )
    prepared = condition(native)
    original_forward = native.forward
    calls = []

    def eos_schedule(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        logits = output.logits.clone()
        logits[:, -1] = -1000
        logits[0, -1, 31] = 1000  # Row zero emits EOS immediately.
        logits[1, -1, 31 if len(calls) == 2 else 3] = 1000
        calls.append(1)
        return type(output)(logits, output.last_hidden_state)

    native.forward = eos_schedule
    native.train()
    native.code_predictor.eval()  # Preserve mixed submodule modes.
    before = [module.training for module in native.modules()]
    result = generate_speech_from_condition(native, codec, prepared, max_frames=4)
    assert result.code_lengths.tolist() == [0, 2]
    assert result.reached_eos.tolist() == [True, True]
    assert result.waveforms[0].shape == (1, 0)
    assert result.waveforms[1].shape[-1] > 0
    assert [module.training for module in native.modules()] == before
    calls.clear()
    result = generate_speech_from_condition(native, codec, prepared, max_frames=1)
    assert result.reached_eos.tolist() == [True, False]
    assert result.code_lengths.tolist() == [0, 1]


@pytest.mark.parametrize("length", [0, 2])
def test_empty_codec_eos_and_partial_head_supervision(length):
    native, _ = pair()
    prepared = condition(native, 1)
    codes = torch.zeros((1, length, 16), dtype=torch.long)
    mask = torch.ones_like(codes, dtype=torch.bool)
    mask[..., 2] = False
    outputs = speech_teacher_force(
        native, prepared, codes, torch.tensor([length]), codebook_mask=mask
    )
    assert outputs.first_labels[0, -1] == 31
    assert outputs.first_logits.shape[:2] == (1, length + 1)
    assert (outputs.residual_labels[..., 1] == -100).all()


def test_greedy_all_codebooks_match_official_cached_generation():
    native, hf = pair()
    torch.manual_seed(71)
    prepared = condition(native, 1)
    codec = Qwen3OmniMoeCode2Wav(
        Qwen3OmniMoeCode2WavConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=24,
            num_hidden_layers=1,
            num_quantizers=16,
            codebook_size=23,
            decoder_dim=16,
            upsample_rates=[2, 2],
            upsampling_ratios=[2, 2],
            sliding_window=3,
        )
    )
    actual = generate_speech_from_condition(
        native, codec, prepared, max_frames=3, repetition_penalty=1
    )
    expected = hf.generate(
        inputs_embeds=prepared.inputs_embeds.detach(),
        talker_input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        trailing_text_hidden=prepared.trailing_text_hidden.detach(),
        tts_pad_embed=prepared.tts_pad_embed.detach(),
        max_new_tokens=4,
        do_sample=False,
        repetition_penalty=1,
        eos_token_id=31,
        pad_token_id=35,
        suppress_tokens=[token for token in range(23, 41) if token != 31],
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    frames = [hidden[-1] for hidden in expected.hidden_states if hidden[-1] is not None]
    codes = torch.stack(frames, 1) if frames else actual.codes.new_empty((1, 0, 16))
    torch.testing.assert_close(actual.codes[:, : codes.shape[1]], codes, rtol=0, atol=0)
    assert codes.shape[1] == int(actual.code_lengths[0])


def test_invalid_codec_inputs_fail_before_forward():
    native, _ = pair()
    prepared = condition(native, 1)
    codes = torch.zeros(1, 2, 16, dtype=torch.long)
    with pytest.raises(ValueError, match="RVQ"):
        speech_teacher_force(native, prepared, codes + 23, torch.tensor([2]))
    with pytest.raises(ValueError, match="lengths"):
        speech_teacher_force(native, prepared, codes, torch.tensor([3]))
    with pytest.raises(ValueError, match="supervision"):
        speech_teacher_force(
            native,
            prepared,
            codes,
            torch.tensor([1]),
            codebook_mask=torch.ones_like(codes, dtype=torch.bool),
        )
