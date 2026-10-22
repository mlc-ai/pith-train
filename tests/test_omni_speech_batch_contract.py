"""Real pure-input validation contracts; no distributed/FSDP acceptance claim."""

import copy
import random

import numpy as np
import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

from pithtrain.models.qwen3_omni_speech import prepare_talker_condition, speech_teacher_force
from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration
from pithtrain.modules.omni_speech_batch_contract import validate_speech_batch
from tests.test_omni_speech_integration import config as talker_config
from tests.test_qwen3_omni_multimodal import KINDS, context


class ConfigOnly:
    """Any access other than metadata fails, including parameters and forward."""

    def __init__(self, config):
        object.__setattr__(self, "config", config)

    def __getattribute__(self, name):
        if name != "config":
            raise AssertionError(f"Validator accessed model.{name}")
        return object.__getattribute__(self, name)


def sample(kind="text", frames=2, length=None):
    thinker, inputs = context(kind, frames=17)
    thinker.text_config.vocab_size = 256
    talker = talker_config()
    for name in (
        "image_token_id",
        "video_token_id",
        "audio_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "audio_start_token_id",
        "audio_end_token_id",
        "position_id_per_seconds",
    ):
        setattr(talker, name, getattr(thinker, name))
    cfg = Qwen3OmniMoeConfig(
        thinker_config=thinker.to_dict(),
        talker_config=talker.to_dict(),
        code2wav_config=dict(
            num_quantizers=16, codebook_size=23, upsampling_ratios=[2, 2], upsample_rates=[2, 2]
        ),
        im_start_token_id=100,
        user_token_id=101,
        assistant_token_id=102,
        tts_bos_token_id=200,
        tts_eos_token_id=201,
        tts_pad_token_id=202,
    )
    ids = torch.cat(
        (
            torch.tensor([[100, 101, 198]]),
            inputs["input_ids"],
            torch.tensor([[100, 102, 198, 40, 41]]),
        ),
        1,
    )
    inputs.update(input_ids=ids, attention_mask=torch.ones_like(ids))
    positions = {
        key: inputs[key]
        for key in ("image_grid_thw", "video_grid_thw", "use_audio_in_video")
        if key in inputs
    }
    if "feature_attention_mask" in inputs:
        positions["audio_seqlens"] = inputs["feature_attention_mask"].sum(-1)
    if "video_second_per_grid" in inputs:
        positions["second_per_grids"] = inputs["video_second_per_grid"]
    batch = dict(
        input_ids=ids,
        model_context=inputs,
        prompt_lengths=[ids.shape[1] - 2],
        labels=torch.full_like(ids, -100),
        codes=torch.zeros(1, frames, 16, dtype=torch.long),
        code_lengths=torch.tensor([frames if length is None else length]),
        position_kwargs=positions,
    )
    return ConfigOnly(cfg), batch


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ("codec_only", "text_and_codec"))
def test_original_media_geometry_and_real_talker_prompt(kind, mode):
    torch.set_num_threads(1)
    owner, batch = sample(kind)
    assert validate_speech_batch(batch, owner, mode, "ETHAN") is None
    # Exercise the actual downstream prefix and teacher-forcing contracts on CPU.
    cfg = owner.config
    talker = Qwen3OmniMoeTalkerForConditionalGeneration(cfg.talker_config)
    width = cfg.talker_config.thinker_hidden_size
    embeddings = torch.zeros(*batch["input_ids"].shape, width)
    condition = prepare_talker_condition(
        talker,
        embeddings,
        embeddings,
        batch["input_ids"],
        config=cfg,
        text_embedding=torch.nn.Embedding(256, width),
        prompt_lengths=batch["prompt_lengths"],
        attention_mask=batch["model_context"]["attention_mask"],
        position_kwargs=batch["position_kwargs"],
    )
    result = speech_teacher_force(talker, condition, batch["codes"], batch["code_lengths"])
    assert result.first_logits.shape == (1, 3, 41)
    assert result.residual_logits.shape == (1, 2, 15, 23)


def mutate(batch, field):
    if field == "codes_dtype":
        batch["codes"] = batch["codes"].float()
    elif field == "length_dtype":
        batch["code_lengths"] = batch["code_lengths"].int()
    elif field == "length_range":
        batch["code_lengths"][0] = 3
    elif field == "masked_bad_id":
        batch["codes"][0, 0, 1] = 99
        batch["codebook_mask"] = torch.zeros_like(batch["codes"], dtype=torch.bool)
    elif field == "mask_padding":
        batch["code_lengths"][0] = 1
        batch["codebook_mask"] = torch.ones_like(batch["codes"], dtype=torch.bool)
    elif field == "mask_dtype":
        batch["codebook_mask"] = torch.ones_like(batch["codes"])
    elif field == "missing_context_ids":
        del batch["model_context"]["input_ids"]
    elif field == "context_ids_drift":
        batch["model_context"]["input_ids"] = batch["input_ids"] + 1
    elif field == "prompt_float":
        batch["prompt_lengths"] = [float(batch["prompt_lengths"][0])]
    elif field == "prompt_boundary":
        batch["prompt_lengths"][0] -= 1
    elif field == "empty_response":
        batch["prompt_lengths"][0] = batch["input_ids"].shape[1]
    elif field == "no_user":
        batch["input_ids"][0, 1] = 103
    elif field == "padded_thinker":
        batch["model_context"]["attention_mask"][0, 0] = 0
    elif field == "text_label_range":
        batch["labels"][0, 0] = 256
    elif field == "text_label_negative":
        batch["labels"][0, 0] = -1
    elif field == "text_label_dtype":
        batch["labels"] = batch["labels"].int()
    elif field == "text_label_shape":
        batch["labels"] = batch["labels"][:, :-1]
    elif field == "position_input_override":
        batch["position_kwargs"]["input_ids"] = batch["input_ids"]
    elif field == "position_window":
        batch["position_kwargs"]["audio_n_window"] = 49
    elif field == "bad_groups":
        batch["codes"] = batch["codes"][..., :-1]
    else:
        raise AssertionError(field)


@pytest.mark.parametrize(
    "field",
    (
        "codes_dtype",
        "length_dtype",
        "length_range",
        "masked_bad_id",
        "mask_padding",
        "mask_dtype",
        "missing_context_ids",
        "context_ids_drift",
        "prompt_float",
        "prompt_boundary",
        "empty_response",
        "no_user",
        "padded_thinker",
        "text_label_range",
        "text_label_negative",
        "text_label_dtype",
        "text_label_shape",
        "position_input_override",
        "position_window",
        "bad_groups",
    ),
)
def test_reject_before_forward_for_actual_downstream_errors(field):
    owner, batch = sample()
    mutate(batch, field)
    with pytest.raises(ValueError):
        validate_speech_batch(batch, owner, "text_and_codec", "ethan")


@pytest.mark.parametrize(
    "field",
    ("image_grid_thw", "video_grid_thw", "audio_seqlens", "second_per_grids", "use_audio_in_video"),
)
def test_talker_position_metadata_must_match_thinker(field):
    owner, batch = sample("all")
    batch["position_kwargs"][field] = batch["position_kwargs"][field].clone()
    if field == "use_audio_in_video":
        batch["position_kwargs"][field] = False
    else:
        batch["position_kwargs"][field].reshape(-1)[0] += 1
    with pytest.raises(ValueError):
        validate_speech_batch(batch, owner, "codec_only", "ethan")


def test_media_must_remain_in_user_conditioning_prefix():
    owner, batch = sample("image")
    ids = torch.tensor(
        [[100, 103, 198, 13, 10, 10, 14, 100, 101, 198, 3, 4, 100, 102, 198, 40, 41]]
    )
    batch.update(input_ids=ids, labels=torch.full_like(ids, -100), prompt_lengths=[15])
    batch["model_context"].update(input_ids=ids, attention_mask=torch.ones_like(ids))
    with pytest.raises(ValueError, match="Unused|missing"):
        validate_speech_batch(batch, owner, "codec_only", "ethan")


@pytest.mark.parametrize("frames,length", ((0, 0), (3, 0), (3, 1), (3, 3)))
def test_eos_lengths_and_padding_are_valid_without_routing_on_target_counts(frames, length):
    owner, batch = sample(frames=frames, length=length)
    batch["codes"][:, length:] = -999  # Never embedded by teacher forcing.
    batch["codebook_mask"] = torch.zeros_like(batch["codes"], dtype=torch.bool)
    validate_speech_batch(batch, owner, "text_and_codec", "ethan")


def test_waveform_does_not_invent_an_objective_input_schema():
    owner, batch = sample()
    assert (
        validate_speech_batch(
            {"codes": batch["codes"], "spectral_target": object()}, owner, "waveform", "unused"
        )
        is None
    )
    for bad in (
        batch["codes"][:, :0],
        batch["codes"].float(),
        batch["codes"] + 23,
        batch["codes"][..., :15],
    ):
        with pytest.raises(ValueError):
            validate_speech_batch({"codes": bad}, owner, "waveform", "unused")


def test_configuration_speaker_and_special_embedding_bounds_are_checked():
    owner, batch = sample()
    with pytest.raises(ValueError, match="speaker"):
        validate_speech_batch(batch, owner, "codec_only", "unknown")
    owner.config.tts_pad_token_id = 256
    with pytest.raises(ValueError, match="vocabulary"):
        validate_speech_batch(batch, owner, "codec_only", "ethan")


def test_no_input_model_rng_autograd_or_collective_side_effects(monkeypatch):
    owner, batch = sample("all")
    before = copy.deepcopy(batch)
    torch_rng, python_rng, numpy_rng = (
        torch.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight attempted a forward/collective")

    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.distributed, "all_gather_object", forbidden)
    monkeypatch.setattr(torch.distributed, "all_reduce", forbidden)
    validate_speech_batch(batch, owner, "text_and_codec", "ethan")

    def same(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b) and a.dtype == b.dtype and a.grad is None
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                same(a[key], b[key])
        else:
            assert a == b

    same(batch, before)
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    actual = np.random.get_state()
    assert (
        numpy_rng[0] == actual[0]
        and np.array_equal(numpy_rng[1], actual[1])
        and numpy_rng[2:] == actual[2:]
    )
    assert not torch.cuda.is_initialized()
