"""Parameter-free preflight for the complete native Qwen3-Omni speech task.

Call locally before any FSDP forward, then collectively reject local errors.
This validates data, not distributed execution or an arbitrary waveform loss.
Only ``model.config`` is read: no module method, parameter, RNG or collective.
"""

import math

import torch

from pithtrain.models.qwen3_omni_audio_inputs import audio_feature_lengths
from pithtrain.models.qwen3_omni_multimodal_inputs import (
    multimodal_position_ids,
    synchronized_video,
)
from pithtrain.models.qwen3_omni_talker_positions import TalkerPositionBuilder


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, name, *, device=None, dtype=None, shape=None):
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    _require(value.device.type != "meta", f"{name} must contain actual data")
    _require(device is None or value.device == device, f"{name} is on a different device")
    _require(dtype is None or value.dtype == dtype, f"{name} has an invalid dtype")
    _require(shape is None or tuple(value.shape) == tuple(shape), f"{name} has an invalid shape")
    return value


def _codec(batch, config, mode):
    codes = _tensor(batch.get("codes"), "codes", dtype=torch.long)
    _require(codes.ndim == 3 and codes.shape[0] == 1, "codes must be int64 [1,time,groups]")
    frames, groups = codes.shape[1:]
    if mode == "waveform":
        wave = config.code2wav_config
        _require(frames > 0 and groups == wave.num_quantizers, "Invalid waveform time/groups")
        _require(
            not ((codes < 0) | (codes >= wave.codebook_size)).any(),
            "Waveform codes must exclude EOS/padding and fit the codebook",
        )
        # Same integer geometry as code2wav.output_length, without invoking a module.
        length = frames * math.prod(wave.upsampling_ratios)
        for rate in wave.upsample_rates:
            length = (length - 1) * rate
        _require(length > 0, "Code sequence is too short for the configured waveform decoder")
        return codes
    talker = config.talker_config
    _require(
        groups == talker.num_code_groups == talker.code_predictor_config.num_code_groups,
        "Codec group count differs from Talker/predictor",
    )
    lengths = _tensor(
        batch.get("code_lengths"), "code_lengths", device=codes.device, dtype=torch.long, shape=(1,)
    )
    _require(not ((lengths < 0) | (lengths > frames)).any(), "Codec lengths exceed padded frames")
    valid = torch.arange(frames, device=codes.device)[None] < lengths[:, None]
    mask = batch.get("codebook_mask")
    if mask is not None:
        _tensor(mask, "codebook_mask", device=codes.device, dtype=torch.bool, shape=codes.shape)
        _require(not (mask & ~valid.unsqueeze(-1)).any(), "Codec mask extends beyond valid frames")
    active = codes[valid]
    _require(
        not ((active < 0) | (active >= talker.code_predictor_config.vocab_size)).any(),
        "Active codec IDs exceed the residual vocabulary, including unsupervised groups",
    )
    _require(
        not (active[:, 0] >= talker.text_config.vocab_size).any(),
        "First codec IDs exceed the Talker vocabulary",
    )
    return codes


def _prompt_ids(batch, ids, config, speaker):
    """Validate the exact token-only branch of prepare_talker_condition (B=1)."""
    talker = config.talker_config
    _require(isinstance(speaker, str), "speaker must be a string")
    speaker_id = talker.speaker_id.get(speaker.lower())
    _require(speaker_id is not None, "Unknown Qwen3-Omni speaker")
    for name in ("tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id"):
        value = getattr(config, name)
        _require(
            type(value) is int and 0 <= value < config.thinker_config.text_config.vocab_size,
            f"{name} exceeds the Thinker embedding vocabulary",
        )
    for value in (
        talker.codec_nothink_id,
        talker.codec_think_bos_id,
        talker.codec_think_eos_id,
        speaker_id,
        talker.codec_pad_id,
        talker.codec_bos_id,
        talker.codec_eos_token_id,
    ):
        _require(
            type(value) is int and 0 <= value < talker.text_config.vocab_size,
            "Codec prefix/EOS/speaker ID exceeds the Talker embedding vocabulary",
        )
    lengths = batch.get("prompt_lengths")
    if isinstance(lengths, torch.Tensor):
        _tensor(lengths, "prompt_lengths", device=ids.device, shape=(1,))
        _require(lengths.dtype in (torch.int32, torch.int64), "prompt_lengths must be integers")
        lengths = lengths.tolist()
    _require(
        isinstance(lengths, (list, tuple)) and len(lengths) == 1 and type(lengths[0]) is int,
        "prompt_lengths must contain exactly one integer",
    )
    end = lengths[0]
    tokens = ids[0]
    _require(0 < end < ids.shape[1], "Prompt must leave a nonempty assistant response")
    starts = torch.nonzero(
        (tokens[:-1] == config.im_start_token_id) & (tokens[1:] == config.assistant_token_id),
        as_tuple=False,
    ).flatten()
    starts = starts[starts < end]
    _require(starts.numel() > 0 and end == int(starts[-1]) + 3, "Invalid assistant header boundary")
    start = int(starts[-1])
    user_indices, role = [], None
    for index in range(start):
        if tokens[index] == config.im_start_token_id and index + 1 < start:
            role = int(tokens[index + 1])
        if role == config.user_token_id:
            user_indices.append(index)
    _require(bool(user_indices), "Speech conditioning requires at least one user turn")
    return torch.cat((tokens[user_indices], tokens.new_full((9,), config.tts_pad_token_id)))[None]


def _positions(batch, context, prefix, config):
    kwargs = batch.get("position_kwargs")
    kwargs = {} if kwargs is None else kwargs
    _require(isinstance(kwargs, dict), "position_kwargs must be a dictionary")
    allowed = {
        "image_grid_thw",
        "video_grid_thw",
        "audio_seqlens",
        "second_per_grids",
        "use_audio_in_video",
        "audio_n_window",
    }
    _require(not set(kwargs) - allowed, "Unsupported Talker position kwargs")
    expected = {
        target: context[source]
        for source, target in (
            ("image_grid_thw", "image_grid_thw"),
            ("video_grid_thw", "video_grid_thw"),
            ("video_second_per_grid", "second_per_grids"),
        )
        if context.get(source) is not None
    }
    if context.get("input_features") is not None:
        expected["audio_seqlens"] = audio_feature_lengths(context, config.thinker_config)
    for name in ("image_grid_thw", "video_grid_thw", "audio_seqlens", "second_per_grids"):
        actual, wanted = kwargs.get(name), expected.get(name)
        if wanted is None:
            _require(actual is None, f"Unused Talker position field: {name}")
        else:
            _tensor(actual, name, device=prefix.device, shape=wanted.shape)
            _require(
                actual.dtype == wanted.dtype, f"Talker {name} dtype differs from Thinker media"
            )
            _require(torch.equal(actual, wanted), f"Talker {name} differs from Thinker media")
    _require(
        synchronized_video(kwargs) == synchronized_video(context),
        "Talker and Thinker synchronized-video flags differ",
    )
    window = kwargs.get("audio_n_window", 50)
    _require(type(window) is int, "audio_n_window must be an integer")
    positions, _ = TalkerPositionBuilder(
        config.talker_config, audio_n_window=window
    ).get_rope_index(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        **{key: value for key, value in kwargs.items() if key != "audio_n_window"},
    )
    _require(
        positions.shape == (3, 1, prefix.shape[1]) and torch.isfinite(positions).all(),
        "Talker position geometry is invalid",
    )


def validate_speech_batch(batch, model, mode, speaker="ethan"):
    """Validate one native B=1 microbatch; return None without reading model weights.

    Arbitrary waveform objectives retain their caller-defined input contract.
    Rank-uniform invocation ordering and synchronized rejection belong to the
    FSDP caller; this function never invokes a distributed operation.
    """
    _require(isinstance(batch, dict), "Speech microbatch must be a dictionary")
    _require(mode in {"codec_only", "text_and_codec", "waveform"}, "Unknown speech mode")
    config = model.config
    _require(config.model_type == "qwen3_omni_moe", "Only Qwen3-Omni is supported")
    codes = _codec(batch, config, mode)
    if mode == "waveform":
        return
    ids = _tensor(batch.get("input_ids"), "input_ids", device=codes.device)
    _require(ids.dtype in (torch.int32, torch.int64), "input_ids must be integer")
    context = batch.get("model_context")
    _require(isinstance(context, dict), "Explicit model_context is required")
    _tensor(context.get("input_ids"), "model_context.input_ids", device=ids.device, shape=ids.shape)
    for name, value in context.items():
        if isinstance(value, torch.Tensor):
            _tensor(value, name, device=ids.device)
    # Reuse all native media shapes, delimiters, positions and finite-value checks.
    multimodal_position_ids(ids, context, config.thinker_config)
    if mode == "text_and_codec":
        labels = _tensor(
            batch.get("labels"), "labels", device=ids.device, dtype=torch.long, shape=ids.shape
        )
        _require(
            not (
                ((labels < 0) & (labels != -100))
                | (labels >= config.thinker_config.text_config.vocab_size)
            ).any(),
            "Text labels must be -100 or valid vocabulary IDs",
        )
    prefix = _prompt_ids(batch, ids, config, speaker)
    # The Talker retains user turns only. Reject media attached to discarded
    # assistant/system turns rather than carrying unused grids into its RoPE.
    prefix_context = dict(context, input_ids=prefix, attention_mask=torch.ones_like(prefix))
    prefix_context.pop("position_ids", None)
    multimodal_position_ids(prefix, prefix_context, config.thinker_config)
    _positions(batch, context, prefix, config)
