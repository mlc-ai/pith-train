"""Validated audio+text placeholder geometry and differentiable feature insertion."""

import torch

from pithtrain.models.qwen3_omni_audio import audio_output_lengths


def audio_feature_lengths(context, config, *, require_features=True):
    mask = context.get("feature_attention_mask")
    if (
        mask is None
        or mask.ndim != 2
        or mask.shape[0] == 0
        or mask.shape[1] == 0
        or not ((mask == 0) | (mask == 1)).all()
    ):
        raise ValueError("feature_attention_mask must be a nonempty binary [audios, frames] mask")
    lengths = mask.long().sum(-1)
    if not (lengths > 0).all() or not torch.equal(
        mask.bool(), torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
    ):
        raise ValueError("Audio feature masks must contain nonempty right-padded sequences")
    declared = context.get("audio_feature_lengths")
    if declared is not None and (
        declared.dtype not in (torch.int32, torch.int64)
        or declared.shape != lengths.shape
        or not torch.equal(declared, lengths)
    ):
        raise ValueError("audio_feature_lengths differ from feature_attention_mask")
    features = context.get("input_features")
    if require_features and features is None:
        raise ValueError("Audio placeholders require input_features")
    if features is not None and (
        features.shape != (mask.shape[0], config.audio_config.num_mel_bins, mask.shape[1])
        or not features.is_floating_point()
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Invalid input_features shape, dtype or finite values")
    return lengths


def validate_audio_inputs(input_ids, context, config, *, require_features=True):
    allowed = {
        "input_ids",
        "attention_mask",
        "input_features",
        "feature_attention_mask",
        "audio_feature_lengths",
        "position_ids",
    }
    if set(context) - allowed:
        raise ValueError(f"Unsupported audio model inputs: {sorted(set(context) - allowed)}")
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape[1] == 0
        or input_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Native audio requires one nonempty integer sample per microbatch")
    shared_ids = context.get("input_ids")
    if shared_ids is not None and not torch.equal(shared_ids, input_ids):
        raise ValueError("model_context input_ids must match model_inputs")
    if input_ids.min() < 0 or input_ids.max() >= config.text_config.vocab_size:
        raise ValueError("input_ids are outside the text vocabulary")
    mask = context.get("attention_mask")
    if mask is not None and (mask.shape != input_ids.shape or not (mask == 1).all()):
        raise ValueError("Native audio microbatches must be unpadded")
    for name in ("image_token_id", "video_token_id"):
        if (input_ids == getattr(config, name)).any():
            raise ValueError("Audio-only training does not consume image/video tokens")
    lengths = audio_feature_lengths(context, config, require_features=require_features)
    counts = audio_output_lengths(lengths, config.audio_config.n_window).tolist()
    tokens = input_ids[0].tolist()
    clip, cursor = 0, 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if token == config.audio_token_id or token == config.audio_end_token_id:
            raise ValueError("Audio placeholders/end tokens must belong to a delimited audio span")
        if token != config.audio_start_token_id:
            cursor += 1
            continue
        if clip >= len(counts):
            raise ValueError("More audio spans than input feature sequences")
        end = cursor + 1 + counts[clip]
        if (
            tokens[cursor + 1 : end] != [config.audio_token_id] * counts[clip]
            or end >= len(tokens)
            or tokens[end] != config.audio_end_token_id
        ):
            raise ValueError(
                "Audio placeholder count or delimiters differ from encoder output length"
            )
        cursor, clip = end + 1, clip + 1
    if clip != len(counts):
        raise ValueError("Unused audio features or missing audio spans")
    positions = (
        torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, 1, -1)
    )
    supplied = context.get("position_ids")
    if supplied is not None and (
        supplied.shape != positions.shape or not torch.equal(supplied, positions)
    ):
        raise ValueError("position_ids do not match audio/text geometry")
    return positions


def flatten_audio_features(context):
    """Match the official processor mask extraction without loading padded frames."""
    mask = context["feature_attention_mask"].bool()
    return context["input_features"].transpose(1, 2)[mask].T


def insert_audio_features(embeddings, input_ids, audio_token_id, features):
    mask = input_ids == audio_token_id
    if features.shape != (int(mask.sum()), embeddings.shape[-1]):
        raise ValueError("Audio features and placeholder count/hidden size differ")
    return embeddings.masked_scatter(mask[..., None], features.to(embeddings.dtype))
