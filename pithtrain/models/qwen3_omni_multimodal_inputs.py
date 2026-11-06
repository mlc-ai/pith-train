"""Native Thinker text/image/audio/video geometry and activation insertion.

Geometry follows the public Transformers 5.17 Qwen3-Omni implementation.
The parser additionally verifies every media token and delimiter before scheduling.
"""

import torch

from pithtrain.models.qwen3_omni_audio import audio_output_lengths
from pithtrain.models.qwen3_omni_audio_inputs import audio_feature_lengths, flatten_audio_features
from pithtrain.models.qwen3_omni_vision import VisionOutput, image_grid, video_grid


def synchronized_video(context):
    flag = context.get("use_audio_in_video", False)
    if isinstance(flag, torch.Tensor):
        if flag.numel() != 1 or flag.dtype != torch.bool:
            raise ValueError("use_audio_in_video must be one boolean")
        flag = flag.item()
    if type(flag) is not bool:
        raise ValueError("use_audio_in_video must be one boolean")
    return flag


def _vision_positions(grid, seconds, config, device):
    temporal, height, width = grid
    height //= config.vision_config.spatial_merge_size
    width //= config.vision_config.spatial_merge_size
    t, h, w = torch.meshgrid(
        torch.arange(temporal, device=device, dtype=torch.float32),
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack(
        ((t * seconds * config.position_id_per_seconds).flatten(), h.flatten(), w.flatten())
    )


def multimodal_position_ids(input_ids, context, config, *, require_media=True):
    """Validate one unpadded microbatch, returning exact floating T/H/W positions.

    Each synchronized video contains vision-start, audio-start, time-merged video
    and audio placeholders, audio-end, vision-end. Ties put video before audio.
    Independent image/audio/video spans can appear in any order in the same row.
    """
    allowed = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "video_second_per_grid",
        "input_features",
        "feature_attention_mask",
        "audio_feature_lengths",
        "use_audio_in_video",
    }
    if set(context) - allowed:
        raise ValueError(f"Unsupported multimodal inputs: {sorted(set(context) - allowed)}")
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape[1] == 0
        or input_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Native multimodal training needs one nonempty integer row")
    if input_ids.min() < 0 or input_ids.max() >= config.text_config.vocab_size:
        raise ValueError("input_ids are outside the text vocabulary")
    shared = context.get("input_ids")
    if shared is not None and not torch.equal(input_ids, shared):
        raise ValueError("model_context input_ids must match model_inputs")
    mask = context.get("attention_mask")
    if mask is not None and (mask.shape != input_ids.shape or not (mask == 1).all()):
        raise ValueError("Native multimodal microbatches must be unpadded")
    merge = config.vision_config.spatial_merge_size
    images = (
        []
        if context.get("image_grid_thw") is None
        else image_grid(context["image_grid_thw"], merge)
    )
    videos = (
        []
        if context.get("video_grid_thw") is None
        else video_grid(context["video_grid_thw"], merge)
    )
    seconds = context.get("video_second_per_grid")
    if videos:
        if (
            seconds is None
            or seconds.shape != (len(videos),)
            or not torch.isfinite(seconds).all()
            or not (seconds > 0).all()
        ):
            raise ValueError("Every video needs positive finite video_second_per_grid")
    elif seconds is not None:
        raise ValueError("Unused video_second_per_grid")
    vision = config.vision_config
    patch_width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    for key, grids in (("pixel_values", images), ("pixel_values_videos", videos)):
        pixels = context.get(key)
        if require_media and bool(grids) != (pixels is not None):
            raise ValueError(f"{key} and its grid must be supplied together")
        if pixels is not None and (
            pixels.shape != (sum(t * h * w for t, h, w in grids), patch_width)
            or not pixels.is_floating_point()
            or not torch.isfinite(pixels).all()
        ):
            raise ValueError(f"Invalid {key} shape, dtype or finite values")
    has_audio = any(
        context.get(k) is not None
        for k in ("input_features", "feature_attention_mask", "audio_feature_lengths")
    )
    counts = []
    if has_audio:
        lengths = audio_feature_lengths(context, config, require_features=require_media)
        counts = audio_output_lengths(lengths, config.audio_config.n_window).tolist()
    sync = synchronized_video(context)
    if sync and not videos:
        raise ValueError("use_audio_in_video requires video inputs")
    tokens = input_ids[0].tolist()
    device = input_ids.device
    pieces, cursor, offset = [], 0, 0.0
    image_index = video_index = audio_index = 0
    reserved = {
        config.image_token_id,
        config.video_token_id,
        config.audio_token_id,
        config.vision_end_token_id,
        config.audio_end_token_id,
    }

    def linear(count, start):
        return torch.arange(count, device=device, dtype=torch.float32)[None].expand(3, -1) + start

    while cursor < len(tokens):
        token = tokens[cursor]
        if token in reserved:
            raise ValueError("Media placeholder/end token outside a delimited span")
        if token not in (config.vision_start_token_id, config.audio_start_token_id):
            pieces.append(linear(1, offset))
            cursor, offset = cursor + 1, offset + 1
            continue
        if token == config.audio_start_token_id:
            if audio_index >= len(counts):
                raise ValueError("More audio spans than audio features")
            count = counts[audio_index]
            content = [config.audio_token_id] * count
            media = linear(count, 0)
            opening, ending = [token], [config.audio_end_token_id]
            audio_index += 1
        else:
            kind = tokens[cursor + 1] if cursor + 1 < len(tokens) else None
            if kind == config.image_token_id:
                if image_index >= len(images):
                    raise ValueError("More image spans than image grids")
                media = _vision_positions(images[image_index], 1.0, config, device)
                content = [kind] * media.shape[1]
                opening, ending = [token], [config.vision_end_token_id]
                image_index += 1
            elif kind in (config.video_token_id, config.audio_start_token_id):
                if video_index >= len(videos):
                    raise ValueError("More video spans than video grids")
                video = _vision_positions(videos[video_index], seconds[video_index], config, device)
                video_index += 1
                if kind == config.video_token_id:
                    if sync:
                        raise ValueError("Synchronized video requires an audio-start delimiter")
                    media, content = video, [kind] * video.shape[1]
                    opening, ending = [token], [config.vision_end_token_id]
                else:
                    if not sync or audio_index >= len(counts):
                        raise ValueError("Synchronized video needs its flag and audio features")
                    audio = linear(counts[audio_index], 0)
                    audio_index += 1
                    # Stable merge, including the official video-before-audio tie rule.
                    v, a, merged, content = 0, 0, [], []
                    while v < video.shape[1] or a < audio.shape[1]:
                        take_video = a == audio.shape[1] or (
                            v < video.shape[1] and video[0, v] <= audio[0, a]
                        )
                        if take_video:
                            merged.append(video[:, v : v + 1])
                            content.append(config.video_token_id)
                            v += 1
                        else:
                            merged.append(audio[:, a : a + 1])
                            content.append(config.audio_token_id)
                            a += 1
                    media = torch.cat(merged, dim=1)
                    opening = [token, config.audio_start_token_id]
                    ending = [config.audio_end_token_id, config.vision_end_token_id]
            else:
                raise ValueError("Vision-start must delimit image, video, or synchronized AV")
        expected = opening + content + ending
        if tokens[cursor : cursor + len(expected)] != expected:
            raise ValueError("Media delimiters, placeholder count or AV temporal ordering differ")
        pieces.append(linear(len(opening), offset))
        offset += len(opening)
        pieces.append(media + offset)
        # The HF AV implementation appends the unconsumed suffix as one block,
        # otherwise its last token is the final block. Reproduce that end offset.
        if len(opening) == 2:
            last_kind = content[-1]
            tail = len(content) - 1
            while tail > 0 and content[tail - 1] == last_kind:
                tail -= 1
            # Only the suffix after the other modality is exhausted is batched.
            end_offset = media[:, tail:].max().item() + offset + 1
        else:
            end_offset = media.max().item() + offset + 1
        pieces.append(linear(len(ending), end_offset))
        offset = end_offset + len(ending)
        cursor += len(expected)
    if (image_index, video_index, audio_index) != (len(images), len(videos), len(counts)):
        raise ValueError("Unused media grids or audio features")
    positions = torch.cat(pieces, dim=1).unsqueeze(1)
    supplied = context.get("position_ids")
    if supplied is not None and (
        supplied.shape != positions.shape or not torch.equal(supplied, positions)
    ):
        raise ValueError("position_ids do not match multimodal geometry")
    return positions


def pack_multimodal_features(
    embeddings, input_ids, config, image_output, video_output, audio_output
):
    """Insert final encoder outputs and sparse DeepStack lanes without cached state."""
    count = len(config.vision_config.deepstack_visual_indexes)
    lanes = [torch.zeros_like(embeddings) for _ in range(count)]
    for token_id, output in (
        (config.image_token_id, image_output),
        (config.video_token_id, video_output),
    ):
        if output is None:
            continue
        mask = input_ids == token_id
        features = output.pooler_output
        if (
            features.shape != (int(mask.sum()), embeddings.shape[-1])
            or len(output.deepstack_features) != count
        ):
            raise ValueError(
                "Vision encoder output differs from placeholders/DeepStack configuration"
            )
        embeddings = embeddings.masked_scatter(mask[..., None], features.to(embeddings.dtype))
        for index, values in enumerate(output.deepstack_features):
            if values.shape != features.shape:
                raise ValueError("DeepStack feature shape differs")
            lanes[index] = lanes[index].masked_scatter(mask[..., None], values.to(embeddings.dtype))
    if audio_output is not None:
        mask = input_ids == config.audio_token_id
        features = audio_output.last_hidden_state
        if features.shape != (int(mask.sum()), embeddings.shape[-1]):
            raise ValueError("Audio encoder output differs from placeholders")
        embeddings = embeddings.masked_scatter(mask[..., None], features.to(embeddings.dtype))
    return torch.cat((embeddings, *lanes), dim=-1)


def encode_multimodal(visual, audio_tower, embeddings, input_ids, context, config):
    """Call each FSDP encoder exactly once per microbatch, including text-only.

    Missing modalities use the smallest valid dummy input. A zero-valued graph
    connection from every output creates real zero gradients for every unused
    encoder parameter. This keeps forward all-gathers and backward reductions in
    the same order on ranks whose deterministic mixture samples differ.
    """
    grids, pixels, image_count = [], [], 0
    for grid_key, pixel_key in (
        ("image_grid_thw", "pixel_values"),
        ("video_grid_thw", "pixel_values_videos"),
    ):
        if context.get(grid_key) is not None:
            grids.append(context[grid_key])
            pixels.append(context[pixel_key])
            if grid_key == "image_grid_thw":
                image_count = (
                    int(context[grid_key].prod(-1).sum())
                    // config.vision_config.spatial_merge_size**2
                )
    has_vision = bool(grids)
    if not has_vision:
        vision = config.vision_config
        merge = vision.spatial_merge_size
        grids = [input_ids.new_tensor([[1, merge, merge]])]
        pixels = [
            embeddings.new_zeros(
                (merge**2, vision.in_channels * vision.temporal_patch_size * vision.patch_size**2)
            )
        ]
    # Zero dependencies impose the same backward order (audio -> visual ->
    # embedding) as well as the same forward order, even across absent modalities.
    vision_pixels = torch.cat(pixels) + embeddings.sum().to(pixels[0].dtype) * 0
    kwargs = {"grid_splits": tuple(len(grid) for grid in grids)} if len(grids) > 1 else {}
    output = visual(vision_pixels, torch.cat(grids), **kwargs)
    vision_anchor = output.pooler_output.sum()
    for values in output.deepstack_features:
        vision_anchor = vision_anchor + values.sum()
    image_output = video_output = None
    if has_vision:

        def take(start, end):
            return VisionOutput(
                output.last_hidden_state,
                output.pooler_output[start:end],
                tuple(values[start:end] for values in output.deepstack_features),
            )

        if image_count:
            image_output = take(0, image_count)
        if context.get("video_grid_thw") is not None:
            video_output = take(image_count, None)
    else:
        # Include all DeepStack outputs; an earlier block's merger is otherwise unused.
        embeddings = embeddings + vision_anchor.to(embeddings.dtype) * 0
    has_audio = context.get("input_features") is not None
    if has_audio:
        features = flatten_audio_features(context)
        lengths = context["feature_attention_mask"].long().sum(-1)
    else:
        features = embeddings.new_zeros((config.audio_config.num_mel_bins, 1))
        lengths = input_ids.new_ones(1)
    features = features + vision_anchor.to(features.dtype) * 0
    audio_output = audio_tower(features, lengths)
    audio_anchor = audio_output.last_hidden_state.sum()
    if not has_audio:
        audio_output = None
    payload = pack_multimodal_features(
        embeddings, input_ids, config, image_output, video_output, audio_output
    )
    return payload + audio_anchor.to(payload.dtype) * 0
