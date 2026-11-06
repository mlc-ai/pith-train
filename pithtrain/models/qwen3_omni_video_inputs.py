"""Native video-only Thinker geometry, preserving fractional temporal positions."""

import torch

from pithtrain.models.qwen3_omni_vision import video_grid


def video_position_ids(input_ids, video_grid_thw, second_per_grid, config, attention_mask=None):
    """Match HF 5.17 video RoPE: temporal grid time, then spatial merged coordinates.

    Temporal positions remain FP32: fractional frame intervals must not be cast
    to integer token positions. Each new text span starts after the maximum of
    all three preceding vision axes, including its temporal positions.
    """
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must have integer shape [batch, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain tokens")
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        if (
            attention_mask.shape != input_ids.shape
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise ValueError("attention_mask must be binary and match input_ids")
        mask = attention_mask.bool()
    grids = video_grid(video_grid_thw, config.vision_config.spatial_merge_size)
    if (
        not isinstance(second_per_grid, torch.Tensor)
        or second_per_grid.ndim != 1
        or len(second_per_grid) != len(grids)
        or second_per_grid.dtype == torch.bool
        or not torch.isfinite(second_per_grid).all()
        or not (second_per_grid > 0).all()
    ):
        raise ValueError(
            "video_second_per_grid must contain one finite positive interval per video"
        )
    rate = config.position_id_per_seconds
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0 < rate < float("inf"):
        raise ValueError("position_id_per_seconds must be finite and positive")
    positions = torch.zeros((3, *input_ids.shape), dtype=torch.float32, device=input_ids.device)
    video_index = 0
    merge = config.vision_config.spatial_merge_size
    for batch_index, row in enumerate(input_ids):
        tokens = row[mask[batch_index]].tolist()
        axes, cursor, offset = [], 0, 0.0
        while cursor < len(tokens):
            if tokens[cursor] != config.video_token_id:
                axes.append(torch.full((3, 1), offset, device=row.device, dtype=torch.float32))
                cursor, offset = cursor + 1, offset + 1
                continue
            if cursor == 0 or tokens[cursor - 1] != config.vision_start_token_id:
                raise ValueError("Every video run must follow vision_start_token_id")
            if video_index >= len(grids):
                raise ValueError("More video placeholder runs than grids")
            time, height, width = grids[video_index]
            height, width = height // merge, width // merge
            count = time * height * width
            if tokens[cursor : cursor + count] != [config.video_token_id] * count:
                raise ValueError("Video placeholder count does not match grid feature count")
            if (
                cursor + count >= len(tokens)
                or tokens[cursor + count] != config.vision_end_token_id
            ):
                raise ValueError("Video run must end with vision_end_token_id at its feature count")
            interval = second_per_grid[video_index].to(device=row.device, dtype=torch.float32)
            temporal = torch.arange(time, device=row.device, dtype=torch.float32) * interval * rate
            h, w = torch.meshgrid(
                torch.arange(height, device=row.device, dtype=torch.float32),
                torch.arange(width, device=row.device, dtype=torch.float32),
                indexing="ij",
            )
            block = (
                torch.stack(
                    (
                        temporal.repeat_interleave(height * width),
                        h.flatten().repeat(time),
                        w.flatten().repeat(time),
                    )
                )
                + offset
            )
            axes.append(block)
            offset = float(block.max()) + 1
            cursor += count
            video_index += 1
        if axes:
            positions[:, batch_index, mask[batch_index]] = torch.cat(axes, dim=1)
    if video_index != len(grids):
        raise ValueError("Unused video grids")
    return positions


def validate_video_inputs(input_ids, context, config, *, require_pixels=True):
    """Reject unsupported media/packing before any distributed model collectives."""
    allowed = {
        "input_ids",
        "attention_mask",
        "pixel_values_videos",
        "video_grid_thw",
        "video_second_per_grid",
        "position_ids",
    }
    if set(context) - allowed:
        raise ValueError(f"Unsupported video model inputs: {sorted(set(context) - allowed)}")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Native video training requires one nonempty sample per microbatch")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must have integer dtype")
    if input_ids.min() < 0 or input_ids.max() >= config.text_config.vocab_size:
        raise ValueError("input_ids are outside the text vocabulary")
    shared = context.get("input_ids")
    if shared is not None and not torch.equal(shared, input_ids):
        raise ValueError("model_context input_ids must match model_inputs")
    mask = context.get("attention_mask")
    if mask is not None and (mask.shape != input_ids.shape or not (mask == 1).all()):
        raise ValueError("Native video microbatches must be unpadded")
    if any(
        (input_ids == getattr(config, name)).any() for name in ("audio_token_id", "image_token_id")
    ):
        raise ValueError("Native video mode does not support audio/image tokens")
    grid = context.get("video_grid_thw")
    if grid is None or not (input_ids == config.video_token_id).any():
        raise ValueError("Native video mode requires video grids and placeholders")
    positions = video_position_ids(
        input_ids, grid, context.get("video_second_per_grid"), config, mask
    )
    pixels = context.get("pixel_values_videos")
    if require_pixels and pixels is None:
        raise ValueError("pixel_values_videos is required on the input stage")
    if pixels is not None:
        vision = config.vision_config
        patches = sum(t * h * w for t, h, w in video_grid(grid, vision.spatial_merge_size))
        width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        if (
            pixels.shape != (patches, width)
            or not pixels.is_floating_point()
            or not torch.isfinite(pixels).all()
        ):
            raise ValueError("Invalid video pixel_values_videos shape, dtype or finite values")
    supplied = context.get("position_ids")
    if supplied is not None and (
        supplied.shape != positions.shape or not torch.equal(supplied, positions)
    ):
        raise ValueError("position_ids do not match video/text geometry")
    return positions
