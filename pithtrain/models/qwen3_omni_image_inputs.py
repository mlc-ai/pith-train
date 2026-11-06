"""Image-only Thinker geometry and differentiable activation transport."""

import torch

from pithtrain.models.qwen3_omni_vision import image_grid


def image_position_ids(input_ids, image_grid_thw, config, attention_mask=None):
    """Construct HF-compatible T/H/W positions; validate each placeholder run.

    Images have T=1. Padding positions are zero for image batches (one for text,
    matching the pinned HF implementation). The production image stage currently
    requires one unpadded sample per microbatch.
    """
    if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must have integer shape [batch, sequence]")
    mask = (
        torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask is None
        else attention_mask.bool()
    )
    if mask.shape != input_ids.shape:
        raise ValueError("attention_mask shape must match input_ids")
    if image_grid_thw is None:
        if (input_ids[mask] == config.image_token_id).any():
            raise ValueError("Image placeholders require image_grid_thw")
        positions = mask.long().cumsum(-1) - 1
        return positions.masked_fill(~mask, 1).unsqueeze(0).expand(3, -1, -1)
    grids = image_grid(image_grid_thw, config.vision_config.spatial_merge_size)
    positions = torch.zeros((3, *input_ids.shape), dtype=torch.long, device=input_ids.device)
    image_index = 0
    merge = config.vision_config.spatial_merge_size
    for batch_index, row in enumerate(input_ids):
        tokens = row[mask[batch_index]].tolist()
        axes, cursor, offset = [], 0, 0
        while cursor < len(tokens):
            token = tokens[cursor]
            if token != config.image_token_id:
                axes.append(torch.full((3, 1), offset, device=row.device, dtype=torch.long))
                cursor, offset = cursor + 1, offset + 1
                continue
            if cursor == 0 or tokens[cursor - 1] != config.vision_start_token_id:
                raise ValueError("Every image run must follow vision_start_token_id")
            if image_index >= len(grids):
                raise ValueError("More image placeholder runs than grids")
            _, height, width = grids[image_index]
            height, width = height // merge, width // merge
            count = height * width
            if tokens[cursor : cursor + count] != [config.image_token_id] * count:
                raise ValueError("Image placeholder count does not match grid feature count")
            if (
                cursor + count >= len(tokens)
                or tokens[cursor + count] != config.vision_end_token_id
            ):
                raise ValueError(
                    "Image run must end with vision_end_token_id at its grid feature count"
                )
            h, w = torch.meshgrid(
                torch.arange(height, device=row.device),
                torch.arange(width, device=row.device),
                indexing="ij",
            )
            axes.append(
                torch.stack((torch.zeros_like(h).flatten(), h.flatten(), w.flatten())) + offset
            )
            offset += max(height, width)
            cursor += count
            image_index += 1
        positions[:, batch_index, mask[batch_index]] = torch.cat(axes, dim=1)
    if image_index != len(grids):
        raise ValueError("Unused image grids")
    return positions


def validate_image_inputs(input_ids, context, config, *, require_pixels=True):
    """Fail before scheduling; never silently consume audio/video or padding."""
    allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "position_ids"}
    if set(context) - allowed:
        raise ValueError(f"Unsupported image model inputs: {sorted(set(context) - allowed)}")
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be nonempty [batch, sequence]")
    shared_ids = context.get("input_ids")
    if shared_ids is not None and not torch.equal(input_ids, shared_ids):
        raise ValueError("model_context input_ids must match model_inputs")
    if input_ids.min() < 0 or input_ids.max() >= config.text_config.vocab_size:
        raise ValueError("input_ids are outside the text vocabulary")
    if input_ids.shape[0] != 1:
        raise ValueError("Native image training requires one sample per microbatch")
    mask = context.get("attention_mask")
    if mask is not None and (mask.shape != input_ids.shape or not (mask == 1).all()):
        raise ValueError("Native image microbatches must be unpadded")
    for name in ("audio_token_id", "video_token_id"):
        if (input_ids == getattr(config, name)).any():
            raise ValueError("Native image training does not support audio/video tokens")
    grid, pixels = context.get("image_grid_thw"), context.get("pixel_values")
    positions = image_position_ids(input_ids, grid, config, mask)
    if require_pixels and (grid is None) != (pixels is None):
        raise ValueError("pixel_values and image_grid_thw must be supplied together")
    if pixels is not None:
        patches = sum(
            h * w for _, h, w in image_grid(grid, config.vision_config.spatial_merge_size)
        )
        vision = config.vision_config
        width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        if (
            pixels.shape != (patches, width)
            or not pixels.is_floating_point()
            or not torch.isfinite(pixels).all()
        ):
            raise ValueError("Invalid image pixel_values shape, dtype or finite values")
    supplied = context.get("position_ids")
    if supplied is not None and (
        supplied.shape != positions.shape or not torch.equal(supplied, positions)
    ):
        raise ValueError("position_ids do not match image/text geometry")
    return positions


def pack_image_features(embeddings, input_ids, image_token_id, vision_output, deepstack_count):
    """Carry DeepStack through ordinary BSH activation edges and their backward.

    Width is H*(1+K): decoder hidden states, followed by K sparse feature lanes.
    Lanes are zero at text tokens. This intentionally spends bandwidth in the
    first implementation to reuse the engine's per-microbatch activation lifetime
    and P2P autograd, rather than storing feature tensors in module attributes.
    """
    mask = input_ids == image_token_id
    lanes = []
    if vision_output is not None:
        features = vision_output.pooler_output
        if features.shape != (int(mask.sum()), embeddings.shape[-1]):
            raise ValueError("Image features and placeholder count/hidden size differ")
        if len(vision_output.deepstack_features) != deepstack_count:
            raise ValueError("DeepStack feature count differs from configuration")
        embeddings = embeddings.masked_scatter(mask[..., None], features.to(embeddings.dtype))
        for values in vision_output.deepstack_features:
            if values.shape != features.shape:
                raise ValueError("DeepStack projection shape mismatch")
            lanes.append(
                torch.zeros_like(embeddings).masked_scatter(
                    mask[..., None], values.to(embeddings.dtype)
                )
            )
    else:
        if mask.any():
            raise ValueError("Image placeholders have no encoder output")
        lanes = [torch.zeros_like(embeddings) for _ in range(deepstack_count)]
    return torch.cat((embeddings, *lanes), dim=-1)


def inject_deepstack(hidden_states, lanes, layer_index, hidden_size, deepstack_count):
    # HF injects projected vision blocks after decoder layers 0..K-1, in order.
    if layer_index < deepstack_count:
        hidden_states = (
            hidden_states + lanes[..., layer_index * hidden_size : (layer_index + 1) * hidden_size]
        )
    return torch.cat((hidden_states, lanes), dim=-1)
