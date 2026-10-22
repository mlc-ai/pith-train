"""Native visual encoders for Qwen3-Omni; no Transformers execution dependency.

Parameter names and math follow Transformers 5.17 Qwen3OmniMoeVisionEncoder
(Apache-2.0). Geometry is computed per call; no image features are cached on a
module. Attention segments are independent images or temporal frames, never the packed batch.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class VisionOutput(NamedTuple):
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: tuple[torch.Tensor, ...]


def image_grid(grid_thw, merge_size):
    """Validate before allocation/attention. Video support is intentionally separate."""
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.shape[0] == 0:
        raise ValueError("image_grid_thw must be nonempty [images, 3]")
    if grid_thw.dtype not in (torch.int32, torch.int64):
        raise ValueError("image_grid_thw must have integer dtype")
    grids = grid_thw.tolist()
    for t, h, w in grids:
        if t != 1 or h <= 0 or w <= 0 or h % merge_size or w % merge_size:
            raise ValueError(
                "Image grids require T=1 and positive H/W divisible by spatial_merge_size"
            )
    return grids


def video_grid(grid_thw, merge_size):
    """Temporal grids retain frame boundaries in the visual attention sequence."""
    if (
        not isinstance(grid_thw, torch.Tensor)
        or grid_thw.ndim != 2
        or grid_thw.shape[1] != 3
        or grid_thw.shape[0] == 0
    ):
        raise ValueError("video_grid_thw must be nonempty [videos, 3]")
    if grid_thw.dtype not in (torch.int32, torch.int64):
        raise ValueError("video_grid_thw must have integer dtype")
    grids = grid_thw.tolist()
    if any(t <= 0 or h <= 0 or w <= 0 or h % merge_size or w % merge_size for t, h, w in grids):
        raise ValueError("Video grids require positive T/H/W and spatially divisible H/W")
    return grids


def vision_geometry(grid_thw, merge_size, side, *, video=False):
    """Block-major axial positions and align-corners bilinear embedding taps."""
    coords, indices, weights, lengths = [], [], [], []
    device = grid_thw.device
    grids = video_grid(grid_thw, merge_size) if video else image_grid(grid_thw, merge_size)
    for t, h, w in grids:
        row, col = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        pos = torch.stack([v.reshape(shape).transpose(1, 2).flatten() for v in (row, col)], dim=-1)
        coords.append(pos.repeat(t, 1))
        taps, fractions = [], []
        for axis, size in enumerate((h, w)):
            source = pos[:, axis].float() * (side - 1) / max(size - 1, 1)
            low = source.floor().long()
            taps.append(torch.stack((low, (low + 1).clamp(max=side - 1)), dim=1))
            delta = source - low
            fractions.append(torch.stack((1 - delta, delta), dim=1))
        indices.append((taps[0][:, :, None] * side + taps[1][:, None, :]).flatten(1).repeat(t, 1))
        weights.append(
            (fractions[0][:, :, None] * fractions[1][:, None, :]).flatten(1).repeat(t, 1)
        )
        lengths.extend([h * w] * t)
    return torch.cat(coords), torch.cat(indices), torch.cat(weights), tuple(lengths)


class Qwen3OmniMoeVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel, stride=kernel, bias=True)

    def forward(self, pixels):
        patches = pixels.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(patches.to(self.proj.weight.dtype)).reshape(-1, self.embed_dim)


class Qwen3OmniMoeVisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_heads // 2
        self.base = config.rope_parameters["rope_theta"]
        if config.rope_parameters["rope_type"] != "axial":
            raise ValueError("Native vision supports axial RoPE only")

    @torch.no_grad()
    def forward(self, positions):
        inv = self.base ** (
            -torch.arange(0, self.dim, 2, device=positions.device, dtype=torch.float32) / self.dim
        )
        angles = (positions.float()[..., None] * inv).flatten(1)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()


class Qwen3OmniMoeVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_implementation = getattr(config, "_attn_implementation", None) or "sdpa"
        if self.attn_implementation not in {"eager", "sdpa"}:
            raise NotImplementedError(
                f"Native vision attention does not implement {self.attn_implementation!r}"
            )

    def forward(self, hidden_states, lengths, position_embeddings):
        q, k, v = self.qkv(hidden_states).reshape(-1, 3, self.num_heads, self.head_dim).unbind(1)
        cos, sin = (value[:, None].float() for value in position_embeddings)

        def rotate(value):
            dtype = value.dtype
            # Share one FP32 cast across both branches, as HF does. Two casts
            # would separately round their backward contributions to BF16.
            value = value.float()
            first, second = value.chunk(2, dim=-1)
            return (value * cos + torch.cat((-second, first), -1) * sin).to(dtype)

        q, k = rotate(q), rotate(k)
        outputs = []
        for queries, keys, values in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
            if self.attn_implementation == "eager":
                # Honor explicit HF eager semantics, including intermediate
                # rounding. The default production path below remains SDPA.
                queries, keys, values = (
                    tensor.transpose(0, 1).unsqueeze(0) for tensor in (queries, keys, values)
                )
                scores = torch.matmul(queries, keys.transpose(2, 3)) * self.head_dim**-0.5
                probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(queries.dtype)
                outputs.append(
                    torch.matmul(probabilities, values).transpose(1, 2).contiguous().squeeze(0)
                )
                continue
            outputs.append(
                F.scaled_dot_product_attention(
                    # Match HF's [batch=1, heads, tokens, dim] SDPA layout.
                    # A rank-3 input can select a different SDPA implementation.
                    queries.transpose(0, 1).unsqueeze(0),
                    keys.transpose(0, 1).unsqueeze(0),
                    values.transpose(0, 1).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        return self.proj(torch.cat(outputs).reshape(hidden_states.shape))


class Qwen3OmniMoeVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        activations = {
            "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "relu": nn.ReLU(),
        }
        if config.hidden_act not in activations:
            raise ValueError(f"Unsupported vision activation: {config.hidden_act}")
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = activations[config.hidden_act]

    def forward(self, hidden_states):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen3OmniMoeVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Epsilon 1e-6 is architectural in the pinned HF vision implementation.
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3OmniMoeVisionAttention(config)
        self.mlp = Qwen3OmniMoeVisionMLP(config)

    def forward(self, hidden_states, lengths, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), lengths, position_embeddings
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3OmniMoeVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size),
        )

    def forward(self, hidden_states):
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.reshape(-1, self.hidden_size)
        return self.mlp(self.ln_q(hidden_states).reshape(-1, self.hidden_size))


class Qwen3OmniMoeVisionEncoder(nn.Module):
    input_modalities = ("image",)
    supports_video = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_heads or (config.hidden_size // config.num_heads) % 4:
            raise ValueError("Vision head dimension must be divisible by four")
        self.side = math.isqrt(config.num_position_embeddings)
        if self.side**2 != config.num_position_embeddings:
            raise ValueError("Vision position embeddings must form a square")
        indexes = tuple(config.deepstack_visual_indexes)
        if indexes != tuple(sorted(set(indexes))) or any(
            i < 0 or i >= config.depth for i in indexes
        ):
            raise ValueError("DeepStack indexes must be unique, ascending vision block indexes")
        if any(
            type(v) is not int or v <= 0
            for v in (
                config.patch_size,
                config.temporal_patch_size,
                config.spatial_merge_size,
                config.depth,
            )
        ):
            raise ValueError("Vision patch sizes, merge size and depth must be positive integers")
        self.deepstack_visual_indexes = indexes
        self.patch_embed = Qwen3OmniMoeVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = Qwen3OmniMoeVisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([Qwen3OmniMoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3OmniMoeVisionPatchMerger(config)
        self.merger_list = nn.ModuleList(
            [Qwen3OmniMoeVisionPatchMerger(config, use_postshuffle_norm=True) for _ in indexes]
        )

    def forward(self, pixel_values, grid_thw):
        config = self.config
        grids = (video_grid if self.supports_video else image_grid)(
            grid_thw, config.spatial_merge_size
        )
        patches = sum(t * h * w for t, h, w in grids)
        width = config.in_channels * config.temporal_patch_size * config.patch_size**2
        if pixel_values.shape != (patches, width):
            raise ValueError(
                f"pixel_values shape must be {(patches, width)}, got {tuple(pixel_values.shape)}"
            )
        if not pixel_values.is_floating_point() or not torch.isfinite(pixel_values).all():
            raise ValueError("pixel_values must be finite floating point values")
        positions, indices, weights, lengths = vision_geometry(
            grid_thw.to(pixel_values.device),
            config.spatial_merge_size,
            self.side,
            video=self.supports_video,
        )
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + (self.pos_embed(indices) * weights[..., None]).sum(1).to(hidden.dtype)
        rotary = self.rotary_pos_emb(positions)
        deepstack = []
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, lengths, rotary)
            if i in self.deepstack_visual_indexes:
                deepstack.append(self.merger_list[self.deepstack_visual_indexes.index(i)](hidden))
        return VisionOutput(hidden, self.merger(hidden), tuple(deepstack))


class Qwen3OmniMoeVideoEncoder(Qwen3OmniMoeVisionEncoder):
    """Same HF visual parameters; temporal patches attend within each frame."""

    input_modalities = ("video",)
    supports_video = True
