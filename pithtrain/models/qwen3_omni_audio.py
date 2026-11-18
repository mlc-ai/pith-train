"""Native Qwen3-Omni audio encoder, matching Transformers 5.17 (Apache-2.0).

All computation is PyTorch. Transformers is used only by the separate parity
tests. The released encoder uses 100-frame convolution chunks, independent
attention windows, and per-chunk sinusoidal positions.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class AudioOutput(NamedTuple):
    last_hidden_state: torch.Tensor


def audio_output_lengths(lengths, n_window=50):
    """Released HF length rule, including the separately convolved tail."""
    if n_window != 50:
        raise ValueError("Native audio supports the released n_window=50 configuration")
    return lengths // 100 * 13 + (lengths % 100 + 7) // 8


def validate_audio_config(config):
    positive = (
        "num_mel_bins",
        "d_model",
        "encoder_attention_heads",
        "encoder_ffn_dim",
        "encoder_layers",
        "output_dim",
        "max_source_positions",
        "conv_chunksize",
        "downsample_hidden_size",
    )
    if any(
        type(getattr(config, name)) is not int or getattr(config, name) <= 0 for name in positive
    ):
        raise ValueError("Audio encoder dimensions and chunk sizes must be positive integers")
    if config.d_model < 4 or config.d_model % 2 or config.d_model % config.encoder_attention_heads:
        raise ValueError(
            "Audio d_model must be even, at least four, and divisible by attention heads"
        )
    if config.n_window != 50:
        raise ValueError("Native audio supports the released n_window=50 configuration")
    if (
        type(config.n_window_infer) is not int
        or config.n_window_infer < 100
        or config.n_window_infer % 100
        or config.max_source_positions < 13
    ):
        raise ValueError(
            "Audio inference windows must be positive multiples of 100; positions need 13 slots"
        )
    if config.dropout or config.attention_dropout or config.activation_dropout:
        raise ValueError("Native audio requires zero dropout")
    if config.activation_function not in {"gelu", "gelu_pytorch_tanh", "silu", "relu"}:
        raise ValueError(f"Unsupported audio activation: {config.activation_function}")


def activation(name):
    return {
        "gelu": nn.GELU(),
        "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
    }[name]


def audio_layout(input_features, feature_lens, config):
    """Validate flattened mel features and derive convolution/attention boundaries."""
    if (
        feature_lens.ndim != 1
        or feature_lens.numel() == 0
        or feature_lens.dtype not in (torch.int32, torch.int64)
        or not (feature_lens > 0).all()
    ):
        raise ValueError("feature_lens must be a nonempty positive integer vector")
    lengths = feature_lens.tolist()
    if (
        input_features.shape != (config.num_mel_bins, sum(lengths))
        or not input_features.is_floating_point()
        or not torch.isfinite(input_features).all()
    ):
        raise ValueError("input_features must be finite [mel_bins, sum(feature_lens)]")
    chunks = [min(100, length - offset) for length in lengths for offset in range(0, length, 100)]
    padded = nn.utils.rnn.pad_sequence(input_features.T.split(chunks), batch_first=True).transpose(
        1, 2
    )
    cnn_lengths = [(size + 7) // 8 for size in chunks]
    stride = max(cnn_lengths)
    valid = torch.cat(
        [
            torch.arange(size, device=input_features.device) + i * stride
            for i, size in enumerate(cnn_lengths)
        ]
    )
    window = stride * (config.n_window_infer // 100)
    outputs = audio_output_lengths(feature_lens).tolist()
    windows = tuple(
        min(window, size - offset) for size in outputs for offset in range(0, size, window)
    )
    return padded, valid, windows


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels):
        super().__init__()
        self.length, self.channels = length, channels

    def forward(self, length, device, dtype):
        # Rebuild from constants so meta/to_empty initialization and dtype moves
        # cannot destroy or prematurely round this nonpersistent positional table.
        increment = math.log(10000) / (self.channels // 2 - 1)
        inverse = torch.exp(-increment * torch.arange(self.channels // 2, device=device).float())
        angles = torch.arange(length, device=device)[:, None] * inverse[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=1).to(dtype)


class Qwen3OmniMoeAudioAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.d_model // self.num_heads
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden_states, lengths):
        shape = (-1, self.num_heads, self.head_dim)
        q, k, v = (
            projection(hidden_states).reshape(shape)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        outputs = []
        for queries, keys, values in zip(
            q.split(lengths), k.split(lengths), v.split(lengths), strict=True
        ):
            queries, keys, values = (
                tensor.transpose(0, 1).unsqueeze(0) for tensor in (queries, keys, values)
            )
            # Preserve the reference's scaling/FP32-softmax arithmetic. Reordering
            # via SDPA changes the near-zero key-bias gradients even on CPU.
            scores = torch.matmul(queries, keys.transpose(2, 3)) * self.head_dim**-0.5
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(queries.dtype)
            outputs.append(torch.matmul(probabilities, values).transpose(1, 2).contiguous())
        return self.out_proj(torch.cat(outputs, dim=1).reshape(hidden_states.shape))


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3OmniMoeAudioAttention(config)
        # HF audio uses the PyTorch LayerNorm default epsilon (1e-5).
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.activation_fn = activation(config.activation_function)

    def forward(self, hidden_states, lengths):
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states), lengths
        )
        hidden_states = hidden_states + self.fc2(
            self.activation_fn(self.fc1(self.final_layer_norm(hidden_states)))
        )
        if hidden_states.dtype == torch.float16:
            bound = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = hidden_states.clamp(-bound, bound)
        return hidden_states


class Qwen3OmniMoeAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_audio_config(config)
        self.config = config
        hidden = config.downsample_hidden_size
        self.positional_embedding = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model
        )
        self.layers = nn.ModuleList(
            Qwen3OmniMoeAudioEncoderLayer(config) for _ in range(config.encoder_layers)
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.conv2d1 = nn.Conv2d(1, hidden, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(hidden, hidden, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            hidden * ((config.num_mel_bins + 7) // 8), config.d_model, bias=False
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = activation(config.activation_function)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def forward(self, input_features, feature_lens):
        padded, valid, lengths = audio_layout(input_features, feature_lens, self.config)
        padded = padded.unsqueeze(1).to(self.conv2d1.weight.dtype)
        convolved = []
        for chunk in padded.split(self.config.conv_chunksize):
            hidden = F.gelu(self.conv2d1(chunk))
            hidden = F.gelu(self.conv2d2(hidden))
            convolved.append(F.gelu(self.conv2d3(hidden)))
        hidden = torch.cat(convolved)
        batch, channels, frequency, steps = hidden.shape
        hidden = self.conv_out(
            hidden.permute(0, 3, 1, 2).contiguous().view(batch, steps, channels * frequency)
        )
        hidden = hidden + self.positional_embedding(steps, hidden.device, hidden.dtype)[None]
        hidden = hidden.reshape(-1, hidden.shape[-1]).index_select(0, valid)
        for layer in self.layers:
            hidden = layer(hidden, lengths)
        hidden = self.proj2(self.act(self.proj1(self.ln_post(hidden))))
        return AudioOutput(hidden)
