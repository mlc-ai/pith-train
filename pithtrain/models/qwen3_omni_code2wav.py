# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native differentiable Qwen3-Omni codec-to-waveform decoder.

Adapted from Transformers 5.17's Qwen3-Omni implementation. Computation and
streaming are native PyTorch; no Hugging Face model is constructed. Parameters
retain the released 26291f7 checkpoint names. This dense audio component is
independent of the MoE text pipeline and does not encode waveforms into codes.
"""

import math
from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import nn


def validate_code2wav_config(config):
    positive = (
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "num_hidden_layers",
        "num_quantizers",
        "codebook_size",
        "decoder_dim",
        "sliding_window",
    )
    if any(
        type(getattr(config, name)) is not int or getattr(config, name) <= 0 for name in positive
    ):
        raise ValueError("Code2wav dimensions and sliding window must be positive integers")
    for name in ("upsample_rates", "upsampling_ratios"):
        rates = getattr(config, name)
        if not rates or any(type(rate) is not int or rate <= 0 for rate in rates):
            raise ValueError("Code2wav upsampling rates must be nonempty positive integers")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if (
        head_dim <= 0
        or head_dim % 2
        or config.hidden_size % config.num_attention_heads
        or config.num_attention_heads % config.num_key_value_heads
    ):
        raise ValueError("Code2wav requires even head dimensions and divisible attention heads")
    if config.decoder_dim % (2 ** len(config.upsample_rates)):
        raise ValueError("Code2wav decoder_dim must divide into all decoder channels")
    if config.hidden_act not in {"silu", "gelu", "gelu_pytorch_tanh", "relu"}:
        raise ValueError(f"Unsupported code2wav activation: {config.hidden_act}")
    rope = getattr(config, "rope_parameters", None) or {}
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError("Code2wav currently implements the released default RoPE")
    theta = rope.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("Code2wav RoPE theta must be finite and positive")
    if not 0 <= config.attention_dropout < 1:
        raise ValueError("Code2wav attention dropout must be in [0, 1)")


class Qwen3OmniMoeCausalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - stride

    def forward(self, hidden):
        length = hidden.shape[-1]
        frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal = (math.ceil(frames) - 1) * self.stride + self.kernel_size - self.padding
        return self.conv(F.pad(hidden, (self.padding, ideal - length))).contiguous()


class Qwen3OmniMoeCausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.left_pad = self.right_pad = math.ceil(kernel_size - stride)

    def forward(self, hidden):
        hidden = self.conv(hidden)
        return hidden[..., self.left_pad : hidden.shape[-1] - self.right_pad].contiguous()


class Qwen3OmniMoeConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = Qwen3OmniMoeCausalConvNet(dim, dim, 7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Fixed ConvNeXt initialization in the upstream architecture.
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden):
        residual = hidden
        hidden = self.norm(self.dwconv(hidden).permute(0, 2, 1))
        hidden = self.pwconv2(self.act(self.pwconv1(hidden))) * self.gamma
        return residual + hidden.permute(0, 2, 1)


class Qwen3OmniMoeCode2WavRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden):
        dtype = hidden.dtype
        hidden = hidden.float()
        hidden = hidden * torch.rsqrt(
            hidden.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * hidden.to(dtype)


class Qwen3OmniMoeCode2WavLayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_initial_scale)
        )

    def forward(self, hidden):
        return self.scale * hidden


class Qwen3OmniMoeCode2WavAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.sliding_window = config.sliding_window
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(self, hidden, rotary):
        batch, length, _ = hidden.shape
        q = self.q_norm(
            self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(hidden)
            .view(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos, sin = (x.unsqueeze(1) for x in rotary)
        q_half = torch.cat((-q[..., self.head_dim // 2 :], q[..., : self.head_dim // 2]), dim=-1)
        k_half = torch.cat((-k[..., self.head_dim // 2 :], k[..., : self.head_dim // 2]), dim=-1)
        q, k = q * cos + q_half * sin, k * cos + k_half * sin
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        positions = torch.arange(length, device=hidden.device)
        distance = positions[:, None] - positions[None, :]
        allowed = (distance >= 0) & (distance < self.sliding_window)
        mask = torch.zeros((length, length), dtype=q.dtype, device=q.device)
        mask.masked_fill_(~allowed, torch.finfo(q.dtype).min)
        weights = (q @ k.transpose(-1, -2)) * self.scaling + mask
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, -1)
        return self.o_proj(output)


class Qwen3OmniMoeCode2WavMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = config.hidden_act

    def forward(self, hidden):
        gate = self.gate_proj(hidden)
        if self.activation == "silu":
            gate = F.silu(gate)
        elif self.activation == "relu":
            gate = F.relu(gate)
        else:
            gate = F.gelu(
                gate, approximate="tanh" if self.activation == "gelu_pytorch_tanh" else "none"
            )
        return self.down_proj(gate * self.up_proj(hidden))


class Qwen3OmniMoeCode2WavTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3OmniMoeCode2WavAttention(config)
        self.mlp = Qwen3OmniMoeCode2WavMlp(config)
        self.input_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3OmniMoeCode2WavRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)
        self.mlp_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)

    def forward(self, hidden, rotary):
        hidden = hidden + self.self_attn_layer_scale(
            self.self_attn(self.input_layernorm(hidden), rotary)
        )
        return hidden + self.mlp_layer_scale(self.mlp(self.post_attention_layernorm(hidden)))


class Qwen3OmniMoeCode2WavTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            Qwen3OmniMoeCode2WavTransformerLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        self.theta = (getattr(config, "rope_parameters", None) or {}).get(
            "rope_theta", getattr(config, "rope_theta", 10000.0)
        )

    def forward(self, hidden):
        # Recompute constants in FP32 so meta/to_empty and BF16 moves cannot corrupt RoPE.
        device_type = hidden.device.type if hidden.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            inverse = 1.0 / (
                self.theta
                ** (torch.arange(0, self.head_dim, 2, device=hidden.device).float() / self.head_dim)
            )
            positions = torch.arange(hidden.shape[1], device=hidden.device).float()
            angles = (inverse[None, :, None] @ positions[None, None, :]).transpose(1, 2)
            angles = torch.cat((angles, angles), dim=-1)
            rotary = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, rotary)
        return self.norm(hidden)


class Qwen3OmniMoeSnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, hidden):
        alpha = self.alpha[None, :, None].exp()
        beta = self.beta[None, :, None].exp()
        # Upstream SnakeBeta epsilon, not a checkpoint configuration knob.
        return hidden + (1.0 / (beta + 1e-9)) * torch.sin(hidden * alpha).square()


class Qwen3OmniMoeCode2WavDecoderResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.act1 = Qwen3OmniMoeSnakeBeta(dim)
        self.conv1 = Qwen3OmniMoeCausalConvNet(dim, dim, 7, dilation=dilation)
        self.act2 = Qwen3OmniMoeSnakeBeta(dim)
        self.conv2 = Qwen3OmniMoeCausalConvNet(dim, dim, 1)

    def forward(self, hidden):
        return hidden + self.conv2(self.act2(self.conv1(self.act1(hidden))))


class Qwen3OmniMoeCode2WavDecoderBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = in_dim // 2
        rate = config.upsample_rates[layer_idx]
        self.block = nn.ModuleList(
            [
                Qwen3OmniMoeSnakeBeta(in_dim),
                Qwen3OmniMoeCausalTransConvNet(in_dim, out_dim, 2 * rate, rate),
                # Upstream decoder uses these three fixed dilation rates.
                *(Qwen3OmniMoeCode2WavDecoderResidualUnit(out_dim, d) for d in (1, 3, 9)),
            ]
        )

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)
        return hidden


class Qwen3OmniMoeCode2Wav(nn.Module):
    """Decode integer codecs [batch, quantizer, time] to waveforms [batch, 1, samples]."""

    def __init__(self, config):
        super().__init__()
        validate_code2wav_config(config)
        self.config = config
        self.total_upsample = math.prod(
            list(config.upsample_rates) + list(config.upsampling_ratios)
        )
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel(config)
        self.code_embedding = nn.Embedding(
            config.codebook_size * config.num_quantizers, config.hidden_size
        )
        self.upsample = nn.ModuleList(
            nn.ModuleList(
                [
                    Qwen3OmniMoeCausalTransConvNet(
                        config.hidden_size, config.hidden_size, factor, factor
                    ),
                    Qwen3OmniMoeConvNeXtBlock(config.hidden_size),
                ]
            )
            for factor in config.upsampling_ratios
        )
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        self.decoder = nn.ModuleList(
            [
                Qwen3OmniMoeCausalConvNet(config.hidden_size, config.decoder_dim, 7),
                *(
                    Qwen3OmniMoeCode2WavDecoderBlock(config, i)
                    for i in range(len(config.upsample_rates))
                ),
                Qwen3OmniMoeSnakeBeta(output_dim),
                Qwen3OmniMoeCausalConvNet(output_dim, 1, 7),
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(
                module.weight, std=getattr(self.config, "initializer_range", 0.02) or 0.02
            )
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _validate_codes(self, codes, *, values=True, empty_time=False):
        if not isinstance(codes, torch.Tensor) or codes.dtype != torch.long or codes.ndim != 3:
            raise ValueError("Code2wav requires int64 codes [batch, quantizer, time]")
        if (
            codes.shape[0] == 0
            or codes.shape[1] != self.config.num_quantizers
            or (codes.shape[2] == 0 and not empty_time)
        ):
            raise ValueError(
                "Code2wav requires nonempty batch/time and the configured quantizer count"
            )
        if values and (torch.any(codes < 0) or torch.any(codes >= self.config.codebook_size)):
            raise ValueError(
                "Code2wav codec ids must be within the configured codebook, without EOS/padding"
            )

    def output_length(self, code_length):
        """Exact length of one forward call, including each transposed-convolution crop."""
        if type(code_length) is not int or code_length <= 0:
            raise ValueError("code_length must be a positive integer")
        length = code_length * math.prod(self.config.upsampling_ratios)
        for rate in self.config.upsample_rates:
            length = (length - 1) * rate
        return length

    def forward(self, codes):
        self._validate_codes(codes)
        if self.output_length(codes.shape[-1]) <= 0:
            raise ValueError("Code sequence is too short for the configured decoder")
        # Rebuild the nonpersistent integer offset for safe meta/to_empty initialization.
        offset = (
            torch.arange(self.config.num_quantizers, device=codes.device)[None, :, None]
            * self.config.codebook_size
        )
        hidden = self.code_embedding(codes + offset).mean(1)
        hidden = self.pre_transformer(hidden).permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        for block in self.decoder:
            hidden = block(hidden)
        return hidden.clamp(min=-1, max=1)

    def decode_stream(
        self, chunks: Iterable[torch.Tensor], left_context_size=25
    ) -> Iterator[torch.Tensor]:
        """Yield causal audio chunks, retaining only the specified number of prior codes.

        Chunk boundaries are significant: this reproduces HF's recompute/crop
        policy, not a full-forward-equivalent convolution/KV cache. It remains
        differentiable; callers choose inference_mode when gradients are unwanted.
        """
        if type(left_context_size) is not int or left_context_size < 0:
            raise ValueError("left_context_size must be a nonnegative integer")
        context = None
        for codes in chunks:
            self._validate_codes(codes, empty_time=True)
            if context is not None and (
                context.shape[:2] != codes.shape[:2] or context.device != codes.device
            ):
                raise ValueError(
                    "All streaming chunks must use the same batch, quantizers and device"
                )
            if codes.shape[-1] == 0:
                continue
            count = 0 if context is None else context.shape[-1]
            combined = codes if context is None else torch.cat((context, codes), dim=-1)
            wav = self(combined)[..., count * self.total_upsample :]
            context = (
                combined[..., -left_context_size:].clone()
                if left_context_size
                else codes[..., :0].clone()
            )
            yield wav

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        """Batched HF-compatible chunk decoding; output retains per-chunk crop losses."""
        self._validate_codes(codes)
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        chunks = (
            codes[..., start : start + chunk_size]
            for start in range(0, codes.shape[-1], chunk_size)
        )
        return torch.cat(list(self.decode_stream(chunks, left_context_size)), dim=-1)

    def decode_batch(self, codes, lengths, chunk_size=300, left_context_size=25):
        """Decode ragged padded codecs, returning one [1, samples] tensor per item.

        Lengths exclude EOS/padding. Padding values are never embedded. A zero
        length returns empty audio and does not invoke the model for that row.
        """
        self._validate_codes(codes, values=False, empty_time=True)
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.dtype != torch.long
            or lengths.shape != (codes.shape[0],)
            or torch.any(lengths < 0)
            or torch.any(lengths > codes.shape[-1])
        ):
            raise ValueError("lengths must contain one int64 valid codec length per batch item")
        if (
            type(chunk_size) is not int
            or chunk_size <= 0
            or type(left_context_size) is not int
            or left_context_size < 0
        ):
            raise ValueError("Invalid chunk size or left context")
        return [
            self.chunked_decode(codes[i : i + 1, :, :length], chunk_size, left_context_size)[0]
            if length
            else self.code_embedding.weight.new_empty((1, 0))
            for i, length in enumerate(lengths.tolist())
        ]
