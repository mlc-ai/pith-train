"""Dense Qwen3.5 text tower, including checkpoints published as Qwen3.8-27B."""

import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from pithtrain.contexts import distributed, training
from pithtrain.kernels import stage_compile_options
from pithtrain.models.qwen35_moe import (
    Qwen35MoeAttention,
    Qwen35MoeGatedDeltaNet,
    Qwen35MoeModel,
    Qwen35MoeRMSNorm,
    Qwen35MoeRotaryEmbedding,
)
from pithtrain.operators.cp_sequence import contiguous_to_zigzag, zigzag_to_contiguous
from pithtrain.operators.linear import BF16Linear
from pithtrain.operators.silu_mul import silu_mul

__all__ = ["Qwen35Model", "Qwen35TextModel"]


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported Qwen3.5 MLP activation: {config.hidden_act}")
        self.gate_proj = training.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = training.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = training.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_x = training.Linear.quantize_input(x)
        gate = self.gate_proj(x, quantized_x)
        up = self.up_proj(x, quantized_x)
        return self.down_proj(silu_mul(gate, up))


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen35MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen35MoeAttention(config)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        layer_types = config.layer_types
        linear_at = lambda i: 0 <= i < len(layer_types) and layer_types[i] == "linear_attention"
        sharded = distributed.cp_size > 1
        self.to_contiguous = sharded and self.is_linear and not linear_at(layer_idx - 1)
        self.to_zigzag = sharded and self.is_linear and not linear_at(layer_idx + 1)

    @torch.compile(fullgraph=True, options=stage_compile_options() or None)
    def forward_stage1(self, hidden_states, rotary_posemb, cu_seqlens=None):
        if self.to_contiguous:
            hidden_states = zigzag_to_contiguous(hidden_states, distributed.cp_group)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear:
            hidden_states = self.linear_attn(hidden_states, cu_seqlens)
        else:
            hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
        residual = residual + hidden_states
        return self.post_attention_layernorm(residual), residual, None

    def forward_stage3(
        self, gathered_tokens, expert_idxs=None, expand_idx=None, expert_counts=None
    ):
        return self.mlp(gathered_tokens)

    @torch.compile(fullgraph=True, options=stage_compile_options() or None)
    def forward_stage5(self, moe_outs, moe_local_idxs, topk_weight, residual):
        hidden_states = residual + moe_outs
        if self.to_zigzag:
            hidden_states = contiguous_to_zigzag(hidden_states, distributed.cp_group)
        return hidden_states

    def reference_forward(self, hidden_states, rotary_posemb, cu_seqlens=None):
        return self.forward(hidden_states, rotary_posemb, cu_seqlens, checkpoint_delta=True)

    def forward_stock_stages(self, hidden_states, rotary_posemb, cu_seqlens=None):
        hidden_states, residual, _ = self.forward_stage1(hidden_states, rotary_posemb, cu_seqlens)
        hidden_states = self.forward_stage3(hidden_states)
        return self.forward_stage5(hidden_states, None, None, residual)

    def forward(self, hidden_states, rotary_posemb, cu_seqlens=None, *, checkpoint_delta=False):
        if self.to_contiguous:
            hidden_states = zigzag_to_contiguous(hidden_states, distributed.cp_group)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear:
            delta = self.linear_attn if checkpoint_delta else self.linear_attn.block
            hidden_states = delta(hidden_states, cu_seqlens)
        else:
            hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if self.to_zigzag:
            hidden_states = contiguous_to_zigzag(hidden_states, distributed.cp_group)
        return hidden_states


class Qwen35Model(Qwen35MoeModel):
    """Reuse hybrid attention and pipeline plumbing; no routed or shared experts are created."""

    decoder_layer_class = Qwen35DecoderLayer


class Qwen35TextModel(nn.Module):
    """The complete dense text tower with a standard autograd forward."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = Qwen35MoeRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            Qwen35DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = BF16Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(tokens)
        rotary_posemb = tuple(t.unsqueeze(0) for t in self.rotary_emb(tokens.size(1)))
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_posemb)
        return self.lm_head(self.norm(hidden_states))
