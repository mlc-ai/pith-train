"""
Qwen/Qwen3-30B-A3B and Qwen/Qwen3-235B-A22B.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from pithtrain.contexts import distributed, training
from pithtrain.kernels import (
    bf16_experts_supported,
    fp8_fused_experts_supported,
    fp8_quant_grouped,
)
from pithtrain.models.interface import RoutingInfo
from pithtrain.modules.load_balance import (
    MoELoadBalanceLossInjector,
    MoELoadBalanceLossTracker,
    replay_indices,
)
from pithtrain.modules.recompute import norm_recompute
from pithtrain.operators.cp_sequence import zigzag_spans, zigzag_varlen_index
from pithtrain.operators.ep_dispatch import prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from pithtrain.operators.fp8_experts import fp8_scatter_quant, fp8_swiglu_experts
from pithtrain.operators.linear import BF16Linear
from pithtrain.operators.ring_attention import ring_attention_func, ring_attention_varlen_func
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)
from pithtrain.pipeline.dualpipev import layer_partition
from pithtrain.pipeline.execution import ChunkRecord, model_forward

__all__ = ["Qwen3MoeGate", "Qwen3MoeModel"]


class Qwen3MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        inv_freq = self.compute_rope_params(config)
        self.set_cos_sin(config, inv_freq)

    def compute_rope_params(self, config: Qwen3MoeConfig) -> torch.Tensor:
        base, dim = config.rope_scaling["rope_theta"], config.head_dim
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def set_cos_sin(self, config: Qwen3MoeConfig, inv_freq: torch.Tensor) -> None:
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin", emb.sin().to(torch.bfloat16), persistent=False)

    def forward(self, S: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:S], self.sin[:S]


class Qwen3MoeExperts(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, num_experts: int):
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        self.top_k = getattr(config, "num_experts_per_tok", num_experts)
        self.gate_proj = training.GroupedLinear(num_experts, hidden_size, moe_intermediate_size)
        self.up_proj = training.GroupedLinear(num_experts, hidden_size, moe_intermediate_size)
        self.down_proj = training.GroupedLinear(num_experts, moe_intermediate_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        group_indices = precompute_group_indices(grouped_mm_offs, x.shape[0])
        if training.fp8 and fp8_fused_experts_supported() and x.shape[0]:
            args = (x, grouped_mm_offs, ks, ks_tensor, group_indices, quantized_input)
            return fp8_swiglu_experts(self.gate_proj, self.up_proj, self.down_proj, *args)
        kwargs = dict(grouped_mm_offs=grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=group_indices)  # fmt: skip
        use_shared_quant = training.fp8 and x.shape[0]
        quantized_x = fp8_quant_grouped(x.detach(), grouped_mm_offs) if use_shared_quant else None  # fmt: skip
        gate = self.gate_proj(x, **kwargs, quantized_input=quantized_x)
        up = self.up_proj(x, **kwargs, quantized_input=quantized_x)
        return self.down_proj(silu_mul(gate, up), **kwargs)

    def uses_bf16_cute(self) -> bool:
        return not training.fp8 and bf16_experts_supported(self.gate_proj.num_groups)

    def forward_from_tokens(
        self,
        gathered: torch.Tensor,
        expert_idxs: torch.Tensor,
        expand_idx: torch.Tensor | None,
        expert_counts: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Stage 3 from the received tokens on Hopper: returns (expert outputs, reverse_shuffle_idxs),
        or the outputs in slot order with None for the BF16 CuTe experts. A method of its own,
        registered with FSDP, so the FP8 input never meets FSDP's bf16 input cast.
        """
        if not training.fp8:
            from pithtrain.operators.bf16_experts import bf16_swiglu_experts

            args = (gathered, expert_idxs, expand_idx, self.top_k)
            return bf16_swiglu_experts(self.gate_proj, self.up_proj, self.down_proj, *args), None
        G = self.gate_proj.num_groups
        x, reverse_shuffle_idxs, offs, ks, ks_tensor, quantized_input = fp8_scatter_quant(gathered, expert_idxs, G, expand_idx, expert_counts)  # fmt: skip
        # self.forward, not self(...): calling the module re-enters FSDP's input cast.
        return self.forward(x, offs, ks=ks, ks_tensor=ks_tensor, quantized_input=quantized_input), reverse_shuffle_idxs  # fmt: skip


class Qwen3MoeGate(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.load_balance_loss_fn = None
        self.router_replay = None
        self.weight = nn.Parameter(torch.empty((self.num_experts, config.hidden_size)), requires_grad=True)  # fmt: skip

    def forward(
        self, hidden_states: torch.Tensor, replay_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        if replay_idx is not None:
            topk_idx = replay_idx
            topk_weight = scores.gather(-1, topk_idx)
        if self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        if self.load_balance_loss_fn is None:
            return topk_idx, topk_weight, None
        lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.num_experts, self.num_experts_per_tok)  # fmt: skip
        # Token-weight the injected lb gradient so train_step's 1/num_tokens grad scale leaves it
        # correctly normalized (it bypasses the token-weighted objective). lb_loss stays unscaled.
        topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss * topk_weight.shape[0])
        return topk_idx, topk_weight, lb_loss


class Qwen3MoeMoE(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts_per_rank = config.num_experts // distributed.ep_size
        self.experts = Qwen3MoeExperts(config, self.experts_per_rank)
        self.gate = Qwen3MoeGate(config)

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        replay_idx = replay_indices(self.gate, hidden_states, self.num_experts_per_tok)
        topk_idx, topk_weight, lb_loss = self.gate(hidden_states, replay_idx)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        expert_idxs = topk_idx.view(-1)
        replicated_tokens = hidden_states.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, hidden_states.shape[-1])  # fmt: skip
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(replicated_tokens, expert_idxs, self.experts_per_rank)  # fmt: skip
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]
        y = (outs.view(*topk_idx.shape, -1) * topk_weight.unsqueeze(dim=-1)).sum(dim=1).to(outs.dtype)  # fmt: skip
        return y.view(*orig_shape)


class Qwen3MoeAttention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        attention_bias = config.attention_bias
        self.q_proj = training.Linear(hidden_size, self.num_heads * self.head_dim, bias=attention_bias)  # fmt: skip
        self.k_proj = training.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)  # fmt: skip
        self.v_proj = training.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)  # fmt: skip
        self.o_proj = training.Linear(self.num_heads * self.head_dim, hidden_size, bias=attention_bias)  # fmt: skip
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_posemb(
        q: torch.Tensor, k: torch.Tensor, rotary_posemb: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = rotary_posemb
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        q_embed = (q * cos) + (Qwen3MoeAttention.rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3MoeAttention.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.size()
        quantized_hidden = training.Linear.quantize_input(hidden_states)
        query_states = self.q_proj(hidden_states, quantized_hidden).view(B, S, self.num_heads, self.head_dim)  # fmt: skip
        key_states = self.k_proj(hidden_states, quantized_hidden).view(B, S, self.num_kv_heads, self.head_dim)  # fmt: skip
        value_states = self.v_proj(hidden_states, quantized_hidden).view(B, S, self.num_kv_heads, self.head_dim)  # fmt: skip
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = self.apply_rotary_posemb(query_states, key_states, rotary_posemb)
        if distributed.cp_size > 1 and cu_seqlens is not None:
            attn_output = ring_attention_varlen_func(query_states.squeeze(0), key_states.squeeze(0), value_states.squeeze(0), cu_seqlens, sm_scale=self.scaling, cp_group=distributed.cp_group).unsqueeze(0)  # fmt: skip
        elif distributed.cp_size > 1:
            attn_output = ring_attention_func(query_states, key_states, value_states, sm_scale=self.scaling, cp_group=distributed.cp_group)  # fmt: skip
        elif cu_seqlens is not None:
            attn_output = flash_attn_varlen_func(query_states.squeeze(0), key_states.squeeze(0), value_states.squeeze(0), cu_seqlens, S, softmax_scale=self.scaling, causal=True).unsqueeze(0)  # fmt: skip
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, softmax_scale=self.scaling, causal=True)  # fmt: skip
        attn_output = attn_output.reshape(B, S, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_id: int):
        super().__init__()
        self.idx = layer_id
        self.self_attn = Qwen3MoeAttention(config)
        self.mlp = Qwen3MoeMoE(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.compile(fullgraph=True, options={"cpp_wrapper": True})
    def forward_stage1_compute(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
        replay_idx: torch.Tensor | None = None,
    ):
        H = self.input_layernorm.normalized_shape[0]
        return norm_recompute(self.stage1_compute, H, hidden_states, rotary_posemb, cu_seqlens, replay_idx)  # fmt: skip

    def stage1_compute(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None,
        replay_idx: torch.Tensor | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        topk_idx, topk_weight, lb_loss = self.mlp.gate(hidden_states, replay_idx)
        return hidden_states, residual, topk_idx, topk_weight, lb_loss

    def forward_stage1(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingInfo | None]:
        replay_idx = replay_indices(self.mlp.gate, hidden_states, self.mlp.num_experts_per_tok)
        outs = self.forward_stage1_compute(hidden_states, rotary_posemb, cu_seqlens, replay_idx)
        return self.stage1_dispatch(*outs)

    def forward_stage5_stage1(
        self,
        prev_layer: "Qwen3MoeDecoderLayer",
        moe_outs: torch.Tensor,
        moe_local_idxs: torch.Tensor | None,
        topk_weight: torch.Tensor | None,
        residual: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingInfo | None]:
        """
        prev_layer.forward_stage5 then self.forward_stage1 as one compiled region, so each layer
        and micro-batch enters one compiled graph (and one compiled backward) instead of two.
        """
        replay_idx = replay_indices(self.mlp.gate, residual, self.mlp.num_experts_per_tok)
        outs = self.forward_stage5_stage1_compute(prev_layer, moe_outs, moe_local_idxs, topk_weight, residual, rotary_posemb, cu_seqlens, replay_idx)  # fmt: skip
        return self.stage1_dispatch(*outs)

    @torch.compile(fullgraph=True, options={"cpp_wrapper": True})
    def forward_stage5_stage1_compute(
        self,
        prev_layer: "Qwen3MoeDecoderLayer",
        moe_outs: torch.Tensor,
        moe_local_idxs: torch.Tensor | None,
        topk_weight: torch.Tensor | None,
        residual: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None,
        replay_idx: torch.Tensor | None,
    ):
        hidden_states = prev_layer.stage5_compute(moe_outs, moe_local_idxs, topk_weight, residual)
        H = self.input_layernorm.normalized_shape[0]
        return norm_recompute(self.stage1_compute, H, hidden_states, rotary_posemb, cu_seqlens, replay_idx)  # fmt: skip

    def stage1_dispatch(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        lb_loss: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingInfo | None]:
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        dispatch_tokens, routing = prepare_dispatch(hidden_states, topk_idx, topk_weight, self.mlp.num_experts, distributed.ep_size, self.mlp.experts_per_rank, distributed.ep_group)  # fmt: skip
        return dispatch_tokens, residual, routing

    def keeps_stage3_input(self) -> bool:
        """
        Whether forward_stage3 keeps gathered_tokens for the backward (the BF16 CuTe experts read
        them again in their weight gradient), so the pipeline must not free them.
        """
        return not training.fp8 and self.mlp.experts.uses_bf16_cute()

    def forward_stage3(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: torch.Tensor | None = None,
        expand_idx: torch.Tensor | None = None,
        expert_counts: list[int] | None = None,
    ) -> torch.Tensor:
        if training.fp8 and fp8_fused_experts_supported():
            expand = expand_idx if distributed.ep_size > 1 else None
            outs, reverse_shuffle_idxs = self.mlp.experts.forward_from_tokens(gathered_tokens, expert_idxs, expand, expert_counts)  # fmt: skip
            del gathered_tokens
            return padded_index_gather(outs, reverse_shuffle_idxs)
        if self.keeps_stage3_input():
            expand = expand_idx if distributed.ep_size > 1 else None
            return self.mlp.experts.forward_from_tokens(gathered_tokens, expert_idxs, expand, None)[0]  # fmt: skip
        if distributed.ep_size > 1:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank, group_counts=expert_counts)  # fmt: skip
        del gathered_tokens
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        return padded_index_gather(outs, reverse_shuffle_idxs)

    @torch.compile(fullgraph=True, options={"cpp_wrapper": True})
    def forward_stage5(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: torch.Tensor | None,
        topk_weight: torch.Tensor | None,
        residual: torch.Tensor,
    ):
        return self.stage5_compute(moe_outs, moe_local_idxs, topk_weight, residual)

    def stage5_compute(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: torch.Tensor | None,
        topk_weight: torch.Tensor | None,
        residual: torch.Tensor,
    ):
        if distributed.ep_size == 1:
            weighted = moe_outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)
            return residual + weighted.sum(dim=1).to(moe_outs.dtype).view(*residual.shape)
        permuted_probs = topk_weight.view(-1)[moe_local_idxs]
        token_indices = moe_local_idxs // topk_weight.shape[1]
        weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
        aggregated = moe_outs.new_zeros(topk_weight.shape[0], moe_outs.shape[-1])
        aggregated.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
        return residual + aggregated.view(*residual.shape)

    def reference_forward(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.reference_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, phase: int):
        super().__init__()
        match phase:
            case 0:
                stage_count = distributed.pp_size * 2
                stage_index = distributed.pp_rank
            case 1:
                stage_count = distributed.pp_size * 2
                stage_index = stage_count - 1 - distributed.pp_rank
            case -1:
                stage_count = 1
                stage_index = 0
            case _:
                raise ValueError("phase must be 0, 1, or -1, got %d" % phase)
        self.stage_index, self.stage_count = stage_index, stage_count
        self.hidden_size = config.hidden_size
        self.chunk_record: ChunkRecord | None = None

        self.rotary_emb = Qwen3MoeRotaryEmbedding(config)
        self.embed_tokens, self.norm, self.lm_head = None, None, None
        if stage_index == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if stage_index == stage_count - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = BF16Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(i): Qwen3MoeDecoderLayer(config, i)
                for i in layer_partition(config.num_hidden_layers, stage_count, stage_index)
            }
        )

    def forward_posemb(
        self, S: int, cu_seqlens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = distributed.device
        cp_rank, cp_size = distributed.cp_rank, distributed.cp_size
        if cu_seqlens is not None:
            index = zigzag_varlen_index(cp_rank, cp_size, cu_seqlens, S * cp_size)
            position_ids = index - cu_seqlens[torch.searchsorted(cu_seqlens, index, right=True) - 1]
        else:
            spans = zigzag_spans(cp_rank, cp_size, S * cp_size)
            position_ids = torch.cat([torch.arange(span.start, span.stop, device=device) for span in spans])  # fmt: skip
        cos, sin = self.rotary_emb(S * cp_size)
        return cos[position_ids].unsqueeze(0), sin[position_ids].unsqueeze(0)

    def forward_prolog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(hidden_states)

    def forward_epilog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        return model_forward(self, hidden_states, self.chunk_record, cu_seqlens)

    def reference_forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.stage_index == 0:
            hidden_states = self.forward_prolog(hidden_states)
        rotary_posemb = self.forward_posemb(hidden_states.shape[1], cu_seqlens)
        for _, layer in self.layers.items():
            hidden_states = layer.reference_forward(hidden_states, rotary_posemb, cu_seqlens)
        if self.stage_index == self.stage_count - 1:
            hidden_states = self.forward_epilog(hidden_states)
        return hidden_states
