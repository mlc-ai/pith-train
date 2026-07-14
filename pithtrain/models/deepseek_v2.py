"""deepseek-ai/DeepSeek-V2 and deepseek-ai/DeepSeek-V2-Lite."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

from pithtrain.contexts import distributed, training
from pithtrain.dualpipe.dualpipev import layer_partition
from pithtrain.dualpipe.execution import ChunkRecord, model_forward
from pithtrain.models.interface import RoutingInfo
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.ep_dispatch import prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from pithtrain.operators.ring_attention import mla_ring_attention_func
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)


class DeepSeekV2RotaryEmbedding(nn.Module):
    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        inv_freq, attn_scale = self.compute_rope_params(config)
        self.set_cos_sin(config, inv_freq, attn_scale)

    @staticmethod
    def yarn_find_correction_range(
        beta_fast: float, beta_slow: float, dim: int, base: float, max_position_embeddings: int
    ) -> tuple[int, int]:
        def correction_dim(num_rotations: float) -> float:
            log_num = math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
            log_den = math.log(base)
            return dim * log_num / (2 * log_den)

        lo = math.floor(correction_dim(beta_fast))
        hi = math.ceil(correction_dim(beta_slow))
        return max(lo, 0), min(hi, dim - 1)

    @staticmethod
    def yarn_get_mscale(factor: float, mscale: float) -> float:
        return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0

    @staticmethod
    def yarn_linear_ramp_mask(lo: float, hi: float, dim: int) -> torch.Tensor:
        hi = hi + 0.001 if lo == hi else hi
        linear_func = (torch.arange(dim, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(linear_func, 0, 1)

    def compute_rope_params(self, config: DeepseekV2Config) -> tuple[torch.Tensor, float]:
        rope_scaling = config.rope_scaling
        base, dim = rope_scaling["rope_theta"], config.qk_rope_head_dim
        match rope_scaling["rope_type"]:
            case "yarn":
                factor = rope_scaling["factor"]
                original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
                beta_fast, beta_slow = rope_scaling["beta_fast"], rope_scaling["beta_slow"]
                mscale, mscale_all_dim = rope_scaling["mscale"], rope_scaling["mscale_all_dim"]
                freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                freq_inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # fmt: skip
                lo, hi = self.yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)  # fmt: skip
                inv_freq_mask = 1.0 - self.yarn_linear_ramp_mask(lo, hi, dim // 2).to(torch.float32)
                inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
                attn_scale = float(self.yarn_get_mscale(factor, mscale) / self.yarn_get_mscale(factor, mscale_all_dim))  # fmt: skip
                return inv_freq, attn_scale
            case "default":
                return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), 1.0
            case other:
                raise ValueError(f"unsupported rope_type: {other!r}")

    def set_cos_sin(
        self, config: DeepseekV2Config, inv_freq: torch.Tensor, attn_scale: float
    ) -> None:
        t = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * attn_scale).to(torch.bfloat16)
        sin = (emb.sin() * attn_scale).to(torch.bfloat16)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, S: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:S], self.sin[:S]


class DeepSeekV2MLP(nn.Module):
    def __init__(self, config: DeepseekV2Config, intermediate_size: int | None = None):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = training.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = training.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = training.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(silu_mul(g, u))

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class DeepSeekV2Experts(nn.Module):
    def __init__(self, config: DeepseekV2Config, num_experts: int):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size
        self.gate_proj = training.GroupedLinear(num_experts, hidden_size, intermediate_size)
        self.up_proj = training.GroupedLinear(num_experts, hidden_size, intermediate_size)
        self.down_proj = training.GroupedLinear(num_experts, intermediate_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0])
        kwargs = dict(grouped_mm_offs=grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi)
        g = self.gate_proj(x, **kwargs)
        u = self.up_proj(x, **kwargs)
        return self.down_proj(silu_mul(g, u), **kwargs)


class DeepSeekV2MoEGate(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.load_balance_loss_fn = None
        self.router_replay = None
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)), requires_grad=True)  # fmt: skip

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)  # fmt: skip
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        if self.topk_method == "group_limited_greedy":
            n_tokens = scores.shape[0]
            group_scores = scores.view(n_tokens, self.num_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(n_tokens, self.num_group, self.num_experts // self.num_group).reshape(n_tokens, -1)  # fmt: skip
            scores = scores.masked_fill(~score_mask.bool(), 0.0)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.router_replay is not None:
            topk_idx = self.router_replay(topk_idx)
            topk_weight = scores.gather(-1, topk_idx)
        topk_weight = topk_weight * self.routed_scaling_factor
        if self.load_balance_loss_fn is None:
            return topk_idx, topk_weight, None
        lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.n_routed_experts, self.top_k)
        # Token-weight the injected lb gradient so train_step's 1/num_tokens grad scale leaves it
        # correctly normalized (it bypasses the token-weighted criterion). lb_loss stays unscaled.
        topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss * topk_weight.shape[0])
        return topk_idx, topk_weight, lb_loss


class DeepSeekV2MoE(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts_per_rank = config.n_routed_experts // distributed.ep_size
        self.n_routed_experts = config.n_routed_experts
        self.experts = DeepSeekV2Experts(config, self.experts_per_rank)
        self.gate = DeepSeekV2MoEGate(config)
        intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = DeepSeekV2MLP(config=config, intermediate_size=intermediate_size)

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, lb_loss = self.gate(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        expert_idxs = topk_idx.view(-1)
        replicated_tokens = hidden_states.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, hidden_states.shape[-1])  # fmt: skip
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(replicated_tokens, expert_idxs, self.experts_per_rank)  # fmt: skip
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]
        y = (outs.view(*topk_idx.shape, -1) * topk_weight.unsqueeze(dim=-1)).sum(dim=1).to(outs.dtype)  # fmt: skip
        y = y.view(*orig_shape) + self.shared_experts(identity)
        return y


class DeepSeekV2Attention(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        if self.q_lora_rank is None:
            self.q_proj = training.Linear(hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = training.Linear(hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = training.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)  # fmt: skip
        self.kv_a_proj_with_mqa = training.Linear(hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False)  # fmt: skip
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = training.Linear(config.kv_lora_rank, self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), bias=False)  # fmt: skip
        self.o_proj = training.Linear(self.num_heads * self.v_head_dim, hidden_size, bias=False)
        self.softmax_scale = self.q_head_dim ** (-0.5)

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
        B, S, H, D = q.shape
        q = q.view(B, S, H, D // 2, 2).transpose(4, 3).reshape(B, S, H, D)
        B, S, H, D = k.shape
        k = k.view(B, S, H, D // 2, 2).transpose(4, 3).reshape(B, S, H, D)
        q_embed = (q * cos) + (DeepSeekV2Attention.rotate_half(q) * sin)
        k_embed = (k * cos) + (DeepSeekV2Attention.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(B, S, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)  # fmt: skip
        k_pe = k_pe.view(B, S, 1, self.qk_rope_head_dim)
        normed_kv = self.kv_a_layernorm(compressed_kv)
        q_pe, k_pe = self.apply_rotary_posemb(q_pe, k_pe, rotary_posemb)
        if distributed.cp_size > 1:
            kv_b_quant = self.kv_b_proj._get_quantized_weight() if training.fp8 else None
            attn_output = mla_ring_attention_func(q_nope, q_pe, normed_kv.contiguous(), k_pe.contiguous(), self.kv_b_proj.weight, sm_scale=self.softmax_scale, qk_nope_head_dim=self.qk_nope_head_dim, v_head_dim=self.v_head_dim, cp_group=distributed.cp_group, kv_b_quant=kv_b_quant)  # fmt: skip
        else:
            kv = self.kv_b_proj(normed_kv).view(B, S, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)  # fmt: skip
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            q = torch.cat([q_nope, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.num_heads, -1)], dim=-1)
            if cu_seqlens is not None:
                attn_output = flash_attn_varlen_func(q.squeeze(0), k.squeeze(0), value_states.squeeze(0).contiguous(), cu_seqlens, S, softmax_scale=self.softmax_scale, causal=True).unsqueeze(0)  # fmt: skip
            else:
                attn_output = flash_attn_func(q, k, value_states.contiguous(), softmax_scale=self.softmax_scale, causal=True)  # fmt: skip
        attn_output = attn_output.reshape(B, S, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class DeepSeekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_id: int):
        super().__init__()
        self.idx = layer_id
        self.self_attn = DeepSeekV2Attention(config=config)
        has_experts = layer_id >= config.first_k_dense_replace
        self.mlp = DeepSeekV2MoE(config) if has_experts else DeepSeekV2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.compile(fullgraph=True)
    def forward_stage1_compute(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, DeepSeekV2MLP):
            return hidden_states, residual, None, None, None
        residual = residual + self.mlp.shared_experts(hidden_states)
        topk_idx, topk_weight, lb_loss = self.mlp.gate(hidden_states)
        return hidden_states, residual, topk_idx, topk_weight, lb_loss

    def forward_stage1(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingInfo | None]:
        hidden_states, residual, topk_idx, topk_weight, lb_loss = self.forward_stage1_compute(hidden_states, rotary_posemb, cu_seqlens)  # fmt: skip
        if isinstance(self.mlp, DeepSeekV2MLP):
            return hidden_states, residual, None
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        dispatch_tokens, routing = prepare_dispatch(hidden_states, topk_idx, topk_weight, self.mlp.n_routed_experts, distributed.ep_size, self.mlp.experts_per_rank, distributed.ep_group)  # fmt: skip
        return dispatch_tokens, residual, routing

    def forward_stage3(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: torch.Tensor | None = None,
        expand_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.mlp, DeepSeekV2MLP):
            return self.mlp(gathered_tokens)
        if distributed.ep_size > 1:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)  # fmt: skip
        del gathered_tokens
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        return padded_index_gather(outs, reverse_shuffle_idxs)

    @torch.compile(fullgraph=True)
    def forward_stage5(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: torch.Tensor | None,
        topk_weight: torch.Tensor | None,
        residual: torch.Tensor,
    ):
        if not isinstance(self.mlp, DeepSeekV2MoE):
            return residual + moe_outs
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


class DeepSeekV2Model(nn.Module):
    def __init__(self, config: DeepseekV2Config, phase: int):
        super().__init__()
        match phase:
            case 0:
                stage_count = distributed.pp_size * 2
                stage_index = distributed.pp_rank
            case 1:
                stage_count = distributed.pp_size * 2
                stage_index = stage_count - 1 - distributed.pp_rank
            case -1:
                # non-pipelined reference: a single stage owns the whole model
                stage_count = 1
                stage_index = 0
            case _:
                raise ValueError("phase must be 0, 1, or -1, got %d" % phase)
        self.stage_index, self.stage_count = stage_index, stage_count
        self.chunk_record: ChunkRecord | None = None

        self.rotary_emb = DeepSeekV2RotaryEmbedding(config)
        self.embed_tokens, self.norm, self.lm_head = None, None, None
        if stage_index == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if stage_index == stage_count - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(i): DeepSeekV2DecoderLayer(config, i)
                for i in layer_partition(config.num_hidden_layers, stage_count, stage_index)
            }
        )

    def forward_posemb(
        self, S: int, cu_seqlens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = distributed.device
        if cu_seqlens is not None:
            starts, ends = cu_seqlens[:-1], cu_seqlens[1:]
            lengths = ends - starts
            position_ids = torch.arange(S, device=device) - torch.repeat_interleave(starts, lengths)
            cos, sin = self.rotary_emb(S)
            return cos[position_ids].unsqueeze(0), sin[position_ids].unsqueeze(0)
        cp_size, block_size = distributed.cp_size, S // 2
        front_start = distributed.cp_rank * block_size
        back_start = (2 * cp_size - distributed.cp_rank - 1) * block_size
        front_end, back_end = front_start + block_size, back_start + block_size
        position_ids = torch.cat([torch.arange(front_start, front_end, device=device), torch.arange(back_start, back_end, device=device)])  # fmt: skip
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
