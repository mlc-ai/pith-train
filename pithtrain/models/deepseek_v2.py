"""deepseek-ai/DeepSeek-V2."""

import math
from dataclasses import fields
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

from pithtrain.contexts import distributed
from pithtrain.dualpipe.execution import EpilogArgs, IntermediateTensors, PrologArgs, PrologOuts
from pithtrain.dualpipe.layer_partition import layer_partition
from pithtrain.dualpipe.modeling import decoder_layer_backward, decoder_layer_forward
from pithtrain.dualpipe.utils import run_backward
from pithtrain.layers.factory import ModelImplMode, get_group_linear_cls, get_linear_cls
from pithtrain.models.interface import ForwardAttnOutput
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.ep_dispatch import moe_ep_prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.ring_attention import mla_ring_attention_func
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)


class DeepSeekV2RotaryEmbedding(nn.Module):
    """
    Rotary embedding for DeepSeek-V2-Lite.
    """

    def __init__(self, config: DeepseekV2Config) -> None:
        super().__init__()
        inv_freq, attn_scale = self.compute_rope_params(config)
        self.set_cos_sin(config, inv_freq, attn_scale)

    @staticmethod
    def yarn_find_correction_range(beta_fast: float, beta_slow: float, dim: int, base: float, max_position_embeddings: int) -> Tuple[int, int]:
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

    def compute_rope_params(self, config: DeepseekV2Config) -> Tuple[torch.Tensor, float]:
        rope_scaling = config.rope_scaling
        base, dim = rope_scaling["rope_theta"], config.qk_rope_head_dim
        match rope_scaling["rope_type"]:
            case "yarn":
                factor = rope_scaling["factor"]
                original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
                beta_fast, beta_slow = rope_scaling["beta_fast"], rope_scaling["beta_slow"]
                mscale, mscale_all_dim = rope_scaling["mscale"], rope_scaling["mscale_all_dim"]
                freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                freq_inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                lo, hi = self.yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
                inv_freq_mask = 1.0 - self.yarn_linear_ramp_mask(lo, hi, dim // 2).to(torch.float32)
                inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
                attn_scale = float(self.yarn_get_mscale(factor, mscale) / self.yarn_get_mscale(factor, mscale_all_dim))
                return inv_freq, attn_scale
            case "default":
                return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), 1.0
            case other:
                raise ValueError(f"unsupported rope_type: {other!r}")

    def set_cos_sin(self, config: DeepseekV2Config, inv_freq: torch.Tensor, attn_scale: float) -> None:
        t = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * attn_scale).to(torch.bfloat16)
        sin = (emb.sin() * attn_scale).to(torch.bfloat16)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


class DeepSeekV2MLP(nn.Module):
    def __init__(self, config: DeepseekV2Config, hidden_size: Optional[int] = None, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        LinearCls = get_linear_cls()
        self.gate_proj = LinearCls(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = LinearCls(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = LinearCls(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(silu_mul(g, u))


class DeepSeekV2Experts(nn.Module):
    def __init__(self, config: DeepseekV2Config, num_experts: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        GroupLinearCls = get_group_linear_cls()
        self.gate_proj = GroupLinearCls(num_experts, self.hidden_size, self.intermediate_size)
        self.up_proj = GroupLinearCls(num_experts, self.hidden_size, self.intermediate_size)
        self.down_proj = GroupLinearCls(num_experts, self.intermediate_size, self.hidden_size)

    def forward(self, x: torch.Tensor, grouped_mm_offs: torch.Tensor, ks: list | None = None, ks_tensor: torch.Tensor | None = None) -> torch.Tensor:
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
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)), requires_grad=True)

    @torch.compile(fullgraph=True)
    def compute(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        _, _, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        if self.topk_method == "group_limited_greedy":
            n_tokens = scores.shape[0]
            group_scores = scores.view(n_tokens, self.num_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(n_tokens, self.num_group, self.num_experts // self.num_group).reshape(n_tokens, -1)
            scores = scores.masked_fill(~score_mask.bool(), 0.0)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.router_replay is not None:
            topk_idx = self.router_replay(topk_idx)
            topk_weight = scores.gather(-1, topk_idx)
        topk_weight = topk_weight * self.routed_scaling_factor

        if self.training and self.load_balance_loss_fn is not None:
            lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.n_routed_experts, self.top_k)
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)

        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)

        return topk_idx, topk_weight


class DeepSeekV2MoEWithGroupGeMM(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts_per_rank = config.n_routed_experts // distributed.ep_size
        self.n_routed_experts = config.n_routed_experts

        self.experts = DeepSeekV2Experts(config, self.experts_per_rank)
        self.gate = DeepSeekV2MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepSeekV2MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        assert distributed.ep_size == 1, "reference implementation only supports ep_size=1"
        expert_idxs = topk_ids.view(-1)
        sorted_tokens = x.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, x.shape[-1])
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]

        final_out = (outs.view(*topk_ids.shape, -1) * topk_weight.unsqueeze(dim=-1)).sum(dim=1).to(outs.dtype)
        return final_out


class DeepSeekV2Attention(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank

        LinearCls = get_linear_cls()
        if self.q_lora_rank is None:
            self.q_proj = LinearCls(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = LinearCls(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = LinearCls(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = LinearCls(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = LinearCls(config.kv_lora_rank, self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), bias=False)

        self.o_proj = LinearCls(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.softmax_scale = self.q_head_dim ** (-0.5)
        # When fp8 training is on, kv_b_proj is an FP8Linear; the pass-latent ring decompresses
        # the rotated latent via the FP8 deep_gemm path instead of a silent bf16 F.linear.
        self._fp8 = ModelImplMode.fp8_training == "deep-gemm"

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        q_embed = (q * cos) + (DeepSeekV2Attention.rotate_half(q) * sin)
        k_embed = (k * cos) + (DeepSeekV2Attention.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        normed_kv = self.kv_a_layernorm(compressed_kv)
        cos, sin = rotary_posemb
        q_pe, k_pe = self.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, unsqueeze_dim=2)

        if distributed.cp_size > 1:
            kv_b_quant = self.kv_b_proj._get_quantized_weight() if self._fp8 else None
            attn_output = mla_ring_attention_func(q_nope, q_pe, normed_kv.contiguous(), k_pe.contiguous(), self.kv_b_proj.weight, sm_scale=self.softmax_scale, qk_nope_head_dim=self.qk_nope_head_dim, v_head_dim=self.v_head_dim, cp_group=distributed.cp_group, kv_b_quant=kv_b_quant)
        else:
            kv = self.kv_b_proj(normed_kv).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            q = torch.cat([q_nope, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.num_heads, -1)], dim=-1)
            attn_output = flash_attn_func(q, k, value_states.contiguous(), softmax_scale=self.softmax_scale, causal=True)

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class DeepSeekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_id: int):
        super().__init__()
        self.idx = layer_id
        self.self_attn = DeepSeekV2Attention(config=config)

        self.mlp = DeepSeekV2MoEWithGroupGeMM(config) if config.n_routed_experts is not None and layer_id >= config.first_k_dense_replace and layer_id % config.moe_layer_freq == 0 else DeepSeekV2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.compile(fullgraph=True)
    def _forward_stage1_compute(self, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states=hidden_states, rotary_posemb=rotary_posemb)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if hasattr(self.mlp, "shared_experts"):
            residual = residual + self.mlp.shared_experts(hidden_states)

        return hidden_states, residual

    def forward_stage1(self, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]) -> ForwardAttnOutput:
        hidden_states, residual = self._forward_stage1_compute(hidden_states, rotary_posemb)

        assert isinstance(self.mlp, (DeepSeekV2MLP, DeepSeekV2MoEWithGroupGeMM))
        if isinstance(self.mlp, DeepSeekV2MLP):
            return ForwardAttnOutput(hidden_states, None, None, None, None, None, residual)

        topk_ids, topk_weight = self.mlp.gate(hidden_states)
        sorted_tokens, idxs, expert_idxs, expand_idx, dedup_input_splits, dedup_output_splits, input_splits, output_splits = moe_ep_prepare_dispatch(hidden_states, topk_ids, self.mlp.n_routed_experts, distributed.ep_size, self.mlp.experts_per_rank, distributed.ep_group)
        return ForwardAttnOutput(sorted_tokens, idxs, topk_weight, output_splits, input_splits, expert_idxs, residual, expand_idx, dedup_input_splits, dedup_output_splits)

    def forward_stage3(self, gathered_tokens: torch.Tensor, expert_idxs: Optional[torch.Tensor] = None, expand_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(self.mlp, DeepSeekV2MLP):
            assert expert_idxs is None
            return self.mlp(gathered_tokens)

        assert expert_idxs is not None
        if expand_idx is not None:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
        del gathered_tokens  # free expanded tokens; no longer needed after scatter
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = padded_index_gather(outs, reverse_shuffle_idxs)
        return outs

    @torch.compile(fullgraph=True)
    def forward_stage5(self, moe_outs: torch.Tensor, moe_local_idxs: Optional[torch.Tensor], topk_weight: Optional[torch.Tensor], residual: torch.Tensor):

        def moe_finalize(moe_outs: torch.Tensor, moe_local_idxs: Optional[torch.Tensor], topk_weight: Optional[torch.Tensor]) -> torch.Tensor:
            if distributed.ep_size > 1:
                assert moe_local_idxs is not None
                seq_len, topk = topk_weight.shape
                permuted_probs = topk_weight.view(-1)[moe_local_idxs]
                token_indices = moe_local_idxs // topk
                weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
                result = moe_outs.new_zeros(seq_len, moe_outs.shape[-1])
                result.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
                return result
            else:
                assert moe_local_idxs is None
                new_x = moe_outs
                final_out = new_x.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(dim=-1)
                final_out = final_out.sum(dim=1).to(new_x.dtype)
                return final_out

        if isinstance(self.mlp, DeepSeekV2MoEWithGroupGeMM):
            hidden_states = moe_finalize(moe_outs, moe_local_idxs, topk_weight).view(*residual.shape)
        else:
            assert moe_local_idxs is None
            assert topk_weight is None
            hidden_states = moe_outs

        hidden_states = residual + hidden_states
        return hidden_states

    def reference_forward(self, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states=hidden_states, rotary_posemb=rotary_posemb)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DeepSeekV2Model(nn.Module):
    def __init__(self, config: DeepseekV2Config, phase: int):
        super().__init__()
        num_stages = distributed.pp_size * 2
        stage_id = distributed.pp_rank if phase == 0 else num_stages - 1 - distributed.pp_rank
        self.stage_id = stage_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) if stage_id == 0 else None
        num_local_layers = layer_partition(config.num_hidden_layers, num_stages)
        layer_id_begin = sum(num_local_layers[:stage_id])
        layer_id_end = layer_id_begin + num_local_layers[stage_id]
        self.layers = nn.ModuleDict({str(i): DeepSeekV2DecoderLayer(config, i) for i in range(layer_id_begin, layer_id_end)})
        if stage_id == num_stages - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

        self.rotary_emb = DeepSeekV2RotaryEmbedding(config)

    def rotary_posemb(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = hidden_states.shape[1]
        cp_size = distributed.cp_size
        block = seq_len // 2
        front_start, back_start = distributed.cp_rank * block, (2 * cp_size - distributed.cp_rank - 1) * block
        position_ids = torch.cat([torch.arange(front_start, front_start + block, device=hidden_states.device), torch.arange(back_start, back_start + block, device=hidden_states.device)])
        cos, sin = self.rotary_emb(seq_len=seq_len * cp_size)
        return cos[position_ids].unsqueeze(0), sin[position_ids].unsqueeze(0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate_tensors: Optional[IntermediateTensors] = getattr(self, "_intermediate_tensors", None)

        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(hidden_states)

        rotary_posemb = self.rotary_posemb(hidden_states)

        if intermediate_tensors is None:
            for _, layer in self.layers.items():
                ret = decoder_layer_forward(layer, hidden_states, rotary_posemb)
                hidden_states = ret[0] if isinstance(ret, tuple) else ret
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
                hidden_states = self.lm_head(hidden_states)
            return hidden_states

        layer_idx = 0
        if self.embed_tokens is not None:
            intermediate_tensors.prolog.args = PrologArgs()
            intermediate_tensors.prolog.outs = PrologOuts(hidden_states)
        for _, layer in self.layers.items():
            ret = decoder_layer_forward(layer, hidden_states, rotary_posemb)
            if len(ret) == 2:
                hidden_states, layer_record = ret
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(layer_record):
                    src_rec = getattr(layer_record, field.name)
                    dst_rec = getattr(dst, field.name)
                    for rf in fields(src_rec):
                        if hasattr(src_rec, rf.name):
                            setattr(dst_rec, rf.name, getattr(src_rec, rf.name))
            else:
                hidden_states = ret[0]
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(dst):
                    record = getattr(dst, field.name)
                    for rf in fields(record):
                        setattr(record, rf.name, None)
            layer_idx += 1

        if self.norm is not None:
            assert self.lm_head is not None
            if not ModelImplMode.use_reference_fwd:
                hidden_states = hidden_states.detach().requires_grad_()
            intermediate_tensors.epilog.args = EpilogArgs(hidden_states)
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states

    @staticmethod
    def backward(module: "DeepSeekV2Model", dy: Optional[List[torch.Tensor]], loss: Optional[torch.Tensor], intermediate_tensors: IntermediateTensors):
        assert (dy is None) != (loss is None), "Either dy or loss should be provided"
        if loss is not None:
            assert module.norm is not None
            assert module.lm_head is not None
            loss.backward()
            loss.detach_()
            dy = (intermediate_tensors.epilog.args.hidden_states.grad,)
            intermediate_tensors.epilog.args = None
            loss = None
        else:
            assert module.norm is None
            assert module.lm_head is None

        dx = dy
        layers_list = [layer for _, layer in module.layers.items()]
        for layer, intermediate_tensors_layer in zip(reversed(layers_list), reversed(intermediate_tensors.layers)):
            dx = (decoder_layer_backward(layer, dx, loss, intermediate_tensors_layer),)

        final_grads = dx
        if module.embed_tokens is not None:
            record = intermediate_tensors.prolog
            run_backward(record.outs, dx)
            record.args = None
            record.outs = None
            final_grads = (None,)
        return final_grads
