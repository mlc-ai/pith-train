"""Native Qwen3-Omni Thinker text backbone, following Transformers' text classes.

This stage supports text pretraining. Audio/vision encoders, DeepStack feature
insertion, audiovisual position construction and the Talker remain separate work.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig

from pithtrain.contexts import distributed, training
from pithtrain.models.interface import RoutingInfo
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.cp_sequence import zigzag_spans
from pithtrain.operators.ep_dispatch import prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from pithtrain.operators.grouped_linear import GroupedLinearFunc
from pithtrain.operators.ring_attention import ring_attention_func
from pithtrain.operators.token_scatter import padded_index_gather, scatter_for_grouped_gemm
from pithtrain.pipeline.dualpipev import layer_partition
from pithtrain.pipeline.execution import ChunkRecord, model_forward


class Qwen3OmniMoeThinkerTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3OmniMoeTextConfig):
        super().__init__()
        rope = config.rope_parameters
        if rope["rope_type"] != "default":
            raise NotImplementedError("Thinker currently supports default RoPE only")
        self.base, self.head_dim = rope["rope_theta"], config.head_dim
        self.mrope_section = tuple(rope["mrope_section"])
        if len(self.mrope_section) != 3 or sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError("mrope_section must partition half of head_dim into T/H/W sections")

    @torch.no_grad()
    def forward(self, position_ids, dtype):
        # HF's interleaved T/H/W recomposition; text gives all three axes the
        # same positions. Recompute frequencies in FP32 so model.to(bfloat16)
        # cannot round an inverse-frequency buffer before the trigonometry.
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=position_ids.device)
                / self.head_dim
            )
        )
        freq = position_ids.float().unsqueeze(-1) * inv_freq
        outputs = []
        for values in (freq.cos(), freq.sin()):
            mixed = values[0].clone()
            for axis in (1, 2):
                section = slice(axis, self.mrope_section[axis] * 3, 3)
                mixed[..., section] = values[axis, ..., section]
            outputs.append(torch.cat((mixed, mixed), dim=-1).to(dtype))
        return tuple(outputs)


class Qwen3OmniMoeThinkerTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        value = hidden_states.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(dtype)


class Qwen3OmniMoeThinkerTextExperts(nn.Module):
    def __init__(self, config, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * config.moe_intermediate_size, config.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, config.hidden_size, config.moe_intermediate_size)
        )

    def forward(self, x, grouped_mm_offs, ks=None, ks_tensor=None):
        if x.shape[0] == 0:
            gate, up = F.linear(x, self.gate_up_proj[0]).chunk(2, dim=-1)
            return F.linear(F.silu(gate) * up, self.down_proj[0])
        gate, up = GroupedLinearFunc.apply(x, self.gate_up_proj, grouped_mm_offs).chunk(2, dim=-1)
        # Match HF's BF16 SiLU output before multiplication. Fusing the two
        # roundings can perturb later top-k routes and their expert gradients.
        activated = F.silu(gate) * up
        return GroupedLinearFunc.apply(activated, self.down_proj, grouped_mm_offs)


class Qwen3OmniMoeThinkerTextTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.load_balance_loss_fn = self.router_replay = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, hidden_states.shape[-1]), self.weight)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        if self.router_replay is not None:
            indices = self.router_replay(indices)
            weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        # Omni casts selected probabilities back to the router-logit dtype.
        weights = weights.to(logits.dtype)
        lb_loss = None
        if self.load_balance_loss_fn is not None:
            lb_loss = self.load_balance_loss_fn(scores, indices, self.num_experts, self.top_k)
            weights = MoELoadBalanceLossInjector.apply(weights, lb_loss * weights.shape[0])
        return indices, weights, lb_loss


class Qwen3OmniMoeThinkerTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        if config.num_experts % distributed.ep_size:
            raise ValueError("num_experts must be divisible by expert_parallel_size")
        self.experts_per_rank = config.num_experts // distributed.ep_size
        self.experts = Qwen3OmniMoeThinkerTextExperts(config, self.experts_per_rank)
        self.gate = Qwen3OmniMoeThinkerTextTopKRouter(config)

    def reference_forward(self, hidden_states):
        shape = hidden_states.shape
        indices, weights, lb_loss = self.gate(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        flat = hidden_states.reshape(-1, shape[-1])
        repeated = flat[:, None].expand(-1, self.num_experts_per_tok, -1).reshape(-1, shape[-1])
        tokens, reverse, offs, ks, sizes = scatter_for_grouped_gemm(
            repeated, indices.reshape(-1), self.experts_per_rank
        )
        outputs = padded_index_gather(self.experts(tokens, offs, ks, sizes), reverse)
        weighted = outputs.view(*indices.shape, -1) * weights.unsqueeze(-1)
        return weighted.sum(1).view(shape)


class Qwen3OmniMoeThinkerTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = training.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = training.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = training.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    reference_forward = forward


class Qwen3OmniMoeThinkerTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads, self.num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.q_proj = training.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = training.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = training.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = training.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3OmniMoeThinkerTextRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3OmniMoeThinkerTextRMSNorm(self.head_dim, config.rms_norm_eps)

    @staticmethod
    def rotate_half(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, hidden_states, rotary_posemb, cu_seqlens=None):
        b, s, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim)
        cos, sin = (value.unsqueeze(2) for value in rotary_posemb)
        q, k = q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin
        if distributed.cp_size > 1:
            output = ring_attention_func(
                q, k, v, sm_scale=self.scaling, cp_group=distributed.cp_group
            )
        elif cu_seqlens is not None:
            output = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens,
                s,
                softmax_scale=self.scaling,
                causal=True,
            ).unsqueeze(0)
        else:
            output = flash_attn_func(q, k, v, softmax_scale=self.scaling, causal=True)
        return self.o_proj(output.reshape(b, s, -1))


class Qwen3OmniMoeThinkerTextDecoderLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.idx = layer_id
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(config, layer_id)
        self.is_moe = (
            layer_id not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_id + 1) % config.decoder_sparse_step == 0
        )
        self.mlp = (
            Qwen3OmniMoeThinkerTextSparseMoeBlock(config)
            if self.is_moe
            else Qwen3OmniMoeThinkerTextMLP(config)
        )
        self.input_layernorm = Qwen3OmniMoeThinkerTextRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3OmniMoeThinkerTextRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    @torch.compile(fullgraph=True)
    def forward_stage1_compute(self, hidden_states, rotary_posemb, cu_seqlens=None):
        residual = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), rotary_posemb, cu_seqlens
        )
        hidden_states = self.post_attention_layernorm(residual)
        routing = self.mlp.gate(hidden_states) if self.is_moe else (None, None, None)
        return hidden_states, residual, *routing

    def forward_stage1(
        self, hidden_states, rotary_posemb, cu_seqlens=None
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingInfo | None]:
        hidden_states, residual, indices, weights, lb_loss = self.forward_stage1_compute(
            hidden_states, rotary_posemb, cu_seqlens
        )
        if not self.is_moe:
            return hidden_states, residual, None
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        tokens, routing = prepare_dispatch(
            hidden_states,
            indices,
            weights,
            self.mlp.num_experts,
            distributed.ep_size,
            self.mlp.experts_per_rank,
            distributed.ep_group,
        )
        return tokens, residual, routing

    def forward_stage3(self, gathered_tokens, expert_idxs=None, expand_idx=None):
        if not self.is_moe:
            return self.mlp(gathered_tokens)
        if distributed.ep_size > 1:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        tokens, reverse, offs, ks, sizes = scatter_for_grouped_gemm(
            gathered_tokens, expert_idxs, self.mlp.experts_per_rank
        )
        return padded_index_gather(self.mlp.experts(tokens, offs, ks, sizes), reverse)

    @torch.compile(fullgraph=True)
    def forward_stage5(self, moe_outs, moe_local_idxs, topk_weight, residual):
        if not self.is_moe:
            return residual + moe_outs
        if distributed.ep_size == 1:
            weighted = moe_outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)
            return residual + weighted.sum(1).view_as(residual)
        probs = topk_weight.reshape(-1)[moe_local_idxs]
        weighted = moe_outs * probs.unsqueeze(-1)
        token_indices = moe_local_idxs // topk_weight.shape[1]
        result = moe_outs.new_zeros(topk_weight.shape[0], moe_outs.shape[-1])
        result.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
        return residual + result.view_as(residual)

    def reference_forward(self, hidden_states, rotary_posemb, cu_seqlens=None):
        residual = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), rotary_posemb, cu_seqlens
        )
        return residual + self.mlp.reference_forward(self.post_attention_layernorm(residual))


class Qwen3OmniMoeThinkerTextModel(nn.Module):
    input_modalities = ("text",)

    def __init__(self, config: Qwen3OmniMoeTextConfig, phase: int):
        super().__init__()
        if training.fp8:
            raise NotImplementedError("Thinker fused experts currently support BF16 training only")
        if config.tie_word_embeddings or config.attention_dropout or config.hidden_act != "silu":
            raise NotImplementedError("Thinker requires untied embeddings, zero dropout and SiLU")
        if config.decoder_sparse_step < 1:
            raise ValueError("decoder_sparse_step must be positive")
        if phase == -1:
            self.stage_count, self.stage_index = 1, 0
        elif phase in (0, 1):
            self.stage_count = 2 * distributed.pp_size
            self.stage_index = (
                distributed.pp_rank if phase == 0 else self.stage_count - 1 - distributed.pp_rank
            )
        else:
            raise ValueError("phase must be -1, 0 or 1")
        self.hidden_size = config.hidden_size
        self.chunk_record: ChunkRecord | None = None
        self.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config)
        self.embed_tokens = self.norm = self.lm_head = None
        if self.stage_index == 0:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, config.pad_token_id
            )
        if self.stage_index == self.stage_count - 1:
            self.norm = Qwen3OmniMoeThinkerTextRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(i): Qwen3OmniMoeThinkerTextDecoderLayer(config, i)
                for i in layer_partition(
                    config.num_hidden_layers, self.stage_count, self.stage_index
                )
            }
        )

    def forward_posemb(self, length, cu_seqlens=None):
        device = distributed.device
        if cu_seqlens is not None:
            if distributed.cp_size != 1:
                raise NotImplementedError("Packed Thinker sequences require CP=1")
            starts, ends = cu_seqlens[:-1], cu_seqlens[1:]
            positions = torch.arange(length, device=device) - torch.repeat_interleave(
                starts, ends - starts
            )
        else:
            spans = zigzag_spans(
                distributed.cp_rank, distributed.cp_size, length * distributed.cp_size
            )
            positions = torch.cat(
                [torch.arange(span.start, span.stop, device=device) for span in spans]
            )
        return self.rotary_emb(positions.view(1, 1, -1).expand(3, -1, -1), training.PARAM_DTYPE)

    def forward_prolog(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward_epilog(self, hidden_states):
        return self.lm_head(self.norm(hidden_states))

    def forward(self, hidden_states, cu_seqlens=None):
        return model_forward(self, hidden_states, self.chunk_record, cu_seqlens)

    def reference_forward(self, hidden_states, cu_seqlens=None):
        if self.stage_index == 0:
            hidden_states = self.forward_prolog(hidden_states)
        rotary_posemb = self.forward_posemb(hidden_states.shape[1], cu_seqlens)
        for layer in self.layers.values():
            hidden_states = layer.reference_forward(hidden_states, rotary_posemb, cu_seqlens)
        if self.stage_index == self.stage_count - 1:
            hidden_states = self.forward_epilog(hidden_states)
        return hidden_states
