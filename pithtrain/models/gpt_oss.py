"""openai/gpt-oss-20b and openai/gpt-oss-120b."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from pithtrain.contexts import distributed, training
from pithtrain.dualpipe.dualpipev import layer_partition
from pithtrain.dualpipe.execution import ChunkRecord, model_forward
from pithtrain.dualpipe.utils import FP8WeightCacheControl
from pithtrain.models.interface import RoutingInfo
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.clamped_swiglu import clamped_swiglu
from pithtrain.operators.deepgemm_quantize import fused_blockwise_transpose_cast_to_fp8_batched
from pithtrain.operators.ep_dispatch import prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from pithtrain.operators.grouped_linear import FP8GroupedLinearFunc, GroupedLinearFunc
from pithtrain.operators.indexed_bias_add import indexed_bias_add
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)


class GptOssRotaryEmbedding(nn.Module):
    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        inv_freq, attn_scale = self.compute_rope_params(config)
        self.set_cos_sin(config, inv_freq, attn_scale)

    @staticmethod
    def yarn_find_correction_range(
        beta_fast: float,
        beta_slow: float,
        dim: int,
        base: float,
        initial_context_length: int,
        truncate: bool,
    ) -> tuple[float, float]:
        def correction_dim(num_rotations: float) -> float:
            log_num = math.log(initial_context_length / (num_rotations * 2 * math.pi))
            log_den = math.log(base)
            return dim * log_num / (2 * log_den)

        low = correction_dim(beta_fast)
        high = correction_dim(beta_slow)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    def compute_rope_params(self, config: GptOssConfig) -> tuple[torch.Tensor, float]:
        rope_scaling = config.rope_scaling
        base, dim = rope_scaling["rope_theta"], config.head_dim
        match rope_scaling["rope_type"]:
            case "yarn":
                scaling_factor = rope_scaling["factor"]
                beta_fast, beta_slow, truncate = rope_scaling["beta_fast"], rope_scaling["beta_slow"], rope_scaling["truncate"]  # fmt: skip
                freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                freq_inter = 1.0 / (scaling_factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # fmt: skip
                low, high = self.yarn_find_correction_range(beta_fast, beta_slow, dim, base, config.initial_context_length, truncate)  # fmt: skip
                inv_freq_mask = 1.0 - self.yarn_linear_ramp_mask(low, high, dim // 2).to(torch.float32)  # fmt: skip
                inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
                attn_scale = 0.1 * math.log(scaling_factor) + 1.0
                return inv_freq, attn_scale
            case "default":
                return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), 1.0
            case other:
                raise ValueError(f"unsupported rope_type: {other!r}")

    def set_cos_sin(self, config: GptOssConfig, inv_freq: torch.Tensor, attn_scale: float) -> None:
        t = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * attn_scale).to(torch.bfloat16)
        sin = (emb.sin() * attn_scale).to(torch.bfloat16)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, S: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:S], self.sin[:S]


class GptOssExperts(nn.Module):
    def __init__(self, config: GptOssConfig, num_experts: int):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.num_experts = num_experts
        self.swiglu_limit = float(config.swiglu_limit)
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))  # fmt: skip
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(num_experts, 2 * intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.down_proj_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))
        # Raw nn.Parameter (fused gate_up) so training.GroupedLinear can't wrap it; the fp8
        # quantized-weight cache lives here, keyed by projection name + version (clear() nulls it).
        self._wq_cache: dict[str, tuple] | None = None
        self._wq_version: int = -1

    def _quantized_weight(self, name: str, weight: torch.Tensor) -> tuple:
        if torch.compiler.is_compiling():
            return fused_blockwise_transpose_cast_to_fp8_batched(weight)
        ver = FP8WeightCacheControl.version
        cache = self._wq_cache
        if self._wq_version != ver or cache is None:
            cache = self._wq_cache = {}
            self._wq_version = ver
        if name not in cache:
            cache[name] = fused_blockwise_transpose_cast_to_fp8_batched(weight)
        return cache[name]

    def _group_linear(
        self,
        x: torch.Tensor,
        weight: nn.Parameter,
        name: str,
        offs: torch.Tensor,
        ks: list | None,
        ks_tensor: torch.Tensor | None,
        group_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x @ weight[0].transpose(-2, -1)
        if training.fp8:
            return FP8GroupedLinearFunc.apply(x, weight, offs, ks, ks_tensor, self._quantized_weight(name, weight), group_indices)  # fmt: skip
        return GroupedLinearFunc.apply(x, weight, offs)

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        group_ids = torch.searchsorted(
            grouped_mm_offs.to(torch.int64),
            torch.arange(x.shape[0], device=x.device, dtype=torch.int64),
            right=True,
        ).clamp_(max=self.num_experts - 1)
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0]) if training.fp8 else None
        gate_up = self._group_linear(x, self.gate_up_proj, "gate_up_proj", grouped_mm_offs, ks, ks_tensor, gi)  # fmt: skip
        gate_up = indexed_bias_add(gate_up, self.gate_up_proj_bias, group_ids, grouped_mm_offs)
        activated = clamped_swiglu(gate_up, 1.702, self.swiglu_limit)
        out = self._group_linear(activated, self.down_proj, "down_proj", grouped_mm_offs, ks, ks_tensor, gi)  # fmt: skip
        out = indexed_bias_add(out, self.down_proj_bias, group_ids, grouped_mm_offs)
        return out


class GptOssTopKRouter(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_experts = config.num_local_experts
        self.num_experts = num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.load_balance_loss_fn = None
        self.router_replay = None
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    @torch.compile(fullgraph=True)
    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states, self.weight, self.bias)
        topk_logits, topk_idx = torch.topk(logits, k=self.num_experts_per_tok, dim=-1, sorted=True)
        if self.router_replay is not None:
            topk_idx = self.router_replay(topk_idx)
            topk_logits = logits.gather(-1, topk_idx)
        topk_weight = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        if self.load_balance_loss_fn is None:
            return topk_idx, topk_weight, None
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.num_experts, self.num_experts_per_tok)  # fmt: skip
        # Token-weight the injected lb gradient so train_step's 1/num_tokens grad scale leaves it
        # correctly normalized (it bypasses the token-weighted criterion). lb_loss stays unscaled.
        topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss * topk_weight.shape[0])
        return topk_idx, topk_weight, lb_loss


class GptOssMoE(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts_per_rank = self.num_experts // distributed.ep_size
        self.experts = GptOssExperts(config, self.experts_per_rank)
        self.router = GptOssTopKRouter(config)

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, lb_loss = self.router(hidden_states)
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


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOssConfig, is_sliding: bool):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_sliding = is_sliding
        self.sliding_window = config.sliding_window
        self.q_proj = training.Linear(hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)  # fmt: skip
        self.k_proj = training.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)  # fmt: skip
        self.v_proj = training.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)  # fmt: skip
        self.o_proj = training.Linear(self.num_heads * self.head_dim, hidden_size, bias=config.attention_bias)  # fmt: skip
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))

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
        q_embed = (q * cos) + (GptOssAttention.rotate_half(q) * sin)
        k_embed = (k * cos) + (GptOssAttention.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        query_states, key_states = self.apply_rotary_posemb(query_states, key_states, rotary_posemb)
        window_size = (self.sliding_window - 1, 0) if self.is_sliding else (None, None)
        sinks = self.sinks.to(query_states.dtype)
        if cu_seqlens is not None:
            attn_output = flash_attn_varlen_func(query_states.squeeze(0), key_states.squeeze(0), value_states.squeeze(0), cu_seqlens, S, softmax_scale=self.scaling, causal=True, window_size=window_size, learnable_sink=sinks).unsqueeze(0)  # fmt: skip
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, softmax_scale=self.scaling, causal=True, window_size=window_size, learnable_sink=sinks)  # fmt: skip
        attn_output = attn_output.reshape(B, S, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.idx = layer_idx
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = GptOssAttention(config, is_sliding)
        self.mlp = GptOssMoE(config)
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
        return hidden_states, residual

    def forward_stage1(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingInfo | None]:
        hidden_states, residual = self.forward_stage1_compute(hidden_states, rotary_posemb, cu_seqlens)  # fmt: skip
        topk_idx, topk_weight, lb_loss = self.mlp.router(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        dispatch_tokens, routing = prepare_dispatch(hidden_states, topk_idx, topk_weight, self.mlp.num_experts, distributed.ep_size, self.mlp.experts_per_rank, distributed.ep_group)  # fmt: skip
        return dispatch_tokens, residual, routing

    def forward_stage3(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: torch.Tensor | None = None,
        expand_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
    ) -> torch.Tensor:
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


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig, phase: int):
        super().__init__()
        if distributed.cp_size > 1:
            raise NotImplementedError("GptOssModel doesn't support context parallelism.")

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

        self.rotary_emb = GptOssRotaryEmbedding(config)
        self.embed_tokens, self.norm, self.lm_head = None, None, None
        if stage_index == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if stage_index == stage_count - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(i): GptOssDecoderLayer(config, i)
                for i in layer_partition(config.num_hidden_layers, stage_count, stage_index)
            }
        )

    def forward_posemb(
        self, S: int, cu_seqlens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.rotary_emb(S)
        if cu_seqlens is not None:
            device = distributed.device
            starts, ends = cu_seqlens[:-1], cu_seqlens[1:]
            lengths = ends - starts
            position_ids = torch.arange(S, device=device) - torch.repeat_interleave(starts, lengths)
            cos, sin = cos[position_ids], sin[position_ids]
        return cos.unsqueeze(0), sin.unsqueeze(0)

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
