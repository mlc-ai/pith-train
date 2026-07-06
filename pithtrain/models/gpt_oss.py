"""openai/gpt-oss-120b and openai/gpt-oss-20b."""

import math

import torch
import torch.nn.functional as F
from flash_attn.cute.interface import flash_attn_func
from torch import nn

from pithtrain.contexts import distributed, training
from pithtrain.dualpipe.dualpipev import layer_partition
from pithtrain.dualpipe.execution import ChunkRecord, record_forward
from pithtrain.dualpipe.utils import FP8WeightCacheControl
from pithtrain.models.interface import MoERouting
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.clamped_swiglu import clamped_swiglu
from pithtrain.operators.deepgemm_quantize import fused_blockwise_transpose_cast_to_fp8_batched
from pithtrain.operators.ep_dispatch import prepare_dispatch
from pithtrain.operators.grouped_linear import FP8GroupedLinearFunc, GroupedLinearFunc
from pithtrain.operators.indexed_bias_add import indexed_bias_add
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

SWIGLU_ALPHA = 1.702  # gpt-oss architectural constant (sigmoid approximation of GELU).


class GptOssRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 150000.0,
        scaling_factor: float = 32.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        truncate: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.truncate = truncate
        self._set_cos_sin_cache(max_position_embeddings, device, torch.get_default_dtype())

    @staticmethod
    def yarn_find_correction_range(low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int, truncate: bool) -> tuple[float, float]:
        def correction_dim(num_rotations: float) -> float:
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
        low = correction_dim(low_rot)
        high = correction_dim(high_rot)
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

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device | None, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

        low, high = self.yarn_find_correction_range(self.beta_fast, self.beta_slow, dim, self.base, self.original_max_position_embeddings, self.truncate)
        inv_freq_mask = 1.0 - self.yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        concentration = 0.1 * math.log(self.scaling_factor) + 1.0  # YaRN concentration factor (mscale).

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * concentration).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * concentration).to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


class GptOssExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(num_experts, 2 * intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.down_proj_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))

        # gpt-oss stores expert projections as raw nn.Parameter (HF layout with fused gate_up),
        # so the training.GroupedLinear module wrapper does not apply. FP8GroupedLinearFunc is
        # dispatched directly on these parameters and the quantized-weight cache is hosted here,
        # keyed by projection name since one module owns two weights. Version-keyed like
        # grouped_linear._get_quantized_weight; FP8WeightCacheControl.clear resets _wq_cache=None.
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
            return FP8GroupedLinearFunc.apply(
                x, weight, offs, ks, ks_tensor, self._quantized_weight(name, weight), group_indices
            )
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

        # Hopper SM90 needs explicit per-row group indices for m_grouped FP8 GEMM;
        # Blackwell ignores it. Computed once and shared across both projections.
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0]) if training.fp8 else None

        gate_up = self._group_linear(x, self.gate_up_proj, "gate_up_proj", grouped_mm_offs, ks, ks_tensor, gi)
        gate_up = indexed_bias_add(gate_up, self.gate_up_proj_bias, group_ids, grouped_mm_offs)
        activated = clamped_swiglu(gate_up, SWIGLU_ALPHA, self.swiglu_limit)

        out = self._group_linear(activated, self.down_proj, "down_proj", grouped_mm_offs, ks, ks_tensor, gi)
        out = indexed_bias_add(out, self.down_proj_bias, group_ids, grouped_mm_offs)
        return out


class GptOssTopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.load_balance_loss_fn = None
        self.router_replay = None
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    @torch.compile(fullgraph=True)
    def compute(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        logits = F.linear(hidden_states, self.weight, self.bias)

        topk_logits, topk_idx = torch.topk(logits, k=self.num_experts_per_tok, dim=-1, sorted=True)
        if self.router_replay is not None:
            topk_idx = self.router_replay(topk_idx)
            topk_logits = logits.gather(-1, topk_idx)
        topk_weight = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

        if self.training and self.load_balance_loss_fn is not None:
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.num_experts, self.num_experts_per_tok)
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        return topk_idx, topk_weight


class GptOssMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        intermediate_size: int,
        swiglu_limit: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.experts_per_rank = num_experts // distributed.ep_size
        self.experts = GptOssExperts(self.experts_per_rank, hidden_size, intermediate_size, swiglu_limit)
        self.router = GptOssTopKRouter(hidden_size, num_experts, num_experts_per_tok)

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.router(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        expert_idxs = topk_idx.view(-1)
        replicated_tokens = hidden_states.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, hidden_states.shape[-1])
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(replicated_tokens, expert_idxs, self.experts_per_rank)
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]
        y = (outs.view(*topk_idx.shape, -1) * topk_weight.unsqueeze(dim=-1)).sum(dim=1).to(outs.dtype)
        return y.view(*orig_shape)


class GptOssAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = True,
        is_sliding: bool = False,
        sliding_window: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window

        self.q_proj = training.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = training.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = training.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = training.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        # learnable per-head attention sink, fused into the softmax denominator so a head can attend to ~nothing.
        self.sinks = nn.Parameter(torch.zeros(num_attention_heads))

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_posemb(q: torch.Tensor, k: torch.Tensor, rotary_posemb: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = rotary_posemb
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        q_embed = (q * cos) + (GptOssAttention.rotate_half(q) * sin)
        k_embed = (k * cos) + (GptOssAttention.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, hidden_states: torch.Tensor, rotary_posemb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        query_states, key_states = self.apply_rotary_posemb(query_states, key_states, rotary_posemb)

        # FA-4 expects (B, S, H, D); GQA is auto-detected from H_q vs H_kv.
        # Sliding window: (W-1, 0) means each query attends to W tokens (self + W-1 prior).
        window_size: tuple[int | None, int | None] = (
            (self.sliding_window - 1, 0) if self.is_sliding else (None, None)
        )
        # FA-4 requires learnable_sink to match q/k/v dtype; the parameter itself stays in
        # fp32 for optimizer numerical stability. flash_attn_func returns (out, lse).
        attn_output, _ = flash_attn_func(
            query_states,
            key_states,
            value_states,
            softmax_scale=self.scaling,
            causal=True,
            window_size=window_size,
            learnable_sink=self.sinks.to(query_states.dtype),
        )

        attn_output = attn_output.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class GptOssDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        swiglu_limit: float,
        rms_norm_eps: float,
        attention_bias: bool,
        layer_idx: int,
        is_sliding: bool = False,
        sliding_window: int = 128,
    ):
        super().__init__()
        self.idx = layer_idx
        self.hidden_size = hidden_size

        self.self_attn = GptOssAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            is_sliding=is_sliding,
            sliding_window=sliding_window,
        )
        self.mlp = GptOssMLP(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            swiglu_limit=swiglu_limit,
        )
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward_stage1_compute(self, hidden_states: torch.Tensor, rotary_posemb: tuple[torch.Tensor, torch.Tensor]):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_posemb)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual

    def forward_stage1(self, hidden_states: torch.Tensor, rotary_posemb: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, MoERouting | None]:
        hidden_states, residual = self.forward_stage1_compute(hidden_states, rotary_posemb)
        topk_idx, topk_weight = self.mlp.router(hidden_states)
        dispatch_tokens, routing = prepare_dispatch(hidden_states, topk_idx, topk_weight, self.mlp.num_experts, distributed.ep_size, self.mlp.experts_per_rank, distributed.ep_group)
        return dispatch_tokens, residual, routing

    def forward_stage3(self, gathered_tokens: torch.Tensor, expert_idxs: torch.Tensor | None = None, expand_idx: torch.Tensor | None = None) -> torch.Tensor:
        if distributed.ep_size > 1:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
        del gathered_tokens
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        return padded_index_gather(outs, reverse_shuffle_idxs)

    @torch.compile(fullgraph=True)
    def forward_stage5(self, moe_outs: torch.Tensor, moe_local_idxs: torch.Tensor | None, topk_weight: torch.Tensor | None, residual: torch.Tensor) -> torch.Tensor:
        if distributed.ep_size == 1:
            weighted = moe_outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)
            return residual + weighted.sum(dim=1).to(moe_outs.dtype).view(*residual.shape)
        permuted_probs = topk_weight.view(-1)[moe_local_idxs]
        token_indices = moe_local_idxs // topk_weight.shape[1]
        weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
        aggregated = moe_outs.new_zeros(topk_weight.shape[0], moe_outs.shape[-1])
        aggregated.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
        return residual + aggregated.view(*residual.shape)

    def reference_forward(self, hidden_states: torch.Tensor, rotary_posemb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_posemb)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.reference_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GptOssModel(nn.Module):
    def __init__(self, config, phase: int):
        super().__init__()
        if distributed.cp_size > 1:
            raise NotImplementedError("GptOssModel does not support context parallelism yet.")
        match phase:
            case 0:
                stage_count = distributed.pp_size * 2
                stage_index = distributed.pp_rank
            case 1:
                stage_count = distributed.pp_size * 2
                stage_index = stage_count - 1 - distributed.pp_rank
            case _:
                stage_count = 1
                stage_index = 0
        self.stage_index, self.stage_count = stage_index, stage_count
        self.chunk_record: ChunkRecord | None = None

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        intermediate_size = config.intermediate_size
        num_experts = getattr(config, "num_local_experts", 128)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 4)
        swiglu_limit = float(getattr(config, "swiglu_limit", 7.0))
        rms_norm_eps = config.rms_norm_eps
        attention_bias = getattr(config, "attention_bias", True)
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        sliding_window = getattr(config, "sliding_window", 128)
        rope_theta = getattr(config, "rope_theta", 150000.0)
        rope_scaling = getattr(config, "rope_scaling", None) or {}

        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "sliding_attention" if i % 2 == 0 else "full_attention"
                for i in range(config.num_hidden_layers)
            ]

        self.rotary_emb = GptOssRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=float(rope_scaling.get("factor", 32.0)),
            original_max_position_embeddings=int(rope_scaling.get("original_max_position_embeddings", 4096)),
            beta_fast=float(rope_scaling.get("beta_fast", 32.0)),
            beta_slow=float(rope_scaling.get("beta_slow", 1.0)),
            truncate=bool(rope_scaling.get("truncate", False)),
        )
        self.embed_tokens, self.norm, self.lm_head = None, None, None
        if stage_index == 0:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        if stage_index == stage_count - 1:
            self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.layers = nn.ModuleDict(
            {
                str(i): GptOssDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    swiglu_limit=swiglu_limit,
                    rms_norm_eps=rms_norm_eps,
                    attention_bias=attention_bias,
                    layer_idx=i,
                    is_sliding=(layer_types[i] == "sliding_attention"),
                    sliding_window=sliding_window,
                )
                for i in layer_partition(config.num_hidden_layers, stage_count, stage_index)
            }
        )

    def forward_posemb(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = hidden_states.shape[1]
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        return cos[:seq_len].unsqueeze(0), sin[:seq_len].unsqueeze(0)

    def forward_prolog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(hidden_states)

    def forward_epilog(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return record_forward(self, hidden_states, self.chunk_record)

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.stage_index == 0:
            hidden_states = self.forward_prolog(hidden_states)
        rotary_posemb = self.forward_posemb(hidden_states)
        for _, layer in self.layers.items():
            hidden_states = layer.reference_forward(hidden_states, rotary_posemb)
        if self.stage_index == self.stage_count - 1:
            hidden_states = self.forward_epilog(hidden_states)
        return hidden_states
