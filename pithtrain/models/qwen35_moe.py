"""
Qwen/Qwen3.5-35B-A3B, Qwen/Qwen3.5-122B-A10B, and Qwen/Qwen3.5-397B-A17B (text tower).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from pithtrain.contexts import distributed, training
from pithtrain.kernels import bf16_experts_supported, fp8_fused_experts_supported
from pithtrain.models.interface import RoutingInfo
from pithtrain.modules.load_balance import (
    MoELoadBalanceLossInjector,
    MoELoadBalanceLossTracker,
    replay_indices,
)
from pithtrain.operators.causal_conv1d import causal_conv1d
from pithtrain.operators.cp_sequence import (
    contiguous_to_zigzag,
    prepend_conv_state,
    zigzag_spans,
    zigzag_to_contiguous,
)
from pithtrain.operators.ep_dispatch import prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from pithtrain.operators.fp8_experts import fp8_scatter_quant, fp8_swiglu_gate_up_experts
from pithtrain.operators.gated_delta_rule import gated_delta_rule
from pithtrain.operators.linear import BF16Linear
from pithtrain.operators.ring_attention import ring_attention_func
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)
from pithtrain.pipeline.dualpipev import layer_partition, layer_sizes
from pithtrain.pipeline.execution import ChunkRecord, model_forward

__all__ = ["Qwen35MoeGatedDeltaNet", "Qwen35MoeModel", "Qwen35MoeTopKRouter"]


def pipeline_layer_sizes(num_layers: int, stage_count: int) -> list[int]:
    """
    Moves one layer from the last stage (norm, lm_head over a 248k vocabulary and the loss, on
    pipeline rank 0) to the middle stage (on the last pipeline rank).
    """
    sizes = layer_sizes(num_layers, stage_count)
    if stage_count < 4:
        return sizes
    moved = sizes.copy()
    moved[-1] -= 1
    moved[stage_count // 2] += 1
    balanced = all(abs(moved[i] - moved[stage_count - 1 - i]) <= 1 for i in range(stage_count))
    return moved if balanced else sizes


def recompute_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
    ops = torch.ops.pithtrain
    if func in (
        ops.qwen35_causal_conv1d.default,
        ops.fp8_linear_fwd_prequantized.default,
    ):  # fmt: skip
        return CheckpointPolicy.MUST_RECOMPUTE
    if func in (
        ops.gated_delta_rule_fwd.default, ops.gated_delta_rule_dense_fwd.default,
        ops.prepend_conv_state.default,
        ops.fp8_quantize_activation.default, ops.flash_attn4_mha_fwd.default,
        ops.flash_attn4_varlen_fwd.default, ops.zigzag_ring_fwd.default,
    ):  # fmt: skip
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_SAVE


def selective_recompute_contexts():
    """
    Selective checkpoint contexts for the token-mixing blocks (Gated DeltaNet, attention). The
    backward recomputes the GDN conv and the FP8 input projections (in_proj_qkv / in_proj_z,
    q / k / v_proj) from their saved FP8 input instead of keeping their bf16 outputs. The
    recurrence, attention, the CP exchanges and the input quantization are saved.
    """
    return create_selective_checkpoint_contexts(recompute_policy)


def document_positions(cu_seqlens: torch.Tensor, S: int) -> torch.Tensor:
    """
    Per-document token positions without a host sync or data-dependent output shape.
    """
    token_idx = torch.arange(S, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    return token_idx - cu_seqlens[torch.searchsorted(cu_seqlens, token_idx, right=True) - 1]


class Qwen35MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        output = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen35MoeRMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        output = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * self.weight.float() * F.silu(gate.float())
        return output.type_as(x)


class Qwen35MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig) -> None:
        super().__init__()
        rope_params = config.rope_parameters
        rotary_dim = int(config.head_dim * rope_params.get("partial_rotary_factor", 1.0))
        base = rope_params["rope_theta"]
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))  # fmt: skip
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin", emb.sin().to(torch.bfloat16), persistent=False)

    def forward(self, S: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:S], self.sin[:S]


class Qwen35MoeGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1)  # fmt: skip
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = Qwen35MoeRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)  # fmt: skip
        self.in_proj_qkv = training.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)  # fmt: skip
        self.in_proj_z = training.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = training.Linear(self.value_dim, self.hidden_size, bias=False)

    def split_qkv(self, mixed_qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        B, S, _ = mixed_qkv.shape
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)  # fmt: skip
        query = F.normalize(query.reshape(B, S, -1, self.head_k_dim), dim=-1)
        key = F.normalize(key.reshape(B, S, -1, self.head_k_dim), dim=-1)
        # Contiguous here, where Inductor writes it straight from the fused SiLU; the delta-rule op
        # would otherwise make the same copy eagerly in forward and again in backward.
        value = value.reshape(B, S, -1, self.head_v_dim).contiguous()
        return query, key, value

    def conv_qkv(self, mixed_qkv: torch.Tensor, offset: int) -> tuple[torch.Tensor, ...]:
        """
        The channels-last conv is bit-identical to self.conv1d on the transposed input sliced to
        [offset, offset + S).
        """
        conv_out = causal_conv1d(mixed_qkv, self.conv1d.weight, offset)
        return self.split_qkv(F.silu(conv_out))

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        if cu_seqlens is not None or not torch.is_grad_enabled():
            return self.block(hidden_states, cu_seqlens)
        return checkpoint(self.block, hidden_states, None, use_reentrant=False, context_fn=selective_recompute_contexts)  # fmt: skip

    def block(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        quantized_hidden = training.Linear.quantize_input(hidden_states)
        mixed_qkv = self.in_proj_qkv(hidden_states, quantized_hidden)
        z = self.in_proj_z(hidden_states, quantized_hidden).reshape(B, S, -1, self.head_v_dim)
        b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
        # Under CP, the tokens prepended from the previous rank replace the conv's zero padding, so
        # the valid output starts at offset.
        offset = 0
        if distributed.cp_size > 1:
            offset = self.conv_kernel_size - 1
            mixed_qkv = prepend_conv_state(mixed_qkv, offset, distributed.cp_group)
        if cu_seqlens is None:
            query, key, value = self.conv_qkv(mixed_qkv, offset)
        else:
            # Mask taps crossing document boundaries. Round after SiLU to match Inductor's fused
            # conv/SiLU; an intermediate bf16 cast makes eager and compiled results differ.
            kernel_size, positions = self.conv_kernel_size, document_positions(cu_seqlens, S)
            padded, conv_weight = F.pad(mixed_qkv.transpose(1, 2).float(), (kernel_size - 1, 0)), self.conv1d.weight.float()  # fmt: skip
            conv_out = sum(padded[..., i : i + S] * conv_weight[..., i] * (positions >= kernel_size - 1 - i) for i in range(kernel_size))  # fmt: skip
            query, key, value = self.split_qkv(F.silu(conv_out).to(mixed_qkv.dtype).transpose(1, 2))
        beta = b.sigmoid()
        # .float() on A_log guards against -inf when loaded in low precision.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        if cu_seqlens is not None:
            # exp(-128) underflows fp32 to zero, resetting state at document boundaries. A finite
            # gate avoids FLA's inf-inf; its magnitude preserves chunk-cumsum precision.
            g = torch.where(positions[:, None] == 0, -128.0, g)
        if distributed.cp_size > 1:
            core_attn_out = gated_delta_rule(query, key, value, g, beta, distributed.cp_group)
        else:
            core_attn_out = gated_delta_rule(query, key, value, g, beta)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(B, S, -1)
        return self.out_proj(core_attn_out)


class Qwen35MoeAttention(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        attention_bias = config.attention_bias
        self.q_proj = training.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=attention_bias)  # fmt: skip
        self.k_proj = training.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)  # fmt: skip
        self.v_proj = training.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)  # fmt: skip
        self.o_proj = training.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias)  # fmt: skip
        self.q_norm = Qwen35MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_posemb(
        q: torch.Tensor, k: torch.Tensor, rotary_posemb: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = rotary_posemb
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = torch.cat([(q_rot * cos) + (Qwen35MoeAttention.rotate_half(q_rot) * sin), q_pass], dim=-1)  # fmt: skip
        k_embed = torch.cat([(k_rot * cos) + (Qwen35MoeAttention.rotate_half(k_rot) * sin), k_pass], dim=-1)  # fmt: skip
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not (training.fp8 and torch.is_grad_enabled()):
            return self.block(hidden_states, rotary_posemb, cu_seqlens)
        return checkpoint(self.block, hidden_states, rotary_posemb, cu_seqlens, use_reentrant=False, context_fn=selective_recompute_contexts)  # fmt: skip

    def block(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.size()
        quantized_hidden = training.Linear.quantize_input(hidden_states)
        query_states, gate = torch.chunk(self.q_proj(hidden_states, quantized_hidden).view(B, S, -1, self.head_dim * 2), 2, dim=-1)  # fmt: skip
        gate = gate.reshape(B, S, -1)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(self.k_proj(hidden_states, quantized_hidden).view(B, S, self.num_kv_heads, self.head_dim))  # fmt: skip
        value_states = self.v_proj(hidden_states, quantized_hidden).view(B, S, self.num_kv_heads, self.head_dim)  # fmt: skip
        query_states, key_states = self.apply_rotary_posemb(query_states, key_states, rotary_posemb)
        if distributed.cp_size > 1:
            attn_output = ring_attention_func(query_states, key_states, value_states, sm_scale=self.scaling, cp_group=distributed.cp_group)  # fmt: skip
        elif cu_seqlens is not None:
            attn_output = flash_attn_varlen_func(query_states.squeeze(0), key_states.squeeze(0), value_states.squeeze(0), cu_seqlens, S, softmax_scale=self.scaling, causal=True).unsqueeze(0)  # fmt: skip
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, softmax_scale=self.scaling, causal=True)  # fmt: skip
        attn_output = attn_output.reshape(B, S, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen35MoeMLP(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, intermediate_size: int | None = None):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.shared_expert_intermediate_size
        self.gate_proj = training.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = training.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = training.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_x = training.Linear.quantize_input(x)
        return self.down_proj(silu_mul(self.gate_proj(x, quantized_x), self.up_proj(x, quantized_x)))  # fmt: skip


class Qwen35MoeExperts(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.top_k = getattr(config, "num_experts_per_tok", num_experts)
        self.gate_up_proj = training.GroupedLinear(num_experts, self.hidden_size, 2 * self.moe_intermediate_size)  # fmt: skip
        self.down_proj = training.GroupedLinear(num_experts, self.moe_intermediate_size, self.hidden_size)  # fmt: skip

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
            return fp8_swiglu_gate_up_experts(self.gate_up_proj, self.down_proj, *args)
        kwargs = dict(grouped_mm_offs=grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=group_indices)  # fmt: skip
        gate_up = self.gate_up_proj(x, **kwargs)
        gate = gate_up[:, : self.moe_intermediate_size].contiguous()
        up = gate_up[:, self.moe_intermediate_size :].contiguous()
        return self.down_proj(silu_mul(gate, up), **kwargs)

    def uses_bf16_cute(self) -> bool:
        return not training.fp8 and bf16_experts_supported(self.gate_up_proj.num_groups)

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
            return bf16_swiglu_experts(self.gate_up_proj, None, self.down_proj, *args), None
        x, reverse_shuffle_idxs, offs, ks, ks_tensor, quantized_input = fp8_scatter_quant(gathered, expert_idxs, self.num_experts, expand_idx, expert_counts)  # fmt: skip
        # self.forward, not self(...): calling the module re-enters FSDP's input cast.
        return self.forward(x, offs, ks=ks, ks_tensor=ks_tensor, quantized_input=quantized_input), reverse_shuffle_idxs  # fmt: skip


class Qwen35MoeTopKRouter(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.load_balance_loss_fn = None
        self.router_replay = None
        self.weight = nn.Parameter(torch.empty((config.num_experts, config.hidden_size)), requires_grad=True)  # fmt: skip

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
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        if self.load_balance_loss_fn is None:
            return topk_idx, topk_weight, None
        lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.num_experts, self.num_experts_per_tok)  # fmt: skip
        # Token-weight the injected lb gradient so train_step's 1/num_tokens grad scale leaves it
        # correctly normalized (it bypasses the token-weighted objective). lb_loss stays unscaled.
        topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss * topk_weight.shape[0])
        return topk_idx, topk_weight, lb_loss


class Qwen35MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts_per_rank = self.num_experts // distributed.ep_size

        self.gate = Qwen35MoeTopKRouter(config)
        self.experts = Qwen35MoeExperts(config, self.experts_per_rank)
        self.shared_expert = Qwen35MoeMLP(config, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def shared_out(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared = self.shared_expert(hidden_states)
        return torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared

    def reference_forward(
        self, hidden_states: torch.Tensor, replay_idx: torch.Tensor | None
    ) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, lb_loss = self.gate(hidden_states, replay_idx)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        expert_idxs = topk_idx.view(-1)
        sorted_tokens = hidden_states.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, hidden_states.shape[-1])  # fmt: skip
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)  # fmt: skip
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]
        y = (outs.view(*topk_idx.shape, -1) * topk_weight.unsqueeze(dim=-1)).sum(dim=1).to(outs.dtype)  # fmt: skip
        return y.view(*orig_shape) + self.shared_out(identity)


class Qwen35MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.idx = layer_idx
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen35MoeGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen35MoeAttention(config)
        self.mlp = Qwen35MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # fmt: skip

        layer_types = config.layer_types
        linear_at = lambda i: 0 <= i < len(layer_types) and layer_types[i] == "linear_attention"
        sharded = distributed.cp_size > 1
        self.to_contiguous = sharded and self.is_linear and not linear_at(layer_idx - 1)
        self.to_zigzag = sharded and self.is_linear and not linear_at(layer_idx + 1)

    @torch.compile(fullgraph=True)
    def forward_stage1_compute(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
        replay_idx: torch.Tensor | None = None,
    ):
        if self.to_contiguous:
            hidden_states = zigzag_to_contiguous(hidden_states, distributed.cp_group)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear:
            hidden_states = self.linear_attn(hidden_states, cu_seqlens)
        else:
            hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        residual = residual + self.mlp.shared_out(hidden_states)
        topk_idx, topk_weight, lb_loss = self.mlp.gate(hidden_states, replay_idx)
        return hidden_states, residual, topk_idx, topk_weight, lb_loss

    def resolve_replay(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """
        Convert the store's zigzag routes to contiguous order for linear-attention gates.
        """
        replay_idx = replay_indices(self.mlp.gate, hidden_states, self.mlp.num_experts_per_tok)
        if replay_idx is not None and self.is_linear and distributed.cp_size > 1:
            replay_idx = zigzag_to_contiguous(replay_idx.unflatten(0, hidden_states.shape[:2]), distributed.cp_group).flatten(0, 1)  # fmt: skip
        return replay_idx

    def forward_stage1(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingInfo | None]:
        replay_idx = self.resolve_replay(hidden_states)
        hidden_states, residual, topk_idx, topk_weight, lb_loss = self.forward_stage1_compute(hidden_states, rotary_posemb, cu_seqlens, replay_idx)  # fmt: skip
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

    # Not fused with the next layer's forward_stage1 into one compiled region as qwen3_moe does:
    # for this model the fused region changes the forward bits at its layer boundaries.
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
            hidden_states = residual + weighted.sum(dim=1).to(moe_outs.dtype).view(*residual.shape)  # fmt: skip
        else:
            permuted_probs = topk_weight.view(-1)[moe_local_idxs]
            token_indices = moe_local_idxs // topk_weight.shape[1]
            weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
            aggregated = moe_outs.new_zeros(topk_weight.shape[0], moe_outs.shape[-1])
            aggregated.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
            hidden_states = residual + aggregated.view(*residual.shape)
        if self.to_zigzag:
            hidden_states = contiguous_to_zigzag(hidden_states, distributed.cp_group)
        return hidden_states

    def reference_forward(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.to_contiguous:
            hidden_states = zigzag_to_contiguous(hidden_states, distributed.cp_group)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear:
            hidden_states = self.linear_attn(hidden_states, cu_seqlens)
        else:
            hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        replay_idx = self.resolve_replay(hidden_states)
        hidden_states = residual + self.mlp.reference_forward(hidden_states, replay_idx)
        if self.to_zigzag:
            hidden_states = contiguous_to_zigzag(hidden_states, distributed.cp_group)
        return hidden_states


class Qwen35MoeModel(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, phase: int):
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

        self.rotary_emb = Qwen35MoeRotaryEmbedding(config)
        self.embed_tokens, self.norm, self.lm_head = None, None, None
        if stage_index == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if stage_index == stage_count - 1:
            self.norm = Qwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = BF16Linear(config.hidden_size, config.vocab_size, bias=False)
        sizes = pipeline_layer_sizes(config.num_hidden_layers, stage_count)
        self.layers = nn.ModuleDict(
            {
                str(i): Qwen35MoeDecoderLayer(config, i)
                for i in layer_partition(config.num_hidden_layers, stage_count, stage_index, sizes)
            }
        )

    def forward_posemb(
        self, S: int, cu_seqlens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = distributed.device
        assert cu_seqlens is None or distributed.cp_size == 1
        if cu_seqlens is not None:
            position_ids = document_positions(cu_seqlens, S)
            cos, sin = self.rotary_emb(S)
            return cos[position_ids].unsqueeze(0), sin[position_ids].unsqueeze(0)
        cp_size = distributed.cp_size
        spans = zigzag_spans(distributed.cp_rank, cp_size, S * cp_size)
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
