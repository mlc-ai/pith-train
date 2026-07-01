"""
Qwen3.5-MoE (text tower).

A hybrid MoE: each decoder layer is a token mixer followed by a shared-expert
MoE block. The mixer is Gated DeltaNet (linear attention) on most layers and
full softmax attention (GQA) on the rest. Context parallelism is not supported.
"""

from dataclasses import fields
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

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
from pithtrain.operators.gated_delta_rule import gated_delta_rule
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

torch._dynamo.allow_in_graph(MoELoadBalanceLossInjector)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class Qwen35MoeRMSNorm(nn.Module):
    """
    RMSNorm with a ``(1 + weight)`` scale and zero-initialised weight.

    This is the GLM-style parameterisation Qwen3.5 ships: the released
    checkpoint stores ``weight`` centered at 0, and the effective scale is
    ``1 + weight``. A plain ``nn.RMSNorm`` (scale ``weight``) would load the
    checkpoint off by 1.0 on every norm, so this must be matched exactly.
    """

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
    """
    RMSNorm (standard ``weight`` scale, ones-init) gated by ``silu(gate)``.

    Used on the Gated DeltaNet output: ``rmsnorm(x) * silu(gate)``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        output = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * self.weight.float() * F.silu(gate.float())
        return output.type_as(x)


# ---------------------------------------------------------------------------
# Rotary position embedding (partial; full-attention layers only)
# ---------------------------------------------------------------------------


class Qwen35MoeRotaryEmbedding(nn.Module):
    """
    Partial rotary embedding.

    Only the first ``int(head_dim * partial_rotary_factor)`` dims of each head
    are rotated. For text-only inputs the model's interleaved MRoPE reduces to
    standard RoPE (the temporal/height/width position grids are identical), so
    a single position grid is sufficient here.
    """

    def __init__(self, head_dim: int, partial_rotary_factor: float, max_position_embeddings: int, base: float, device: Optional[torch.device] = None):
        super().__init__()
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply partial rotary embedding to BSHD query/key tensors.

    ``cos``/``sin`` have shape ``[batch, seq, rotary_dim]`` (``rotary_dim`` may
    be smaller than ``head_dim``); only the leading ``rotary_dim`` channels of
    each head are rotated, the rest pass through unchanged.
    """
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Gated DeltaNet (linear attention)
# ---------------------------------------------------------------------------


class Qwen35MoeGatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear-attention token mixer.
    """

    def __init__(self, config, layer_idx: int):
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
        self.conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))

        self.norm = Qwen35MoeRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        LinearCls = get_linear_cls()
        self.in_proj_qkv = LinearCls(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = LinearCls(self.hidden_size, self.value_dim, bias=False)
        # in_proj_a / in_proj_b have num_v_heads (=32) outputs: too small for the
        # 128-element FP8 block scaling, so keep them as plain bf16 Linear.
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = LinearCls(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [b, conv_dim, seq]
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)

        # Causal depthwise conv + SiLU, truncated back to seq_len (padding=k-1).
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[..., :seq_len]).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = F.normalize(query.reshape(batch_size, seq_len, -1, self.head_k_dim), dim=-1)
        key = F.normalize(key.reshape(batch_size, seq_len, -1, self.head_k_dim), dim=-1)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # .float() on A_log guards against -inf when loaded in low precision.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query, key = query.repeat_interleave(repeats, dim=2), key.repeat_interleave(repeats, dim=2)

        core_attn_out = gated_delta_rule(query, key, value, g, beta)

        core_attn_out, z = core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# Full softmax attention (output-gated GQA)
# ---------------------------------------------------------------------------


class Qwen35MoeAttention(nn.Module):
    """
    Grouped-query attention with a per-head sigmoid output gate.

    ``q_proj`` emits twice the query width: the first half is the query, the
    second half is the gate applied (after ``sigmoid``) to the attention
    output before ``o_proj``.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        attention_bias = getattr(config, "attention_bias", False)
        LinearCls = get_linear_cls()
        self.q_proj = LinearCls(self.hidden_size, self.num_heads * self.head_dim * 2, bias=attention_bias)
        self.k_proj = LinearCls(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.v_proj = LinearCls(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.o_proj = LinearCls(self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias)

        self.q_norm = Qwen35MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states, gate = torch.chunk(self.q_proj(hidden_states).view(bsz, seq_len, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(bsz, seq_len, -1)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim))
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = flash_attn_func(query_states, key_states, value_states, softmax_scale=self.scaling, causal=True)

        attn_output = attn_output.reshape(bsz, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# MoE: dense shared expert, grouped experts, router
# ---------------------------------------------------------------------------


class Qwen35MoeMLP(nn.Module):
    """
    Dense SwiGLU MLP (used for the shared expert).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        LinearCls = get_linear_cls()
        self.gate_proj = LinearCls(hidden_size, intermediate_size, bias=False)
        self.up_proj = LinearCls(hidden_size, intermediate_size, bias=False)
        self.down_proj = LinearCls(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))


class Qwen35MoeExperts(nn.Module):
    """
    Routed experts: a fused gate/up grouped linear plus a down grouped linear.

    ``gate_up_proj`` is one grouped GEMM producing ``[.., 2*inter]``, split
    non-interleaved (gate = first half, up = second half). FP8 vs BF16 and the
    quantized-weight cache are handled by ``get_group_linear_cls()``.
    """

    def __init__(self, num_experts: int, hidden_size: int, moe_intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        GroupLinearCls = get_group_linear_cls()
        self.gate_up_proj = GroupLinearCls(num_experts, hidden_size, 2 * moe_intermediate_size)
        self.down_proj = GroupLinearCls(num_experts, moe_intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor, grouped_mm_offs: torch.Tensor, ks: list | None = None, ks_tensor: torch.Tensor | None = None) -> torch.Tensor:
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0])
        kwargs = dict(grouped_mm_offs=grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi)
        gate_up = self.gate_up_proj(x, **kwargs)
        # Non-interleaved fused gate/up: column slices are non-contiguous, and silu_mul requires contiguous inputs.
        gate = gate_up[:, : self.moe_intermediate_size].contiguous()
        up = gate_up[:, self.moe_intermediate_size :].contiguous()
        return self.down_proj(silu_mul(gate, up), **kwargs)


class Qwen35MoeTopKRouter(nn.Module):
    """
    Softmax-then-top-k router with sum-normalised weights.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.load_balance_loss_fn = None
        self.router_replay = None
        self.weight = nn.Parameter(torch.empty((config.num_experts, config.hidden_size)), requires_grad=True)

    @torch.compile(fullgraph=True)
    def compute(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        if self.router_replay is not None:
            topk_idx = self.router_replay(topk_idx)
            topk_weight = scores.gather(-1, topk_idx)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        if self.training and self.load_balance_loss_fn is not None:
            lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.num_experts, self.num_experts_per_tok)
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        return topk_idx, topk_weight


class Qwen35MoeSparseMoeBlock(nn.Module):
    """
    Routed experts + a sigmoid-gated shared expert.
    """

    def __init__(self, config, ep_size: int = 1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # ep_size sizes the local expert weights (per-instance config); the ep_group
        # collective is read from the distributed context at its point of use.
        self.ep_size = ep_size
        self.experts_per_rank = self.num_experts // ep_size

        self.gate = Qwen35MoeTopKRouter(config)
        self.experts = Qwen35MoeExperts(self.experts_per_rank, self.hidden_size, config.moe_intermediate_size)
        self.shared_expert = Qwen35MoeMLP(self.hidden_size, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def shared_out(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid-gated shared-expert output (per-token scalar gate).
        """
        shared = self.shared_expert(hidden_states)
        return torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Reference (non-pipelined, ep_size==1) full MoE forward.
        """
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return y + self.shared_out(identity)

    def moe_infer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        assert self.ep_size == 1, "Reference implementation only supports ep_size=1"
        expert_idxs = topk_ids.view(-1)
        sorted_tokens = x.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, x.shape[-1])
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]
        final_out = (outs.view(*topk_ids.shape, -1) * topk_weight.unsqueeze(dim=-1)).sum(dim=1).to(outs.dtype)
        return final_out


# ---------------------------------------------------------------------------
# Decoder layer (5-stage DualPipeV split)
# ---------------------------------------------------------------------------


class Qwen35MoeDecoderLayer(nn.Module):
    """
    A single hybrid decoder layer split into the 5 DualPipeV stages.

    Every layer is MoE. The token mixer is either ``linear_attn`` (Gated
    DeltaNet) or ``self_attn`` (full attention), selected by ``layer_type``.
    """

    def __init__(self, config, layer_idx: int, ep_size: int = 1):
        super().__init__()
        self.idx = layer_idx
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"

        if self.is_linear:
            self.linear_attn = Qwen35MoeGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen35MoeAttention(config)

        self.mlp = Qwen35MoeSparseMoeBlock(config, ep_size=ep_size)

        self.input_layernorm = Qwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.compile(fullgraph=True)
    def _forward_attn_compute(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear:
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings=self._position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Shared expert folds into the residual here so it overlaps the stage-2
        # all-to-all dispatch of the routed tokens.
        residual = residual + self.mlp.shared_out(hidden_states)
        return hidden_states, residual

    def forward_attn(self, hidden_states: torch.Tensor) -> ForwardAttnOutput:
        """
        Stage 1: LN + mixer + LN + shared expert + routing/dispatch prep.
        """
        hidden_states, residual = self._forward_attn_compute(hidden_states)

        topk_ids, topk_weight = self.mlp.gate(hidden_states)
        sorted_tokens, idxs, expert_idxs, expand_idx, dedup_input_splits, dedup_output_splits, input_splits, output_splits = moe_ep_prepare_dispatch(hidden_states, topk_ids, self.mlp.num_experts, self.mlp.ep_size, self.mlp.experts_per_rank, distributed.ep_group)
        return ForwardAttnOutput(sorted_tokens, idxs, topk_weight, output_splits, input_splits, expert_idxs, residual, expand_idx, dedup_input_splits, dedup_output_splits)

    def forward_mlp(self, gathered_tokens: torch.Tensor, expert_idxs: Optional[torch.Tensor] = None, expand_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Stage 3: scatter-by-expert + grouped GEMM + unshuffle.
        """
        assert expert_idxs is not None
        if expand_idx is not None:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
        del gathered_tokens
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = padded_index_gather(outs, reverse_shuffle_idxs)
        return outs

    @torch.compile(fullgraph=True)
    def forward_aggregate(self, moe_outs: torch.Tensor, moe_local_idxs: Optional[torch.Tensor], topk_weight: Optional[torch.Tensor], residual: torch.Tensor) -> torch.Tensor:
        """
        Stage 5: weighted expert sum + residual (shared expert already in residual).
        """
        if self.mlp.ep_size > 1:
            assert moe_local_idxs is not None
            seq_len, topk = topk_weight.shape
            permuted_probs = topk_weight.view(-1)[moe_local_idxs]
            token_indices = moe_local_idxs // topk
            weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
            hidden_states = moe_outs.new_zeros(seq_len, moe_outs.shape[-1])
            hidden_states.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
            hidden_states = hidden_states.view(*residual.shape)
        else:
            assert moe_local_idxs is None
            final_out = moe_outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(dim=-1)
            hidden_states = final_out.sum(dim=1).to(moe_outs.dtype).view(*residual.shape)

        return residual + hidden_states

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Non-pipelined eager forward for correctness validation / inference.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear:
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings=self._position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Qwen35MoeModel(nn.Module):
    """
    Qwen3.5-MoE text model for DualPipeV pipeline + expert parallelism.
    """

    def __init__(self, config, num_stages: int, stage_id: int):
        super().__init__()
        if distributed.cp_size > 1:
            raise NotImplementedError("Qwen35MoeModel does not support context parallelism (linear attention needs a bespoke sequence-sharded recurrence).")
        self.config = config
        self.stage_id = stage_id
        self.num_stages = num_stages

        hidden_size = config.hidden_size
        head_dim = config.head_dim
        vocab_size = config.vocab_size
        rope_params = config.rope_parameters
        ep_size = getattr(config, "ep_size", 1)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size) if stage_id == 0 else None

        num_local_layers = layer_partition(config.num_hidden_layers, num_stages)
        layer_id_begin = sum(num_local_layers[:stage_id])
        layer_id_end = layer_id_begin + num_local_layers[stage_id]

        self.layers = nn.ModuleDict({
            str(i): Qwen35MoeDecoderLayer(config, layer_idx=i, ep_size=ep_size)
            for i in range(layer_id_begin, layer_id_end)
        })

        if stage_id == num_stages - 1:
            self.norm = Qwen35MoeRMSNorm(hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

        self.rotary_emb = Qwen35MoeRotaryEmbedding(head_dim, partial_rotary_factor=rope_params.get("partial_rotary_factor", 1.0), max_position_embeddings=config.max_position_embeddings, base=rope_params["rope_theta"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate_tensors: Optional[IntermediateTensors] = getattr(self, "_intermediate_tensors", None)

        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(hidden_states)

        _, seq_len, _ = hidden_states.shape

        # No CP: a single contiguous position grid. Full-attention layers read
        # _position_embeddings; linear-attention layers ignore it.
        position_ids = torch.arange(seq_len, device=hidden_states.device)
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        position_embeddings = (cos[position_ids].unsqueeze(0), sin[position_ids].unsqueeze(0))
        for _, layer in self.layers.items():
            layer._position_embeddings = position_embeddings

        if intermediate_tensors is None:
            for _, layer in self.layers.items():
                ret = decoder_layer_forward(layer, hidden_states)
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
            ret = decoder_layer_forward(layer, hidden_states)
            if len(ret) == 2:
                hidden_states, layer_record = ret
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(layer_record):
                    src_rec = getattr(layer_record, field.name)
                    dst_rec = getattr(dst, field.name)
                    for rf in fields(src_rec):
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
    def backward(module: "Qwen35MoeModel", dy: Optional[List[torch.Tensor]], loss: Optional[torch.Tensor], intermediate_tensors: IntermediateTensors):
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
            for rf in fields(record):
                setattr(record, rf.name, None)
            final_grads = (None,)

        return final_grads
