"""
<HF model card URL or name>.

STRUCTURAL OUTLINE - NOT a working file.

This template encodes only the framework-side contracts that every
PithTrain model must satisfy:
  - the decoder-layer stage interface (`forward_stage1` / `forward_stage3`
    / `forward_stage5` + `reference_forward`; see `pithtrain/models/interface.py`),
  - `@torch.compile(fullgraph=True)` on the hot compute regions,
  - shared-expert placement inside `forward_stage1_compute`,
  - the model-level `forward` (recorded pipeline path) vs `reference_forward`
    (plain autograd path) split, and the phase -> stage mapping.

Everything that defines *this* model (class names, attribute names,
tensor shapes, RoPE variant, attention kernel, expert activation, etc.)
is marked `TODO_HF` - fill in from HuggingFace's `modeling_<model>.py`
**before** looking at other PithTrain models.  Do not start from Qwen3 /
DeepSeek-V2 / GPT-OSS and rename; that path has produced silent
state_dict mismatches.  See `reference/conventions.md`.

The engine splits each decoder layer into five stages so it can overlap
one micro-batch's compute with another's communication:

  - Stage 1: pre-dispatch compute (LN + attention + routing).  `forward_stage1`.
  - Stage 2: dispatch all-to-all.  Driven by the engine, NOT a layer method.
  - Stage 3: expert compute.  `forward_stage3`.
  - Stage 4: combine all-to-all.  Driven by the engine, NOT a layer method.
  - Stage 5: post-combine compute (weighted sum + residual).  `forward_stage5`.
"""

import torch
import torch.nn.functional as F
from torch import nn

from pithtrain.contexts import distributed, training
from pithtrain.dualpipe.dualpipev import layer_partition
from pithtrain.dualpipe.execution import ChunkRecord, record_forward
from pithtrain.models.interface import RoutingInfo
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.ep_dispatch import prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from pithtrain.operators.ring_attention import ring_attention_func
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

# TODO_HF: import HF's config class, e.g.
#   from transformers.models.<model>.configuration_<model> import <Prefix>Config
# and annotate the classes below with it instead of the bare `config` parameter.


# -----------------------------------------------------------------------------
# Rotary Position Embedding
#
# TODO_HF: copy HF's <Prefix>RotaryEmbedding implementation (LLaMA-style,
# YaRN, linear-scaled, etc.).  Match HF's *class name* exactly - YaRN is
# an implementation detail, not part of the name.  The cos/sin cache is a
# non-persistent buffer sized to `max_position_embeddings`; `forward(S)`
# returns the leading `S` rows (the model applies per-document / per-CP
# position selection in `forward_posemb`, not here).
# -----------------------------------------------------------------------------


class HFPrefixRotaryEmbedding(nn.Module):  # TODO_HF rename to match HF class
    def __init__(self, config) -> None:
        super().__init__()
        inv_freq = self.compute_rope_params(config)
        self.set_cos_sin(config, inv_freq)

    def compute_rope_params(self, config) -> torch.Tensor:
        # TODO_HF: match HF's inv_freq computation (base, head_dim, any scaling).
        base, dim = config.rope_theta, config.head_dim
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def set_cos_sin(self, config, inv_freq: torch.Tensor) -> None:
        t = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin", emb.sin().to(torch.bfloat16), persistent=False)

    def forward(self, S: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:S], self.sin[:S]


# -----------------------------------------------------------------------------
# Experts
#
# TODO_HF: mirror HF's expert MLP - class name, attribute names, fused vs
# split projections, activation.
#
# Two storage strategies, pick the one that matches HF's checkpoint layout:
#
#   (a) Split projections -> `training.GroupedLinear(num_experts, in, out)`
#       (shown below, Qwen3-style).  GroupedLinear owns the fp8/bf16 backend
#       swap and the per-group GEMM; you just call it with the grouped-mm
#       metadata.
#
#   (b) Fused gate+up or per-expert bias -> raw `nn.Parameter` of shape
#       `[E, out, in]` plus a private `_group_linear` helper that dispatches
#       on `training.fp8` (GPT-OSS-style; see gpt_oss.py `GptOssExperts`).
#       If HF stores `[E, in, out]`, the transpose lives ONLY in dcp2hf,
#       never here.
#
# CRITICAL: if the expert forward has a bias-add or elementwise post-op,
# truncate `x` to `sum(ks)` before the grouped GEMM - grouped_mm may write
# NaN into padding rows.  See reference/pitfalls.md nan-padding.
# -----------------------------------------------------------------------------


class HFPrefixExperts(nn.Module):  # TODO_HF rename to match HF class
    def __init__(self, config, num_experts: int):
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        # TODO_HF: match HF's projection names and fusion.
        self.gate_proj = training.GroupedLinear(num_experts, hidden_size, moe_intermediate_size)
        self.up_proj = training.GroupedLinear(num_experts, hidden_size, moe_intermediate_size)
        self.down_proj = training.GroupedLinear(num_experts, moe_intermediate_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0])
        kwargs = dict(grouped_mm_offs=grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi)
        # TODO_HF: implement the HF expert forward (SwiGLU shown).
        g = self.gate_proj(x, **kwargs)
        u = self.up_proj(x, **kwargs)
        return self.down_proj(silu_mul(g, u), **kwargs)


# -----------------------------------------------------------------------------
# Router / Gate
#
# TODO_HF: match HF's class name EXACTLY (e.g. `<Prefix>TopKRouter`,
# `<Prefix>Gate`) and the attribute the MoE block uses to hold it
# (`self.router` vs `self.gate` - follow HF).  State_dict keys depend on
# both.  The router is a raw `nn.Parameter` weight (+ optional bias), not a
# `training.Linear`.
# -----------------------------------------------------------------------------


class HFPrefixGate(nn.Module):  # TODO_HF rename to match HF
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.load_balance_loss_fn = None
        self.router_replay = None
        self.weight = nn.Parameter(torch.empty((self.num_experts, config.hidden_size)))
        # TODO_HF: add `self.bias = nn.Parameter(torch.zeros(self.num_experts))` if HF has it.

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # TODO_HF: match HF's routing math exactly. Two common shapes:
        #   (a) softmax-then-topk (Qwen3, shown): topk over softmax scores, optional renorm.
        #   (b) topk-then-softmax (GPT-OSS): topk over raw logits, softmax over the topk only.
        logits = F.linear(hidden_states, self.weight, None)  # TODO_HF: add bias if HF has it
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        if self.router_replay is not None:
            topk_idx = self.router_replay(topk_idx)
            topk_weight = scores.gather(-1, topk_idx)
        if self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        if self.load_balance_loss_fn is None:
            return topk_idx, topk_weight, None
        lb_loss = self.load_balance_loss_fn(
            scores, topk_idx, self.num_experts, self.num_experts_per_tok
        )
        # Token-weight the injected lb gradient so train_step's 1/num_tokens grad scale leaves it
        # correctly normalized (it bypasses the token-weighted criterion). lb_loss stays unscaled.
        topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss * topk_weight.shape[0])
        return topk_idx, topk_weight, lb_loss


# -----------------------------------------------------------------------------
# MoE Block (the `mlp` attribute on the decoder layer)
#
# TODO_HF: match HF's MoE class name (e.g. `<Prefix>MoE`, `<Prefix>MoEBlock`).
# `experts_per_rank` is the EP-local expert count; EP degree comes from the
# `distributed` runtime context, never a constructor arg.
# -----------------------------------------------------------------------------


class HFPrefixMoE(nn.Module):  # TODO_HF rename to match HF
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts_per_rank = config.num_experts // distributed.ep_size
        self.experts = HFPrefixExperts(config, self.experts_per_rank)
        # TODO_HF: name this attribute per HF: self.router OR self.gate.
        self.gate = HFPrefixGate(config)
        # TODO_HF (if applicable): self.shared_experts = <SharedMLP>(config)

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, lb_loss = self.gate(hidden_states)  # TODO_HF: .gate vs .router
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        expert_idxs = topk_idx.view(-1)
        replicated_tokens = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.num_experts_per_tok, -1)
            .reshape(-1, hidden_states.shape[-1])
        )
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(replicated_tokens, expert_idxs, self.experts_per_rank)
        )
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]
        y = (outs.view(*topk_idx.shape, -1) * topk_weight.unsqueeze(dim=-1)).sum(dim=1).to(outs.dtype)
        return y.view(*orig_shape)


# -----------------------------------------------------------------------------
# Attention
#
# TODO_HF: copy HF's attention class structure.  The framework is BSHD end
# to end; only attention reshapes to the kernel's varlen layout.  Pick the
# kernel dispatch that matches HF's features:
#   - Context parallelism (cp_size > 1)     -> ring_attention_func
#   - Packed / variable-length (cu_seqlens) -> flash_attn_varlen_func (squeeze
#                                             batch dim of 1, unsqueeze back)
#   - Dense causal                          -> flash_attn_func
#   - Sliding window / attention sinks      -> pass window_size / learnable_sink
#                                             to the flash_attn kernels (see gpt_oss.py)
# -----------------------------------------------------------------------------


class HFPrefixAttention(nn.Module):  # TODO_HF rename to match HF
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        attention_bias = config.attention_bias
        self.q_proj = training.Linear(hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = training.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.v_proj = training.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.o_proj = training.Linear(self.num_heads * self.head_dim, hidden_size, bias=attention_bias)
        # TODO_HF: add model-specific Parameters/norms (e.g. q_norm/k_norm, per-head sinks).

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
        q_embed = (q * cos) + (HFPrefixAttention.rotate_half(q) * sin)
        k_embed = (k * cos) + (HFPrefixAttention.rotate_half(k) * sin)
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
        # TODO_HF: apply q_norm/k_norm here if HF has them.
        query_states, key_states = self.apply_rotary_posemb(query_states, key_states, rotary_posemb)
        if distributed.cp_size > 1:
            attn_output = ring_attention_func(query_states, key_states, value_states, sm_scale=self.scaling, cp_group=distributed.cp_group)
        elif cu_seqlens is not None:
            attn_output = flash_attn_varlen_func(query_states.squeeze(0), key_states.squeeze(0), value_states.squeeze(0), cu_seqlens, S, softmax_scale=self.scaling, causal=True).unsqueeze(0)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, softmax_scale=self.scaling, causal=True)
        attn_output = attn_output.reshape(B, S, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


# -----------------------------------------------------------------------------
# Decoder Layer - the stage-method surface (LayerProtocol).
# -----------------------------------------------------------------------------


class HFPrefixDecoderLayer(nn.Module):  # TODO_HF rename to match HF
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.idx = layer_id
        self.self_attn = HFPrefixAttention(config)
        self.mlp = HFPrefixMoE(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # ----- Stage 1 ---------------------------------------------------------
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
        # TODO_HF (if applicable): residual = residual + self.mlp.shared_experts(hidden_states)
        topk_idx, topk_weight, lb_loss = self.mlp.gate(hidden_states)  # TODO_HF: .gate vs .router
        return hidden_states, residual, topk_idx, topk_weight, lb_loss

    def forward_stage1(
        self,
        hidden_states: torch.Tensor,
        rotary_posemb: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RoutingInfo | None]:
        hidden_states, residual, topk_idx, topk_weight, lb_loss = self.forward_stage1_compute(hidden_states, rotary_posemb, cu_seqlens)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        dispatch_tokens, routing = prepare_dispatch(hidden_states, topk_idx, topk_weight, self.mlp.num_experts, distributed.ep_size, self.mlp.experts_per_rank, distributed.ep_group)
        return dispatch_tokens, residual, routing

    # ----- Stage 3 ---------------------------------------------------------
    def forward_stage3(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: torch.Tensor | None = None,
        expand_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if distributed.ep_size > 1:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
        del gathered_tokens
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        return padded_index_gather(outs, reverse_shuffle_idxs)

    # ----- Stage 5 ---------------------------------------------------------
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

    # ----- Reference (non-pipelined) ---------------------------------------
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
        return residual + hidden_states


# -----------------------------------------------------------------------------
# Full Model - one pipeline stage (chunk) of the model, wired for DualPipeV.
#
# DualPipeV cuts the model into `2 * pp_size` chunks and gives each pp rank
# two of them (a V-shape).  `phase` selects which chunk this instance is:
#   phase 0  -> the "descending" chunk (stage_index = pp_rank),
#   phase 1  -> the "ascending" chunk  (stage_index = 2*pp_size-1-pp_rank),
#   phase -1 -> the whole model as a single stage (non-pipelined reference).
# -----------------------------------------------------------------------------


class HFPrefixModel(nn.Module):  # TODO_HF rename: typically `<Prefix>Model`
    def __init__(self, config, phase: int):
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

        self.rotary_emb = HFPrefixRotaryEmbedding(config)
        self.embed_tokens, self.norm, self.lm_head = None, None, None
        if stage_index == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if stage_index == stage_count - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(i): HFPrefixDecoderLayer(config, i)
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
        position_ids = torch.cat([torch.arange(front_start, front_end, device=device), torch.arange(back_start, back_end, device=device)])
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
        return record_forward(self, hidden_states, self.chunk_record, cu_seqlens)

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
