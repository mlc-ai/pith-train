"""
<HF model card URL or name>.

STRUCTURAL OUTLINE — NOT a working file.

This template encodes only the framework-side contracts that every
PithTrain model must satisfy:
  • the 5-stage decoder-layer interface,
  • `@torch.compile(fullgraph=True)` on the three hot regions,
  • shared-expert placement inside `_forward_attn_compute`,
  • the model-level `forward` stage-record copy and `backward`.

Everything that defines *this* model (class names, attribute names,
tensor shapes, RoPE variant, attention kernel, expert activation, etc.)
is marked `TODO_HF` — fill in from HuggingFace's `modeling_<model>.py`
**before** looking at other PithTrain models.  Do not start from Qwen3 /
DeepSeek-V2 / GPT-OSS and rename; that path has produced silent
state_dict mismatches.  See `reference/conventions.md`.
"""

from dataclasses import fields
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from pithtrain.dualpipe.execution import EpilogArgs, IntermediateTensors, PrologArgs, PrologOuts
from pithtrain.dualpipe.layer_partition import layer_partition
from pithtrain.dualpipe.modeling import decoder_layer_backward, decoder_layer_forward
from pithtrain.dualpipe.utils import run_backward
from pithtrain.layers.factory import ModelImplMode, get_linear_cls
from pithtrain.models.interface import ForwardAttnOutput
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.ep_dispatch import moe_ep_prepare_dispatch
from pithtrain.operators.token_scatter import padded_index_gather, scatter_for_grouped_gemm

torch._dynamo.allow_in_graph(MoELoadBalanceLossInjector)


# -----------------------------------------------------------------------------
# Rotary Position Embedding
#
# TODO_HF: copy HF's <Prefix>RotaryEmbedding implementation (LLaMA-style,
# YaRN, linear-scaled, etc.).  Match HF's *class name* exactly — YaRN is
# an implementation detail, not part of the name.
# -----------------------------------------------------------------------------


class HFPrefixRotaryEmbedding(nn.Module):  # TODO_HF rename to match HF class
    """<one-line summary>."""

    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # TODO_HF: build cos/sin cache identical to HF.

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO_HF: return cos[:seq_len], sin[:seq_len] in x's dtype.
        raise NotImplementedError


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# -----------------------------------------------------------------------------
# Experts
#
# TODO_HF: mirror HF's Experts module — class name, attribute names, fused
# vs split projections.  Storage layout must be `[E, out, in]` (training-
# framework consensus; see reference/conventions.md).  If HF stores
# `[E, in, out]`, the transpose lives ONLY in dcp2hf, never here.
#
# If HF fuses gate+up into `gate_up_proj`, keep it fused.  Split the
# output at forward time for different post-ops.
#
# CRITICAL: if expert forward has bias-add or elementwise post-ops,
# truncate x to sum(ks) before the grouped GEMM (see reference/pitfalls.md).
# -----------------------------------------------------------------------------


class HFPrefixExperts(nn.Module):  # TODO_HF rename to match HF class
    """<one-line summary>."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # TODO_HF: match HF's weight shapes and names.
        # Example (GPT-OSS style, fused gate+up, `[E, out, in]` layout,
        # per-expert bias):
        #   self.gate_up_proj = nn.Parameter(
        #       torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        #   self.gate_up_proj_bias = nn.Parameter(
        #       torch.zeros(num_experts, 2 * intermediate_size))
        #   self.down_proj = nn.Parameter(
        #       torch.empty(num_experts, hidden_size, intermediate_size))
        #   self.down_proj_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def _grouped_mm(
        self, x: torch.Tensor, weight: nn.Parameter, offs: torch.Tensor
    ) -> torch.Tensor:
        # Empty-input guard: grouped_mm dislikes an M=0 input.
        if x.shape[0] == 0:
            return x @ weight[0].transpose(-2, -1)
        # `[E, out, in]` storage → transpose view for F.grouped_mm.
        return F.grouped_mm(x, weight.transpose(-2, -1), offs=offs)

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NaN-padding protection: truncate to sum(ks) before matmul.
        # See reference/pitfalls.md §nan-padding.
        if ks is not None:
            actual_m = sum(ks)
            if actual_m < x.shape[0]:
                x = x[:actual_m]

        # TODO_HF: implement the HF expert forward.
        # Typical SwiGLU pattern:
        #   gate_up = self._grouped_mm(x, self.gate_up_proj, grouped_mm_offs)
        #   gate, up = gate_up[:, ::2], gate_up[:, 1::2]     # if fused
        #   activated = silu(gate) * up    # or clamped-SwiGLU, etc.
        #   out = self._grouped_mm(activated, self.down_proj, grouped_mm_offs)
        #   return out
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Router / Gate
#
# TODO_HF: match HF's class name EXACTLY (e.g. `<Prefix>TopKRouter`,
# `<Prefix>Gate`, etc.) and the attribute name in the MLP module
# (`self.mlp.router` or `self.mlp.gate`).  State_dict keys depend on this.
# -----------------------------------------------------------------------------


class HFPrefixRouter(nn.Module):  # TODO_HF rename to match HF
    """<one-line summary>."""

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.load_balance_loss_fn = None  # set externally by setup_model
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        # TODO_HF: add `self.bias = nn.Parameter(torch.zeros(num_experts))` if HF has it.

    @torch.compile(fullgraph=True)
    def compute(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Top-k routing — must be fullgraph-compilable.

        TODO_HF: match HF's exact routing math (pre-softmax top-k,
        post-softmax top-k, norm_topk_prob, etc.).
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        # TODO_HF: compute logits, top-k idx, top-k weight per HF's spec.
        # Common patterns:
        #   (a) GPT-OSS: softmax over topk_logits only
        #       logits = F.linear(hidden_states, self.weight, self.bias)
        #       topk_logits, topk_idx = torch.topk(logits, k=..., dim=-1, sorted=True)
        #       topk_weight = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        #
        #   (b) Qwen3: softmax-then-topk, optional renormalise
        #       scores = F.linear(hidden_states, self.weight, None).softmax(
        #           dim=-1, dtype=torch.float32)
        #       topk_weight, topk_idx = torch.topk(scores, k=..., dim=-1, sorted=False)
        #       if self.norm_topk_prob:
        #           topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        logits = F.linear(hidden_states, self.weight, None)  # TODO_HF: add bias if HF has it
        raise NotImplementedError  # complete per HF's spec

        # Load-balance loss injection — copy this block verbatim.
        if self.training and self.load_balance_loss_fn is not None:
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            lb_loss = self.load_balance_loss_fn(
                scores, topk_idx, self.num_experts, self.num_experts_per_tok
            )
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        return topk_idx, topk_weight


# -----------------------------------------------------------------------------
# MoE Block (the `mlp` attribute on the decoder layer)
#
# TODO_HF: match HF's MLP class name (e.g. `<Prefix>MLP`,
# `<Prefix>MoEBlock`) and the attribute that holds the router
# (`self.router` vs `self.gate` — follow HF).
# -----------------------------------------------------------------------------


class HFPrefixMLP(nn.Module):  # TODO_HF rename to match HF
    """MoE block with EP support."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        intermediate_size: int,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Protocol contract (DecoderLayerMlpProtocol):
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.ep_rank = ep_group.rank() if ep_group is not None else 0
        self.experts_per_rank = num_experts // ep_size

        self.experts = HFPrefixExperts(self.experts_per_rank, hidden_size, intermediate_size)
        # TODO_HF: name this attribute per HF: self.router OR self.gate.
        self.router = HFPrefixRouter(hidden_size, num_experts, num_experts_per_tok)

        # TODO_HF (if applicable): self.shared_experts = <SharedMLP>(...)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Non-pipelined reference forward — used by reference_forward + test."""
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.router(hidden_states)  # TODO_HF: .gate vs .router
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self._moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return y

    def _moe_infer(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        assert self.ep_size == 1, "Reference implementation only supports ep_size=1"
        expert_idxs = topk_ids.view(-1)
        sorted_tokens = (
            x.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, x.shape[-1])
        )
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)
        )
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]

        final_out = (
            (outs.view(*topk_ids.shape, -1) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(outs.dtype)
        )
        return final_out


# -----------------------------------------------------------------------------
# Attention
#
# TODO_HF: copy HF's attention class structure.  Pick the kernel that
# matches HF's features:
#   • Standard GQA / MHA, no sinks → flash_attn_func
#   • Sliding window or sinks        → flex_attention (+ LSE renorm post-op)
#   • Context parallelism needed     → ring_attention_func (conditional unwrap)
#
# DO NOT use `score_mod` closures that capture Parameters.  If HF does
# something like "add a learned bias to attention scores", do it as a
# post-op on the kernel output with the LSE trick — see reference/compile.md.
# -----------------------------------------------------------------------------


class HFPrefixAttention(nn.Module):  # TODO_HF rename to match HF
    """<one-line summary>."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        LinearCls = get_linear_cls()
        self.q_proj = LinearCls(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = LinearCls(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = LinearCls(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = LinearCls(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        # TODO_HF: add any model-specific Parameters (e.g. per-head sinks).

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        # TODO_HF: additional kwargs your kernel needs (block_mask, etc.)
    ) -> torch.Tensor:
        # TODO_HF: Q/K/V projection, RoPE, kernel call, o_proj.
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Decoder Layer — the 5-stage protocol surface.
# -----------------------------------------------------------------------------


class HFPrefixDecoderLayer(nn.Module):  # TODO_HF rename to match HF
    """Implements `DecoderLayerProtocol` (pithtrain/models/interface.py)."""

    def __init__(
        self,
        # TODO_HF: the parameter list should reflect the HF config.
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
        attention_bias: bool,
        layer_idx: int,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.idx = layer_idx  # protocol field — nvtx range labels use this
        self.hidden_size = hidden_size

        self.self_attn = HFPrefixAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
        )

        self.mlp = HFPrefixMLP(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            ep_size=ep_size,
            ep_group=ep_group,
        )

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Conditional unwrap — only when a self-compiling attention kernel
        # is active.  Leave this out if HF's attention doesn't need it.
        # Example for ring attention:
        #   if getattr(self.self_attn, "use_ring_attn", False):
        #       self._forward_attn_compute = self._forward_attn_compute.__wrapped__.__get__(
        #           self, type(self),
        #       )

    # ----- Stage 1 ---------------------------------------------------------
    @torch.compile(fullgraph=True)
    def _forward_attn_compute(self, hidden_states: torch.Tensor):
        """LN + Attn + LN (+ shared experts if any).  Must be fullgraph-compilable.

        NOTE: shared experts — if the model has them — are folded into
        residual HERE, not in forward_aggregate.  Their compute overlaps
        with the stage-2 all-to-all dispatch of the routed tokens.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError("Position embeddings must be set before calling forward_attn")

        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Shared-expert fold (keep this stanza; it's a no-op when no shared experts).
        if hasattr(self.mlp, "shared_experts"):
            residual = residual + self.mlp.shared_experts(hidden_states)

        return hidden_states, residual

    def forward_attn(self, hidden_states: torch.Tensor) -> ForwardAttnOutput:
        hidden_states, residual = self._forward_attn_compute(hidden_states)

        # TODO_HF (rare): dense fallback for non-MoE layers.  Most pure-MoE
        # models don't need this; remove if every layer is MoE.
        if not hasattr(self.mlp, "experts"):
            return ForwardAttnOutput(
                hidden_states,
                None,
                None,
                None,
                None,
                None,
                residual,
            )

        # Routing + dispatch prep
        topk_ids, topk_weight = self.mlp.router(hidden_states)  # TODO_HF: .gate vs .router
        (
            sorted_tokens,
            idxs,
            expert_idxs,
            expand_idx,
            dedup_input_splits,
            dedup_output_splits,
            input_splits,
            output_splits,
        ) = moe_ep_prepare_dispatch(
            hidden_states,
            topk_ids,
            self.mlp.num_experts,
            self.mlp.ep_size,
            self.mlp.experts_per_rank,
            self.mlp.ep_group,
        )
        return ForwardAttnOutput(
            sorted_tokens,
            idxs,
            topk_weight,
            output_splits,
            input_splits,
            expert_idxs,
            residual,
            expand_idx,
            dedup_input_splits,
            dedup_output_splits,
        )

    # ----- Stage 3 ---------------------------------------------------------
    def forward_mlp(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not hasattr(self.mlp, "experts"):
            assert expert_idxs is None
            return self.mlp(gathered_tokens)

        assert expert_idxs is not None
        if expand_idx is not None:
            # `padded_index_gather`, NOT raw gathered_tokens[expand_idx].
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
        )
        del gathered_tokens
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = padded_index_gather(outs, reverse_shuffle_idxs)
        return outs

    # ----- Stage 5 ---------------------------------------------------------
    @torch.compile(fullgraph=True)
    def forward_aggregate(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted expert sum + residual.  Shared experts are ALREADY folded into
        residual inside `_forward_attn_compute` — do NOT re-add them here.
        """
        if hasattr(self.mlp, "experts"):
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
                final_out = moe_outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)
                hidden_states = final_out.sum(dim=1).to(moe_outs.dtype).view(*residual.shape)
        else:
            assert moe_local_idxs is None and topk_weight is None
            hidden_states = moe_outs

        return residual + hidden_states

    # ----- Reference (non-pipelined) ---------------------------------------
    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Plain forward — used by single-GPU test and the reference model in test_fsdp."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError("Position embeddings must be set before calling reference_forward")

        # TODO_HF: if self_attn has a ring-attention path, disable it here (see
        # Qwen3 / DeepSeek-V2 for the `_disable_ring_attn` flag pattern).
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# -----------------------------------------------------------------------------
# Full Model — one pipeline stage of the model, wired for DualPipeV.
# -----------------------------------------------------------------------------


class HFPrefixModel(nn.Module):  # TODO_HF rename: typically `<Prefix>Model`
    """<one-line summary>."""

    def __init__(
        self,
        config,
        num_stages: int,
        stage_id: int,
        cp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.config = config
        self.stage_id = stage_id
        self.num_stages = num_stages

        # TODO_HF: read fields off `config`.  Use `getattr(config, X, default)`
        # for fields that are optional in HF's config.
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        intermediate_size = config.intermediate_size
        num_experts = getattr(config, "num_local_experts", None) or config.num_experts
        num_experts_per_tok = config.num_experts_per_tok
        rms_norm_eps = config.rms_norm_eps
        attention_bias = getattr(config, "attention_bias", False)
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        rope_theta = getattr(config, "rope_theta", 10000.0)

        ep_size = getattr(config, "ep_size", 1)

        # First stage has embed_tokens; last stage has norm + lm_head.
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size) if stage_id == 0 else None

        num_local_layers = layer_partition(config.num_hidden_layers, num_stages)
        layer_id_begin = sum(num_local_layers[:stage_id])
        layer_id_end = layer_id_begin + num_local_layers[stage_id]

        # nn.ModuleDict keyed by absolute layer_id string (required by FSDP wrapping).
        self.layers = nn.ModuleDict(
            {
                str(i): HFPrefixDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    rms_norm_eps=rms_norm_eps,
                    attention_bias=attention_bias,
                    layer_idx=i,
                    ep_size=ep_size,
                    ep_group=ep_group,
                )
                for i in range(layer_id_begin, layer_id_end)
            }
        )

        if stage_id == num_stages - 1:
            self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

        self.rotary_emb = HFPrefixRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        # TODO_HF: any other model-level state (e.g. block_mask cache for
        # flex_attention).

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.  Two modes:
        • Plain (intermediate_tensors is None): reference / eager inference.
        • Recorded (intermediate_tensors is set): the 5-stage path — we
          must copy every stage record into the pre-allocated slot.
        """
        intermediate_tensors: Optional[IntermediateTensors] = getattr(
            self, "_intermediate_tensors", None
        )

        if self.embed_tokens is not None:
            input_ids = hidden_states
            hidden_states = self.embed_tokens(input_ids)

        bsz, seq_len, _ = hidden_states.shape

        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        position_embeddings = (cos[:seq_len].unsqueeze(0), sin[:seq_len].unsqueeze(0))

        for layer_idx_str, layer in self.layers.items():
            layer._position_embeddings = position_embeddings
            # TODO_HF: set any other per-forward state on layers here
            # (e.g. layer._block_mask, layer type flags, etc.).

        # Plain forward — falls into this branch under `reference_forward`
        # and under plain eager inference.
        if intermediate_tensors is None:
            for _, layer in self.layers.items():
                ret = decoder_layer_forward(layer, hidden_states)
                hidden_states = ret[0] if isinstance(ret, tuple) else ret
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
                hidden_states = self.lm_head(hidden_states)
            return hidden_states

        # Recorded forward — the 5-stage path.  The stage-record copy is
        # critical: iterate EVERY field of every record, including stages
        # 2 and 4 which only have `.ctx`.  See reference/protocol.md.
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
    def backward(
        module: "HFPrefixModel",
        dy: Optional[List[torch.Tensor]],
        loss: Optional[torch.Tensor],
        intermediate_tensors: IntermediateTensors,
    ):
        """Backward pass.  Copy this verbatim; only the class name in
        `module: "HFPrefixModel"` changes per model.
        """
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
        for layer, intermediate_tensors_layer in zip(
            reversed(layers_list), reversed(intermediate_tensors.layers)
        ):
            dx = (decoder_layer_backward(layer, dx, loss, intermediate_tensors_layer),)

        final_grads = dx
        if module.embed_tokens is not None:
            record = intermediate_tensors.prolog
            run_backward(record.outs, dx)
            for rf in fields(record):
                setattr(record, rf.name, None)
            final_grads = (None,)

        return final_grads
