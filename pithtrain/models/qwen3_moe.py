"""Qwen/Qwen3-30B-A3B."""

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
from pithtrain.layers.factory import ModelImplMode, get_group_linear_cls, get_linear_cls
from pithtrain.models.interface import ForwardAttnOutput
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.ep_dispatch import moe_ep_prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.ring_attention import ring_attention_func
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

torch._dynamo.allow_in_graph(MoELoadBalanceLossInjector)


class Qwen3MoeRotaryEmbedding(nn.Module):
    """Standard Rotary Position Embedding for Qwen3."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 40960,
        base: float = 1000000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the cos/sin cache up to ``seq_len`` positions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, used only for dtype.
        seq_len : int
            Required cache length (global, i.e. includes CP offset).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Cosine and sine tables of shape ``[seq_len, head_dim]``.
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape [batch, seq_len, num_heads, head_dim].
    k : torch.Tensor
        Key tensor of shape [batch, seq_len, num_kv_heads, head_dim].
    cos : torch.Tensor
        Cosine embedding of shape [batch, seq_len, head_dim].
    sin : torch.Tensor
        Sine embedding of shape [batch, seq_len, head_dim].

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3MoeMLP(nn.Module):
    """Standard dense MLP for Qwen3 (used when layer is not MoE)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        LinearCls = get_linear_cls()
        self.gate_proj = LinearCls(hidden_size, intermediate_size, bias=False)
        self.up_proj = LinearCls(hidden_size, intermediate_size, bias=False)
        self.down_proj = LinearCls(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))


class Qwen3MoeExperts(nn.Module):
    """Expert layers using grouped linear operations for efficient computation."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size

        GroupLinearCls = get_group_linear_cls()
        self.gate_proj = GroupLinearCls(num_experts, hidden_size, moe_intermediate_size)
        self.up_proj = GroupLinearCls(num_experts, hidden_size, moe_intermediate_size)
        self.down_proj = GroupLinearCls(num_experts, moe_intermediate_size, hidden_size)

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


class Qwen3MoeGate(nn.Module):
    """Top-K routing gate for MoE with softmax normalization."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.load_balance_loss_fn = None
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size)), requires_grad=True)

    @torch.compile(fullgraph=True)
    def compute(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Gate math + lb_loss injection (compiled).

        Includes linear + softmax + topk + normalize + load-balance loss
        computation + injection. Only MoELoadBalanceLossTracker.add() (a
        class-level side effect) stays outside in forward().

        Note: norm_topk_prob is applied before lb_loss injection. This is
        safe because MoELoadBalanceLossInjector is identity in forward and
        ones_like(lb_loss) in backward - gradient on topk_weight is unchanged.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            topk_idx, topk_weight, lb_loss (None when not training or no loss fn).
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)

        if self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        if self.training and self.load_balance_loss_fn is not None:
            lb_loss = self.load_balance_loss_fn(
                scores, topk_idx, self.num_experts, self.num_experts_per_tok
            )
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and expert indices.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [batch, seq_len, hidden_size].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            topk_idx: Expert indices of shape [batch*seq_len, num_experts_per_tok].
            topk_weight: Routing weights of shape [batch*seq_len, num_experts_per_tok].
        """
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)

        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)

        return topk_idx, topk_weight


class Qwen3MoeMoE(nn.Module):
    """Mixture of Experts block with expert parallelism support."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.ep_size = ep_size
        self.ep_group = ep_group
        self.ep_rank = ep_group.rank() if ep_group is not None else 0
        self.experts_per_rank = num_experts // ep_size

        self.experts = Qwen3MoeExperts(
            self.experts_per_rank,
            hidden_size,
            moe_intermediate_size,
        )
        self.gate = Qwen3MoeGate(hidden_size, num_experts, num_experts_per_tok, norm_topk_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return y

    def moe_infer(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """MoE inference with grouped GEMM."""
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


class Qwen3MoeAttention(nn.Module):
    """Grouped Query Attention (GQA) for Qwen3 using Flash Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5
        self.cp_group = cp_group
        self.use_ring_attn = cp_group is not None and cp_group.size() > 1

        LinearCls = get_linear_cls()
        self.q_proj = LinearCls(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = LinearCls(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = LinearCls(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = LinearCls(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass for GQA attention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [batch, seq_len, hidden_size].
        position_embeddings : Tuple[torch.Tensor, torch.Tensor]
            Tuple of (cos, sin) for rotary embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not self.use_ring_attn:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=self.scaling,
                causal=True,
            )
        else:
            attn_output = ring_attention_func(
                query_states,
                key_states,
                value_states,
                sm_scale=self.scaling,
                cp_group=self.cp_group,
            )

        attn_output = attn_output.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3MoeDecoderLayer(nn.Module):
    """
    Decoder layer for Qwen3 MoE model.

    Implements the required protocol methods for DualPipeV:
    - forward_attn: LN + Attn + LN + Expert selection
    - forward_mlp: MLP/Expert computation
    - forward_aggregate: Weighted expert output + residual
    - reference_forward: Standard forward pass
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        rms_norm_eps: float,
        attention_bias: bool,
        norm_topk_prob: bool,
        layer_idx: int,
        decoder_sparse_step: int = 1,
        mlp_only_layers: Optional[List[int]] = None,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.idx = layer_idx
        self.hidden_size = hidden_size

        self.self_attn = Qwen3MoeAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            cp_group=cp_group,
        )

        mlp_only_layers = mlp_only_layers or []
        use_moe = (
            num_experts > 0
            and (layer_idx + 1) % decoder_sparse_step == 0
            and layer_idx not in mlp_only_layers
        )

        if use_moe:
            self.mlp = Qwen3MoeMoE(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
                norm_topk_prob=norm_topk_prob,
                ep_size=ep_size,
                ep_group=ep_group,
            )
        else:
            self.mlp = Qwen3MoeMLP(hidden_size, intermediate_size)

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    @torch.compile(fullgraph=True)
    def _forward_attn_compute(
        self,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError("Position embeddings must be set before calling forward_attn")

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        return hidden_states, residual

    def forward_attn(
        self,
        hidden_states: torch.Tensor,
    ) -> ForwardAttnOutput:
        """LN + Attn + LN + Expert selection."""
        hidden_states, residual = self._forward_attn_compute(hidden_states)

        if isinstance(self.mlp, Qwen3MoeMLP):
            return ForwardAttnOutput(
                hidden_states,  # sorted_tokens
                None,
                None,
                None,
                None,
                None,  # expert_idxs
                residual,
            )

        topk_ids, topk_weight = self.mlp.gate(hidden_states)
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

    def forward_mlp(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MLP/Expert forward."""
        if isinstance(self.mlp, Qwen3MoeMLP):
            assert expert_idxs is None
            return self.mlp(gathered_tokens)

        assert expert_idxs is not None
        if expand_idx is not None:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
        )
        del gathered_tokens  # free expanded tokens; no longer needed after scatter
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = padded_index_gather(outs, reverse_shuffle_idxs)
        return outs

    @torch.compile(fullgraph=True)
    def forward_aggregate(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted expert output + residual connection."""
        if isinstance(self.mlp, Qwen3MoeMoE):
            if self.mlp.ep_size > 1:
                assert moe_local_idxs is not None
                seq_len, topk = topk_weight.shape
                # Memory-efficient equivalent of
                # new_x[moe_local_idxs] = moe_outs followed by weighted sum.
                permuted_probs = topk_weight.view(-1)[moe_local_idxs]
                token_indices = moe_local_idxs // topk
                weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
                hidden_states = moe_outs.new_zeros(seq_len, moe_outs.shape[-1])
                hidden_states.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
                hidden_states = hidden_states.view(*residual.shape)
            else:
                assert moe_local_idxs is None
                new_x = moe_outs
                final_out = new_x.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(dim=-1)
                final_out = final_out.sum(dim=1).to(new_x.dtype)
                hidden_states = final_out.view(*residual.shape)
        else:
            assert moe_local_idxs is None
            assert topk_weight is None
            hidden_states = moe_outs

        hidden_states = residual + hidden_states
        return hidden_states

    def reference_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reference forward implementation for correctness validation.
        Uses standard flash attention (no ring attention) to stay independent.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError("Position embeddings must be set before calling reference_forward")

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3MoeModel(nn.Module):
    """
    Qwen3 MoE model for DualPipeV pipeline parallelism.

    This model supports stage partitioning for pipeline parallelism and
    expert parallelism for MoE layers.
    """

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
        self.cp_group = cp_group
        self.cp_rank = cp_group.rank() if cp_group is not None else 0
        self.cp_size = cp_group.size() if cp_group is not None else 1

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        intermediate_size = config.intermediate_size
        num_experts = config.num_experts
        num_experts_per_tok = config.num_experts_per_tok
        moe_intermediate_size = config.moe_intermediate_size
        rms_norm_eps = config.rms_norm_eps
        attention_bias = getattr(config, "attention_bias", False)
        norm_topk_prob = getattr(config, "norm_topk_prob", True)
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        rope_theta = getattr(config, "rope_theta", 1000000.0)

        ep_size = getattr(config, "ep_size", 1)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size) if stage_id == 0 else None

        num_local_layers = layer_partition(config.num_hidden_layers, num_stages)
        layer_id_begin = sum(num_local_layers[:stage_id])
        layer_id_end = layer_id_begin + num_local_layers[stage_id]

        self.layers = nn.ModuleDict(
            {
                str(i): Qwen3MoeDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    moe_intermediate_size=moe_intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    attention_bias=attention_bias,
                    norm_topk_prob=norm_topk_prob,
                    layer_idx=i,
                    decoder_sparse_step=decoder_sparse_step,
                    mlp_only_layers=mlp_only_layers,
                    ep_size=ep_size,
                    cp_group=cp_group,
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

        self.rotary_emb = Qwen3MoeRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the model.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor. If stage_id == 0, this should be input_ids.
            Otherwise, it should be hidden states from the previous stage.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        intermediate_tensors: Optional[IntermediateTensors] = getattr(
            self, "_intermediate_tensors", None
        )

        if self.embed_tokens is not None:
            input_ids = hidden_states
            hidden_states = self.embed_tokens(input_ids)

        bsz, seq_len, _ = hidden_states.shape

        # Zigzag CP layout: the local seq_len tokens come from two non-contiguous
        # global chunks. Build the global position IDs by concatenating the
        # front block and the mirror back block, then gather cos/sin by position.
        block = seq_len // 2
        global_seq_len = seq_len * self.cp_size
        front_start = self.cp_rank * block
        back_start = (2 * self.cp_size - self.cp_rank - 1) * block
        position_ids = torch.cat(
            [
                torch.arange(front_start, front_start + block, device=hidden_states.device),
                torch.arange(back_start, back_start + block, device=hidden_states.device),
            ]
        )
        cos, sin = self.rotary_emb(hidden_states, seq_len=global_seq_len)
        position_embeddings = (cos[position_ids].unsqueeze(0), sin[position_ids].unsqueeze(0))

        for layer_idx_str, layer in self.layers.items():
            layer._position_embeddings = position_embeddings

        if intermediate_tensors is None:
            if self.embed_tokens is not None:
                pass
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
    def backward(
        module: "Qwen3MoeModel",
        dy: Optional[List[torch.Tensor]],
        loss: Optional[torch.Tensor],
        intermediate_tensors: IntermediateTensors,
    ):
        """Backward pass for the model."""
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
