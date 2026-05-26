"""deepseek-ai/DeepSeek-V2-Lite."""

import math
from dataclasses import fields
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from pithtrain.dualpipe.execution import EpilogArgs, IntermediateTensors, PrologArgs, PrologOuts
from pithtrain.dualpipe.layer_partition import layer_partition
from pithtrain.dualpipe.modeling import decoder_layer_backward, decoder_layer_forward
from pithtrain.dualpipe.utils import run_backward
from pithtrain.layers.factory import ModelImplMode, get_group_linear_cls, get_linear_cls
from pithtrain.models.interface import ForwardAttnOutput
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.ep_dispatch import moe_ep_prepare_dispatch
from pithtrain.operators.flash_attn_v4 import mla_flash_attn_func
from pithtrain.operators.ring_attention import mla_ring_attention_func
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

torch._dynamo.allow_in_graph(MoELoadBalanceLossInjector)


class DeepseekV2LiteRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2LiteYarnRotaryEmbedding(DeepseekV2LiteRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekV2LiteMLP(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        LinearCls = get_linear_cls()
        self.gate_proj = LinearCls(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = LinearCls(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = LinearCls(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(silu_mul(g, u))


class DeepseekV2LiteExperts(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        num_experts: int,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        GroupLinearCls = get_group_linear_cls()
        self.gate_proj = GroupLinearCls(num_experts, self.hidden_size, self.intermediate_size)
        self.up_proj = GroupLinearCls(num_experts, self.hidden_size, self.intermediate_size)
        self.down_proj = GroupLinearCls(num_experts, self.intermediate_size, self.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ):
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0])
        kwargs = dict(grouped_mm_offs=grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi)
        g = self.gate_proj(x, **kwargs)
        u = self.up_proj(x, **kwargs)
        return self.down_proj(silu_mul(g, u), **kwargs)


class DeepseekV2LiteMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.load_balance_loss_fn = None
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size)), requires_grad=True
        )

    @torch.compile(fullgraph=True)
    def compute(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Gate math + lb_loss injection (compiled).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            topk_idx, topk_weight, lb_loss (None when not training or no loss fn).
        """
        _, _, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight * self.routed_scaling_factor

        if self.training and self.load_balance_loss_fn is not None:
            lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.n_routed_experts, self.top_k)
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states):
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)

        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)

        return topk_idx, topk_weight


class DeepseekV2LiteMoEWithGroupGeMM(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        ep_group: Optional[dist.ProcessGroup] = None,
        layer_id: int = 0,
    ):
        super().__init__()
        self.config = config
        self.ep_group = ep_group
        self.num_experts_per_tok = config.num_experts_per_tok
        self.ep_size = getattr(config, "ep_size", 1)
        self.ep_rank = ep_group.rank() if ep_group is not None else 0
        self.experts_per_rank = config.n_routed_experts // self.ep_size
        self.n_routed_experts = config.n_routed_experts

        self.experts = DeepseekV2LiteExperts(
            config,
            self.experts_per_rank,
            intermediate_size=config.moe_intermediate_size,
        )
        self.gate = DeepseekV2LiteMoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2LiteMLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x, topk_ids, topk_weight):
        assert self.ep_size == 1, "reference implementation only supports ep_size=1"
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


class DeepseekV2LiteAttention(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        layer_id: int = 0,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.cp_group = cp_group
        self.use_ring_attn = cp_group is not None and cp_group.size() > 1

        LinearCls = get_linear_cls()
        self.q_proj = LinearCls(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = LinearCls(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = LinearCls(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = LinearCls(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.softmax_scale = self.q_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        normed_kv = self.kv_a_layernorm(compressed_kv)
        cos, sin = position_embeddings
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, unsqueeze_dim=2)

        if self.use_ring_attn:
            # MLA context parallelism: rotate the compressed latent (normed_kv + shared
            # k_pe) around the ring and decompress on each rank via kv_b, instead of
            # decompressing to full per-head K/V before rotating. ~9x less ring traffic
            # for DeepSeek-V2-Lite.
            attn_output = mla_ring_attention_func(
                q_nope,
                q_pe,
                normed_kv.contiguous(),
                k_pe.contiguous(),
                self.kv_b_proj.weight,
                sm_scale=self.softmax_scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                v_head_dim=self.v_head_dim,
                cp_group=self.cp_group,
            )
        else:
            kv = self.kv_b_proj(normed_kv).view(
                bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            attn_output = mla_flash_attn_func(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                value_states,
                softmax_scale=self.softmax_scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                causal=True,
            )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class DeepseekV2LiteDecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        layer_id: int,
        ep_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.idx = layer_id
        self.self_attn = DeepseekV2LiteAttention(
            config=config, layer_id=layer_id, cp_group=cp_group
        )

        self.mlp = (
            DeepseekV2LiteMoEWithGroupGeMM(config, ep_group, layer_id)
            if (
                config.n_routed_experts is not None
                and layer_id >= config.first_k_dense_replace
                and layer_id % config.moe_layer_freq == 0
            )
            else DeepseekV2LiteMLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.self_attn.use_ring_attn:
            self._forward_attn_compute = self._forward_attn_compute.__wrapped__.__get__(
                self, type(self)
            )

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
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if hasattr(self.mlp, "shared_experts"):
            residual = residual + self.mlp.shared_experts(hidden_states)

        return hidden_states, residual

    def forward_attn(
        self,
        hidden_states: torch.Tensor,
    ):
        """LN + Attn + LN + Expert selection"""
        hidden_states, residual = self._forward_attn_compute(hidden_states)

        assert isinstance(self.mlp, (DeepseekV2LiteMLP, DeepseekV2LiteMoEWithGroupGeMM))
        if isinstance(self.mlp, DeepseekV2LiteMLP):
            return ForwardAttnOutput(
                hidden_states,  # sorted_tokens
                None,  # idxs
                None,  # topk_weight
                None,  # output_splits
                None,  # input_splits
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
            self.mlp.n_routed_experts,
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
    ):
        """MLP forward"""
        if isinstance(self.mlp, DeepseekV2LiteMLP):
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
    ):
        """
        Weighted expert output + residual connection.
        Shared expert output is already folded into residual by forward_attn.
        """

        def moe_finalize(moe_outs, moe_local_idxs, topk_weight) -> torch.Tensor:
            if self.mlp.ep_size > 1:
                assert moe_local_idxs is not None
                seq_len, topk = topk_weight.shape
                # Memory-efficient equivalent of
                # new_x[moe_local_idxs] = moe_outs followed by weighted sum.
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

        if isinstance(self.mlp, DeepseekV2LiteMoEWithGroupGeMM):
            hidden_states = moe_finalize(moe_outs, moe_local_idxs, topk_weight).view(
                *residual.shape
            )
        else:
            assert moe_local_idxs is None
            assert topk_weight is None
            hidden_states = moe_outs

        hidden_states = residual + hidden_states
        return hidden_states

    def reference_forward(
        self,
        hidden_states: torch.Tensor,
    ):
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DeepseekV2LiteModel(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        num_stages: int,
        stage_id: int,
        ep_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.cp_group = cp_group
        self.cp_rank = cp_group.rank() if cp_group is not None else 0
        self.cp_size = cp_group.size() if cp_group is not None else 1
        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size) if stage_id == 0 else None
        )
        # Compute the local layer range for this stage
        # We first equally distribute the layers to each stage.
        # For the remaining layers,
        # - when i is even, the i-th remaining layer goes to the (i // 2)-th layer
        # counting from the beginning.
        # - when i is odd, the i-th remaining layer goes to the (num_stages - 1 - i // 2)-th layer
        # counting from the beginning.
        #
        # The main reason of this partition is because stage 0 may have layers that
        # do not use MoE, which already computes less than other stages.
        # Naive layer partition may cause stage -1 to have fewer layers than other stages,
        # which further leads to imbalance. So we use this partition, trying to achieve
        # a more balanced partition.
        num_local_layers = layer_partition(config.num_hidden_layers, num_stages)
        layer_id_begin = sum(num_local_layers[:stage_id])
        layer_id_end = layer_id_begin + num_local_layers[stage_id]
        self.layers = nn.ModuleDict(
            {
                str(i): DeepseekV2LiteDecoderLayer(config, i, ep_group, cp_group=cp_group)
                for i in range(layer_id_begin, layer_id_end)
            }
        )
        if stage_id == num_stages - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

        scaling_factor = config.rope_scaling["factor"]
        rope_kwargs = {
            key: config.rope_scaling[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in config.rope_scaling
        }
        rope_theta = getattr(config, "rope_theta", None) or (config.rope_scaling or {}).get(
            "rope_theta"
        )
        self.rotary_emb = DeepseekV2LiteYarnRotaryEmbedding(
            config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            scaling_factor=scaling_factor,
            base=rope_theta,
            **rope_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        # Get pre-allocated intermediate_tensors from module attribute (set by DualPipeV)
        intermediate_tensors: Optional[IntermediateTensors] = getattr(
            self, "_intermediate_tensors", None
        )

        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(hidden_states)

        seq_len = hidden_states.shape[1]
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
        position_embeddings = (
            cos[position_ids].unsqueeze(0).to(dtype=hidden_states.dtype),
            sin[position_ids].unsqueeze(0).to(dtype=hidden_states.dtype),
        )
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
                # Copy into pre-allocated slot
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(layer_record):
                    src_rec = getattr(layer_record, field.name)
                    dst_rec = getattr(dst, field.name)
                    for rf in fields(src_rec):
                        if hasattr(src_rec, rf.name):
                            setattr(dst_rec, rf.name, getattr(src_rec, rf.name))
            else:
                hidden_states = ret[0]
                # Clear pre-allocated slot (layer didn't produce intermediate)
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
        module: "DeepseekV2LiteModel",
        dy: Optional[List[torch.Tensor]],
        loss: Optional[torch.Tensor],
        intermediate_tensors: IntermediateTensors,
    ):
        assert (dy is None) != (loss is None), "Either dy or loss should be provided"
        if loss is not None:
            assert module.norm is not None
            assert module.lm_head is not None
            loss.backward()
            loss.detach_()
            dy = (intermediate_tensors.epilog.args.hidden_states.grad,)
            # Clear tensor refs but keep pre-allocated record
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
            # Clear tensor refs but keep pre-allocated record
            record.args = None
            record.outs = None
            final_grads = (None,)
        return final_grads
