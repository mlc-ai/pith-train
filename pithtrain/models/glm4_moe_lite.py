"""
zai-org/GLM-4.7-Flash (Glm4MoeLiteForCausalLM).

The decoder is DeepSeek-V2's: multi-head latent attention with a low-rank query, interleaved rotary
embedding, a dense first layer and shared experts next to the routed ones. Only the router differs:
sigmoid scores, a per-expert correction bias that steers the choice but not the weights
(DeepSeek-V3's aux-loss-free "noaux_tc"), top-k weights renormalized then scaled.

The checkpoint also holds one multi-token-prediction layer, model.layers.47, which this model does
not instantiate; its keys are expected unmapped on import.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig

from pithtrain.models.deepseek_v2 import (
    DeepSeekV2DecoderLayer,
    DeepSeekV2Model,
    DeepSeekV2MoE,
)
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector

__all__ = ["Glm4MoeLiteModel", "Glm4MoeLiteTopkRouter"]


class Glm4MoeLiteTopkRouter(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig):
        super().__init__()
        # config.json names the formula; Glm4MoeLiteConfig keeps the field only when read from it.
        topk_method = getattr(config, "topk_method", "noaux_tc")
        if topk_method != "noaux_tc":
            raise ValueError(f"Unsupported GLM-4.7 topk_method: {topk_method}")
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.load_balance_loss_fn = None
        self.router_replay = None
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)), requires_grad=True)  # fmt: skip
        # A buffer in HF, updated there by its own bias schedule and frozen at inference. A
        # parameter here, so the checkpoint, the HF import and the weight sync carry it; frozen,
        # as Megatron trains this model with a bias update rate of 0, and read in fp32 like HF's
        # (apply_fsdp keeps a frozen parameter in its stored dtype).
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts), requires_grad=False)  # fmt: skip

    def forward(
        self, hidden_states: torch.Tensor, replay_idx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)  # fmt: skip
        scores = logits.sigmoid()
        if replay_idx is not None:
            topk_idx = replay_idx
        else:
            # The bias only steers the choice; the two best experts of each group rank the groups.
            scores_for_choice = scores + self.e_score_correction_bias.type(torch.float32)
            num_tokens = scores.shape[0]
            group_scores = scores_for_choice.view(num_tokens, self.num_group, -1).topk(2, dim=-1)[0].sum(dim=-1)  # fmt: skip
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(num_tokens, self.num_group, self.num_experts // self.num_group).reshape(num_tokens, -1)  # fmt: skip
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
            topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weight = scores.gather(-1, topk_idx)
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        if self.load_balance_loss_fn is None:
            return topk_idx, topk_weight, None
        # The balance loss reads the scores as a distribution over the experts.
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        lb_loss = self.load_balance_loss_fn(probs, topk_idx, self.n_routed_experts, self.top_k)
        # Token-weight the injected lb gradient so train_step's 1/num_tokens grad scale leaves it
        # correctly normalized (it bypasses the token-weighted objective). lb_loss stays unscaled.
        topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss * topk_weight.shape[0])
        return topk_idx, topk_weight, lb_loss


class Glm4MoeLiteMoE(DeepSeekV2MoE):
    gate_class = Glm4MoeLiteTopkRouter


class Glm4MoeLiteDecoderLayer(DeepSeekV2DecoderLayer):
    moe_class = Glm4MoeLiteMoE


class Glm4MoeLiteModel(DeepSeekV2Model):
    decoder_layer_class = Glm4MoeLiteDecoderLayer

    def __init__(self, config: Glm4MoeLiteConfig, phase: int):
        # DeepSeekV2Attention pairs the rotary dims as HF's interleaved variant does.
        if not getattr(config, "rope_interleave", True):
            raise ValueError("GLM-4.7 without rope_interleave is not supported")
        super().__init__(config, phase)
