"""Talker trunk/first-code head on the existing five-stage DualPipeV protocol.

This adapter deliberately owns only model.* and codec_head.*. The complete
speech task composes the two conditioning projections and residual predictor;
loading this partial adapter is not a complete Talker checkpoint conversion.
"""

import torch
from torch import nn

from pithtrain.contexts import distributed, training
from pithtrain.models.qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextAttention,
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeThinkerTextRMSNorm,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
    _combine_expert_order,
)
from pithtrain.operators.linear import BF16Linear
from pithtrain.pipeline.dualpipev import layer_partition


class Qwen3OmniMoeTalkerPipelineMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.shared_expert_intermediate_size
        self.gate_proj = training.Linear(config.hidden_size, width, bias=False)
        self.up_proj = training.Linear(config.hidden_size, width, bias=False)
        self.down_proj = training.Linear(width, config.hidden_size, bias=False)

    def forward(self, value):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))


class Qwen3OmniMoeTalkerPipelineMoeBlock(Qwen3OmniMoeThinkerTextSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.shared_expert = Qwen3OmniMoeTalkerPipelineMLP(config)
        self.shared_expert_gate = training.Linear(config.hidden_size, 1, bias=False)

    def shared_forward(self, hidden):
        return self.shared_expert_gate(hidden).sigmoid() * self.shared_expert(hidden)

    def reference_forward(self, hidden):
        # Retain the HF addition order: routed + shared, then the layer residual.
        return super().reference_forward(hidden) + self.shared_forward(hidden)


class Qwen3OmniMoeTalkerPipelineDecoderLayer(Qwen3OmniMoeThinkerTextDecoderLayer):
    def __init__(self, config, layer_id):
        nn.Module.__init__(self)
        self.idx = layer_id
        self.is_moe = True
        self.self_attn = Qwen3OmniMoeThinkerTextAttention(config, layer_id)
        self.mlp = Qwen3OmniMoeTalkerPipelineMoeBlock(config)
        self.input_layernorm = Qwen3OmniMoeThinkerTextRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3OmniMoeThinkerTextRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    @torch.compile(fullgraph=True, options={"emulate_precision_casts": True})
    def forward_stage1_compute(
        self, hidden_states, rotary_posemb, cu_seqlens=None, replay_idx=None
    ):
        residual = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), rotary_posemb, cu_seqlens
        )
        hidden_states = self.post_attention_layernorm(residual)
        shared = self.mlp.shared_forward(hidden_states)
        routing = self.mlp.gate(hidden_states, replay_idx)
        # Carry both tensors through the existing residual channel. Folding
        # shared into residual here would change HF's BF16 addition order.
        return hidden_states, torch.cat((residual, shared), -1), *routing

    @torch.compile(fullgraph=True, options={"emulate_precision_casts": True})
    def forward_stage5(self, moe_outs, moe_local_idxs, topk_weight, residual):
        residual, shared = residual.chunk(2, dim=-1)
        routed = _combine_expert_order(
            moe_outs,
            topk_weight,
            moe_local_idxs if distributed.ep_size > 1 else None,
            residual.dtype,
        ).view_as(residual)
        return residual + (routed + shared)


class Qwen3OmniMoeTalkerTextModel(Qwen3OmniMoeThinkerTextModel):
    """First-code-only pipeline adapter with HF model.* parameter names."""

    input_modalities = ("codec",)
    checkpoint_component_scope = "talker_trunk_and_first_code_head"

    def __init__(self, config, phase):
        nn.Module.__init__(self)
        if training.fp8:
            raise NotImplementedError("Talker fused experts currently require BF16")
        if config.tie_word_embeddings or config.attention_dropout or config.hidden_act != "silu":
            raise NotImplementedError(
                "Talker pipeline requires untied embeddings, zero dropout and SiLU"
            )
        if config.mlp_only_layers or config.decoder_sparse_step != 1:
            raise NotImplementedError(
                "Released Talker uses MoE plus a shared expert in every layer"
            )
        if config.sliding_window is not None:
            raise NotImplementedError("Talker trunk uses full attention")
        if config.shared_expert_intermediate_size <= 0:
            raise ValueError("Talker requires a positive shared expert width")
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
        self.chunk_record = None
        self.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(config)
        self.model = nn.Module()
        self.model.codec_embedding = None
        self.model.norm = None
        self.codec_head = None
        if self.stage_index == 0:
            self.model.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        if self.stage_index == self.stage_count - 1:
            self.model.norm = Qwen3OmniMoeThinkerTextRMSNorm(
                config.hidden_size, config.rms_norm_eps
            )
            self.codec_head = BF16Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.layers = nn.ModuleDict(
            {
                str(i): Qwen3OmniMoeTalkerPipelineDecoderLayer(config, i)
                for i in layer_partition(
                    config.num_hidden_layers, self.stage_count, self.stage_index
                )
            }
        )

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.codec_embedding

    @property
    def norm(self):
        return self.model.norm

    @property
    def lm_head(self):
        return self.codec_head

    def fsdp_edge_modules(self):
        # Do not shard the enclosing model before its expert/layer children.
        return tuple(
            module
            for module in (self.model.codec_embedding, self.model.norm, self.codec_head)
            if module is not None
        )
