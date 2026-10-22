"""Native Thinker image training, using the existing staged text decoder.

The expanded activation transports differentiable DeepStack lanes through the
existing scheduler. All feature state belongs to a microbatch's activation graph.
CP, packed/padded samples, audio/video and FP8 are explicitly unsupported here.
"""

from functools import partial

import torch

from pithtrain.contexts import distributed, training
from pithtrain.models.qwen3_omni_image_inputs import (
    inject_deepstack,
    pack_image_features,
    validate_image_inputs,
)
from pithtrain.models.qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextModel,
)
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVisionEncoder
from pithtrain.pipeline.execution import model_forward


class Qwen3OmniMoeThinkerImageDecoderLayer(Qwen3OmniMoeThinkerTextDecoderLayer):
    def __init__(self, config, layer_id, deepstack_count):
        super().__init__(config, layer_id)
        self.text_hidden_size = config.hidden_size
        self.deepstack_count = deepstack_count

    def forward_stage1(self, payload, rotary_posemb, cu_seqlens=None):
        hidden, lanes = payload[..., : self.text_hidden_size], payload[..., self.text_hidden_size :]
        dispatch, residual, routing = super().forward_stage1(hidden, rotary_posemb, cu_seqlens)
        return dispatch, torch.cat((residual, lanes), -1), routing

    # Preserve explicit BF16 rounding at residual, routing and DeepStack boundaries.
    @torch.compile(fullgraph=True, options={"emulate_precision_casts": True})
    def forward_stage5(self, moe_outs, moe_local_idxs, topk_weight, residual):
        hidden, lanes = (
            residual[..., : self.text_hidden_size],
            residual[..., self.text_hidden_size :],
        )
        # Inline the shared arithmetic inside this single fullgraph region.
        hidden = Qwen3OmniMoeThinkerTextDecoderLayer._forward_stage5(
            self, moe_outs, moe_local_idxs, topk_weight, hidden
        )
        return inject_deepstack(
            hidden, lanes, self.idx, self.text_hidden_size, self.deepstack_count
        )

    def reference_forward_with_hidden(self, payload, rotary_posemb, cu_seqlens=None):
        """Return both the transported payload and HF's pre-DeepStack hidden state."""
        hidden, lanes = payload[..., : self.text_hidden_size], payload[..., self.text_hidden_size :]
        hidden = super().reference_forward(hidden, rotary_posemb, cu_seqlens)
        injected = inject_deepstack(
            hidden, lanes, self.idx, self.text_hidden_size, self.deepstack_count
        )
        return injected, hidden

    def reference_forward(self, payload, rotary_posemb, cu_seqlens=None):
        return self.reference_forward_with_hidden(payload, rotary_posemb, cu_seqlens)[0]


class Qwen3OmniMoeThinkerImageModel(Qwen3OmniMoeThinkerTextModel):
    # Paired image+text records only. Mixed image/text DP batches need an
    # explicit unused-encoder collective contract before enabling that mixture.
    input_modalities = ("image",)
    vision_encoder_class = Qwen3OmniMoeVisionEncoder

    def __init__(self, config, phase):
        if distributed.cp_size != 1:
            raise NotImplementedError("Native image training currently requires CP=1")
        text = config.text_config
        count = len(config.vision_config.deepstack_visual_indexes)
        if (
            config.vision_config.out_hidden_size != text.hidden_size
            or count > text.num_hidden_layers
        ):
            raise ValueError("Vision projection width/DeepStack count must fit the text decoder")
        super().__init__(
            text,
            phase,
            decoder_layer_factory=partial(
                Qwen3OmniMoeThinkerImageDecoderLayer, deepstack_count=count
            ),
        )
        self.image_config = config
        self.text_hidden_size, self.deepstack_count = text.hidden_size, count
        self.hidden_size = text.hidden_size * (1 + count)
        self.visual = (
            self.vision_encoder_class(config.vision_config) if self.stage_index == 0 else None
        )

    @staticmethod
    def validate_media_batch(input_ids, context, config):
        return validate_image_inputs(input_ids, context, config)

    def forward_prolog(self, input_ids, model_context=None):
        context = model_context or {}
        validate_image_inputs(input_ids, context, self.image_config)
        embeddings = self.embed_tokens(input_ids)
        pixels = context.get("pixel_values")
        visual = None if pixels is None else self.visual(pixels, context["image_grid_thw"])
        return pack_image_features(
            embeddings, input_ids, self.image_config.image_token_id, visual, self.deepstack_count
        )

    def forward_posemb(self, length, cu_seqlens=None, model_context=None):
        if cu_seqlens is not None:
            raise NotImplementedError("Packed native image training is not implemented")
        if model_context is None:
            return super().forward_posemb(length)
        positions = validate_image_inputs(
            model_context["input_ids"], model_context, self.image_config, require_pixels=False
        )
        return self.rotary_emb(positions, training.PARAM_DTYPE)

    def forward_epilog(self, payload):
        return super().forward_epilog(payload[..., : self.text_hidden_size])

    def forward(self, hidden_states, cu_seqlens=None, model_context=None):
        return model_forward(self, hidden_states, self.chunk_record, cu_seqlens, model_context)

    def reference_forward(self, hidden_states, cu_seqlens=None, model_context=None):
        if self.stage_index == 0:
            hidden_states = self.forward_prolog(hidden_states, model_context)
        rotary = self.forward_posemb(hidden_states.shape[1], cu_seqlens, model_context)
        for layer in self.layers.values():
            hidden_states = layer.reference_forward(hidden_states, rotary, cu_seqlens)
        if self.stage_index == self.stage_count - 1:
            hidden_states = self.forward_epilog(hidden_states)
        return hidden_states
