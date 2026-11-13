"""Unified native Thinker pretraining for text, image, audio and synchronized video.

All encoder state travels through this microbatch's ordinary activation graph.
CP, packing and padding are deliberately rejected by the initial media contract.
"""

from pithtrain.contexts import training
from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from pithtrain.models.qwen3_omni_image import Qwen3OmniMoeThinkerImageModel
from pithtrain.models.qwen3_omni_multimodal_inputs import (
    encode_multimodal,
    multimodal_position_ids,
)
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVideoEncoder


class Qwen3OmniMoeThinkerMultimodalModel(Qwen3OmniMoeThinkerImageModel):
    input_modalities = ("text", "image", "audio", "video")
    vision_encoder_class = Qwen3OmniMoeVideoEncoder

    def __init__(self, config, phase):
        if training.fp8:
            raise NotImplementedError("Native multimodal training currently requires BF16")
        if config.audio_config.output_dim != config.text_config.hidden_size:
            raise ValueError("Audio projection width must equal text hidden size")
        super().__init__(config, phase)
        self.audio_tower = (
            Qwen3OmniMoeAudioEncoder(config.audio_config) if self.stage_index == 0 else None
        )

    @staticmethod
    def validate_media_batch(input_ids, context, config):
        return multimodal_position_ids(input_ids, context, config)

    def forward_prolog(self, input_ids, media_context=None):
        context = media_context or {}
        multimodal_position_ids(input_ids, context, self.image_config)
        return encode_multimodal(
            self.visual,
            self.audio_tower,
            self.embed_tokens(input_ids),
            input_ids,
            context,
            self.image_config,
        )

    def forward_posemb(self, length, cu_seqlens=None, media_context=None):
        if cu_seqlens is not None:
            raise NotImplementedError("Packed native multimodal training is not implemented")
        if media_context is None:
            return super().forward_posemb(length)
        positions = multimodal_position_ids(
            media_context["input_ids"], media_context, self.image_config, require_media=False
        )
        return self.rotary_emb(positions, training.PARAM_DTYPE)
