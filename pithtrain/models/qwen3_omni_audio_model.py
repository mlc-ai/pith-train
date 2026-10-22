"""Native paired audio/text Thinker training on the existing DualPipeV path."""

from pithtrain.contexts import distributed, training
from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from pithtrain.models.qwen3_omni_audio_inputs import (
    audio_feature_lengths,
    flatten_audio_features,
    insert_audio_features,
    validate_audio_inputs,
)
from pithtrain.models.qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel
from pithtrain.pipeline.execution import model_forward


class Qwen3OmniMoeThinkerAudioModel(Qwen3OmniMoeThinkerTextModel):
    # Each DP sample has audio. Mixed audio/text-only batches require a separate
    # unused-encoder collective contract, just as the image implementation does.
    input_modalities = ("audio",)

    def __init__(self, config, phase):
        if distributed.cp_size != 1:
            raise NotImplementedError("Native audio training currently requires CP=1")
        if config.audio_config.output_dim != config.text_config.hidden_size:
            raise ValueError("Audio projection width must match the text decoder")
        super().__init__(config.text_config, phase)
        self.audio_config = config
        self.audio_tower = (
            Qwen3OmniMoeAudioEncoder(config.audio_config) if self.stage_index == 0 else None
        )

    @staticmethod
    def validate_media_batch(input_ids, context, config):
        return validate_audio_inputs(input_ids, context, config)

    def forward_prolog(self, input_ids, model_context=None):
        context = model_context or {}
        validate_audio_inputs(input_ids, context, self.audio_config)
        lengths = audio_feature_lengths(context, self.audio_config)
        features = self.audio_tower(flatten_audio_features(context), lengths).last_hidden_state
        return insert_audio_features(
            self.embed_tokens(input_ids), input_ids, self.audio_config.audio_token_id, features
        )

    def forward_posemb(self, length, cu_seqlens=None, model_context=None):
        if cu_seqlens is not None:
            raise NotImplementedError("Packed native audio training is not implemented")
        if model_context is None:
            raise ValueError("Native audio positions require the shared model_context")
        positions = validate_audio_inputs(
            model_context["input_ids"], model_context, self.audio_config, require_features=False
        )
        if positions.shape[-1] != length:
            raise ValueError("Audio context sequence length differs from model activation")
        return self.rotary_emb(positions, training.PARAM_DTYPE)

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
