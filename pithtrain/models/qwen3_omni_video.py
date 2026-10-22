"""Native video-conditioned Thinker with differentiable staged DeepStack."""

from pithtrain.contexts import training
from pithtrain.models.qwen3_omni_image import Qwen3OmniMoeThinkerImageModel
from pithtrain.models.qwen3_omni_image_inputs import pack_image_features
from pithtrain.models.qwen3_omni_video_inputs import validate_video_inputs
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVideoEncoder


class Qwen3OmniMoeThinkerVideoModel(Qwen3OmniMoeThinkerImageModel):
    """Video+paired-text, CP1/MB1; audio tracks require their separate consumer."""

    input_modalities = ("video",)
    vision_encoder_class = Qwen3OmniMoeVideoEncoder

    @staticmethod
    def validate_media_batch(input_ids, context, config):
        return validate_video_inputs(input_ids, context, config)

    def forward_prolog(self, input_ids, model_context=None):
        context = model_context or {}
        validate_video_inputs(input_ids, context, self.image_config)
        features = self.visual(context["pixel_values_videos"], context["video_grid_thw"])
        return pack_image_features(
            self.embed_tokens(input_ids),
            input_ids,
            self.image_config.video_token_id,
            features,
            self.deepstack_count,
        )

    def forward_posemb(self, length, cu_seqlens=None, model_context=None):
        if cu_seqlens is not None:
            raise NotImplementedError("Packed native video training is not implemented")
        if model_context is None:
            raise ValueError("Native video requires temporal grid metadata")
        positions = validate_video_inputs(
            model_context["input_ids"], model_context, self.image_config, require_pixels=False
        )
        return self.rotary_emb(positions, training.PARAM_DTYPE)
