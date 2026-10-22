"""Select the implemented text decoder from a composite Hugging Face config."""


def resolve_text_model_config(config):
    """Unwrap Omni/Thinker configs without enabling their media components."""
    if config.model_type == "qwen3_omni_moe":
        config = getattr(config, "thinker_config", None)
        if getattr(config, "model_type", None) != "qwen3_omni_moe_thinker":
            raise ValueError("Omni config requires a Thinker config")
    if config.model_type == "qwen3_omni_moe_thinker":
        config = getattr(config, "text_config", None)
        if getattr(config, "model_type", None) != "qwen3_omni_moe_text":
            raise ValueError("Thinker config requires an Omni text config")
    return config
