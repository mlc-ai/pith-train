"""Independent CPU parity for native audio, geometry and input-feature gradients."""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HFAudio,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFPositions,
)

from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder, audio_output_lengths
from pithtrain.models.qwen3_omni_audio_inputs import (
    flatten_audio_features,
    insert_audio_features,
    validate_audio_inputs,
)
from pithtrain.modules.model_config import resolve_text_model_config

ROOT = Path(__file__).resolve().parents[1]


def audio_config(**kwargs):
    values = dict(
        num_mel_bins=16,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=24,
        d_model=16,
        output_dim=24,
        downsample_hidden_size=4,
        max_source_positions=13,
        n_window=50,
        n_window_infer=200,
        conv_chunksize=2,
    )
    values.update(kwargs)
    return Qwen3OmniMoeAudioEncoderConfig(**values)


def thinker_config():
    return Qwen3OmniMoeThinkerConfig(
        audio_config=audio_config(),
        audio_token_id=10,
        audio_start_token_id=11,
        audio_end_token_id=12,
        image_token_id=13,
        video_token_id=14,
        vision_start_token_id=15,
        text_config=dict(vocab_size=32, hidden_size=24),
    )


def paired(config, device="cpu", dtype=torch.float32):
    torch.manual_seed(91)
    config = copy.deepcopy(config)
    config._attn_implementation = "eager"
    reference = HFAudio(config).to(device=device, dtype=dtype)
    native = Qwen3OmniMoeAudioEncoder(config).to(device=device, dtype=dtype)
    native.load_state_dict(reference.state_dict(), strict=True)
    assert dict(native.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    return native, reference


def assert_gradients(native, reference, rtol=2e-4, atol=2e-6):
    for name, parameter in native.named_parameters():
        expected = reference.get_parameter(name)
        assert parameter.grad is not None and expected.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and torch.isfinite(expected.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=rtol, atol=atol, msg=name)


@pytest.mark.parametrize("lengths", [[1], [99], [100], [101], [201, 17], [300, 100]])
@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_encoder_output_every_gradient_and_optimizer_update(lengths, activation):
    native, reference = paired(audio_config(activation_function=activation))
    features = torch.randn(16, sum(lengths), requires_grad=True)
    expected_features = features.detach().clone().requires_grad_()
    sizes = torch.tensor(lengths)
    actual = native(features, sizes).last_hidden_state
    expected = reference(expected_features, feature_lens=sizes).last_hidden_state
    assert actual.shape == (
        sum((length // 100 * 13 + (length % 100 + 7) // 8) for length in lengths),
        24,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert_gradients(native, reference)
    torch.testing.assert_close(features.grad, expected_features.grad, rtol=2e-4, atol=2e-6)
    before = native.conv2d1.weight.detach().clone()
    for model in (native, reference):
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0).step()
    assert not torch.equal(before, native.conv2d1.weight)
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter, reference.get_parameter(name), rtol=2e-4, atol=2e-6, msg=name
        )


def audio_batch():
    config = thinker_config()
    lengths = torch.tensor([101, 17])
    mask = (torch.arange(101)[None] < lengths[:, None]).long()
    tokens = torch.tensor([[7, 11, *([10] * 14), 12, 8, 11, 10, 10, 10, 12, 9]])
    return (
        config,
        tokens,
        dict(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            input_features=torch.randn(2, 16, 101),
            feature_attention_mask=mask,
        ),
    )


def test_audio_positions_match_hf_and_processor_flattening():
    config, tokens, context = audio_batch()
    positions = validate_audio_inputs(tokens, context, config)
    owner = SimpleNamespace(
        config=config, spatial_merge_size=config.vision_config.spatial_merge_size
    )
    expected, _ = HFPositions.get_rope_index(owner, tokens, attention_mask=torch.ones_like(tokens))
    torch.testing.assert_close(positions.float(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        flatten_audio_features(context),
        torch.cat(
            (context["input_features"][0, :, :101], context["input_features"][1, :, :17]), dim=1
        ),
        rtol=0,
        atol=0,
    )


def test_insertion_propagates_to_audio_parameters_and_only_valid_frames():
    config, tokens, context = audio_batch()
    native, reference = paired(config.audio_config)
    context["input_features"].requires_grad_()
    expected_input = context["input_features"].detach().clone().requires_grad_()
    lengths = context["feature_attention_mask"].sum(-1)
    features = native(flatten_audio_features(context), lengths).last_hidden_state
    expected_features = reference(
        expected_input.permute(0, 2, 1)[context["feature_attention_mask"].bool()].T,
        feature_lens=lengths,
    ).last_hidden_state
    embeddings = torch.randn(1, tokens.shape[1], 24, requires_grad=True)
    expected_embeddings = embeddings.detach().clone().requires_grad_()
    actual = insert_audio_features(embeddings, tokens, 10, features)
    expected = expected_embeddings.clone()
    expected[tokens == 10] = expected_features
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert_gradients(native, reference)
    torch.testing.assert_close(
        context["input_features"].grad, expected_input.grad, rtol=2e-4, atol=2e-6
    )
    torch.testing.assert_close(embeddings.grad, expected_embeddings.grad, rtol=0, atol=0)
    assert context["input_features"].grad[1, :, 17:].count_nonzero() == 0


def test_two_live_audio_microbatches_keep_independent_graphs():
    native, _ = paired(audio_config())
    independent = copy.deepcopy(native)
    first, second = torch.randn(16, 101), torch.randn(16, 17)
    a = native(first, torch.tensor([101])).last_hidden_state
    b = native(second, torch.tensor([17])).last_hidden_state
    b.square().sum().backward()
    a.square().sum().backward()
    expected = (
        independent(first, torch.tensor([101])).last_hidden_state.square().sum()
        + independent(second, torch.tensor([17])).last_hidden_state.square().sum()
    )
    expected.backward()
    assert_gradients(native, independent)


def training_function(name, namespace):
    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "training", "exec"), namespace)
    return namespace[name]


def test_audio_registration_preserves_existing_model_selection():
    audio, image, text = object(), object(), object()
    choose = training_function(
        "model_class_for_config",
        dict(
            Qwen3OmniMoeThinkerAudioModel=audio,
            Qwen3OmniMoeThinkerImageModel=image,
            Qwen3OmniMoeThinkerTextModel=text,
            resolve_text_model_config=resolve_text_model_config,
        ),
    )
    config = thinker_config()
    for source in (config, SimpleNamespace(model_type="qwen3_omni_moe", thinker_config=config)):
        assert choose(source) is text
        assert choose(source, native_vision=True) is image
        assert choose(source, native_audio=True) is audio
        with pytest.raises(ValueError, match="combined media"):
            choose(source, native_audio=True, native_vision=True)
    with pytest.raises(ValueError, match="composite"):
        choose(config.text_config, native_audio=True)


def test_audio_modules_are_adamw_parameters_and_initialize_after_meta_allocation():
    choose = training_function("is_muon_param", {"torch": torch})
    assert not choose("module.0.audio_tower.conv2d1.weight", torch.empty(4, 1, 3, 3))
    assert not choose("module.0.audio_tower.layers.0.self_attn.q_proj.weight", torch.empty(16, 16))
    assert choose("module.0.layers.0.self_attn.q_proj.weight", torch.empty(16, 16))
    initialize = training_function("init_weights", {"torch": torch, "nn": torch.nn, "math": math})
    with torch.device("meta"):
        model = torch.nn.Module()
        model.audio_tower = Qwen3OmniMoeAudioEncoder(audio_config())
    model.to_empty(device="cpu")
    initialize(model, 4)
    for name, value in model.named_parameters():
        assert torch.isfinite(value).all(), name
        if value.ndim == 1:
            torch.testing.assert_close(
                value, torch.zeros_like(value) if name.endswith("bias") else torch.ones_like(value)
            )
        else:
            assert value.std() > 0, name
    assert torch.isfinite(
        model.audio_tower(torch.randn(16, 101), torch.tensor([101])).last_hidden_state
    ).all()


def test_later_pipeline_chunks_validate_without_feature_payload():
    config, tokens, context = audio_batch()
    expected = validate_audio_inputs(tokens, context, config)
    del context["input_features"]
    torch.testing.assert_close(
        validate_audio_inputs(tokens, context, config, require_features=False), expected
    )
    with pytest.raises(ValueError, match="input_features"):
        validate_audio_inputs(tokens, context, config)


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "zero",
        "negative",
        "float_lengths",
        "wrong_mels",
        "wrong_total",
        "nan",
        "integer_features",
    ],
)
def test_bad_encoder_inputs_rejected(change):
    native, _ = paired(audio_config())
    sizes, features = torch.tensor([17]), torch.randn(16, 17)
    if change == "empty":
        sizes = torch.tensor([], dtype=torch.long)
    elif change == "zero":
        sizes[0] = 0
    elif change == "negative":
        sizes[0] = -1
    elif change == "float_lengths":
        sizes = sizes.float()
    elif change == "wrong_mels":
        features = features[:15]
    elif change == "wrong_total":
        sizes[0] = 16
    elif change == "nan":
        features[0, 0] = float("nan")
    elif change == "integer_features":
        features = features.long()
    with pytest.raises(ValueError):
        native(features, sizes)


@pytest.mark.parametrize(
    "change",
    [
        "count",
        "start",
        "end",
        "unused",
        "missing",
        "mask_hole",
        "mask_empty",
        "mask_value",
        "nan",
        "shape",
        "lengths",
        "padding",
        "video",
        "extra",
        "positions",
        "shared_ids",
    ],
)
def test_bad_audio_microbatch_rejected(change):
    config, tokens, context = audio_batch()
    if change == "count":
        tokens[0, 2] = 9
    elif change == "start":
        tokens[0, 1] = 9
    elif change == "end":
        tokens[0, 16] = 9
    elif change == "unused":
        context["feature_attention_mask"] = context["feature_attention_mask"][:1]
        context["input_features"] = context["input_features"][:1]
    elif change == "missing":
        del context["input_features"]
    elif change == "mask_hole":
        context["feature_attention_mask"][0, 1] = 0
    elif change == "mask_empty":
        context["feature_attention_mask"][0] = 0
    elif change == "mask_value":
        context["feature_attention_mask"][0, 0] = 2
    elif change == "nan":
        context["input_features"][0, 0, 0] = float("nan")
    elif change == "shape":
        context["input_features"] = context["input_features"][:, :15]
    elif change == "lengths":
        context["audio_feature_lengths"] = torch.tensor([99, 17])
    elif change == "padding":
        context["attention_mask"][0, -1] = 0
    elif change == "video":
        tokens[0, -1] = config.video_token_id
    elif change == "extra":
        context["pixel_values"] = torch.ones(1)
    elif change == "positions":
        context["position_ids"] = torch.zeros(3, 1, tokens.shape[1])
    elif change == "shared_ids":
        context["input_ids"] = tokens.flip(-1)
    with pytest.raises(ValueError):
        validate_audio_inputs(tokens, context, config)


@pytest.mark.parametrize(
    "change",
    [
        dict(n_window=40),
        dict(n_window_infer=50),
        dict(n_window_infer=150),
        dict(d_model=15),
        dict(d_model=2),
        dict(max_source_positions=12),
        dict(conv_chunksize=0),
        dict(encoder_layers=0),
        dict(dropout=0.1),
        dict(activation_function="unknown"),
    ],
)
def test_unsupported_audio_config_rejected(change):
    with pytest.raises(ValueError):
        Qwen3OmniMoeAudioEncoder(audio_config(**change))


def test_audio_has_no_transformers_execution_imports_or_cached_feature_state():
    for name in ("qwen3_omni_audio.py", "qwen3_omni_audio_inputs.py", "qwen3_omni_audio_model.py"):
        tree = ast.parse((ROOT / "pithtrain/models" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("transformers")
            if isinstance(node, ast.Assign):
                assert all(
                    not (
                        isinstance(target, ast.Attribute)
                        and target.attr in {"input_features", "model_context", "audio_features"}
                    )
                    for target in node.targets
                )


def test_released_output_lengths_include_chunk_boundaries():
    sizes = torch.tensor([1, 7, 8, 9, 99, 100, 101, 199, 200, 201])
    torch.testing.assert_close(
        audio_output_lengths(sizes), torch.tensor([1, 1, 1, 2, 13, 13, 14, 26, 26, 27])
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_matches_hf_arithmetic_including_near_zero_key_bias(dtype):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioAttention as HFAttention,
    )

    from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioAttention

    config = audio_config()
    config._attn_implementation = "eager"
    reference = HFAttention(config).to(dtype=dtype)
    native = Qwen3OmniMoeAudioAttention(config).to(dtype=dtype)
    native.load_state_dict(reference.state_dict(), strict=True)
    hidden = torch.randn(43, config.d_model, dtype=dtype, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    actual = native(hidden, (26, 13, 4))
    expected = reference(expected_hidden, torch.tensor([0, 26, 39, 43], dtype=torch.int32))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert_gradients(native, reference, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad, rtol=0, atol=0)
