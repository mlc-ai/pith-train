"""Independent CPU HF gates for temporal vision, positions, and trainable DeepStack."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFPositions,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as HFVision,
)

from pithtrain.models.qwen3_omni_video_inputs import validate_video_inputs, video_position_ids
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVideoEncoder
from pithtrain.modules.model_config import resolve_text_model_config
from tests.test_qwen3_omni_vision import assert_gradients, thinker_config, vision_config


def hf_video_positions(tokens, grids, intervals, config, mask):
    owner = SimpleNamespace(
        config=config, spatial_merge_size=config.vision_config.spatial_merge_size
    )
    owner.get_llm_pos_ids_for_vision = HFPositions.get_llm_pos_ids_for_vision.__get__(owner)
    return HFPositions.get_rope_index(
        owner, tokens, video_grid_thw=grids, second_per_grids=intervals, attention_mask=mask
    )[0]


@pytest.mark.parametrize(
    "grids", [[[2, 4, 6]], [[3, 2, 2], [2, 6, 4]], [[1, 8, 10]], [[1, 4, 4], [2, 4, 6]]]
)
@pytest.mark.parametrize("activation", ["gelu_pytorch_tanh", "gelu", "silu"])
def test_video_outputs_all_parameter_gradients_and_update(grids, activation):
    torch.manual_seed(91)
    config = vision_config(hidden_act=activation)
    config._attn_implementation = "eager"
    reference = HFVision(config)
    native = Qwen3OmniMoeVideoEncoder(config)
    native.load_state_dict(reference.state_dict(), strict=True)
    assert dict(native.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    grid = torch.tensor(grids)
    pixels = torch.randn(sum(t * h * w for t, h, w in grids), 24, requires_grad=True)
    expected_pixels = pixels.detach().clone().requires_grad_()
    actual, expected = native(pixels, grid), reference(expected_pixels, grid)
    actual_outputs = (actual.last_hidden_state, actual.pooler_output, *actual.deepstack_features)
    expected_outputs = (
        expected.last_hidden_state,
        expected.pooler_output,
        *expected.deepstack_features,
    )
    probes = [torch.randn_like(value) for value in actual_outputs]
    for a, b in zip(actual_outputs, expected_outputs, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    sum((v * p).sum() for v, p in zip(actual_outputs, probes)).backward()
    sum((v * p).sum() for v, p in zip(expected_outputs, probes)).backward()
    assert_gradients(native, reference)
    torch.testing.assert_close(pixels.grad, expected_pixels.grad, rtol=2e-4, atol=2e-6)
    before = {name: p.detach().clone() for name, p in native.named_parameters()}
    opt = torch.optim.AdamW(native.parameters(), lr=1e-3, weight_decay=0)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0)
    opt.step()
    ref_opt.step()
    for name, p in native.named_parameters():
        assert not torch.equal(before[name], p), f"No update: {name}"
        torch.testing.assert_close(p, reference.get_parameter(name), rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("rate", [25, 3])
@pytest.mark.parametrize("intervals", [[0.13, 0.37], [0.001, 0.04], [2.0, 1.0]])
@pytest.mark.parametrize("padding", [False, True])
def test_video_temporal_positions_match_hf(rate, intervals, padding):
    config = thinker_config()
    config.position_id_per_seconds = rate
    grids = torch.tensor([[3, 2, 4], [2, 4, 4]])
    tokens = torch.tensor([[7, 11, *([14] * 6), 12, 8, 11, *([14] * 8), 12, 9, 0]])
    mask = torch.ones_like(tokens)
    mask[:, -1] = int(not padding)
    intervals = torch.tensor(intervals)
    positions = video_position_ids(tokens, grids, intervals, config, mask)
    torch.testing.assert_close(
        positions, hf_video_positions(tokens, grids, intervals, config, mask), rtol=0, atol=0
    )
    assert positions.dtype == torch.float32


def test_visual_attention_is_independent_per_frame():
    torch.manual_seed(3)
    model = Qwen3OmniMoeVideoEncoder(vision_config())
    pixels = torch.randn(48, 24, requires_grad=True)
    grid = torch.tensor([[3, 4, 4]])
    output = model(pixels, grid)
    output.pooler_output[:4].square().sum().backward()
    assert pixels.grad[:16].abs().sum() > 0
    assert torch.count_nonzero(pixels.grad[16:]) == 0
    changed = pixels.detach().clone()
    changed[16:] *= 3
    other = model(changed, grid)
    for a, b in zip(
        (output.pooler_output, *output.deepstack_features),
        (other.pooler_output, *other.deepstack_features),
        strict=True,
    ):
        torch.testing.assert_close(a[:4], b[:4], rtol=0, atol=0)
        assert not torch.equal(a[4:], b[4:])


def test_two_live_video_graphs_have_independent_backward():
    model = Qwen3OmniMoeVideoEncoder(vision_config())
    alone = copy.deepcopy(model)
    grid = torch.tensor([[2, 4, 4]])
    a, b = torch.randn(32, 24), torch.randn(32, 24)

    def objective(output):
        return sum(v.square().sum() for v in (output.pooler_output, *output.deepstack_features))

    first, second = model(a, grid), model(b, grid)
    objective(second).backward()
    objective(first).backward()
    (objective(alone(a, grid)) + objective(alone(b, grid))).backward()
    assert_gradients(model, alone)


def video_fixture():
    config = thinker_config()
    tokens = torch.tensor([[7, 11, 14, 14, 14, 14, 12, 8]])
    inputs = dict(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        video_grid_thw=torch.tensor([[2, 2, 4]]),
        pixel_values_videos=torch.randn(16, 24),
        video_second_per_grid=torch.tensor([0.13]),
    )
    return config, tokens, inputs


@pytest.mark.parametrize(
    "change",
    [
        "count",
        "nan",
        "padding",
        "audio",
        "image",
        "extra",
        "position",
        "missing_pixels",
        "missing_grid",
        "missing_interval",
        "negative_interval",
        "nan_interval",
        "interval_count",
        "zero_frames",
        "odd_grid",
        "bos",
        "eos",
        "mismatched_ids",
    ],
)
def test_invalid_video_inputs_fail_before_collectives(change):
    config, tokens, context = video_fixture()
    if change == "count":
        tokens[0, 3] = 9
    elif change == "nan":
        context["pixel_values_videos"][0, 0] = float("nan")
    elif change == "padding":
        context["attention_mask"][0, -1] = 0
    elif change in ("audio", "image"):
        tokens[0, -1] = getattr(config, change + "_token_id")
    elif change == "extra":
        context["image_grid_thw"] = torch.tensor([[1, 2, 4]])
    elif change == "position":
        context["position_ids"] = torch.zeros(3, 1, tokens.shape[1])
    elif change.startswith("missing_"):
        del context[
            dict(
                pixels="pixel_values_videos",
                grid="video_grid_thw",
                interval="video_second_per_grid",
            )[change[8:]]
        ]
    elif change in ("negative_interval", "nan_interval", "interval_count"):
        context["video_second_per_grid"] = {
            "negative_interval": torch.tensor([-1.0]),
            "nan_interval": torch.tensor([float("nan")]),
            "interval_count": torch.tensor([0.1, 0.2]),
        }[change]
    elif change in ("zero_frames", "odd_grid"):
        context["video_grid_thw"][0, 0 if change == "zero_frames" else 1] = (
            0 if change == "zero_frames" else 3
        )
    elif change in ("bos", "eos"):
        tokens[0, 1 if change == "bos" else -2] = 9
    elif change == "mismatched_ids":
        context["input_ids"] = tokens + 1
    with pytest.raises(ValueError):
        validate_video_inputs(tokens, context, config)


def test_later_pipeline_stage_uses_metadata_without_pixels():
    config, tokens, inputs = video_fixture()
    expected = validate_video_inputs(tokens, inputs, config)
    del inputs["pixel_values_videos"]
    torch.testing.assert_close(
        validate_video_inputs(tokens, inputs, config, require_pixels=False), expected
    )
    assert (expected != expected.long()).any()


def test_video_registration_and_default_selection():
    path = Path(__file__).parents[1] / "pithtrain/modules/training.py"
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "model_class_for_config"
    )
    image, text, video = object(), object(), object()
    ns = dict(
        Qwen3OmniMoeThinkerImageModel=image,
        Qwen3OmniMoeThinkerTextModel=text,
        Qwen3OmniMoeThinkerVideoModel=video,
        resolve_text_model_config=resolve_text_model_config,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)
    choose, config = ns[function.name], thinker_config()
    assert choose(config) is text
    assert choose(config, native_vision=True) is image
    assert choose(config, native_video=True) is video
    with pytest.raises(ValueError, match="Select one"):
        choose(config, native_video=True, native_vision=True)
    with pytest.raises(ValueError, match="composite"):
        choose(config.text_config, native_video=True)


def test_video_consumer_has_no_hf_import_or_cached_features():
    root = Path(__file__).parents[1] / "pithtrain/models"
    for file in ("qwen3_omni_video.py", "qwen3_omni_video_inputs.py"):
        tree = ast.parse((root / file).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("transformers")
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"video_features", "model_context"}
