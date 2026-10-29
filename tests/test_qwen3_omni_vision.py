"""Executable CPU numerical gates for native vision and image consumer geometry."""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFPositions,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as HFVision,
)

from pithtrain.models.qwen3_omni_image_inputs import (
    image_position_ids,
    inject_deepstack,
    pack_image_features,
    validate_image_inputs,
)
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVisionEncoder, VisionOutput


def vision_config(**kwargs):
    values = dict(
        depth=3,
        hidden_size=32,
        num_heads=4,
        intermediate_size=48,
        out_hidden_size=24,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=36,
        deepstack_visual_indexes=[0, 2],
    )
    values.update(kwargs)
    return Qwen3OmniMoeVisionEncoderConfig(**values)


def paired(config, device="cpu", dtype=torch.float32):
    torch.manual_seed(91)
    config = copy.deepcopy(config)
    config._attn_implementation = "eager"
    reference = HFVision(config).to(device=device, dtype=dtype)
    native = Qwen3OmniMoeVisionEncoder(config).to(device=device, dtype=dtype)
    native.load_state_dict(reference.state_dict(), strict=True)
    assert dict(native.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    return native, reference


def assert_gradients(native, reference, rtol=2e-4, atol=2e-6):
    for name, parameter in native.named_parameters():
        expected = reference.get_parameter(name)
        assert parameter.grad is not None and expected.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=rtol, atol=atol, msg=name)


@pytest.mark.parametrize("grids", [[[1, 4, 6]], [[1, 2, 2], [1, 6, 4]], [[1, 8, 10]]])
@pytest.mark.parametrize("activation", ["gelu_pytorch_tanh", "gelu", "silu"])
def test_vision_outputs_all_parameter_gradients_and_update(grids, activation):
    config = vision_config(hidden_act=activation)
    native, reference = paired(config)
    grid = torch.tensor(grids)
    count = sum(h * w for _, h, w in grids)
    pixels = torch.randn(count, 24, requires_grad=True)
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
    torch.optim.AdamW(native.parameters(), lr=1e-3, weight_decay=0).step()
    for name, p in native.named_parameters():
        assert not torch.equal(before[name], p), f"No update: {name}"


def thinker_config():
    return Qwen3OmniMoeThinkerConfig(
        vision_config=vision_config(),
        image_token_id=10,
        vision_start_token_id=11,
        vision_end_token_id=12,
        audio_token_id=13,
        video_token_id=14,
    )


def hf_positions(tokens, grids, config, mask):
    owner = SimpleNamespace(
        config=config, spatial_merge_size=config.vision_config.spatial_merge_size
    )
    owner.get_llm_pos_ids_for_vision = HFPositions.get_llm_pos_ids_for_vision.__get__(owner)
    return HFPositions.get_rope_index(owner, tokens, image_grid_thw=grids, attention_mask=mask)[0]


@pytest.mark.parametrize("padding", [False, True])
def test_positions_match_independent_hf_for_multiple_images(padding):
    config = thinker_config()
    tokens = torch.tensor([[7, 11, 10, 10, 12, 8, 11, 10, 10, 10, 10, 12, 9, 0]])
    grid = torch.tensor([[1, 2, 4], [1, 4, 4]])
    mask = torch.ones_like(tokens)
    mask[:, -1] = int(not padding)
    torch.testing.assert_close(
        image_position_ids(tokens, grid, config, mask).float(),
        hf_positions(tokens, grid, config, mask),
        rtol=0,
        atol=0,
    )


def test_text_positions_match_hf():
    tokens = torch.tensor([[1, 2, 3, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    config = thinker_config()
    torch.testing.assert_close(
        image_position_ids(tokens, None, config, mask).float(),
        hf_positions(tokens, None, config, mask),
        rtol=0,
        atol=0,
    )


def test_two_live_microbatch_graphs_do_not_share_features_or_gradients():
    native, _ = paired(vision_config())
    alone = copy.deepcopy(native)
    grid = torch.tensor([[1, 4, 4]])
    a, b = torch.randn(16, 24), torch.randn(16, 24)
    first, second = native(a, grid), native(b, grid)
    first_alone, second_alone = alone(a, grid), alone(b, grid)
    assert not torch.equal(first.pooler_output, second.pooler_output)

    # Reverse order simulates overlapping forward/backward lifetimes.
    def objective(output):
        return output.pooler_output.square().sum() + sum(
            v.square().sum() for v in output.deepstack_features
        )

    objective(second).backward()
    objective(first).backward()
    (objective(first_alone) + objective(second_alone)).backward()
    assert_gradients(native, alone)


def test_deepstack_transport_matches_explicit_injection_and_backward():
    torch.manual_seed(3)
    embeddings = torch.randn(1, 6, 24, requires_grad=True)
    features = [torch.randn(2, 24, requires_grad=True) for _ in range(3)]
    tokens = torch.tensor([[7, 11, 10, 10, 12, 8]])
    output = VisionOutput(torch.empty(0), features[0], tuple(features[1:]))
    payload = pack_image_features(embeddings, tokens, 10, output, 2)
    actual = payload
    expected = embeddings.clone().masked_scatter((tokens == 10)[..., None], features[0])
    for layer in range(4):
        actual = inject_deepstack(actual[..., :24] * 1.1, actual[..., 24:], layer, 24, 2)
        expected = expected * 1.1
        if layer < 2:
            expected = expected + torch.zeros_like(expected).masked_scatter(
                (tokens == 10)[..., None], features[layer + 1]
            )
    torch.testing.assert_close(actual[..., :24], expected)
    grads = torch.autograd.grad(
        actual[..., :24].square().sum(), (embeddings, *features), retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected.square().sum(), (embeddings, *features))
    for a, b in zip(grads, expected_grads):
        torch.testing.assert_close(a, b)
        assert a.abs().sum() > 0


@pytest.mark.parametrize("grids", [[[2, 4, 4]], [[1, 0, 4]], [[1, 3, 4]], [], [[1, 4]]])
def test_invalid_grids_fail_before_compute(grids):
    with pytest.raises(ValueError):
        Qwen3OmniMoeVisionEncoder(vision_config())(torch.randn(16, 24), torch.tensor(grids))


@pytest.mark.parametrize(
    "change",
    ["count", "nan", "padding", "video", "extra", "position", "missing_pixels", "missing_grid"],
)
def test_bad_image_microbatches_rejected(change):
    config = thinker_config()
    tokens = torch.tensor([[7, 11, 10, 10, 12, 8]])
    context = dict(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
        pixel_values=torch.randn(8, 24),
    )
    if change == "count":
        tokens[0, 3] = 9
    elif change == "nan":
        context["pixel_values"][0, 0] = float("nan")
    elif change == "padding":
        context["attention_mask"][0, -1] = 0
    elif change == "video":
        tokens[0, -1] = config.video_token_id
    elif change == "extra":
        context["pixel_values_videos"] = torch.randn(8, 24)
    elif change == "position":
        context["position_ids"] = torch.zeros(3, 1, 6)
    elif change == "missing_pixels":
        del context["pixel_values"]
    elif change == "missing_grid":
        del context["image_grid_thw"]
    with pytest.raises(ValueError):
        validate_image_inputs(tokens, context, config)


@pytest.mark.parametrize(
    "change",
    [
        dict(deepstack_visual_indexes=[2, 0]),
        dict(deepstack_visual_indexes=[0, 0]),
        dict(deepstack_visual_indexes=[3]),
        dict(num_position_embeddings=35),
        dict(hidden_size=30),
        dict(hidden_act="unknown"),
    ],
)
def test_unsupported_vision_config_rejected(change):
    with pytest.raises(ValueError):
        Qwen3OmniMoeVisionEncoder(vision_config(**change))


def test_shared_token_context_cannot_disagree_with_embedding_input():
    config = thinker_config()
    tokens = torch.tensor([[7, 8]])
    with pytest.raises(ValueError, match="must match"):
        validate_image_inputs(tokens, {"input_ids": torch.tensor([[8, 7]])}, config)


def test_empty_image_token_sequence_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        validate_image_inputs(torch.empty(1, 0, dtype=torch.long), {}, thinker_config())


@pytest.mark.parametrize("seed", [18, 33])
def test_mixed_media_preserves_bf16_gradient_boundaries_in_one_call(seed):
    from torch.func import functional_call

    native, reference = paired(vision_config())
    torch.manual_seed(seed)
    grid = torch.tensor([[1, 4, 4]])
    pixels = [torch.randn(16, 24).bfloat16() for _ in range(2)]

    class SeparateMedia(torch.nn.Module):
        def __init__(self, visual):
            super().__init__()
            self.visual = visual

        def forward(self):
            outputs = [self.visual(value, grid) for value in pixels]
            return tuple(
                torch.cat(values)
                for values in zip(
                    *[(v.pooler_output, *v.deepstack_features) for v in outputs], strict=True
                )
            )

    oracle = SeparateMedia(reference)
    expected = functional_call(oracle, {n: p.bfloat16() for n, p in oracle.named_parameters()}, ())
    calls = []
    hook = native.register_forward_hook(lambda *args: calls.append(1))
    try:
        output = functional_call(
            native,
            {n: p.bfloat16() for n, p in native.named_parameters()},
            (torch.cat(pixels), torch.cat((grid, grid))),
            {"grid_splits": (1, 1)},
        )
    finally:
        hook.remove()
    actual = (output.pooler_output, *output.deepstack_features)
    assert calls == [1]
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    probes = [torch.randn_like(value) for value in expected]
    torch.autograd.backward(expected, probes)
    torch.autograd.backward(actual, probes)
    assert_gradients(native, reference, rtol=0, atol=0)
