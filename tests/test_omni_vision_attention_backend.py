"""Explicit native visual attention backends must follow the HF configuration."""

import copy

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionAttention as HFAttention,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as HFVision,
)

from pithtrain.models.qwen3_omni_vision import (
    Qwen3OmniMoeVideoEncoder,
    Qwen3OmniMoeVisionAttention,
    Qwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeVisionRotaryEmbedding,
    vision_geometry,
)
from tests.test_qwen3_omni_vision import vision_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grids", [[[1, 8, 10]], [[1, 2, 2], [1, 4, 6]]])
def test_explicit_eager_matches_hf_output_and_every_gradient_exactly(monkeypatch, dtype, grids):
    torch.manual_seed(91)
    config = vision_config()
    config._attn_implementation = "eager"
    reference = HFAttention(config).to(dtype)
    native = Qwen3OmniMoeVisionAttention(config).to(dtype)
    native.load_state_dict(reference.state_dict(), strict=True)
    pos, _, _, lengths = vision_geometry(torch.tensor(grids), 2, 6)
    embeddings = Qwen3OmniMoeVisionRotaryEmbedding(config)(pos)
    cu_seqlens = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0)
    inputs = torch.randn(len(pos), config.hidden_size, dtype=dtype, requires_grad=True)
    expected_inputs = inputs.detach().clone().requires_grad_()

    def unexpected_sdpa(*args, **kwargs):
        raise AssertionError("Explicit eager configuration must not execute SDPA")

    monkeypatch.setattr(F, "scaled_dot_product_attention", unexpected_sdpa)
    actual = native(inputs, lengths, embeddings)
    expected = reference(expected_inputs, cu_seqlens, position_embeddings=embeddings)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cotangent = torch.randn_like(actual)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=0, atol=0)
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad, reference.get_parameter(name).grad, rtol=0, atol=0, msg=name
        )


@pytest.mark.parametrize("backend", [None, "sdpa"])
def test_default_and_explicit_sdpa_preserve_native_sdpa_path(monkeypatch, backend):
    config = vision_config()
    config._attn_implementation = backend
    model = Qwen3OmniMoeVisionAttention(config)
    original = F.scaled_dot_product_attention
    calls = []

    def observe(*args, **kwargs):
        calls.append((args[0].shape, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", observe)
    pos, _, _, lengths = vision_geometry(torch.tensor([[1, 2, 2], [1, 4, 6]]), 2, 6)
    result = model(
        torch.randn(len(pos), config.hidden_size),
        lengths,
        Qwen3OmniMoeVisionRotaryEmbedding(config)(pos),
    )
    assert result.shape == (28, config.hidden_size)
    assert len(calls) == 2
    assert all(
        len(shape) == 4 and shape[0] == 1 and shape[1] == config.num_heads for shape, _ in calls
    )
    assert [shape[2] for shape, _ in calls] == [4, 24]
    assert all(options == {"dropout_p": 0.0, "is_causal": False} for _, options in calls)


def test_unsupported_explicit_attention_backend_fails_before_computation():
    config = vision_config()
    config._attn_implementation = "flash_attention_2"
    with pytest.raises(NotImplementedError, match="Native vision attention"):
        Qwen3OmniMoeVisionAttention(config)


def test_original_silu_non_square_encoder_case_matches_every_gradient_exactly():
    torch.manual_seed(91)
    config = vision_config(hidden_act="silu")
    config._attn_implementation = "eager"
    reference = HFVision(config)
    native = Qwen3OmniMoeVisionEncoder(config)
    native.load_state_dict(reference.state_dict(), strict=True)
    pixels = torch.randn(80, 24, requires_grad=True)
    expected_pixels = pixels.detach().clone().requires_grad_()
    actual, expected = (
        native(pixels, torch.tensor([[1, 8, 10]])),
        reference(expected_pixels, torch.tensor([[1, 8, 10]])),
    )
    left = (actual.last_hidden_state, actual.pooler_output, *actual.deepstack_features)
    right = (expected.last_hidden_state, expected.pooler_output, *expected.deepstack_features)
    probes = [torch.randn_like(value) for value in left]
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sum((value * probe).sum() for value, probe in zip(left, probes)).backward()
    sum((value * probe).sum() for value, probe in zip(right, probes)).backward()
    torch.testing.assert_close(pixels.grad, expected_pixels.grad, rtol=0, atol=0)
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad, reference.get_parameter(name).grad, rtol=0, atol=0, msg=name
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("native_backend", [None, "sdpa"])
@pytest.mark.parametrize(
    "grids",
    [
        [[1, 8, 10]],
        [[1, 2, 2], [1, 4, 6]],
        [[2, 4, 6], [3, 2, 2]],
    ],
)
def test_sdpa_full_encoder_matches_official_hf_every_output_and_gradient(
    dtype, native_backend, grids
):
    """Independent official encoder verifies image/frame segmentation and all VJPs."""
    torch.manual_seed(91)
    config = vision_config()
    config._attn_implementation = "sdpa"
    # Match the production BF16 reference construction: casting the whole
    # module would also round HF's non-persistent FP32 RoPE buffers to BF16.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        reference = HFVision(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert reference.rotary_pos_emb.inv_freq.dtype == torch.float32
    native_config = copy.deepcopy(config)
    native_config._attn_implementation = native_backend
    model_type = (
        Qwen3OmniMoeVideoEncoder if any(t > 1 for t, _, _ in grids) else Qwen3OmniMoeVisionEncoder
    )
    native = model_type(native_config).to(dtype)
    native.load_state_dict(reference.state_dict(), strict=True)
    assert dict(native.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    grid = torch.tensor(grids)
    pixels = torch.randn(sum(t * h * w for t, h, w in grids), 24, dtype=dtype, requires_grad=True)
    ref_pixels = pixels.detach().clone().requires_grad_()
    actual, expected = native(pixels, grid), reference(ref_pixels, grid)
    left = (actual.last_hidden_state, actual.pooler_output, *actual.deepstack_features)
    right = (expected.last_hidden_state, expected.pooler_output, *expected.deepstack_features)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    cotangents = [torch.randn_like(value) for value in left]
    torch.autograd.backward(left, cotangents)
    torch.autograd.backward(right, cotangents)
    assert pixels.grad is not None and ref_pixels.grad is not None
    torch.testing.assert_close(pixels.grad, ref_pixels.grad, rtol=0, atol=0)
    for name, parameter in native.named_parameters():
        expected_gradient = reference.get_parameter(name).grad
        assert parameter.grad is not None and expected_gradient is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0, msg=name)
