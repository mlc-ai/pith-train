"""CPU parity against the independent Transformers 5.17 code2wav reference."""

import ast
import copy
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav as HFCode2Wav,
)

from pithtrain.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav


def tiny_config(**changes):
    values = dict(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=24,
        num_hidden_layers=2,
        num_quantizers=3,
        codebook_size=8,
        decoder_dim=16,
        upsample_rates=[2, 2],
        upsampling_ratios=[2, 2],
        sliding_window=3,
        attention_dropout=0.0,
    )
    values.update(changes)
    config = Qwen3OmniMoeCode2WavConfig(**values)
    config._attn_implementation = "eager"
    return config


def paired(config=None):
    config = tiny_config() if config is None else config
    torch.manual_seed(91)
    reference = HFCode2Wav(copy.deepcopy(config)).float()
    # Excite all branches; tiny HF initial convolutions otherwise yield vanishing
    # gradients that make an absolute-only comparison uninformative.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith((".alpha", ".beta")):
                parameter.uniform_(-0.2, 0.2)
            elif name.endswith((".scale", ".gamma")):
                parameter.fill_(0.4)
            elif "norm" in name and name.endswith("weight"):
                parameter.fill_(1.0)
            else:
                parameter.uniform_(-0.15, 0.15)
    native = Qwen3OmniMoeCode2Wav(copy.deepcopy(config)).float()
    native.load_state_dict(reference.state_dict(), strict=True)
    assert dict(native.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    return native, reference


def compare_gradients(native, reference):
    for name, actual in native.named_parameters():
        expected = reference.get_parameter(name)
        assert actual.grad is not None and expected.grad is not None, name
        assert torch.isfinite(actual.grad).all() and torch.isfinite(expected.grad).all(), name
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7, msg=name)
        denominator = actual.grad.double().square().sum() + expected.grad.double().square().sum()
        if denominator > 0:
            nse = (actual.grad.double() - expected.grad.double()).square().sum() / denominator
            assert nse < 1e-8, (name, nse.item())


@pytest.mark.parametrize("length", [1, 5, 9])
@pytest.mark.parametrize("activation,bias", [("silu", False), ("gelu", True)])
def test_forward_all_gradients_and_first_adam_update(length, activation, bias):
    native, reference = paired(tiny_config(hidden_act=activation, attention_bias=bias))
    codes = torch.randint(0, 8, (2, 3, length))
    actual, expected = native(codes), reference(codes)
    assert actual.shape == (2, 1, native.output_length(length))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    target = torch.randn_like(actual)
    ((actual - target).square().sum()).backward()
    ((expected - target).square().sum()).backward()
    compare_gradients(native, reference)
    before = {name: p.detach().clone() for name, p in native.named_parameters()}
    for model in (native, reference):
        torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0, foreach=False).step()
    assert any(not torch.equal(before[name], p) for name, p in native.named_parameters())
    for name, actual in native.named_parameters():
        torch.testing.assert_close(
            actual, reference.get_parameter(name), rtol=1e-5, atol=1e-7, msg=name
        )


@pytest.mark.parametrize("chunk_size,left_context", [(1, 0), (4, 0), (4, 2), (4, 25), (20, 25)])
def test_chunk_stream_reference_and_every_gradient(chunk_size, left_context):
    native, reference = paired()
    codes = torch.randint(0, 8, (2, 3, 9))
    actual = native.chunked_decode(codes, chunk_size, left_context)
    expected = reference.chunked_decode(codes, chunk_size, left_context)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    streamed = torch.cat(
        list(native.decode_stream(codes.split(chunk_size, dim=-1), left_context)), dim=-1
    )
    torch.testing.assert_close(streamed, actual, rtol=0, atol=0)
    target = torch.randn_like(actual)
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    compare_gradients(native, reference)


def test_stream_is_lazy_and_causal():
    native, reference = paired()
    codes = torch.randint(0, 8, (2, 3, 9))
    supplied = []

    def source():
        for index, chunk in enumerate(codes.split(4, dim=-1)):
            supplied.append(index)
            yield chunk

    iterator = native.decode_stream(source(), left_context_size=2)
    first = next(iterator)
    assert supplied == [0]
    torch.testing.assert_close(first, reference(codes[..., :4]), rtol=1e-5, atol=1e-7)
    changed = codes.clone()
    changed[..., 4:] = (changed[..., 4:] + 1) % 8
    torch.testing.assert_close(
        native(codes)[..., : native.output_length(4)],
        native(changed)[..., : native.output_length(4)],
        rtol=0,
        atol=0,
    )


def test_ragged_batch_ignores_poisoned_padding_and_allows_empty_audio():
    native, reference = paired()
    codes = torch.randint(0, 8, (3, 3, 9))
    lengths = torch.tensor([9, 3, 0])
    codes[1, :, 3:] = 2150  # EOS/special token must not be embedded.
    codes[2] = -100
    waves = native.decode_batch(codes, lengths, chunk_size=4, left_context_size=2)
    for i, length in enumerate(lengths[:2]):
        expected = reference.chunked_decode(codes[i : i + 1, :, :length], 4, 2)[0]
        torch.testing.assert_close(waves[i], expected, rtol=1e-5, atol=1e-7)
    assert waves[2].shape == (1, 0)


def test_released_default_meta_parameter_shapes_and_nonpersistent_offsets():
    config = Qwen3OmniMoeCode2WavConfig()
    config._attn_implementation = "eager"
    with torch.device("meta"):
        native, reference = Qwen3OmniMoeCode2Wav(config), HFCode2Wav(copy.deepcopy(config))
    actual = {name: tuple(p.shape) for name, p in native.state_dict().items()}
    expected = {name: tuple(p.shape) for name, p in reference.state_dict().items()}
    assert actual == expected
    assert len(actual) == 230
    assert sum(p.numel() for p in native.parameters()) == 216016577
    assert native.total_upsample == 1920
    assert native.output_length(1) == 1365
    assert native.output_length(300) == 575445
    assert not any("code_offset" in name or "inv_freq" in name for name in actual)


def test_meta_materialization_and_state_round_trip(tmp_path):
    native, reference = paired()
    with torch.device("meta"):
        restored = Qwen3OmniMoeCode2Wav(tiny_config())
    restored.to_empty(device="cpu")
    path = tmp_path / "code2wav.pt"
    torch.save(native.state_dict(), path)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    codes = torch.randint(0, 8, (2, 3, 7))
    torch.testing.assert_close(restored(codes), reference(codes), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "codes",
    [
        torch.zeros((1, 3, 2)),
        torch.zeros((1, 3), dtype=torch.long),
        torch.zeros((1, 2, 2), dtype=torch.long),
        torch.zeros((0, 3, 2), dtype=torch.long),
        torch.zeros((1, 3, 0), dtype=torch.long),
        torch.full((1, 3, 2), -1),
        torch.full((1, 3, 2), 8),
    ],
)
def test_invalid_codes_rejected(codes):
    with pytest.raises(ValueError):
        Qwen3OmniMoeCode2Wav(tiny_config())(codes)


@pytest.mark.parametrize(
    "changes",
    [
        dict(num_quantizers=0),
        dict(codebook_size=0),
        dict(upsample_rates=[0]),
        dict(upsampling_ratios=[]),
        dict(decoder_dim=15),
        dict(sliding_window=0),
        dict(hidden_size=15),
        dict(num_key_value_heads=3),
        dict(hidden_act="mish"),
        dict(attention_dropout=1),
    ],
)
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        Qwen3OmniMoeCode2Wav(tiny_config(**changes))


def test_unsupported_rope_rejected():
    config = tiny_config()
    config.rope_parameters["rope_type"] = "linear"
    with pytest.raises(NotImplementedError, match="RoPE"):
        Qwen3OmniMoeCode2Wav(config)


@pytest.mark.parametrize("chunk_size,left_context", [(0, 2), (-1, 2), (2, -1), (True, 2)])
def test_invalid_chunk_policy_rejected(chunk_size, left_context):
    native = Qwen3OmniMoeCode2Wav(tiny_config())
    with pytest.raises(ValueError):
        native.chunked_decode(torch.zeros((1, 3, 2), dtype=torch.long), chunk_size, left_context)


@pytest.mark.parametrize(
    "lengths",
    [torch.tensor([3]), torch.tensor([-1, 2]), torch.tensor([3, 2]), torch.tensor([1.0, 2.0])],
)
def test_invalid_ragged_lengths_rejected(lengths):
    native = Qwen3OmniMoeCode2Wav(tiny_config())
    with pytest.raises(ValueError):
        native.decode_batch(torch.zeros((2, 3, 2), dtype=torch.long), lengths)


def test_stream_batch_change_rejected():
    native = Qwen3OmniMoeCode2Wav(tiny_config())
    with pytest.raises(ValueError, match="same batch"):
        list(
            native.decode_stream(
                [torch.zeros((1, 3, 2), dtype=torch.long), torch.zeros((2, 3, 2), dtype=torch.long)]
            )
        )


def test_native_module_does_not_import_transformers():
    path = Path(__file__).resolve().parents[1] / "pithtrain/models/qwen3_omni_code2wav.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("transformers")
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("transformers") for alias in node.names)


def test_empty_stream_and_zero_length_eos_batch():
    native, reference = paired()
    empty = torch.empty((2, 3, 0), dtype=torch.long)
    assert list(native.decode_stream([])) == []
    assert list(native.decode_stream([empty, empty])) == []
    waves = native.decode_batch(empty, torch.zeros(2, dtype=torch.long))
    assert [tuple(w.shape) for w in waves] == [(1, 0), (1, 0)]
    codes = torch.randint(0, 8, (2, 3, 4))
    streamed = list(native.decode_stream([empty, codes, empty]))
    assert len(streamed) == 1
    torch.testing.assert_close(streamed[0], reference(codes), rtol=1e-5, atol=1e-7)
