"""CPU geometry/encoder composition gates; distributed GPU acceptance is separate."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as HFAudio,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFPositions,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as HFVision,
)

from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from pithtrain.models.qwen3_omni_audio_inputs import flatten_audio_features
from pithtrain.models.qwen3_omni_multimodal_inputs import (
    encode_multimodal,
    multimodal_position_ids,
)
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVideoEncoder
from tests.test_qwen3_omni_audio import audio_config
from tests.test_qwen3_omni_vision import vision_config


def config():
    return Qwen3OmniMoeThinkerConfig(
        vision_config=vision_config(),
        audio_config=audio_config(),
        text_config=dict(hidden_size=24, vocab_size=32),
        image_token_id=10,
        video_token_id=11,
        audio_token_id=12,
        vision_start_token_id=13,
        vision_end_token_id=14,
        audio_start_token_id=15,
        audio_end_token_id=16,
        position_id_per_seconds=13,
    )


KINDS = ("text", "image", "audio", "video", "image+audio", "image+video", "av", "all")


def context(kind, *, seconds=0.73, frames=201):
    """Small controlled streams with non-square grids and two different audio clips."""
    cfg, out, tokens = config(), {}, [1, 2]
    image = kind in ("image", "image+audio", "image+video", "all")
    audio = kind in ("audio", "image+audio", "all")
    video = kind in ("video", "image+video", "av", "all")
    sync = kind in ("av", "all")
    lengths = []
    if audio:
        lengths.append(17)
        tokens += [15] + [12] * 3 + [16, 3]
    if image:
        out["image_grid_thw"] = torch.tensor([[1, 2, 4]])
        out["pixel_values"] = torch.randn(8, 24)
        tokens += [13, 10, 10, 14, 4]
    if video:
        out["video_grid_thw"] = torch.tensor([[3, 4, 6]])
        out["pixel_values_videos"] = torch.randn(72, 24)
        out["video_second_per_grid"] = torch.tensor([seconds])
        if sync:
            count = frames // 100 * 13 + (frames % 100 + 7) // 8
            lengths.append(frames)
            # Stable independent event sort implements the processor's merge.
            times = torch.arange(3).float() * out["video_second_per_grid"][0] * 13
            events = [(float(t), 0, 11) for t in times for _ in range(6)]
            events += [(float(index), 1, 12) for index in range(count)]
            events.sort(key=lambda item: item[:2])
            tokens += [13, 15] + [item[2] for item in events] + [16, 14, 5]
            out["use_audio_in_video"] = torch.tensor(True)
        else:
            tokens += [13] + [11] * 18 + [14, 5]
    if lengths:
        out["feature_attention_mask"] = (
            torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
        )
        out["input_features"] = torch.randn(len(lengths), 16, max(lengths))
    tokens += [6, 7, 8]
    out["input_ids"] = torch.tensor([tokens])
    out["attention_mask"] = torch.ones_like(out["input_ids"])
    return cfg, out


def hf_positions(cfg, inputs):
    owner = SimpleNamespace(config=cfg, spatial_merge_size=cfg.vision_config.spatial_merge_size)
    owner.get_llm_pos_ids_for_vision = HFPositions.get_llm_pos_ids_for_vision.__get__(owner)
    lengths = inputs.get("feature_attention_mask")
    return HFPositions.get_rope_index(
        owner,
        inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs["attention_mask"],
        audio_seqlens=None if lengths is None else lengths.sum(-1),
        second_per_grids=inputs.get("video_second_per_grid"),
        use_audio_in_video=bool(inputs.get("use_audio_in_video", False)),
    )[0]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seconds,frames", [(0.125, 8), (0.73, 201), (2.5, 99)])
def test_exact_hf_positions_for_mixed_inputs_and_av(kind, seconds, frames):
    cfg, inputs = context(kind, seconds=seconds, frames=frames)
    actual = multimodal_position_ids(inputs["input_ids"], inputs, cfg)
    torch.testing.assert_close(actual, hf_positions(cfg, inputs), rtol=0, atol=0)
    # All decoder stages can reconstruct positions without large encoder payloads.
    metadata = {
        k: v
        for k, v in inputs.items()
        if k not in ("pixel_values", "pixel_values_videos", "input_features")
    }
    torch.testing.assert_close(
        actual,
        multimodal_position_ids(inputs["input_ids"], metadata, cfg, require_media=False),
        rtol=0,
        atol=0,
    )


class Encoders(nn.Module):
    def __init__(self, cfg, *, reference=False):
        super().__init__()
        self.config = copy.deepcopy(cfg)
        self.config.vision_config._attn_implementation = "eager"
        self.config.audio_config._attn_implementation = "eager"
        self.visual = (HFVision if reference else Qwen3OmniMoeVideoEncoder)(
            self.config.vision_config
        )
        self.audio_tower = (HFAudio if reference else Qwen3OmniMoeAudioEncoder)(
            self.config.audio_config
        )
        self.embed_tokens = nn.Embedding(32, 24)

    def forward(self, inputs):
        ids = inputs["input_ids"]
        return encode_multimodal(
            self.visual, self.audio_tower, self.embed_tokens(ids), ids, inputs, self.config
        )

    def reference(self, inputs):
        """Independent HF encoders and explicit sparse insertion, without our helper."""
        ids = inputs["input_ids"]
        hidden = self.embed_tokens(ids)
        lanes = [torch.zeros_like(hidden) for _ in range(2)]
        for token, pixels, grids in (
            (10, "pixel_values", "image_grid_thw"),
            (11, "pixel_values_videos", "video_grid_thw"),
        ):
            if pixels not in inputs:
                continue
            output = self.visual(inputs[pixels], inputs[grids])
            mask = (ids == token)[..., None]
            hidden = hidden.masked_scatter(mask, output.pooler_output)
            for i, feature in enumerate(output.deepstack_features):
                lanes[i] = lanes[i].masked_scatter(mask, feature)
        if "input_features" in inputs:
            features = flatten_audio_features(inputs)
            lengths = inputs["feature_attention_mask"].sum(-1)
            output = self.audio_tower(features, feature_lens=lengths).last_hidden_state
            hidden = hidden.masked_scatter((ids == 12)[..., None], output)
        return torch.cat((hidden, *lanes), dim=-1)


@pytest.mark.parametrize("kind", KINDS)
def test_composition_matches_hf_every_gradient_and_uniform_encoder_calls(kind):
    torch.manual_seed(49)
    cfg, inputs = context(kind)
    native, reference = Encoders(cfg), Encoders(cfg, reference=True)
    native.load_state_dict(reference.state_dict(), strict=True)
    calls = []
    backwards = []
    native.visual.register_forward_pre_hook(lambda *args: calls.append("visual"))
    native.audio_tower.register_forward_pre_hook(lambda *args: calls.append("audio"))
    native.visual.register_full_backward_hook(lambda *args: backwards.append("visual"))
    native.audio_tower.register_full_backward_hook(lambda *args: backwards.append("audio"))
    native.embed_tokens.register_full_backward_hook(lambda *args: backwards.append("embedding"))
    actual, expected = native(inputs), reference.reference(inputs)
    assert calls == ["visual", "audio"]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    assert backwards == ["audio", "visual", "embedding"]
    (expected * probe).sum().backward()
    for name, param in native.named_parameters():
        expected_grad = reference.get_parameter(name).grad
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        if expected_grad is None:
            # HF omits the inactive encoder. Ours must provide exact zero, not None.
            assert (
                name.startswith("visual.")
                and "pixel_values" not in inputs
                and "pixel_values_videos" not in inputs
            ) or (name.startswith("audio_tower.") and "input_features" not in inputs), name
            assert torch.count_nonzero(param.grad) == 0, name
        else:
            torch.testing.assert_close(param.grad, expected_grad, rtol=2e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize(
    "fault", ["order", "audio_count", "wrapper", "flag", "nan", "padding", "unused", "position"]
)
def test_bad_av_fails_before_encoder_collectives(fault):
    cfg, inputs = context("all")
    if fault == "order":
        row = inputs["input_ids"][0]
        i = ((row[:-1] == 11) & (row[1:] == 12)).nonzero()[0].item()
        row[i], row[i + 1] = 12, 11
    elif fault == "audio_count":
        inputs["feature_attention_mask"][-1, -10:] = False
    elif fault == "wrapper":
        inputs["input_ids"][0, -5] = 1
    elif fault == "flag":
        inputs["use_audio_in_video"] = torch.tensor(False)
    elif fault == "nan":
        inputs["input_features"][0, 0, 0] = float("nan")
    elif fault == "padding":
        inputs["attention_mask"][0, -1] = 0
    elif fault == "unused":
        inputs["image_grid_thw"] = torch.cat([inputs["image_grid_thw"]] * 2)
    else:
        inputs["position_ids"] = torch.zeros(3, 1, inputs["input_ids"].shape[1])
    with pytest.raises(ValueError):
        multimodal_position_ids(inputs["input_ids"], inputs, cfg)


def test_interleaved_microbatch_graphs_and_optimizer_resume(tmp_path):
    torch.manual_seed(71)
    cfg = config()
    model = Encoders(cfg)
    sequential = copy.deepcopy(model)
    inputs = [context(kind)[1] for kind in ("text", "audio", "image", "av")]
    outputs = [model(batch) for batch in inputs]
    for output in reversed(outputs):
        output.square().sum().backward()
    for batch in inputs:
        sequential(batch).square().sum().backward()
    for name, param in model.named_parameters():
        torch.testing.assert_close(
            param.grad, sequential.get_parameter(name).grad, rtol=2e-5, atol=2e-6
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = tmp_path / "mixed.pt"
    torch.save(
        dict(model=model.state_dict(), optimizer=optimizer.state_dict(), next_index=4), checkpoint
    )
    restored = Encoders(cfg)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(state["model"], strict=True)
    restored_optimizer.load_state_dict(state["optimizer"])
    assert state["next_index"] == len(inputs)
    batch = context("all")[1]
    before, after = model(batch), restored(batch)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    before.square().sum().backward()
    after.square().sum().backward()
    optimizer.step()
    restored_optimizer.step()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, restored.get_parameter(name), rtol=0, atol=0)
