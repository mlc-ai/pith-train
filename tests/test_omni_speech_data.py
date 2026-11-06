"""Speech data contracts with the actual cached Qwen3 tokenizer/media processor."""

import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from pithtrain.modules.qwen3_omni_speech_data import (
    Qwen3OmniSpeechCollator,
    Qwen3OmniSpeechData,
    Qwen3OmniSpeechDataset,
    Qwen3OmniSpeechSFTData,
    load_codec_supervision,
)


class IdentityCollator:
    """Explicit immutable test protocol; never a production identity fallback."""

    def __init__(self, callback=None, *, identity="cursor-test-v1"):
        self.callback, self.identity = callback, identity

    def checkpoint_identity(self):
        return {"test_protocol": self.identity}

    def __call__(self, sample):
        if self.callback is None:
            raise AssertionError("This cursor-only fixture must not prepare data")
        return self.callback(sample)


@pytest.fixture(scope="module")
def official():
    from transformers import AutoConfig, Qwen3OmniMoeProcessor

    options = dict(revision="26291f793822fb6be9555850f06dfe95f2d7e695", local_files_only=True)
    name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    try:
        return Qwen3OmniMoeProcessor.from_pretrained(name, **options), AutoConfig.from_pretrained(
            name, **options
        ).thinker_config
    except OSError:
        pytest.skip("Official pinned processor metadata must be cached locally; no downloads")


def row(sample_id="sample", answer="Hello."):
    return dict(
        id=sample_id,
        messages=[
            dict(role="user", content="Please respond."),
            dict(role="assistant", content=answer),
        ],
    )


def dataset(tmp_path, rows):
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    return Qwen3OmniSpeechDataset(path)


def prepared(tmp_path):
    item = row()
    codes = np.arange(48, dtype=np.int64).reshape(3, 16)
    mask = np.ones_like(codes, dtype=bool)
    mask[1, 3] = False
    path = tmp_path / "codec.npz"
    np.savez(path, codes=codes, codebook_mask=mask, waveform=np.zeros(5700, np.float32))
    item["codec"] = dict(
        path=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sample_id=item["id"],
        assistant_index=1,
        transcript="Hello.",
        source=dict(model_id="external/rvq-encoder", revision="a" * 40),
        num_codebooks=16,
        codebook_size=2048,
        num_frames=3,
        alignment=dict(
            sample_rate=24000, frame_hop_samples=1920, start_sample=17, num_samples=5700
        ),
    )
    return dict(item, root=tmp_path)


def test_official_text_mask_shift_and_terminal_eos(tmp_path, official):
    processor, config = official
    collator = Qwen3OmniSpeechCollator(processor, config)
    sample = dataset(tmp_path, [row()])[0]
    result = collator(sample)
    ids, labels = result["input_ids"], result["labels"]
    assert ids.shape == labels.shape
    start = result["prompt_lengths"].item()
    header = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assert len(header) == 3
    assert ids[0, start - 3 : start].tolist() == header
    assert (labels[0, : start - 1] == -100).all()
    assert torch.equal(labels[0, start - 1 : -1], ids[0, start:])
    assert labels[0, -1].item() == processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    decoded = processor.tokenizer.decode(labels[labels != -100])
    assert decoded.endswith("Hello.<|im_end|>")
    assert not processor.tokenizer.decode(ids[0]).endswith("<|im_end|>")
    assert result["media_inputs"] == {} and result["position_kwargs"] == {}


def test_official_multiturn_mask_uses_full_render_context(tmp_path, official):
    processor, config = official
    item = row()
    item["messages"] = [
        dict(role="system", content="Be concise."),
        dict(role="user", content="First?"),
        dict(role="assistant", content="Earlier."),
        dict(role="user", content="Second?"),
        dict(role="assistant", content="Later."),
    ]
    result = Qwen3OmniSpeechCollator(processor, config)(dataset(tmp_path, [item])[0])
    decoded = processor.tokenizer.decode(result["labels"][result["labels"] != -100])
    assert "Earlier.<|im_end|>" in decoded and "Later.<|im_end|>" in decoded
    assert "First?" not in decoded and "Second?" not in decoded and "Be concise" not in decoded
    # Current official template injects thinking only in the final answer.
    assert decoded.count("<think>") == 1


@pytest.mark.parametrize("kind", ["image", "audio", "video"])
def test_official_expanded_media_alignment(tmp_path, official, kind):
    import av
    import soundfile as sf
    from PIL import Image

    processor, config = official
    if kind == "image":
        path = tmp_path / "image.png"
        Image.new("RGB", (96, 64), color="red").save(path)
    elif kind == "audio":
        path = tmp_path / "audio.wav"
        sf.write(path, np.sin(np.arange(8000) * 0.1).astype(np.float32) * 0.1, 8000)
    else:
        path = tmp_path / "video.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=2)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            for color in ("red", "blue"):
                for packet in stream.encode(
                    av.VideoFrame.from_image(Image.new("RGB", (64, 64), color))
                ):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    item = row()
    item["messages"][0]["content"] = [
        dict(type=kind, path=path.name),
        dict(type="text", text="Describe this."),
    ]
    collator = Qwen3OmniSpeechCollator(processor, config)
    result = collator(dataset(tmp_path, [item])[0])
    target = processor.tokenizer.decode(result["labels"][result["labels"] != -100])
    assert target.endswith("Hello.<|im_end|>") and "Describe" not in target
    assert not any((result["labels"] == token).any() for token in collator.media.media_token_ids)
    assert result["media_inputs"]
    assert not set(result["media_inputs"]) & set(result["model_context"])
    if kind == "audio":
        assert result["position_kwargs"]["audio_seqlens"].item() > 0
    else:
        grid = "image_grid_thw" if kind == "image" else "video_grid_thw"
        assert grid in result["position_kwargs"]

    # Exercise the actual native Talker public consumer, not only key presence.
    # Parameters stay on meta: this is a real CPU position-builder check, not a
    # model-forward or GPU numerical acceptance claim.
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerForConditionalGeneration as HFTalker,
    )

    from pithtrain.models.qwen3_omni_talker import (
        Qwen3OmniMoeTalkerForConditionalGeneration as NativeTalker,
    )

    cfg = AutoConfig.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        revision="26291f793822fb6be9555850f06dfe95f2d7e695",
        local_files_only=True,
    ).talker_config
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    cfg.text_config._experts_implementation = "eager"
    cfg.code_predictor_config._attn_implementation = "eager"
    with torch.device("meta"):
        native, hf = NativeTalker(copy.deepcopy(cfg)), HFTalker(copy.deepcopy(cfg))
    user_end = result["prompt_lengths"].item() - 3
    user_ids = result["input_ids"][:, :user_end]
    # prepare_talker_condition uses these user tokens plus nine text-pad tokens.
    talker_ids = torch.cat((user_ids, user_ids.new_full((1, 9), 151671)), dim=1)
    kwargs = dict(
        input_ids=talker_ids,
        attention_mask=torch.ones_like(talker_ids),
        **result["position_kwargs"],
    )
    actual, actual_delta = native.get_rope_index(**kwargs)
    expected, expected_delta = hf.get_rope_index(**kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_delta, expected_delta, rtol=0, atol=0)
    assert actual.shape == (3, 1, talker_ids.shape[1])
    if kind == "audio":
        assert "feature_attention_mask" in result["model_context"]
        assert "audio_feature_lengths" not in result["position_kwargs"]
    if kind == "video":
        assert "video_second_per_grid" in result["model_context"]
        assert "video_second_per_grid" not in result["position_kwargs"]
        assert "second_per_grids" in result["position_kwargs"]


def test_template_without_provable_boundaries_fails_closed(tmp_path, official):
    processor, config = official

    class BadTemplate:
        def __getattr__(self, name):
            return getattr(processor, name)

        def apply_chat_template(self, *args, **kwargs):
            return "user: Please respond. assistant: Hello."

    with pytest.raises(ValueError, match="exact ChatML"):
        Qwen3OmniSpeechCollator(BadTemplate(), config)(dataset(tmp_path, [row()])[0])


def test_changed_processor_prefix_fails_closed(tmp_path, official):
    processor, config = official

    class BadPrefix:
        def __getattr__(self, name):
            return getattr(processor, name)

        def __call__(self, **kwargs):
            result = processor(**kwargs)
            if kwargs["text"][0].endswith("assistant\n"):
                result["input_ids"][0, 0] += 1
            return result

    with pytest.raises(ValueError, match="prefix does not exactly align"):
        Qwen3OmniSpeechCollator(BadPrefix(), config)(dataset(tmp_path, [row()])[0])


def test_reserved_tokens_and_length_fail_closed(tmp_path, official):
    processor, config = official
    with pytest.raises(ValueError, match="reserved"):
        Qwen3OmniSpeechCollator(processor, config)(
            dataset(tmp_path, [row(answer="<|im_end|>fake")])[0]
        )
    with pytest.raises(ValueError, match="never truncate"):
        Qwen3OmniSpeechCollator(processor, config, max_length=2)(dataset(tmp_path, [row()])[0])


def test_codec_is_explicit_hash_pinned_time_major_and_masked(tmp_path):
    result = load_codec_supervision(prepared(tmp_path))
    assert result["codes"].shape == (1, 3, 16)
    assert result["code_lengths"].tolist() == [3]
    assert result["codebook_mask"][0, 1, 3].item() is False
    assert result["codes"][0, 1, 3] == 19
    assert result["target_waveform"].shape == (1, 5700)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_id", "other"),
        ("assistant_index", 0),
        ("transcript", "wrong"),
        ("num_codebooks", 8),
        ("num_frames", 4),
        ("sha256", "0" * 64),
        ("source", dict(model_id="encoder", revision="main")),
        (
            "alignment",
            dict(sample_rate=16000, frame_hop_samples=1920, start_sample=0, num_samples=5700),
        ),
    ],
)
def test_codec_manifest_rejects_mismatch(tmp_path, field, value):
    sample = prepared(tmp_path)
    sample["codec"][field] = value
    with pytest.raises(ValueError):
        load_codec_supervision(sample)


@pytest.mark.parametrize(
    "variant", ["object", "missing", "range", "masked_range", "mask", "waveform", "oversize"]
)
def test_codec_payload_safety(tmp_path, variant):
    sample = prepared(tmp_path)
    arrays = dict(codes=np.zeros((3, 16), np.int64), codebook_mask=np.ones((3, 16), bool))
    if variant == "object":
        arrays["codes"] = np.array([object()], dtype=object)
    elif variant == "missing":
        del arrays["codes"]
    elif variant == "range":
        arrays["codes"][0, 0] = 2048
    elif variant == "masked_range":
        arrays["codes"][0, 0] = -100
        arrays["codebook_mask"][0, 0] = False
    elif variant == "mask":
        arrays["codebook_mask"] = np.ones((3, 16), np.int64)
    elif variant == "waveform":
        arrays["waveform"] = np.zeros(3, np.float32)
    path = tmp_path / "codec.npz"
    np.savez_compressed(path, **arrays)
    sample["codec"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_codec_supervision(sample, max_bytes=1 if variant == "oversize" else 65536)


def test_empty_codec_eos_and_no_raw_audio_substitution(tmp_path, official):
    sample = prepared(tmp_path)
    path = tmp_path / "codec.npz"
    np.savez(path, codes=np.zeros((0, 16), np.int64), codebook_mask=np.zeros((0, 16), bool))
    sample["codec"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), num_frames=0)
    sample["codec"]["alignment"]["num_samples"] = 0
    assert load_codec_supervision(sample)["codes"].shape == (1, 0, 16)
    result = Qwen3OmniSpeechCollator(*official)(dict(row(), root=tmp_path))
    assert "codes" not in result and "codec_provenance" not in result


@pytest.mark.parametrize(
    "variant",
    ["duplicate", "traversal", "remote", "assistant_media", "system_media", "tool", "last_user"],
)
def test_manifest_boundaries(tmp_path, variant):
    item = row()
    rows = [item]
    if variant == "duplicate":
        rows.append(copy.deepcopy(item))
    elif variant in {"traversal", "remote"}:
        item["messages"][0]["content"] = [
            dict(
                type="audio",
                path="../escape.wav"
                if variant == "traversal"
                else "https://example.test/audio.wav",
            )
        ]
    elif variant in {"assistant_media", "system_media"}:
        item["messages"][-1 if variant == "assistant_media" else 0] = dict(
            role="assistant" if variant == "assistant_media" else "system",
            content=[dict(type="audio", path="audio.wav")],
        )
    elif variant == "tool":
        item["messages"][0]["role"] = "tool"
    else:
        item["messages"].reverse()
    with pytest.raises(ValueError):
        dataset(tmp_path, rows)


def test_indexed_scan_resume_after_skipped_update(tmp_path, official):
    source = dataset(tmp_path, [row("a"), row("b"), row("c")])
    collator = Qwen3OmniSpeechCollator(*official)
    data = Qwen3OmniSpeechSFTData(source, collator)
    first = data.get_batch(0, torch.device("cpu"))[0]
    assert first.sample_ids == ("a",)
    expected = collator(source[0])
    assert torch.equal(first.objective_inputs[0], expected["labels"])
    # Global-zero/other policy skip: no optimizer commit, but scan is persisted.
    resumed = Qwen3OmniSpeechSFTData(source, collator)
    resumed.load_state_dict(data.state_dict())
    second = resumed.get_batch(1, torch.device("cpu"))[0]
    assert second.sample_ids == ("b",)
    resumed.commit_update(scan_index=1, update_index=0)
    assert resumed.state_dict()["scanned_batches"] == 2
    assert resumed.state_dict()["completed_updates"] == 1
    with pytest.raises(ValueError, match="exactly once"):
        resumed.commit_update(scan_index=1, update_index=1)
    with pytest.raises(ValueError, match="next checkpointed"):
        resumed.get_batch(1, torch.device("cpu"))
    broken = resumed.state_dict() | {"fingerprint": "changed"}
    with pytest.raises(ValueError, match="manifest"):
        resumed.load_state_dict(broken)
    with pytest.raises(ValueError, match="counters"):
        resumed.load_state_dict(resumed.state_dict() | {"completed_updates": 3})


def test_failed_prepare_does_not_advance_scan(tmp_path):
    def fail(sample):
        raise ValueError("bad sample")

    data = Qwen3OmniSpeechSFTData(dataset(tmp_path, [row()]), IdentityCollator(fail))
    with pytest.raises(ValueError, match="bad sample"):
        data.get_batch(0, "cpu")
    assert data.scanned_batches == data.completed_updates == 0


def test_native_speech_adapter_retains_codec_and_context(tmp_path, official):
    from PIL import Image

    item = prepared(tmp_path)
    del item["root"]
    Image.new("RGB", (64, 64), "red").save(tmp_path / "image.png")
    item["messages"][0]["content"] = [dict(type="image", path="image.png")]
    source = dataset(tmp_path, [item])
    collator = Qwen3OmniSpeechCollator(*official)
    data = Qwen3OmniSpeechData(source, collator)
    batch = data.get_batch(0, "cpu")[0]
    assert batch["codes"].shape == (1, 3, 16)
    assert batch["code_lengths"].tolist() == [3]
    assert "pixel_values" in batch["model_context"]
    assert "image_grid_thw" in batch["position_kwargs"]
    assert torch.equal(batch["input_ids"], batch["model_context"]["input_ids"])
    assert torch.equal(batch["labels"], collator(source[0])["labels"])
    assert batch["target_waveform"].shape == (1, 5700)
    data.commit_update(scan_index=0, update_index=0)
    resumed = Qwen3OmniSpeechData(source, collator)
    resumed.load_state_dict(data.state_dict())
    assert resumed.state_dict() == data.state_dict()
    with pytest.raises(ValueError, match="exactly once"):
        resumed.commit_update(scan_index=0, update_index=1)


def test_speech_adapter_rejects_missing_codec_without_advancing(tmp_path, official):
    source = dataset(tmp_path, [row()])
    data = Qwen3OmniSpeechData(source, Qwen3OmniSpeechCollator(*official))
    with pytest.raises(ValueError, match="explicit external codec"):
        data.get_batch(0, "cpu")
    assert data.scanned_batches == 0


def test_manifest_cannot_change_after_indexing(tmp_path):
    source = dataset(tmp_path, [row()])
    source.path.write_text(json.dumps(row(answer="Changed.")) + "\n")
    with pytest.raises(ValueError, match="changed after indexing"):
        source[0]


@pytest.mark.parametrize(
    "state",
    [
        dict(scanned_batches=True, completed_updates=0, last_update_scan=-1),
        dict(scanned_batches=1, completed_updates=0, last_update_scan=0),
        dict(scanned_batches=3, completed_updates=0, last_update_scan=-1),
    ],
)
def test_cursor_rejects_invalid_checkpoint(tmp_path, state):
    source = dataset(tmp_path, [row()])
    data = Qwen3OmniSpeechSFTData(source, IdentityCollator())
    with pytest.raises(ValueError):
        data.load_state_dict(state | dict(fingerprint=source.fingerprint))


def test_checkpoint_preflight_is_pure_and_failed_load_atomic(tmp_path):
    source = dataset(tmp_path, [row(), row("second")])
    data = Qwen3OmniSpeechSFTData(source, IdentityCollator())
    initial = data.state_dict()
    valid = initial | dict(scanned_batches=2, completed_updates=1, last_update_scan=1)
    data.validate_state_dict(valid)
    assert data.state_dict() == initial
    data.load_state_dict(valid)
    assert data.state_dict() == valid
    invalid = valid | dict(last_update_scan=2)
    with pytest.raises(ValueError):
        data.validate_state_dict(invalid)
    assert data.state_dict() == valid
    with pytest.raises(ValueError):
        data.load_state_dict(invalid)
    assert data.state_dict() == valid


def test_unsupervised_known_code_remains_a_causal_teacher_force_input(tmp_path):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTalkerConfig,
    )

    from pithtrain.models.qwen3_omni_speech import TalkerCondition, speech_teacher_force
    from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration

    torch.manual_seed(93)
    torch.set_num_threads(1)
    text = dict(
        vocab_size=41,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        attention_dropout=0.0,
        rope_parameters=dict(
            rope_type="default", rope_theta=1000000, mrope_section=[2, 1, 1], interleaved=True
        ),
    )
    predictor = dict(
        vocab_size=23,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_code_groups=16,
        attention_dropout=0.0,
        rope_parameters=dict(rope_type="default", rope_theta=1000000),
    )
    cfg = Qwen3OmniMoeTalkerConfig(
        text_config=text,
        code_predictor_config=predictor,
        thinker_hidden_size=20,
        num_code_groups=16,
        codec_eos_token_id=31,
    )
    model = Qwen3OmniMoeTalkerForConditionalGeneration(cfg).float().eval()
    condition = TalkerCondition(
        torch.randn(1, 5, 24),
        torch.zeros(1, 5, dtype=torch.long),
        torch.ones(1, 5, dtype=torch.long),
        torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5),
        torch.randn(1, 2, 24),
        torch.randn(1, 1, 24),
    )
    sample = prepared(tmp_path)
    codes = np.arange(48, dtype=np.int64).reshape(3, 16) % 23
    mask = np.ones_like(codes, dtype=bool)
    mask[1, 10] = False
    path = tmp_path / "codec.npz"

    def load(values):
        np.savez(path, codes=values, codebook_mask=mask)
        sample["codec"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load_codec_supervision(sample)

    batch = load(codes)
    assert torch.equal(batch["codes"][0], torch.from_numpy(codes))
    baseline = speech_teacher_force(
        model,
        condition,
        batch["codes"],
        batch["code_lengths"],
        codebook_mask=batch["codebook_mask"],
    )
    unmasked = speech_teacher_force(model, condition, batch["codes"], batch["code_lengths"])
    assert baseline.residual_labels[0, 1, 9] == -100
    assert unmasked.residual_labels[0, 1, 9] == codes[1, 10]
    torch.testing.assert_close(baseline.first_logits, unmasked.first_logits, rtol=0, atol=0)
    torch.testing.assert_close(baseline.residual_logits, unmasked.residual_logits, rtol=0, atol=0)
    changed = codes.copy()
    changed[1, 10] = (changed[1, 10] + 1) % 23
    after_batch = load(changed)
    after = speech_teacher_force(
        model,
        condition,
        after_batch["codes"],
        after_batch["code_lengths"],
        codebook_mask=after_batch["codebook_mask"],
    )
    torch.testing.assert_close(
        baseline.first_logits[:, :2], after.first_logits[:, :2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        baseline.residual_logits[:, 1, :10], after.residual_logits[:, 1, :10], rtol=0, atol=0
    )
    assert (baseline.residual_logits[:, 1, 10] - after.residual_logits[:, 1, 10]).abs().max() > 1e-8
    assert (baseline.first_logits[:, 2] - after.first_logits[:, 2]).abs().max() > 1e-8


@pytest.mark.parametrize("pp_size", [1, 2, 3, 4])
def test_grouped_sft_satisfies_actual_native_schedule_floor(tmp_path, official, pp_size):
    import ast
    from pathlib import Path

    # Execute the real pure schedule helper without importing GPU-only operators.
    pipeline = Path(__file__).resolve().parents[1] / "pithtrain/pipeline/dualpipev.py"
    tree = ast.parse(pipeline.read_text())
    minimum = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "min_chunks"
    )
    scope = {}
    exec(compile(ast.Module(body=[minimum], type_ignores=[]), str(pipeline), "exec"), scope)
    floor = scope["min_chunks"](pp_size)
    group = 2 * pp_size
    source = dataset(tmp_path, [row(str(i), answer=f"Answer {i}.") for i in range(group * 2)])
    collator = Qwen3OmniSpeechCollator(*official)
    data = Qwen3OmniSpeechSFTData(source, collator, microbatches_per_scan=group)
    assert data.total_scans == 2
    batches = data.get_batch(0, "cpu")
    assert len(batches) == group and len(batches) >= floor
    assert [batch.sample_ids for batch in batches] == [(str(i),) for i in range(group)]
    for index, batch in enumerate(batches):
        prepared = collator(source[index])
        assert torch.equal(batch.objective_inputs[0], prepared["labels"])
        assert torch.equal(batch.model_inputs[0], prepared["input_ids"])
        assert batch.model_inputs[0].shape[0] == 1
    assert data.scanned_batches == 1 and data.completed_updates == 0


def test_grouped_scan_resume_preserves_rows_after_skipped_update(tmp_path, official):
    source = dataset(tmp_path, [row(str(i)) for i in range(4)])
    collator = Qwen3OmniSpeechCollator(*official)
    data = Qwen3OmniSpeechSFTData(source, collator, microbatches_per_scan=2)
    assert [batch.sample_ids for batch in data.get_batch(0, "cpu")] == [("0",), ("1",)]
    # Persist a global-zero scan, with no successful optimizer update.
    state = data.state_dict()
    assert state["microbatches_per_scan"] == 2
    resumed = Qwen3OmniSpeechSFTData(source, collator, microbatches_per_scan=2)
    resumed.load_state_dict(state)
    assert [batch.sample_ids for batch in resumed.get_batch(1, "cpu")] == [("2",), ("3",)]
    resumed.commit_update(scan_index=1, update_index=0)
    assert resumed.completed_updates == 1 and resumed.scanned_batches == 2
    saved = resumed.state_dict()
    with pytest.raises(IndexError, match="exhausted"):
        resumed.get_batch(2, "cpu")
    assert resumed.state_dict() == saved
    different_group = Qwen3OmniSpeechSFTData(source, collator, microbatches_per_scan=1)
    before = different_group.state_dict()
    with pytest.raises(ValueError, match="manifest"):
        different_group.load_state_dict(saved)
    assert different_group.state_dict() == before
    with pytest.raises(ValueError, match="counters"):
        resumed.load_state_dict(saved | {"scanned_batches": 3})
    assert resumed.state_dict() == saved


def test_grouped_scan_failed_later_prepare_is_atomic(tmp_path, official):
    source = dataset(tmp_path, [row("first"), row("second")])
    collator = Qwen3OmniSpeechCollator(*official)
    fail_second = True
    calls = []

    def prepare(sample):
        calls.append(sample["id"])
        if fail_second and sample["id"] == "second":
            raise ValueError("second preparation failed")
        return collator(sample)

    data = Qwen3OmniSpeechSFTData(source, IdentityCollator(prepare), microbatches_per_scan=2)
    original = data.state_dict()
    with pytest.raises(ValueError, match="second preparation failed"):
        data.get_batch(0, "cpu")
    assert calls == ["first", "second"] and data.state_dict() == original
    fail_second = False
    assert [batch.sample_ids for batch in data.get_batch(0, "cpu")] == [("first",), ("second",)]
    assert data.scanned_batches == 1


def test_incomplete_group_rejected_before_preparation(tmp_path):
    source = dataset(tmp_path, [row(str(i)) for i in range(3)])
    calls = []
    with pytest.raises(ValueError, match="incomplete final scan"):
        Qwen3OmniSpeechSFTData(source, lambda sample: calls.append(sample), microbatches_per_scan=2)
    assert not calls


@pytest.mark.parametrize("group", [0, -1, True, 2.0])
def test_invalid_group_size_rejected(tmp_path, group):
    with pytest.raises(ValueError, match="microbatches_per_scan"):
        Qwen3OmniSpeechSFTData(dataset(tmp_path, [row()]), None, microbatches_per_scan=group)


def test_group_checkpoint_is_explicit_and_speech_stays_singleton(tmp_path):
    source = dataset(tmp_path, [row(), row("second")])
    data = Qwen3OmniSpeechData(source, IdentityCollator())
    assert data.microbatches_per_scan == 1 and data.total_scans == 2
    saved = data.state_dict()
    assert saved["microbatches_per_scan"] == 1
    for invalid in (
        dict(saved, microbatches_per_scan=True),
        {key: value for key, value in saved.items() if key != "microbatches_per_scan"},
    ):
        with pytest.raises(ValueError, match="manifest"):
            data.load_state_dict(invalid)
        assert data.state_dict() == saved
    with pytest.raises(TypeError):
        Qwen3OmniSpeechData(source, IdentityCollator(), microbatches_per_scan=2)


def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_same(before):
    assert random.getstate() == before[0]
    now = np.random.get_state()
    assert now[0] == before[1][0] and np.array_equal(now[1], before[1][1])
    assert now[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


def _real_media(tmp_path, kind):
    from PIL import Image

    if kind == "image":
        path = tmp_path / "identity.png"
        Image.new("RGB", (64, 64), "red").save(path)
    elif kind == "audio":
        import soundfile as sf

        path = tmp_path / "identity.wav"
        sf.write(path, np.zeros(8000, np.float32), 8000)
    else:
        import av

        path = tmp_path / "identity.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=2)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            for color in ("red", "blue"):
                for packet in stream.encode(
                    av.VideoFrame.from_image(Image.new("RGB", (64, 64), color))
                ):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return path


@pytest.mark.parametrize("kind", ["image", "audio", "video"])
@pytest.mark.parametrize("operation", ["read", "restore"])
def test_media_replacement_rejected_before_processing_or_cursor_rng_mutation(
    tmp_path, official, kind, operation
):
    path = _real_media(tmp_path, kind)
    item = row()
    item["messages"][0]["content"] = [{"type": kind, "path": path.name}]
    source = dataset(tmp_path, [item])
    collator = Qwen3OmniSpeechCollator(*official)
    data = Qwen3OmniSpeechSFTData(source, collator)
    saved, rng = data.state_dict(), _rng_snapshot()
    original = path.read_bytes()
    path.write_bytes(original + b"changed-media-bytes")
    assert dataset(tmp_path, [item]).fingerprint != source.fingerprint
    with pytest.raises(ValueError, match="media identity changed"):
        data.get_batch(0, "cpu") if operation == "read" else data.load_state_dict(saved)
    assert data.state_dict() == saved
    _assert_rng_same(rng)


@pytest.mark.parametrize(
    "field",
    [
        "template",
        "processor",
        "tokenizer",
        "media_limit",
        "model_config",
        "revision",
        "derived_filter",
    ],
)
@pytest.mark.parametrize("operation", ["read", "restore", "fresh_restore"])
def test_preprocessing_identity_changes_fail_closed(tmp_path, official, field, operation):
    processor, config = copy.deepcopy(official)
    collator = Qwen3OmniSpeechCollator(processor, config)
    source = dataset(tmp_path, [row()])
    data = Qwen3OmniSpeechSFTData(source, collator)
    saved = data.state_dict()
    if field == "template":
        processor.chat_template += " "
    elif field == "processor":
        processor.feature_extractor.dither = 0.25
    elif field == "tokenizer":
        processor.tokenizer.add_tokens(["identity_added_token"])
    elif field == "media_limit":
        collator.media.max_pixels += 1
    elif field == "model_config":
        config.audio_config.n_window += 1
    elif field == "derived_filter":
        processor.feature_extractor.mel_filters[0, 0] += 1.0
    else:
        config._commit_hash = "b" * 40
    if operation == "fresh_restore":
        data = Qwen3OmniSpeechSFTData(source, collator)
    before, rng = data.state_dict(), _rng_snapshot()
    with pytest.raises(ValueError, match="identity changed|manifest"):
        data.get_batch(0, "cpu") if operation == "read" else data.load_state_dict(saved)
    assert data.state_dict() == before
    _assert_rng_same(rng)


def test_media_changed_during_decoder_use_rejected_and_rng_rolled_back(
    tmp_path, official, monkeypatch
):
    path = _real_media(tmp_path, "image")
    item = row()
    item["messages"][0]["content"] = [{"type": "image", "path": path.name}]
    source = dataset(tmp_path, [item])
    collator = Qwen3OmniSpeechCollator(*official)
    data = Qwen3OmniSpeechSFTData(source, collator)
    original = collator._process

    def replace_after_decode(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b"replaced-during-processing")
        random.random()
        np.random.rand()
        torch.rand(2)
        return result

    monkeypatch.setattr(collator, "_process", replace_after_decode)
    saved, rng = data.state_dict(), _rng_snapshot()
    with pytest.raises(ValueError, match="media identity changed"):
        data.get_batch(0, "cpu")
    assert data.state_dict() == saved
    _assert_rng_same(rng)


def test_restore_revalidates_nonnext_media_and_codec_archive(tmp_path, official):
    sample = prepared(tmp_path)
    sample.pop("root")
    source = dataset(tmp_path, [row("first"), sample])
    data = Qwen3OmniSpeechSFTData(source, Qwen3OmniSpeechCollator(*official))
    saved, rng = data.state_dict(), _rng_snapshot()
    path = tmp_path / "codec.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="codec media identity changed"):
        data.validate_state_dict(saved)
    assert data.state_dict() == saved
    _assert_rng_same(rng)


def test_group_preflight_checks_every_asset_before_first_collator_call(tmp_path):
    path = _real_media(tmp_path, "image")
    second = row("second")
    second["messages"][0]["content"] = [{"type": "image", "path": path.name}]
    calls = []
    data = Qwen3OmniSpeechSFTData(
        dataset(tmp_path, [row("first"), second]),
        IdentityCollator(lambda sample: calls.append(sample)),
        microbatches_per_scan=2,
    )
    saved = data.state_dict()
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="media identity changed"):
        data.get_batch(0, "cpu")
    assert not calls and data.state_dict() == saved


@pytest.mark.parametrize("collator", [None, lambda sample: sample])
def test_strict_reader_never_accepts_unidentified_callable(tmp_path, collator):
    with pytest.raises(ValueError, match="checkpoint_identity"):
        Qwen3OmniSpeechSFTData(dataset(tmp_path, [row()]), collator)


def test_old_weak_reader_checkpoint_rejected_atomically(tmp_path, official):
    data = Qwen3OmniSpeechSFTData(dataset(tmp_path, [row()]), Qwen3OmniSpeechCollator(*official))
    saved = data.state_dict()
    with pytest.raises(ValueError, match="manifest"):
        data.load_state_dict(
            {key: value for key, value in saved.items() if key != "collator_fingerprint"}
        )
    assert data.state_dict() == saved
