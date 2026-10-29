"""Real mux/decode/processor tests for synchronized video/audio timelines."""

from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from pithtrain.models.qwen3_omni_multimodal_inputs import multimodal_position_ids
from pithtrain.modules.qwen3_omni_av import decode_video_audio
from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator
from tests import test_qwen3_omni_data as data_fixtures
from tests.test_qwen3_omni_multimodal import hf_positions

processor_config = data_fixtures.processor_config


def write_av(path, *, audio_start=0, audio_samples=16000):
    with av.open(str(path), "w", format="matroska") as container:
        video = container.add_stream("ffv1", rate=4)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("pcm_s16le", rate=16000)
        audio.layout = "mono"
        for index in range(4):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), index * 50, dtype=np.uint8), format="rgb24"
            )
            frame.pts = index
            frame.time_base = Fraction(1, 4)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        values = (np.sin(np.arange(audio_samples) * (2 * np.pi * 440 / 16000)) * 5000).astype(
            np.int16
        )
        frame = av.AudioFrame.from_ndarray(values[None], format="s16", layout="mono")
        frame.sample_rate = 16000
        frame.pts = round(audio_start * 16000)
        frame.time_base = Fraction(1, 16000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def test_audio_timestamps_align_to_video_timeline_with_explicit_silence(tmp_path):
    path = tmp_path / "offset.mkv"
    write_av(path, audio_start=0.125, audio_samples=12000)
    waveform, details = decode_video_audio(
        path, start_seconds=0, duration_seconds=1, sampling_rate=16000, max_seconds=2
    )
    assert len(waveform) == 16000
    assert details["audio_start_seconds"] == 0.125
    assert details["audio_end_seconds"] == 0.875
    assert details["silence_samples"] == 4000
    assert (waveform[:2000] == 0).all() and (waveform[-2000:] == 0).all()
    assert np.abs(waveform[2000:-2000]).sum() > 0


def test_audio_outside_video_is_rejected_instead_of_truncated(tmp_path):
    path = tmp_path / "outside.mkv"
    write_av(path, audio_start=0.5)
    with pytest.raises(ValueError, match="outside"):
        decode_video_audio(
            path, start_seconds=0, duration_seconds=1, sampling_rate=16000, max_seconds=2
        )


def test_real_synchronized_processor_tokens_positions_and_tensor_context(
    tmp_path, processor_config
):
    processor, cfg = processor_config
    path = tmp_path / "synced.mkv"
    write_av(path)
    sample = dict(
        id="av", text="The frames brighten with a tone.", media=[dict(type="video", path=path)]
    )
    with pytest.raises(ValueError, match="audio track"):
        Qwen3OmniCollator(processor, cfg)([sample])
    batch = Qwen3OmniCollator(
        processor,
        cfg,
        use_audio_in_video=True,
        max_length=512,
        min_pixels=4096,
        max_pixels=4096,
        video_fps=4,
    )([sample])
    inputs = batch.model_inputs
    assert inputs["use_audio_in_video"].dtype == torch.bool
    assert all(isinstance(value, torch.Tensor) for value in inputs.values())
    assert (inputs["input_ids"] == cfg.video_token_id).any()
    assert (inputs["input_ids"] == cfg.audio_token_id).any()
    actual = multimodal_position_ids(inputs["input_ids"], inputs, cfg)
    torch.testing.assert_close(actual, hf_positions(cfg, inputs), rtol=0, atol=0)
    assert batch.media_info[0][0]["audio"]["samples"] == 16000
    assert batch.media_info[0][0]["timestamps"] == [0.0, 0.25, 0.5, 0.75]
    targets = batch.labels[batch.labels != -100].tolist()
    expected = processor.tokenizer.encode(
        sample["text"] + processor.tokenizer.eos_token, add_special_tokens=False
    )
    assert targets == expected
