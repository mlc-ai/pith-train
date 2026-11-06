"""Decode a video's audio onto its sampled video timeline, without truncation."""

import math

import av
import numpy as np


def decode_video_audio(path, *, start_seconds, duration_seconds, sampling_rate, max_seconds):
    """Preserve timestamp gaps as silence and reject ambiguous/disjoint streams.

    Video sampling uses its first frame as t=0. Audio is resampled to the
    processor rate and placed by presentation timestamp on that same timeline.
    Leading/trailing silence is explicit; audio content outside the video's
    interval is rejected instead of silently discarded.
    """
    if not 0 < duration_seconds <= max_seconds:
        raise ValueError("Synchronized video duration exceeds the audio duration limit")
    count = math.ceil(duration_seconds * sampling_rate)
    result = np.zeros(count, dtype=np.float32)
    written = np.zeros(count, dtype=np.bool_)
    previous = None
    with av.open(str(path)) as container:
        if len(container.streams.audio) != 1:
            raise ValueError("Synchronized video requires exactly one audio stream")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)

        def place(frame):
            nonlocal previous
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Synchronized audio frame lacks a presentation timestamp")
            begin = round((float(frame.pts * frame.time_base) - start_seconds) * sampling_rate)
            values = frame.to_ndarray().reshape(-1)
            if not np.isfinite(values).all() or not len(values):
                raise ValueError("Empty or nonfinite synchronized audio")
            if previous is not None and begin < previous - 1:
                raise ValueError("Audio timestamps overlap or are not increasing")
            previous = begin + len(values)
            # One resampler rounding sample is allowed at either edge.
            if begin < -1 or begin + len(values) > count + 1:
                raise ValueError("Audio content falls outside the video timeline")
            left, right = max(begin, 0), min(begin + len(values), count)
            if right <= left or written[left:right].any():
                raise ValueError("Audio stream is disjoint or overlaps on the video timeline")
            result[left:right] = values[left - begin : right - begin]
            written[left:right] = True

        for source in container.decode(audio=0):
            for frame in resampler.resample(source):
                place(frame)
        for frame in resampler.resample(None):
            place(frame)
    if not written.any():
        raise ValueError("Synchronized video has no decodable audio")
    return result, dict(
        sampling_rate=sampling_rate,
        samples=count,
        audio_start_seconds=float(np.flatnonzero(written)[0] / sampling_rate),
        audio_end_seconds=float((np.flatnonzero(written)[-1] + 1) / sampling_rate),
        silence_samples=int((~written).sum()),
    )
