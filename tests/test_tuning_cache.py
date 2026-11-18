"""
The launch-config tuning cache (CPU only): jobs writing it at once keep every entry and never leave
a partial file, and a corrupt file reads as empty. The module is loaded from its file, since
importing pithtrain.kernels probes the GPU; it imports nothing from pithtrain.
"""

import importlib.util
import json
import multiprocessing
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "tuning", Path(__file__).resolve().parents[1] / "pithtrain/kernels/tuning.py"
)
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)
WRITERS, ROUNDS = 4, 25


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / "GPU-0123456789abcdef.json"
    monkeypatch.setattr(tuning, "cache_file", lambda: path)
    return path


def write_entries(writer: int) -> None:
    for step in range(ROUNDS):
        tuning.write_cache([("family", "fp", (writer, step), 3, writer % 2)])


def test_concurrent_writers_keep_every_entry(cache):
    tuning.write_cache([("family", "fp", ("before",), 5, 0)])
    context = multiprocessing.get_context("fork")
    writers = [context.Process(target=write_entries, args=(writer,)) for writer in range(WRITERS)]
    for process in writers:
        process.start()
    while any(process.is_alive() for process in writers):
        json.loads(cache.read_text())  # never partial
    for process in writers:
        process.join()
        assert process.exitcode == 0
    entries = tuning.read_cache()
    expected = {(writer, step) for writer in range(WRITERS) for step in range(ROUNDS)}
    assert {key for _, _, key, _, _ in entries} == expected | {("before",)}
    assert len(entries) == len(expected) + 1
    assert not list(cache.parent.glob("*.tmp"))


def test_write_replaces_same_entry(cache):
    tuning.write_cache([("family", "fp", (1, 2), 3, 0), ("family", "fp", (1, 2), 4, 0)])
    tuning.write_cache([("family", "fp", (1, 2), 3, 1)])
    assert sorted(tuning.read_cache()) == [
        ("family", "fp", (1, 2), 3, 1),
        ("family", "fp", (1, 2), 4, 0),
    ]


@pytest.mark.parametrize("text", ['[["family", "fp", [1, 2], 3', "", "null", "[1, 2]", '{"a": 1}'])
def test_corrupt_file_reads_empty(cache, text):
    cache.write_text(text)
    assert tuning.read_cache() == []
    tuning.write_cache([("family", "fp", (1, 2), 3, 0)])
    assert tuning.read_cache() == [("family", "fp", (1, 2), 3, 0)]
