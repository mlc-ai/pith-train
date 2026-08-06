from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd

from pithtrain.tasks.tokenize_corpus import read_file


@pytest.mark.parametrize("suffix", [".jsonl.zst", ".jsonl.zstd"])
def test_read_file_supports_compressed_jsonl(tmp_path: Path, suffix: str):
    path = tmp_path / f"data{suffix}"
    with zstd.open(path, "wt") as f:
        f.write('{"text": "alpha"}\n{"text": "beta"}\n')

    assert list(read_file(path)) == ["alpha", "beta"]


def test_read_file_supports_jsonl(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "alpha"}\n{"text": "beta"}\n')

    assert list(read_file(path)) == ["alpha", "beta"]


def test_read_file_supports_parquet_text_column(tmp_path: Path):
    path = tmp_path / "data.parquet"
    table = pa.table({"text": ["alpha", "beta"], "metadata": [{"x": 1}, {"x": 2}]})
    pq.write_table(table, path)

    assert list(read_file(path)) == ["alpha", "beta"]


def test_read_file_rejects_parquet_without_text_column(tmp_path: Path):
    path = tmp_path / "data.parquet"
    table = pa.table({"content": ["alpha"]})
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="text column"):
        list(read_file(path))
