"""Offline examples of the existing text pretraining data format and label shift."""

import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from pithtrain.modules.dataset import MemmapDataset
from pithtrain.tasks.tokenize_corpus import Worker, Writer, read_file


def test_text_to_next_token_samples(tmp_path):
    # A local toy tokenizer keeps the real Worker/Writer path offline and the IDs readable.
    vocab = {"[UNK]": 0, "[EOS]": 1, "alpha": 2, "beta": 3, "gamma": 4, "delta": 5, "epsilon": 6}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)

    source = tmp_path / "text.jsonl"
    documents = ["alpha beta", "gamma delta epsilon"]
    source.write_text("".join(json.dumps({"text": text}) + "\n" for text in documents))
    Worker(str(tokenizer_path))
    path = tmp_path / "text.bin"
    writer = Writer(path)
    for text in read_file(source):
        tokens, _ = Worker.encode(text)
        writer.append(tokens)
    writer.flush()

    # The .bin contains two .npy arrays: a token stream, then document end offsets.
    with path.open("rb") as stream:
        stored_tokens, splits = np.load(stream), np.load(stream)
    assert stored_tokens.tolist() == [2, 3, 1, 4, 5, 6, 1]
    assert stored_tokens.dtype == np.uint8
    assert splits.tolist() == [3, 7]

    dataset = MemmapDataset(path, sequence_length=3)
    assert len(dataset) == 2
    inputs, labels = dataset[0]
    assert inputs.tolist() == [2, 3, 1]
    # EOS -> gamma is a normal target: dense pretraining does not mask document boundaries.
    assert labels.tolist() == [3, 1, 4]
    inputs, labels = dataset[1]
    assert inputs.tolist() == [4, 5, 6]
    assert labels.tolist() == [5, 6, 1]


def test_chunk_labels_follow_the_original_stream(tmp_path):
    # IDs above uint16's range must survive storage (e.g. a full Omni tokenizer vocabulary).
    path = tmp_path / "wide-vocab.bin"
    writer = Writer(path)
    writer.append(
        np.array([1001, 151645, 1002, 1003, 1004, 1005, 151645, 1006, 152063], dtype=np.uint32)
    )
    writer.flush()
    dataset = MemmapDataset(path, sequence_length=8)
    assert len(dataset) == 1
    inputs, labels = dataset[0]
    assert inputs.tolist() == [1001, 151645, 1002, 1003, 1004, 1005, 151645, 1006]
    assert labels.tolist() == [151645, 1002, 1003, 1004, 1005, 151645, 1006, 152063]

    # CP=2 rank 0 reads the first and last two positions. The target after 151645
    # in the front chunk is 1002 in the original stream, not the back chunk's 151645.
    front_inputs, front_labels = dataset.get_chunk(0, 0, 2)
    back_inputs, back_labels = dataset.get_chunk(0, 6, 2)
    assert torch.cat([front_inputs, back_inputs]).tolist() == [1001, 151645, 151645, 1006]
    assert torch.cat([front_labels, back_labels]).tolist() == [151645, 1002, 1006, 152063]


@pytest.mark.parametrize("n_tokens,expected_samples", [(4, 0), (5, 1), (8, 1), (9, 2)])
def test_samples_require_one_extra_target_token(tmp_path, n_tokens, expected_samples):
    path = tmp_path / "tail.bin"
    writer = Writer(path)
    writer.append(np.arange(n_tokens, dtype=np.uint32))
    writer.flush()
    dataset = MemmapDataset(path, sequence_length=4)
    # No padding: each retained sequence needs its final next-token label in the shard.
    assert len(dataset) == expected_samples
    if expected_samples:
        inputs, labels = dataset[expected_samples - 1]
        assert inputs.shape == labels.shape == (4,)
        assert labels[-1].item() == expected_samples * 4
