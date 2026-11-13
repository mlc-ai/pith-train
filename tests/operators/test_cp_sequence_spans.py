"""CPU coverage of the token positions shared by loaders and model RoPE."""

import pytest

from pithtrain.operators.cp_sequence import zigzag_spans


@pytest.mark.parametrize("length", [0, 1, 2, 69, 128, 1063, 2048])
def test_unsharded_positions_keep_every_token(length):
    front, back = zigzag_spans(0, 1, length)
    assert [*front, *back] == list(range(length))


@pytest.mark.parametrize("cp_size", [2, 3, 4, 8])
def test_sharded_positions_keep_zigzag_pairing(cp_size):
    # Label blocks independently, as a caller would shard the global sequence.
    blocks = [list(range(start, start + 32)) for start in range(0, 64 * cp_size, 32)]
    positions = []
    for rank in range(cp_size):
        front, back = zigzag_spans(rank, cp_size, 64 * cp_size)
        local = [*front, *back]
        assert local == blocks[rank] + blocks[-rank - 1]
        positions.extend(local)
    assert sorted(positions) == list(range(64 * cp_size))
