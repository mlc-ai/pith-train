"""CPU checks of the actual setup_model entry before any GPU/model allocation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


class ModelConstructionReached(Exception):
    pass


def setup_entry(cp_size):
    path = Path(__file__).resolve().parents[1] / "pithtrain/modules/training.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup_model"
    )
    calls = []
    model_config = SimpleNamespace(hidden_size=256, model_type="qwen3_moe")

    def construct(config, phase):
        calls.append((config.max_position_embeddings, phase))
        raise ModelConstructionReached

    # Execute the actual entry function. Stand-ins avoid GPU-only module imports;
    # the constructor sentinel stops before weights, FSDP or kernels are created.
    namespace = dict(
        training=SimpleNamespace(),
        distributed=SimpleNamespace(cp_size=cp_size, cp_group=None),
        nn=SimpleNamespace(Linear=object()),
        FP8Linear=object(),
        FP8GroupedLinear=object(),
        GroupedLinear=object(),
        AutoConfig=SimpleNamespace(from_pretrained=lambda _: model_config),
        model_class_for_config=lambda _: construct,
    )
    exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
    return namespace["setup_model"], calls


@pytest.mark.parametrize(
    "cp_size,length",
    [(1, 1), (1, 2), (1, 69), (1, 128), (1, 2049), (2, 128), (4, 128), (8, 128)],
)
def test_model_entry_accepts_unsharded_and_valid_sharded_lengths(cp_size, length):
    setup_model, calls = setup_entry(cp_size)
    cfg = SimpleNamespace(fp8=False, model="fixture-config", sequence_length=length)
    with pytest.raises(ModelConstructionReached):
        setup_model(cfg, SimpleNamespace())
    assert calls == [(length, 0)]


@pytest.mark.parametrize(
    "cp_size,length",
    [
        (2, 1),
        (2, 2),
        (2, 69),
        (2, 70),
        (2, 127),
        (4, 4),
        (4, 68),
        (4, 127),
        (8, 8),
        (8, 120),
    ],
)
def test_model_entry_rejects_uneven_multi_rank_cp_before_allocation(cp_size, length):
    setup_model, calls = setup_entry(cp_size)
    cfg = SimpleNamespace(fp8=False, model="fixture-config", sequence_length=length)
    with pytest.raises(ValueError, match="must be divisible"):
        setup_model(cfg, SimpleNamespace())
    assert calls == []
