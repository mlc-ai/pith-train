"""CPU contracts for the production stable-index helper; no CUDA execution."""

import pytest
import torch

from pithtrain.operators.stable_scatter import stable_group_order


@pytest.mark.parametrize("rows", [0, 1, 64, 129, 513])
@pytest.mark.parametrize("experts", [1, 4, 16])
def test_stable_order_and_padded_positions(rows, experts):
    generator = torch.Generator().manual_seed(rows + experts)
    ids = torch.randint(experts, (rows,), generator=generator)
    counts = torch.bincount(ids, minlength=experts)
    order, starts = stable_group_order(ids, counts)
    expected = sorted(range(rows), key=lambda index: (int(ids[index]), index))
    assert order.tolist() == expected
    assert starts.tolist() == [int(counts[:index].sum()) for index in range(experts)]
    padded = ((counts + 127) // 128) * 128
    padded_starts = padded.cumsum(0) - padded
    positions = torch.arange(rows) - starts[ids[order]] + padded_starts[ids[order]]
    assert positions.unique().numel() == rows
    if rows:
        assert positions.min() >= 0
        assert positions.max() < padded.sum()
    for expert in range(experts):
        selected = order[ids[order] == expert]
        assert selected.tolist() == torch.where(ids == expert)[0].tolist()


def test_fullgraph_cpu_inductor_helper():
    ids = torch.tensor([3, 1, 3, 0, 1, 0, 3], dtype=torch.int64)
    counts = torch.tensor([2, 2, 0, 3], dtype=torch.int64)
    compiled = torch.compile(stable_group_order, backend="inductor", fullgraph=True)
    expected = stable_group_order(ids, counts)
    actual = compiled(ids, counts)
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 1, 32, 129])
@pytest.mark.parametrize("ep_size,experts,top_k", [(1, 4, 2), (2, 4, 2), (4, 16, 4)])
def test_stable_ep_metadata_reconstructs_original_routes(rows, ep_size, experts, top_k):
    from pithtrain.operators.stable_scatter import stable_dedup_prepare_dispatch

    generator = torch.Generator().manual_seed(91 + rows)
    ids = (
        torch.stack([torch.randperm(experts, generator=generator)[:top_k] for _ in range(rows)])
        if rows
        else torch.empty((0, top_k), dtype=torch.int64)
    )
    per_rank = experts // ep_size
    result = stable_dedup_prepare_dispatch(ids, experts, ep_size, per_rank)
    tokens, dedup, dispatch, slots, expand, meta = result
    expected_slots = sorted(range(rows * top_k), key=lambda slot: (int(ids.view(-1)[slot]), slot))
    assert slots.tolist() == expected_slots
    expected_dispatch = []
    dedup_start = 0
    for rank in range(ep_size):
        rank_tokens = [
            row
            for row in range(rows)
            if any(int(expert) // per_rank == rank for expert in ids[row])
        ]
        expected_dispatch.extend(rank_tokens)
        assert dedup[rank].item() == len(rank_tokens)
        rank_slots = [
            slot for slot in expected_slots if int(ids.view(-1)[slot]) // per_rank == rank
        ]
        assert tokens[rank].item() == len(rank_slots)
        for position, slot in enumerate(expected_slots):
            if slot in rank_slots:
                assert dispatch[dedup_start + expand[position]].item() == slot // top_k
        dedup_start += len(rank_tokens)
    assert dispatch[:dedup_start].tolist() == expected_dispatch
    assert dispatch.numel() == rows * ep_size
    counts = [sum(int(expert) == e for expert in ids.view(-1)) for e in range(experts)]
    expected_meta = []
    for rank in range(ep_size):
        expected_meta.extend(counts[rank * per_rank : (rank + 1) * per_rank])
        expected_meta.append(dedup[rank].item())
    assert meta.tolist() == expected_meta
    for again, original in zip(
        stable_dedup_prepare_dispatch(ids, experts, ep_size, per_rank), result, strict=True
    ):
        assert torch.equal(again, original)


def test_omni_production_routing_requests_stable_layout():
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "pithtrain/models/qwen3_omni_moe.py"
    tree = ast.parse(source.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"prepare_dispatch", "scatter_for_grouped_gemm"}
    ]
    assert len(calls) == 3
    for call in calls:
        assert any(
            keyword.arg == "deterministic"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
