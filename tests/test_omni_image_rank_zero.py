"""Rank-zero fixture and CPU collective evidence, not native GPU training."""

import ast
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.training_data import global_target_count
from tests.acceptance.omni_image_evidence import (
    fixed_image_batches,
    target_partition_evidence,
    validate_target_layout,
)


def source_batches():
    result = []
    for image, count in enumerate((10, 12)):
        labels = torch.full((1, 20), -100)
        labels[:, :count] = 1
        result.append(SimpleNamespace(labels=labels, sample_ids=(f"image-{image}",)))
    return result


def microbatches(batches):
    return [
        Microbatch(
            model_inputs=(torch.ones_like(batch.labels),),
            cu_seqlens=None,
            objective_inputs=(batch.labels,),
        )
        for batch in batches
    ]


@pytest.mark.parametrize("dp_size", [2, 4])
def test_entire_rank_is_zero_while_other_ranks_keep_alternating_images(dp_size):
    original = source_batches()
    batches = fixed_image_batches(original, dp_size, "rank-zero")
    layout = validate_target_layout(batches, dp_size, "rank-zero")
    assert layout["rank_target_totals"][0] == 0
    for rank in range(dp_size):
        local = batches[rank::dp_size]
        assert all(a.sample_ids != b.sample_ids for a, b in zip(local, local[1:]))
        if rank == 0:
            assert all((b.labels == -100).all() for b in local)
        else:
            assert any((b.labels != -100).any() for b in local)
    assert [int((b.labels != -100).sum()) for b in original] == [10, 12]
    if dp_size == 2:
        assert layout["microbatch_target_counts"] == [0, 12, 0, 5, 0, 12, 0, 10]
        assert layout["rank_target_totals"] == [0, 39]


def test_old_single_microbatch_mask_cannot_pass_as_rank_zero():
    old = fixed_image_batches(source_batches(), 2, "nonuniform")
    assert validate_target_layout(old, 2, "nonuniform")["rank_target_totals"] == [28, 39]
    with pytest.raises(AssertionError, match="entire DP rank"):
        validate_target_layout(old, 2, "rank-zero")


@pytest.mark.parametrize(
    "bad", ["one_leaked_label", "globally_zero", "wrong_partition", "pp_duplicate"]
)
def test_faults_cannot_be_reported_as_rank_zero_acceptance(bad):
    batches = fixed_image_batches(source_batches(), 2, "rank-zero")
    layout = validate_target_layout(batches, 2, "rank-zero")
    with pytest.raises(AssertionError):
        if bad == "one_leaked_label":
            batches[6].labels[0, 0] = 1
            validate_target_layout(batches, 2, "rank-zero")
        elif bad == "globally_zero":
            for batch in batches:
                batch.labels.fill_(-100)
            validate_target_layout(batches, 2, "rank-zero")
        else:
            local = batches[1::2] if bad == "wrong_partition" else batches[::2]
            count = layout["global_target_count"] * (2 if bad == "pp_duplicate" else 1)
            target_partition_evidence(microbatches(local), layout, 0, count)


def test_rank_zero_requires_another_dp_rank():
    with pytest.raises(AssertionError, match="multiple DP"):
        fixed_image_batches(source_batches(), 1, "rank-zero")


def _rank_zero_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    batches = fixed_image_batches(source_batches(), 2, "rank-zero")
    layout = validate_target_layout(batches, 2, "rank-zero")
    local = microbatches(batches[rank % 2 :: 2])
    count = global_target_count(local, groups[rank // 2])
    evidence = target_partition_evidence(local, layout, rank % 2, count)
    evidence["pp_rank"] = rank // 2
    all_evidence = [None] * 4
    dist.all_gather_object(all_evidence, evidence)
    assert [row["local_target_count"] for row in all_evidence] == [0, 39, 0, 39]
    assert [row["global_target_count"] for row in all_evidence] == [39] * 4
    Path(output, f"rank-{rank}.json").write_text(json.dumps(evidence))
    dist.destroy_process_group()


def test_rank_zero_actual_target_count_on_four_cpu_pp_dp_ranks(tmp_path):
    mp.spawn(_rank_zero_worker, args=(tmp_path / "rendezvous", tmp_path), nprocs=4, join=True)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)]
    assert [row["local_target_count"] for row in reports] == [0, 39, 0, 39]
    assert all(row["global_target_count"] == 39 for row in reports)


def test_rank_zero_uses_the_same_native_ce_gradient_and_optimizer_gate():
    # Structural evidence only: zero-label batches are not filtered and there is
    # one shared actual model.step/objective/gradient/update path for both cases.
    root = Path(__file__).resolve().parent / "acceptance"
    tree = ast.parse((root / "omni_image_staged.py").read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    calls = [n for n in ast.walk(run) if isinstance(n, ast.Call)]
    fixed = next(n for n in calls if ast.unparse(n.func) == "fixed_image_batches")
    assert ast.unparse(fixed.args[-1]) == "args.scenario"
    steps = [n for n in calls if ast.unparse(n.func) == "training.model.step"]
    assert len(steps) == 1 and ast.unparse(steps[0].args[1]) == "pretrain_lm.objective"
    for name in (
        "target_partition_evidence",
        "expected_shard",
        "optimizer.step",
        "module_update_evidence",
        "global_target_count",
    ):
        assert any(ast.unparse(n.func) == name for n in calls)
    runner = (root / "run_week4_h100.sh").read_text()
    for profile in ("2 1 1 rank-zero", "2 1 2 rank-zero", "4 2 2 rank-zero"):
        assert profile in runner
    assert '--scenario "$scenario"' in runner
