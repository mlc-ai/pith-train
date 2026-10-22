"""Actual CPU tests of evidence gates; not native decoder/GPU execution."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.training_data import global_target_count
from tests.acceptance.omni_image_evidence import (
    OptimizerStepEvidence,
    fixture_image_index,
    module_update_evidence,
    snapshot,
)


def parameters():
    values = {name: torch.nn.Parameter(torch.ones(2)) for name in ("left.weight", "right.weight")}
    for value in values.values():
        value.grad = torch.ones_like(value)
    return values


def test_each_projection_requires_its_own_gradient_and_update():
    values = parameters()
    before = snapshot(values)
    torch.optim.AdamW(values.values(), lr=0.1, weight_decay=0).step()
    evidence = module_update_evidence(values, before, ("left.", "right."))
    assert evidence.keys() == {"left.", "right."}
    assert all(
        row["gradient_squared_sum"] > 0 and row["update_squared_sum"] > 0
        for row in evidence.values()
    )


@pytest.mark.parametrize("frozen", ["left.weight", "right.weight"])
def test_one_updated_projection_cannot_hide_a_frozen_projection(frozen):
    values = parameters()
    before = snapshot(values)
    with torch.no_grad():
        for name, value in values.items():
            if name != frozen:
                value.add_(1)
    with pytest.raises(AssertionError, match="No optimizer update"):
        module_update_evidence(values, before, ("left.", "right."))


@pytest.mark.parametrize("bad", ["missing", "zero", "nan"])
def test_parameter_update_cannot_hide_a_bad_gradient(bad):
    values = parameters()
    before = snapshot(values)
    with torch.no_grad():
        for value in values.values():
            value.add_(1)
    values["right.weight"].grad = {
        "missing": None,
        "zero": torch.zeros(2),
        "nan": torch.full((2,), float("nan")),
    }[bad]
    with pytest.raises(AssertionError, match="gradient"):
        module_update_evidence(values, before, ("left.", "right."))


def test_missing_component_is_not_success():
    values = parameters()
    before = snapshot(values)
    torch.optim.AdamW(values.values(), lr=0.1, weight_decay=0).step()
    with pytest.raises(AssertionError, match="Uncovered"):
        module_update_evidence(values, before, ("left.", "right.", "missing."))


def test_observer_records_all_four_actual_optimizer_steps():
    values = parameters()
    opt = torch.optim.AdamW(values.values(), lr=0.1, weight_decay=0)
    observer = OptimizerStepEvidence(values, ("left.", "right."))
    pre = opt.register_step_pre_hook(observer.before_step)
    post = opt.register_step_post_hook(observer.after_step)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        sum(value.square().sum() for value in values.values()).backward()
        opt.step()
    pre.remove()
    post.remove()
    assert len(observer.steps) == 4 and observer.before is None


def _zero_target_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    micro = Microbatch(
        model_inputs=(torch.ones(1, 3, dtype=torch.long),),
        cu_seqlens=None,
        objective_inputs=(torch.full((1, 3), -100),),
    )
    try:
        global_target_count([micro], groups[rank // 2])
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Global all-zero labels were accepted")
    errors = [None] * 4
    dist.all_gather_object(errors, message)
    assert len(set(errors)) == 1 and "no valid next-token targets" in message
    Path(output, f"rank-{rank}.json").write_text(json.dumps(dict(status="PASSED", error=message)))
    dist.destroy_process_group()


def test_global_all_zero_targets_reject_identically_on_four_cpu_ranks(tmp_path):
    mp.spawn(_zero_target_worker, args=(tmp_path / "rendezvous", tmp_path), nprocs=4, join=True)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)]
    assert all(report["status"] == "PASSED" for report in reports)


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_every_dp_rank_receives_different_consecutive_images(dp_size):
    selected = [fixture_image_index(index, dp_size) for index in range(8)]
    assert selected.count(0) == selected.count(1) == 4
    for rank in range(dp_size):
        local = selected[rank::dp_size]
        assert len(local) >= 2 and all(a != b for a, b in zip(local, local[1:]))
