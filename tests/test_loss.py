"""Compare stage-wide token normalization with one concatenated-batch objective.

CPU collectives stand in for DP2 x CP2; the real multi-process/GPU tests exercise
communication. This oracle covers the dense equivalence and unequal masked counts.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.modules import loss
from pithtrain.modules.microbatch import Microbatch


@pytest.mark.parametrize("masked", [False, True])
def test_stage_normalization_matches_global_objective(monkeypatch, masked):
    inputs = torch.arange(32, dtype=torch.float64).reshape(4, 4, 2) / 32
    targets = torch.arange(16).reshape(4, 4) % 3
    if masked:
        targets[0] = -100
        targets[1, :2] = -100
        targets[2, 0] = -100
    weight = (torch.arange(6, dtype=torch.float64).reshape(3, 2) / 10).requires_grad_()
    sums = [
        F.cross_entropy(F.linear(x, weight), y, reduction="sum") for x, y in zip(inputs, targets)
    ]
    reference = F.cross_entropy(F.linear(inputs.flatten(0, 1), weight), targets.flatten())
    expected_grad = torch.autograd.grad(reference, weight, retain_graph=True)[0]
    summed_grad = torch.autograd.grad(sum(sums), weight)[0]
    stage_group = object()
    monkeypatch.setattr(loss.dist, "is_initialized", lambda: True)

    def all_reduce(value, *, group):
        assert group is stage_group, "PP replicas must not participate in this reduction"
        reduced = targets.ne(-100).sum() if value.dtype == torch.int64 else sum(sums).detach()
        value.copy_(reduced)

    monkeypatch.setattr(loss.dist, "all_reduce", all_reduce)
    for rank, labels in enumerate(targets):
        batch = Microbatch(
            model_inputs=(inputs[rank],), objective_inputs=(labels,), cu_seqlens=None
        )
        count = loss.global_target_count([batch], stage_group)
        if not masked:
            assert count == labels.numel() * 2 * 2  # old local_count * DP * CP
        torch.testing.assert_close(summed_grad / count, expected_grad)
        mean = loss.global_loss_mean(sums[rank], count, stage_group)
        torch.testing.assert_close(mean.double(), reference.detach(), rtol=1e-6, atol=1e-6)


def test_no_global_targets_fails_before_optimization():
    labels = torch.full((1, 4), -100)
    batch = Microbatch(model_inputs=(labels,), objective_inputs=(labels,), cu_seqlens=None)
    with pytest.raises(ValueError, match="no valid next-token targets"):
        loss.global_target_count([batch])
