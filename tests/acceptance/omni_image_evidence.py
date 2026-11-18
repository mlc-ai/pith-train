"""CPU-testable evidence checks shared by the unexecuted H100 image gates."""

import copy

import torch


def training_groups(config):
    """Every declared trainable component, including individual DeepStack heads."""
    return (
        "visual.patch_embed.",
        "visual.pos_embed.",
        "visual.merger.",
        *(f"visual.blocks.{i}." for i in range(config.vision_config.depth)),
        *(
            f"visual.merger_list.{i}."
            for i in range(len(config.vision_config.deepstack_visual_indexes))
        ),
        "model.embed_tokens.",
        "model.norm.",
        "lm_head.",
        *(f"model.layers.{i}." for i in range(config.text_config.num_hidden_layers)),
    )


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def snapshot(parameters):
    return {name: local_tensor(p).detach().clone() for name, p in parameters.items()}


def module_update_evidence(parameters, before, groups, reduce=None):
    """Require finite, nonzero gradients AND updates in every component.

    On FSDP/EP/PP, reduce adds scalar evidence across owning ranks. It is only a
    coverage witness, not a gradient normalization formula or parity comparison.
    Nonfinite/missing-gradient errors also enter the reduction so all ranks see
    the same failure before a later collective. Empty local shards are allowed.
    """
    assert parameters and parameters.keys() == before.keys()
    device = local_tensor(next(iter(parameters.values()))).device
    metrics = torch.zeros(len(groups), 4, device=device, dtype=torch.float64)
    for name, parameter in parameters.items():
        owner = [i for i, prefix in enumerate(groups) if name.startswith(prefix)]
        assert len(owner) == 1, f"Missing/ambiguous training component: {name}"
        index = owner[0]
        value = local_tensor(parameter).detach()
        old = before[name]
        grad = None if parameter.grad is None else local_tensor(parameter.grad).detach()
        valid = torch.isfinite(value).all() & torch.isfinite(old).all()
        if grad is None:
            metrics[index, 3] += 1
        else:
            valid = valid & torch.isfinite(grad).all()
            metrics[index, 1] += grad.double().nan_to_num().square().sum()
        metrics[index, 3] += (~valid).to(torch.float64)
        metrics[index, 0] += value.numel()
        metrics[index, 2] += (value.double() - old.double()).nan_to_num().square().sum()
    if reduce is not None:
        reduce(metrics)
    result = {}
    for prefix, (elements, grad_sq, update_sq, errors) in zip(
        groups, metrics.tolist(), strict=True
    ):
        assert errors == 0, f"Missing/nonfinite gradients or values: {prefix}"
        assert elements > 0, f"Uncovered training component: {prefix}"
        assert grad_sq > 0, f"Zero gradient component: {prefix}"
        assert update_sq > 0, f"No optimizer update component: {prefix}"
        result[prefix] = dict(
            elements=elements, gradient_squared_sum=grad_sq, update_squared_sum=update_sq
        )
    return result


class OptimizerStepEvidence:
    """Observe the real optimizer without replacing train_step or its backward."""

    def __init__(self, parameters, groups, reduce=None):
        self.parameters, self.groups, self.reduce = parameters, groups, reduce
        self.steps, self.before = [], None

    def before_step(self, optimizer, args, kwargs):
        assert self.before is None
        self.before = snapshot(self.parameters)

    def after_step(self, optimizer, args, kwargs):
        assert self.before is not None
        self.steps.append(
            module_update_evidence(self.parameters, self.before, self.groups, self.reduce)
        )
        self.before = None


def fixture_image_index(global_index, dp_size):
    """Alternate local images even when global batches are strided across DP."""
    return (global_index // dp_size + global_index % dp_size) % 2


def fixed_image_batches(batches, dp_size, scenario):
    """Shared by CPU audits and the real staged gate; never skips zero-label work."""
    assert len(batches) == 2 and dp_size in (1, 2, 4)
    assert scenario in ("nonuniform", "rank-zero")
    assert scenario != "rank-zero" or dp_size > 1, "rank-zero requires multiple DP ranks"
    result = []
    for index in range(8):
        batch = copy.deepcopy(batches[fixture_image_index(index, dp_size)])
        batch.labels = batch.labels.clone()
        if index == 0 or (scenario == "rank-zero" and index % dp_size == 0):
            batch.labels.fill_(-100)
        elif index % 3 == 0:
            batch.labels[:, ::2] = -100
        result.append(batch)
    validate_target_layout(result, dp_size, scenario)
    return result


def validate_target_layout(batches, dp_size, scenario):
    """Fail before native execution if the advertised rank-zero case is absent."""
    assert len(batches) == 8 and dp_size in (1, 2, 4)
    assert scenario in ("nonuniform", "rank-zero")
    counts = [int((batch.labels != -100).sum()) for batch in batches]
    totals = [sum(counts[rank::dp_size]) for rank in range(dp_size)]
    assert sum(totals) > 0, "Expected globally nonzero targets"
    if scenario == "rank-zero":
        assert dp_size > 1 and totals[0] == 0, "Expected the entire DP rank 0 to have zero targets"
        assert all(value > 0 for value in totals[1:]), "Other DP ranks must retain targets"
    else:
        assert counts[0] == 0 and all(value > 0 for value in totals)
    return dict(
        microbatch_target_counts=counts, rank_target_totals=totals, global_target_count=sum(totals)
    )


def target_partition_evidence(microbatches, layout, dp_rank, global_count):
    """Check actual objective_inputs, including each PP copy, before model.step."""
    counts = [int((batch.objective_inputs[0] != -100).sum()) for batch in microbatches]
    dp_size = len(layout["rank_target_totals"])
    assert counts == layout["microbatch_target_counts"][dp_rank::dp_size], (
        "Wrong DP label partition"
    )
    local_count = sum(counts)
    assert local_count == layout["rank_target_totals"][dp_rank]
    assert int(global_count) == layout["global_target_count"], "PP replicas duplicated targets"
    return dict(
        dp_rank=dp_rank, local_target_count=local_count, global_target_count=int(global_count)
    )
