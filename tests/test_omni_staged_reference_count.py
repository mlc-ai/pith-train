"""Exercise single-modality and mixed HF blocks against the production count producer."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pithtrain.modules import training_data
from pithtrain.modules.microbatch import Microbatch
from tests.acceptance.omni_image_evidence import (
    fixed_image_batches,
    target_partition_evidence,
    validate_target_layout,
)
from tests.acceptance.omni_reference_precision import FP32GradientReference


class TokenReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 5, dtype=torch.bfloat16)

    def forward(self, inputs):
        return self.projection(inputs["features"])


def staged_reference_block(driver):
    """Use the actual caller without constructing native CUDA-only training."""
    path = Path(__file__).parent / "acceptance" / driver
    entrypoint = "run" if driver == "omni_image_staged.py" else "staged"
    module = ast.parse(path.read_text())
    run = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint
    )
    constructors = [
        node
        for node in run.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "FP32GradientReference"
    ]
    assert len(constructors) == 1
    starts = [
        index
        for index, node in enumerate(run.body)
        if isinstance(node, ast.Assign) and ast.unparse(node) == "hf_loss = 0"
    ]
    assert len(starts) == 1
    start = starts[0]
    end = next(
        index
        for index in range(start + 1, len(run.body))
        if isinstance(run.body[index], ast.Assign)
        and isinstance(run.body[index].value, ast.Call)
        and ast.unparse(run.body[index].value.func) == "training.model.step"
    )
    return compile(
        ast.Module(body=[constructors[0], *run.body[start:end]], type_ignores=[]), str(path), "exec"
    )


@pytest.mark.parametrize(
    ("dp_size", "scenario"),
    [(1, "nonuniform"), (2, "nonuniform"), (4, "nonuniform"), (2, "rank-zero"), (4, "rank-zero")],
)
def test_actual_staged_caller_accepts_production_float_count(dp_size, scenario, monkeypatch):
    check_actual_staged_caller("omni_image_staged.py", dp_size, scenario, monkeypatch)


@pytest.mark.parametrize(
    ("dp_size", "scenario"),
    [(1, "nonuniform"), (2, "nonuniform"), (4, "nonuniform"), (2, "rank-zero"), (4, "rank-zero")],
)
def test_actual_mixed_caller_accepts_production_float_count(dp_size, scenario, monkeypatch):
    check_actual_staged_caller("omni_multimodal_training.py", dp_size, scenario, monkeypatch)


def check_actual_staged_caller(driver, dp_size, scenario, monkeypatch):
    torch.manual_seed(42)
    source = [
        SimpleNamespace(
            model_inputs={"features": torch.randn(1, 4, 3, dtype=torch.bfloat16)},
            labels=torch.tensor([[0, 1, 2, -100]]),
        ),
        SimpleNamespace(
            model_inputs={"features": torch.randn(1, 4, 3, dtype=torch.bfloat16)},
            labels=torch.tensor([[1, 2, 3, 4]]),
        ),
    ]
    batches = fixed_image_batches(source, dp_size, scenario)
    layout = validate_target_layout(batches, dp_size, scenario)
    expected_count = sum(int((batch.labels != -100).sum()) for batch in batches)
    assert type(layout["global_target_count"]) is int
    assert expected_count == layout["global_target_count"] > 0
    assert layout["microbatch_target_counts"][0] == 0
    counts = []
    monkeypatch.setattr(training_data.dist, "is_initialized", lambda: True)
    for rank in range(dp_size):
        local = [
            Microbatch(model_inputs=(), cu_seqlens=None, objective_inputs=(batch.labels,))
            for batch in batches[rank::dp_size]
        ]
        local_count = layout["rank_target_totals"][rank]
        expected_group = object()

        def reduce_counts(value, *, group):
            assert group is expected_group
            assert value.ndim == 0 and value.dtype == torch.int64
            assert value.item() == local_count
            value.add_(expected_count - local_count)

        monkeypatch.setattr(training_data.dist, "all_reduce", reduce_counts)
        count = training_data.global_target_count(local, expected_group)
        assert count.shape == torch.Size([]) and count.dtype == torch.float32
        assert count.item() == expected_count
        partition = target_partition_evidence(local, layout, rank, count)
        assert partition["local_target_count"] == local_count
        counts.append(count)
    if scenario == "rank-zero":
        assert layout["rank_target_totals"][0] == 0 and counts[0].item() > 0

    module = TokenReference()
    independent = copy.deepcopy(module)
    # Preserve the strict positive-integer API. The old actual caller fails here.
    with pytest.raises(ValueError, match="positive integer scalar"):
        FP32GradientReference(copy.deepcopy(module)).normalize_gradients_(counts[0])
    oracle = {
        name: torch.zeros_like(parameter, dtype=torch.float32)
        for name, parameter in independent.named_parameters()
    }
    expected_loss = 0
    for batch in batches:
        logits = independent(batch.model_inputs)
        loss = F.cross_entropy(
            logits.float().flatten(0, 1), batch.labels.flatten(), reduction="sum"
        )
        expected_loss += loss.detach()
        loss.backward()
        for name, parameter in independent.named_parameters():
            assert parameter.grad is not None
            oracle[name].add_(parameter.grad.float())
        independent.zero_grad(set_to_none=True)
    namespace = dict(
        hf_reference=module,
        FP32GradientReference=FP32GradientReference,
        global_batches=batches,
        layout=layout,
        count=counts[0],
        distributed=SimpleNamespace(device=torch.device("cpu")),
        F=F,
    )
    exec(staged_reference_block(driver), namespace)
    reference = namespace["reference"]
    assert type(reference) is FP32GradientReference and reference.normalized
    assert reference.master_parameters.keys() == oracle.keys()
    for name, parameter in reference.master_parameters.items():
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        torch.testing.assert_close(parameter.grad, oracle[name] / expected_count, rtol=0, atol=0)
    torch.testing.assert_close(namespace["hf_loss"], expected_loss / expected_count, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_production_global_zero_count_remains_rejected(monkeypatch):
    monkeypatch.setattr(training_data.dist, "is_initialized", lambda: False)
    zero = Microbatch(
        model_inputs=(), cu_seqlens=None, objective_inputs=(torch.full((1, 3), -100),)
    )
    with pytest.raises(ValueError, match="no valid next-token targets"):
        training_data.global_target_count([zero])
