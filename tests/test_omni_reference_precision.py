"""CPU contracts for staged HF reference precision, independent of native kernels."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.acceptance.omni_reference_precision import FP32GradientReference


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1, dtype=torch.bfloat16))
        self.register_buffer("position", torch.tensor(1.0001, dtype=torch.float32))

    def forward(self, value):
        assert self.weight.dtype == torch.bfloat16
        assert self.position.dtype == torch.float32
        return self.weight * value


def test_fp32_accumulation_keeps_small_microbatch_contributions():
    original = Scalar()
    wrapped = FP32GradientReference(original)
    for contribution in (256, 1, -256):
        wrapped(torch.tensor(contribution, dtype=torch.bfloat16)).backward()
        original(torch.tensor(contribution, dtype=torch.bfloat16)).backward()
    assert original.weight.grad.item() == 0  # The previous BF16 accumulation loses the 1.
    assert wrapped.master_parameters["weight"].grad.item() == 1
    assert wrapped.master_parameters["weight"].grad.dtype == torch.float32
    wrapped.normalize_gradients_(3)
    torch.testing.assert_close(
        wrapped.master_parameters["weight"].grad, torch.tensor(1 / 3), rtol=0, atol=0
    )


def test_global_normalization_with_nonuniform_and_zero_target_microbatches():
    torch.manual_seed(19)
    original = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 5)).to(torch.bfloat16)
    reference = FP32GradientReference(original)
    independent = copy.deepcopy(original)
    oracle = {
        name: torch.zeros_like(p, dtype=torch.float32) for name, p in independent.named_parameters()
    }
    counts, loss_sums = [], []
    for valid in (0, 1, 3, 2):
        features = torch.randn(3, 3, dtype=torch.bfloat16)
        labels = torch.tensor([0, 1, 2])
        labels[valid:] = -100
        counts.append(int((labels != -100).sum()))
        actual, expected = reference(features), independent(features)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss = F.cross_entropy(actual.float(), labels, reduction="sum")
        loss_sums.append(loss.detach())
        loss.backward()
        F.cross_entropy(expected.float(), labels, reduction="sum").backward()
        for name, parameter in independent.named_parameters():
            assert parameter.grad is not None
            oracle[name].add_(parameter.grad.float())
        independent.zero_grad(set_to_none=True)
    assert counts == [0, 1, 3, 2] and loss_sums[0] == 0
    reference.normalize_gradients_(torch.tensor(sum(counts), dtype=torch.int64))
    assert reference.master_parameters.keys() == oracle.keys()
    for name, parameter in reference.master_parameters.items():
        assert parameter.dtype == parameter.grad.dtype == torch.float32
        torch.testing.assert_close(parameter.grad, oracle[name] / sum(counts), rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in original.parameters())


def test_forward_preserves_all_parameters_buffers_and_hf_module_state():
    original = Scalar()
    parameters = {
        name: (parameter, parameter.clone()) for name, parameter in original.named_parameters()
    }
    buffers = {name: (value, value.clone()) for name, value in original.named_buffers()}
    reference = FP32GradientReference(original)
    reference(torch.tensor(2, dtype=torch.bfloat16)).backward()
    assert reference.master_parameters.keys() == parameters.keys()
    for name, (identity, value) in parameters.items():
        assert original.get_parameter(name) is identity
        torch.testing.assert_close(original.get_parameter(name), value, rtol=0, atol=0)
        assert identity.grad is None
    for name, (identity, value) in buffers.items():
        assert original.get_buffer(name) is identity
        torch.testing.assert_close(original.get_buffer(name), value, rtol=0, atol=0)
    reference.normalize_gradients_(2)
    with pytest.raises(RuntimeError, match="already normalized"):
        reference.normalize_gradients_(2)
    with pytest.raises(RuntimeError, match="Clear gradients"):
        reference(torch.tensor(2, dtype=torch.bfloat16))
    reference.zero_grad()
    assert all(p.grad is None for p in reference.master_parameters.values())
    reference(torch.tensor(3, dtype=torch.bfloat16)).backward()
    assert reference.master_parameters["weight"].grad.item() == 3


@pytest.mark.parametrize("count", [0, -1, 1.5, True, torch.ones(1), torch.tensor(1.0)])
def test_invalid_global_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive integer scalar"):
        FP32GradientReference(Scalar()).normalize_gradients_(count)


def test_non_bf16_reference_is_rejected():
    with pytest.raises(ValueError, match="BF16"):
        FP32GradientReference(nn.Linear(2, 2))


def test_original_synthetic_audio_hf_all_92_parameter_gradients_match_fp32_oracle():
    from tests.acceptance.omni_audio_reference import HFReference, fixture

    torch.set_num_threads(2)
    config, batches = fixture()
    original = HFReference(config)
    independent = copy.deepcopy(original)
    reference = FP32GradientReference(original)
    expected = {
        name: torch.zeros_like(p, dtype=torch.float32) for name, p in independent.named_parameters()
    }
    assert len(expected) == 92
    count = sum(int((labels != -100).sum()) for _, labels in batches)
    for inputs, labels in batches:
        actual, target = reference(inputs), independent(inputs)
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
        F.cross_entropy(actual.float().flatten(0, 1), labels.flatten(), reduction="sum").backward()
        F.cross_entropy(target.float().flatten(0, 1), labels.flatten(), reduction="sum").backward()
        for name, parameter in independent.named_parameters():
            assert parameter.grad is not None
            expected[name].add_(parameter.grad.float())
        independent.zero_grad(set_to_none=True)
    reference.normalize_gradients_(count)
    assert reference.master_parameters.keys() == expected.keys()
    for name, parameter in reference.master_parameters.items():
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32, name
        torch.testing.assert_close(parameter.grad, expected[name] / count, rtol=0, atol=0)
    assert all(p.grad is None for p in original.parameters())
    assert not torch.cuda.is_initialized()
