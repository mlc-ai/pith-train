"""Independent CPU token/codec mathematics; no native model or CUDA substitutes."""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.tasks.omni_objectives import (
    assistant_targets,
    auxiliary_loss_sum,
    codec_loss_sums,
    global_target_counts,
    normalized_loss,
    speech_loss_sums,
    text_loss_sum,
)


def test_assistant_only_golden_labels_shift_once_keep_prompt_and_eos():
    tokens = torch.tensor([[0, 0, 11, 12, 21, 22, 2], [11, 12, 21, 2, 0, 0, 0]])
    assistant = torch.tensor(
        [
            [False, False, False, False, True, True, True],
            [False, False, True, True, False, False, False],
        ]
    )
    attention = tokens.ne(0)
    original = tokens.clone()
    inputs, labels = assistant_targets(tokens, assistant, attention_mask=attention)
    assert torch.equal(inputs, tokens[:, :-1])
    assert torch.equal(
        labels, torch.tensor([[-100, -100, -100, 21, 22, 2], [-100, 21, 2, -100, -100, -100]])
    )
    assert torch.equal(tokens, original)
    logits = torch.randn(2, 6, 25, dtype=torch.float64, requires_grad=True)
    result = text_loss_sum(logits, labels)
    # Select original supervised predictions directly; do not reuse a label-shift helper.
    selected = torch.stack([logits[0, 3], logits[0, 4], logits[0, 5], logits[1, 1], logits[1, 2]])
    expected = F.cross_entropy(selected, torch.tensor([21, 22, 2, 21, 2]), reduction="sum")
    torch.testing.assert_close(result.loss_sum, expected, rtol=1e-14, atol=1e-14)
    assert result.target_count.item() == 5


@pytest.mark.parametrize("mask", [None, torch.ones(1, 3), torch.ones(1, 2, dtype=torch.bool)])
def test_assistant_mask_must_be_explicit_boolean_and_aligned(mask):
    with pytest.raises((ValueError, TypeError)):
        assistant_targets(torch.tensor([[1, 2, 3]]), mask)


@pytest.mark.parametrize("sizes", [(10, 90), (1, 9, 30, 60), (0, 100)])
def test_microbatch_partition_preserves_global_token_loss_and_gradients(sizes):
    torch.manual_seed(91)
    logits = torch.randn(100, 9, dtype=torch.float64, requires_grad=True)
    oracle = logits.detach().clone().requires_grad_()
    labels = torch.arange(100) % 9
    pieces = [
        text_loss_sum(x, y) for x, y in zip(logits.split(sizes), labels.split(sizes), strict=True)
    ]
    sums = torch.stack([p.loss_sum for p in pieces]).sum()
    count = torch.stack([p.target_count for p in pieces]).sum()
    actual = normalized_loss(sums, count, weighting="combined")
    expected = F.cross_entropy(oracle, labels)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(logits.grad, oracle.grad, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ignored_targets_retain_finite_zero_cotangents(dtype):
    logits = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    loss = text_loss_sum(logits, torch.full((2, 3), -100))
    assert loss.loss_sum.dtype == torch.float32 and loss.target_count.item() == 0
    loss.loss_sum.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("weighting", ["combined", "per_head"])
def test_codec_time_codebook_mask_matches_independent_logprob_gradient(weighting):
    torch.manual_seed(37)
    logits = torch.randn(1, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    oracle = logits.detach().clone().requires_grad_()
    labels = torch.tensor([[[1, 2, 3], [2, 1, 2], [3, 0, 1], [4, 3, -100]]])
    mask = torch.tensor(
        [[[True, True, False], [False, True, False], [False, True, False], [True, True, False]]]
    )
    result = codec_loss_sums(logits, labels, mask)
    assert result.target_count.tolist() == [2, 4, 0]
    actual = normalized_loss(result.loss_sum, result.target_count, weighting=weighting)
    logp = oracle.log_softmax(-1)
    first = -(logp[0, 0, 0, 1] + logp[0, 3, 0, 4])
    second = -(logp[0, 0, 1, 2] + logp[0, 1, 1, 1] + logp[0, 2, 1, 0] + logp[0, 3, 1, 3])
    expected = (first + second) / 6 if weighting == "combined" else (first / 2 + second / 4) / 2
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(logits.grad, oracle.grad, rtol=1e-14, atol=1e-14)
    assert torch.count_nonzero(logits.grad[:, :, 2]) == 0


def test_first_codec_group_eos_and_residual_vocab_have_independent_counts():
    first = torch.randn(1, 3, 11, dtype=torch.float64, requires_grad=True)
    residual = torch.randn(1, 2, 2, 5, dtype=torch.float64, requires_grad=True)
    first_labels = torch.tensor([[4, 8, 10]])  # Last target is the first-group EOS.
    residual_labels = torch.tensor([[[2, -100], [1, 3]]])
    result = speech_loss_sums(first, first_labels, residual, residual_labels)
    assert result.target_count.tolist() == [3, 2, 1]
    expected = F.cross_entropy(first[0], first_labels[0], reduction="sum")
    expected += F.cross_entropy(residual[0, :, 0], torch.tensor([2, 1]), reduction="sum")
    expected += F.cross_entropy(residual[0, 1, 1].unsqueeze(0), torch.tensor([3]), reduction="sum")
    torch.testing.assert_close(result.loss_sum.sum(), expected, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match=r"T\+1"):
        speech_loss_sums(first[:, :-1], first_labels[:, :-1], residual, residual_labels)


def test_empty_residual_frames_keep_first_group_eos_objective():
    first = torch.randn(1, 1, 7, requires_grad=True)
    residual = torch.empty(1, 0, 2, 5, requires_grad=True)
    result = speech_loss_sums(
        first, torch.tensor([[6]]), residual, torch.empty(1, 0, 2, dtype=torch.long)
    )
    assert result.target_count.tolist() == [1, 0, 0]
    result.loss_sum.sum().backward()
    assert residual.grad.shape == residual.shape
    assert torch.isfinite(first.grad).all()


def test_zero_local_count_still_uses_explicit_collective(monkeypatch):
    from pithtrain.tasks import omni_objectives as module

    calls, group = [], object()
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)

    def reduce(value, *, op, group):
        calls.append((value.clone(), op, group))
        value.add_(torch.tensor([3, 5]))

    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    local = torch.zeros(2, dtype=torch.int64)
    total = global_target_counts(local, token_group=group)
    assert total.tolist() == [3, 5] and local.tolist() == [0, 0]
    assert len(calls) == 1 and calls[0][2] is group
    with pytest.raises(ValueError, match="explicitly"):
        global_target_counts(local, token_group=None)
    assert len(calls) == 1


@pytest.mark.parametrize("weighting", ["combined", "per_head"])
def test_global_zero_normalization_is_finite_zero_and_not_an_update_policy(weighting):
    sums = torch.tensor([0.0, 0.0], requires_grad=True)
    result = normalized_loss(sums, torch.zeros(2, dtype=torch.long), weighting=weighting)
    result.backward()
    assert result.item() == 0 and torch.equal(sums.grad, torch.zeros_like(sums))


@pytest.mark.parametrize(
    "case", ["float_labels", "wrong_shape", "negative_label", "float_mask", "bad_weighting"]
)
def test_invalid_objective_contracts_fail_closed(case):
    logits, labels = torch.randn(1, 2, 5), torch.tensor([[1, 2]])
    with pytest.raises((ValueError, TypeError)):
        if case == "float_labels":
            text_loss_sum(logits, labels.float())
        elif case == "wrong_shape":
            text_loss_sum(logits, labels[:, :1])
        elif case == "negative_label":
            text_loss_sum(logits, torch.tensor([[-1, 2]]))
        elif case == "float_mask":
            text_loss_sum(logits, labels, torch.ones_like(labels).float())
        else:
            normalized_loss(torch.tensor(1.0), torch.tensor(1), weighting="local_mean")


def test_auxiliary_mask_uses_own_count_not_assistant_denominator():
    values = torch.tensor([[1.0, 3.0, 7.0, 11.0]], dtype=torch.float64, requires_grad=True)
    result = auxiliary_loss_sum(values.square(), torch.tensor([[True, False, True, True]]))
    assert result.target_count.item() == 3  # Independent of any text/assistant mask.
    normalized_loss(result.loss_sum, result.target_count, weighting="combined").backward()
    torch.testing.assert_close(
        values.grad, torch.tensor([[2 / 3, 0, 14 / 3, 22 / 3]], dtype=torch.float64)
    )
    empty = auxiliary_loss_sum(values, torch.zeros_like(values, dtype=torch.bool))
    assert empty.target_count.item() == 0 and empty.loss_sum.item() == 0
    with pytest.raises(ValueError, match="boolean"):
        auxiliary_loss_sum(values, torch.ones_like(values))
