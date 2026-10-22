"""Aligned text/codec objectives, independent of model construction and CUDA kernels.

Only ``assistant_targets`` shifts raw tokens. Every loss below consumes labels
already aligned with its logits. Auxiliary MoE losses are deliberately separate.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F

IGNORE_INDEX = -100


def _masked_labels(labels: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if labels.dtype != torch.long:
        raise TypeError("Target labels must be int64")
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != labels.shape or mask.device != labels.device:
            raise ValueError("Target mask must be boolean with the labels' shape and device")
        labels = labels.masked_fill(~mask, IGNORE_INDEX)
    if ((labels < 0) & (labels != IGNORE_INDEX)).any():
        raise ValueError("Negative labels must use ignore_index=-100")
    return labels


def assistant_targets(
    input_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift raw [B,L] tokens exactly once and supervise assistant target positions.

    The processor supplies the assistant mask (including EOS only if desired).
    Prompt tokens remain in model input. Padding is excluded as a target and as
    its predecessor, so left padding cannot create a spurious first-token loss.
    This does not construct an attention mask or claim packed-document isolation.
    """
    if input_ids.ndim != 2 or input_ids.size(1) < 2 or input_ids.dtype != torch.long:
        raise ValueError("Raw input_ids must be int64 [B,L] with L >= 2")
    if (input_ids < 0).any():
        raise ValueError("Raw input_ids must contain vocabulary IDs, not ignored labels")
    if assistant_mask is None:
        raise ValueError("An explicit boolean assistant mask is required")
    labels = _masked_labels(input_ids, assistant_mask)
    if attention_mask is not None:
        _masked_labels(input_ids, attention_mask)
        valid = attention_mask[:, 1:] & attention_mask[:, :-1]
        labels = labels[:, 1:].masked_fill(~valid, IGNORE_INDEX)
    else:
        labels = labels[:, 1:].clone()
    return input_ids[:, :-1], labels


@dataclass(frozen=True, slots=True)
class TargetLoss:
    """Differentiable loss sums and int64 counts; scalar for text, [G] for codec."""

    loss_sum: torch.Tensor
    target_count: torch.Tensor


def text_loss_sum(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
) -> TargetLoss:
    """Sum CE over aligned targets; all-ignored input retains a zero-gradient graph."""
    labels = _masked_labels(labels, mask)
    if not logits.is_floating_point() or logits.ndim != labels.ndim + 1:
        raise ValueError("Floating logits must have shape labels.shape + [vocabulary]")
    if logits.shape[:-1] != labels.shape or logits.device != labels.device or logits.size(-1) < 1:
        raise ValueError("Logits and labels must have matching target dimensions and device")
    # Accumulate half/bfloat16 CE in FP32, without demoting FP64 reference inputs.
    scores = logits if logits.dtype == torch.float64 else logits.float()
    loss = F.cross_entropy(
        scores.reshape(-1, scores.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return TargetLoss(loss, labels.ne(IGNORE_INDEX).sum(dtype=torch.int64))


def codec_loss_sums(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None
) -> TargetLoss:
    """Per-codebook sums/counts for [B,T,G,V] logits and [B,T,G] aligned targets.

    A boolean time/codebook mask is combined with existing -100 labels. No time
    or codebook shift occurs here; teacher-forcing alignment belongs to the model.
    """
    if logits.ndim != 4 or labels.ndim != 3 or logits.shape[:-1] != labels.shape:
        raise ValueError("Codec logits/labels must be [B,T,G,V] / [B,T,G]")
    labels = _masked_labels(labels, mask)
    if logits.size(2) == 0:
        if not logits.is_floating_point() or logits.device != labels.device:
            raise ValueError("Codec logits must be floating point on the labels' device")
        sums = logits.sum(dim=(0, 1, 3))
        return TargetLoss(sums, labels.ne(IGNORE_INDEX).sum(dim=(0, 1), dtype=torch.int64))
    heads = [text_loss_sum(logits[:, :, h], labels[:, :, h]) for h in range(logits.size(2))]
    return TargetLoss(
        torch.stack([head.loss_sum for head in heads]),
        torch.stack([head.target_count for head in heads]),
    )


def speech_loss_sums(
    first_logits: torch.Tensor,
    first_labels: torch.Tensor,
    residual_logits: torch.Tensor,
    residual_labels: torch.Tensor,
    *,
    first_mask: torch.Tensor | None = None,
    residual_mask: torch.Tensor | None = None,
) -> TargetLoss:
    """Separate first-group/EOS [B,T+1,V0] and residual [B,T,G-1,Vr] heads.

    V0 and Vr may differ. The returned [G] statistics retain each head's own
    count so callers explicitly choose combined-target or per-head weighting.
    """
    if (
        first_logits.ndim != 3
        or residual_logits.ndim != 4
        or first_logits.size(0) != residual_logits.size(0)
        or first_logits.size(1) != residual_logits.size(1) + 1
        or first_logits.device != residual_logits.device
    ):
        raise ValueError("Speech heads must align [B,T+1,V0] with [B,T,G-1,Vr]")
    first = text_loss_sum(first_logits, first_labels, first_mask)
    residual = codec_loss_sums(residual_logits, residual_labels, residual_mask)
    return TargetLoss(
        torch.cat((first.loss_sum.reshape(1), residual.loss_sum)),
        torch.cat((first.target_count.reshape(1), residual.target_count)),
    )


def global_target_counts(local_counts: torch.Tensor, *, token_group) -> torch.Tensor:
    """Sum even zero counts over explicitly selected nonduplicate DP x CP shards.

    PP copies must not be counted again; EP is not an additional data axis. Each
    PP stage must receive equivalent label metadata and call on its own group.
    ``None`` is allowed only for a non-distributed local run, not implicit WORLD.
    """
    if local_counts.dtype != torch.int64 or local_counts.ndim > 1 or (local_counts < 0).any():
        raise ValueError("Counts must be nonnegative int64 scalars or per-head vectors")
    total = local_counts.detach().clone()
    if dist.is_initialized():
        if token_group is None:
            raise ValueError("Select the nonduplicated DP x CP token group explicitly")
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=token_group)
    return total


def normalized_loss(
    loss_sums: torch.Tensor,
    global_counts: torch.Tensor,
    *,
    weighting: Literal["combined", "per_head"],
) -> torch.Tensor:
    """Normalize once using global counts, never an average of rank-local means.

    ``combined`` weights every active target equally. ``per_head`` gives equal
    weight to each globally nonempty head's token mean. All-zero counts return
    differentiable zero, but the task must uniformly skip optimizer/scheduler
    updates (including weight decay). Replica gradient reduction must be SUM.
    """
    if (
        not loss_sums.is_floating_point()
        or global_counts.dtype != torch.int64
        or loss_sums.shape != global_counts.shape
        or loss_sums.device != global_counts.device
        or global_counts.ndim > 1
        or (global_counts < 0).any()
    ):
        raise ValueError(
            "Loss sums and nonnegative int64 global counts must have equal shape/device"
        )
    if weighting == "combined":
        count = global_counts.sum()
        total = loss_sums.sum()
        return torch.where(count > 0, total / count.clamp_min(1), total * 0)
    if weighting == "per_head":
        active = global_counts > 0
        means = loss_sums / global_counts.clamp_min(1)
        return torch.where(active, means, 0).sum() / active.sum().clamp_min(1)
    raise ValueError("Select weighting='combined' or 'per_head' explicitly")


def auxiliary_loss_sum(per_item_loss: torch.Tensor, mask: torch.Tensor) -> TargetLoss:
    """Explicit auxiliary sum/count boundary, separate from assistant targets.

    The caller defines the actual auxiliary formula and its own item mask. This
    helper does not implement a MoE formula or wire the existing gradient injector
    into SFT. Compose and globally normalize each objective with its own counts.
    """
    if not per_item_loss.is_floating_point() or (
        mask.dtype != torch.bool
        or mask.shape != per_item_loss.shape
        or mask.device != per_item_loss.device
    ):
        raise ValueError("Auxiliary loss needs floating values and an aligned boolean mask")
    values = per_item_loss if per_item_loss.dtype == torch.float64 else per_item_loss.float()
    return TargetLoss(torch.where(mask, values, 0).sum(), mask.sum(dtype=torch.int64))
