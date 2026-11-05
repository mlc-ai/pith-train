"""Token-weighted loss and gradient normalization across one pipeline stage."""

import torch
import torch.distributed as dist


def global_target_count(microbatches, group=None):
    """Sum non-ignored labels over the stage's DP x CP group, excluding PP replicas.

    For equal local counts this is local_count * dp_size * cp_size, matching
    dense training. Masked media batches can have unequal (or zero) local counts,
    so multiplying one rank's count by the group size would be incorrect.
    """
    count = sum((mb.objective_inputs[0] != -100).sum() for mb in microbatches)
    if dist.is_initialized():
        dist.all_reduce(count, group=group)
    if count.item() == 0:
        raise ValueError("The global batch has no valid next-token targets")
    return count.to(dtype=torch.float32)


def global_loss_mean(local_loss_sum, global_target_count, group=None):
    """Sum rank-local losses and divide once by the already-global target count."""
    total = local_loss_sum.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(total, group=group)
    return total / global_target_count
