"""MoE routing load balance losses."""

from typing import List, Optional, Tuple

import torch
import torch.distributed


def _expert_token_counts(idx_flat: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Equivalent to ``torch.bincount(idx_flat, minlength=n_bins)`` but without
    the synchronous ``input.max().item()`` that ATen's CUDA bincount issues to
    size its output. Assumes all values in ``[0, n_bins)``."""
    out = torch.zeros(n_bins, dtype=torch.long, device=idx_flat.device)
    # Stride-0 view of a single 1 avoids materializing a full-size ones tensor.
    ones = torch.ones(1, dtype=torch.long, device=idx_flat.device).expand_as(idx_flat)
    out.scatter_add_(0, idx_flat, ones)
    return out


class MoELoadBalanceLossInjector(torch.autograd.Function):
    """
    Attach a load balance loss to the computation graph. The output tensor
    passes through unchanged; in the backward pass the loss gradient is injected.
    """

    @staticmethod
    def forward(ctx, output: torch.Tensor, lb_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lb_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (lb_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(lb_loss)


class MicroBatchLoadBalanceLoss:
    """
    Micro-batch load balance loss (https://arxiv.org/abs/2101.03961).

        loss = coeff * E * sum_i(f_i * P_i)

    where f_i is the fraction of tokens dispatched to expert i and P_i is
    the average router probability for expert i.
    """

    def __init__(self, lb_coef: float) -> None:
        self.lb_coef = lb_coef

    def __call__(
        self,
        scores: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        num_tokens = scores.shape[0]

        tokens_per_expert = _expert_token_counts(topk_idx.reshape(-1), num_experts)
        tokens_per_expert = tokens_per_expert.to(scores.dtype)
        f = tokens_per_expert / (num_tokens * top_k)
        p = scores.mean(dim=0)

        return self.lb_coef * num_experts * torch.dot(f, p)

    def reset(self) -> None:
        pass


class GlobalBatchLoadBalanceLoss:
    """
    Global-batch load balance loss (https://arxiv.org/abs/2501.11873).

    Synchronises expert selection frequencies across DP x EP ranks via
    all-reduce and accumulates counts across gradient-accumulation
    micro-steps. The buffer is cleared after each optimizer step via
    MoELoadBalanceLossTracker.reset().

    Accumulation state uses pre-allocated tensors with in-place ops so that
    __call__ is compatible with torch.compile (no Python attribute mutation).
    Call init_buffers() before the first forward pass.
    """

    def __init__(
        self,
        lb_coef: float,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.lb_coef = lb_coef
        self.process_group = process_group
        self.group_size = (
            torch.distributed.get_world_size(process_group) if process_group is not None else 1
        )
        self._count_buffer: Optional[torch.Tensor] = None
        self._total_tokens: Optional[torch.Tensor] = None

    def init_buffers(self, num_experts: int, device: torch.device) -> None:
        """Pre-allocate accumulation buffers. Must be called before __call__."""
        if self._count_buffer is None:
            # float32 to match scores dtype (softmax always outputs float32).
            self._count_buffer = torch.zeros(num_experts, dtype=torch.float32, device=device)
            self._total_tokens = torch.zeros((), dtype=torch.int32, device=device)

    def __call__(
        self,
        scores: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        num_tokens = scores.shape[0]

        tokens_per_expert = _expert_token_counts(topk_idx.reshape(-1), num_experts)
        tokens_per_expert = tokens_per_expert.to(scores.dtype)

        # Synchronise counts across DP x EP ranks.
        if self.process_group is not None:
            torch.distributed.all_reduce(tokens_per_expert, group=self.process_group)

        # Accumulate in the gradient-accumulation buffer (in-place tensor ops).
        self._count_buffer.add_(tokens_per_expert.detach())
        self._total_tokens.add_(num_tokens * top_k * self.group_size)

        f = self._count_buffer / self._total_tokens
        p = scores.mean(dim=0)

        return self.lb_coef * num_experts * torch.dot(f, p)

    def reset(self) -> None:
        if self._count_buffer is not None:
            self._count_buffer.zero_()
            self._total_tokens.zero_()


class SequenceLevelLoadBalanceLoss:
    """
    Sequence-level load balance loss (https://arxiv.org/abs/2405.04434).

    Computes the standard load balance loss independently per sequence, then
    averages over the batch.  This matches the DeepSeek-V2 formulation where
    T is the number of tokens in a *single sequence*, not the micro-batch.

    With CP, each CP rank sees only a chunk of the sequence.
    The expert fraction f is all-reduced across CP ranks and normalized
    by the full sequence length so that the LB gradient direction at the
    gate weight matches CP=1.
    """

    def __init__(
        self,
        lb_coef: float,
        sequence_length: int,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.lb_coef = lb_coef
        self.sequence_length = sequence_length
        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1

    def _all_reduce_expert_counts(self, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """All-reduce expert counts across CP ranks (noop when cp_group is None)."""
        if self.cp_group is not None:
            torch.distributed.all_reduce(tokens_per_expert, group=self.cp_group)
        return tokens_per_expert

    def __call__(
        self,
        scores: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        sequence_length = self.sequence_length
        bsz = scores.shape[0] // sequence_length

        # Per-expert token counts per sequence: offset topk indices into a
        # flattened [B*E] space so we can bincount in a single call.
        batch_offsets = torch.arange(bsz, device=topk_idx.device)
        batch_offsets = batch_offsets.unsqueeze(1).expand(-1, sequence_length * top_k) * num_experts
        flat_idx = topk_idx.view(bsz, -1) + batch_offsets
        tokens_per_expert = _expert_token_counts(flat_idx.reshape(-1), bsz * num_experts)
        tokens_per_expert = tokens_per_expert.to(scores.dtype).view(bsz, num_experts)

        tokens_per_expert = self._all_reduce_expert_counts(tokens_per_expert)

        f = tokens_per_expert / (sequence_length * self.cp_size * top_k)
        p = scores.view(bsz, sequence_length, num_experts).mean(dim=1)

        per_seq = (f * p).sum(dim=1) * num_experts
        return self.lb_coef * per_seq.mean()

    def reset(self) -> None:
        pass


class MoELoadBalanceLossTracker:
    """Tracks load balance loss instances and accumulates loss values for logging."""

    instances: List = []
    losses: List = []

    @classmethod
    def register(cls, instance) -> None:
        cls.instances.append(instance)

    @classmethod
    def reset(cls) -> None:
        for inst in cls.instances:
            inst.reset()

    @classmethod
    def add(cls, loss: torch.Tensor) -> None:
        cls.losses.append(loss.detach())

    @classmethod
    def get_total_count_and_clear(cls) -> Tuple[float, int]:
        """Return (total, count) of accumulated losses and clear the log."""
        if not cls.losses:
            return 0.0, 0
        total = torch.stack(cls.losses).sum().item()
        count = len(cls.losses)
        cls.losses.clear()
        return total, count


def make_load_balance_loss_fn(
    lb_type: str,
    lb_coef: float,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    sequence_length: int = 0,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    Create a load balance loss callable for MoE gates.

    Returns a callable with signature (scores, topk_idx, num_experts, top_k) -> loss.
    """
    match lb_type:
        case "micro-batch":
            loss = MicroBatchLoadBalanceLoss(lb_coef)
        case "global-batch":
            loss = GlobalBatchLoadBalanceLoss(lb_coef, process_group)
        case "sequence":
            loss = SequenceLevelLoadBalanceLoss(lb_coef, sequence_length, cp_group)
        case _:
            raise ValueError(f"Unknown moe_load_balance_type: {lb_type!r}")
    MoELoadBalanceLossTracker.register(loss)
    return loss
