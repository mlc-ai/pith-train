from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed
import torch.distributed._functional_collectives as funcol

__all__ = [
    "GlobalBatchLoadBalanceLoss",
    "MicroBatchLoadBalanceLoss",
    "MoELoadBalanceLossInjector",
    "MoELoadBalanceLossTracker",
    "SequenceLevelLoadBalanceLoss",
    "force_balance",
    "make_load_balance_loss_fn",
    "replay_indices",
]


def expert_token_counts(idx_flat: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    torch.bincount(idx_flat, minlength=n_bins) without the host sync of CUDA bincount. Every value
    must lie in [0, n_bins).
    """
    out = torch.zeros(n_bins, dtype=torch.long, device=idx_flat.device)
    # Stride-0 view of a single 1 avoids materializing a full-size ones tensor.
    ones = torch.ones(1, dtype=torch.long, device=idx_flat.device).expand_as(idx_flat)
    out.scatter_add_(0, idx_flat, ones)
    return out


class MoELoadBalanceLossInjector(torch.autograd.Function):
    """
    Attach a load-balance loss to the graph: output passes through unchanged, and the backward
    gives lb_loss a gradient of one.
    """

    @staticmethod
    def forward(ctx, output: torch.Tensor, lb_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lb_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (lb_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(lb_loss)


torch._dynamo.allow_in_graph(MoELoadBalanceLossInjector)


class MicroBatchLoadBalanceLoss:
    """
    Micro-batch load-balance loss (https://arxiv.org/abs/2101.03961):

        loss = lb_coef * E * sum_i(f_i * P_i)

    where f_i is the fraction of tokens dispatched to expert i and P_i is the mean router
    probability of expert i, both over the micro-batch.
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

        tokens_per_expert = expert_token_counts(topk_idx.reshape(-1), num_experts)
        tokens_per_expert = tokens_per_expert.to(scores.dtype)
        f = tokens_per_expert / (num_tokens * top_k)
        p = scores.mean(dim=0)

        return self.lb_coef * num_experts * torch.dot(f, p)

    def reset(self) -> None:
        pass


class GlobalBatchLoadBalanceLoss:
    """
    Global-batch load-balance loss (https://arxiv.org/abs/2501.11873): f_i is the fraction of tokens
    routed to expert i over process_group and over the micro-batches of the step, which reset()
    clears. Call init_buffers before the first forward; the counts live in preallocated tensors
    updated in place, so __call__ compiles.
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
        self.count_buffer: Optional[torch.Tensor] = None
        self.total_tokens: Optional[torch.Tensor] = None

    def init_buffers(self, num_experts: int, device: torch.device) -> None:
        if self.count_buffer is None:
            # float32 to match the dtype of scores, a softmax output.
            self.count_buffer = torch.zeros(num_experts, dtype=torch.float32, device=device)
            self.total_tokens = torch.zeros((), dtype=torch.int32, device=device)

    def __call__(
        self,
        scores: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        num_tokens = scores.shape[0]

        tokens_per_expert = expert_token_counts(topk_idx.reshape(-1), num_experts)
        tokens_per_expert = tokens_per_expert.to(scores.dtype)

        # A functional collective keeps the gate fullgraph-traceable; counts need no gradient.
        if self.process_group is not None:
            tokens_per_expert = funcol.all_reduce(tokens_per_expert, "sum", self.process_group)

        self.count_buffer.add_(tokens_per_expert.detach())
        self.total_tokens.add_(num_tokens * top_k * self.group_size)

        f = self.count_buffer / self.total_tokens
        p = scores.mean(dim=0)

        return self.lb_coef * num_experts * torch.dot(f, p)

    def reset(self) -> None:
        if self.count_buffer is not None:
            self.count_buffer.zero_()
            self.total_tokens.zero_()


class SequenceLevelLoadBalanceLoss:
    """
    Sequence-level load-balance loss (https://arxiv.org/abs/2405.04434): the load-balance loss of
    each sequence of sequence_length tokens, averaged over the batch. Under context parallelism f
    is summed over cp_group and normalized by the full sequence length, so the loss matches CP=1.
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

    def all_reduce_expert_counts(self, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        # A functional collective keeps the gate fullgraph-traceable; counts need no gradient.
        if self.cp_size > 1:
            tokens_per_expert = funcol.all_reduce(tokens_per_expert, "sum", self.cp_group)
        return tokens_per_expert

    def __call__(
        self,
        scores: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        sequence_length = self.sequence_length
        batch_size = scores.shape[0] // sequence_length

        # Offset each sequence's indices into its own block of num_experts bins, so one count
        # covers the whole batch.
        batch_offsets = torch.arange(batch_size, device=topk_idx.device)
        batch_offsets = batch_offsets.unsqueeze(1).expand(-1, sequence_length * top_k) * num_experts
        flat_idx = topk_idx.view(batch_size, -1) + batch_offsets
        tokens_per_expert = expert_token_counts(flat_idx.reshape(-1), batch_size * num_experts)
        tokens_per_expert = tokens_per_expert.to(scores.dtype).view(batch_size, num_experts)

        tokens_per_expert = self.all_reduce_expert_counts(tokens_per_expert)

        f = tokens_per_expert / (sequence_length * self.cp_size * top_k)
        p = scores.view(batch_size, sequence_length, num_experts).mean(dim=1)

        per_seq = (f * p).sum(dim=1) * num_experts
        return self.lb_coef * per_seq.mean()

    def reset(self) -> None:
        pass


class MoELoadBalanceLossTracker:
    """
    The registry of load-balance losses, reset after each optimizer step, and the loss values
    accumulated for logging.
    """

    instances: List = []
    losses: List = []

    @classmethod
    def register(cls, instance) -> None:
        cls.instances.append(instance)

    @classmethod
    def reset(cls) -> None:
        for instance in cls.instances:
            instance.reset()

    @classmethod
    def add(cls, loss: torch.Tensor) -> None:
        cls.losses.append(loss.detach())

    @classmethod
    def get_total_count_and_clear(cls) -> Tuple[float, int]:
        """
        Syncs with the device.
        """
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


def force_balance(num_experts: int) -> Callable[[int, int], torch.Tensor]:
    """
    A benchmark-only router replay that gives every expert an equal load. The returned function maps
    (num_tokens, top_k) to the (num_tokens, top_k) expert indices a gate should use.
    """
    cache: dict[tuple[int, int], torch.Tensor] = {}

    def replay(num_tokens: int, top_k: int) -> torch.Tensor:
        indices = cache.get((num_tokens, top_k))
        if indices is None:
            stride = num_experts // top_k
            device = torch.cuda.current_device()
            token = torch.arange(num_tokens, device=device).unsqueeze(1)
            slot = torch.arange(top_k, device=device).unsqueeze(0)
            indices = cache[(num_tokens, top_k)] = (token + slot * stride) % num_experts
        return indices

    return replay


def replay_indices(
    gate: torch.nn.Module, hidden_states: torch.Tensor, top_k: int
) -> torch.Tensor | None:
    """
    Call it in eager mode, so the compiled gate takes the indices as an input.
    router_replay(num_tokens, top_k) returns expert ids [num_tokens, top_k] as a contiguous int64
    or int32 tensor on the current device, rows in loader token order (zigzag under CP). Layers
    with another token order relayout the rows. Called once per gate per forward, never from a
    backward; each gate's store keys routes by layer and training.current_microbatch.
    """
    if gate.router_replay is None:
        return None
    return gate.router_replay(hidden_states.shape[:-1].numel(), top_k)
