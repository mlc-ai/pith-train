"""
Muon optimizer for the 2-D (and stacked 3-D expert) hidden weights: SGD momentum, then a
Newton-Schulz orthogonalization of the update. The other parameters go to AdamW, which
make_muon_optimizer in training.py composes with Muon.

Under FSDP2 the orthogonalization needs the full matrix, so each weight is gathered on every rank
of its shard group, orthogonalized by one owner rank and broadcast, the owners taken round-robin.
The result is bit-identical to every rank doing all of them.
"""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

__all__ = [
    "Muon",
    "clip_grad_norm",
    "local_grads",
    "muon_scale_factor",
    "orthogonalized_updates",
    "scale_grads_",
    "zeropower_via_newtonschulz5",
]


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Approximately orthogonalize G with steps quintic Newton-Schulz iterations in bf16, batched over
    the leading dims: the singular values are driven toward 1. Returns a bf16 tensor of G's shape.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # The Frobenius prescale bounds the spectral norm by 1, where the iteration converges.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_scale_factor(n: int, m: int) -> float:
    """
    Moonlight's update scale 0.2 * sqrt(max(n, m)), which brings the RMS of an orthogonalized n x m
    update to that of an AdamW step, so Muon and AdamW share one learning rate.
    """
    return 0.2 * max(n, m) ** 0.5


def orthogonalize_scaled(full, dtype, steps):
    orth = zeropower_via_newtonschulz5(full, steps=steps).to(dtype)
    return (orth * muon_scale_factor(full.size(-2), full.size(-1))).contiguous()


def broadcast_owned(results, updates, params, idxs, owner_of, group, group_size):
    """
    Fill results[i] for every i in idxs on every rank of group: each owner broadcasts the results
    it computed as one flat buffer. Assumes one dtype per group.
    """
    group_rank = dist.get_rank(group)
    for owner in range(group_size):
        owned = [i for i in idxs if owner_of[i] == owner]
        if not owned:
            continue
        shapes = [tuple(updates[i].shape) for i in owned]
        numels = [math.prod(shape) for shape in shapes]
        dtype, device = params[owned[0]].dtype, params[owned[0]].device
        if group_rank == owner:
            flat = torch.cat([results[i].reshape(-1) for i in owned])
        else:
            flat = torch.empty(sum(numels), dtype=dtype, device=device)
        dist.broadcast(flat, src=dist.get_global_rank(group, owner), group=group)
        offset = 0
        for i, shape, numel in zip(owned, shapes, numels):
            results[i] = flat[offset : offset + numel].view(shape)
            offset += numel


def orthogonalized_updates(params, updates, steps: int = 5, chunk_size: int | None = None):
    """
    The orthogonalized and scaled update of each updates[i] (a DTensor or a plain tensor), as a
    full tensor in params[i].dtype. Weights sharded over a 1-D mesh are split round-robin across
    its ranks, chunk_size at a time; collective over each such mesh.
    """
    results = [None] * len(updates)

    buckets: dict[int, tuple] = {}  # id(group) -> (group, group size, group rank, idxs)
    local = []
    for i, update in enumerate(updates):
        if isinstance(update, DTensor) and update.device_mesh.ndim == 1:
            group = update.device_mesh.get_group()
            if dist.get_world_size(group) > 1:
                key = id(group)
                if key not in buckets:
                    buckets[key] = (group, dist.get_world_size(group), dist.get_rank(group), [])
                buckets[key][3].append(i)
                continue
        local.append(i)

    for i in local:
        update = updates[i]
        full = update.full_tensor() if isinstance(update, DTensor) else update
        results[i] = orthogonalize_scaled(full, params[i].dtype, steps)

    for group, group_size, group_rank, idxs in buckets.values():
        chunk = chunk_size or max(4 * group_size, 8)
        owner_of = {idx: pos % group_size for pos, idx in enumerate(idxs)}
        for start in range(0, len(idxs), chunk):
            window = idxs[start : start + chunk]
            # No collective between the owners' orthogonalizations, so they run concurrently.
            fulls = {i: updates[i].full_tensor() for i in window}
            for i in window:
                if owner_of[i] == group_rank:
                    results[i] = orthogonalize_scaled(fulls[i], params[i].dtype, steps)
            fulls.clear()
        broadcast_owned(results, updates, params, idxs, owner_of, group, group_size)

    return results


class Muon(torch.optim.Optimizer):
    """
    Collective when the parameters are sharded. Keeps one fp32 momentum_buffer per parameter.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Orthogonalize all updates in one call so the broadcasts stay in lockstep across ranks.
        params, updates, hparams = [], [], []
        for group in self.param_groups:
            beta, lr, weight_decay = group["momentum"], group["lr"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                params.append(param)
                updates.append(self.compute_update(param, beta))
                hparams.append((lr, weight_decay))

        if params:
            orths = orthogonalized_updates(params, updates, steps=5)
            for param, orth, (lr, weight_decay) in zip(params, orths, hparams):
                self.apply_update(param, orth, lr, weight_decay)

        return loss

    def compute_update(self, param, beta):
        """
        Returns the Nesterov momentum update, in fp32.
        """
        state = self.state[param]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
        buf = state["momentum_buffer"]
        grad = param.grad.to(torch.float32)
        buf.lerp_(grad, 1 - beta)
        return grad.lerp(buf, beta)

    def apply_update(self, param, orth, lr, weight_decay):
        """
        orth is the full, unsharded update; a DTensor param takes its own shard of it.
        """
        if isinstance(param, DTensor):
            replicated = [Replicate()] * param.device_mesh.ndim
            orth = DTensor.from_local(
                orth, param.device_mesh, replicated, run_check=False
            ).redistribute(placements=param.placements)
        param.mul_(1 - lr * weight_decay)
        param.add_(orth, alpha=-lr)


def local_grads(model: torch.nn.Module) -> list[torch.Tensor]:
    grads = []
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad
        grads.append(grad._local_tensor if isinstance(grad, DTensor) else grad)
    return grads


@torch.no_grad()
def scale_grads_(grads: list[torch.Tensor], scale: torch.Tensor) -> None:
    """
    scale is a 0-d tensor. Bit-identical to grad.mul_(scale) on each gradient.
    """
    groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for grad in grads:
        groups.setdefault((grad.device, grad.dtype), []).append(grad)
    for group in groups.values():
        torch._foreach_mul_(group, scale)


@torch.no_grad()
def clip_grad_norm(
    model: torch.nn.Module, max_norm: float, norm_type: float = 2.0, hsdp_replica: int = 1
) -> torch.Tensor:
    """
    Clip the gradients in place to a global norm of max_norm over all ranks and return the total
    norm before clipping, a 0-d tensor. Collective over the world. At hsdp_replica above 1 each
    gradient element sits on that many ranks, so the summed power is divided by that count.

    Without a host sync: the gradients are always multiplied, by 1.0 (bit-preserving) when no
    clipping is due or the norm is NaN.
    """
    grads = local_grads(model)
    if not grads:
        first_param = next(model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    local_norm = torch.nn.utils.get_total_norm(grads, norm_type=norm_type)
    local_norm_sq = local_norm**norm_type
    torch.distributed.all_reduce(local_norm_sq, op=torch.distributed.ReduceOp.SUM)
    local_norm_sq = local_norm_sq / hsdp_replica
    total_norm = (local_norm_sq ** (1.0 / norm_type)).clamp(min=1e-6)
    clip_coef = (max_norm / total_norm).clamp(max=1.0)
    clip_coef = torch.where(clip_coef < 1.0, clip_coef, torch.ones_like(clip_coef))
    scale_grads_(grads, clip_coef)
    return total_norm
