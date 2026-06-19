"""
Muon optimizer: SGD momentum, then orthogonalize the update with a 5-step
Newton-Schulz iteration.

Only 2D hidden weights are Muon-eligible (:func:`is_muon_param`); embeddings,
the LM head, norms, biases, and the MoE router go to a separate AdamW.
:func:`partition_muon_params` splits a model into the two lists; the training
setup composes ``[Muon, AdamW]`` and steps/schedules/checkpoints them together.

Under FSDP2 a weight is a sharded DTensor and Newton-Schulz needs the full
matrix. Instead of every rank gathering and orthogonalizing every weight
(duplicating the compute G times, G = FSDP group size),
:func:`orthogonalized_updates` round-robins the weights across the mesh so
each rank does ~1/G of them and broadcasts the result -- a bit-identical update
(NS is deterministic on the gathered matrix), faster as G grows. G==1 weights
are done locally.

The update is scaled by Moonlight's ``0.2 * sqrt(max(n, m))`` so its RMS
matches AdamW's and both share one LR. ``zeropower_via_newtonschulz5`` is
verbatim from the public reference (mirrored by DeepSpeed).
"""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization, batched over leading dims, bf16.

    Drives singular values toward ~1 (not exactly UV^T, but close enough). The
    transpose handles tall matrices; the Frobenius prescale keeps spectral norm
    <= 1.
    """
    assert G.ndim >= 2  # batches over leading dims (e.g. 3D stacked experts)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic update term
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_scale_factor(n: int, m: int) -> float:
    """Moonlight scale ``0.2 * sqrt(max(n, m))``: brings the orthogonalized
    update's RMS to ~0.2 (~ an AdamW step) so Muon and AdamW share one LR."""
    return 0.2 * max(n, m) ** 0.5


# ---------------------------------------------------------------------------
# Distributed orthogonalization (parameter-parallel Newton-Schulz)
#
# Round-robin the weights across each one's FSDP mesh so the compute-bound NS
# is shared, not duplicated on every rank. Within a chunk: gather every weight,
# then each rank orthogonalizes only the ones it owns, then broadcast the
# results -- with no collective between the orthogonalizations, so they run
# concurrently across ranks instead of being serialized. Chunking bounds memory.
# ---------------------------------------------------------------------------


def _orthogonalize(full, dtype, steps):
    orth = zeropower_via_newtonschulz5(full, steps=steps).to(dtype)
    return (orth * muon_scale_factor(full.size(-2), full.size(-1))).contiguous()


def _broadcast_owned(results, updates, params, idxs, owner_of, group, group_size):
    """Broadcast the per-owner results so every rank ends up with all of them.
    Each rank concatenates the weights it owns into one buffer and broadcasts it
    once; the others pre-size the buffer from each weight's ``.shape``/dtype and
    slice it back out (N -> G collectives). Assumes one dtype per mesh group."""
    group_rank = dist.get_rank(group)
    for owner in range(group_size):
        owned = [i for i in idxs if owner_of[i] == owner]
        if not owned:
            continue
        shapes = [tuple(updates[i].shape) for i in owned]
        numels = [math.prod(s) for s in shapes]
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
    """Orthogonalize each ``updates[i]`` (post-momentum, DTensor or plain),
    distributing the NS across each weight's 1-D FSDP mesh. Returns the full
    orthogonalized+scaled tensor per weight (in ``params[i].dtype``); the caller
    reshards/applies it."""
    results = [None] * len(updates)

    # Bucket sharded (G>1) weights by mesh group; the rest are local.
    buckets: dict[int, tuple] = {}  # group id -> (group, G, rank, idxs)
    local = []
    for i, u in enumerate(updates):
        if isinstance(u, DTensor) and u.device_mesh.ndim == 1:
            group = u.device_mesh.get_group()
            if dist.get_world_size(group) > 1:
                key = id(group)
                if key not in buckets:
                    buckets[key] = (group, dist.get_world_size(group), dist.get_rank(group), [])
                buckets[key][3].append(i)
                continue
        local.append(i)

    # Local weights (plain or G==1): no duplication, orthogonalize in place.
    for i in local:
        u = updates[i]
        full = u.full_tensor() if isinstance(u, DTensor) else u
        results[i] = _orthogonalize(full, params[i].dtype, steps)

    for group, group_size, group_rank, idxs in buckets.values():
        chunk = chunk_size or max(4 * group_size, 8)
        owner_of = {idx: pos % group_size for pos, idx in enumerate(idxs)}
        for start in range(0, len(idxs), chunk):
            window = idxs[start : start + chunk]
            # Gather the chunk; each rank orthogonalizes the weights it owns.
            fulls = {i: updates[i].full_tensor() for i in window}
            for i in window:
                if owner_of[i] == group_rank:
                    results[i] = _orthogonalize(fulls[i], params[i].dtype, steps)
            fulls.clear()
        # Broadcast so every rank ends up with all the orthogonalized weights.
        _broadcast_owned(results, updates, params, idxs, owner_of, group, group_size)

    return results


# ---------------------------------------------------------------------------
# The Muon optimizer
# ---------------------------------------------------------------------------


class Muon(torch.optim.Optimizer):
    """Muon for 2D (and batched-3D-expert) hidden weights. One fp32
    ``momentum_buffer`` per param; each step is momentum -> Newton-Schulz ->
    scale. Pass only Muon-eligible params (:func:`is_muon_param`)."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute every update first, then orthogonalize the batch together so
        # the per-mesh round-robin and broadcasts stay in lockstep across ranks.
        params, updates, hparams = [], [], []
        for group in self.param_groups:
            beta, lr, wd = group["momentum"], group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                params.append(p)
                updates.append(self._compute_update(p, beta))
                hparams.append((lr, wd))

        if params:
            orths = orthogonalized_updates(params, updates, steps=5)
            for p, orth, (lr, wd) in zip(params, orths, hparams):
                self._apply(p, orth, lr, wd)

        return loss

    def _compute_update(self, p, beta):
        """SGD momentum on the local shard (no comm); returns the pre-NS update.
        Grad is cast to fp32 to match the fp32 momentum buffer (grads may be
        bf16)."""
        state = self.state[p]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
        buf = state["momentum_buffer"]
        grad = p.grad.to(torch.float32)
        buf.lerp_(grad, 1 - beta)
        return grad.lerp(buf, beta)

    def _apply(self, p, orth, lr, wd):
        """Decoupled weight decay + the step. ``orth`` is the full update;
        reshard to ``p``'s placements (a local narrow) when ``p`` is sharded."""
        if isinstance(p, DTensor):
            replicated = [Replicate()] * p.device_mesh.ndim
            orth = DTensor.from_local(
                orth, p.device_mesh, replicated, run_check=False
            ).redistribute(placements=p.placements)
        p.mul_(1 - lr * wd)
        p.add_(orth, alpha=-lr)


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------


def is_muon_param(name: str, param: torch.Tensor) -> bool:
    """True if ``param`` is a 2D hidden weight Muon should optimize.

    Muon: attention q/k/v/o and MLA projections, dense/shared-expert
    gate/up/down, and the stacked 3D expert weights. AdamW (False): all 1D
    params (norms, biases, sinks), embeddings, the LM head, the MoE gate/router,
    and the 2D stacked expert biases (caught by the ``_bias`` check).
    """
    if param.ndim < 2:
        return False
    if name.endswith(".bias") or name.endswith("_bias"):
        return False
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return False
    if ".gate.weight" in name or ".router.weight" in name:
        return False
    return True


def partition_muon_params(model: torch.nn.Module):
    """Split ``model``'s requires-grad params into ``(muon_params, aux_params)``
    via :func:`is_muon_param`. Asserts the Muon list is non-empty."""
    muon_params, aux_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (muon_params if is_muon_param(name, param) else aux_params).append(param)

    assert muon_params, (
        "Muon selected no parameters; check the model classification in is_muon_param."
    )
    return muon_params, aux_params
