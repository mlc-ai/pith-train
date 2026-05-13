"""Spawn workers and run them under PithTrain's distributed context."""

import os
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from torch.multiprocessing.spawn import spawn

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx, distributed_context

# Snapshot launcher-provided values at module load. Within a single pytest
# session, multiple launch calls would otherwise pollute each other's env via
# setdefault no-ops when mesh_extent differs between parametrizations.
LAUNCHER_WORLD_SIZE = os.environ.get("WORLD_SIZE")
LAUNCHER_LOCAL_WORLD_SIZE = os.environ.get("LOCAL_WORLD_SIZE")


def cosine_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Return 1 - cosine similarity: 0 if a and b point the same direction, 2 if
    opposite. Scale-invariant, so magnitude shifts in low-precision do not inflate it.
    """
    a, b = a.double().flatten(), b.double().flatten()
    return float(1.0 - torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def entrypoint(i: int, cfg: DistributedCfg, worker: Callable, *args) -> None:
    node_rank = int(os.environ["NODE_RANK"])
    world_local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    os.environ["RANK"] = str(node_rank * world_local_size + i)
    os.environ["LOCAL_RANK"] = str(i)

    parent_cfg = SimpleNamespace(distributed=cfg)
    parent_ctx = SimpleNamespace(distributed=DistributedCtx())
    with distributed_context(parent_cfg, parent_ctx) as ctx:
        worker(ctx, *args)


def launch(cfg: DistributedCfg, worker: Callable, *args) -> None:
    """
    Spawn workers and call worker(ctx, *args) inside each. Skip the test if
    the distributed runtime cannot provide pp * cp * ep ranks.
    """
    mesh_extent = 1
    mesh_extent *= cfg.pipeline_parallel_size
    mesh_extent *= cfg.context_parallel_size
    mesh_extent *= cfg.expert_parallel_size

    os.environ["WORLD_SIZE"] = LAUNCHER_WORLD_SIZE or str(mesh_extent)
    os.environ["LOCAL_WORLD_SIZE"] = LAUNCHER_LOCAL_WORLD_SIZE or os.environ["WORLD_SIZE"]
    os.environ.setdefault("NODE_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "15213")
    os.environ.setdefault("TORCHELASTIC_RUN_ID", "pytest")

    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    if world_size < mesh_extent:
        pytest.skip(f"require {mesh_extent} ranks, got {world_size}")
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(f"require {local_world_size} GPUs, got {torch.cuda.device_count()}")
    if world_size % mesh_extent != 0:
        raise ValueError(f"{world_size=} not divisible by {mesh_extent=}")
    if world_size % local_world_size != 0:
        raise ValueError(f"{world_size=} not divisible by {local_world_size=}")

    spawn(entrypoint, args=(cfg, worker, *args), nprocs=local_world_size)
