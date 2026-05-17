"""PithTrain distributed module."""

import atexit
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Generator, Literal

import torch

from pithtrain.config import SlottedDefault


@dataclass(init=False, slots=True)
class DistributedCfg(SlottedDefault):
    """
    Configuration for distributed runtime.

    Parallelism degrees (PP, CP, EP), FSDP2 sharding strategy, and operation timeout. DP is
    inferred from the world size.
    """

    pipeline_parallel_size: int = 1
    """
    Degree of pipeline parallelism (PP).

    Partition the model layers across ranks; each rank holds a consecutive slice. Forward and
    backward execution is scheduled by DualPipeV.
    """

    context_parallel_size: int = 1
    """
    Degree of context parallelism (CP).

    Shard the sequence dimension across CP ranks. K/V exchange uses ring attention with a zigzag
    token layout.
    """

    expert_parallel_size: int = 1
    """
    Degree of expert parallelism (EP).

    Distribute the MoE experts across ranks; non-expert layers are unaffected. Token routing uses
    EP dispatch and combine kernels with token deduplication.
    """

    timeout: timedelta = timedelta(minutes=15)
    """
    Timeout for distributed operations.

    Applied to NCCL collectives and the watchdog heartbeat. Scale up for multi-node runs; keep
    small to fail fast.
    """

    sharding_strategy: Literal["fsdp", "hsdp"] = "fsdp"
    """
    FSDP2 sharding strategy.

    - "fsdp": shard parameters across the full FSDP mesh (dp x cp x ep for non-MoE; dp x cp
      for MoE experts). Lowest memory.
    - "hsdp": shard within the inner mesh (cp x ep for non-MoE; cp for MoE) and replicate
      across dp. Pick when one DP replica fits the model.
    """


@dataclass(init=False, slots=True)
class DistributedCtx:
    """
    Context for distributed runtime.

    Hold the torchrun ranks alongside the (PP, DP, CP, EP) device mesh, providing a single source
    of truth that the training loop, model constructors, and collectives reference.
    """

    rank: int
    """Global worker rank."""

    world_size: int
    """Total number of workers."""

    local_rank: int
    """Worker rank within the node."""

    local_world_size: int
    """Number of workers on the node."""

    device_mesh: torch.distributed.DeviceMesh
    """4D mesh over (PP, DP, CP, EP) axes."""

    pp_rank: int
    pp_size: int
    dp_rank: int
    dp_size: int
    cp_rank: int
    cp_size: int
    ep_rank: int
    ep_size: int


def setup_torch_runtime() -> None:
    """Apply torch runtime tuning: enable TF32 matmul and raise the dynamo recompile cap."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 64


def setup_default_process_group(cfg: DistributedCfg, ctx: DistributedCtx) -> None:
    """
    Initialize the default process group from torchrun environment variables.

    Read global/local rank info into ctx, apply NCCL env tuning, register cleanup at exit, and set
    the current CUDA device from the local rank.
    """
    assert torch.cuda.is_available(), "CUDA is not available."
    assert "TORCHELASTIC_RUN_ID" in os.environ, "Not launched with torchrun."

    ctx.rank = int(os.environ["RANK"])
    ctx.world_size = int(os.environ["WORLD_SIZE"])
    ctx.local_rank = int(os.environ["LOCAL_RANK"])
    ctx.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(int(cfg.timeout.total_seconds()))

    kwargs = dict()
    kwargs["backend"] = "nccl"
    kwargs["device_id"] = ctx.local_rank
    kwargs["timeout"] = cfg.timeout
    torch.distributed.init_process_group(**kwargs)
    atexit.register(torch.distributed.destroy_process_group)
    torch.cuda.set_device(ctx.local_rank)


def setup_failfast_excepthook() -> None:
    """
    Install a fail-fast excepthook that bypasses the NCCL drain on uncaught exceptions.

    Default torch.distributed shutdown can hang indefinitely while draining in-flight NCCL work
    that peers will never satisfy. Hard-exiting bypasses the drain so NCCL wor on other ranks
    fail fast instead of hanging.
    """
    original = sys.excepthook

    def excepthook(exc_type, exc_value, exc_tb, *_):
        try:
            original(exc_type, exc_value, exc_tb)
        except Exception:
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(1)

    sys.excepthook = excepthook
    threading.excepthook = lambda args: excepthook(*args)


def setup_device_mesh(cfg: DistributedCfg, ctx: DistributedCtx) -> None:
    """
    Build the (PP, DP, CP, EP) device mesh and read per-axis ranks and sizes into ctx.

    Mesh dimensions go outer-to-inner: PP, DP, CP, EP. CP and EP sit innermost so frequent
    collectives (ring K/V exchange, MoE all-to-all) stay within the NVLink domain.
    """
    ctx.pp_size = pp_size = cfg.pipeline_parallel_size
    ctx.cp_size = cp_size = cfg.context_parallel_size
    ctx.ep_size = ep_size = cfg.expert_parallel_size
    world_size = ctx.world_size

    divisor = pp_size * cp_size * ep_size
    if world_size % divisor != 0:
        raise RuntimeError(f"{world_size=} not divisible by {pp_size=} * {cp_size=} * {ep_size=}")
    ctx.dp_size = world_size // divisor

    kwargs = dict()
    kwargs["device_type"] = "cuda"
    kwargs["mesh_shape"] = (ctx.pp_size, ctx.dp_size, ctx.cp_size, ctx.ep_size)
    kwargs["mesh_dim_names"] = ("pp", "dp", "cp", "ep")
    ctx.device_mesh = torch.distributed.init_device_mesh(**kwargs)

    ctx.dp_rank = ctx.device_mesh.get_local_rank("dp")
    ctx.pp_rank = ctx.device_mesh.get_local_rank("pp")
    ctx.cp_rank = ctx.device_mesh.get_local_rank("cp")
    ctx.ep_rank = ctx.device_mesh.get_local_rank("ep")


@contextmanager
def distributed_context(cfg: object, ctx: object) -> Generator[DistributedCtx, None, None]:
    """Context manager for distributed runtime."""
    assert hasattr(cfg, "distributed") and isinstance(cfg.distributed, DistributedCfg)
    assert hasattr(ctx, "distributed") and isinstance(ctx.distributed, DistributedCtx)
    setup_torch_runtime()
    setup_default_process_group(cfg.distributed, ctx.distributed)
    setup_failfast_excepthook()
    setup_device_mesh(cfg.distributed, ctx.distributed)
    yield ctx.distributed
