"""PithTrain distributed module."""

import atexit
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Generator, Literal

import torch

from pithtrain.config import SlottedDefault
from pithtrain.modules import shutdown


@dataclass(init=False, slots=True)
class DistributedCfg(SlottedDefault):
    """Configuration for distributed runtime."""

    pipeline_parallel_size: int = 1
    """
    Degree of pipeline parallelism.

    Pipeline Parallelism (PP) is a technique that assigns consecutive layers or segments of a neural network
    to different GPUs. This division allows each GPU to process different stages of the network sequentially.

    For example, if a model has 12 layers and the pipeline_parallel_size is set to 4, then each GPU will
    handle 3 layers.
    """

    context_parallel_size: int = 1
    """
    Degree of context parallelism.

    Context Parallelism (CP) splits the sequence dimension across GPUs. Each GPU processes a chunk
    of the full sequence. Ring attention is used to compute full causal attention across the
    distributed sequence chunks, passing K/V around a ring of CP ranks.
    """

    expert_parallel_size: int = 1
    """
    Degree of expert parallelism.

    Expert Parallelism (EP) is a type of model parallelism that distributes experts of an MoE across GPUs.
    Unlike other model-parallel techniques, EP is applied to only the expert layers thus does not impact
    the parallel mapping of the rest of the layers.

    For example, if the model has 8 experts, then setting expert_parallel_size to 4 results in each GPU
    processing 2 experts. The number of experts should be divisible by the expert parallel size.
    """

    nccl_timeout_seconds: int = 180
    """
    Timeout for NCCL collective operations and watchdog heartbeat, in seconds.

    Bounds how long a rank waits on a hung collective before NCCL's watchdog
    aborts the job. Applies both to the per-collective timeout passed to
    ``torch.distributed.init_process_group`` and to ``TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC``.

    Scale up for large multi-node or deep-pipeline runs where first-iteration
    setup (NCCL channel build, FSDP all-gather, checkpoint load) can be slow.
    Keep small for single-node runs to fail fast.
    """

    sharding_strategy: Literal["fsdp", "hsdp"] = "fsdp"
    """
    Sharding strategy for FSDP2.

    - ``"fsdp"`` (default): Fully Sharded Data Parallel. Shards weights, gradients,
      and optimizer states across the full data-parallel mesh (``dp x cp x ep``
      for non-MoE parameters; ``dp x cp`` for MoE expert weights). Maximizes
      memory savings.

    - ``"hsdp"``: Hybrid Sharded Data Parallel. Shards within the inner mesh
      dimension (``cp x ep`` for non-MoE; ``cp`` for MoE) and replicates across
      the ``dp`` dimension. Lower memory savings than FSDP, but uses a smaller
      FSDP group. Useful when the model fits comfortably in a single DP replica.
    """


@dataclass(init=False, slots=True)
class DistributedCtx:
    """Context for distributed runtime."""

    rank: int
    """Global rank of the current process."""

    world_size: int
    """Total number of workers in the distributed job."""

    local_rank: int
    """Local rank of the current process on the node."""

    local_world_size: int
    """Number of workers on the current node."""

    dp_rank: int
    """Rank of the current process in the data parallel group."""

    dp_size: int
    """Size of the data parallel group."""

    pp_rank: int
    """Rank of the current process in the pipeline parallel group."""

    pp_size: int
    """Size of the pipeline parallel group."""

    cp_rank: int
    """Rank of the current process in the context parallel group."""

    cp_size: int
    """Size of the context parallel group."""

    ep_rank: int
    """Rank of the current process in the expert parallel group."""

    ep_size: int
    """Size of the expert parallel group."""

    device_mesh: torch.distributed.DeviceMesh
    """Collection of process groups for multi-dimensional parallelism."""


def setup_default_process_group(cfg: DistributedCfg, ctx: DistributedCtx) -> None:
    """
    Setup the default process group.

    This function initializes the default process group using environment variables by torchrun.
    It also sets the current CUDA device based on the LOCAL_RANK environment variable. A cleanup
    function is registered to destroy the process group at program exit.
    """
    assert torch.cuda.is_available(), "CUDA is not available."
    assert "TORCHELASTIC_RUN_ID" in os.environ, "Not launched with torchrun."

    ctx.rank = int(os.environ["RANK"])
    ctx.world_size = int(os.environ["WORLD_SIZE"])
    ctx.local_rank = int(os.environ["LOCAL_RANK"])
    ctx.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    shutdown.set_heartbeat_timeout(cfg.nccl_timeout_seconds)
    kwargs = dict()
    kwargs["backend"] = "nccl"
    kwargs["device_id"] = ctx.local_rank
    kwargs["timeout"] = timedelta(seconds=cfg.nccl_timeout_seconds)
    torch.distributed.init_process_group(**kwargs)
    # See pithtrain.modules.shutdown for why os._exit(1), not destroy/abort.
    shutdown.install_failfast_excepthook()
    atexit.register(torch.distributed.destroy_process_group)
    torch.cuda.set_device(ctx.local_rank)


def setup_device_mesh(cfg: DistributedCfg, ctx: DistributedCtx) -> None:
    """
    Setup the device mesh.

    Process groups are created in the following order. EP and CP are the inner-most
    dimensions to keep their frequent communications within the NVLink domain.

    Mesh shape: ``(PP, DP, CP, EP)``

    1. Pipeline Parallel (PP) - outermost
    2. Data Parallel (DP)
    3. Context Parallel (CP) - ring attention KV exchange
    4. Expert Parallel (EP) - innermost, MoE all-to-all
    """
    ctx.ep_size = cfg.expert_parallel_size
    ctx.pp_size = cfg.pipeline_parallel_size
    ctx.cp_size = cfg.context_parallel_size
    ctx.dp_size = ctx.world_size // (ctx.ep_size * ctx.pp_size * ctx.cp_size)

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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 64
    setup_default_process_group(cfg.distributed, ctx.distributed)
    setup_device_mesh(cfg.distributed, ctx.distributed)
    yield ctx.distributed
