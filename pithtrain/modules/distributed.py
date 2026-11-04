import atexit
import os
import sys
from dataclasses import dataclass
from datetime import timedelta

import torch

from pithtrain.config import SlottedDefault
from pithtrain.contexts import distributed

__all__ = [
    "DistributedCfg",
    "setup_torch_runtime",
    "bind_cpu_affinity",
    "setup_default_process_group",
    "setup_device_mesh",
    "setup_distributed",
]


@dataclass(init=False, slots=True)
class DistributedCfg(SlottedDefault):
    pipeline_parallel_size: int = 1
    """
    Degree of pipeline parallelism (PP).

    Split the model layers across ranks, two V-shaped chunks per rank, scheduled by DualPipeV.
    """

    context_parallel_size: int = 1
    """
    Degree of context parallelism (CP).

    Shard the sequence across CP ranks in a zigzag layout; attention exchanges K/V in a ring.
    """

    expert_parallel_size: int = 1
    """
    Degree of expert parallelism (EP).

    Distribute the routed experts across ranks; tokens reach them through all-to-all dispatch and
    combine. Non-expert layers are unaffected.
    """

    timeout: timedelta = timedelta(minutes=15)
    """
    Timeout of every collective, passed to init_process_group.
    """

    hsdp_replica: int = 1
    """
    Number of replicas each FSDP shard group is split into.

    At 1, FSDP shards the attention parameters over the dp x cp stage and the expert parameters
    over the dp axis of the expert view. Above 1, both groups split into this many replicas (both
    must divide by it): FSDP shards within one replica and all-reduces across them, trading memory
    for a cheaper gradient reduction.
    """


def setup_torch_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 64


def parse_cpulist(text: str) -> list[int]:
    cpus = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def bind_cpu_affinity(device_id: int) -> None:
    """
    Pin this process to the CPUs local to its GPU. Call it before any thread starts, so every
    thread inherits the set. A process already restricted to a smaller set (srun --cpu-bind,
    numactl, taskset) keeps its binding.
    """
    p = torch.cuda.get_device_properties(device_id)
    path = "/sys/bus/pci/devices/%04x:%02x:%02x.0/local_cpulist" % (
        p.pci_domain_id, p.pci_bus_id, p.pci_device_id
    )  # fmt: skip
    try:
        with open(path) as f:
            cpus = parse_cpulist(f.read())
    except OSError:
        return
    if not cpus or not set(cpus) <= os.sched_getaffinity(0):
        return
    os.sched_setaffinity(0, cpus)


def setup_default_process_group(cfg: DistributedCfg, device_id: int) -> None:
    """
    The excepthook hard-exits on an uncaught exception, since a collective teardown would wait on
    peers that already died.
    """
    kwargs = dict(backend="nccl", device_id=device_id, timeout=cfg.timeout)
    torch.distributed.init_process_group(**kwargs)
    atexit.register(torch.distributed.destroy_process_group)

    original = sys.excepthook

    def excepthook(exc_type, exc_value, exc_tb):
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


def setup_device_mesh(cfg: DistributedCfg, device_id: int) -> None:
    """
    MoE parallel folding (https://arxiv.org/abs/2504.14960): attention and the experts get separate
    meshes over the same ranks, and the cp and ep groups are each a contiguous rank block.
    """
    distributed.rank = torch.distributed.get_rank()
    distributed.world_size = torch.distributed.get_world_size()
    distributed.device = torch.device("cuda", device_id)
    torch.cuda.set_device(distributed.device)

    pp_size = cfg.pipeline_parallel_size
    cp_size = cfg.context_parallel_size
    ep_size = cfg.expert_parallel_size

    world_size = distributed.world_size
    if world_size % pp_size != 0:
        raise RuntimeError(f"{world_size=} not divisible by {pp_size=}")
    stage_size = world_size // pp_size
    if stage_size % cp_size != 0:
        raise RuntimeError(f"{stage_size=} (world_size // pp_size) not divisible by {cp_size=}")
    if stage_size % ep_size != 0:
        raise RuntimeError(f"{stage_size=} (world_size // pp_size) not divisible by {ep_size=}")
    attn_dp_size = stage_size // cp_size
    expt_dp_size = stage_size // ep_size

    # Both meshes build a pp communicator; only the one on attn_mesh is used.
    init = torch.distributed.init_device_mesh
    attn_mesh = init("cuda", (pp_size, attn_dp_size, cp_size), mesh_dim_names=("pp", "dp", "cp"))
    expt_mesh = init("cuda", (pp_size, expt_dp_size, ep_size), mesh_dim_names=("pp", "dp", "ep"))
    distributed.attn_mesh, distributed.expt_mesh = attn_mesh, expt_mesh

    distributed.pp_size, distributed.pp_rank = pp_size, attn_mesh.get_local_rank("pp")
    distributed.pp_group = attn_mesh.get_group("pp")

    distributed.cp_size, distributed.cp_rank = cp_size, attn_mesh.get_local_rank("cp")
    distributed.cp_group = attn_mesh.get_group("cp")

    distributed.ep_size, distributed.ep_rank = ep_size, expt_mesh.get_local_rank("ep")
    distributed.ep_group = expt_mesh.get_group("ep")

    # No dp process group: FSDP reduces over a mesh. Only the attention dp decides what a rank
    # loads, so only it is published.
    distributed.dp_size, distributed.dp_rank = attn_dp_size, attn_mesh.get_local_rank("dp")


def setup_distributed(cfg: object) -> None:
    assert hasattr(cfg, "distributed") and isinstance(cfg.distributed, DistributedCfg)
    setup_torch_runtime()
    device_id = int(os.environ["LOCAL_RANK"])
    bind_cpu_affinity(device_id)
    setup_default_process_group(cfg.distributed, device_id)
    setup_device_mesh(cfg.distributed, device_id)
