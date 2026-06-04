"""PithTrain training module."""

import gc
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal, Optional, Union

import numpy as np
import torch
import torch.distributed.fsdp
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from transformers import AutoConfig

from pithtrain.config import SlottedDefault
from pithtrain.dualpipe import DualPipeV, set_p2p_tensor_dtype, set_p2p_tensor_shapes
from pithtrain.models.deepseek_v2_lite import DeepseekV2LiteModel
from pithtrain.models.gpt_oss import GptOssModel
from pithtrain.models.qwen3_moe import Qwen3MoeModel
from pithtrain.modules.dataset import ConcatDataset, MemmapDataset
from pithtrain.modules.load_balance import make_load_balance_loss_fn

from .distributed import DistributedCfg, DistributedCtx


@dataclass(init=False, slots=True)
class TrainingCfg(SlottedDefault):
    dataset: Path
    """The root directory hosting the tokenized dataset."""

    sequence_length: int
    """The sequence length for each training sample."""

    seed: int = 1234
    """The random seed for reproducibility."""

    min_lr: float
    """The minimum learning rate to start with and decay to."""

    max_lr: float
    """The maximum learning rate."""

    warmup_steps: int
    """The number of steps for linear warmup of the learning rate."""

    max_steps: int
    """The maximum number of training steps."""

    micro_batch_size: int
    """The size of each micro-batch used during training."""

    global_batch_size: int
    """
    The size of the global batch used during training.

    Gradients will be accumulated over multiple micro-batches to achieve this batch size.
    """

    optimizer: Literal["Adam"]
    """The optimizer to use during training."""

    scheduler: Literal["CosineAnnealing", "Constant"]
    """The learning rate scheduler to use after linear warmup."""

    model: Union[
        Path,
        Literal[
            "deepseek-ai/DeepSeek-V2-Lite",
            "Qwen/Qwen3-30B-A3B",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ],
    ]
    """
    The model to use for training. Can be a HuggingFace model ID
    (e.g. ``"Qwen/Qwen3-30B-A3B"``) or a local path to a config JSON file
    (e.g. ``"examples/pretrain_lm/qwen3-30b-a3b/config.json"``).
    """

    save_interval: Optional[int] = None
    """
    The interval (in steps) at which to save checkpoints. When None,
    checkpoint saving is disabled but loading still occurs from
    ``save_location`` (if set). This is useful for validation runs
    that need to load a pretrained checkpoint without writing new ones.
    """

    save_location: Optional[Path] = None
    """
    The directory for checkpoint storage. Checkpoints are loaded from
    and saved to ``<save_location>/torch-dcp/step-XXXXXXXX``. When
    None, both loading and saving are disabled and the model trains
    from scratch.
    """

    moe_load_balance_coef: float = 0.0
    """
    Coefficient for the MoE load balance loss.
    Set to 0 to disable. Typical values are 1e-2 to 1e-1.
    """

    moe_load_balance_type: Literal["micro-batch", "global-batch", "sequence"] = "micro-batch"
    """
    Load balance loss strategy for MoE layers.

    * "micro-batch" - Micro-batch loss computed per micro-batch
      (https://arxiv.org/abs/2101.03961).
    * "global-batch" - Global-batch loss that synchronises expert selection
      frequencies across DP x EP ranks and accumulates across gradient
      accumulation steps (https://arxiv.org/abs/2501.11873).
    * "sequence" - Sequence-level loss computed independently per sequence
      then averaged over the batch (https://arxiv.org/abs/2405.04434).
    """

    fp8_training: Literal["deep-gemm", "disabled"] = "disabled"
    """
    FP8 training backend: ``"disabled"`` (BF16 only) or ``"deep-gemm"`` (128-element
    block scaling via DeepGEMM). Supports SM90 (Hopper) and SM100+ (Blackwell).
    """

    init_std: float = 0.02
    """
    Standard deviation for weight initialization.
    Input layers use N(0, init_std). Output layers use N(0, init_std / sqrt(2 * num_layers)).
    """

    nsys_start: Optional[int] = None
    """
    Training step at which to start the CUDA profiler (for Nsight Systems).

    The profiler starts at the beginning of this step. Set to ``None`` to disable.
    """

    nsys_stop: Optional[int] = None
    """
    Training step at which to stop the CUDA profiler (for Nsight Systems).

    The profiler stops at the beginning of this step, so this step and subsequent
    steps are not profiled. To profile a single step `N`, set `nsys_start=N` and
    `nsys_stop=N+1`. Set to ``None`` to disable.
    """

    memory_profile_start: Optional[int] = None
    """
    Training step at which to start recording CUDA memory allocation history.

    When set, ``torch.cuda.memory._record_memory_history`` is called at the
    beginning of this step with full stack traces for both allocations and frees.
    Set to ``None`` to disable.
    """

    memory_profile_stop: Optional[int] = None
    """
    Training step at which to stop recording and dump the memory snapshot.

    At the beginning of this step the recorded history is dumped to
    ``memory_profile_output`` and recording is disabled. To profile a single
    step ``N``, set ``memory_profile_start=N`` and ``memory_profile_stop=N+1``.
    Set to ``None`` to disable.
    """

    memory_profile_output: Path = Path.cwd()
    """
    Output directory for the CUDA memory snapshot. Each rank writes a pickle
    file named ``snapshot-rank00000.pickle`` etc. into this directory.
    The snapshot can be visualized at https://pytorch.org/memory_viz.
    """


@dataclass(init=False, slots=True)
class TrainingCtx:
    dataset: ConcatDataset
    """The concatenated dataset for training."""

    model: DualPipeV
    """The model being trained."""

    optimizer: Optimizer
    """The optimizer used for training."""

    scheduler: LRScheduler
    """The learning rate scheduler used for training."""

    step: int
    """The current training step."""


def setup_dataset(cfg: TrainingCfg, ctx: TrainingCtx) -> None:
    memmap_datasets = []
    for file in sorted(cfg.dataset.rglob("*.bin")):
        memmap_datasets.append(MemmapDataset(file, cfg.sequence_length))
    ctx.dataset = ConcatDataset(memmap_datasets, cfg.seed)


def init_weights(model: nn.Module, num_layers: int, init_std: float = 0.02) -> None:
    """
    Apply scaled normal weight initialization.

    * **Input layers** (embedding, QKV projections, gate/up projections,
      MoE gate, lm_head): ``N(0, init_std)``
    * **Output layers** (attention output projection ``o_proj``, MLP/expert
      down projection ``down_proj``): ``N(0, init_std / sqrt(2 * num_layers))``
    * **1-D parameters** (layer-norm weights, biases): left unchanged.

    Parameters
    ----------
    model : nn.Module
        A single pipeline-stage module (e.g. ``DeepseekV2LiteModel``).
    num_layers : int
        Total number of transformer layers in the *full* model (not just this
        stage).  Used to compute the output-layer scaling factor.
    init_std : float
        Standard deviation for input-layer initialisation (default ``0.02``).
    """
    # Scale down residual-stream projections (o_proj, down_proj) to bound variance growth.
    output_std = init_std / math.sqrt(2.0 * num_layers)
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # skip biases, layer-norm weights, etc.
        if "o_proj" in name or "down_proj" in name:
            torch.nn.init.normal_(param, mean=0.0, std=output_std)
        else:
            torch.nn.init.normal_(param, mean=0.0, std=init_std)


def apply_fsdp(
    model,
    mesh: DeviceMesh,
    sharding_strategy: Literal["fsdp", "hsdp"] = "fsdp",
):
    # MoE params: unique per EP rank, replicated across DP x CP.
    # Non-MoE params: replicated across DP x CP x EP.
    # FSDP shards along the replicated dims:
    #   "fsdp": 1D mesh; FSDP2 shards across all participants.
    #   "hsdp": 2D mesh; FSDP2 shards along the inner dim and replicates
    #           along the outer (dp) dim. For non-MoE, cp and ep are folded
    #           into a single inner shard dim via _concatenate.
    if sharding_strategy == "fsdp":
        moe_fsdp_mesh = mesh["dp", "cp"]._flatten()
        other_fsdp_mesh = mesh["dp", "cp", "ep"]._flatten()
    elif sharding_strategy == "hsdp":
        moe_fsdp_mesh = mesh["dp", "cp"]
        cp_ep_mesh = mesh["cp", "ep"]._flatten("cp_ep")
        other_fsdp_mesh = DeviceMesh._concatenate([mesh["dp"], cp_ep_mesh])
    else:
        raise ValueError(f"Unknown sharding_strategy: {sharding_strategy!r}")
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    # FSDP recommends shard models from the bottom to the top.
    for i in range(2):
        assert isinstance(model[i], (DeepseekV2LiteModel, GptOssModel, Qwen3MoeModel))
        if model[i].embed_tokens is not None:
            fully_shard(
                model[i].embed_tokens,
                mesh=other_fsdp_mesh,
                reshard_after_forward=True,
                mp_policy=mp,
            )
        if model[i].norm is not None:
            assert model[i].lm_head is not None
            fully_shard(
                model[i].norm,
                mesh=other_fsdp_mesh,
                reshard_after_forward=True,
                mp_policy=mp,
            )
            fully_shard(
                model[i].lm_head,
                mesh=other_fsdp_mesh,
                reshard_after_forward=True,
                mp_policy=mp,
            )
        for layer in model[i].layers.values():
            if hasattr(layer.mlp, "experts"):
                fully_shard(
                    layer.mlp.experts,
                    mesh=moe_fsdp_mesh,
                    reshard_after_forward=False,
                    mp_policy=mp,
                )
            fully_shard(layer, mesh=other_fsdp_mesh, reshard_after_forward=False, mp_policy=mp)
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_attn")
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_mlp")
            torch.distributed.fsdp.register_fsdp_forward_method(layer, "forward_aggregate")
        fully_shard(model[i], mesh=other_fsdp_mesh, reshard_after_forward=False, mp_policy=mp)
    return model


def setup_model(
    cfg: TrainingCfg,
    ctx: TrainingCtx,
    distributed_cfg: DistributedCfg,
    distributed: DistributedCtx,
) -> None:
    from pithtrain.dualpipe.utils import FP8WeightCacheControl
    from pithtrain.layers.factory import ModelImplMode

    ModelImplMode.fp8_training = cfg.fp8_training
    if cfg.fp8_training != "disabled":
        FP8WeightCacheControl.enabled = True

    if ModelImplMode.fp8_training == "deep-gemm":
        try:
            import deep_gemm  # noqa: F401
        except ImportError:
            raise ImportError(
                "fp8_training='deep-gemm' requires the 'deep-gemm' package. "
                "Install it by running: uv sync"
            )
    elif ModelImplMode.fp8_training != "disabled":
        raise ValueError(
            f"Invalid fp8_training={cfg.fp8_training!r}. Expected one of: 'disabled', 'deep-gemm'."
        )

    pp_size = distributed.pp_size
    pp_rank = distributed.pp_rank
    cp_size = distributed.cp_size
    ep_size = distributed.ep_size

    device_mesh = distributed.device_mesh
    pp_group = device_mesh.get_group("pp")
    cp_group = device_mesh.get_group("cp") if cp_size > 1 else None
    ep_group = device_mesh.get_group("ep")

    modules = []
    module_config = AutoConfig.from_pretrained(cfg.model)
    module_config.ep_size = ep_size
    assert hasattr(module_config, "hidden_size")
    assert isinstance(module_config.hidden_size, int)
    if cfg.sequence_length % (2 * cp_size) != 0:
        raise ValueError(
            f"sequence_length ({cfg.sequence_length}) must be divisible by "
            f"2 * context_parallel_size ({2 * cp_size}); zigzag ring attention "
            f"splits the sequence into 2*cp_size equal chunks"
        )

    hidden_size = module_config.hidden_size

    if module_config.model_type == "deepseek_v2":
        ModelClass = DeepseekV2LiteModel
        model_kwargs = {"cp_group": cp_group}
    elif module_config.model_type == "qwen3_moe":
        ModelClass = Qwen3MoeModel
        model_kwargs = {"cp_group": cp_group}
    elif module_config.model_type == "gpt_oss":
        ModelClass = GptOssModel
        model_kwargs = {"cp_group": cp_group}
    else:
        raise ValueError(f"Unsupported model_type: {module_config.model_type}")

    modules.append(
        ModelClass(module_config, pp_size * 2, pp_rank, ep_group=ep_group, **model_kwargs)
    )
    modules.append(
        ModelClass(
            module_config, pp_size * 2, pp_size * 2 - 1 - pp_rank, ep_group=ep_group, **model_kwargs
        )
    )

    # Apply scaled normal weight initialization before FSDP sharding.
    num_layers = module_config.num_hidden_layers
    for module in modules:
        init_weights(module, num_layers, cfg.init_std)

    modules = nn.Sequential(*modules)
    apply_fsdp(modules, device_mesh, distributed_cfg.sharding_strategy)

    local_seq_len = cfg.sequence_length // cp_size
    # sequence_length = cfg.sequence_length, TODO this is kept here for stripe context parallelism
    micro_batch_size = cfg.micro_batch_size

    # Propagate MoE load balance loss to gate modules.
    if cfg.moe_load_balance_coef > 0:
        dp_ep_group = device_mesh["dp", "ep"]._flatten().get_group()
        for i in range(2):
            for layer in modules[i].layers.values():
                gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
                if gate is not None:
                    loss_fn = make_load_balance_loss_fn(
                        cfg.moe_load_balance_type,
                        cfg.moe_load_balance_coef,
                        dp_ep_group,
                        sequence_length=local_seq_len,
                        cp_group=cp_group,
                    )
                    if hasattr(loss_fn, "init_buffers"):
                        loss_fn.init_buffers(gate.num_experts, gate.weight.device)
                    gate.load_balance_loss_fn = loss_fn
                    if cp_group is not None:
                        gate.compute = gate.compute.__wrapped__.__get__(gate, type(gate))

    ctx.model = DualPipeV(modules, pp_group=pp_group, ep_group=ep_group)
    set_p2p_tensor_shapes([(micro_batch_size, local_seq_len, hidden_size)])
    set_p2p_tensor_dtype(torch.bfloat16)


def setup_optimizer(cfg: TrainingCfg, ctx: TrainingCtx) -> None:
    model, max_lr = ctx.model, cfg.max_lr
    ctx.optimizer = Adam(model.parameters(), lr=max_lr)


def setup_scheduler(cfg: TrainingCfg, ctx: TrainingCtx) -> None:
    min_lr, max_lr = cfg.min_lr, cfg.max_lr
    warmup_steps, max_steps = cfg.warmup_steps, cfg.max_steps
    warmup = LinearLR(ctx.optimizer, min_lr / max_lr, 1.0, warmup_steps)
    match cfg.scheduler:
        case "CosineAnnealing":
            stable = CosineAnnealingLR(ctx.optimizer, max_steps - warmup_steps, min_lr)
        case "Constant":
            stable = LinearLR(ctx.optimizer, 1.0, 1.0, max_steps - warmup_steps)
        case _:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")
    ctx.scheduler = SequentialLR(ctx.optimizer, [warmup, stable], [warmup_steps])


@contextmanager
def training_context(cfg: object, ctx: object) -> Generator[TrainingCtx, None, None]:
    """Context manager for training."""
    assert hasattr(cfg, "training") and isinstance(cfg.training, TrainingCfg)
    assert hasattr(ctx, "training") and isinstance(ctx.training, TrainingCtx)
    assert hasattr(ctx, "distributed") and isinstance(ctx.distributed, DistributedCtx)
    ctx.training.step = 0
    setup_dataset(cfg.training, ctx.training)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    setup_model(cfg.training, ctx.training, cfg.distributed, ctx.distributed)
    setup_optimizer(cfg.training, ctx.training)
    setup_scheduler(cfg.training, ctx.training)
    try:
        gc.disable()
        yield ctx.training
    finally:
        gc.enable()
