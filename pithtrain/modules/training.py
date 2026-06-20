"""PithTrain training module."""

from __future__ import annotations

import gc
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Literal, Optional, Union

import numpy as np
import torch
import torch.distributed.fsdp
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers import AutoConfig

from pithtrain.config import SlottedDefault
from pithtrain.dualpipe import DualPipeV, set_p2p_tensor_dtype, set_p2p_tensor_shapes
from pithtrain.models.deepseek_v2_lite import DeepseekV2LiteModel
from pithtrain.models.gpt_oss import GptOssModel
from pithtrain.models.qwen3_moe import Qwen3MoeModel
from pithtrain.modules.dataset import ConcatDataset, MemmapDataset
from pithtrain.modules.load_balance import make_load_balance_loss_fn
from pithtrain.modules.optimizer import Muon

from .distributed import DistributedCfg, DistributedCtx


def is_muon_param(name: str, param: torch.Tensor) -> bool:
    """
    True if Muon should optimize ``param`` (False routes it to AdamW):

    * Muon: 2D hidden weights (attention q/k/v/o, MLA projections, dense and
      shared-expert gate/up/down, stacked 3D expert weights).
    * AdamW: everything else (1D norms/biases/sinks, embeddings, LM head, MoE
      gate/router, 2D stacked expert biases).
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


def make_muon_optimizer(
    cfg: TrainingCfg, ctx: TrainingCtx, *, weight_decay: float = 0.1
) -> tuple[Optimizer, ...]:
    """
    Muon for the 2D hidden weights, AdamW for the rest (the :func:`is_muon_param`
    split). Weight decay (0.1, per "Muon is Scalable for LLM Training") applies to
    both: decaying the RMSNorm gamma keeps per-layer output RMS from blowing up.
    """
    muon_params, adamw_params = [], []
    for name, param in ctx.model.named_parameters():
        if not param.requires_grad:
            continue
        if is_muon_param(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    kwargs = dict(lr=cfg.lr, weight_decay=weight_decay)
    return Muon(muon_params, **kwargs), AdamW(adamw_params, **kwargs)


def make_adamw_optimizer(
    cfg: TrainingCfg, ctx: TrainingCtx, *, weight_decay: float = 0.1
) -> tuple[Optimizer, ...]:
    """AdamW over all parameters."""
    kwargs = dict(lr=cfg.lr, weight_decay=weight_decay)
    return (AdamW(ctx.model.parameters(), **kwargs),)


def make_wsd_scheduler(
    cfg: TrainingCfg,
    ctx: TrainingCtx,
    *,
    start_lr: float = 0.0,
    warmup_ratio: float = 0.0,
    final_lr: float = 0.0,
    decay_ratio: float = 0.1,
    decay_shape: Literal["cosine", "linear"] = "cosine",
) -> tuple[LRScheduler, ...]:
    """
    Warmup-stable-decay: linear warmup, hold lr, then cosine or linear decay.
    Set decay_ratio = 1 - warmup_ratio for a pure anneal.
    """
    if decay_shape not in ("cosine", "linear"):
        raise ValueError(f"Unknown decay_shape: {decay_shape!r}")
    if warmup_ratio + decay_ratio > 1:
        raise ValueError("warmup_ratio + decay_ratio must be <= 1")

    max_steps = cfg.max_steps
    warmup_steps, decay_steps = round(warmup_ratio * max_steps), round(decay_ratio * max_steps)
    stable_steps = max_steps - warmup_steps - decay_steps
    start_factor, final_factor = start_lr / cfg.lr, final_lr / cfg.lr

    def lr_lambda(step: int) -> float:
        # Warmup start_lr -> lr.
        if step < warmup_steps:
            return start_factor + (1 - start_factor) * (step / warmup_steps)
        # Stable at lr.
        if decay_steps == 0 or step < warmup_steps + stable_steps:
            return 1.0
        # Decay lr -> final_lr.
        t = min((step - warmup_steps - stable_steps) / decay_steps, 1.0)
        match decay_shape:
            case "cosine":
                return final_factor + (1 - final_factor) * 0.5 * (1 + math.cos(math.pi * t))
            case "linear":
                return 1.0 + (final_factor - 1.0) * t

    return tuple(LambdaLR(opt, lr_lambda) for opt in ctx.optimizers)


def make_constant_scheduler(cfg: TrainingCfg, ctx: TrainingCtx) -> tuple[LRScheduler, ...]:
    """Hold lr constant for the whole run: no warmup, no decay."""
    return tuple(LambdaLR(opt, lambda _: 1.0) for opt in ctx.optimizers)


@dataclass(init=False, slots=True)
class TrainingCfg(SlottedDefault):
    dataset: Path
    """The root directory hosting the tokenized dataset."""

    sequence_length: int
    """The sequence length for each training sample."""

    seed: int = 1234
    """The random seed for reproducibility."""

    lr: float
    """The base learning rate to construct the optimizer."""

    max_steps: int
    """The maximum number of training steps."""

    micro_batch_size: int
    """The size of each micro-batch used during training."""

    global_batch_size: int
    """
    The size of the global batch used during training.

    Gradients will be accumulated over multiple micro-batches to achieve this batch size.
    """

    optimizer: Callable[[TrainingCfg, TrainingCtx], tuple[Optimizer, ...]]
    """
    Builder for the optimizer(s). Use a built-in below or make your own:

    * :func:`make_muon_optimizer`: Muon + AdamW, split by :func:`is_muon_param`.
    * :func:`make_adamw_optimizer`: AdamW over all parameters.
    """

    scheduler: Callable[[TrainingCfg, TrainingCtx], tuple[LRScheduler, ...]]
    """
    Builder for the scheduler(s), one per optimizer. Use a built-in below or make your own:

    * :func:`make_wsd_scheduler`: warmup, stable hold, then cosine/linear decay.
    * :func:`make_constant_scheduler`: hold lr constant for the whole run.
    """

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

    optimizers: tuple[Optimizer, ...]
    """Optimizer(s): Muon composes two (Muon + AdamW); AdamW is a 1-tuple."""

    schedulers: tuple[LRScheduler, ...]
    """Scheduler(s), one per optimizer (same warmup/decay shape)."""

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

    ctx.model = DualPipeV(modules, pp_group=pp_group, ep_group=ep_group)
    set_p2p_tensor_shapes([(micro_batch_size, local_seq_len, hidden_size)])
    set_p2p_tensor_dtype(torch.bfloat16)


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
    ctx.training.optimizers = cfg.training.optimizer(cfg.training, ctx.training)
    ctx.training.schedulers = cfg.training.scheduler(cfg.training, ctx.training)
    try:
        gc.disable()
        yield ctx.training
    finally:
        gc.enable()
