from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed.fsdp
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers import AutoConfig

from pithtrain.config import SlottedDefault
from pithtrain.contexts import distributed, training
from pithtrain.models.deepseek_v2 import DeepSeekV2Model
from pithtrain.models.gpt_oss import GptOssModel
from pithtrain.models.qwen3_moe import Qwen3MoeModel
from pithtrain.models.qwen35_moe import Qwen35MoeModel
from pithtrain.modules.fp8_param_gather import (
    install_fp8_param_gather,
    install_raw_fp8_param_gather,
)
from pithtrain.modules.fsdp_forward import install_lean_hooks, register_fsdp_forward_method
from pithtrain.modules.load_balance import force_balance, make_load_balance_loss_fn
from pithtrain.modules.optimizer import Muon
from pithtrain.modules.wgrad_accumulation import install_raw_main_grad, install_wgrad_accumulation
from pithtrain.operators.grouped_linear import FP8GroupedLinear, GroupedLinear
from pithtrain.operators.linear import BF16Linear, FP8Linear
from pithtrain.pipeline import DualPipeV

from .distributed import DistributedCfg

__all__ = [
    "TrainingCfg",
    "is_muon_param",
    "make_muon_optimizer",
    "make_adamw_optimizer",
    "make_wsd_scheduler",
    "make_constant_scheduler",
    "init_weights",
    "apply_fsdp",
    "setup_model",
    "offload_training_state",
    "reload_training_state",
    "setup_training",
]

PIPELINE_STAGE_MODELS = (DeepSeekV2Model, GptOssModel, Qwen3MoeModel, Qwen35MoeModel)


def is_muon_param(name: str, param: torch.Tensor) -> bool:
    if param.ndim < 2:
        return False
    if name.endswith(".bias") or name.endswith("_bias"):
        return False
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return False
    if ".gate.weight" in name or ".router.weight" in name:
        return False
    return True


def make_muon_optimizer(cfg: TrainingCfg, *, weight_decay: float = 0.1) -> tuple[Optimizer, ...]:
    """
    The default weight_decay of 0.1 follows "Muon is Scalable for LLM Training"; decaying the norm
    weights too bounds the output RMS.
    """
    muon_params, adamw_params = [], []
    for name, param in training.model.named_parameters():
        if not param.requires_grad:
            continue
        if is_muon_param(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    kwargs = dict(lr=cfg.lr, weight_decay=weight_decay)
    return Muon(muon_params, **kwargs), AdamW(adamw_params, **kwargs)


def make_adamw_optimizer(cfg: TrainingCfg, *, weight_decay: float = 0.1) -> tuple[Optimizer, ...]:
    kwargs = dict(lr=cfg.lr, weight_decay=weight_decay)
    return (AdamW(training.model.parameters(), fused=True, **kwargs),)


def make_wsd_scheduler(
    cfg: TrainingCfg,
    *,
    start_lr: float = 0.0,
    warmup_ratio: float = 0.0,
    final_lr: float = 0.0,
    decay_ratio: float = 0.1,
    decay_shape: Literal["cosine", "linear"] = "cosine",
) -> tuple[LRScheduler, ...]:
    """
    Warmup-stable-decay: linear warmup from start_lr, hold lr, then cosine or linear decay to
    final_lr. Set decay_ratio = 1 - warmup_ratio for a pure anneal.
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
        if step < warmup_steps:
            return start_factor + (1 - start_factor) * (step / warmup_steps)
        if decay_steps == 0 or step < warmup_steps + stable_steps:
            return 1.0
        t = min((step - warmup_steps - stable_steps) / decay_steps, 1.0)
        match decay_shape:
            case "cosine":
                return final_factor + (1 - final_factor) * 0.5 * (1 + math.cos(math.pi * t))
            case "linear":
                return 1.0 + (final_factor - 1.0) * t

    return tuple(LambdaLR(opt, lr_lambda) for opt in training.optimizers)


def make_constant_scheduler(cfg: TrainingCfg) -> tuple[LRScheduler, ...]:
    return tuple(LambdaLR(opt, lambda _: 1.0) for opt in training.optimizers)


@dataclass(init=False, slots=True)
class TrainingCfg(SlottedDefault):
    sequence_length: int
    """
    The number of tokens in each training sample. For packed variable-length samples it is an
    upper bound on the tokens in one micro-batch.
    """

    seed: int = 1234

    lr: float

    max_steps: int

    micro_batch_size: int

    global_batch_size: int
    """
    The number of samples in a step, across all data-parallel ranks; gradients accumulate over the
    micro-batches that make it up.
    """

    optimizer: Callable[[TrainingCfg], tuple[Optimizer, ...]]
    """
    Builds the optimizers from this config: make_muon_optimizer, make_adamw_optimizer or your own.
    """

    scheduler: Callable[[TrainingCfg], tuple[LRScheduler, ...]]
    """
    Builds one scheduler per optimizer from this config: make_wsd_scheduler,
    make_constant_scheduler or your own.
    """

    model: Path
    """
    A HuggingFace model directory. Only its config.json is read; the weights come from the
    checkpoint or from init_weights.
    """

    save_interval: Optional[int] = None
    """
    Steps between checkpoint saves. When None, nothing is saved, but a checkpoint is still loaded
    from save_location if set.
    """

    save_location: Optional[Path] = None
    """
    The checkpoint directory; checkpoints are loaded from and saved to
    <save_location>/torch-dcp/<step>. When None, the model trains from scratch and saves nothing.
    """

    moe_load_balance_coef: float = 0.0
    """
    Coefficient of the MoE load balance loss, typically 1e-2 to 1e-1; 0 disables it.
    """

    moe_load_balance_type: Literal["micro-batch", "global-batch", "sequence"] = "micro-batch"
    """
    Scope of the MoE load balance loss.

    micro-batch: one loss per micro-batch (https://arxiv.org/abs/2101.03961).
    global-batch: expert frequencies summed over the dp x cp ranks and the step's micro-batches
    (https://arxiv.org/abs/2501.11873).
    sequence: one loss per sequence, averaged over the batch (https://arxiv.org/abs/2405.04434).
    """

    benchmark: bool = False
    """
    Force-balance MoE routing so every expert gets an equal token load, as NeMo Megatron-Bridge
    does for its MoE benchmarks. For throughput measurement only; wrong for real training.
    """

    fp8: bool = False
    """
    Train the linear layers in FP8 through DeepGEMM with 128-element block scaling; False is BF16.
    """

    init_std: float = 0.02

    nsys_start: Optional[int] = None
    """
    Step at whose start the CUDA profiler starts, for Nsight Systems; None disables it.
    """

    nsys_stop: Optional[int] = None
    """
    Step at whose start the CUDA profiler stops, so this step is not profiled. To profile step N
    alone, set nsys_start=N and nsys_stop=N+1.
    """

    memory_profile_start: Optional[int] = None
    """
    Step at whose start CUDA memory history recording starts, with stack traces of allocations
    and frees; None disables it.
    """

    memory_profile_stop: Optional[int] = None
    """
    Step at whose start the recorded history is dumped to memory_profile_output and recording
    stops. To profile step N alone, set memory_profile_start=N and memory_profile_stop=N+1.
    """

    memory_profile_output: Path = Path.cwd()
    """
    Directory of the CUDA memory snapshots, one snapshot-rank<rank>.pickle per rank. View them at
    https://pytorch.org/memory_viz in a browser.
    """


def init_weights(model: nn.Module, num_layers: int, init_std: float = 0.02, seed: int = 0) -> None:
    """
    Initialize this rank's V-chunks (an nn.Sequential of two models) in place, before FSDP shards
    them. Every parameter of 2 or more dims draws from N(0, init_std), except o_proj and down_proj,
    which use N(0, init_std / sqrt(2 * num_layers)) with num_layers of the full model. 1D parameters
    stay as they are.

    Each parameter draws from a stream keyed by its name in the full model, and each routed expert
    of a 3D stacked expert weight from one keyed also by its global expert index, so every parallel
    layout starts from the same weights. The 2D stacked expert biases of gpt-oss draw as one tensor,
    so they differ across EP sizes.
    """
    output_std = init_std / math.sqrt(2.0 * num_layers)
    for chunk_name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        name = chunk_name.split(".", 1)[1]  # drop the V-chunk index
        std = output_std if "o_proj" in name or "down_proj" in name else init_std
        # Routed experts are stacked (local experts, ...), the local ones of this EP rank.
        if ".experts." in name and param.dim() == 3:
            first_expert = distributed.ep_rank * param.shape[0]
            for local_idx in range(param.shape[0]):
                generator = weight_generator(seed, name, param.device, first_expert + local_idx)
                torch.nn.init.normal_(param[local_idx], mean=0.0, std=std, generator=generator)
        else:
            generator = weight_generator(seed, name, param.device)
            torch.nn.init.normal_(param, mean=0.0, std=std, generator=generator)


def weight_generator(seed: int, name: str, device: torch.device, *index: int) -> torch.Generator:
    key = (*name.encode(), *index)
    state = np.random.SeedSequence(seed, spawn_key=key).generate_state(1, np.uint64)[0]
    return torch.Generator(device=device).manual_seed(int(state))


def split_replicas(mesh: DeviceMesh, hsdp_replica: int) -> DeviceMesh:
    """
    The (replica, shard) layout FSDP2 reads as replicate-then-shard; each shard group is a
    contiguous block of the mesh.
    """
    size = mesh.size()
    assert size % hsdp_replica == 0, f"{size=} not divisible by {hsdp_replica=}"
    return mesh._unflatten(0, (hsdp_replica, size // hsdp_replica), ("replica", "shard"))


def apply_fsdp(model, hsdp_replica: int = 1):
    """
    Shard the two V-chunks in model with FSDP2 in place and return model. The routed expert
    weights shard over the dp axis of the expert mesh, everything else over the dp x cp stage, and
    gradients reduce as a plain sum. Under FP8 the FP8 weights are all-gathered in FP8.
    """
    expt_fsdp_mesh = distributed.expt_mesh["dp"]
    if hsdp_replica > 1:
        # A separate name: _unflatten takes the flattened dim away from its mesh, and setup_model
        # reads the default flattening for the load-balance group.
        attn_fsdp_mesh = distributed.attn_mesh["dp", "cp"]._flatten("hsdp_stage")
        attn_fsdp_mesh = split_replicas(attn_fsdp_mesh, hsdp_replica)
        expt_fsdp_mesh = split_replicas(expt_fsdp_mesh, hsdp_replica)
    else:
        attn_fsdp_mesh = distributed.attn_mesh["dp", "cp"]._flatten()
    mp = MixedPrecisionPolicy(
        param_dtype=training.PARAM_DTYPE,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    install_lean_hooks()
    # FSDP2 shards submodules before their parents; the walk stops at a sharded submodule, so
    # the expert parameters stay out of the layer's group.
    for i in range(2):
        assert isinstance(model[i], PIPELINE_STAGE_MODELS)
        # Embeddings, final norm and LM head run once per step, so they reshard after forward.
        for name, child in model[i].named_children():
            if name != "layers" and next(child.parameters(), None) is not None:
                fully_shard(child, mesh=attn_fsdp_mesh, reshard_after_forward=True, mp_policy=mp)
        for layer in model[i].layers.values():
            if hasattr(layer.mlp, "experts"):
                fully_shard(
                    layer.mlp.experts,
                    mesh=expt_fsdp_mesh,
                    reshard_after_forward=False,
                    mp_policy=mp,
                )
                if hasattr(layer.mlp.experts, "forward_from_tokens"):
                    register_fsdp_forward_method(layer.mlp.experts, "forward_from_tokens")
            fully_shard(layer, mesh=attn_fsdp_mesh, reshard_after_forward=False, mp_policy=mp)
            register_fsdp_forward_method(layer, "forward_stage1")
            register_fsdp_forward_method(layer, "forward_stage3")
            register_fsdp_forward_method(layer, "forward_stage5")
        fully_shard(model[i], mesh=attn_fsdp_mesh, reshard_after_forward=False, mp_policy=mp)

    # Sum, not average: the two classes reduce over groups of different size, yet each sum is the
    # global gradient, so the step's single divide by the global token count normalizes both.
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)

    # Gathering FP8 weights in FP8 gives the same operands as quantizing the gathered bf16 weight.
    if training.fp8:
        install_fp8_param_gather(model, (FP8Linear, FP8GroupedLinear))
        # gpt-oss raw expert parameters; must follow the call above, which resets the registry.
        install_raw_fp8_param_gather(model)
    return model


def setup_model(
    cfg: TrainingCfg,
    distributed_cfg: DistributedCfg,
) -> None:
    training.fp8 = cfg.fp8
    training.Linear = FP8Linear if cfg.fp8 else BF16Linear
    training.GroupedLinear = FP8GroupedLinear if cfg.fp8 else GroupedLinear

    cp_size = distributed.cp_size
    cp_group = distributed.cp_group

    modules = []
    module_config = AutoConfig.from_pretrained(cfg.model)
    module_config.max_position_embeddings = cfg.sequence_length

    assert hasattr(module_config, "hidden_size")
    assert isinstance(module_config.hidden_size, int)
    if cfg.sequence_length % (2 * cp_size) != 0:
        raise ValueError(
            f"sequence_length ({cfg.sequence_length}) must be divisible by "
            f"2 * context_parallel_size ({2 * cp_size}); zigzag ring attention "
            f"splits the sequence into 2*cp_size equal chunks"
        )

    if module_config.model_type == "deepseek_v2":
        ModelClass = DeepSeekV2Model
    elif module_config.model_type == "qwen3_moe":
        ModelClass = Qwen3MoeModel
    elif module_config.model_type == "gpt_oss":
        ModelClass = GptOssModel
    elif module_config.model_type == "qwen3_5_moe_text":
        ModelClass = Qwen35MoeModel
    else:
        raise ValueError(f"Unsupported model_type: {module_config.model_type}")

    modules.append(ModelClass(module_config, phase=0))
    modules.append(ModelClass(module_config, phase=1))

    modules = nn.Sequential(*modules)
    init_weights(modules, module_config.num_hidden_layers, cfg.init_std, cfg.seed)
    apply_fsdp(modules, distributed_cfg.hsdp_replica)
    install_wgrad_accumulation(modules)
    install_raw_main_grad(modules)  # gpt-oss raw expert weights; no-op elsewhere

    # The two V-chunks build identical read-only rotary caches; share one to save memory.
    if hasattr(modules[0], "rotary_emb") and hasattr(modules[1], "rotary_emb"):
        modules[1].rotary_emb = modules[0].rotary_emb

    local_seq_len = cfg.sequence_length // cp_size

    if cfg.moe_load_balance_coef > 0:
        # With hsdp_replica 1 this reuses the flat dp x cp mesh apply_fsdp built,
        # adding no communicator.
        lb_group = distributed.attn_mesh["dp", "cp"]._flatten().get_group()
        for i in range(2):
            for layer in modules[i].layers.values():
                gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
                if gate is not None:
                    loss_fn = make_load_balance_loss_fn(
                        cfg.moe_load_balance_type,
                        cfg.moe_load_balance_coef,
                        lb_group,
                        sequence_length=local_seq_len,
                        cp_group=cp_group,
                    )
                    if hasattr(loss_fn, "init_buffers"):
                        loss_fn.init_buffers(gate.num_experts, gate.weight.device)
                    gate.load_balance_loss_fn = loss_fn

    if cfg.benchmark:
        for module in modules.modules():
            if hasattr(module, "router_replay"):
                module.router_replay = force_balance(module.num_experts)

    training.model = DualPipeV(modules)


offloaded_state: list[tuple[torch.UntypedStorage, torch.Tensor]] = []
"""
Each evicted CUDA storage with its pinned host copy; non-empty only while offloaded.
"""

offload_buffer_pool: dict[int, list[torch.Tensor]] = {}
"""
Free pinned host buffers by byte size, kept for the life of the process, because page-locking is
the expensive part of an offload.
"""


def take_pinned_buffer(num_bytes: int) -> torch.Tensor:
    free = offload_buffer_pool.setdefault(num_bytes, [])
    if free:
        return free.pop()
    return torch.empty(num_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)


def storage_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def training_state_storages() -> list[torch.UntypedStorage]:
    """
    Deduplicated by address, because FSDP2's padded shard buffer and a parameter's local tensor
    share one allocation.
    """
    storages: dict[int, torch.UntypedStorage] = {}

    def collect(tensor: torch.Tensor) -> None:
        local = tensor._local_tensor if isinstance(tensor, DTensor) else tensor
        if local.device.type != "cuda":
            return
        storage = local.untyped_storage()
        if storage.size() > 0:
            storages.setdefault(storage.data_ptr(), storage)

    for param in training.model.parameters():
        collect(param)
    for optimizer in training.optimizers:
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    collect(value)

    return list(storages.values())


def offload_training_state() -> None:
    """
    Copy this rank's parameter shards and optimizer state to pinned host memory and free their
    GPU storages. Tensors keep their identity, so reload_training_state restores them in place.
    Gradients stay on the GPU. Nothing may read or write the state until reload_training_state
    clears training.offloaded. A no-op while already offloaded.
    """
    if training.offloaded:
        return

    # An unsharded parameter is a separate allocation that the storage walk would miss.
    for module in training.model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()

    evicted = []
    for storage in training_state_storages():
        host = take_pinned_buffer(storage.size())
        host.copy_(storage_bytes(storage), non_blocking=True)
        evicted.append((storage, host))

    # Publish and free only after all copies finish; a pinning failure must allow retry.
    torch.cuda.synchronize()
    offloaded_state.extend(evicted)
    training.offloaded = True
    for storage, _ in offloaded_state:
        storage.resize_(0)
    torch.cuda.empty_cache()


def reload_training_state() -> None:
    if not training.offloaded:
        return

    for storage, host in offloaded_state:
        storage.resize_(host.numel())
        storage_bytes(storage).copy_(host, non_blocking=True)

    # The copies are asynchronous, so no buffer may return to the pool until they land.
    torch.cuda.synchronize()
    for _, host in offloaded_state:
        offload_buffer_pool[host.numel()].append(host)
    offloaded_state.clear()
    training.offloaded = False


def setup_training(cfg: object) -> None:
    assert hasattr(cfg, "training") and isinstance(cfg.training, TrainingCfg)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    setup_model(cfg.training, cfg.distributed)
    training.optimizers = cfg.training.optimizer(cfg.training)
    training.schedulers = cfg.training.scheduler(cfg.training)
