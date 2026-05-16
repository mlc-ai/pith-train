"""Pretrain a language model."""

import gc
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.cuda
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import wandb
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pithtrain.config import SlottedDefault
from pithtrain.modules.checkpoint import (
    to_canonical_model,
    to_canonical_optim,
    to_localized_model,
    to_localized_optim,
)
from pithtrain.modules.distributed import DistributedCfg, DistributedCtx, distributed_context
from pithtrain.modules.load_balance import MoELoadBalanceLossTracker
from pithtrain.modules.logging import LoggingCfg, LoggingCtx, activate_wandb, logging_context
from pithtrain.modules.training import TrainingCfg, TrainingCtx, training_context
from pithtrain.operators.cross_entropy import cross_entropy


@dataclass(init=False, slots=True)
class PretrainLanguageModelCfg(SlottedDefault):
    """Configuration for pretraining a language model."""

    distributed: DistributedCfg = field(default_factory=DistributedCfg)
    """Distributed training configuration."""

    training: TrainingCfg = field(default_factory=TrainingCfg)
    """Training configuration including model, optimizer, and dataset settings."""

    logging: LoggingCfg = field(default_factory=LoggingCfg)
    """Logging configuration."""


@dataclass(init=False, slots=True)
class PretrainLanguageModelCtx(SlottedDefault):
    """Context for pretraining a language model."""

    logging: LoggingCtx = field(default_factory=LoggingCtx)
    """Active logging context."""

    distributed: DistributedCtx = field(default_factory=DistributedCtx)
    """Active distributed context."""

    training: TrainingCtx = field(default_factory=TrainingCtx)
    """Active training context."""


def get_global_batch(
    cfg: PretrainLanguageModelCfg, ctx: PretrainLanguageModelCtx, device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Gather this rank's portion of the global batch on pipeline parallel rank 0."""
    if ctx.distributed.pp_rank != 0:
        return None, None

    # short-hands
    step = ctx.training.step
    micro_batch_size = cfg.training.micro_batch_size
    global_batch_size = cfg.training.global_batch_size
    dp_size = ctx.distributed.dp_size
    dp_rank = ctx.distributed.dp_rank
    ep_size = ctx.distributed.ep_size
    ep_rank = ctx.distributed.ep_rank
    sequence_length = cfg.training.sequence_length
    dataset = ctx.training.dataset

    # arithmetic for dataset indices
    effective_batch_size = micro_batch_size * dp_size * ep_size
    local_batch_size = global_batch_size // (dp_size * ep_size)
    start0 = step * global_batch_size + (dp_rank * ep_size + ep_rank) * micro_batch_size

    # Zigzag context-parallel sharding. We split the global sequence into
    # 2*cp_size equal chunks and assign rank r the pair (r, 2*cp_size-r-1).
    # The local sequence is the concatenation of the "front" chunk and the
    # mirror "back" chunk, balancing the causal workload across CP ranks
    # (see pithtrain/operators/ring_attention.py for the matching attention
    # implementation). For cp_size == 1 this reduces to a contiguous read.
    cp_size = ctx.distributed.cp_size
    cp_rank = ctx.distributed.cp_rank
    block = sequence_length // (2 * cp_size)
    local_seq_len = 2 * block
    front_offset = cp_rank * block
    back_offset = (2 * cp_size - cp_rank - 1) * block

    # single allocation on host, then one HtoD transfer per tensor
    local_tokens = torch.empty((local_batch_size, local_seq_len), dtype=torch.long)
    local_labels = torch.empty((local_batch_size, local_seq_len), dtype=torch.long)

    # fill in one pass: k iterates over our rank-local batch rows. Each sample
    # is two memmap reads (front block + back block) followed by an in-place
    # concat into the pre-allocated host buffer.
    for k in range(local_batch_size):
        acc, off = divmod(k, micro_batch_size)
        index = start0 + acc * effective_batch_size + off
        tokens_a, labels_a = dataset.get_chunk(index, front_offset, block)
        tokens_b, labels_b = dataset.get_chunk(index, back_offset, block)
        local_tokens[k, :block] = tokens_a
        local_tokens[k, block:] = tokens_b
        local_labels[k, :block] = labels_a
        local_labels[k, block:] = labels_b

    local_tokens = local_tokens.to(device, non_blocking=True)
    local_labels = local_labels.to(device, non_blocking=True)

    return local_tokens, local_labels


def criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output = output.view(-1, output.size(-1))
    target = target.view(-1)
    return cross_entropy(output, target, ignore_index=-100)


@torch.no_grad()
def clip_grad_norm_(model: nn.Module, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """
    Clip gradients by global norm across all ranks (FSDP + pipeline).
    Returns the total gradient norm before clipping.
    """
    grads = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        if isinstance(g, DTensor):
            g = g.to_local()
        grads.append(g)
    if not grads:
        first_param = next(model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    local_norm = torch.nn.utils.get_total_norm(grads, norm_type=norm_type)
    # Global L2 norm: all-reduce sum of squared norms across all ranks (FSDP + pipeline).
    local_norm_sq = local_norm**norm_type
    torch.distributed.all_reduce(local_norm_sq, op=torch.distributed.ReduceOp.SUM)
    total_norm = (local_norm_sq ** (1.0 / norm_type)).clamp(min=1e-6)
    clip_coef = (max_norm / total_norm).clamp(max=1.0)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return total_norm


class AppState(Stateful):
    """Stateful object to save and load the checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        model_only: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_only = model_only

    def state_dict(self):
        """
        Serialize the model, optimizer, and scheduler to a state dictionary.

        Both model and optimizer states are converted to canonical (PP-independent) format:
        the module.{N}. prefix is stripped so FQNs use global layer IDs (e.g.
        layers.0.weight), and stacked expert weights are expanded to individual
        expert tensors with global IDs.

        When ``model_only`` is set (e.g. loading a checkpoint converted from
        HuggingFace that has no optimizer/scheduler), only model keys are
        advertised so DCP's planner does not look for missing optimizer keys.
        """
        if self.model_only:
            model_state, _ = get_state_dict(self.model, self.optimizer)
            return {"model": to_canonical_model(model_state, self.model)}
        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        model_state = to_canonical_model(model_state, self.model)
        optim_state = to_canonical_optim(optim_state, self.model)
        sched_state = self.scheduler.state_dict()
        return {"model": model_state, "optimizer": optim_state, "scheduler": sched_state}

    def load_state_dict(self, state_dict):
        """
        Restore the model, optimizer, and scheduler from the checkpoint.

        Canonical (PP-independent) FQNs are mapped back to local FQNs using the
        current model structure.  The optimizer param_groups are rebuilt from
        the current model so that DCP's cross-rank deduplication of non-tensor
        metadata does not cause FQN mismatches.

        Released checkpoints from HuggingFace may not necessarily include the states of the
        optimizer and scheduler, so we skip the loading of these states if they are missing.
        """
        model_state = to_localized_model(state_dict["model"], self.model)
        optim_state = state_dict.get("optimizer")
        sched_state = state_dict.get("scheduler")

        if optim_state:
            optim_state = to_localized_optim(optim_state, self.model)
            kwargs = dict(model_state_dict=model_state, optim_state_dict=optim_state)
            set_state_dict(self.model, self.optimizer, **kwargs)
        else:
            options = StateDictOptions(strict=False)
            set_model_state_dict(self.model, model_state, options=options)
        if sched_state:
            self.scheduler.load_state_dict(sched_state)


def raise_if_dataset_insufficient(
    cfg: PretrainLanguageModelCfg, ctx: PretrainLanguageModelCtx
) -> None:
    """Raise if configured run requires more samples than available in dataset."""
    global_batch_size = cfg.training.global_batch_size
    max_steps = cfg.training.max_steps

    assert global_batch_size > 0, f"{global_batch_size=}"

    required_samples = max_steps * global_batch_size
    dataset_size = len(ctx.training.dataset)

    if dataset_size >= required_samples:
        return

    message = (
        "Dataset is too small for this run: available-samples=%s, required-samples=%s "
        "(max_steps=%s x global_batch_size=%s)."
        % (
            format(dataset_size, ","),
            format(required_samples, ","),
            format(max_steps, ","),
            format(global_batch_size, ","),
        )
    )
    if ctx.distributed.rank == 0:
        raise RuntimeError(message)
    raise SystemExit(1)


def save_checkpoint(cfg: PretrainLanguageModelCfg, ctx: PretrainLanguageModelCtx) -> None:
    """
    Save the checkpoint at the current step.

    Uses cpu_offload=True (with the default full_state_dict=False)
    so that each rank's local FSDP shards are moved to CPU -- no GPU
    all-gather is performed.  Expert DTensors are split into per-expert
    entries locally (via unwrap_dtensor_experts in resharding.py),
    so each rank writes only the expert keys it owns.  Non-expert
    DTensors are kept as CPU DTensors and DCP saves each rank's shard.
    """
    stdout = ctx.logging.stdout
    assert cfg.training.save_location is not None
    save_location = Path(cfg.training.save_location, "torch-dcp", "step-%08d" % ctx.training.step)
    model = ctx.training.model
    optimizer = ctx.training.optimizer
    scheduler = ctx.training.scheduler

    options = StateDictOptions(cpu_offload=True)
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    state_dict = dict()
    state_dict["app"] = dict()
    state_dict["app"]["model"] = to_canonical_model(model_state, model)
    state_dict["app"]["optimizer"] = to_canonical_optim(optim_state, model)
    state_dict["app"]["scheduler"] = scheduler.state_dict()

    stdout.info("Save checkpoint: %s" % save_location)
    t0 = time.monotonic()
    gc.collect()
    torch.cuda.empty_cache()
    dcp.save(state_dict, checkpoint_id=save_location)
    rank = torch.distributed.get_rank()
    rng_path = Path(save_location, "rng-rank-%05d.pt" % rank)
    torch.save(torch.cuda.get_rng_state(), rng_path)
    dt = torch.tensor(time.monotonic() - t0, device="cuda")
    dt_min, dt_max = dt.clone(), dt.clone()
    torch.distributed.all_reduce(dt_min, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(dt_max, op=torch.distributed.ReduceOp.MAX)
    stdout.info("Save checkpoint: Elapsed min=%.1fs, max=%.1fs" % (dt_min.item(), dt_max.item()))


def load_checkpoint(cfg: PretrainLanguageModelCfg, ctx: PretrainLanguageModelCtx) -> None:
    """Load the checkpoint from the latest step."""
    stdout = ctx.logging.stdout
    if cfg.training.save_location is None:
        stdout.info("No save_location set; training from scratch.")
        return
    path2step = lambda p: int(p.stem.removeprefix("step-"))
    checkpoints = Path(cfg.training.save_location, "torch-dcp").glob("step-*")
    checkpoints = sorted(checkpoints, key=path2step)
    if not checkpoints:
        stdout.info("No checkpoint found; training from scratch.")
        return
    load_location = checkpoints.pop()
    stdout.info("Load checkpoint: %s" % load_location)
    t0 = time.monotonic()
    torch.cuda.empty_cache()
    metadata = FileSystemReader(str(load_location)).read_metadata()
    model_only = all(k.startswith("app.model.") for k in metadata.state_dict_metadata)
    model, optimizer, scheduler = ctx.training.model, ctx.training.optimizer, ctx.training.scheduler
    app_state = AppState(model, optimizer, scheduler, model_only=model_only)
    dcp.load({"app": app_state}, checkpoint_id=load_location)
    rank = torch.distributed.get_rank()
    rng_path = Path(load_location, "rng-rank-%05d.pt" % rank)
    if rng_path.exists():
        rng_state = torch.load(rng_path, weights_only=True)
        torch.cuda.set_rng_state(rng_state)
    ctx.training.step = path2step(load_location)
    dt = torch.tensor(time.monotonic() - t0, device="cuda")
    dt_min, dt_max = dt.clone(), dt.clone()
    torch.distributed.all_reduce(dt_min, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(dt_max, op=torch.distributed.ReduceOp.MAX)
    stdout.info("Load checkpoint: Elapsed min=%.1fs, max=%.1fs" % (dt_min.item(), dt_max.item()))


def train_step(cfg: PretrainLanguageModelCfg, ctx: PretrainLanguageModelCtx) -> None:
    """Execute one step of training."""
    # Start the nsys and the memory profiler.
    start = cfg.training.nsys_start
    if start is not None and ctx.training.step == start:
        torch.cuda.cudart().cudaProfilerStart()
    start = cfg.training.memory_profile_start
    if start is not None and ctx.training.step == start:
        torch.cuda.memory._record_memory_history(max_entries=65536, stacks="python")

    device = torch.cuda.current_device()
    t0 = time.time()

    torch.cuda.memory.reset_peak_memory_stats()

    model = ctx.training.model
    optimizer = ctx.training.optimizer
    scheduler = ctx.training.scheduler
    model.train()

    dp_size = ctx.distributed.dp_size
    ep_size = ctx.distributed.ep_size
    micro_batch_size = cfg.training.micro_batch_size
    global_batch_size = cfg.training.global_batch_size
    assert global_batch_size % (micro_batch_size * dp_size * ep_size) == 0

    # Gather the data for this rank's portion of the global batch.
    accumulate_steps = global_batch_size // (micro_batch_size * dp_size * ep_size)
    global_tokens, global_labels = get_global_batch(cfg, ctx, device)

    # Run the forward and backward pass.
    loss, _ = model.step(
        global_tokens,
        num_chunks=accumulate_steps,
        criterion=criterion,
        labels=(global_labels,),
        return_outputs=False,
    )

    # Average loss across CP ranks for correct logging.
    cp_size = ctx.distributed.cp_size
    if loss is not None and cp_size > 1:
        cp_group = ctx.distributed.device_mesh.get_group("cp")
        torch.distributed.all_reduce(loss, group=cp_group)
        loss /= cp_size

    # Scale gradients back so the effective loss is the mean over chunks.
    if accumulate_steps > 1:
        scale = 1.0 / accumulate_steps
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    # Clip the gradients.
    gradient_norm = clip_grad_norm_(model, max_norm=1.0, norm_type=2)

    # Take an optimization step.
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    MoELoadBalanceLossTracker.reset()

    # Measure the elapsed time in seconds.
    t1 = time.time()
    dt = torch.tensor(t1 - t0, device=device)
    torch.distributed.all_reduce(dt, op=torch.distributed.ReduceOp.MAX)
    elapsed = dt.item()

    # Measure the peak GPU memory allocated.
    peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
    peak_gpu_mem = torch.tensor(peak_gpu_mem, device=device)
    torch.distributed.all_reduce(peak_gpu_mem, op=torch.distributed.ReduceOp.MAX)

    # Collect the mean load balance loss (reduced across all ranks).
    # The tracked values include the coefficient: lb_coef * E * dot(f, p).
    # We divide it out so the logged metric is E * dot(f, p), where 1.0
    # represents perfect balance (matches Megatron-LM convention).
    moe_load_balance_coef = cfg.training.moe_load_balance_coef
    lb_total, lb_count = MoELoadBalanceLossTracker.get_total_count_and_clear()
    if moe_load_balance_coef > 0:
        lb_stats = torch.tensor([lb_total, lb_count], device=device)
        torch.distributed.all_reduce(lb_stats, op=torch.distributed.ReduceOp.SUM)
        lb_loss = (lb_stats[0] / lb_stats[1]).item() / moe_load_balance_coef
    else:
        lb_loss = 0.0

    # Print the loss and learning rate on rank 0.
    logger = ctx.logging.stdout
    if ctx.distributed.rank == 0:
        step = ctx.training.step
        max_steps = cfg.training.max_steps
        loss, lr = torch.mean(loss).item(), scheduler.get_last_lr()[0]
        tokens_per_second = global_batch_size * cfg.training.sequence_length / elapsed
        statements = []
        statements.append("step %08d/%08d" % (step + 1, max_steps))
        statements.append("step-time %.3f sec" % elapsed)
        statements.append("cross-entropy-loss %.4f" % loss)
        if moe_load_balance_coef > 0:
            statements.append("load-balance-loss %.6f" % lb_loss)
        statements.append("learning-rate %.6e" % lr)
        statements.append("gradient-norm %.4f" % gradient_norm.item())
        statements.append("tokens-per-second %s" % format(tokens_per_second, ",.0f"))
        statements.append("peak-gpu-memory %.2f GB" % peak_gpu_mem)
        logger.info(" | ".join(statements))
        # Lazily initialize WandB on the first successful step.
        activate_wandb(cfg, ctx)
        if ctx.logging.wandb is not None:
            metrics = dict()
            metrics["train/step"] = step
            metrics["train/cross-entropy-loss"] = loss
            if moe_load_balance_coef > 0:
                metrics["train/load-balance-loss"] = lb_loss
            metrics["train/learning-rate"] = lr
            metrics["train/gradient-norm"] = gradient_norm
            metrics["infra/tokens-per-second"] = tokens_per_second
            metrics["infra/peak-gpu-memory"] = peak_gpu_mem
            metrics["infra/step-time"] = elapsed
            wandb.log(metrics)

    # Increment the step counter.
    ctx.training.step += 1

    # Stop the nsys and the memory profiler.
    stop = cfg.training.nsys_stop
    if stop is not None and ctx.training.step == stop:
        torch.cuda.cudart().cudaProfilerStop()
    stop = cfg.training.memory_profile_stop
    if stop is not None and ctx.training.step == stop:
        rank = ctx.distributed.rank
        cfg.training.memory_profile_output.mkdir(parents=True, exist_ok=True)
        path = Path(cfg.training.memory_profile_output, "snapshot-rank%05d.pickle" % rank)
        torch.cuda.memory._dump_snapshot(str(path))
        torch.cuda.memory._record_memory_history(enabled=None)

    # We should save the checkpoint if any of the following conditions is true:
    # 1. The current step is a multiple of save_interval.
    # 2. The current step is the last step (max_steps).
    # Skip entirely if save_interval is None.
    if cfg.training.save_interval is not None:
        should_save = False
        should_save |= ctx.training.step % cfg.training.save_interval == 0
        should_save |= ctx.training.step == cfg.training.max_steps
        if should_save:
            save_checkpoint(cfg, ctx)

    # Run deferred GC here so cyclic collection never fires mid-forward/backward.
    gc.collect()


@record
def launch(cfg: PretrainLanguageModelCfg) -> None:
    """Launch the pretraining of a language model."""
    with ExitStack() as stack:
        ctx = PretrainLanguageModelCtx()
        stack.enter_context(logging_context(cfg, ctx))
        stack.enter_context(distributed_context(cfg, ctx))
        stack.enter_context(training_context(cfg, ctx))
        logger = ctx.logging.stdout
        logger.info("launch(cfg=%s)" % cfg)
        load_checkpoint(cfg, ctx)
        raise_if_dataset_insufficient(cfg, ctx)
        while ctx.training.step < cfg.training.max_steps:
            train_step(cfg, ctx)
