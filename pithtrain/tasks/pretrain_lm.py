"""
Pretrain a language model.
"""

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch
import torch.cuda
import wandb
from torch.distributed.elastic.multiprocessing.errors import record

from pithtrain.config import SlottedDefault
from pithtrain.contexts import distributed, logging, training
from pithtrain.modules.checkpoint import (
    find_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from pithtrain.modules.dataset import ConcatDataset, MemmapDataset
from pithtrain.modules.distributed import DistributedCfg, setup_distributed
from pithtrain.modules.hf_loader import load_hf_into_model
from pithtrain.modules.load_balance import MoELoadBalanceLossTracker
from pithtrain.modules.logging import LoggingCfg, activate_wandb, setup_logging
from pithtrain.modules.optimizer import clip_grad_norm
from pithtrain.modules.training import TrainingCfg, setup_training
from pithtrain.operators.cp_sequence import zigzag_spans
from pithtrain.operators.cross_entropy import cross_entropy
from pithtrain.pipeline import Microbatch


@dataclass(init=False, slots=True)
class PretrainLMCfg(SlottedDefault):
    """
    Configuration for pretraining a language model.
    """

    distributed: DistributedCfg = field(default_factory=DistributedCfg)
    """
    Distributed training configuration.
    """

    training: TrainingCfg = field(default_factory=TrainingCfg)
    """
    Model, optimizer, scheduler and checkpointing configuration.
    """

    logging: LoggingCfg = field(default_factory=LoggingCfg)
    """
    Logging configuration.
    """

    dataset: Path
    """
    The root directory hosting the tokenized corpus, globbed for *.bin shards.
    """


def setup_dataset(cfg: PretrainLMCfg) -> ConcatDataset:
    """
    Build the shuffled concatenation of every tokenized shard under the corpus root.
    """
    files = sorted(cfg.dataset.rglob("*.bin"))
    memmaps = [MemmapDataset(file, cfg.training.sequence_length) for file in files]
    dataset = ConcatDataset(memmaps, cfg.training.seed)
    required = cfg.training.max_steps * cfg.training.global_batch_size
    assert len(dataset) >= required, f"corpus has {len(dataset)} samples, run needs {required}"
    return dataset


def get_global_batch(
    cfg: PretrainLMCfg, dataset: ConcatDataset, step: int, device: torch.device
) -> List[Microbatch]:
    """
    Gather the portion of the global batch belonging to this rank, already split into micro-batches.

    dp_rank alone decides which data this rank loads: the expert rank names the experts a rank
    hosts, never the data it sees. Every pipeline rank loads the same samples, since the offsets
    below follow the step and the data and context ranks and never the pipeline rank, so each
    builds an identical list and DualPipeV needs no broadcast to learn the shapes. Only the first
    rank consumes the tensors, since under the V-shape it holds both the embedding and the loss.
    """
    # short-hands
    micro_batch_size = cfg.training.micro_batch_size
    global_batch_size = cfg.training.global_batch_size
    dp_size = distributed.dp_size
    dp_rank = distributed.dp_rank
    sequence_length = cfg.training.sequence_length

    # arithmetic for dataset indices
    effective_batch_size = micro_batch_size * dp_size
    local_batch_size = global_batch_size // dp_size
    start0 = step * global_batch_size + dp_rank * micro_batch_size

    # two blocks per rank under zigzag CP; at cp_size 1 this is one contiguous read
    front, back = zigzag_spans(distributed.cp_rank, distributed.cp_size, sequence_length)
    block = len(front)
    local_seq_len = 2 * block

    # single allocation on host, then one HtoD transfer per tensor
    local_tokens = torch.empty((local_batch_size, local_seq_len), dtype=torch.long)
    local_labels = torch.empty((local_batch_size, local_seq_len), dtype=torch.long)

    # fill in one pass: k iterates over our rank-local batch rows. Each sample
    # is two memmap reads (front block + back block) followed by an in-place
    # concat into the pre-allocated host buffer.
    for k in range(local_batch_size):
        acc, off = divmod(k, micro_batch_size)
        index = start0 + acc * effective_batch_size + off
        tokens_a, labels_a = dataset.get_chunk(index, front.start, block)
        tokens_b, labels_b = dataset.get_chunk(index, back.start, block)
        local_tokens[k, :block] = tokens_a
        local_tokens[k, block:] = tokens_b
        local_labels[k, :block] = labels_a
        local_labels[k, block:] = labels_b

    local_tokens = local_tokens.to(device, non_blocking=True)
    local_labels = local_labels.to(device, non_blocking=True)

    # Rows are already micro-batch major, so a plain split reproduces the partitioning the pipeline
    # applies itself: rows [i * mbs, (i + 1) * mbs) belong to micro-batch i.
    return [
        Microbatch(
            model_inputs=(local_tokens[i : i + micro_batch_size],),
            cu_seqlens=None,
            objective_inputs=(local_labels[i : i + micro_batch_size],),
        )
        for i in range(0, local_batch_size, micro_batch_size)
    ]


def objective(
    model_outputs: Tuple[torch.Tensor, ...],
    objective_inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cross-entropy objective for language-model pretraining.

    Returns the loss summed over the tokens of this micro-batch, so gradients accumulate across
    micro-batches; the training step divides by the global non-ignored token count for a correct
    token-weighted mean. The second return value is the same loss detached, which the step
    reduces the same way to log the training loss.
    """
    (logits,) = model_outputs
    (labels,) = objective_inputs
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    n_tokens = (labels != -100).sum().clamp_min(1)
    loss = cross_entropy(logits, labels, ignore_index=-100) * n_tokens
    return loss, loss.detach()


def train_step(cfg: PretrainLMCfg, dataset: ConcatDataset, step: int) -> None:
    """
    Execute one step of training.
    """
    # Start the nsys and the memory profiler.
    start = cfg.training.nsys_start
    if start is not None and step == start:
        torch.cuda.cudart().cudaProfilerStart()
        # Pushed right after cudaProfilerStart so it is the earliest in-window NVTX per globalTid
        # (enables pid to mesh-coord lookup); range, not mark, so nsys-ui renders on the thread row.
        d, t, parts = distributed, cfg.training, list()
        parts.append(f"rank={d.rank}")
        parts.append(
            f"pp={d.pp_rank}/{d.pp_size} dp={d.dp_rank}/{d.dp_size} "
            f"cp={d.cp_rank}/{d.cp_size} ep={d.ep_rank}/{d.ep_size}"
        )
        parts.append(f"mbs={t.micro_batch_size} seq={t.sequence_length}")
        torch.cuda.nvtx.range_push("; ".join(parts))
    start = cfg.training.memory_profile_start
    if start is not None and step == start:
        torch.cuda.memory._record_memory_history(max_entries=65536, stacks="python")

    device = torch.cuda.current_device()
    t0 = time.time()

    torch.cuda.memory.reset_peak_memory_stats()

    model, optimizers, schedulers = training.model, training.optimizers, training.schedulers
    model.train()

    dp_size = distributed.dp_size
    micro_batch_size = cfg.training.micro_batch_size
    global_batch_size = cfg.training.global_batch_size
    assert global_batch_size % (micro_batch_size * dp_size) == 0

    # Gather the part of the global batch this rank owns, split into micro-batches.
    microbatches = get_global_batch(cfg, dataset, step, device)

    # Run the forward and backward pass. The objective hands back one detached loss per
    # micro-batch on pipeline rank 0; every other rank gets an empty list.
    objective_outputs = model.step(microbatches, objective)

    # Token-weighted reduction. The objective returns a loss summed over the tokens of each
    # micro-batch, so dividing by the total non-ignored token count yields the correct token-mean
    # regardless of how tokens split across micro-batches. Every rank holds the same labels, so
    # each counts the same total for the gradient scale.
    counted = 0
    for mb in microbatches:
        (labels,) = mb.objective_inputs
        counted = counted + (labels != -100).sum()
    num_tokens = counted.clamp_min(1).to(device=device, dtype=torch.float32)

    cp_size = distributed.cp_size
    if distributed.pp_rank == 0:
        loss = torch.stack(objective_outputs).sum() / num_tokens
        if cp_size > 1:
            torch.distributed.all_reduce(loss, group=distributed.cp_group)
            loss /= cp_size

    # The one gradient normalization. FSDP reduces with a plain sum (see apply_fsdp), so dividing
    # by the global token count leaves every parameter, attn or expt, at the token-weighted mean.
    # Tokens split dp ways across the batch and cp ways along the sequence, so the global count
    # is the local count times dp * cp, which holds only while every rank counts the same number
    # of tokens: true for pretraining, not for packed data with -100 masks.
    scale = 1.0 / (num_tokens * distributed.dp_size * distributed.cp_size)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)

    # Clip the gradients.
    gradient_norm = clip_grad_norm(
        model, max_norm=1.0, norm_type=2, hsdp_replica=cfg.distributed.hsdp_replica
    )

    # Take an optimization step (composed optimizers + their schedulers).
    for optimizer, scheduler in zip(optimizers, schedulers):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
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
    logger = logging.stdout
    if distributed.rank == 0:
        max_steps = cfg.training.max_steps
        loss, lr = loss.item(), schedulers[0].get_last_lr()[0]
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
        activate_wandb(cfg)
        if logging.wandb is not None:
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

    # Everything below counts completed steps, one more than the index of the step just run.
    completed = step + 1

    # Stop the nsys and the memory profiler.
    stop = cfg.training.nsys_stop
    if stop is not None and completed == stop:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    stop = cfg.training.memory_profile_stop
    if stop is not None and completed == stop:
        rank = distributed.rank
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
        should_save |= completed % cfg.training.save_interval == 0
        should_save |= completed == cfg.training.max_steps
        if should_save:
            assert cfg.training.save_location is not None
            save_checkpoint(cfg.training.save_location, completed)

    # Run deferred GC here so cyclic collection never fires mid-forward/backward.
    gc.collect()


@record
def launch(cfg: PretrainLMCfg) -> None:
    """
    Launch the pretraining of a language model.
    """
    setup_logging(cfg)
    setup_distributed(cfg)
    dataset = setup_dataset(cfg)
    setup_training(cfg)
    logger = logging.stdout
    logger.info("launch(cfg=%s)" % cfg)
    step = find_checkpoint(cfg.training.save_location)
    if step is not None:
        load_checkpoint(cfg.training.save_location, step)
    elif cfg.training.hf_import_path is not None:
        load_hf_into_model(cfg.training.hf_import_path, training.model)
    step = step or 0
    gc.disable()
    while step < cfg.training.max_steps:
        train_step(cfg, dataset, step)
        step += 1
    gc.enable()
