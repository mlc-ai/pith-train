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
from pithtrain.kernels import tune_kernels
from pithtrain.modules.checkpoint import (
    find_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from pithtrain.modules.dataset import ConcatDataset, MemmapDataset
from pithtrain.modules.distributed import DistributedCfg, setup_distributed
from pithtrain.modules.fp8_param_gather import precompute_fp8_param_scales
from pithtrain.modules.hf_loader import load_hf_into_model
from pithtrain.modules.load_balance import MoELoadBalanceLossTracker
from pithtrain.modules.logging import LoggingCfg, activate_wandb, setup_logging
from pithtrain.modules.optimizer import clip_grad_norm, local_grads, scale_grads_
from pithtrain.modules.training import TrainingCfg, setup_training
from pithtrain.operators.cp_sequence import zigzag_spans
from pithtrain.operators.cross_entropy import cross_entropy
from pithtrain.pipeline import Microbatch

__all__ = [
    "PretrainLMCfg",
    "launch",
    "setup_dataset",
    "get_global_batch",
    "objective",
    "train_step",
]


@dataclass(init=False, slots=True)
class PretrainLMCfg(SlottedDefault):
    distributed: DistributedCfg = field(default_factory=DistributedCfg)

    training: TrainingCfg = field(default_factory=TrainingCfg)

    logging: LoggingCfg = field(default_factory=LoggingCfg)

    dataset: Path
    """
    The root directory hosting the tokenized corpus, globbed for *.bin shards.
    """


def setup_dataset(cfg: PretrainLMCfg) -> ConcatDataset:
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
    The samples depend only on step and dp_rank, and each sample holds this rank's zigzag CP blocks.
    Every pipeline rank builds the same list, so DualPipeV needs no broadcast of shapes; only
    pipeline rank 0 feeds the tensors to the model.
    """
    micro_batch_size = cfg.training.micro_batch_size
    global_batch_size = cfg.training.global_batch_size
    dp_size = distributed.dp_size
    dp_rank = distributed.dp_rank
    sequence_length = cfg.training.sequence_length

    effective_batch_size = micro_batch_size * dp_size
    local_batch_size = global_batch_size // dp_size
    start0 = step * global_batch_size + dp_rank * micro_batch_size

    # Two blocks per rank under zigzag CP; at cp_size 1 they are adjacent and cover the sequence.
    front, back = zigzag_spans(distributed.cp_rank, distributed.cp_size, sequence_length)
    block = len(front)
    local_seq_len = 2 * block

    local_tokens = torch.empty((local_batch_size, local_seq_len), dtype=torch.long)
    local_labels = torch.empty((local_batch_size, local_seq_len), dtype=torch.long)

    for row in range(local_batch_size):
        microbatch_idx, row_in_microbatch = divmod(row, micro_batch_size)
        index = start0 + microbatch_idx * effective_batch_size + row_in_microbatch
        tokens_a, labels_a = dataset.get_chunk(index, front.start, block)
        tokens_b, labels_b = dataset.get_chunk(index, back.start, block)
        local_tokens[row, :block] = tokens_a
        local_tokens[row, block:] = tokens_b
        local_labels[row, :block] = labels_a
        local_labels[row, block:] = labels_b

    local_tokens = local_tokens.to(device, non_blocking=True)
    local_labels = local_labels.to(device, non_blocking=True)

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
    The loss is summed over the micro-batch's tokens, not averaged: train_step divides by the
    global token count.
    """
    (logits,) = model_outputs
    (labels,) = objective_inputs
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    # Compute the sum before storing BF16 logit gradients. Multiplying a
    # previously rounded BF16 mean by the token count loses information.
    loss = cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
    return loss, loss.detach()


def train_step(cfg: PretrainLMCfg, dataset: ConcatDataset, step: int) -> None:
    start = cfg.training.nsys_start
    if start is not None and step == start:
        torch.cuda.cudart().cudaProfilerStart()
        # The first NVTX range in the profile window names each thread's mesh coordinates; a range,
        # not a mark, so the nsys UI shows it on the thread row.
        mesh, train_cfg, parts = distributed, cfg.training, list()
        parts.append(f"rank={mesh.rank}")
        parts.append(
            f"pp={mesh.pp_rank}/{mesh.pp_size} dp={mesh.dp_rank}/{mesh.dp_size} "
            f"cp={mesh.cp_rank}/{mesh.cp_size} ep={mesh.ep_rank}/{mesh.ep_size}"
        )
        parts.append(f"mbs={train_cfg.micro_batch_size} seq={train_cfg.sequence_length}")
        torch.cuda.nvtx.range_push("; ".join(parts))
    start = cfg.training.memory_profile_start
    if start is not None and step == start:
        torch.cuda.memory._record_memory_history(max_entries=65536, stacks="python")

    device = torch.cuda.current_device()
    step_start_time = time.time()

    torch.cuda.memory.reset_peak_memory_stats()

    model, optimizers, schedulers = training.model, training.optimizers, training.schedulers
    model.train()

    dp_size = distributed.dp_size
    micro_batch_size = cfg.training.micro_batch_size
    global_batch_size = cfg.training.global_batch_size
    assert global_batch_size % (micro_batch_size * dp_size) == 0

    microbatches = get_global_batch(cfg, dataset, step, device)

    # One detached loss per micro-batch on pipeline rank 0, an empty list on the other ranks.
    objective_outputs = model.step(microbatches, objective)

    # Every pipeline rank holds the same labels, so all stages count the same total for the scale.
    counted = 0
    for microbatch in microbatches:
        (labels,) = microbatch.objective_inputs
        counted = counted + (labels != -100).sum()
    num_tokens = counted.clamp_min(1).to(device=device, dtype=torch.float32)

    cp_size = distributed.cp_size
    if distributed.pp_rank == 0:
        # Log the same global token mean that scales the gradients. Each
        # objective output is detached; this reduction only affects reporting.
        loss = torch.stack(objective_outputs).sum()
        if dp_size * cp_size > 1:
            token_group = distributed.attn_mesh["dp", "cp"]._flatten().get_group()
            torch.distributed.all_reduce(loss, group=token_group)
        loss /= num_tokens * dp_size * cp_size

    # The one gradient normalization: FSDP reduces with a plain sum, so this gives the
    # token-weighted mean. The global count is the local count times dp * cp only while every rank
    # counts the same number of tokens: true for pretraining, not for packed data with -100 masks.
    scale = 1.0 / (num_tokens * distributed.dp_size * distributed.cp_size)
    scale_grads_(local_grads(model), scale)

    gradient_norm = clip_grad_norm(
        model, max_norm=1.0, norm_type=2, hsdp_replica=cfg.distributed.hsdp_replica
    )

    for optimizer, scheduler in zip(optimizers, schedulers):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    if training.fp8:
        precompute_fp8_param_scales()
    MoELoadBalanceLossTracker.reset()

    step_end_time = time.time()
    elapsed_tensor = torch.tensor(step_end_time - step_start_time, device=device)
    torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
    elapsed = elapsed_tensor.item()

    peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
    peak_gpu_mem = torch.tensor(peak_gpu_mem, device=device)
    torch.distributed.all_reduce(peak_gpu_mem, op=torch.distributed.ReduceOp.MAX)

    # The tracked values include the coefficient; dividing it out logs E * dot(f, p), where 1.0
    # is perfect balance (the Megatron-LM convention).
    moe_load_balance_coef = cfg.training.moe_load_balance_coef
    lb_total, lb_count = MoELoadBalanceLossTracker.get_total_count_and_clear()
    if moe_load_balance_coef > 0:
        lb_stats = torch.tensor([lb_total, lb_count], device=device)
        torch.distributed.all_reduce(lb_stats, op=torch.distributed.ReduceOp.SUM)
        lb_loss = (lb_stats[0] / lb_stats[1]).item() / moe_load_balance_coef
    else:
        lb_loss = 0.0

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

    completed = step + 1

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

    if cfg.training.save_interval is not None:
        should_save = False
        should_save |= completed % cfg.training.save_interval == 0
        should_save |= completed == cfg.training.max_steps
        if should_save:
            assert cfg.training.save_location is not None
            save_checkpoint(cfg.training.save_location, completed)

    # Collect here so cyclic collection never fires mid-forward or mid-backward.
    gc.collect()


@record
def launch(cfg: PretrainLMCfg) -> None:
    setup_logging(cfg)
    setup_distributed(cfg)
    dataset = setup_dataset(cfg)
    setup_training(cfg)
    logger = logging.stdout
    logger.info("launch(cfg=%s)" % cfg)
    step = find_checkpoint(cfg.training.save_location)
    if step is not None:
        load_checkpoint(cfg.training.save_location, step)
        if training.fp8:
            precompute_fp8_param_scales()
    elif cfg.training.hf_import_path is not None:
        load_hf_into_model(cfg.training.hf_import_path, training.model)
        if training.fp8:
            precompute_fp8_param_scales()
    step = step or 0
    # Freeze the startup heap (model, datasets, compiled graphs) out of the collector, so the
    # per-step gc.collect() in train_step scans only what the step allocated.
    gc.collect()
    gc.freeze()
    gc.disable()
    # Install the cached kernel launch configs now; tune the ones the first step lacked after it.
    tune_kernels()
    first_step = step
    while step < cfg.training.max_steps:
        train_step(cfg, dataset, step)
        if step == first_step:
            tune_kernels()
        step += 1
    gc.enable()
