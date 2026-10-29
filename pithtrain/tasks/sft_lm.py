"""Callable assistant-only SFT task using an explicitly initialized training context.

The caller constructs the Qwen3 model/engine, optimizers, schedulers and reader.
This task reuses ``model.step`` rather than copying the pretraining orchestrator.
The reader supplies already shifted/masked Microbatch labels; no shift occurs in
the objective. FSDP/engine replica gradients must use SUM, as in pretraining.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from pithtrain.config import SlottedDefault
from pithtrain.contexts import distributed, training
from pithtrain.modules.microbatch import Microbatch
from pithtrain.tasks.omni_objectives import IGNORE_INDEX, global_target_counts, text_loss_sum


@dataclass(init=False, slots=True)
class SFTLMCfg(SlottedDefault):
    """Absolute update/scan limits; finite scans prevent looping on ignored batches."""

    max_updates: int = 1
    max_batches: int = 1
    max_grad_norm: float | None = 1.0
    hsdp_replica: int = 1
    moe_load_balance_coef: float = 0.0


@dataclass(slots=True)
class SFTState:
    """Serializable scan/update cursors; save with model/optimizer/data/RNG state."""

    scanned_batches: int = 0
    completed_updates: int = 0

    def state_dict(self) -> dict[str, int]:
        return dict(scanned_batches=self.scanned_batches, completed_updates=self.completed_updates)

    def validate_state_dict(self, state: dict[str, int]) -> None:
        if (
            set(state) != {"scanned_batches", "completed_updates"}
            or any(type(value) is not int or value < 0 for value in state.values())
            or state["completed_updates"] > state["scanned_batches"]
        ):
            raise ValueError("Invalid SFT scan/update checkpoint state")

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.validate_state_dict(state)
        self.scanned_batches = state["scanned_batches"]
        self.completed_updates = state["completed_updates"]


class SFTData(Protocol):
    """Indexed scans are separate from successful optimizer-update commits.

    get_batch(scan_index, ...) must honor the explicit index, including after a
    skipped scan/resume. commit_update gets both zero-based indices; it must not
    infer the scanned data position from the update count. A global-zero scan
    never calls commit_update. The checkpoint callback persists SFTState even
    after such a scan, together with any reader state, avoiding silent replays.
    """

    def get_batch(self, scan_index: int, device: torch.device) -> list[Microbatch]: ...

    def commit_update(self, *, scan_index: int, update_index: int) -> None: ...


class SFTCheckpointState:
    """Concrete local task+reader state for native task-checkpoint version 2."""

    def __init__(self, state: SFTState, data: SFTData):
        self.state, self.data = state, data

    def state_dict(self):
        return dict(
            **self.state.state_dict(),
            data_cursor=self.data.state_dict()["scanned_batches"],
            reader_state=self.data.state_dict(),
        )

    def validate_state_dict(self, payload):
        if set(payload) != {"scanned_batches", "completed_updates", "data_cursor", "reader_state"}:
            raise ValueError("Invalid SFT native task checkpoint keys")
        cursor = {key: payload[key] for key in ("scanned_batches", "completed_updates")}
        self.state.validate_state_dict(cursor)
        self.data.validate_state_dict(payload["reader_state"])
        if (
            type(payload["data_cursor"]) is not int
            or payload["data_cursor"] != cursor["scanned_batches"]
        ):
            raise ValueError("SFT native data cursor differs from committed scans")
        if any(payload["reader_state"].get(key) != value for key, value in cursor.items()):
            raise ValueError("SFT task and reader checkpoint cursors differ")

    def load_state_dict(self, payload):
        self.validate_state_dict(payload)
        self.data.load_state_dict(payload["reader_state"])
        self.state.load_state_dict(
            {key: payload[key] for key in ("scanned_batches", "completed_updates")}
        )


@dataclass(frozen=True, slots=True)
class SFTStepResult:
    updated: bool
    global_targets: int
    loss: torch.Tensor | None


def objective(model_outputs, objective_inputs):
    """DualPipeV-compatible sum objective for already aligned assistant-only labels."""
    (logits,) = model_outputs
    (labels,) = objective_inputs
    loss = text_loss_sum(logits, labels).loss_sum
    return loss, loss.detach()


def reject_unsupported_auxiliary_loss(cfg, model, *, device, include_training_context=True) -> None:
    """Reject unsupported injected MoE loss collectively, before forward/update.

    Auxiliary objectives need their own masks/counts. The existing injector's
    backward is not scaled by the task's token objective, so dividing its gradient
    by assistant targets is incorrect. WORLD here synchronizes capability only;
    token counts still use the explicit nonduplicated DP x CP group.
    """
    unsupported = any(
        getattr(owner, "moe_load_balance_coef", 0.0) != 0
        for owner in ((cfg, training) if include_training_context else (cfg,))
    )
    if model is not None:
        for module in model.modules():
            injected = getattr(module, "load_balance_loss_fn", None)
            if injected is not None and getattr(injected, "lb_coef", None) != 0:
                unsupported = True
    flag = torch.tensor(int(unsupported), device=device, dtype=torch.int64)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        raise NotImplementedError(
            "SFT MoE auxiliary loss normalization is unsupported: use coefficient zero "
            "and no nonzero or unknown load_balance_loss_fn injector"
        )


def reject_nonfinite_gradients(model, *, device, group=None, norm=None) -> None:
    """All participating ranks reject before any optimizer/scheduler mutation."""
    invalid = torch.zeros((), dtype=torch.int64, device=device)
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
        invalid.bitwise_or_((~torch.isfinite(gradient).all()).to(torch.int64))
    if norm is not None:
        invalid.bitwise_or_((~torch.isfinite(norm).all()).to(device=device, dtype=torch.int64))
    if dist.is_initialized():
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=group)
    if invalid.item():
        raise FloatingPointError("Nonfinite gradients or norm: no optimizer/scheduler update")


def _admit_microbatches(microbatches, token_group):
    """Validate local input and exact stage-local DP x CP membership collectively."""
    error, labels, layout = None, [], None
    try:
        if not microbatches:
            raise ValueError(
                "Each rank needs microbatches, using -100 labels for zero local targets"
            )
        labels = [batch.objective_inputs[0] for batch in microbatches]
        if any(label.dtype != torch.long or label.device != labels[0].device for label in labels):
            raise ValueError("SFT labels must be int64 on one device")
        if any(((label < 0) & (label != IGNORE_INDEX)).any() for label in labels):
            raise ValueError("SFT labels must use -100 for ignored targets")
        if (
            not callable(getattr(training.model, "step", None))
            or not training.optimizers
            or len(training.optimizers) != len(training.schedulers)
        ):
            raise ValueError("Initialize the engine and matching optimizer/scheduler tuples")
        if dist.is_initialized():
            if token_group is None:
                raise ValueError("Explicit stage-local DP x CP token group required")
            layout = {
                key: getattr(distributed, key)
                for key in (
                    "rank",
                    "world_size",
                    "pp_rank",
                    "dp_rank",
                    "cp_rank",
                    "pp_size",
                    "dp_size",
                    "cp_size",
                )
            }
            if any(type(value) is not int for value in layout.values()):
                raise ValueError("Runtime topology coordinates must be integers")
            layout["members"] = dist.get_process_group_ranks(token_group)
    except Exception as exc:
        error = str(exc)
    records = [dict(error=error, layout=layout)]
    if dist.is_initialized():
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, dict(error=error, layout=layout))
    errors = [item["error"] for item in records if item["error"]]
    if not errors and dist.is_initialized():
        layouts = [item["layout"] for item in records]
        sizes = tuple(layouts[0][key] for key in ("pp_size", "dp_size", "cp_size"))
        world = len(layouts)
        coordinates = [(item["pp_rank"], item["dp_rank"], item["cp_rank"]) for item in layouts]
        if (
            any(size < 1 for size in sizes)
            or sizes[0] * sizes[1] * sizes[2] != world
            or len(set(coordinates)) != world
        ):
            errors.append("Runtime PP/DP/CP topology must cover world ranks exactly once")
        for rank, item in enumerate(layouts):
            coord = coordinates[rank]
            expected = [
                index for index, other in enumerate(layouts) if other["pp_rank"] == item["pp_rank"]
            ]
            if (
                item["rank"] != rank
                or item["world_size"] != world
                or tuple(item[key] for key in ("pp_size", "dp_size", "cp_size")) != sizes
                or any(not 0 <= value < size for value, size in zip(coord, sizes, strict=True))
                or sorted(item["members"]) != expected
            ):
                errors.append("Token group must cover unique DP x CP ranks at exactly one PP stage")
    if errors:
        raise ValueError("SFT collective admission rejected: " + "; ".join(errors))
    return labels


def train_step(cfg: SFTLMCfg, microbatches: list[Microbatch], *, token_group) -> SFTStepResult:
    """All ranks count first; zero-local ranks still execute the complete engine step."""
    labels = _admit_microbatches(microbatches, token_group)
    reject_unsupported_auxiliary_loss(
        cfg, getattr(training, "model", None), device=labels[0].device
    )
    local_count = sum(label.ne(IGNORE_INDEX).sum(dtype=torch.int64) for label in labels)
    count = global_target_counts(local_count, token_group=token_group)
    targets = int(count.item())
    if dist.is_initialized():
        totals = [None] * dist.get_world_size()
        dist.all_gather_object(totals, targets)
        if len(set(totals)) != 1:
            raise ValueError("SFT PP copies disagree on the global target count")
    if targets == 0:
        return SFTStepResult(False, 0, None)

    model, optimizers, schedulers = training.model, training.optimizers, training.schedulers
    if not optimizers or len(optimizers) != len(schedulers):
        raise ValueError("Initialize matching nonempty optimizer/scheduler tuples")
    model.train()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    outputs = model.step(microbatches, objective)
    loss = None
    if outputs:  # Only the objective-owning PP rank returns losses.
        total = torch.stack(outputs).sum().detach().float()
        if dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=token_group)
        loss = total / targets
    # Engines sum unnormalized microbatch and replica gradients. Divide once,
    # after their reduction, matching the pretraining FP32 master-gradient path.
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(1.0 / targets)
    reject_nonfinite_gradients(model, device=labels[0].device)
    if cfg.max_grad_norm is not None:
        if dist.is_initialized():
            from pithtrain.modules.optimizer import clip_grad_norm

            norm = clip_grad_norm(model, cfg.max_grad_norm, hsdp_replica=cfg.hsdp_replica)
            reject_nonfinite_gradients(model, device=labels[0].device, norm=norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
            )
    for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    return SFTStepResult(True, targets, loss)


def launch(
    cfg: SFTLMCfg,
    data: SFTData,
    *,
    device: torch.device,
    token_group,
    state: SFTState | None = None,
    checkpoint: Callable[[dict[str, int]], None] | None = None,
    checkpoint_root: str | None = None,
    resume_scan: int | None = None,
    checkpoint_contract: dict | None = None,
) -> SFTState:
    """Run SFT against the initialized context; never rebuild or silently rescan it.

    The caller restores model/optimizer/scheduler/RNG/data before passing restored
    state. checkpoint receives a fresh cursor dict after every consumed scan,
    including skipped ones; it owns the actual checkpoint transaction. Reaching
    max_batches may return fewer updates than requested, explicitly in state.
    checkpoint_root additionally uses the native strict task-v2 protocol, with
    an explicit stable contract and resume_scan; scans and updates stay distinct.
    """
    if any(type(value) is not int or value < 0 for value in (cfg.max_updates, cfg.max_batches)):
        raise ValueError("SFT limits must be nonnegative integers")
    if cfg.max_grad_norm is not None and (
        not isinstance(cfg.max_grad_norm, (int, float)) or not 0 < cfg.max_grad_norm < float("inf")
    ):
        raise ValueError("max_grad_norm must be positive and finite, or None")
    state = SFTState() if state is None else state
    state.load_state_dict(state.state_dict())
    adapter = None
    if checkpoint_root is not None:
        if not isinstance(checkpoint_contract, dict) or not checkpoint_contract:
            raise ValueError("Native checkpoint needs an explicit stable model/topology contract")
        from pithtrain.modules.checkpoint import load_checkpoint
        from pithtrain.modules.task_checkpoint import TaskCheckpointState

        checkpoint_contract = dict(
            checkpoint_contract,
            sft_objective=dict(
                version=1,
                alignment="already-shifted-assistant-targets",
                reduction="global-sum-count",
                max_grad_norm=cfg.max_grad_norm,
                hsdp_replica=cfg.hsdp_replica,
                moe_load_balance_coef=cfg.moe_load_balance_coef,
            ),
        )
        adapter = TaskCheckpointState(
            SFTCheckpointState(state, data),
            task_kind="sft",
            task_schema=1,
            contract=checkpoint_contract,
            transaction_key="scanned_batches",
            update_key="completed_updates",
            data_key="data_cursor",
        )
        if resume_scan is not None:
            if type(resume_scan) is not int or resume_scan < 0:
                raise ValueError("resume_scan must identify a committed scan")
            load_checkpoint(
                checkpoint_root,
                resume_scan,
                data_state=adapter,
                strict=True,
                resume_config=checkpoint_contract,
            )
        adapter.validate_state_dict(adapter.state_dict())
    elif resume_scan is not None:
        raise ValueError("resume_scan requires a native checkpoint root")
    while state.completed_updates < cfg.max_updates and state.scanned_batches < cfg.max_batches:
        scan, update = state.scanned_batches, state.completed_updates
        batches = data.get_batch(scan, device)
        result = train_step(cfg, batches, token_group=token_group)
        if result.updated:
            data.commit_update(scan_index=scan, update_index=update)
            state.completed_updates += 1
        state.scanned_batches += 1
        if adapter is not None:
            from pithtrain.modules.checkpoint import save_checkpoint

            save_checkpoint(
                checkpoint_root,
                state.scanned_batches,
                data_state=adapter,
                strict=True,
                resume_config=checkpoint_contract,
            )
        if checkpoint is not None:
            checkpoint(state.state_dict())
    return state
