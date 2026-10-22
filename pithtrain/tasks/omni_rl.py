"""Bounded on-policy GRPO training with an external rollout/reward boundary.

This task trains a Thinker text-token policy, optionally with media context. It
does not implement a serving transport, RVQ/audio policy, or a verl/Slime backend.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.distributed as dist

from pithtrain.config import SlottedDefault
from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.resume_semantics import (
    callable_contract,
    implementation_contract,
    model_contract,
    scheduler_contract,
    seal,
)
from pithtrain.modules.strict_checkpoint import (
    _check_schema,
    _restore_rng,
    _rng_state,
    _validate_rng,
)
from pithtrain.tasks.omni_objectives import global_target_counts


def fingerprint(value) -> str:
    """Hash an explicit JSON config; callers supply actual template/processor/position data."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RolloutContract:
    chat_template_sha256: str
    processor_sha256: str
    position_config_sha256: str
    reference_policy_sha256: str
    reward_config_sha256: str

    def validate(self):
        for name, value in asdict(self).items():
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be an explicit SHA256 fingerprint")


@dataclass(frozen=True)
class PolicyIdentity:
    version: int
    weights_sha256: str
    contract: RolloutContract
    semantics_sha256: str

    def validate(self):
        self.contract.validate()
        if type(self.version) is not int or self.version < 0:
            raise ValueError("Invalid policy version")
        digest = self.weights_sha256
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Policy snapshot must identify exact exported weights")
        digest = self.semantics_sha256
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Policy snapshot must identify exact model/forward semantics")


@dataclass(frozen=True)
class PolicySnapshot:
    identity: PolicyIdentity
    # Transport is caller-owned: a CPU state dict for a toy policy, or a reviewed
    # checkpoint/shard descriptor for a real distributed serving adapter.
    payload: object
    semantics: dict


@dataclass(frozen=True)
class RolloutRequest:
    identity: PolicyIdentity
    request_index: int
    data_cursor: int
    seed: int
    group_size: int
    max_samples: int
    max_sequence_length: int
    dp_rank: int
    pp_rank: int
    rollout_leader: bool


@dataclass
class RolloutBatch:
    identity: PolicyIdentity
    request_index: int
    input_ids: torch.Tensor
    sampled_token_ids: torch.Tensor
    token_mask: torch.Tensor
    old_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    group_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    # Explicit per-microbatch payloads avoid guessing patch/frame slicing.
    model_context: tuple[dict[str, torch.Tensor] | None, ...] | None = None
    media_inputs: tuple[dict[str, torch.Tensor] | None, ...] | None = None


class ExternalRollout(Protocol):
    """Serving integration must enforce its own I/O deadlines and persist its cursor/RNG.

    A native PP adapter calls this on every rank. Only rollout_leader may create
    external requests; followers must receive the same token batch, old/ref
    logprobs and metadata via the caller's transport. Media payload stays on PP0.
    synchronize must acknowledge installed weights, not just echo a requested ID.
    """

    def synchronize(self, snapshot: PolicySnapshot) -> PolicyIdentity: ...

    def generate(self, request: RolloutRequest) -> RolloutBatch: ...

    def state_dict(self) -> dict: ...

    def validate_state_dict(self, state: dict) -> None: ...

    def load_state_dict(self, state: dict) -> None: ...


class ExternalReward(Protocol):
    def score(self, request: RolloutRequest, batch: RolloutBatch) -> torch.Tensor: ...

    def state_dict(self) -> dict: ...

    def validate_state_dict(self, state: dict) -> None: ...

    def load_state_dict(self, state: dict) -> None: ...


@dataclass(init=False, slots=True)
class OmniRLCfg(SlottedDefault):
    max_updates: int = 1
    max_requests: int = 1
    group_size: int = 2
    max_samples: int = 32
    max_sequence_length: int = 4096
    micro_batch_size: int = 1
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    advantage_epsilon: float = 1e-6
    max_grad_norm: float = 1.0
    seed: int = 1234

    def validate(self):
        for name in (
            "max_updates",
            "max_requests",
            "max_samples",
            "max_sequence_length",
            "micro_batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("GRPO requires at least two completions per local group")
        if not 0 < self.clip_epsilon < 1 or not 0 <= self.kl_beta or not 0 < self.advantage_epsilon:
            raise ValueError("Invalid GRPO coefficients")
        if not 0 < self.max_grad_norm or not all(
            math.isfinite(x)
            for x in (self.clip_epsilon, self.kl_beta, self.advantage_epsilon, self.max_grad_norm)
        ):
            raise ValueError("Coefficients must be finite")


def group_relative_advantages(rewards, group_ids, group_size, epsilon):
    """Population-standardized rewards; complete groups must be local to one DP rank."""
    if type(group_size) is not int or group_size < 2 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid group size or advantage epsilon")
    if rewards.ndim != 1 or rewards.numel() != len(group_ids) or not rewards.is_floating_point():
        raise ValueError("One floating reward per completion is required")
    if rewards.requires_grad or not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be detached and finite")
    result = torch.empty_like(rewards)
    for group in dict.fromkeys(group_ids):
        indices = [i for i, value in enumerate(group_ids) if value == group]
        if len(indices) != group_size:
            raise ValueError(
                "Incomplete local reward group; groups spanning DP ranks are unsupported"
            )
        values = rewards[indices]
        result[indices] = (values - values.mean()) / (values.std(unbiased=False) + epsilon)
    return result


def grpo_loss_sum(logits, sampled_tokens, old, reference, mask, advantages, cfg):
    """Token-sum clipped surrogate plus k3 KL; no sequence shift or hidden averaging."""
    if logits.ndim != 3 or logits.shape[:2] != sampled_tokens.shape:
        raise ValueError("Expected aligned B,S,V logits and B,S sampled token IDs")
    if mask.dtype != torch.bool or mask.shape != sampled_tokens.shape:
        raise ValueError("Expected aligned boolean token mask")
    if (
        old.shape != mask.shape
        or reference.shape != mask.shape
        or advantages.shape != (mask.shape[0],)
    ):
        raise ValueError("Old/reference logprobs or advantages are not aligned")
    if old.requires_grad or reference.requires_grad or advantages.requires_grad:
        raise ValueError("Rollout/reference/reward tensors must be detached")
    if not torch.isfinite(logits).all() or not torch.isfinite(advantages).all():
        raise ValueError("Nonfinite policy logits or advantages")
    if not mask.any():
        return logits.sum() * 0.0
    if (
        sampled_tokens.dtype != torch.long
        or (sampled_tokens[mask] < 0).any()
        or (sampled_tokens[mask] >= logits.shape[-1]).any()
    ):
        raise ValueError("Sampled token outside policy vocabulary")
    if not torch.isfinite(old[mask]).all() or not torch.isfinite(reference[mask]).all():
        raise ValueError("Nonfinite valid old/reference logprobs")
    safe_tokens = torch.where(mask, sampled_tokens, 0)
    current = logits.float().log_softmax(-1).gather(-1, safe_tokens.unsqueeze(-1)).squeeze(-1)
    current = current[mask]
    old, reference = old[mask].float(), reference[mask].float()
    advantage = advantages[:, None].expand_as(mask)[mask].float()
    ratio = (current - old).exp()
    delta = reference - current
    kl = delta.exp() - delta - 1
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantage
    terms = -torch.minimum(unclipped, clipped) + cfg.kl_beta * kl
    if not torch.isfinite(terms).all():
        raise ValueError("Nonfinite GRPO ratio/KL; refusing an overflowed update")
    return terms.sum()


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _weights_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _collective_error(error, group=None):
    errors = [str(error) if error is not None else None]
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, errors[0], group=group)
        errors = gathered
    if any(errors):
        raise ValueError(f"RL validation failed before the next collective/update: {errors}")


def _optimizer_contract(model, optimizers):
    names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    covered, contracts = [], []
    for optimizer in optimizers:
        groups = []
        for group in optimizer.param_groups:
            ids = [id(p) for p in group["params"]]
            if any(i not in names for i in ids):
                raise ValueError("Optimizer contains a foreign/frozen policy parameter")
            covered.extend(ids)
            groups.append(
                dict(
                    names=[names[i] for i in ids],
                    options={k: v for k, v in group.items() if k not in {"params", "lr"}},
                )
            )
        contracts.append(
            dict(
                type=f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                defaults=copy.deepcopy(optimizer.defaults),
                groups=groups,
            )
        )
    if not covered or len(covered) != len(set(covered)) or set(covered) != set(names):
        raise ValueError("Optimizer policy coverage is missing or duplicated")
    return contracts


def _local_rng_state():
    # CPU runs must not initialize CUDA. Once initialized, preserve every visible
    # default CUDA generator, including generators used by a rollout provider.
    state = _rng_state(torch.device("cpu"))
    state["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return state


def _validate_local_rng(state):
    # Validate using isolated generators before mutating any live model/provider
    # or process-global random state. CUDA visibility must match the saved run.
    _check_schema(state, _local_rng_state(), path="app.resume.rng")
    _validate_rng(state, torch.device("cpu"))
    if state["cuda"] is not None:
        for index, value in enumerate(state["cuda"]):
            torch.Generator(device=f"cuda:{index}").set_state(value)


def _restore_local_rng(state):
    _restore_rng(state, torch.device("cpu"))
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class TorchPolicyAdapter:
    """Actual local training/checkpoint adapter for caller-owned forward models.

    Distributed production uses PithTrainPolicyAdapter, not implicit DDP averaging.
    """

    dp_rank = pp_rank = 0
    pp_size = 1
    token_group = validation_group = None

    def __init__(self, model, optimizer, *, forward=None, scheduler=None):
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Use the explicit PithTrain adapter for distributed policy training")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.optimizer_contract = _optimizer_contract(model, (optimizer,))
        self.forward = forward
        devices = {value.device for value in (*model.parameters(), *model.buffers())}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise ValueError("Local RL policy requires one CPU or CUDA model device")
        self.device = next(iter(devices))

    def model_semantics(self):
        model = model_contract(self.model)
        forward = (
            dict(default="model(*inputs, **context)", adapter=implementation_contract(type(self)))
            if self.forward is None
            else callable_contract(self.forward, bound_objects=((self.model, model),))
        )
        return seal(dict(schema=1, model=model, forward=forward))

    def resume_semantics(self):
        return dict(model=self.model_semantics(), scheduler=scheduler_contract(self.scheduler))

    def snapshot(self, version, contract):
        weights = _cpu_tree(self.model.state_dict())
        semantics = self.model_semantics()
        identity = PolicyIdentity(version, _weights_digest(weights), contract, semantics["sha256"])
        return PolicySnapshot(identity, weights, semantics)

    def validate_batch_layout(self, batch, cfg, rewards):
        return None

    def backward(self, microbatches, objective):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        outputs = []
        for mb in microbatches:
            context = mb.context_for_stage(0) or {}
            result = (
                self.model(*mb.model_inputs, **context)
                if self.forward is None
                else self.forward(mb.model_inputs, context)
            )
            logits = result[0] if isinstance(result, tuple) else result
            loss, metric = objective((logits,), mb.objective_inputs)
            loss.backward()
            outputs.append(metric)
        return outputs

    def update(self, count, max_grad_norm):
        for p in self.model.parameters():
            if p.requires_grad:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise ValueError("Missing/nonfinite policy gradient")
                p.grad.div_(count)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Nonfinite policy after optimizer update; restore before continuing")
        return float(norm)

    def save(self, path, state):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        payload = dict(
            schema=3,
            semantics=self.resume_semantics(),
            model=_cpu_tree(self.model.state_dict()),
            optimizer=_cpu_tree(self.optimizer.state_dict()),
            optimizer_contract=self.optimizer_contract,
            scheduler=None if self.scheduler is None else self.scheduler.state_dict(),
            rl=state.state_dict(),
            rng=_local_rng_state(),
        )
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        raw = buffer.getvalue()
        (path / "state.pt").write_bytes(raw)
        (path / "sha256").write_text(hashlib.sha256(raw).hexdigest())

    def load(self, path, state):
        path = Path(path)
        raw = (path / "state.pt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != (path / "sha256").read_text():
            raise ValueError("RL checkpoint checksum mismatch")
        # RNG byte tensors stay on CPU; model/optimizer loaders copy their own
        # tensors to the model device. Mapping all tensors to CUDA corrupts RNG.
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if payload["schema"] != 3:
            raise ValueError("Unsupported RL checkpoint schema")
        if payload["semantics"] != self.resume_semantics():
            raise ValueError("RL model/forward/scheduler semantics differ")
        _validate_local_rng(payload["rng"])
        state.validate_state_dict(payload["rl"])
        if payload["optimizer_contract"] != self.optimizer_contract:
            raise ValueError("RL optimizer type/options/parameter order differ")
        template = self.optimizer.state_dict()
        saved = payload["optimizer"]
        if len(saved["param_groups"]) != len(template["param_groups"]) or any(
            left["params"] != right["params"]
            for left, right in zip(saved["param_groups"], template["param_groups"], strict=True)
        ):
            raise ValueError("RL optimizer parameter membership differs")
        for item in saved["state"].values():
            if any(
                isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in item.values()
            ):
                raise ValueError("RL optimizer checkpoint has nonfinite state")
        if type(self.optimizer) is torch.optim.AdamW:
            membership = {
                index: (parameter, group)
                for saved_group, group in zip(
                    template["param_groups"], self.optimizer.param_groups, strict=True
                )
                for index, parameter in zip(saved_group["params"], group["params"], strict=True)
            }
            expected = set(membership) if payload["rl"]["updates"] else set()
            if set(saved["state"]) != expected:
                raise ValueError("RL Adam state coverage differs from completed update count")
            for index, item in saved["state"].items():
                parameter, group = membership[index]
                moments = {"exp_avg", "exp_avg_sq"} | (
                    {"max_exp_avg_sq"} if group["amsgrad"] else set()
                )
                if (
                    set(item) != moments | {"step"}
                    or item["step"].numel() != 1
                    or item["step"].item() != payload["rl"]["updates"]
                ):
                    raise ValueError("RL Adam step differs from committed policy version")
                if any(
                    item[key].shape != parameter.shape or item[key].dtype != parameter.dtype
                    for key in moments
                ):
                    raise ValueError("RL Adam moment shape/dtype differs")
        current = self.model.state_dict()
        if current.keys() != payload["model"].keys() or any(
            current[k].shape != v.shape
            or current[k].dtype != v.dtype
            or not torch.isfinite(v).all()
            for k, v in payload["model"].items()
        ):
            raise ValueError("RL checkpoint model layout/values differ")
        if (self.scheduler is None) != (payload["scheduler"] is None):
            raise ValueError("RL scheduler presence differs")
        old = (
            _cpu_tree(current),
            copy.deepcopy(self.optimizer.state_dict()),
            None if self.scheduler is None else copy.deepcopy(self.scheduler.state_dict()),
            state.state_dict(),
            _local_rng_state(),
        )
        try:
            self.model.load_state_dict(payload["model"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(payload["scheduler"])
            state.load_state_dict(payload["rl"])
            _restore_local_rng(payload["rng"])
        except BaseException:
            self.model.load_state_dict(old[0])
            self.optimizer.load_state_dict(old[1])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(old[2])
            state.load_state_dict(old[3])
            _restore_local_rng(old[4])
            raise


class PithTrainPolicyAdapter:
    """Native DualPipeV/FSDP-SUM adapter; CP1, locally complete reward groups.

    The caller configures setup_training and supplies a collective-safe publisher
    of real policy weights. No automatic full-model CPU/GPU copy is made here.
    Actual serving transport and GPU validation remain separate integration work.
    """

    def __init__(
        self,
        model,
        optimizers,
        schedulers,
        *,
        device,
        token_group,
        pp_group,
        pp_rank,
        pp_size,
        dp_rank,
        export_policy,
        clip_grad,
        validation_group=None,
        training_config=None,
    ):
        if token_group is None:
            raise ValueError("Explicit stage-local DP token group required; WORLD may duplicate PP")
        self.model, self.optimizers, self.schedulers = model, optimizers, schedulers
        if not optimizers or len(optimizers) != len(schedulers):
            raise ValueError("Matching nonempty optimizer/scheduler tuples required")
        self.optimizer_contract = _optimizer_contract(model, optimizers)
        self.device, self.token_group, self.pp_group = device, token_group, pp_group
        self.pp_rank, self.pp_size, self.dp_rank = pp_rank, pp_size, dp_rank
        self.export_policy, self.clip_grad = export_policy, clip_grad
        self.validation_group = validation_group
        self.training_config = copy.deepcopy(training_config)

    @classmethod
    def from_training_context(cls, *, export_policy, training_cfg, hsdp_replica=1):
        from pithtrain.contexts import distributed, training
        from pithtrain.modules.optimizer import clip_grad_norm

        if distributed.cp_size != 1:
            raise ValueError("Native GRPO context parallel rollout slicing is not implemented")
        if hsdp_replica != 1:
            raise ValueError("Native GRPO transactional resume currently requires HSDP replicas=1")
        if training_cfg.moe_load_balance_coef != 0 or training_cfg.benchmark:
            raise ValueError(
                "This GRPO objective requires auxiliary load-balance loss off and real routing"
            )
        return cls(
            training.model,
            training.optimizers,
            training.schedulers,
            device=torch.device("cuda", torch.cuda.current_device()),
            token_group=distributed.attn_mesh["dp", "cp"]._flatten().get_group(),
            pp_group=distributed.pp_group,
            pp_rank=distributed.pp_rank,
            pp_size=distributed.pp_size,
            dp_rank=distributed.dp_rank,
            export_policy=export_policy,
            training_config=training_cfg.to_json_dict(),
            clip_grad=lambda maximum: clip_grad_norm(training.model, maximum, 2, hsdp_replica),
        )

    def model_semantics(self):
        return seal(
            dict(
                schema=1,
                model=model_contract(self.model),
                forward=dict(
                    adapter=implementation_contract(type(self)),
                    step=implementation_contract(type(self.model).step),
                ),
                training=self.training_config,
            )
        )

    def resume_semantics(self):
        return dict(
            model=self.model_semantics(),
            schedulers=[scheduler_contract(scheduler) for scheduler in self.schedulers],
        )

    def snapshot(self, version, contract):
        # This is a collective publication boundary, not a hidden collective in
        # state_dict/resume_config. Bind every PP/EP shard's numerical contract.
        local, error = None, None
        try:
            local = self.model_semantics()
        except Exception as exc:
            error = str(exc)
        _collective_error(error, self.validation_group)
        ranks = [local]
        if dist.is_initialized():
            ranks = [None] * dist.get_world_size(self.validation_group)
            dist.all_gather_object(ranks, local, group=self.validation_group)
        semantics = seal(dict(schema=1, kind="native_policy_ranks", ranks=ranks))
        snapshot = self.export_policy(version, contract, semantics)
        if not isinstance(snapshot, PolicySnapshot):
            raise ValueError("Policy publisher must return a real exported-weight snapshot")
        if (
            snapshot.semantics != semantics
            or snapshot.identity.semantics_sha256 != semantics["sha256"]
        ):
            raise ValueError("Policy publisher omitted or changed model/forward semantics")
        return snapshot

    def validate_batch_layout(self, batch, cfg, rewards):
        if dist.is_initialized():
            peers = [None] * dist.get_world_size(self.token_group)
            dist.all_gather_object(peers, (self.pp_rank, self.dp_rank), group=self.token_group)
            error = None
            if any(pp != self.pp_rank for pp, _ in peers) or len({dp for _, dp in peers}) != len(
                peers
            ):
                error = "Token-count group must contain unique DP ranks at one PP stage"
            _collective_error(error, self.validation_group)
        if self.pp_size > 1:
            if self.pp_group is None:
                raise ValueError("Explicit PP group required for duplicate-batch validation")
            payload = dict(
                identity=asdict(batch.identity),
                sample_ids=batch.sample_ids,
                group_ids=batch.group_ids,
                request=batch.request_index,
                tensors=_weights_digest(
                    {
                        name: getattr(batch, name).detach().cpu()
                        for name in (
                            "input_ids",
                            "sampled_token_ids",
                            "token_mask",
                            "old_logprobs",
                            "reference_logprobs",
                        )
                    }
                ),
                context=[_weights_digest(_cpu_tree(x or {})) for x in (batch.model_context or ())],
                rewards=_weights_digest({"rewards": rewards.detach().cpu()}),
            )
            gathered = [None] * self.pp_size
            dist.all_gather_object(gathered, fingerprint(payload), group=self.pp_group)
            if len(set(gathered)) != 1:
                raise ValueError(
                    "Pipeline ranks received different rollout tokens/positions/policy"
                )
        if self.pp_rank != 0 and batch.media_inputs:
            raise ValueError("Only PP0 may carry encoder media payloads")

    def backward(self, microbatches, objective):
        self.model.train()
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        return self.model.step(microbatches, objective)

    def update(self, count, max_grad_norm):
        error = None
        for p in self.model.parameters():
            if p.requires_grad:
                if (
                    p.grad is None
                    or not torch.isfinite(
                        p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                    ).all()
                ):
                    error = "Missing/nonfinite native policy gradient"
                else:
                    p.grad.div_(count)  # FSDP already SUM-reduced; no extra DP/EP/PP factor.
        _collective_error(error, self.validation_group)
        norm = self.clip_grad(max_grad_norm)
        _collective_error(
            None if torch.isfinite(norm).all() else "Nonfinite clipped gradient norm",
            self.validation_group,
        )
        for optimizer, scheduler in zip(self.optimizers, self.schedulers, strict=True):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        error = None
        for p in self.model.parameters():
            local = p.to_local() if hasattr(p, "to_local") else p
            if not torch.isfinite(local).all():
                error = "Nonfinite native policy after optimizer update; restore required"
        _collective_error(error, self.validation_group)
        return float(norm)

    def save(self, path, state):
        from pithtrain.modules.checkpoint import save_checkpoint

        save_checkpoint(
            Path(path),
            state.requests,
            data_state=self._checkpoint_state(state),
            strict=True,
            resume_config=state.resume_config(),
        )

    def load(self, path, state, *, request_index):
        from pithtrain.modules.checkpoint import load_checkpoint

        load_checkpoint(
            Path(path),
            request_index,
            data_state=self._checkpoint_state(state),
            strict=True,
            resume_config=state.resume_config(),
        )

    @staticmethod
    def _checkpoint_state(state):
        from pithtrain.modules.task_checkpoint import TaskCheckpointState

        return TaskCheckpointState(
            state,
            task_kind="omni_grpo",
            task_schema=1,
            contract=state.resume_config(),
            transaction_key="requests",
            update_key="updates",
            data_key="data_cursor",
            policy_version_key="policy_version",
        )


class OmniRLTrainer:
    """One bounded request -> reward -> update transaction, with explicit resume state."""

    def __init__(self, cfg, *, contract, backend, rollout, reward):
        cfg.validate()
        contract.validate()
        self.cfg, self.contract, self.backend = cfg, contract, backend
        self.rollout, self.reward = rollout, reward
        self.requests = self.updates = self.policy_version = self.data_cursor = 0
        self._in_flight = self._failed = False

    def resume_config(self):
        return dict(
            algorithm="token_mean_grpo_clipped_k3_v1",
            cfg=self.cfg.to_json_dict(),
            contract=asdict(self.contract),
            backend=type(self.backend).__name__,
            torch_version=str(torch.__version__),
            optimizer_contract=copy.deepcopy(self.backend.optimizer_contract),
            semantics=self.backend.resume_semantics(),
        )

    def state_dict(self):
        if self._in_flight or self._failed:
            raise ValueError("Cannot checkpoint an in-flight/failed RL transaction")
        return dict(
            schema=1,
            config=self.resume_config(),
            requests=self.requests,
            updates=self.updates,
            policy_version=self.policy_version,
            data_cursor=self.data_cursor,
            rollout=_cpu_tree(self.rollout.state_dict()),
            reward=_cpu_tree(self.reward.state_dict()),
        )

    def validate_state_dict(self, state):
        keys = {
            "schema",
            "config",
            "requests",
            "updates",
            "policy_version",
            "data_cursor",
            "rollout",
            "reward",
        }
        if set(state) != keys or state["schema"] != 1 or state["config"] != self.resume_config():
            raise ValueError("RL resume algorithm/config/fingerprints/backend differ")
        if any(
            type(state[k]) is not int or state[k] < 0
            for k in ("requests", "updates", "policy_version", "data_cursor")
        ):
            raise ValueError("Invalid RL resume counters")
        if (
            not state["policy_version"]
            == state["updates"]
            <= state["requests"]
            <= self.cfg.max_requests
        ):
            raise ValueError("RL policy version/update/request counters are inconsistent")
        if state["updates"] > self.cfg.max_updates or state["data_cursor"] < state["requests"]:
            raise ValueError("RL checkpoint exceeds configured bounds or data cursor")
        if (
            not state["requests"] * self.cfg.group_size
            <= state["data_cursor"]
            <= state["requests"] * self.cfg.max_samples
        ):
            raise ValueError("RL request/data cursor bounds disagree")
        if state["data_cursor"] % self.cfg.group_size:
            raise ValueError("RL cursor does not contain complete reward groups")
        self.rollout.validate_state_dict(state["rollout"])
        self.reward.validate_state_dict(state["reward"])

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.rollout.load_state_dict(_cpu_tree(state["rollout"]))
        self.reward.load_state_dict(_cpu_tree(state["reward"]))
        for key in ("requests", "updates", "policy_version", "data_cursor"):
            setattr(self, key, state[key])
        self._in_flight = self._failed = False

    def _validate_batch(self, request, batch):
        if batch.identity != request.identity or batch.request_index != request.request_index:
            raise ValueError("Stale/mismatched rollout policy, fingerprints or request cursor")
        shape = batch.input_ids.shape
        if (
            len(shape) != 2
            or not 0 < shape[0] <= request.max_samples
            or not 0 < shape[1] <= request.max_sequence_length
        ):
            raise ValueError("Rollout exceeds bounded sample/sequence dimensions")
        if shape[0] % self.cfg.micro_batch_size or shape[0] % self.cfg.group_size:
            raise ValueError("Rollout sample count must fit complete groups and microbatches")
        for name in ("sampled_token_ids", "token_mask", "old_logprobs", "reference_logprobs"):
            tensor = getattr(batch, name)
            if tensor.shape != shape or tensor.requires_grad:
                raise ValueError(f"Invalid aligned/detached rollout field {name}")
        if (
            batch.input_ids.dtype != torch.long
            or batch.sampled_token_ids.dtype != torch.long
            or batch.token_mask.dtype != torch.bool
        ):
            raise ValueError("Rollout token/mask dtype mismatch")
        if (
            len(batch.sample_ids) != shape[0]
            or len(set(batch.sample_ids)) != shape[0]
            or len(batch.group_ids) != shape[0]
        ):
            raise ValueError("Missing/duplicate completion identity")
        if any(
            not isinstance(value, str) or not value
            for value in (*batch.sample_ids, *batch.group_ids)
        ):
            raise ValueError("Completion/group IDs must be nonempty strings")
        microbatches = shape[0] // self.cfg.micro_batch_size
        if (
            isinstance(self.backend, PithTrainPolicyAdapter)
            and microbatches < 2 * self.backend.pp_size
        ):
            raise ValueError("DualPipeV requires at least 2*PP microbatches per rank")
        for payload in (batch.model_context, batch.media_inputs):
            if payload is not None:
                if len(payload) != microbatches:
                    raise ValueError("Explicit context/media entry required for every microbatch")
                if any(
                    row is not None
                    and (
                        not isinstance(row, dict)
                        or any(
                            not isinstance(v, torch.Tensor) or v.requires_grad for v in row.values()
                        )
                    )
                    for row in payload
                ):
                    raise ValueError("Context/media must contain detached tensors")
        if self.backend.pp_rank != 0 and batch.media_inputs:
            raise ValueError("Only PP0 may carry encoder media payloads")
        for values in (batch.old_logprobs, batch.reference_logprobs):
            if (
                not values.is_floating_point()
                or not torch.isfinite(values[batch.token_mask]).all()
                or (values[batch.token_mask] > 1e-6).any()
            ):
                raise ValueError("Invalid selected-token rollout/reference logprob")

    def train_step(self):
        if self._in_flight or self._failed:
            raise ValueError(
                "Create a fresh trainer and restore the last checkpoint after a failed/in-flight transaction"
            )
        if self.requests >= self.cfg.max_requests or self.updates >= self.cfg.max_updates:
            raise ValueError("Bounded RL request/update limit reached")
        self._in_flight = True
        try:
            error = None
            try:
                snapshot = self.backend.snapshot(self.policy_version, self.contract)
                snapshot.identity.validate()
                if (
                    seal(snapshot.semantics["identity"]) != snapshot.semantics
                    or snapshot.identity.semantics_sha256 != snapshot.semantics["sha256"]
                ):
                    raise ValueError("Published policy semantics digest does not match contract")
                if (
                    snapshot.identity.version != self.policy_version
                    or snapshot.identity.contract != self.contract
                ):
                    raise ValueError("Published policy version/config does not match training")
                if self.rollout.synchronize(snapshot) != snapshot.identity:
                    raise ValueError("Rollout backend did not acknowledge exact policy weights")
                request = RolloutRequest(
                    snapshot.identity,
                    self.requests,
                    self.data_cursor,
                    self.cfg.seed + self.requests,
                    self.cfg.group_size,
                    self.cfg.max_samples,
                    self.cfg.max_sequence_length,
                    self.backend.dp_rank,
                    self.backend.pp_rank,
                    self.backend.pp_rank == 0,
                )
                batch = self.rollout.generate(request)
                self._validate_batch(request, batch)
                rewards = self.reward.score(request, batch)
                advantages = group_relative_advantages(
                    rewards, batch.group_ids, self.cfg.group_size, self.cfg.advantage_epsilon
                )
            except Exception as caught:
                error = caught
            _collective_error(error, self.backend.validation_group)
            if dist.is_initialized():
                identities = [None] * dist.get_world_size(self.backend.validation_group)
                identity = dict(policy=asdict(batch.identity))
                if isinstance(self.backend, PithTrainPolicyAdapter):
                    identity["batch_shape"] = list(batch.input_ids.shape)
                    identity["micro_batch_size"] = self.cfg.micro_batch_size
                dist.all_gather_object(identities, identity, group=self.backend.validation_group)
                if any(value != identities[0] for value in identities):
                    raise ValueError(
                        "Training ranks do not share one policy version/weights/contract and native padded batch layout"
                    )
            error = None
            try:
                self.backend.validate_batch_layout(batch, self.cfg, rewards)
            except Exception as caught:
                error = caught
            _collective_error(error, self.backend.validation_group)
            local_count = batch.token_mask.sum(dtype=torch.int64).to(self.backend.device)
            count = global_target_counts(local_count, token_group=self.backend.token_group)
            if count.item() == 0:
                self.requests += 1
                self.data_cursor += batch.input_ids.shape[0]
                return dict(
                    status="SKIPPED_GLOBAL_ZERO",
                    requests=self.requests,
                    updates=self.updates,
                    policy_version=self.policy_version,
                    global_tokens=0,
                )
            microbatches = []
            for begin in range(0, batch.input_ids.shape[0], self.cfg.micro_batch_size):
                end = begin + self.cfg.micro_batch_size

                def move(tensor):
                    return tensor[begin:end].to(self.backend.device)

                index = begin // self.cfg.micro_batch_size
                context = None if batch.model_context is None else batch.model_context[index]
                media = None if batch.media_inputs is None else batch.media_inputs[index]
                context = {k: v.to(self.backend.device) for k, v in (context or {}).items()}
                microbatches.append(
                    Microbatch(
                        model_inputs=(move(batch.input_ids),),
                        cu_seqlens=None,
                        objective_inputs=(
                            move(batch.sampled_token_ids),
                            move(batch.old_logprobs),
                            move(batch.reference_logprobs),
                            move(batch.token_mask),
                            move(advantages),
                        ),
                        model_context=context or None,
                        media_inputs=None
                        if not media
                        else {k: v.to(self.backend.device) for k, v in media.items()},
                        sample_ids=batch.sample_ids[begin:end],
                    )
                )

            def objective(outputs, inputs):
                loss = grpo_loss_sum(outputs[0], *inputs, self.cfg)
                return loss, loss.detach()

            metrics = self.backend.backward(microbatches, objective)
            norm = self.backend.update(count, self.cfg.max_grad_norm)
            self.requests += 1
            self.updates += 1
            self.policy_version += 1
            self.data_cursor += batch.input_ids.shape[0]
            return dict(
                status="UPDATED",
                requests=self.requests,
                updates=self.updates,
                policy_version=self.policy_version,
                global_tokens=int(count.item()),
                local_loss_sum=sum(float(x) for x in metrics),
                gradient_norm=norm,
                weight_source="caller_policy",
                external_backend="CALLER_PROVIDED",
            )
        except BaseException:
            self._failed = True
            raise
        finally:
            self._in_flight = False


def launch(cfg, *, contract, backend, rollout, reward, trainer=None):
    """Run at most max_requests transactions and max_updates actual optimizer steps."""
    trainer = trainer or OmniRLTrainer(
        cfg, contract=contract, backend=backend, rollout=rollout, reward=reward
    )
    if (
        trainer.cfg.to_json_dict() != cfg.to_json_dict()
        or trainer.contract != contract
        or trainer.backend is not backend
        or trainer.rollout is not rollout
        or trainer.reward is not reward
    ):
        raise ValueError("Existing RL trainer does not match launch configuration/providers")
    reports = []
    while trainer.requests < cfg.max_requests and trainer.updates < cfg.max_updates:
        reports.append(trainer.train_step())
    return trainer, reports
