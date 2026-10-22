"""Tiny native DualPipeV SFT/GRPO numerical and fresh-process resume acceptance.

prepare/plan are CPU-only. produce/restore require explicitly scheduled CUDA;
this module allocates no cloud resources, downloads no weights and runs no child
jobs. Synthetic ID prompts exercise real autoregressive policy sampling, not
conversational quality. An external rollout service remains NOT_INTEGRATED.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "omni_native_posttraining_v1"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
OFFICIAL_SHA = "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"
TRANSFORMERS = "5.17.0"
PROFILES = ((1, 1, 1), (4, 1, 2), (4, 2, 2))
SEED, SEQUENCE, PROMPT, BATCH, UPDATES, REQUESTS = 1234, 32, 28, 8, 4, 5
GRADIENT_NSE, LOSS_RTOL, LOSS_ATOL = 1e-2, 1e-3, 1e-3
COMMAND_TIMEOUT_SECONDS, KILL_GRACE_SECONDS = 180, 10
CONFIG_PATH = REPO / "tests/configs/qwen3_omni_text/tiny.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def source_identity(expected):
    require(re.fullmatch("[0-9a-f]{40}", expected) is not None, "Full expected commit required")
    require(git("rev-parse", "HEAD") == expected, "Source commit differs")
    require(not git("status", "--porcelain"), "Exact clean committed source required")
    return dict(commit=expected, tree=git("rev-parse", "HEAD^{tree}"), driver=sha(__file__))


def profile(world, pp, ep):
    require((world, pp, ep) in PROFILES, "Unreviewed posttraining topology")
    dp = world // pp
    local = BATCH // dp
    require(local >= 2 * pp and local % 2 == 0, "Incomplete pipeline/reward microbatches")
    return dict(world=world, pp=pp, ep=ep, cp=1, hsdp=1, dp=dp, local_batch=local)


def token_mask(request, dp_rank, topology):
    require(type(request) is int and 0 <= request < REQUESTS, "Request outside bounded schedule")
    require(0 <= dp_rank < topology["dp"], "Invalid data rank")
    mask = torch.zeros(topology["local_batch"], SEQUENCE, dtype=torch.bool)
    if request == 2 or (request == 1 and topology["dp"] > 1 and dp_rank == 0):
        return mask
    for row in range(mask.shape[0]):
        length = 2 + (row + dp_rank + request) % 3
        mask[row, PROMPT - 1 : PROMPT - 1 + length] = True
    return mask


def reward_tokens(tokens, mask):
    """Decidable reward: fraction of valid generated token IDs that are even."""
    require(tokens.shape == mask.shape and mask.dtype == torch.bool, "Reward layout differs")
    counts = mask.sum(-1)
    return ((tokens.remainder(2) == 0) & mask).float().sum(-1) / counts.clamp(min=1)


@torch.no_grad()
def sample_autoregressive(policy, prompts, generator, completion_length):
    """Sample from each newly extended prefix, retaining the full policy support."""
    ids = prompts
    for _ in range(completion_length):
        probabilities = policy(ids)[:, -1].float().softmax(-1).cpu()
        sampled = torch.multinomial(probabilities, 1, generator=generator)
        ids = torch.cat((ids, sampled.to(ids.device)), dim=1)
    return ids


def independent_grpo(logits, tokens, old, reference, mask, rewards, *, clip=0.2, beta=0.02):
    """Independent scalar objective; no task loss/advantage helper is called."""
    require(rewards.ndim == 1 and rewards.numel() % 2 == 0, "Complete pairs required")
    grouped = rewards.reshape(-1, 2)
    advantages = (
        (grouped - grouped.mean(-1, keepdim=True))
        / (grouped.std(-1, correction=0, keepdim=True) + 1e-6)
    ).reshape(-1)
    return independent_token_objective(
        logits, tokens, old, reference, mask, advantages, clip=clip, beta=beta
    )


def independent_token_objective(
    logits, tokens, old, reference, mask, advantages, *, clip=0.2, beta=0.02
):
    if not mask.any():
        return logits.sum() * 0
    current = logits.float()[mask].log_softmax(-1).gather(-1, tokens[mask, None]).squeeze(-1)
    ratio = (current - old[mask]).exp()
    advantage = advantages[:, None].expand_as(mask)[mask]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    delta = reference[mask] - current
    return (-surrogate + beta * (delta.exp() - delta - 1)).sum()


def gradient_evidence(actual, expected, shapes):
    """Complete report before aggregate rejection; zero/tiny tensors are compared."""
    from tests.acceptance.omni_topology_gradients import calculate_difference

    issues = []
    for label, values in (("native", actual), ("reference", expected)):
        if set(values) != set(shapes):
            issues.append(
                dict(
                    side=label,
                    missing=sorted(set(shapes) - set(values)),
                    extra=sorted(set(values) - set(shapes)),
                )
            )
    rows = {}
    for name, shape in shapes.items():
        left, right = actual.get(name), expected.get(name)
        if (
            left is None
            or right is None
            or tuple(left.shape) != tuple(shape)
            or tuple(right.shape) != tuple(shape)
        ):
            rows[name] = dict(status="FAIL", reason="missing gradient or wrong shape")
            continue
        if (
            left.dtype != right.dtype
            or not torch.isfinite(left).all()
            or not torch.isfinite(right).all()
        ):
            rows[name] = dict(status="FAIL", reason="dtype or nonfinite gradient")
            continue
        error = calculate_difference(left, right)
        rows[name] = dict(
            status="PASS" if error < GRADIENT_NSE else "FAIL",
            nse=error,
            native_norm=float(left.double().norm()),
            reference_norm=float(right.double().norm()),
            native_maxabs=float(left.abs().max()),
            reference_maxabs=float(right.abs().max()),
            diff_maxabs=float((left.double() - right.double()).abs().max()),
            shape=list(shape),
        )
    return dict(
        status="FAIL"
        if issues or any(row["status"] != "PASS" for row in rows.values())
        else "PASS",
        threshold=GRADIENT_NSE,
        expected_count=len(shapes),
        checked_count=sum("nse" in row for row in rows.values()),
        issues=issues,
        parameters=rows,
    )


def merge_canonical(parts, shapes):
    result = {}
    for part in parts:
        for name, tensor in part.items():
            if name in result:
                require(
                    torch.equal(result[name], tensor), f"Replica canonical tensor differs: {name}"
                )
            else:
                result[name] = tensor
    require(set(result) == set(shapes), "Missing/unexpected canonical parameters")
    require(
        all(tuple(value.shape) == tuple(shapes[key]) for key, value in result.items()),
        "Canonical shape differs",
    )
    return result


def root_phase(function):
    """Propagate rank-zero reference/fixture failures before peers enter collectives."""
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = (None, function())
        except Exception as error:
            payload[0] = (f"{type(error).__name__}: {error}", None)
    dist.broadcast_object_list(payload, src=0)
    error, value = payload[0]
    require(error is None, f"Reference rank failed: {error}")
    return value


def gather_native(model, config, *, gradients=False):
    from torch.distributed.tensor import DTensor

    from pithtrain.modules.checkpoint import to_canonical_model
    from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import parameter_shapes

    parameters = dict(model.named_parameters())
    errors = [name for name, value in parameters.items() if gradients and value.grad is None]
    peers = [None] * dist.get_world_size()
    dist.all_gather_object(peers, errors)
    require(not any(peers), f"Missing native gradient before tensor collectives: {peers}")
    localized = {}
    for name, parameter in parameters.items():
        value = parameter.grad if gradients else parameter.detach()
        if isinstance(value, DTensor):
            value = value.full_tensor()
        localized[name] = value.detach().cpu().clone()
    local = to_canonical_model(localized, model)
    parts = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local, parts, dst=0)
    result = None

    def merge():
        nonlocal result
        result = merge_canonical(parts, parameter_shapes(config))
        return True

    root_phase(merge)
    return result


def plan_commands(args):
    base = [
        "--task",
        args.task,
        "--world",
        str(args.world),
        "--pp",
        str(args.pp),
        "--ep",
        str(args.ep),
        "--expected-commit",
        args.expected_commit,
        "--hardware",
        args.hardware,
        "--output",
        str(args.output),
    ]
    return {
        mode: [
            "timeout",
            "--kill-after=10s",
            "180s",
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.world}",
            "-m",
            "tests.acceptance.omni_posttraining",
            mode,
            *base,
        ]
        for mode in ("produce", "restore")
    }


def binding(args):
    return dict(
        schema=SCHEMA,
        source=source_identity(args.expected_commit),
        task=args.task,
        topology=profile(args.world, args.pp, args.ep),
        hardware=args.hardware,
        transformers=TRANSFORMERS,
        official_revision=REVISION,
        official_config_sha256=OFFICIAL_SHA,
        tiny_config_sha256=sha(CONFIG_PATH),
        seed=SEED,
        sequence=SEQUENCE,
        prompt=PROMPT,
        global_batch=BATCH,
        micro_batch=1,
        updates=UPDATES,
        requests=REQUESTS,
        checkpoint_request=3,
        checkpoint_updates=2,
        gradient_nse=GRADIENT_NSE,
        loss_rtol=LOSS_RTOL,
        loss_atol=LOSS_ATOL,
        reference_attention="native-matched-v2",
        parameter_storage="float32",
        compute="bfloat16",
        fixture="synthetic ID prompts within official vocabulary; no conversational dataset claim",
        external_service="NOT_INTEGRATED",
    )


def prepare(args):
    require(not torch.cuda.is_initialized(), "CPU preparation must not initialize CUDA")
    require(importlib.metadata.version("transformers") == TRANSFORMERS, "Transformers pin differs")
    require(
        args.official_config is not None and sha(args.official_config) == OFFICIAL_SHA,
        "Reviewed official config required",
    )
    from tests.acceptance.omni_native_resume import model_config

    metadata = binding(args)
    args.output.mkdir(parents=True, exist_ok=False)
    model_config().save_pretrained(args.output / "model")
    metadata["prepared_config_sha256"] = sha(args.output / "model/config.json")
    metadata["commands"] = plan_commands(args)
    metadata["status"] = "PREPARED_GPU_NOT_RUN"
    write_json(args.output / "prepare.json", metadata)
    require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
    return metadata


def setup(args):
    require(
        os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1",
        "Explicit GPU scheduling opt-in required",
    )
    require(int(os.environ.get("WORLD_SIZE", "0")) == args.world, "Exact torchrun world required")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "Offline HF mode required")
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
        "Process-start CUBLAS lock missing",
    )
    prepared = json.loads((args.output / "prepare.json").read_text())
    require(
        all(prepared.get(k) == v for k, v in binding(args).items()),
        "Prepared source/runtime contract differs",
    )
    require(
        prepared["prepared_config_sha256"] == sha(args.output / "model/config.json"),
        "Prepared model changed",
    )
    require(importlib.metadata.version("transformers") == TRANSFORMERS, "Transformers pin differs")
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import (
        make_adamw_optimizer,
        make_constant_scheduler,
        setup_training,
    )
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg

    cfg = PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size, cfg.distributed.expert_parallel_size = args.pp, args.ep
    cfg.distributed.context_parallel_size = cfg.distributed.hsdp_replica = 1
    cfg.distributed.timeout = timedelta(seconds=90)
    t = cfg.training
    t.model = args.output / "model"
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.sequence_length, t.global_batch_size, t.micro_batch_size = SEQUENCE, BATCH, 1
    t.lr, t.max_steps, t.seed = 1e-5, UPDATES, SEED
    t.moe_load_balance_coef, t.benchmark, t.fp8 = 0.0, False, False
    t.strict_resume = t.deterministic = True
    t.save_interval = t.save_location = None
    cfg.logging.wandb = None
    setup_logging(cfg)
    setup_distributed(cfg)
    require(torch.cuda.get_device_name() == args.hardware, "Actual GPU differs from hardware pin")
    require(torch.cuda.get_device_capability()[0] in (9, 10), "Hopper/Blackwell required")
    setup_training(cfg)
    from pithtrain.contexts import training
    from pithtrain.pipeline import DualPipeV

    require(isinstance(training.model, DualPipeV), "Actual native DualPipeV required")
    require(len(training.optimizers) == 1, "Reviewed all-parameter AdamW optimizer required")
    require(
        all(p.dtype == torch.float32 for p in training.model.parameters()),
        "Native FP32 masters required",
    )
    return cfg, prepared


class HFPolicy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerTextModel,
        )

        local = copy.deepcopy(config)
        local._attn_implementation = local._experts_implementation = "eager"
        self.model = Qwen3OmniMoeThinkerTextModel(local)
        self.lm_head = torch.nn.Linear(local.hidden_size, local.vocab_size, bias=False)

    def forward(self, ids):
        from tests.acceptance.omni_reference_attention import reference_attention_kwargs

        hidden = self.model(
            input_ids=ids, use_cache=False, **reference_attention_kwargs(self)
        ).last_hidden_state
        return self.lm_head(hidden)


class Reference:
    def __init__(self, config, initial):
        from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import canonical_to_hf
        from tests.acceptance.omni_reference_attention import MATCHED, configure_reference_attention
        from tests.acceptance.omni_reference_precision import FP32GradientReference

        self.config = config
        # Preserve FP32 HF RoPE buffers; only parameter dtype is BF16 at birth.
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cpu"), torch.random.fork_rng(devices=[]):
                self.policy = HFPolicy(config)
        finally:
            torch.set_default_dtype(previous)
        self.policy = self.policy.cuda().eval()
        self.attention = configure_reference_attention(self.policy, config, MATCHED)
        self.install(initial)
        self.reference = copy.deepcopy(self.policy).eval()
        self.master = FP32GradientReference(self.policy)
        self.optimizer = torch.optim.AdamW(
            self.master.master_parameters.values(), lr=1e-5, weight_decay=0.1
        )
        self._convert = canonical_to_hf

    def install(self, canonical):
        from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import canonical_to_hf

        self.policy.load_state_dict(canonical_to_hf(canonical, self.config, prefix=""), strict=True)

    @torch.no_grad()
    def rollout(self, request, topology):
        count = topology["local_batch"]
        generator = torch.Generator().manual_seed(request.seed + request.dp_rank * 100003)
        prompts = torch.randint(
            0, self.config.vocab_size, (count // 2, PROMPT), generator=generator
        ).repeat_interleave(2, 0)
        ids = sample_autoregressive(self.policy, prompts.cuda(), generator, SEQUENCE - PROMPT)
        targets = torch.zeros_like(ids)
        targets[:, :-1] = ids[:, 1:]
        old = self.policy(ids).float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ref = (
            self.reference(ids)
            .float()
            .log_softmax(-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )
        return dict(
            input_ids=ids.cpu(),
            tokens=targets.cpu(),
            mask=token_mask(request.request_index, request.dp_rank, topology),
            old=old.cpu(),
            reference=ref.cpu(),
        )

    def gradient(self, canonical, records, task):
        from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import hf_to_canonical

        self.install(canonical)
        converted = self._convert(canonical, self.config, prefix="")
        with torch.no_grad():
            for name, parameter in self.master.master_parameters.items():
                parameter.copy_(converted[name])
        self.master.zero_grad()
        count = sum(int(row["mask"].sum()) for row in records)
        require(count > 0, "Reference gradient needs valid targets")
        total = 0.0
        for row in records:
            rewards = reward_tokens(row["tokens"], row["mask"]).cuda()
            # Complete groups are pairs; native executes one microbatch at a time.
            grouped = rewards.reshape(-1, 2)
            advantages = (
                (grouped - grouped.mean(-1, keepdim=True))
                / (grouped.std(-1, correction=0, keepdim=True) + 1e-6)
            ).reshape(-1)
            for index in range(row["input_ids"].shape[0]):
                ids, labels, mask = (
                    row[key][index : index + 1].cuda() for key in ("input_ids", "tokens", "mask")
                )
                logits = self.master(ids)
                if task == "sft":
                    loss = F.cross_entropy(
                        logits.float().flatten(0, 1),
                        labels.masked_fill(~mask, -100).flatten(),
                        reduction="sum",
                    )
                else:
                    loss = independent_token_objective(
                        logits,
                        labels,
                        row["old"][index : index + 1].cuda(),
                        row["reference"][index : index + 1].cuda(),
                        mask,
                        advantages[index : index + 1],
                    )
                loss.backward()
                total += float(loss.detach())
        self.master.normalize_gradients_(count)
        values = self.master.master_parameters
        require(all(p.grad is not None for p in values.values()), "Missing HF parameter gradient")
        preclip, _ = hf_to_canonical(
            {name: p.grad.detach().cpu().clone() for name, p in values.items()},
            self.config,
            prefix="",
        )
        norm = torch.sqrt(sum(p.grad.square().sum() for p in values.values())).clamp(min=1e-6)
        scale = (1.0 / norm).clamp(max=1.0)
        for parameter in values.values():
            parameter.grad.mul_(scale)
        gradients, _ = hf_to_canonical(
            {name: p.grad.detach().cpu() for name, p in values.items()}, self.config, prefix=""
        )
        return dict(
            gradients=gradients,
            preclip_gradients=preclip,
            loss=total / count,
            count=count,
            preclip_norm=float(norm),
        )


class DecidableReward:
    def __init__(self):
        self.calls = 0

    def score(self, request, batch):
        self.calls += 1
        return reward_tokens(batch.sampled_token_ids, batch.token_mask)

    def state_dict(self):
        return dict(calls=self.calls, contract="even-token-fraction-v1")

    def validate_state_dict(self, state):
        require(
            set(state) == {"calls", "contract"}
            and state["contract"] == "even-token-fraction-v1"
            and type(state["calls"]) is int
            and 0 <= state["calls"] <= REQUESTS,
            "Reward state differs",
        )

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.calls = state["calls"]


class Rollout:
    def __init__(self, reference, topology, observer):
        self.reference, self.topology, self.observer = reference, topology, observer
        self.calls = 0

    def synchronize(self, snapshot):
        self.identity = snapshot.identity
        root_phase(lambda: self.reference.install(snapshot.payload))
        return snapshot.identity

    def generate(self, request):
        from dataclasses import replace

        from pithtrain.tasks.omni_rl import RolloutBatch

        records = root_phase(
            lambda: [
                self.reference.rollout(
                    replace(request, dp_rank=rank, rollout_leader=True), self.topology
                )
                for rank in range(self.topology["dp"])
            ]
        )
        self.observer.prepare(request.request_index, records)
        row = records[request.dp_rank]
        local = self.topology["local_batch"]
        self.calls += 1
        return RolloutBatch(
            request.identity,
            request.request_index,
            row["input_ids"],
            row["tokens"],
            row["mask"],
            row["old"],
            row["reference"],
            tuple(f"dp{request.dp_rank}-g{i // 2}" for i in range(local)),
            tuple(
                f"request{request.request_index}-dp{request.dp_rank}-sample{i}"
                for i in range(local)
            ),
        )

    def state_dict(self):
        return dict(calls=self.calls, sampler="cpu-multinomial-seed-request-dp-v1")

    def validate_state_dict(self, state):
        require(
            set(state) == {"calls", "sampler"}
            and state["sampler"] == "cpu-multinomial-seed-request-dp-v1"
            and type(state["calls"]) is int
            and 0 <= state["calls"] <= REQUESTS,
            "Rollout state differs",
        )

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.calls = state["calls"]


class Observer:
    def __init__(self, args, config, reference):
        self.args, self.config, self.reference = args, config, reference
        self.calls = 0
        self.records = None

    def prepare(self, request, records):
        from pithtrain.contexts import training

        self.request, self.records = request, records
        canonical = gather_native(training.model, self.config)
        self.before = canonical
        if any(row["mask"].any() for row in records):
            self.expected = None

            def compute():
                self.expected = self.reference.gradient(canonical, records, self.args.task)
                return {
                    key: value
                    for key, value in self.expected.items()
                    if key not in {"gradients", "preclip_gradients"}
                }

            self.summary = root_phase(compute)
        else:
            self.summary = dict(count=0, loss=None)

    def before_optimizer_step(self, optimizer, positional, keyword):
        from pithtrain.contexts import training
        from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import parameter_shapes

        actual = gather_native(training.model, self.config, gradients=True)

        def compare():
            report = gradient_evidence(
                actual, self.expected["gradients"], parameter_shapes(self.config)
            )
            report.update(
                request=self.request,
                phase="normalized_clipped_before_optimizer",
                reference=self.summary,
                source=self.args.expected_commit,
            )
            write_json(
                self.args.output / f"{self.args.mode}-gradient-request-{self.request}.json", report
            )
            require(
                report["status"] == "PASS", "Native/HF gradient gate failed; complete report saved"
            )
            return True

        root_phase(compare)
        self.calls += 1

    def before_clip(self):
        from pithtrain.contexts import training
        from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import parameter_shapes

        actual = gather_native(training.model, self.config, gradients=True)

        def compare():
            report = gradient_evidence(
                actual, self.expected["preclip_gradients"], parameter_shapes(self.config)
            )
            report.update(
                request=self.request,
                phase="normalized_before_clipping",
                reference=self.summary,
                source=self.args.expected_commit,
            )
            write_json(
                self.args.output / f"{self.args.mode}-preclip-gradient-request-{self.request}.json",
                report,
            )
            require(
                report["status"] == "PASS",
                "Native/HF pre-clip gradient gate failed; complete report saved",
            )

        root_phase(compare)

    def after_update(self):
        from pithtrain.contexts import training
        from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import (
            hf_to_canonical,
            parameter_shapes,
        )

        actual = gather_native(training.model, self.config)

        def compare():
            self.reference.optimizer.step()
            expected, _ = hf_to_canonical(
                {
                    name: p.detach().cpu()
                    for name, p in self.reference.master.master_parameters.items()
                },
                self.config,
                prefix="",
            )
            left = {name: value - self.before[name] for name, value in actual.items()}
            right = {name: value - self.before[name] for name, value in expected.items()}
            report = gradient_evidence(left, right, parameter_shapes(self.config))
            report.update(
                request=self.request,
                phase="independent_adamw_update_delta",
                source=self.args.expected_commit,
                any_native_parameter_changed=any(
                    torch.count_nonzero(value) for value in left.values()
                ),
            )
            write_json(
                self.args.output / f"{self.args.mode}-update-request-{self.request}.json", report
            )
            require(report["any_native_parameter_changed"], "No native parameter changed")
            require(
                report["status"] == "PASS", "Native/HF update gate failed; complete report saved"
            )

        root_phase(compare)


class SFTData:
    def __init__(self):
        self.scanned_batches = self.completed_updates = 0

    def state_dict(self):
        return dict(scanned_batches=self.scanned_batches, completed_updates=self.completed_updates)

    def validate_state_dict(self, state):
        require(set(state) == {"scanned_batches", "completed_updates"}, "SFT data keys differ")
        require(
            all(type(x) is int for x in state.values())
            and 0 <= state["completed_updates"] <= state["scanned_batches"] <= REQUESTS,
            "SFT data cursor differs",
        )

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.scanned_batches, self.completed_updates = (
            state["scanned_batches"],
            state["completed_updates"],
        )


def sft_records(request, topology, vocab):
    records = []
    for rank in range(topology["dp"]):
        generator = torch.Generator().manual_seed(SEED + request * 100 + rank)
        ids = torch.randint(vocab, (topology["local_batch"], SEQUENCE + 1), generator=generator)
        records.append(
            dict(input_ids=ids[:, :-1], tokens=ids[:, 1:], mask=token_mask(request, rank, topology))
        )
    return records


def snapshot(task_state):
    from pithtrain.contexts import distributed, training
    from pithtrain.modules.distributed_checkpoint import _pack, _runtime

    values, _ = _runtime(
        training.model, training.optimizers, training.schedulers, task_state, distributed.device
    )
    return _pack(values)


def run_gpu(args):
    from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
    from pithtrain.modules.microbatch import Microbatch
    from pithtrain.modules.task_checkpoint import TaskCheckpointState
    from pithtrain.tasks import sft_lm
    from pithtrain.tasks.omni_rl import (
        OmniRLCfg,
        OmniRLTrainer,
        PithTrainPolicyAdapter,
        PolicyIdentity,
        PolicySnapshot,
        RolloutContract,
        _weights_digest,
        fingerprint,
    )
    from tests.acceptance.omni_native_resume import equal, model_config

    cfg, prepared = setup(args)
    from pithtrain.contexts import distributed, training

    rank, device, topology = distributed.rank, distributed.device, prepared["topology"]
    hardware = dict(
        name=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        cuda=torch.version.cuda,
        torch=str(torch.__version__),
        transformers=importlib.metadata.version("transformers"),
        memory_free_total=list(torch.cuda.mem_get_info()),
    )
    if rank == 0:
        for key, command in (
            (
                "driver_gpu_uuid",
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            ),
            ("nvlink_topology", ["nvidia-smi", "topo", "-m"]),
        ):
            proc = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
            hardware[key] = dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    write_json(
        args.output / f"{args.mode}-runtime-rank-{rank:05d}.json",
        dict(source=prepared["source"], topology=topology, hardware=hardware),
    )
    destination = args.output / f"{args.mode}-rank-{rank:05d}.json"
    require(not destination.exists(), "Never overwrite prior acceptance evidence")
    config = model_config()
    initial = gather_native(training.model, config)
    reference = None

    def construct_reference():
        nonlocal reference
        reference = Reference(config, initial)
        return dict(digest=_weights_digest(initial), attention=reference.attention)

    identity = root_phase(construct_reference)
    observer = Observer(args, config, reference)
    from pithtrain.modules import optimizer as native_optimizer

    original_clip = native_optimizer.clip_grad_norm

    def observed_clip(*positional, **keyword):
        observer.before_clip()
        return original_clip(*positional, **keyword)

    native_optimizer.clip_grad_norm = observed_clip
    hook = training.optimizers[0].register_step_pre_hook(observer.before_optimizer_step)
    token_group = distributed.attn_mesh["dp", "cp"]._flatten().get_group()
    resume_config = dict(
        schema=SCHEMA,
        source=prepared["source"],
        task=args.task,
        topology=topology,
        fixture=prepared["fixture"],
        seed=SEED,
        lr=cfg.training.lr,
        reference_weights=identity["digest"],
        max_grad_norm=1.0,
    )
    if args.task == "grpo":
        contract = RolloutContract(
            fingerprint("synthetic-id-prompt-v1"),
            fingerprint(
                dict(
                    vocabulary=config.vocab_size,
                    official_revision=REVISION,
                    source=args.expected_commit,
                    tiny_config=prepared["tiny_config_sha256"],
                    tokenizer_used=False,
                )
            ),
            fingerprint(dict(text_positions="0..S-1", rope=config.rope_parameters)),
            identity["digest"],
            fingerprint("even-token-fraction-v1"),
        )

        def publish(version, active_contract, semantics):
            canonical = gather_native(training.model, config)
            digest = root_phase(lambda: _weights_digest(canonical))
            return PolicySnapshot(
                PolicyIdentity(version, digest, active_contract, semantics["sha256"]),
                canonical,
                semantics,
            )

        backend = PithTrainPolicyAdapter.from_training_context(
            export_policy=publish, training_cfg=cfg.training
        )
        rlcfg = OmniRLCfg()
        rlcfg.max_updates, rlcfg.max_requests, rlcfg.max_samples = (
            UPDATES,
            REQUESTS,
            topology["local_batch"],
        )
        rlcfg.max_sequence_length, rlcfg.seed = SEQUENCE, SEED
        trainer = OmniRLTrainer(
            rlcfg,
            contract=contract,
            backend=backend,
            rollout=Rollout(reference, topology, observer),
            reward=DecidableReward(),
        )
        task_state = backend._checkpoint_state(trainer)
        resume_config = trainer.resume_config()
    else:
        sftcfg = sft_lm.SFTLMCfg()
        sftcfg.max_updates, sftcfg.max_batches, sftcfg.max_grad_norm = UPDATES, REQUESTS, 1.0
        state, data = sft_lm.SFTState(), SFTData()
        task_state = TaskCheckpointState(
            sft_lm.SFTCheckpointState(state, data),
            task_kind="sft",
            task_schema=1,
            contract=resume_config,
            transaction_key="scanned_batches",
            update_key="completed_updates",
            data_key="data_cursor",
        )
    checkpoint = args.output / "checkpoints"
    expected_path = args.output / f"continuous-rank-{rank:05d}.pt"
    expected = None
    start = 0
    if args.mode == "restore":
        expected = torch.load(expected_path, weights_only=True)
        require(expected["producer_pid"] != os.getpid(), "Fresh torchrun process required")
        require(expected["source"] == args.expected_commit, "Resume evidence source differs")
        require(
            all(not optimizer.state for optimizer in training.optimizers),
            "Fresh optimizer required",
        )
        load_checkpoint(
            checkpoint, 3, data_state=task_state, strict=True, resume_config=resume_config
        )
        equal(snapshot(task_state), expected["boundary"], "native boundary")
        root_phase(lambda: reference.optimizer.load_state_dict(expected["reference_optimizer"]))
        start = 3
    results, next_records = [], []
    boundary = None
    reference_boundary = None
    try:
        for request in range(start, REQUESTS):
            before_calls = observer.calls
            before_zero = snapshot(task_state) if request == 2 else None
            if args.task == "grpo":
                result = trainer.train_step()
                loss = result.get("local_loss_sum", 0.0)
                scalar = torch.tensor(
                    loss if distributed.pp_rank == 0 else 0.0, device=device, dtype=torch.float64
                )
                dist.all_reduce(scalar)
                native_loss = (
                    None if not result["global_tokens"] else float(scalar) / result["global_tokens"]
                )
                updated = result["status"] == "UPDATED"
            else:
                records = sft_records(request, topology, config.vocab_size)
                observer.prepare(request, records)
                row = records[distributed.dp_rank]
                batches = [
                    Microbatch(
                        model_inputs=(row["input_ids"][i : i + 1].to(device),),
                        cu_seqlens=None,
                        objective_inputs=(
                            row["tokens"][i : i + 1]
                            .masked_fill(~row["mask"][i : i + 1], -100)
                            .to(device),
                        ),
                    )
                    for i in range(topology["local_batch"])
                ]
                result = sft_lm.train_step(sftcfg, batches, token_group=token_group)
                updated = result.updated
                native_loss = root_phase(
                    lambda: None if result.loss is None else float(result.loss)
                )
                state.scanned_batches += 1
                state.completed_updates += int(updated)
                data.load_state_dict(state.state_dict())
                result = dict(
                    updated=updated,
                    global_tokens=result.global_targets,
                    requests=state.scanned_batches,
                    updates=state.completed_updates,
                )
            require(
                observer.calls - before_calls == int(updated),
                "Optimizer step observer coverage differs",
            )
            require(updated == (request != 2), "Unexpected update/skip schedule")
            count = observer.summary["count"]
            require(result["global_tokens"] == count, "Global normalization count differs")
            row_report = dict(
                request=request,
                updated=updated,
                native_loss=native_loss,
                reference_loss=observer.summary["loss"],
                global_tokens=count,
                local_tokens=int(observer.records[distributed.dp_rank]["mask"].sum()),
            )
            write_json(
                args.output / f"{args.mode}-request-{request}-rank-{rank:05d}.json", row_report
            )
            if count:
                require(
                    abs(native_loss - observer.summary["loss"])
                    <= LOSS_ATOL + LOSS_RTOL * abs(observer.summary["loss"]),
                    "Native/reference loss differs",
                )
                observer.after_update()
            if request == 2:
                after_zero = snapshot(task_state)
                for key in ("model", "optimizer", "scheduler"):
                    equal(after_zero[key], before_zero[key], f"global-zero-{key}")
            results.append(row_report)
            if request >= 3:
                next_records.append(copy.deepcopy(observer.records))
            if request == 2:
                save_checkpoint(
                    checkpoint, 3, data_state=task_state, strict=True, resume_config=resume_config
                )
                boundary = snapshot(task_state)
                if rank == 0:
                    from tests.acceptance.omni_native_resume import cpu_tree

                    reference_boundary = cpu_tree(reference.optimizer.state_dict())
        final = snapshot(task_state)
        if args.mode == "produce":
            torch.save(
                dict(
                    source=args.expected_commit,
                    producer_pid=os.getpid(),
                    boundary=boundary,
                    final=final,
                    next_records=next_records,
                    results=results[3:],
                    reference_optimizer=reference_boundary,
                ),
                expected_path,
            )
        else:
            equal(final, expected["final"], "native fresh-process next updates")
            equal(
                next_records,
                expected["next_records"],
                "fresh autoregressive samples/logprobs/masks",
            )
            equal(results, expected["results"], "fresh update metrics")
        if args.task == "grpo":
            require(
                trainer.updates == trainer.policy_version == UPDATES
                and trainer.requests == REQUESTS,
                "Final policy counters differ",
            )
        else:
            require(
                state.completed_updates == UPDATES and state.scanned_batches == REQUESTS,
                "Final SFT counters differ",
            )
        require(
            all(
                len(opt.state) == len([p for group in opt.param_groups for p in group["params"]])
                and all(value["step"].item() == UPDATES for value in opt.state.values())
                for opt in training.optimizers
            ),
            "Final Adam coverage/update counters differ",
        )
        hardware.update(
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            final_memory_free_total=list(torch.cuda.mem_get_info()),
        )
        write_json(
            destination,
            dict(
                status="PASS",
                task=args.task,
                mode=args.mode,
                source=prepared["source"],
                topology=topology,
                synthetic_fixture=prepared["fixture"],
                weight_source="native random initialization; exact current weights installed into HF rollout",
                native_engine="DualPipeV/FSDP",
                strict_resume=args.mode == "restore",
                results=results,
                gradient_reports=f"{args.mode}-gradient-request-*.json",
                hardware=hardware,
                reference_attention=identity["attention"],
                external_service="NOT_INTEGRATED",
                full_size="NOT_RUN",
            ),
        )
    finally:
        hook.remove()
        native_optimizer.clip_grad_norm = original_clip


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "prepare", "produce", "restore"))
    parser.add_argument("--task", choices=("sft", "grpo"), required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-config", type=Path)
    args = parser.parse_args(argv)
    profile(args.world, args.pp, args.ep)
    require(
        re.fullmatch("[0-9a-f]{40}", args.expected_commit) is not None,
        "Full expected commit required",
    )
    return args


def main():
    args = parse_args()
    started = time.time()
    try:
        if args.mode == "plan":
            print(
                json.dumps(
                    dict(
                        status="NOT_RUN",
                        commands=plan_commands(args),
                        topology=profile(args.world, args.pp, args.ep),
                    ),
                    indent=2,
                )
            )
        elif args.mode == "prepare":
            print(json.dumps(prepare(args), indent=2))
        else:
            run_gpu(args)
    except Exception as error:
        if args.output.exists():
            rank = int(os.environ.get("RANK", "0"))
            path = args.output / f"{args.mode}-failure-rank-{rank:05d}.json"
            if not path.exists():
                write_json(
                    path,
                    dict(
                        status="FAIL",
                        source=args.expected_commit,
                        mode=args.mode,
                        elapsed_seconds=time.time() - started,
                        error=f"{type(error).__name__}: {error}",
                    ),
                )
        raise


if __name__ == "__main__":
    main()
