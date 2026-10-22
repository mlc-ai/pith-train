"""DP2 complete speech FSDP acceptance; no allocation or automatic retry.

Reuse omni_speech_chain.prepare and its immutable media fixture. Produce runs
four real AdamW updates and saves the native task-v2 checkpoint at scan two;
restore is a separate torchrun process and reproduces the subsequent scans.
The independent HFChain evaluates the concatenated DP objective. Every active
canonical parameter has preclip/postclip gradient and actual update-delta NSE
checks. Frozen components are listed separately. CPU contracts are not GPU evidence.

Codec EOS is always supervised in the production model. Therefore codec modes
cannot express a wholly empty objective: text-branch zeros are recorded as such.
Waveform covers rank-local and global zero, including an actual no-update scan.
Integer codec labels are external synthetic supervision, not a differentiable
Talker-to-waveform sampling path or a released-weight quality test.
"""

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from tests.acceptance import omni_speech_chain as chain

SCHEMA = "qwen3_omni_speech_fsdp_acceptance_v1"
MODES = ("text_and_codec", "waveform", "codec_only")
WORLD = 2
GRADIENT_NSE = 1e-2
OUTPUT_NSE = 1e-3
MAX_NORM = 1.0
CAPS = dict(prepare=120, produce=600, restore=600, kill_after=10)


def scans(mode):
    if mode not in MODES:
        raise ValueError("Unsupported speech mode")
    return 5 if mode == "waveform" else 4


def updates_at(mode, scan):
    if type(scan) is not int or not 0 <= scan <= scans(mode):
        raise ValueError("Invalid scan cursor")
    return scan - int(mode == "waveform" and scan >= 3)


def plan(mode):
    scans(mode)
    return dict(
        schema=SCHEMA,
        status="NOT_RUN",
        training_mode=mode,
        topology=dict(dp=WORLD, cp=1, ep=1, pp=1),
        precision="FP32 sharded masters/AdamW; BF16 compute; FP32 SUM gradients",
        updates=4,
        scans=scans(mode),
        checkpoint=dict(
            format=2,
            policy="conditional_adamw_v1",
            bounded_load=True,
            save_scan=2,
            save_updates=2,
            restore="fresh-process",
        ),
        caps_seconds=CAPS,
        stages=["chain.prepare", "produce", "restore"],
        matrix=dict(
            media=list(chain.MEDIA),
            nonuniform="REQUIRED",
            whole_objective_rank_zero="REQUIRED" if mode == "waveform" else "NOT_EXPRESSIBLE_EOS",
            whole_objective_global_zero="REQUIRED" if mode == "waveform" else "NOT_EXPRESSIBLE_EOS",
            text_target_zero="REQUIRED" if mode == "text_and_codec" else "NOT_APPLICABLE",
        ),
        thresholds=dict(
            gradient_nse_strict_lt=GRADIENT_NSE,
            actual_update_delta_nse_strict_lt=GRADIENT_NSE,
            output_and_loss_nse_strict_lt=OUTPUT_NSE,
        ),
        limitations=[
            "tiny dimensions",
            "no external codec encoder",
            "CP/EP/PP must equal 1",
            "no GPU execution established by this plan",
        ],
    )


def scheduled_batches(samples, mode, scan, rank, device):
    if type(rank) is not int or rank not in range(WORLD) or not 0 <= scan < scans(mode):
        raise ValueError("Invalid DP rank/scan")
    batches = [chain.move(item, device) for item in samples]
    for batch in batches:
        # Full physical forward is identical in shape/order on both ranks.
        # Supervision counts, and supplied RVQ target values, differ explicitly.
        batch["codes"] = (batch["codes"] + rank + scan) % 23
        keep = torch.ones_like(batch["codes"], dtype=torch.bool)
        if rank == 1:
            keep[:, 1::2] = False
            positions = (batch["labels"] != -100).nonzero()
            for row, column in positions[1::2]:
                batch["labels"][row, column] = -100
        if (scan == 1 and rank == 1) or scan == 2:
            batch["labels"].fill_(-100)
        batch["codebook_mask"] = keep
        batch["waveform_fraction"] = (
            0 if mode == "waveform" and (scan == 2 or (scan == 1 and rank == 1)) else 2 - rank
        )
    return batches


class ScheduledData:
    def __init__(self, samples, fingerprint, mode, rank):
        self.samples, self.fingerprint, self.mode, self.rank = samples, fingerprint, mode, rank
        self.scanned_batches = self.completed_updates = 0
        self.last_update_scan = -1
        self.observed = {}

    def get_batch(self, scan_index, device):
        if scan_index != self.scanned_batches:
            raise ValueError("Reader scan mismatch")
        result = scheduled_batches(self.samples, self.mode, scan_index, self.rank, device)
        self.scanned_batches += 1
        self.observed[scan_index] = chain.tensor_tree(result)
        return result

    def commit_update(self, *, scan_index, update_index):
        if (
            scan_index != self.scanned_batches - 1
            or update_index != self.completed_updates
            or updates_at(self.mode, scan_index + 1) != update_index + 1
        ):
            raise ValueError("Reader update mismatch")
        self.completed_updates += 1
        self.last_update_scan = scan_index

    def state_dict(self):
        return dict(
            fingerprint=self.fingerprint,
            mode=self.mode,
            rank=self.rank,
            scanned_batches=self.scanned_batches,
            completed_updates=self.completed_updates,
            last_update_scan=self.last_update_scan,
        )

    def validate_state_dict(self, state):
        if set(state) != set(self.state_dict()) or any(
            state[key] != getattr(self, key) for key in ("fingerprint", "mode", "rank")
        ):
            raise ValueError("Reader identity differs")
        scan, update, last = (
            state[k] for k in ("scanned_batches", "completed_updates", "last_update_scan")
        )
        if any(type(x) is not int for x in (scan, update, last)):
            raise ValueError("Reader counters must be integers")
        wanted_last = scan - 1 if scan else -1
        if self.mode == "waveform" and scan == 3:
            wanted_last = 1
        if update != updates_at(self.mode, scan) or last != wanted_last:
            raise ValueError("Reader update/scan schedule differs")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        for key in ("scanned_batches", "completed_updates", "last_update_scan"):
            setattr(self, key, state[key])


def waveform_objective(output, batch):
    from pithtrain.tasks.omni_objectives import TargetLoss

    fraction = batch["waveform_fraction"]
    if type(fraction) is not int or fraction not in (0, 1, 2):
        raise ValueError("Invalid explicit waveform supervision mask")
    target = torch.sin(torch.arange(output.shape[-1], device=output.device).float() * 0.1) * 0.1
    mask = torch.arange(output.shape[-1], device=output.device) < output.shape[-1] * fraction // 2
    mask = mask.expand_as(output)
    return TargetLoss(
        ((output.float() - target).square() * mask).sum(), mask.sum(dtype=torch.int64)
    )


def cpu_tree(value):
    if isinstance(value, DTensor):
        return value.to_local().detach().cpu().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def compare_tensors(actual, expected, *, threshold=GRADIENT_NSE):
    """Visit the union of names before returning failure; no magnitude exemptions."""
    report = {}
    for name in sorted(actual.keys() | expected.keys()):
        left, right = actual.get(name), expected.get(name)
        row = dict(status="FAIL", native_present=left is not None, hf_present=right is not None)
        if left is not None and right is not None:
            try:
                error = chain.nse(left, right)
                row.update(
                    nse=error,
                    native_norm=float(left.double().norm()),
                    hf_norm=float(right.double().norm()),
                    native_max_abs=float(left.double().abs().max()) if left.numel() else 0.0,
                    hf_max_abs=float(right.double().abs().max()) if right.numel() else 0.0,
                    diff_max_abs=float((left.double() - right.double()).abs().max())
                    if left.numel()
                    else 0.0,
                    status="PASS" if error < threshold else "FAIL",
                )
            except (AssertionError, RuntimeError) as exc:
                row["reason"] = str(exc)
        else:
            row["reason"] = "Missing tensor/gradient; conditional absence is not numerical PASS"
        report[name] = row
    return report


def passed(rows):
    return bool(rows) and all(row["status"] == "PASS" for row in rows.values())


def collect_full(parameters, *, gradients=False):
    # Agree on complete structural identities before any full_tensor collective.
    # Missing one rank's gradient is reported on every rank, never silently zeroed.
    local = {}
    for name, parameter in parameters.items():
        descriptor = dict(
            dtensor=isinstance(parameter, DTensor),
            shape=list(parameter.shape),
            dtype=str(parameter.dtype),
        )
        if isinstance(parameter, DTensor):
            descriptor.update(
                placements=[str(item) for item in parameter.placements],
                ranks=parameter.device_mesh.mesh.cpu().flatten().tolist(),
                device_type=parameter.device_mesh.device_type,
            )
        value = parameter.grad if gradients else parameter
        local[name] = dict(
            parameter=descriptor,
            present=value is not None,
            value_is_dtensor=isinstance(value, DTensor),
            value_dtype=str(value.dtype) if value is not None else None,
            value_shape=list(value.shape) if value is not None else None,
        )
    all_presence = [None] * dist.get_world_size()
    dist.all_gather_object(all_presence, local)
    if any(item.keys() != local.keys() for item in all_presence):
        raise RuntimeError("FSDP canonical parameter identities differ across ranks")
    for name in sorted(parameters):
        descriptors = [item[name]["parameter"] for item in all_presence]
        if not descriptors[0]["dtensor"] or any(item != descriptors[0] for item in descriptors):
            raise RuntimeError(f"FSDP parameter layout differs across ranks: {name}")
        for item in all_presence:
            value = item[name]
            if value["present"] and (
                not value["value_is_dtensor"]
                or value["value_dtype"] != descriptors[0]["dtype"]
                or value["value_shape"] != descriptors[0]["shape"]
            ):
                raise RuntimeError(f"FSDP gradient structure differs: {name}")
    result = {}
    for name in sorted(parameters):
        if not all(item[name]["present"] for item in all_presence):
            result[name] = None
            continue
        tensor = parameters[name].grad if gradients else parameters[name]
        result[name] = tensor.full_tensor().detach().cpu().clone()
    return result


def independent_labels(output, batch):
    first, residual = output.first_labels.clone(), output.residual_labels.clone()
    first[:, :-1].masked_fill_(~batch["codebook_mask"][..., 0], -100)
    residual.masked_fill_(~batch["codebook_mask"][..., 1:], -100)
    return first, residual


def cross_entropy_sum(logits, labels):
    count = (labels != -100).sum(dtype=torch.int64)
    return F.cross_entropy(
        logits.float().flatten(0, -2), labels.flatten(), ignore_index=-100, reduction="sum"
    ), count


def output_record(output, batch, mode, *, independent=False):
    if mode == "waveform":
        item = waveform_objective(output, batch)
        return dict(waveform=cpu_tree(output)), (item.loss_sum, item.target_count), None
    first, residual = (
        independent_labels(output, batch)
        if independent
        else (output.first_labels, output.residual_labels)
    )
    first_sum, first_count = cross_entropy_sum(output.first_logits, first)
    residual_sum, residual_count = cross_entropy_sum(output.residual_logits, residual)
    text = (
        cross_entropy_sum(output.thinker_logits, batch["labels"])
        if mode == "text_and_codec"
        else None
    )
    return (
        dict(
            first=cpu_tree(output.first_logits),
            residual=cpu_tree(output.residual_logits),
            text=cpu_tree(output.thinker_logits),
            first_labels=cpu_tree(first),
            residual_labels=cpu_tree(residual),
        ),
        (first_sum + residual_sum, first_count + residual_count),
        text,
    )


def reference_step(hf, optimizer, samples, mode, scan, device):
    from pithtrain.tasks import omni_speech as task

    # The reference is an independent persistent optimizer trajectory, never reset
    # from native after initialization. Do not perturb native RNG while observing.
    rng = task._rng_state()
    try:
        optimizer.zero_grad(set_to_none=True)
        outputs, sums, counts, text_sums, text_counts = {}, [], [], [], []
        rank_counts = []
        for rank in range(WORLD):
            local_count = [0, 0]
            for index, batch in enumerate(scheduled_batches(samples, mode, scan, rank, device)):
                if mode == "waveform":
                    output = hf.decode_training_codes(batch["codes"].transpose(1, 2))
                else:
                    args, kwargs = chain.speech_args(batch)
                    output = hf(*args, **kwargs)
                record, primary, text = output_record(output, batch, mode, independent=True)
                outputs[(rank, index)] = record
                sums.append(primary[0])
                counts.append(primary[1])
                local_count[0] += int(primary[1])
                if text is not None:
                    text_sums.append(text[0])
                    text_counts.append(text[1])
                    local_count[1] += int(text[1])
            rank_counts.append(local_count)
        primary_count = sum(counts)
        text_count = sum(text_counts) if text_counts else 0
        updated = bool(primary_count) or bool(text_count)
        loss = sum(sums) / primary_count.clamp_min(1)
        if text_sums:
            loss = loss + sum(text_sums) / text_count.clamp_min(1)
        parameters = {name: p for name, p in hf.model.named_parameters() if p.requires_grad}
        before = cpu_tree(parameters)
        pre, post = {}, {}
        if updated:
            loss.backward()
            pre = {name: cpu_tree(p.grad) for name, p in parameters.items()}
            norm = torch.sqrt(
                sum(
                    p.grad.double().square().sum()
                    for p in parameters.values()
                    if p.grad is not None
                )
            )
            scale = (MAX_NORM / norm.clamp_min(1e-12)).clamp(max=1.0)
            for p in parameters.values():
                if p.grad is not None:
                    p.grad.mul_(scale.to(p.grad.dtype))
            post = {name: cpu_tree(p.grad) for name, p in parameters.items()}
            optimizer.step()
        after = cpu_tree(parameters)
        return dict(
            outputs=outputs,
            loss=loss.detach().cpu(),
            counts=rank_counts,
            global_counts=[int(primary_count), int(text_count)],
            updated=updated,
            pre=pre,
            post=post,
            delta={name: after[name] - before[name] for name in before},
        )
    finally:
        task._set_rng(rng)


@contextmanager
def observers(model, optimizer, batches, mode, parameters, reference, report):
    from pithtrain.tasks import omni_speech as task

    observed = []
    original_clip = task.clip_sharded_grad_norm

    def record_output(output):
        if len(observed) >= len(batches):
            raise RuntimeError("Unexpected native forward count")
        record, primary, text = output_record(output, batches[len(observed)], mode)
        observed.append((record, primary, text))

    def forward_hook(module, inputs, output):
        record_output(output)

    def clip_observer(owned_model, maximum, *, group):
        if owned_model is not model or maximum != MAX_NORM:
            raise RuntimeError("Unexpected native clipping contract")
        report["preclip_gradients"] = compare_tensors(
            collect_full(parameters, gradients=True), reference["pre"]
        )
        return original_clip(owned_model, maximum, group=group)

    def pre_step(opt, args, kwargs):
        report["postclip_gradients"] = compare_tensors(
            collect_full(parameters, gradients=True), reference["post"]
        )

    hook = (model.code2wav if mode == "waveform" else model).register_forward_hook(forward_hook)
    step_hook = optimizer.register_step_pre_hook(pre_step)
    task.clip_sharded_grad_norm = clip_observer
    try:
        yield observed
    finally:
        task.clip_sharded_grad_norm = original_clip
        hook.remove()
        step_hook.remove()


def optimizer_evidence(parameters, optimizer, expected_steps):
    rows = {}
    for name, parameter in sorted(parameters.items()):
        saved = optimizer.state.get(parameter, {})
        row = dict(status="FAIL", parameter_dtype=str(parameter.dtype), present=bool(saved))
        try:
            step = saved["step"]
            step_value = float(step)
            row["step"] = step_value if math.isfinite(step_value) else str(step_value)
            moments = {}
            for field in ("exp_avg", "exp_avg_sq"):
                value = saved[field]
                local = value.to_local() if isinstance(value, DTensor) else value
                moments[field] = dict(
                    dtype=str(value.dtype), finite=bool(torch.isfinite(local).all())
                )
            row["moments"] = moments
            if (
                row["step"] == expected_steps
                and parameter.dtype == torch.float32
                and all(
                    item["dtype"] == "torch.float32" and item["finite"] for item in moments.values()
                )
            ):
                row["status"] = "PASS"
        except (KeyError, TypeError, ValueError) as exc:
            row["reason"] = str(exc)
        rows[name] = row
    return rows


def state_snapshot(model, optimizer, scheduler, data, state):
    from pithtrain.tasks import omni_speech as task

    return cpu_tree(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            data=data.state_dict(),
            task=state.state_dict(),
            rng=task._rng_state(),
        )
    )


def evidence_save(path, value):
    if path.exists():
        raise FileExistsError(path)
    torch.save(value, path)
    return chain.sha(path)


def evidence_load(path, digest):
    if chain.sha(path) != digest:
        raise chain.PreconditionError("Producer evidence checksum differs")
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_producer(producer, args, fixture_sha):
    expected = dict(
        schema=SCHEMA,
        status="PASS",
        mode="produce",
        source=args.source,
        training_mode=args.training_mode,
        fixture_sha256=fixture_sha,
        runtime_lock_sha256=chain.sha(args.runtime_lock),
        rank=args.rank,
        topology=dict(dp=WORLD, cp=1, ep=1, pp=1),
    )
    if producer.get("runtime", {}).get("hardware") != args.hardware:
        raise chain.PreconditionError("Producer hardware lock differs")
    if any(producer.get(k) != v for k, v in expected.items()) or producer.get("pid") == os.getpid():
        raise chain.PreconditionError("Fresh producer source/fixture/runtime/mode/status differs")


def run_gate(args, samples, native, hf, fixture):
    from pithtrain.modules.omni_speech_fsdp import apply_speech_fsdp
    from pithtrain.tasks import omni_speech as task
    from pithtrain.tasks.sft_lm import SFTState

    mode, rank = args.training_mode, dist.get_rank()
    cfg = task.OmniSpeechCfg(
        mode=mode,
        max_updates=4,
        max_batches=scans(mode),
        max_grad_norm=MAX_NORM,
        waveform_objective_id="speech-sine-prefix-mask-mse-v1",
    )
    task.configure_trainable_components(native, mode)
    task.configure_trainable_components(hf, mode)
    mesh = init_device_mesh("cuda", (WORLD,), mesh_dim_names=("dp",))
    apply_speech_fsdp(native, mesh, mode=mode)
    parameters = {chain.canonical(n): p for n, p in native.named_parameters() if p.requires_grad}
    if parameters.keys() != {n for n, p in hf.model.named_parameters() if p.requires_grad}:
        raise RuntimeError("Full active canonical coverage differs")
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4, weight_decay=0, foreach=False)
    hf_optimizer = torch.optim.AdamW(
        [p for p in hf.parameters() if p.requires_grad], lr=1e-4, weight_decay=0, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    data = ScheduledData(samples, fixture["samples_sha256"], mode, rank)
    state = SFTState()
    contract = dict(
        schema=SCHEMA,
        source=args.source,
        fixture_sha256=chain.sha(args.prepared / "fixture.json"),
        training_mode=mode,
        dp=WORLD,
        cp=1,
        ep=1,
        pp=1,
        hsdp_replica=1,
        runtime_lock_sha256=chain.sha(args.runtime_lock),
        schedule=plan(mode)["matrix"],
    )
    options = dict(
        model=native,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        state=state,
        token_group=mesh.get_group(),
        gradient_reduction="sum",
        waveform_objective=waveform_objective,
        checkpoint_contract=contract,
    )
    result = dict(
        active_parameters=list(sorted(parameters)),
        frozen_parameters=sorted(
            chain.canonical(n) for n, p in native.named_parameters() if not p.requires_grad
        ),
        optimizer="torch.optim.AdamW foreach=False",
        reference_trajectory="independent persistent HFChain",
        checkpoints={},
        evidence={},
        steps=[],
    )
    producer, start = None, 0
    if args.mode == "restore":
        producer = json.loads((args.producer / f"rank-{rank}" / "result.json").read_text())
        validate_producer(producer, args, contract["fixture_sha256"])
        cfg.max_updates = cfg.max_batches = 2
        task.launch(
            cfg, data, **options, checkpoint_root=args.producer / "checkpoints", resume_scan=2
        )
        expected = evidence_load(
            args.producer / f"rank-{rank}" / "scan-2.pt", producer["evidence"]["2"]
        )
        chain.exact(
            state_snapshot(native, optimizer, scheduler, data, state),
            expected["native"],
            "restored-boundary",
        )
        hf.model.load_state_dict(expected["hf_model"], strict=True)
        hf_optimizer.load_state_dict(expected["hf_optimizer"])
        result["restored_boundary"] = "EXACT"
        start = 2
    for scan in range(start, scans(mode)):
        reference = reference_step(hf, hf_optimizer, samples, mode, scan, args.device)
        before = collect_full(parameters)
        before_state = state_snapshot(native, optimizer, scheduler, data, state)
        report = dict(
            scan=scan + 1,
            update=updates_at(mode, scan + 1),
            target_counts=reference["counts"],
            global_target_counts=reference["global_counts"],
            expected_update=reference["updated"],
            status="FAIL",
        )
        local_batches = scheduled_batches(samples, mode, scan, rank, args.device)
        cfg.max_batches, cfg.max_updates = scan + 1, 4
        with observers(
            native, optimizer, local_batches, mode, parameters, reference, report
        ) as observed:
            task.launch(cfg, data, **options, checkpoint_root=args.output / "checkpoints")
        after = collect_full(parameters)
        report["actual_update_delta"] = compare_tensors(
            {n: after[n] - before[n] for n in before}, reference["delta"]
        )
        output_rows, label_errors = {}, []
        local_loss = torch.zeros((), dtype=torch.float32, device=args.device)
        if len(observed) != len(local_batches):
            raise AssertionError("Incomplete native forward observations")
        for index, (record, primary, text) in enumerate(observed):
            expected = reference["outputs"][(rank, index)]
            for key in ("first_labels", "residual_labels"):
                if key in record:
                    try:
                        chain.exact(record[key], expected[key], key)
                    except AssertionError as exc:
                        label_errors.append(str(exc))
            output_rows.update(
                compare_tensors(
                    {
                        f"{chain.MEDIA[index]}.{k}": v
                        for k, v in record.items()
                        if not k.endswith("labels")
                    },
                    {
                        f"{chain.MEDIA[index]}.{k}": v
                        for k, v in expected.items()
                        if not k.endswith("labels")
                    },
                    threshold=OUTPUT_NSE,
                )
            )
            local_loss += primary[0].detach() / max(reference["global_counts"][0], 1)
            if text is not None:
                local_loss += text[0].detach() / max(reference["global_counts"][1], 1)
        native_counts = [
            sum(int(primary[1]) for _, primary, _ in observed),
            sum(int(text[1]) for _, _, text in observed if text is not None),
        ]
        all_counts = [None] * WORLD
        dist.all_gather_object(all_counts, native_counts)
        report["native_target_counts"] = all_counts
        report["count_match"] = all_counts == reference["counts"]
        report["updated_components"] = sorted(
            {
                name.split(".")[0]
                for name in before
                if torch.count_nonzero(after[name] - before[name])
            }
        )
        required_components = {
            "waveform": {"code2wav"},
            "codec_only": {"talker"},
            "text_and_codec": {"thinker", "talker"},
        }[mode]
        report["component_update_match"] = set(report["updated_components"]) == (
            required_components if reference["updated"] else set()
        )
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        report["loss"] = compare_tensors(
            {"loss": local_loss.cpu()}, {"loss": reference["loss"]}, threshold=OUTPUT_NSE
        )
        report["outputs"], report["label_errors"] = output_rows, label_errors
        report["optimizer_parameters"] = optimizer_evidence(
            parameters, optimizer, updates_at(mode, scan + 1)
        )
        current = state_snapshot(native, optimizer, scheduler, data, state)
        if not reference["updated"]:
            for key in ("model", "optimizer", "scheduler", "rng"):
                chain.exact(before_state[key], current[key], f"global-zero.{key}")
            report["global_zero"] = "EXACT_NO_UPDATE_MODEL_OPTIMIZER_SCHEDULER_RNG"
        if state.scanned_batches != scan + 1 or state.completed_updates != updates_at(
            mode, scan + 1
        ):
            raise AssertionError("Actual task cursors differ from explicit schedule")
        checks = [
            report["actual_update_delta"],
            report["outputs"],
            report["loss"],
            report["optimizer_parameters"],
        ]
        if reference["updated"]:
            checks += [report.get("preclip_gradients", {}), report.get("postclip_gradients", {})]
        report["status"] = (
            "PASS"
            if not label_errors
            and report["count_match"]
            and report["component_update_match"]
            and all(passed(rows) for rows in checks)
            else "FAIL"
        )
        payload = dict(
            native=current,
            hf_model=cpu_tree(hf.model.state_dict()),
            hf_optimizer=cpu_tree(hf_optimizer.state_dict()),
            observed=data.observed[scan],
        )
        if producer is not None:
            expected = evidence_load(
                args.producer / f"rank-{rank}" / f"scan-{scan + 1}.pt",
                producer["evidence"][str(scan + 1)],
            )
            chain.exact(payload, expected, f"fresh-resume.scan-{scan + 1}")
            report["fresh_resume"] = "EXACT_NATIVE_HF_OPTIMIZER_SCHEDULER_RNG_DATA"
        digest = evidence_save(args.rank_output / f"scan-{scan + 1}.pt", payload)
        result["evidence"][str(scan + 1)] = digest
        report_path = args.rank_output / f"scan-{scan + 1}.json"
        chain.write_json(report_path, report)
        result["steps"].append(
            dict(
                scan=scan + 1,
                update=report["update"],
                status=report["status"],
                report=report_path.name,
                sha256=chain.sha(report_path),
            )
        )
        failures = [None] * WORLD
        dist.all_gather_object(failures, report["status"])
        if any(item != "PASS" for item in failures):
            raise AssertionError("Strict numerical gate failed; complete per-name reports retained")
    result.update(
        status="PASS",
        checkpoint_contract="native task-v2 conditional_adamw_v1 bounded_load",
        fresh_process=args.mode == "restore",
    )
    return result


def runtime(args):
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != WORLD
        or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != WORLD
    ):
        raise chain.PreconditionError("This gate requires exactly one local DP2 group")
    rank = int(os.environ["LOCAL_RANK"])
    if rank != int(os.environ["RANK"]) or rank not in range(WORLD):
        raise chain.PreconditionError("Only single-node ordered DP2 is supported")
    if importlib.metadata.version("transformers") != "5.17.0":
        raise chain.PreconditionError("Transformers must be exactly 5.17.0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD:
        raise chain.PreconditionError("Expose exactly the two reviewed CUDA devices")
    if torch.cuda.get_device_name(rank) != args.hardware or torch.cuda.get_device_capability(rank)[
        0
    ] not in (9, 10):
        raise chain.PreconditionError("Explicit Hopper/Blackwell hardware lock differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise chain.PreconditionError("Deterministic CUBLAS_WORKSPACE_CONFIG required")
    torch.cuda.set_device(rank)
    args.device = torch.device("cuda", rank)
    random.seed(chain.SEED)
    np.random.seed(chain.SEED)
    torch.manual_seed(chain.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    observed = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        python=platform.python_version(),
        transformers=importlib.metadata.version("transformers"),
        packages=sorted(
            subprocess.check_output(
                [os.sys.executable, "-m", "pip", "freeze"], text=True, timeout=20
            ).splitlines()
        ),
    )
    digest = chain.verify_runtime_lock(args.runtime_lock, observed)
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.free",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    # Host/PCI topology is diagnostic and may be unavailable inside a GPU container.
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5, check=False
    )
    return dict(
        **observed,
        runtime_lock_sha256=digest,
        hardware=args.hardware,
        nvidia_smi=smi.stdout,
        nvlink_topology=dict(
            status="AVAILABLE" if topology.returncode == 0 else "UNAVAILABLE",
            returncode=topology.returncode,
            stdout=topology.stdout,
            stderr=topology.stderr,
        ),
        free_total_bytes=list(torch.cuda.mem_get_info(rank)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "produce", "restore"))
    parser.add_argument("--training-mode", required=True, choices=MODES)
    parser.add_argument("--expected-commit")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--hardware")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        print(json.dumps(plan(args.training_mode), indent=2))
        return
    if (
        not args.expected_commit
        or not args.hardware
        or args.runtime_lock is None
        or args.prepared is None
        or args.output is None
        or (args.mode == "restore" and args.producer is None)
    ):
        parser.error(
            "Explicit source, prepared fixture, runtime lock, hardware, output and restore producer required"
        )
    rank = int(os.environ.get("RANK", "0"))
    args.rank = rank
    args.rank_output = args.output / f"rank-{rank}"
    args.rank_output.mkdir(parents=True, exist_ok=False)
    result = dict(
        **plan(args.training_mode),
        mode=args.mode,
        pid=os.getpid(),
        rank=rank,
        gpu_acceptance="NOT_RUN",
    )
    started, phase = time.monotonic(), "source_precondition"
    try:
        args.source = chain.source_identity(args.expected_commit)
        result["source"] = args.source
        result["runtime"] = runtime(args)
        result["runtime_lock_sha256"] = chain.sha(args.runtime_lock)
        config, samples, fixture = chain.load_fixture(args.prepared, args.expected_commit)
        result["fixture_sha256"] = chain.sha(args.prepared / "fixture.json")
        dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=args.device)
        phase = "model_setup"
        native, hf, attention = chain.build_pair(config, args.device)
        result["reference_attention"] = attention
        phase = "numerical_and_native_resume"
        result.update(run_gate(args, samples, native.model, hf, fixture))
        result["gpu_acceptance"] = "PASS_THIS_GATE_ONLY"
    except Exception as exc:
        result.update(chain.failure_report(phase, exc))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        if torch.cuda.is_initialized():
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        chain.write_json(args.rank_output / "result.json", result)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
