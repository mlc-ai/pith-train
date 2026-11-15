"""Original full-size first-update numerical checks, reused by released continuation.

Extracted without numerical changes from the reviewed eight-H200 driver.
The acceptance entrance is omni_released_continuation, not this helper.
"""

import hashlib
import json
import math
import os
from pathlib import Path

GIB = 2**30


REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


CONFIG_SHA = "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"


TEXT = 30_532_646_912


EXPERTS = 28_991_029_248


EDGE = 622_856_192


NONEXPERT_LAYER = 19_140_864


ENCODERS = {"image": (538_631_408, 351), "video": (538_631_408, 351), "audio": (647_927_168, 525)}


CHUNK_ELEMENTS = 2**20


def estimate(modality):
    encoder, keys = ENCODERS[modality]
    # PP2 V partition remains 24 layers per stage. EP4 owns 1/4 experts;
    # attention-DP4 shards every non-expert parameter four ways.
    heavy_nonexpert = 24 * NONEXPERT_LAYER + EDGE + encoder
    light_nonexpert = 24 * NONEXPERT_LAYER
    heavy_shard = EXPERTS // 8 + heavy_nonexpert // 4
    light_shard = EXPERTS // 8 + light_nonexpert // 4
    heavy_unsharded = EXPERTS // 8 + heavy_nonexpert
    light_unsharded = EXPERTS // 8 + light_nonexpert
    result = {
        "modality": modality,
        "official_revision": REVISION,
        "parameters": TEXT + encoder,
        "parameter_keys": 531 + keys,
        "topology": {
            "world": 8,
            "pp": 2,
            "ep": 4,
            "cp": 1,
            "attention_dp": 4,
            "expert_dp": 1,
            "hsdp_replica": 1,
        },
        "global_batch_size": 16,
        "micro_batch_size": 1,
        "steps": 1,
        "microbatches_per_attention_dp_rank": 4,
        "weight_source": "production_random_fp32",
        "optimizer": "production_AdamW",
        "optimizer_default_cuda_implementation": "foreach",
        "all_ranks_initial_cpu_fp32_bytes": 16 * (heavy_unsharded + light_unsharded),
        "host_limit_bytes": 512 * GIB,
        "host_available_admission_bytes": 192 * GIB,
        "cpu_limit": 32,
        "threads_per_rank": 4,
        "outer_timeout_seconds": 600,
        "kill_after_seconds": 10,
        "estimate_only": True,
        "state_estimates_exclude_activation_and_allocator": True,
        "admission_note": "20S + 6U + original 8 GiB margin; deliberately allows all BF16 unsharded weights and FP32 full gradients to overlap foreach even though step finalization reshards",
        "steady_state": "NOT_RUN",
        "hf_parity": "SEPARATE_GATE",
        "checkpoint": "NOT_RUN",
        "released_weights": "NOT_RUN",
    }
    for pp, shard, unsharded in (
        (0, heavy_shard, heavy_unsharded),
        (1, light_shard, light_unsharded),
    ):
        result.update(
            {
                f"pp{pp}_sharded_elements": shard,
                f"pp{pp}_unsharded_elements": unsharded,
                f"pp{pp}_first_backward_state_bytes": 4 * shard + 6 * unsharded,
                f"pp{pp}_first_optimizer_state_bytes": 16 * shard,
                f"pp{pp}_foreach_sqrt_transient_bytes": 4 * shard,
                f"pp{pp}_first_optimizer_peak_lower_bound_bytes": 20 * shard,
                f"pp{pp}_conservative_retained_full_weight_gradient_bytes": 6 * unsharded,
                f"pp{pp}_initial_free_gpu_admission_bytes": 20 * shard + 6 * unsharded + 8 * GIB,
                f"pp{pp}_second_backward_state_bytes": 12 * shard + 6 * unsharded,
            }
        )
    return result


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_optional_text(path):
    """Return a file's stripped text, or None when it does not exist."""
    path = Path(path)
    return path.read_text().strip() if path.exists() else None


def resource_inventory():
    """Read ancestor cgroup limits and this process's kernel CPU affinity.

    Slurm puts memory limits above the task cgroup and limits CPUs by affinity rather than CFS
    quota, so without a cgroup namespace neither the cgroup root nor the task cgroup holds the
    limit. Host-wide free memory and environment variables are never treated as limits.
    """
    directories = set()
    for line in (read_optional_text("/proc/self/cgroup") or "").splitlines():
        _, controllers, relative = line.split(":", 2)
        for mount in (read_optional_text("/proc/self/mountinfo") or "").splitlines():
            left, right = mount.split(" - ", 1)
            fields, info = left.split(), right.split()
            if info[0] not in ("cgroup", "cgroup2"):
                continue
            if controllers and not set(controllers.split(",")).intersection(info[2].split(",")):
                continue
            if not controllers and info[0] != "cgroup2":
                continue
            root, target = Path(fields[3]), Path(fields[4])
            current = Path(relative)
            if not current.is_relative_to(root):
                continue
            current = target / current.relative_to(root)
            while current.is_relative_to(target):
                directories.add(current)
                if current == target:
                    break
                current = current.parent
    # Also retain the original container-root/v1 paths for namespaced mounts.
    directories.update(
        Path(p) for p in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory", "/sys/fs/cgroup/cpu")
    )
    memories, quotas = [], []
    for directory in sorted(directories):
        raw = read_optional_text(directory / "memory.max")
        used = read_optional_text(directory / "memory.current")
        peak = read_optional_text(directory / "memory.peak")
        if raw is None:
            raw = read_optional_text(directory / "memory.limit_in_bytes")
            used = read_optional_text(directory / "memory.usage_in_bytes")
            peak = read_optional_text(directory / "memory.max_usage_in_bytes")
        if raw and raw != "max" and int(raw) < 2**60:
            memories.append(
                dict(
                    path=str(directory),
                    limit=int(raw),
                    used=int(used) if used else None,
                    peak=int(peak) if peak else None,
                )
            )
        raw_cpu = read_optional_text(directory / "cpu.max")
        if raw_cpu:
            q, period = raw_cpu.split()
        else:
            q = read_optional_text(directory / "cpu.cfs_quota_us")
            period = read_optional_text(directory / "cpu.cfs_period_us")
        if q and q != "max" and int(q) > 0 and period and int(period) > 0:
            quotas.append(int(q) / int(period))
    available = next(
        (
            int(line.split()[1]) * 1024
            for line in (read_optional_text("/proc/meminfo") or "").splitlines()
            if line.startswith("MemAvailable:")
        ),
        None,
    )
    for row in memories:
        if row["used"] is not None:
            remaining = row["limit"] - row["used"]
            available = min(available, remaining) if available is not None else remaining
        else:
            # A missing usage counter must not substitute host-wide free RAM.
            available = 0
    limiting = (
        min(memories, key=lambda row: (row["limit"], -(row["used"] or 0))) if memories else {}
    )
    affinity = len(os.sched_getaffinity(0))
    quota = min(quotas) if quotas else None
    return dict(
        memory_limit_bytes=limiting.get("limit"),
        memory_current_bytes=limiting.get("used"),
        cgroup_lifetime_peak_bytes=limiting.get("peak"),
        memory_limits=memories,
        available_bytes=available,
        cpu_limit=min(affinity, quota) if quota is not None else affinity,
        cpu_quota=quota,
        cpu_affinity_count=affinity,
        cpu_limit_basis="minimum_of_kernel_affinity_and_all_finite_ancestor_quotas",
    )


def resource_errors(resources):
    errors = []
    limit, available = resources["memory_limit_bytes"], resources["available_bytes"]
    if limit is None or not 192 * GIB <= limit <= 512 * GIB:
        errors.append("A readable finite host memory hard limit in [192,512] GiB is required")
    if available is None or available < 192 * GIB:
        errors.append("At least 192 GiB actual available cgroup/host RAM is required")
    if resources["cpu_limit"] is None or not 0 < resources["cpu_limit"] <= 32:
        errors.append("Kernel CPU quota/affinity capacity in (0,32] is required")
    return errors


def component_groups(config, modality):
    text = ["model.embed_tokens.", "model.norm.", "lm_head."]
    text += [f"model.layers.{i}." for i in range(config.text_config.num_hidden_layers)]
    if modality == "audio":
        from tests.acceptance.omni_audio_training import training_groups

        return tuple(training_groups(config))
    return tuple(
        ["visual.patch_embed.", "visual.pos_embed.", "visual.merger."]
        + [f"visual.blocks.{i}." for i in range(config.vision_config.depth)]
        + [
            f"visual.merger_list.{i}."
            for i in range(len(config.vision_config.deepstack_visual_indexes))
        ]
        + text
    )


def canonical(name):
    return name if name.startswith(("visual.", "audio_tower.", "lm_head.")) else "model." + name


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def scan_tensor(value, *, digest=False):
    """Bounded scratch; every local element enters finite/norm/max and optional SHA256."""
    import torch

    value = local_tensor(value).detach()
    if not value.is_contiguous():
        raise AssertionError("Unexpected non-contiguous parameter/gradient; no full copy allowed")
    flat = value.view(-1)
    squared = torch.zeros((), device=value.device, dtype=torch.float64)
    max_abs = torch.zeros((), device=value.device, dtype=torch.float32)
    finite = torch.ones((), device=value.device, dtype=torch.bool)
    nonzero = torch.zeros((), device=value.device, dtype=torch.int64)
    sha = hashlib.sha256() if digest else None
    for offset in range(0, flat.numel(), CHUNK_ELEMENTS):
        chunk = flat[offset : offset + CHUNK_ELEMENTS]
        finite &= torch.isfinite(chunk).all()
        squared += chunk.double().nan_to_num().square().sum()
        max_abs = torch.maximum(max_abs, chunk.abs().nan_to_num().max())
        nonzero += torch.count_nonzero(chunk)
        if sha is not None:
            # One small synchronous copy at a time; no full-model clone or pinned host pool.
            sha.update(chunk.cpu().contiguous().numpy().tobytes())
    result = dict(
        elements=flat.numel(),
        dtype=str(value.dtype),
        finite=bool(finite.item()),
        norm=math.sqrt(float(squared.item())),
        max_abs=float(max_abs.item()),
        nonzero_elements=int(nonzero.item()),
    )
    if sha is not None:
        result["sha256"] = sha.hexdigest()
    # Preserve the diagnostic even on NaN/Inf without producing invalid JSON.
    for key in ("norm", "max_abs"):
        if not math.isfinite(result[key]):
            result[key] = None
    return result


def validate_rows(rows, groups):
    """Pure report validation; do not lose later parameter diagnostics on first failure."""
    errors = []
    summary = {
        prefix: dict(
            elements=0, gradient_nonzero_elements=0, changed_parameter_shards=0, parameter_shards=0
        )
        for prefix in groups
    }
    for rank in rows:
        for name, item in rank["parameters"].items():
            owners = [prefix for prefix in groups if name.startswith(prefix)]
            if len(owners) != 1:
                errors.append(f"rank {rank['rank']}: missing/ambiguous component {name}")
                continue
            result = summary[owners[0]]
            result["elements"] += item["before"]["elements"]
            result["parameter_shards"] += 1
            grad = item.get("gradient")
            if grad is None or not grad["finite"] or grad["dtype"] != "torch.float32":
                errors.append(f"rank {rank['rank']}: missing/nonfinite/non-FP32 gradient {name}")
            else:
                result["gradient_nonzero_elements"] += grad["nonzero_elements"]
            if not item["before"]["finite"] or not item.get("after", {}).get("finite", False):
                errors.append(f"rank {rank['rank']}: nonfinite/missing parameter values {name}")
            if item.get("changed"):
                result["changed_parameter_shards"] += 1
            state = item.get("optimizer_state", {})
            if state.get("step") != 1:
                errors.append(f"rank {rank['rank']}: Adam step is not exactly 1 for {name}")
            for key in ("exp_avg", "exp_avg_sq"):
                v = state.get(key)
                if v is None or not v["finite"] or v["dtype"] != "torch.float32":
                    errors.append(f"rank {rank['rank']}: invalid FP32 Adam {key} for {name}")
    for prefix, result in summary.items():
        if (
            min(
                result["elements"],
                result["gradient_nonzero_elements"],
                result["changed_parameter_shards"],
            )
            <= 0
        ):
            errors.append(f"Missing/zero gradient/no actual update component {prefix}")
    return summary, errors


def inspect_production_adamw(optimizer, parameters, *, expected_lr):
    """Validate defaults and exact inventory; observe, never override, Torch selection.

    Selection is predicted from the actual group arguments, not evidence of a
    completed step. The existing post-step all-parameter state checks remain the
    only acceptance of an executed update. CPU and CUDA can select different paths.
    """
    import inspect

    import torch
    from torch.distributed.tensor import DTensor
    from torch.optim.optimizer import _default_to_fused_or_foreach

    assert type(optimizer) is torch.optim.AdamW, "Production factory must return AdamW"
    assert not optimizer.state, "First-step optimizer state must be empty"
    assert parameters, "Empty canonical parameter inventory"
    assert len(optimizer.param_groups) == 1, "Production factory uses one parameter group"
    expected = list(parameters.values())
    assert len({id(p) for p in expected}) == len(expected), "Canonical parameter alias/duplicate"
    actual = optimizer.param_groups[0]["params"]
    assert len(actual) == len(expected), (
        "Optimizer parameter count differs from canonical inventory"
    )
    assert len({id(p) for p in actual}) == len(actual), "Duplicate optimizer parameter"
    assert {id(p) for p in actual} == {id(p) for p in expected}, (
        "Optimizer parameter identity mismatch"
    )
    assert all(
        isinstance(p, DTensor) and p.dtype == torch.float32 and p.requires_grad for p in actual
    ), "Every production parameter must be trainable FP32 DTensor"
    group = optimizer.param_groups[0]
    defaults = dict(
        foreach=None,
        fused=None,
        differentiable=False,
        capturable=False,
        maximize=False,
        amsgrad=False,
        decoupled_weight_decay=True,
        lr=expected_lr,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    for key, value in defaults.items():
        assert group.get(key) == value, f"Production AdamW default changed: {key}"
    fused, foreach = _default_to_fused_or_foreach(actual, group["differentiable"])
    assert type(fused) is bool and type(foreach) is bool
    assert not fused, "Production default does not opt in to fused AdamW"
    local_bytes = sum(local_tensor(p).numel() * p.element_size() for p in actual)
    return dict(
        status="PASS",
        interpretation="CONTRACT_ONLY_NOT_EXECUTED",
        selection_kind="PREDICTED_FROM_TORCH_DEFAULTS",
        predicted_implementation="foreach" if foreach else "single_tensor",
        fused=fused,
        foreach=foreach,
        overrides_applied=False,
        parameter_count=len(actual),
        parameter_names=sorted(parameters),
        local_parameter_bytes=local_bytes,
        foreach_sqrt_transient_bytes=local_bytes if foreach else 0,
        original_production_options=defaults,
        torch_version=torch.__version__,
        selector_source_sha256=hashlib.sha256(
            inspect.getsource(_default_to_fused_or_foreach).encode()
        ).hexdigest(),
    )
