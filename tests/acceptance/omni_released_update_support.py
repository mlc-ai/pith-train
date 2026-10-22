"""Original full-size first-update numerical checks, reused by released continuation.

Extracted without numerical changes from the reviewed eight-H200 driver.
The acceptance entrance is omni_released_continuation, not this helper.
"""

import hashlib
import json
import math
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


def resource_inventory():
    """Read effective Linux limits without using host-wide free memory as a cgroup limit."""

    def read(path):
        p = Path(path)
        return p.read_text().strip() if p.exists() else None

    raw_limit = read("/sys/fs/cgroup/memory.max")
    raw_used = read("/sys/fs/cgroup/memory.current")
    raw_peak = read("/sys/fs/cgroup/memory.peak")
    raw_cpu = read("/sys/fs/cgroup/cpu.max")
    if raw_limit is None:
        raw_limit = read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        raw_used = read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    limit = int(raw_limit) if raw_limit and raw_limit != "max" else None
    used = int(raw_used) if raw_used else None
    quota = None
    if raw_cpu:
        q, period = raw_cpu.split()
        if q != "max":
            quota = int(q) / int(period)
    else:
        q = read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and period and int(q) > 0:
            quota = int(q) / int(period)
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    if limit is not None and used is not None:
        available = min(available, limit - used) if available is not None else limit - used
    return dict(
        memory_limit_bytes=limit,
        memory_current_bytes=used,
        cgroup_lifetime_peak_bytes=int(raw_peak) if raw_peak else None,
        available_bytes=available,
        cpu_limit=quota,
        cpu_max_raw=raw_cpu,
    )


def resource_errors(resources):
    errors = []
    limit, available = resources["memory_limit_bytes"], resources["available_bytes"]
    if limit is None or not 192 * GIB <= limit <= 512 * GIB:
        errors.append("A readable finite host memory hard limit in [192,512] GiB is required")
    if available is None or available < 192 * GIB:
        errors.append("At least 192 GiB actual available cgroup/host RAM is required")
    if resources["cpu_limit"] is None or not 0 < resources["cpu_limit"] <= 32:
        errors.append("A readable CPU quota in (0,32] is required")
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
