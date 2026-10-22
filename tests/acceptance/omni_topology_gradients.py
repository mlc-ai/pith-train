"""Complete per-parameter evidence; no missing/name/noise-floor skip can pass."""

import math
from pathlib import Path

import torch

from tests.acceptance.omni_topology_contract import (
    GRAD_EPS,
    SMALL_PARAMETERS,
    canonical,
    expected_parameters,
    write_json,
)


def inspect_structure(actual, reference, expected):
    issues = []
    maps = []
    for side, entries in (("actual", list(actual)), ("reference", list(reference))):
        mapping = {}
        for name, parameter in entries:
            if name in mapping:
                issues.append(f"{side}: duplicate parameter name {name}")
            mapping[name] = parameter
        if len(entries) != len(expected):
            issues.append(f"{side}: count {len(entries)} != expected {len(expected)}")
        missing, extra = (
            sorted(expected.keys() - mapping.keys()),
            sorted(mapping.keys() - expected.keys()),
        )
        if missing:
            issues.append(f"{side}: missing keys {missing}")
        if extra:
            issues.append(f"{side}: unexpected keys {extra}")
        for name in sorted(expected.keys() & mapping.keys()):
            p = mapping[name]
            if tuple(p.shape) != tuple(expected[name]):
                issues.append(
                    f"{side}: parameter shape {name}: {tuple(p.shape)} != {expected[name]}"
                )
            if p.grad is None:
                issues.append(f"{side}: missing gradient {name}")
            elif tuple(p.grad.shape) != tuple(expected[name]):
                issues.append(
                    f"{side}: gradient shape {name}: {tuple(p.grad.shape)} != {expected[name]}"
                )
        maps.append(mapping)
    return issues, maps[0], maps[1]


def calculate_difference(x, y):
    """Same FP64 chunked metric and chunk size as frozen tests/test_dualpipev.py."""
    if x.shape != y.shape:
        raise ValueError("Gradient shapes differ")
    x, y = x.reshape(-1), y.reshape(-1)
    dot, squared_norm = 0.0, 0.0
    for start in range(0, x.numel(), 1 << 20):
        xc = x[start : start + (1 << 20)].double()
        yc = y[start : start + (1 << 20)].double()
        dot += torch.dot(xc, yc).item()
        squared_norm += torch.dot(xc, xc).item() + torch.dot(yc, yc).item()
    if not math.isfinite(dot) or not math.isfinite(squared_norm):
        raise ValueError("Non-finite gradient comparison")
    return 0.0 if squared_norm == 0 else 1 - 2 * dot / squared_norm


def tensor_stats(tensor):
    finite, maximum, nonzero = True, 0.0, False
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), 1 << 20):
        part = flat[start : start + (1 << 20)]
        part_finite = bool(torch.isfinite(part).all())
        finite &= part_finite
        if part_finite:
            maximum = max(maximum, float(part.abs().max()))
        nonzero |= bool(torch.any(part != 0))
    return {"finite": finite, "abs_max": maximum if finite else None, "all_zero": not nonzero}


def compare_pair(name, actual, reference, cp_size, metric=calculate_difference):
    if cp_size not in (1, 2):
        raise ValueError("Unsupported CP")
    if tuple(actual.shape) != tuple(reference.shape):
        raise ValueError("Gradient shapes differ")
    actual = actual.detach() / cp_size if cp_size > 1 else actual.detach()
    reference = reference.detach()
    a, r = tensor_stats(actual), tensor_stats(reference)
    reason = None
    if a["all_zero"] and r["all_zero"]:
        reason = "both_zero"
    elif r["finite"] and r["abs_max"] < 1e-5 and ".experts." not in name:
        reason = "reference_abs_max_below_1e-5_non_expert"
    difference = metric(actual, reference) if a["finite"] and r["finite"] else None
    passed = difference is not None and math.isfinite(difference) and difference <= GRAD_EPS
    return {
        "name": name,
        "canonical": canonical(name),
        "shape": list(actual.shape),
        "cp_divisor": cp_size,
        "actual": a,
        "reference": r,
        "legacy_would_skip": reason,
        "compared": difference is not None,
        "difference": difference,
        "threshold": GRAD_EPS,
        "small_gradient_target": canonical(name) in SMALL_PARAMETERS,
        "status": "PASS" if passed else "FAIL",
    }


def compare_cpu(actual, reference, expected, cp_size=1, metric=calculate_difference):
    """CPU fault-injection entry with exactly the same structural and numeric checks."""
    issues, a, r = inspect_structure(actual, reference, expected)
    rows = []
    if not issues:
        rows = [
            compare_pair(n, a[n].grad.cpu(), r[n].grad.cpu(), cp_size, metric)
            for n in sorted(expected)
        ]
    return {
        "status": "FAIL" if issues or any(x["status"] != "PASS" for x in rows) else "PASS",
        "structural_issues": issues,
        "expected_count": len(expected),
        "checked_count": len(rows),
        "parameters": rows,
    }


def runtime_check(actual_module, reference_module, case, out, metric, source=None):
    """Called only inside the explicitly launched original GPU test, on every rank."""
    dist = torch.distributed
    rank = dist.get_rank()
    expected = expected_parameters(case, rank)
    issues, actual, reference = inspect_structure(
        actual_module.named_parameters(remove_duplicate=False),
        reference_module.named_parameters(remove_duplicate=False),
        expected,
    )
    gathered = [None] * case.world
    # All ranks agree on structural success before any full_tensor collectives.
    dist.all_gather_object(gathered, {"rank": rank, "issues": issues})
    report = {
        "schema": 2,
        "source": source,
        "case": case.record(),
        "rank": rank,
        "expected_count": len(expected),
        "structural_issues": issues,
        "all_rank_structural_issues": gathered,
        "parameters": [],
        "gradient_threshold": GRAD_EPS,
        "loss_rtol": 1e-3,
        "loss_atol": 1e-3,
        "loss_two_steps_passed_before_hook": True,
        "status": "FAIL",
    }
    destination = Path(out) / f"rank-{rank}.json"
    if any(row["issues"] for row in gathered):
        report["checked_count"] = 0
        write_json(destination, report)
        raise AssertionError("Strict all-rank parameter structure failed; see rank reports")
    from torch.distributed.tensor import DTensor

    for name in sorted(expected):
        gradient = actual[name].grad
        if isinstance(gradient, DTensor):
            gradient = gradient.full_tensor()
        row = compare_pair(
            name, gradient.detach().cpu(), reference[name].grad.detach().cpu(), case.cp, metric
        )
        report["parameters"].append(row)
        if row["legacy_would_skip"]:
            print(f"[STRICT] Historically omitted gradient was compared: {row}", flush=True)
        elif row["status"] != "PASS":
            print(f"[STRICT] gradient failure: {row}", flush=True)
    rows = report["parameters"]
    report["checked_count"] = len(rows)
    report["small_gradient_targets_checked"] = sorted(
        r["canonical"] for r in rows if r["small_gradient_target"]
    )
    local_pass = all(r["status"] == "PASS" for r in rows)
    gathered = [None] * case.world
    dist.all_gather_object(gathered, {"rank": rank, "passed": local_pass})
    report["all_rank_numerical_results"] = gathered
    report["status"] = "PASS" if all(r["passed"] for r in gathered) else "FAIL"
    write_json(destination, report)
    if report["status"] != "PASS":
        raise AssertionError("Strict gradient acceptance failed; original thresholds retained")
    worst = max(rows, key=lambda r: r["difference"])
    return worst["difference"], worst["name"]
