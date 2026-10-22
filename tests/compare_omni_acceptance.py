"""Compare complete, full precision repeated-baseline and feature/resume reports."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_run(path):
    """Compare the same global objective, while retaining/checking each raw log.

    Old pretraining logs the CP mean for DP rank 0; the feature logs the global
    DP x CP token-weighted mean. Comparing those raw scalars when DP > 1 is not
    a numerical regression test. Independently sum captured objective losses and
    counts over PP=0 here; no production loss-reduction helper is reused.
    """
    reports = [json.loads(file.read_text()) for file in sorted(path.glob("rank*.json"))]
    assert reports, f"No rank reports under {path}"
    report = next(item for item in reports if item["mesh"]["rank"] == 0)
    mesh = report["mesh"]
    pp, dp, cp = (mesh[key] for key in ("pp_size", "dp_size", "cp_size"))
    assert sorted(item["mesh"]["rank"] for item in reports) == list(range(pp * dp * cp))
    start, steps = report["start"], report["steps"]
    for item in reports:
        m = item["mesh"]
        assert [m[key] for key in ("pp_size", "dp_size", "cp_size")] == [pp, dp, cp]
        assert (m["pp_rank"], m["dp_rank"], m["cp_rank"]) == (
            m["rank"] // (dp * cp),
            m["rank"] // cp % dp,
            m["rank"] % cp,
        ), "Unexpected mesh coordinates"
        assert item["legacy"] == report["legacy"]
        assert item["result"] == "PASSED" and item["start"] == start and item["steps"] == steps
        assert len(item["loss_statistics"]) == (steps - start if m["pp_rank"] == 0 else 0)
    assert [row["train/step"] for row in report["rows"]] == list(range(start, steps))
    output_ranks = [item for item in reports if item["mesh"]["pp_rank"] == 0]
    for index, row in enumerate(report["rows"]):
        stats = [item["loss_statistics"][index] for item in output_ranks]
        assert all(math.isfinite(s["loss_sum"]) and s["loss_sum"] >= 0 for s in stats)
        assert all(type(s["target_count"]) is int and s["target_count"] >= 0 for s in stats)
        count = sum(s["target_count"] for s in stats)
        assert count > 0, "No valid global targets"
        global_mean = math.fsum(s["loss_sum"] for s in stats) / count
        expected_log = global_mean
        if report["legacy"]:
            dp0 = [
                item["loss_statistics"][index]
                for item in output_ranks
                if item["mesh"]["dp_rank"] == 0
            ]
            expected_log = math.fsum(s["loss_sum"] / max(s["target_count"], 1) for s in dp0) / cp
        logged = row["train/cross-entropy-loss"]
        assert math.isclose(logged, expected_log, rel_tol=1e-6, abs_tol=1e-7), (
            f"Logged CE disagrees with independently captured objective in {path}, step {start + index}: "
            f"logged={logged}, expected={expected_log}"
        )
        row["logged/cross-entropy-loss"] = logged
        row["train/cross-entropy-loss"] = global_mean
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base0", type=Path)
    p.add_argument("base1", type=Path)
    p.add_argument("feature", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reports = [load_run(path) for path in (args.base0, args.base1, args.feature)]
    steps = reports[0]["steps"]
    for report in reports:
        assert report["result"] == "PASSED" and report["steps"] == steps and report["start"] == 0
        assert [row["train/step"] for row in report["rows"]] == list(range(steps)), "Incomplete run"
    ranks = sorted(args.base0.glob("rank*.json"))
    assert [p.name for p in ranks] == sorted(p.name for p in args.base1.glob("rank*.json"))
    assert [p.name for p in ranks] == sorted(p.name for p in args.feature.glob("rank*.json"))
    for rank in ranks:
        arms = [
            json.loads((path / rank.name).read_text())
            for path in (args.base0, args.base1, args.feature)
        ]
        for arm in arms:
            assert arm["result"] == "PASSED" and arm["steps"] == steps and arm["start"] == 0
        initial = [arm["initial_state"] for arm in arms]
        assert initial[0] == initial[1] == initial[2], f"Unmatched initial weights on {rank.name}"
        hooks = [arm["audited_view_hooks"] for arm in arms]
        assert hooks[0] == hooks[1] == hooks[2] and hooks[0] > 0, (
            f"Unmatched or missing decoder view audits on {rank.name}: {hooks}"
        )
        batches = [arm["batches"] for arm in arms]
        assert batches[0] == batches[1] == batches[2], f"Unmatched inputs on {rank.name}"
    metrics = {}
    passed = True
    for key in ("train/cross-entropy-loss", "train/load-balance-loss", "train/gradient-norm"):
        values = [np.array([row[key] for row in report["rows"]]) for report in reports]
        assert all(np.isfinite(value).all() for value in values)
        floor = float(np.abs(values[0][1:] - values[1][1:]).mean())
        delta = float(np.abs(values[0][1:] - values[2][1:]).mean())
        ratio = delta / floor if floor else (0.0 if delta == 0 else None)
        ok = ratio is not None and ratio < 3
        metrics[key] = dict(
            baseline_mean_abs_delta=floor,
            feature_mean_abs_delta=delta,
            ratio=ratio,
            passed=ok,
            first_step_delta=float(abs(values[0][0] - values[2][0])),
        )
        passed &= ok
    if args.resume:
        resumed = load_run(args.resume)
        start = resumed["start"]
        assert resumed["exact_restore"] and resumed["steps"] == steps
        assert [row["train/step"] for row in resumed["rows"]] == list(range(start, steps))
        for key in ("train/cross-entropy-loss", "train/load-balance-loss", "train/gradient-norm"):
            baseline = [
                np.array([row[key] for row in report["rows"]][start:]) for report in reports[:2]
            ]
            feature = np.array([row[key] for row in reports[2]["rows"]][start:])
            restored = np.array([row[key] for row in resumed["rows"]])
            floor = float(np.abs(baseline[0] - baseline[1]).mean())
            delta = float(np.abs(feature - restored).mean())
            ratio = delta / floor if floor else (0.0 if delta == 0 else None)
            ok = ratio is not None and ratio < 3
            metrics[f"resume/{key}"] = dict(
                baseline_mean_abs_delta=floor, resume_mean_abs_delta=delta, ratio=ratio, passed=ok
            )
            passed &= ok
    output = dict(
        result="PASSED" if passed else "INVESTIGATE",
        steps=steps,
        metrics=metrics,
        ce_definition="Independent sum of objective losses / valid targets across PP0 DP x CP",
        raw_logged_ce={
            name: [row["logged/cross-entropy-loss"] for row in report["rows"]]
            for name, report in zip(("base0", "base1", "feature"), reports)
        },
    )
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    assert passed, (
        "Numerical deltas exceed the repeated-baseline envelope; diagnose before changing gates"
    )


if __name__ == "__main__":
    main()
