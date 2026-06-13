"""Diff two PithTrain training logs: gate on correctness, report on performance."""

import argparse
import re
import sys
from statistics import median, stdev

CORRECTNESS_METRICS = ["cross-entropy-loss", "load-balance-loss"]
PERF_METRICS = ["step-time", "peak-gpu-memory"]
STEP_PATTERN = re.compile(r"step\s+(\d+)/(\d+)")


def parse_log(path):
    steps = []
    with open(path) as f:
        for line in f:
            if "| INFO | step " not in line:
                continue
            metrics = {}
            for part in (p.strip() for p in line.split("|")):
                m = STEP_PATTERN.match(part)
                if m:
                    metrics["step"] = int(m.group(1))
                    continue
                tokens = part.split()
                for i in range(len(tokens) - 1, 0, -1):
                    try:
                        val = float(tokens[i].replace(",", ""))
                    except ValueError:
                        continue
                    metrics[" ".join(tokens[:i])] = val
                    break
            if "step" in metrics:
                steps.append(metrics)
    return steps


def correctness_failures(base_steps, feature_steps, metric, tolerance):
    failures = []
    for base, feature in zip(base_steps, feature_steps):
        step = base["step"]
        bv, fv = base.get(metric), feature.get(metric)
        if bv is None and fv is None:
            continue
        if bv is None or fv is None:
            failures.append(f"  step {step:03d}: {metric} one-sided (base={bv}, feature={fv})")
            continue
        if bv == 0 and fv == 0:
            continue
        denom = abs(bv) if bv != 0 else abs(fv)
        rel_diff = abs(bv - fv) / denom
        if rel_diff > tolerance:
            failures.append(
                f"  step {step:03d}: {metric} diverged — base={bv:.6f}, feature={fv:.6f}, "
                f"rel_diff={rel_diff:.2e} > tolerance={tolerance:.0e}"
            )
    return failures


def print_correctness_table(base_steps, feature_steps):
    print("Correctness (per-step):")
    print("-" * 100)
    header = f"{'step':>5}"
    for metric in CORRECTNESS_METRICS:
        header += f" | {'base ' + metric:>28} {'feature':>12} {'rel_diff':>10}"
    print(header)
    print("-" * 100)
    for base, feature in zip(base_steps, feature_steps):
        row = f"{base['step']:5d}"
        for metric in CORRECTNESS_METRICS:
            bv, fv = base.get(metric), feature.get(metric)
            if bv is not None and fv is not None:
                denom = abs(bv) if bv != 0 else (abs(fv) if fv != 0 else 1.0)
                rd = abs(bv - fv) / denom
                row += f" | {bv:28.6f} {fv:12.6f} {rd:10.2e}"
            elif bv is not None:
                row += f" | {bv:28.6f} {'N/A':>12} {'N/A':>10}"
            elif fv is not None:
                row += f" | {'N/A':>28} {fv:12.6f} {'N/A':>10}"
            else:
                row += f" | {'N/A':>28} {'N/A':>12} {'N/A':>10}"
        print(row)
    print("-" * 100)
    print()


def print_perf_summary(base_steps, feature_steps):
    print("Performance (median ± std over last 10 steps):")
    print("-" * 104)
    print(f"{'metric':>22} | {'base':>26} {'feature':>26} {'delta':>12} {'delta %':>9}")
    print("-" * 104)
    for metric in PERF_METRICS:
        bvs = [s[metric] for s in base_steps[-10:] if metric in s]
        fvs = [s[metric] for s in feature_steps[-10:] if metric in s]
        if not bvs or not fvs:
            print(f"{metric:>22} | {'N/A':>26} {'N/A':>26} {'N/A':>12} {'N/A':>9}")
            continue
        bm, fm = median(bvs), median(fvs)
        bs = stdev(bvs) if len(bvs) > 1 else 0.0
        fs = stdev(fvs) if len(fvs) > 1 else 0.0
        delta = fm - bm
        pct = (delta / bm * 100) if bm != 0 else 0.0
        base_str = f"{bm:.4f} ± {bs:.4f}"
        feat_str = f"{fm:.4f} ± {fs:.4f}"
        print(f"{metric:>22} | {base_str:>26} {feat_str:>26} {delta:+12.4f} {pct:+8.2f}%")
    print("-" * 104)
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare two PithTrain training logs.")
    parser.add_argument("base_log")
    parser.add_argument("feature_log")
    parser.add_argument("--tolerance", type=float, default=5e-3)
    args = parser.parse_args()

    base_steps = parse_log(args.base_log)
    feature_steps = parse_log(args.feature_log)
    print(f"Base log:    {args.base_log} ({len(base_steps)} steps)")
    print(f"Feature log: {args.feature_log} ({len(feature_steps)} steps)")
    print(f"Tolerance:   {args.tolerance:.0e}\n")

    if not base_steps:
        print("FAIL: No training steps found in base log.")
        sys.exit(1)
    if not feature_steps:
        print("FAIL: No training steps found in feature log.")
        sys.exit(1)
    if len(base_steps) != len(feature_steps):
        n = min(len(base_steps), len(feature_steps))
        print(f"WARNING: step counts differ; comparing first {n} steps.\n")

    all_failures = {}
    for metric in CORRECTNESS_METRICS:
        failures = correctness_failures(base_steps, feature_steps, metric, args.tolerance)
        if failures:
            all_failures[metric] = failures

    print_correctness_table(base_steps, feature_steps)
    print_perf_summary(base_steps, feature_steps)

    if not all_failures:
        print("PASS: all correctness metrics within tolerance across all steps.")
        sys.exit(0)
    print("FAIL: correctness metrics diverged beyond tolerance:")
    for metric, failures in all_failures.items():
        print(f"\n  {metric}:")
        for f in failures:
            print(f)
    sys.exit(1)


if __name__ == "__main__":
    main()
