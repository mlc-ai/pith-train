"""
Compare three PithTrain training logs: base0, base1 and feat0.

base0 and base1 are the same code, so their difference is the run-to-run floor. base0-vs-feat0
is judged as a ratio against it.

Usage:  python3 compare.py <base0.log> <base1.log> <feat0.log>
Output: one row per gated metric with envelope, signal, ratio and verdict, the step-1 forward and
        the global gradient norm (both reported, not gated), then PASS or FAIL
Exit:   0 PASS, 1 FAIL
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

METRICS = ["cross-entropy-loss", "load-balance-loss"]
# Reduced over every rank, so it stays comparable when the arms lay data out differently.
REPORTED = ["gradient-norm"]
RATIO_WARN, RATIO_FAIL = 3.0, 5.0


def parse(path):
    text = Path(path).read_text()
    return {m: [float(v) for v in re.findall(rf"{m} ([0-9.]+)", text)] for m in METRICS + REPORTED}


def mean_delta(a, b):
    return statistics.mean(abs(x - y) for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base0_log")
    parser.add_argument("base1_log")
    parser.add_argument("feat0_log")
    args = parser.parse_args()

    runs = [parse(p) for p in (args.base0_log, args.base1_log, args.feat0_log)]
    steps = min(len(r["cross-entropy-loss"]) for r in runs)
    if steps == 0:
        sys.exit("no training steps found in one of the logs")
    for run in runs:  # one window for every metric, so a short arm cannot skew the ratio
        for metric in METRICS + REPORTED:
            run[metric] = run[metric][:steps]
    base0, base1, feat0 = runs
    print(f"{steps} steps compared\n")

    failed = False
    for metric in METRICS:
        envelope = mean_delta(base0[metric], base1[metric])
        signal = mean_delta(base0[metric], feat0[metric])
        # Identical bases are the cleanest floor there is, so a zero envelope is not a failure.
        ratio = signal / envelope if envelope else (0.0 if signal == 0 else float("inf"))
        verdict = "PASS" if ratio < RATIO_WARN else "INVESTIGATE" if ratio < RATIO_FAIL else "FAIL"
        failed |= ratio >= RATIO_FAIL
        print(
            f"{metric:22} envelope {envelope:.4f}  signal {signal:.4f}  ratio {ratio:5.2f}  {verdict}"
        )

    # Step 1 precedes any optimizer update, so its floor is normally zero and the signal is
    # exactly the numerical difference the change makes to the forward. Reported, not gated.
    loss = "cross-entropy-loss"
    print(
        f"{'step-1 forward':22} envelope {abs(base0[loss][0] - base1[loss][0]):.4f}  "
        f"signal {abs(base0[loss][0] - feat0[loss][0]):.4f}"
    )

    # The logged losses are rank 0's own samples (see SKILL.md); the gradient norm is global, so
    # it is the comparison to read when the arms assign samples to ranks differently.
    for metric in REPORTED:
        envelope = mean_delta(base0[metric], base1[metric])
        signal = mean_delta(base0[metric], feat0[metric])
        ratio = signal / envelope if envelope else (0.0 if signal == 0 else float("inf"))
        print(
            f"{metric:22} envelope {envelope:.4f}  signal {signal:.4f}  ratio {ratio:5.2f}  (reported)"
        )

    print("\nFAIL" if failed else "\nPASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
