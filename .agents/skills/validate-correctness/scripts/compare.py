"""
Compare three PithTrain training logs: base0, base1 and feat0.

base0 and base1 are the same code, so their difference is the run-to-run floor. base0-vs-feat0
is judged as a ratio against it.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

CORRECTNESS = ["cross-entropy-loss", "load-balance-loss"]
PERFORMANCE = ["step-time", "peak-gpu-memory"]
RATIO_WARN, RATIO_FAIL = 3.0, 5.0


def parse(path):
    text = Path(path).read_text()
    return {
        m: [float(v) for v in re.findall(rf"{m} ([0-9.]+)", text)]
        for m in CORRECTNESS + PERFORMANCE
    }


def mean_delta(a, b):
    return statistics.mean(abs(x - y) for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base0_log")
    parser.add_argument("base1_log")
    parser.add_argument("feat0_log")
    args = parser.parse_args()

    base0, base1, feat0 = (parse(p) for p in (args.base0_log, args.base1_log, args.feat0_log))
    steps = min(len(r["cross-entropy-loss"]) for r in (base0, base1, feat0))
    if steps == 0:
        sys.exit("FAIL: no training steps found")
    print(f"{steps} steps compared\n")

    failed = False
    for metric in CORRECTNESS:
        envelope = mean_delta(base0[metric], base1[metric])
        signal = mean_delta(base0[metric], feat0[metric])
        ratio = signal / envelope if envelope else float("inf")
        verdict = "PASS" if ratio < RATIO_WARN else "INVESTIGATE" if ratio < RATIO_FAIL else "FAIL"
        failed |= ratio >= RATIO_FAIL
        print(
            f"{metric:22} envelope {envelope:.4f}  signal {signal:.4f}  ratio {ratio:5.2f}  {verdict}"
        )

    for metric in PERFORMANCE:
        b, f = statistics.median(base0[metric]), statistics.median(feat0[metric])
        print(f"{metric:22} base0 {b:10.3f}  feat0 {f:10.3f}  {(f - b) / b * 100:+6.2f}%")

    # Step 1 precedes any optimizer update, so its floor is normally zero and the signal is
    # exactly the numerical difference the change makes to the forward. Reported, not gated:
    # a reordered reduction or a swapped kernel moves it without being a regression.
    loss = "cross-entropy-loss"
    step1_envelope = abs(base0[loss][0] - base1[loss][0])
    step1_signal = abs(base0[loss][0] - feat0[loss][0])
    print(f"{'step-1 forward':22} envelope {step1_envelope:.4f}  signal {step1_signal:.4f}")

    print("\nFAIL" if failed else "\nPASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
