"""
Compare two PithTrain benchmark logs and report the throughput difference.

Both runs use benchmark mode (synthetic tokens, replayed routing), so the loss is meaningless and
only the timing and memory rows carry information. Step 1 is dropped: it carries the JIT warmup and the
torch.compile trace, which dwarf the steady-state step time.

Usage:  python3 compare.py <base.log> <feat.log>
Output: median step-time, tokens-per-second and peak-gpu-memory per arm, plus the delta
Exit:   0. This script measures; it does not gate.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

METRICS = ["step-time", "tokens-per-second", "peak-gpu-memory"]
UNITS = {"step-time": "sec", "peak-gpu-memory": "GB"}


def parse(path):
    text = Path(path).read_text()
    return {
        m: [float(v.replace(",", "")) for v in re.findall(rf"{m} ([0-9.,]+)", text)][1:]
        for m in METRICS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_log")
    parser.add_argument("feat_log")
    args = parser.parse_args()

    base, feat = parse(args.base_log), parse(args.feat_log)
    steps = min(len(base["step-time"]), len(feat["step-time"]))
    if steps == 0:
        sys.exit("no steps past the first found in one of the logs")
    print(f"{steps} steps compared, excluding step 1\n")

    for metric in METRICS:
        b, f = statistics.median(base[metric][:steps]), statistics.median(feat[metric][:steps])
        print(
            f"{metric:22} base {b:12,.3f}  feat {f:12,.3f}  {(f - b) / b * 100:+6.2f}% {UNITS.get(metric, '')}"
        )


if __name__ == "__main__":
    main()
