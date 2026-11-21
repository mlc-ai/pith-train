"""
The regression gate of benchmarks/kernels/bench_spec.py, without torch (CPU-only tests import it):

    python -m benchmarks.kernels.compare results.json benchmarks/kernels/baselines/h200.json

An op regresses when its median exceeds the baseline's by more than 5% plus 3 us
(docs/kernel-styles.md section 9). An op the baseline has, of a case the new run ran, that the new
run did not time (an error, an OOM, a dropped op) also fails, so a broken kernel cannot pass by
vanishing. Cases the new run did not select (--family / --model / --tier subsets) are not compared.
Exits 1 on any regression or missing op, or when the two runs come from different GPUs.
"""

import argparse
import json
import sys
from pathlib import Path

REL = 0.05
ABS_US = 3.0


def compare(new: dict, base: dict, rel: float = REL, abs_us: float = ABS_US) -> list[str]:
    """
    One line per regressed or missing op of new against base.
    """
    timed = {(r["case"], r["op"]): r for r in new["results"] if "median_us" in r}
    ran = {r["case"] for r in new["results"]}
    problems = []
    for prev in base["results"]:
        if "median_us" not in prev or prev["case"] not in ran:
            continue
        key = (prev["case"], prev["op"])
        if key not in timed:
            problems.append(f"{key[0]} {key[1]}: missing (baseline {prev['median_us']:.1f} us)")
            continue
        limit = prev["median_us"] * (1 + rel) + abs_us
        if timed[key]["median_us"] > limit:
            problems.append(
                f"{key[0]} {key[1]}: {timed[key]['median_us']:.1f} us > {limit:.1f} us (baseline {prev['median_us']:.1f})"
            )
    return problems


def gate(new: dict, base: dict) -> int:
    """
    Print the comparison and return the exit status.
    """
    if base["meta"]["gpu"] != new["meta"]["gpu"]:
        print(f"baseline is from {base['meta']['gpu']}, this run from {new['meta']['gpu']}")
        return 1
    problems = compare(new, base)
    for line in problems:
        print("REGRESSION", line)
    matched = sum(1 for r in new["results"] if "median_us" in r)
    print(f"{len(problems)} regressions over {matched} ops")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    return gate(json.loads(args.results.read_text()), json.loads(args.baseline.read_text()))


if __name__ == "__main__":
    sys.exit(main())
