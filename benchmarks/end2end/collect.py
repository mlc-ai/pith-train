"""
Collect the result.json files of one launch.sh output directory into a single JSON file, the form
in which a baseline is committed (benchmarks/end2end/baseline.json) and compare.py reads.

    python3 -m benchmarks.end2end.collect <output-dir> -o benchmarks/end2end/baseline.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(p.read_text()) for p in sorted(args.results.rglob("result.json"))]
    cases = {r["case"]: r for r in results}
    for r in cases.values():
        r.pop("error", None) if r["status"] == "ok" else None
    commits = Counter((r["commit"], r["dirty"]) for r in cases.values())
    packages = Counter(r["environment"]["torch"] for r in cases.values())
    header = dict(commits=[dict(commit=c, dirty=d, cases=n) for (c, d), n in commits.items()])
    header.update(torch=dict(packages), statuses=dict(Counter(r["status"] for r in cases.values())))
    header["dates"] = [
        min(r["date"] for r in cases.values()),
        max(r["date"] for r in cases.values()),
    ]
    args.output.write_text(json.dumps(dict(header=header, cases=cases), indent=1) + "\n")
    print(f"{len(cases)} cases -> {args.output}")


if __name__ == "__main__":
    main()
