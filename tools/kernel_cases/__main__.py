"""
python -m tools.kernel_cases            check that kernel_cases.json is current (exit 1 if stale)
python -m tools.kernel_cases --write    regenerate kernel_cases.json
python -m tools.kernel_cases --summary  case counts per family, model and tier
python -m tools.kernel_cases --export DIR [--family F] [--tier T]
                                        write DIR/<family>.jsonl, one row per case, for an external
                                        harness (TIRx-harness, kernel-design agents)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import SPEC_PATH, dumps, load_spec, select
from .cases import generate


def export(spec: dict, out: Path, families: list[str], tier: str) -> None:
    """
    One JSONL file per family: each row is a case plus the family's kernels, reference and
    tolerance, and the sha256 of the model config it came from, so a consumer can pin it.
    """
    out.mkdir(parents=True, exist_ok=True)
    for family in families:
        meta = spec["families"][family]
        rows = []
        for case in select(family, tier, spec=spec):
            row = dict(case)
            row["kernels"] = meta["kernels"]
            row["reference"] = meta["reference"]
            row["tolerance"] = meta["tolerance"]
            row["config_sha256"] = spec["inputs"]["models"][case["model"]]["sha256"]
            rows.append(json.dumps(row, separators=(",", ":")))
        (out / f"{family}.jsonl").write_text("\n".join(rows) + "\n")
        print(f"{out / family}.jsonl: {len(rows)} cases")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.kernel_cases",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--write", action="store_true", help="regenerate kernel_cases.json")
    parser.add_argument("--summary", action="store_true", help="print case counts")
    parser.add_argument("--export", type=Path, help="write per-family JSONL into this directory")
    parser.add_argument("--family", action="append", help="families to export (default: all)")
    parser.add_argument("--tier", default="spec", choices=["smoke", "spec", "stress"])
    args = parser.parse_args()

    text = dumps(generate())
    if args.write:
        SPEC_PATH.write_text(text)
        print(f"wrote {SPEC_PATH} ({len(json.loads(text)['cases'])} cases)")
    elif not args.summary and args.export is None:
        if not SPEC_PATH.exists() or SPEC_PATH.read_text() != text:
            print(
                f"{SPEC_PATH} is stale: run python -m tools.kernel_cases --write", file=sys.stderr
            )
            return 1
        print(f"{SPEC_PATH} is current")
    spec = json.loads(text)
    if args.summary:
        counts = Counter((c["family"], c["model"], c["tier"]) for c in spec["cases"])
        for family in spec["families"]:
            rows = {k[1:]: v for k, v in counts.items() if k[0] == family}
            if rows:
                total = sum(rows.values())
                detail = ", ".join(f"{m}:{t}={n}" for (m, t), n in sorted(rows.items()))
                print(f"{family}: {total}  ({detail})")
        print(f"total: {len(spec['cases'])}")
    if args.export is not None:
        families = args.family or [f for f in spec["families"] if select(f, args.tier, spec=spec)]
        export(load_spec(), args.export, families, args.tier)
    return 0


if __name__ == "__main__":
    sys.exit(main())
