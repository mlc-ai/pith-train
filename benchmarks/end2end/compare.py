"""
Tabulate end-to-end results, or diff a candidate against a baseline case by case.

    python3 -m benchmarks.end2end.compare <results>                 # one set
    python3 -m benchmarks.end2end.compare <baseline> <candidate>    # candidate vs baseline

Each argument is an output directory of launch.sh (every **/result.json below it) or a collected
JSON file such as benchmarks/end2end/baseline.json. --filter keeps the cases whose path contains
every given substring; --markdown prints a Markdown table.

Speed is reported, not gated: step time is the median of the steps after the warmup, and a few
percent between two runs of identical code is normal. Correctness is flagged: a case that failed
or broke a check (a case the model does not support on a build, status unsupported, is listed
but not flagged, and one that becomes supported is marked "newly supported"), a first-step loss further than --loss-first-tol from the baseline (the first
step precedes any update, so it moves only if the forward changed), or a loss curve further
than --loss-tol from it at any step.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmarks.end2end.common import MAX_STEPS


def load(path: Path) -> dict[str, dict]:
    if path.is_dir():
        results = [json.loads(p.read_text()) for p in sorted(path.rglob("result.json"))]
        return {r["case"]: r for r in results}
    return json.loads(path.read_text())["cases"]


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def problems(r: dict) -> list[str]:
    if r["status"] != "ok":
        return [r["status"]]
    return [k for k, v in r["checks"].items() if v is False]


def table_one(results: dict[str, dict]) -> tuple[list[str], list[list[str]]]:
    head = ["case", "status", "step s", "tok/s/GPU", "peak GB", "loss 1st", "loss last"]
    rows = []
    for case, r in results.items():
        m = r.get("metrics", {})
        status = ",".join(problems(r)) or "ok"
        rows.append([case, status, fmt(m.get("step_time"), ".3f")])
        rows[-1] += [fmt(m.get("tokens_per_second_per_gpu"), ",.0f")]
        rows[-1] += [fmt(m.get("peak_memory_gb"), ".1f"), fmt(m.get("loss_first"), ".4f")]
        rows[-1] += [fmt(m.get("loss_last"), ".4f")]
    return head, rows


def table_two(base: dict, feat: dict, args) -> tuple[list[str], list[list[str]], list[float]]:
    head = ["case", "step s base", "step s feat", "step %", "tok/s/GPU feat", "peak GB d"]
    head += ["loss1 d", "max loss d", "flags"]
    rows, ratios = [], []
    for case in sorted(set(base) | set(feat)):
        b, f = base.get(case), feat.get(case)
        if b is None or f is None:
            rows.append(
                [case] + ["-"] * 7 + ["only in " + ("candidate" if b is None else "baseline")]
            )
            continue
        if b["status"] == "unsupported" or f["status"] == "unsupported":
            note = {
                ("unsupported", "ok"): "newly supported",
                ("ok", "unsupported"): "feat:unsupported",
            }
            note = note.get((b["status"], f["status"]), "unsupported")
            rows.append([case] + ["-"] * 7 + [note if note != "unsupported" else ""])
            if f["status"] == "ok":
                fm = f["metrics"]
                rows[-1][2], rows[-1][4] = (
                    f"{fm['step_time']:.3f}",
                    f"{fm['tokens_per_second_per_gpu']:,.0f}",
                )
            continue
        flags = [f"base:{p}" for p in problems(b)] + [f"feat:{p}" for p in problems(f)]
        if flags:
            rows.append([case] + ["-"] * 7 + [",".join(flags)])
            continue
        bm, fm = b["metrics"], f["metrics"]
        ratio = fm["step_time"] / bm["step_time"]
        ratios.append(ratio)
        d1 = fm["loss_first"] - bm["loss_first"]
        pairs = zip(b["steps"][:MAX_STEPS], f["steps"][:MAX_STEPS])
        dmax = max(abs(x["loss"] - y["loss"]) for x, y in pairs)
        if abs(d1) > args.loss_first_tol:
            flags.append("loss1")
        if dmax > args.loss_tol:
            flags.append("loss")
        rows.append([case, f"{bm['step_time']:.3f}", f"{fm['step_time']:.3f}"])
        rows[-1] += [f"{100 * (ratio - 1):+.1f}", f"{fm['tokens_per_second_per_gpu']:,.0f}"]
        rows[-1] += [f"{fm['peak_memory_gb'] - bm['peak_memory_gb']:+.1f}", f"{d1:+.4f}"]
        rows[-1] += [f"{dmax:.4f}", ",".join(flags)]
    return head, rows, ratios


def render(head: list[str], rows: list[list[str]], markdown: bool) -> str:
    if markdown:
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    return "\n".join(
        "  ".join(str(x).ljust(w) for x, w in zip(row, widths)) for row in [head] + rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--filter", nargs="*", default=[])
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--loss-first-tol", type=float, default=0.01)
    parser.add_argument("--loss-tol", type=float, default=0.1)
    args = parser.parse_args()
    if len(args.results) > 2:
        parser.error("give one result set, or a baseline and a candidate")
    sets = [load(p) for p in args.results]
    sets = [{c: r for c, r in s.items() if all(k in c for k in args.filter)} for s in sets]
    if len(sets) == 1:
        head, rows = table_one(sets[0])
        print(render(head, rows, args.markdown))
        unsupported = sum(row[1] == "unsupported" for row in rows)
        bad = sum(row[1] not in ("ok", "unsupported") for row in rows)
        print(f"\n{len(rows)} cases, {unsupported} unsupported on this build, {bad} not ok")
        return
    head, rows, ratios = table_two(*sets, args)
    print(render(head, rows, args.markdown))
    flagged = sum(bool(row[-1]) for row in rows)
    print(f"\n{len(rows)} cases, {len(ratios)} compared, {flagged} flagged")
    if ratios:
        geomean = math.exp(statistics.fmean(math.log(r) for r in ratios))
        print(
            f"step time, candidate / baseline: geomean {geomean:.4f}, min {min(ratios):.4f}, max {max(ratios):.4f}"
        )


if __name__ == "__main__":
    main()
