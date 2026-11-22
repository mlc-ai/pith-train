"""
The speedup of a kernel change: the same cases of tests/kernels/cases.py timed with the pithtrain
of a base git ref and with the working tree's, on one GPU, interleaved.

    python -m benchmarks.kernels.ab --base upstream/main --family cross_entropy --out ab.json
    python -m benchmarks.kernels.ab --base upstream/main --family moe_experts --op 'quant|silu|swiglu'
    python -m benchmarks.kernels.ab --base-dir ../old-checkout --tier stress --family weight_quant

--base checks the ref out into a worktree (workspace/ab/<sha>, kept for reuse); --base-dir uses a
directory that already holds one. Two worker processes run on the current GPU, one importing
pithtrain from the base tree and one from this tree; everything else (the cases, the op builders of
benchmarks/kernels/bench_spec.py, the libraries in this environment) is this tree's, so the two
differ only in pithtrain. For every case each op is timed ROUNDS times on each side, alternating
which side goes first (base, head, head, base, ...), and the medians of the rounds are compared:
speedup = base / head (above 1: the working tree is faster). An op that fails on one side (missing,
rejected, OOM) is reported with its error.

The summary gives per op name the work-weighted speedup sum(w) / sum(w / speedup), w = work x layers
(FLOPs, or elements for memory-bound ops, per step; bench_spec.py), which is the ratio of the total
time the cases take, weighted by how often a step runs them; and the same over every op.

--base-dir . times the working tree against itself: the noise floor. On the H200 single ops of
about 1 GB (gpt-oss-120b's expert weights, its EP 1 activations) then differ by up to 10% either way
between the two processes, from where their buffers are placed, so judge a change by the weighted
speedup and by ops that move together, not by one such op.

The gate (exit 1): an op that runs at the base but fails in the working tree, or an op name whose
work-weighted speedup is below 1 / (1 + TOLERANCE), i.e. the working tree more than 5% slower on
that op summed over its cases. Kernel PRs post this run against main.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 0.05


def worker(tree: str) -> None:
    """
    Serve timing requests on stdin / stdout: one JSON line {"case": id, "op": regex or null} in,
    one JSON line {op: timing or {"error": ...}} out. pithtrain comes from tree.
    """
    sys.path.insert(0, tree)
    import pithtrain

    assert Path(pithtrain.__file__).resolve().is_relative_to(Path(tree).resolve()), pithtrain.__file__  # fmt: skip
    sys.path.remove(tree)
    import re

    import torch

    from benchmarks.kernels.bench_spec import Timer, build
    from tests.kernels.cases import all_cases

    torch.backends.cuda.matmul.allow_tf32 = False
    by_id = {case["id"]: case for case in all_cases()}
    timer = None
    print(json.dumps(dict(ready=pithtrain.__file__)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if timer is None:
            timer = Timer(request["warmup"], request["reps"])
        case, pattern = by_id[request["case"]], request["op"] and re.compile(request["op"])
        reply, ops = {}, {}
        try:
            ops = build(case)
            for op, entry in ops.items():
                if pattern and not pattern.search(op):
                    continue
                try:
                    reply[op] = dict(timer(entry.fn), work=entry.work, memory_bound=entry.memory_bound)  # fmt: skip
                except Exception as error:
                    reply[op] = dict(error=f"{type(error).__name__}: {error}"[:300])
        except Exception as error:
            reply["*"] = dict(error=f"{type(error).__name__}: {error}"[:300])
        finally:
            ops = None
            torch.cuda.empty_cache()
        print(json.dumps(reply), flush=True)


class Worker:
    def __init__(self, tree: Path):
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
        cmd = [sys.executable, "-m", "benchmarks.kernels.ab", "--worker", str(tree)]
        self.proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)  # fmt: skip
        self.pithtrain = self.read()["ready"]

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited with {self.proc.wait()}")
        return json.loads(line)

    def time(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        return self.read()

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def base_tree(ref: str) -> Path:
    sha = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    tree = REPO_ROOT / "workspace" / "ab" / sha[:12]
    if not (tree / "pithtrain").is_dir():
        subprocess.run(["git", "worktree", "add", "--detach", str(tree), sha], cwd=REPO_ROOT, check=True)  # fmt: skip
    return tree


def summarize(rows: list[dict]) -> dict:
    """
    Per op name and overall: the work-weighted speedup and how many ops it covers.
    """
    groups = defaultdict(list)
    for row in rows:
        if "speedup" in row:
            groups[row["op"]].append(row)
            groups["*"].append(row)
    out = {}
    for name, members in groups.items():
        w = [r["work"] * r["layers"] for r in members]
        weighted = sum(w) / sum(wi / r["speedup"] for wi, r in zip(w, members)) if sum(w) else None
        out[name] = dict(ops=len(members), weighted_speedup=weighted, min=min(r["speedup"] for r in members), max=max(r["speedup"] for r in members))  # fmt: skip
    return out


def regressions(rows: list[dict], summary: dict) -> list[str]:
    """
    Ops that run at the base but fail in the working tree, and op names more than TOLERANCE slower
    work-weighted.
    """
    problems = [f"{r['case']} {r['op']}: {r['head_error']}" for r in rows if "head_error" in r and "base_us" in r]  # fmt: skip
    for name, s in sorted(summary.items()):
        if name != "*" and s["weighted_speedup"] and s["weighted_speedup"] < 1 / (1 + TOLERANCE):
            problems.append(f"{name}: work-weighted speedup x{s['weighted_speedup']:.3f}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--base", help="git ref to compare against")
    parser.add_argument("--base-dir", type=Path, help="a checkout of the base instead of --base")
    parser.add_argument("--family", action="append")
    parser.add_argument("--model")
    parser.add_argument("--tier", default="spec")
    parser.add_argument("--op", help="time only the ops whose name matches this regex")
    parser.add_argument("--shard", help="I/N: every N-th case starting at I")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    from benchmarks.kernels.bench_spec import FAMILIES, cases_of

    if (args.base is None) == (args.base_dir is None):
        parser.error("give one of --base and --base-dir")
    base = Worker(args.base_dir.resolve() if args.base_dir else base_tree(args.base))
    head = Worker(REPO_ROOT)
    rows = []
    for case in cases_of(args.family or FAMILIES, args.tier, args.model, args.shard):
        request = dict(case=case["id"], op=args.op, warmup=args.warmup, reps=args.reps)
        samples = dict(base=defaultdict(list), head=defaultdict(list))
        replies = dict(base={}, head={})
        for r in range(args.rounds):
            order = [("base", base), ("head", head)] if r % 2 == 0 else [("head", head), ("base", base)]  # fmt: skip
            for side, proc in order:
                for op, timing in proc.time(request).items():
                    replies[side][op] = timing
                    if "median_us" in timing:
                        samples[side][op].append(timing["median_us"])
        layers = case.get("weight", {}).get("layers", 1)
        for op in dict.fromkeys([*replies["base"], *replies["head"]]):
            row = dict(case=case["id"], op=op, layers=layers)
            for side in ("base", "head"):
                timing = replies[side].get(op, dict(error="not timed"))
                if samples[side][op]:
                    row[f"{side}_us"] = statistics.median(samples[side][op])
                    row["work"] = timing["work"]
                else:
                    row[f"{side}_error"] = timing.get("error", "not timed")
            if "base_us" in row and "head_us" in row:
                row["speedup"] = row["base_us"] / row["head_us"]
            rows.append(row)
            shown = f"{row['base_us']:10.1f} {row['head_us']:10.1f} us  x{row['speedup']:.3f}" if "speedup" in row else f"base: {row.get('base_error', 'ok')} | head: {row.get('head_error', 'ok')}"  # fmt: skip
            print(f"{case['id']:<52} {op:<36} {shown}", flush=True)
    base.close()
    head.close()
    summary = summarize(rows)
    for name, s in sorted(summary.items()):
        weighted = f"{s['weighted_speedup']:.3f}" if s["weighted_speedup"] else "-"
        print(f"{name:<40} {s['ops']:5d} ops  work-weighted speedup x{weighted}  (min {s['min']:.3f}, max {s['max']:.3f})")  # fmt: skip
    problems = regressions(rows, summary)
    for line in problems:
        print("REGRESSION", line)
    print(f"{len(problems)} regressions")
    if args.out:
        meta = dict(base=args.base or str(args.base_dir), base_pithtrain=base.pithtrain, head_pithtrain=head.pithtrain, rounds=args.rounds, tier=args.tier)  # fmt: skip
        args.out.write_text(json.dumps(dict(meta=meta, summary=summary, results=rows, regressions=problems), indent=1) + "\n")  # fmt: skip
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
