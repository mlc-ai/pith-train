"""
The kernel benchmark gate (benchmarks/kernels/compare.py) against the stored H200 baseline, in the
smoke tier (no GPU work): an unchanged run passes; inflating a few ops past 5% + 3 us flags exactly
those and exits 1, through the function and through the command line; noise within the bound, ops of
cases a run did not select, and timings below the absolute slack pass; every op slower within its
own slack but the work-weighted total past 5% fails; an op that errored or vanished fails.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.kernels.compare import compare, gate, weighted_ratio

pytestmark = pytest.mark.smoke
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "benchmarks" / "kernels" / "baselines" / "h200.json"
BASELINE = json.loads(BASELINE_PATH.read_text())


def timed(run: dict) -> list[dict]:
    return [r for r in run["results"] if "median_us" in r]


def inflate(run: dict, picks: list[dict], factor: float) -> dict:
    out = copy.deepcopy(run)
    keys = {(r["case"], r["op"]) for r in picks}
    for r in out["results"]:
        if (r["case"], r["op"]) in keys:
            r["median_us"] *= factor
    return out


def slow_ops(count: int = 3) -> list[dict]:
    """
    count ops from different families, long enough that 1.3x is far past 5% + 3 us.
    """
    picks, families = [], set()
    for r in timed(BASELINE):
        family = r["case"].split("/")[0]
        if r["median_us"] > 100 and family not in families:
            picks.append(r)
            families.add(family)
        if len(picks) == count:
            return picks
    raise AssertionError("baseline has too few long ops")


def flagged(problems: list[str]) -> set[str]:
    return {line.split(":")[0] for line in problems}


def test_baseline_is_well_formed() -> None:
    assert BASELINE["meta"]["gpu"] == "NVIDIA H200"
    assert len(timed(BASELINE)) >= 100
    assert not [r for r in BASELINE["results"] if "median_us" not in r]


def test_unchanged_run_passes() -> None:
    assert compare(BASELINE, BASELINE) == []
    assert gate(copy.deepcopy(BASELINE), BASELINE) == 0


def test_inflated_ops_are_flagged_exactly() -> None:
    picks = slow_ops()
    new = inflate(BASELINE, picks, 1.3)
    problems = compare(new, BASELINE)
    assert flagged(problems) == {f"{r['case']} {r['op']}" for r in picks}
    assert gate(new, BASELINE) == 1


def test_command_line_exits_nonzero(tmp_path: Path) -> None:
    picks = slow_ops()
    results = tmp_path / "results.json"
    results.write_text(json.dumps(inflate(BASELINE, picks, 1.3)))
    cmd = [sys.executable, "-m", "benchmarks.kernels.compare", str(results), str(BASELINE_PATH)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.startswith("REGRESSION")]
    assert len(lines) == len(picks)
    for r in picks:
        assert any(f"{r['case']} {r['op']}:" in line for line in lines)
    results.write_text(json.dumps(BASELINE))
    assert subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True).returncode == 0


def test_noise_within_bound_passes() -> None:
    new = inflate(BASELINE, timed(BASELINE), 1.04)  # every op 4% slower: inside 5% + 3 us
    assert compare(new, BASELINE) == []


def test_absolute_slack_for_short_ops() -> None:
    base = copy.deepcopy(BASELINE)
    op = timed(base)[0]
    op["median_us"] = 10.0  # a 10 us kernel: 5% is 0.5 us, the 3 us slack dominates
    new = copy.deepcopy(base)
    new_op = next(r for r in new["results"] if (r["case"], r["op"]) == (op["case"], op["op"]))
    new_op["median_us"] = 13.4  # 34% slower, within 10 * 1.05 + 3 = 13.5
    assert compare(new, base) == []
    new_op["median_us"] = 13.6
    assert flagged(compare(new, base)) == {f"{op['case']} {op['op']}"}


def test_errored_or_missing_op_fails() -> None:
    victim = slow_ops(1)[0]
    errored = copy.deepcopy(BASELINE)
    errored["results"] = [
        r for r in errored["results"] if (r["case"], r["op"]) != (victim["case"], victim["op"])
    ]
    errored["results"].append(dict(case=victim["case"], op="*", error="OOM"))
    assert flagged(compare(errored, BASELINE)) == {f"{victim['case']} {victim['op']}"}


def test_unselected_cases_are_not_compared() -> None:
    family = timed(BASELINE)[0]["case"].split("/")[0]
    subset = copy.deepcopy(BASELINE)
    subset["results"] = [r for r in subset["results"] if r["case"].startswith(family + "/")]
    assert subset["results"] and compare(subset, BASELINE) == []


def test_other_gpu_fails() -> None:
    other = copy.deepcopy(BASELINE)
    other["meta"]["gpu"] = "NVIDIA B200"
    assert gate(other, BASELINE) == 1


def test_work_weighted_total() -> None:
    base = copy.deepcopy(BASELINE)
    for r in timed(base):
        r.setdefault("work", 1)
        r.setdefault("layers", 1)
    assert weighted_ratio(base, base) == 1.0
    slower = inflate(base, timed(base), 1.04)
    assert compare(slower, base) == [] and gate(slower, base) == 0
    short = [r for r in timed(base) if r["median_us"] * 0.06 < 3.0]  # 6% slower is inside 5% + 3 us
    slower = inflate(base, short, 1.06)
    for r in timed(base):
        r["work"] = 1 if r in short else 0
    assert compare(slower, base) == []
    assert abs(weighted_ratio(slower, base) - 1.06) < 1e-9
    assert gate(slower, base) == 1
