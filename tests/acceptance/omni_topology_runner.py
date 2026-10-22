"""Bounded sequential topology runner for an already allocated H100 host.

No Modal API, scheduler submission, new credentials, or allocation management.
Example:
  python -m tests.acceptance.omni_topology_runner --expected-commit "$COMMIT" \\
    --case dense-single --out /work/evidence/topology-single
Use --case all only on an existing dedicated eight-H100 allocation.
"""

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from tests.acceptance.omni_topology_contract import (
    BY_NAME,
    CASES,
    GRAD_EPS,
    SMALL_PARAMETERS,
    canonical,
    expected_parameters,
    sha256,
    verify_source,
    write_json,
)


def validate_reports(case, reports, log, exit_code, source):
    problems = []
    if exit_code != 0:
        problems.append(f"process exit {exit_code}")
    if set(reports) != set(range(case.world)):
        problems.append("missing or unexpected rank report")
    for step in (0, 1):
        if f"[INFO] Loss matches the reference at step {step}." not in log:
            problems.append(f"missing original loss assertion completion step {step}")
    small = set()
    for rank in range(case.world):
        if rank not in reports:
            continue
        report = reports[rank]
        if report.get("source") != source:
            problems.append(f"rank {rank}: missing or different commit provenance")
        expected = expected_parameters(case, rank)
        rows = report.get("parameters", [])
        if report.get("rank") != rank or report.get("case") != case.record():
            problems.append(f"rank {rank}: case/rank mismatch")
        if report.get("status") != "PASS" or report.get("structural_issues"):
            problems.append(f"rank {rank}: failed structure/numerics")
        if report.get("checked_count") != len(expected) or report.get("expected_count") != len(
            expected
        ):
            problems.append(f"rank {rank}: bad count")
        names = [r.get("name") for r in rows]
        if len(names) != len(expected) or set(names) != set(expected):
            problems.append(f"rank {rank}: incomplete or duplicated keys")
        if (
            report.get("gradient_threshold") != GRAD_EPS
            or report.get("loss_rtol") != 1e-3
            or report.get("loss_atol") != 1e-3
        ):
            problems.append(f"rank {rank}: changed thresholds")
        if report.get("loss_two_steps_passed_before_hook") is not True:
            problems.append(f"rank {rank}: missing loss evidence")
        for row in rows:
            name = row.get("name")
            difference = row.get("difference")
            valid = (
                name in expected
                and row.get("shape") == list(expected[name])
                and row.get("canonical") == canonical(name)
                and row.get("cp_divisor") == case.cp
                and row.get("status") == "PASS"
                and row.get("compared") is True
                and row.get("actual", {}).get("finite") is True
                and row.get("reference", {}).get("finite") is True
                and row.get("threshold") == GRAD_EPS
                and isinstance(difference, (int, float))
                and math.isfinite(difference)
                and difference <= GRAD_EPS
            )
            if not valid:
                problems.append(f"rank {rank}: invalid/uncompared gradient {name}")
            elif canonical(name) in SMALL_PARAMETERS:
                small.add(canonical(name))
    if small != SMALL_PARAMETERS:
        problems.append("incomplete eight-canonical-parameter coverage")
    return {
        "status": "FAIL" if problems else "PASS",
        "problems": problems,
        "canonical_small_parameters_verified": sorted(small),
        "rank_reports": len(reports),
    }


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def execute(args):
    repo = Path(__file__).resolve().parents[2]
    source = verify_source(repo, args.expected_commit)
    selected = CASES if args.case == ["all"] else tuple(BY_NAME[name] for name in args.case)
    if len(set(case.name for case in selected)) != len(selected):
        raise ValueError("Cases must not be repeated")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    summary = {
        "source": source,
        "started_at_utc": now(),
        "status": "RUNNING",
        "scope": "Native text topology losses and all gradients; not HF parity, resume, or noise acceptance",
        "allocation": "existing dedicated single-node H100 host; managed by caller",
        "maximum_case_seconds": sum(case.timeout + 10 for case in selected),
        "cases": [{"case": case.record(), "status": "NOT_RUN"} for case in CASES],
        "dependencies": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
    }

    def save():
        summary["updated_at_utc"] = now()
        write_json(args.out / "summary.json", summary)

    save()
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled",
        TZ="UTC",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONPATH=str(repo),
        TRITON_CACHE_DIR=str(args.out / "cache/triton"),
        TORCHINDUCTOR_CACHE_DIR=str(args.out / "cache/inductor"),
    )
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP"):
        env.pop(key, None)
    try:
        for case in selected:
            row = next(row for row in summary["cases"] if row["case"]["name"] == case.name)
            row["started_at_utc"] = now()
            save()
            snapshot = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=15,
            )
            (args.out / f"{case.name}.gpu.csv").write_text(snapshot)
            devices = [line.split(",") for line in snapshot.splitlines() if line.strip()]
            if len(devices) < case.world:
                row.update(status="NOT_RUN", reason="Insufficient allocated devices")
                save()
                continue
            selected_devices = devices[: case.world]
            if any("H100" not in device[2] for device in selected_devices):
                raise RuntimeError("Selected devices are not the required H100 hardware")
            if any(int(device[3].strip()) >= 1000 for device in selected_devices):
                raise RuntimeError("Selected devices are busy; no shared GPU run is allowed")
            case_env = dict(
                env,
                **case.environment(),
                CUDA_VISIBLE_DEVICES=",".join(device[0].strip() for device in selected_devices),
            )
            output = args.out / case.name
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={case.world}",
                "--module",
                "tests.acceptance.omni_topology_entry",
                "--expected-commit",
                args.expected_commit,
                "--case",
                case.name,
                "--out",
                str(output),
            ]
            row.update(
                command=command,
                status="RUNNING",
                gpu_snapshot_sha256=sha256(args.out / f"{case.name}.gpu.csv"),
            )
            save()
            logfile = args.out / f"{case.name}.log"
            with logfile.open("w") as stream:
                result = subprocess.run(
                    ["timeout", "--signal=TERM", "--kill-after=10s", f"{case.timeout}s", *command],
                    env=case_env,
                    cwd=repo,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=case.timeout + 30,
                )
            reports = {}
            for rank in range(case.world):
                path = output / f"rank-{rank}.json"
                if path.is_file():
                    reports[rank] = json.loads(path.read_text())
            row.update(
                validate_reports(
                    case, reports, logfile.read_text(errors="replace"), result.returncode, source
                )
            )
            row.update(exit_code=result.returncode, ended_at_utc=now(), log_sha256=sha256(logfile))
            save()
        summary["source_after"] = verify_source(repo, args.expected_commit)
        selected_rows = [
            row
            for row in summary["cases"]
            if row["case"]["name"] in {case.name for case in selected}
        ]
        success = all(row["status"] == "PASS" for row in selected_rows)
        summary["status"] = (
            ("PASS_ALL_13" if len(selected) == 13 else "PASS_SELECTED")
            if success
            else "FAIL_OR_INCOMPLETE"
        )
    except Exception as error:
        summary.update(status="FAIL_OR_INCOMPLETE", error=repr(error))
    finally:
        summary["ended_at_utc"] = now()
        summary["elapsed_seconds"] = round(time.monotonic() - start, 3)
        save()
    return 0 if summary["status"] in ("PASS_ALL_13", "PASS_SELECTED") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="CPU-only list; no GPU imports or launch"
    )
    parser.add_argument("--case", choices=["all", *BY_NAME], nargs="+")
    parser.add_argument("--expected-commit")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.list:
        print(json.dumps([case.record() for case in CASES], indent=2))
        return 0
    if not (args.case and args.expected_commit and args.out):
        parser.error("Execution requires explicit case, exact expected commit, and output")
    if "all" in args.case and args.case != ["all"]:
        parser.error("'all' cannot be combined with other cases")
    args.out = args.out.resolve()
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
