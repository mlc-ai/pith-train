"""
Turn the logs of one end-to-end case into its result.json. launch.sh calls it on node 0 after
every case; run it by hand to re-summarize. The first log is node 0's, which holds rank 0's step
lines; the error of a failed case is gathered from every log, since any node may hold it.

    python3 -m benchmarks.end2end.summarize <node0.log> [<node1.log> ...] --case <case> --output <result.json>
    python3 -m benchmarks.end2end.summarize --remeasure <result.json> ...   # after a fixture change
"""

import argparse
import importlib.metadata
import json
import math
import os
import re
import socket
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks.end2end.common import MAX_STEPS, MODELS, WARMUP_STEPS

CASE = re.compile(
    r"(?P<model>.+)-seq(?P<seq>\d+)-(?P<dtype>bf16|fp8)/(?P<gpu>[a-z0-9]+)-(?P<nodes>\d+)n(?P<gpn>\d+)g/"
    r"pp(?P<pp>\d+)-dp(?P<dp>\d+)-cp(?P<cp>\d+)-ep(?P<ep>\d+)$"
)

STEP = re.compile(
    r"step (?P<step>\d+)/\d+ \| step-time (?P<step_time>[\d.]+) sec"
    r" \| cross-entropy-loss (?P<loss>[-\w.]+)"
    r"(?: \| load-balance-loss (?P<lb_loss>[-\w.]+))?"
    r" \| learning-rate [-+\w.]+"
    r" \| gradient-norm (?P<grad_norm>[-\w.]+)"
    r" \| tokens-per-second (?P<tokens_per_second>[\d,]+)"
    r" \| peak-gpu-memory (?P<peak_memory_gb>[\d.]+) GB"
)

PACKAGES = ["torch", "deep-gemm", "flash-attn-4", "nvidia-cutlass-dsl", "quack-kernels"]
PACKAGES += ["apache-tvm-tirx", "tirx-kernels", "triton", "transformers", "pithtrain"]


def parse_case(case: str) -> dict:
    """
    The case path is the contract: model, sequence length, dtype, hardware and mesh are read from
    it, and tests/test_end2end_cases.py checks that each script agrees with its path.
    """
    m = CASE.fullmatch(case)
    if m is None:
        raise ValueError(f"not a case path: {case}")
    nodes, gpn = int(m["nodes"]), int(m["gpn"])
    world = nodes * gpn
    pp, dp, cp, ep = (int(m[k]) for k in ("pp", "dp", "cp", "ep"))
    return dict(
        model=m["model"],
        sequence_length=int(m["seq"]),
        dtype=m["dtype"],
        gpu=m["gpu"],
        nodes=nodes,
        gpus_per_node=gpn,
        world_size=world,
        pp=pp,
        dp=dp,
        cp=cp,
        ep=ep,
        expert_dp=world // (pp * ep),
        global_batch_size=4 * world,
    )


def parse_steps(text: str) -> list[dict]:
    steps = {}
    for m in STEP.finditer(text):
        row = dict(step=int(m["step"]))
        for key in ("step_time", "loss", "lb_loss", "grad_norm", "peak_memory_gb"):
            row[key] = None if m[key] is None else float(m[key])
        row["tokens_per_second"] = float(m["tokens_per_second"].replace(",", ""))
        steps[row["step"]] = row
    return [steps[k] for k in sorted(steps)]


def environment() -> dict:
    env = dict(host=socket.gethostname(), nodelist=os.environ.get("SLURM_STEP_NODELIST"))
    for name in PACKAGES:
        try:
            env[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            env[name] = None
    query = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"]
    try:
        gpu, driver = subprocess.run(query, capture_output=True, text=True).stdout.split(", ")
        env.update(gpu_name=gpu.strip(), driver=driver.strip())
    except (OSError, ValueError):
        pass
    return env


def summarize(text: str, case: str, exit_code: int, others: tuple[str, ...] = ()) -> dict:
    steps = parse_steps(text)
    max_steps = max((int(n) for n in re.findall(r"step \d+/(\d+)", text)), default=None)
    result = dict(case=case, config=parse_case(case), steps=steps)
    if exit_code == 0 and max_steps is not None and len(steps) == max_steps:
        return measure(result)
    # A model that raises NotImplementedError for this mesh does not support it on this build
    # (e.g. context parallelism for GPT-OSS on main): recorded, not retried, not a regression.
    texts = (text,) + others
    found = (re.search(r"NotImplementedError: (.*)", t) for t in texts)
    unsupported = next((m for m in found if m), None)
    result["status"] = "unsupported" if unsupported else "failed"
    if unsupported:
        result["reason"] = unsupported[1].strip()
    # Any node may hold the root cause, so take the first and last error lines of every log.
    errors = []
    for t in texts:
        hits = [line for line in t.splitlines() if re.search(r"Error|error:|OutOfMemory", line)]
        hits = [line for line in hits if "Warning" not in line]
        errors += hits[:3] + hits[-2:] if len(hits) > 5 else hits
    lines = [line for line in text.splitlines() if line.strip()]
    result["error"] = list(dict.fromkeys(errors))[:20] + lines[-10:]
    return result


def measure(result: dict) -> dict:
    """
    Fill the metrics, checks and status of a complete case from its steps. Only the first
    MAX_STEPS steps count, timing the steps after WARMUP_STEPS, so a case that ran longer (the
    baseline ran 20) is summarized exactly as one that ran MAX_STEPS.
    """
    config, steps = result["config"], result["steps"][:MAX_STEPS]
    timed = [s for s in steps if s["step"] > WARMUP_STEPS]
    step_time = statistics.median(s["step_time"] for s in timed)
    tokens = config["global_batch_size"] * config["sequence_length"]
    losses = [s["loss"] for s in steps]
    vocab = json.loads(Path(MODELS[config["model"]][0], "config.json").read_text())
    vocab = vocab.get("text_config", vocab)["vocab_size"]
    finite = all(math.isfinite(s[k]) for s in steps for k in ("loss", "grad_norm"))
    lb = [s["lb_loss"] for s in steps if s["lb_loss"] is not None]
    result["steps_run"] = len(result["steps"])
    result["timed_steps"] = [timed[0]["step"], timed[-1]["step"]]
    result["metrics"] = dict(
        step_time=step_time,
        step_time_min=min(s["step_time"] for s in timed),
        step_time_max=max(s["step_time"] for s in timed),
        tokens_per_second_per_gpu=tokens / step_time / config["world_size"],
        peak_memory_gb=max(s["peak_memory_gb"] for s in steps),
        first_step_time=steps[0]["step_time"],
        loss_first=losses[0],
        loss_last=losses[-1],
        grad_norm_first=steps[0]["grad_norm"],
    )
    result["checks"] = dict(
        finite=finite,
        loss_decreased=finite and losses[-1] < losses[0],
        # Force-balanced routing pins E * dot(f, p) at 1: proof the case ran force-balanced.
        force_balanced=all(abs(x - 1.0) < 1e-3 for x in lb) if lb else None,
        # From fresh weights the logits are near zero, so the first loss is near ln(vocab_size).
        loss_first_minus_ln_vocab=losses[0] - math.log(vocab),
    )
    ok = result["checks"]["finite"] and result["checks"]["loss_decreased"]
    ok = ok and result["checks"]["force_balanced"] is not False
    result["status"] = "ok" if ok else "check-failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="*")
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--remeasure",
        type=Path,
        nargs="+",
        help="recompute the metrics of these result.json files from their stored steps, in place",
    )
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--commit")
    parser.add_argument("--dirty", type=int, default=0)
    parser.add_argument("--wall-seconds", type=int)
    args = parser.parse_args()
    if args.remeasure:
        for path in args.remeasure:
            result = json.loads(path.read_text())
            if result.get("metrics"):
                path.write_text(json.dumps(measure(result), indent=1) + "\n")
                print(f"{result['case']}: {result['status']}")
        return
    if not (args.logs and args.case and args.output):
        parser.error("give the logs of a case with --case and --output, or --remeasure")
    texts = [p.read_text(errors="replace") if p.exists() else "" for p in args.logs]
    result = summarize(texts[0], args.case, args.exit_code, tuple(texts[1:]))
    result.update(exit_code=args.exit_code, commit=args.commit, dirty=bool(args.dirty))
    result.update(wall_seconds=args.wall_seconds, date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    result["environment"] = environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.case}: {result['status']}")


if __name__ == "__main__":
    main()
