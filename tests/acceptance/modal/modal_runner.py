"""Explicit, task-scoped Modal launches with one shared local reservation ledger."""

import argparse
import contextlib
import datetime
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE = "nvidia/cuda@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c"
FILE_LIMIT = 50 * 1024**2
TRANSFER_LIMIT = 1024**3
PROFILES = {
    "single": dict(
        gpu="H100",
        gpu_count=1,
        cpu_request=2,
        cpu_limit=4,
        memory_request_gib=16,
        memory_limit_gib=32,
        lifetime=1800,
        reserve=10.0,
    ),
    "distributed": dict(
        gpu="H100:4",
        gpu_count=4,
        cpu_request=8,
        cpu_limit=16,
        memory_request_gib=64,
        memory_limit_gib=128,
        lifetime=7200,
        reserve=65.0,
    ),
    "folded": dict(
        gpu="H100:8",
        gpu_count=8,
        cpu_request=16,
        cpu_limit=32,
        memory_request_gib=64,
        memory_limit_gib=128,
        lifetime=1800,
        reserve=25.0,
    ),
    "full": dict(
        gpu="H200",
        gpu_count=1,
        cpu_request=8,
        cpu_limit=16,
        memory_request_gib=128,
        memory_limit_gib=256,
        lifetime=3600,
        reserve=35.0,
    ),
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sdk():
    # Local initialization, inspection and --help do not import/authenticate Modal.
    modal = importlib.import_module("modal")
    if modal.__version__ != "1.4.0":
        raise RuntimeError("Use the pinned controller dependency modal==1.4.0")
    return modal


@contextlib.contextmanager
def lock(run_dir):
    path = run_dir / "controller.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "Another controller holds this ledger; do not start a second run"
        ) from exc
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def validate_budget(ledger):
    total = ledger["total_budget_usd"]
    amounts = [entry["reservation_usd"] for entry in ledger["entries"]]
    values = [total, ledger["reserved_upper_bound"], ledger["unreserved_remaining"], *amounts]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values
    ):
        raise ValueError("Budget values must be finite and nonnegative")
    if not 0 < total <= 300:
        raise ValueError("The shared budget must be at most USD 300")
    if len({entry["id"] for entry in ledger["entries"]}) != len(ledger["entries"]):
        raise ValueError("Reservation IDs must be unique")
    if not math.isclose(sum(amounts), ledger["reserved_upper_bound"], abs_tol=1e-8):
        raise ValueError("Reservation total does not match the ledger")
    if not math.isclose(total - sum(amounts), ledger["unreserved_remaining"], abs_tol=1e-8):
        raise ValueError("Remaining budget does not match the ledger")
    policy = ledger["policy"]
    if (
        policy["max_simultaneous_paid_sandboxes"] != 1
        or policy["max_gpu_count"] > 8
        or not 0 <= policy["repair_reserve_usd"] <= total
    ):
        raise ValueError("Invalid concurrency, GPU or repair-reserve policy")
    rates = ledger["rates_usd_per_second"]
    required = {
        "h100_each",
        "h200_each",
        "function_cpu_core",
        "function_memory_gib",
        "sandbox_cpu_core",
        "sandbox_memory_gib",
    }
    if set(rates) != required or not all(
        math.isfinite(value) and value > 0 for value in rates.values()
    ):
        raise ValueError("Missing or invalid verified resource prices")
    if not math.isfinite(ledger["egress_usd_per_gib"]) or ledger["egress_usd_per_gib"] < 0:
        raise ValueError("Invalid egress price")


def initialize(run_dir, commit, app_name):
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Provide the final 40-character source commit")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,49}", app_name):
        raise ValueError("Use a short alphanumeric/hyphen Modal application name")
    run_dir.mkdir(parents=True, exist_ok=True)
    with lock(run_dir):
        if any((run_dir / name).exists() for name in ("budget.json", "state.json")):
            raise RuntimeError("Initialization cannot overwrite an existing budget or state")
        ledger = read(ROOT / "budget.template.json")
        state = read(ROOT / "state.template.json")
        state.update(source_commit=commit, app_name=app_name, run_key=uuid.uuid4().hex[:12])
        validate_budget(ledger)
        write(run_dir / "budget.json", ledger)
        write(run_dir / "state.json", state)
    print("Initialized local files. Verify current prices in budget.json before explicit create.")


def reservation(ledger, state, profile):
    validate_budget(ledger)
    if ledger["prices_verified_utc_date"] != utc()[:10]:
        raise RuntimeError("Verify all prices/terms today and set prices_verified_utc_date first")
    settings = PROFILES[profile]
    rates = ledger["rates_usd_per_second"]
    compute_bound = settings["lifetime"] * (
        settings["gpu_count"] * rates["h200_each" if settings["gpu"] == "H200" else "h100_each"]
        + settings["cpu_limit"] * rates["sandbox_cpu_core"]
        + settings["memory_limit_gib"] * rates["sandbox_memory_gib"]
    )
    if settings["gpu_count"] > ledger["policy"]["max_gpu_count"]:
        raise RuntimeError("GPU count exceeds shared policy")
    # Legacy build charges, transfer, storage and billing uncertainty share the
    # allowance. New image builds are disabled. No credits are subtracted.
    if compute_bound + max(5.0, ledger["egress_usd_per_gib"]) >= settings["reserve"]:
        raise RuntimeError("Updated prices leave insufficient build/transfer/storage allowance")
    if state["phase"] in ("image_build_failed", "image_lookup_failed"):
        attempts = state.get("image_lookup_attempts", state.get("image_build_attempts", 0))
        if state["profile"] != profile or attempts >= 3:
            raise RuntimeError("Image retry requires the same profile and at most three attempts")
        record = next(entry for entry in ledger["entries"] if entry["id"] == state["reservation"])
        if record["reservation_usd"] != settings["reserve"]:
            raise RuntimeError("Retry reservation differs from the original")
        return record
    if state["phase"] not in ("idle", "terminated"):
        raise RuntimeError("Previous launch is active or uncertain; reconcile it before creating")
    if (
        ledger["unreserved_remaining"] - settings["reserve"]
        < ledger["policy"]["repair_reserve_usd"]
    ):
        raise RuntimeError("Launch would spend the shared repair reserve")
    record = {
        "id": f"{state['run_key']}-{profile}-{len(ledger['entries']) + 1}",
        "started_utc": utc(),
        "status": "RESERVED",
        "reservation_usd": settings["reserve"],
        "compute_bound_usd": compute_bound,
        "transfer_reserved_bytes": 0,
        "resources": settings.copy(),
        "source_commit": state["source_commit"],
        "rates_usd_per_second": rates.copy(),
        "egress_usd_per_gib": ledger["egress_usd_per_gib"],
        "prices_verified_utc_date": ledger["prices_verified_utc_date"],
    }
    ledger["entries"].append(record)
    ledger["reserved_upper_bound"] += settings["reserve"]
    ledger["unreserved_remaining"] -= settings["reserve"]
    validate_budget(ledger)
    return record


def create(run_dir, profile, image_id):
    if not re.fullmatch(r"im-[A-Za-z0-9]+", image_id):
        raise ValueError("Provide the immutable ID of an already successfully built image")
    state, ledger = read(run_dir / "state.json"), read(run_dir / "budget.json")
    modal = sdk()
    if state["phase"] == "terminated" and modal.Sandbox.from_id(state["sandbox_id"]).poll() is None:
        raise RuntimeError("Previously stopped Sandbox is still running")
    retry = state["phase"] in ("image_build_failed", "image_lookup_failed")
    if retry and state.get("runtime_image", image_id) != image_id:
        raise RuntimeError("Retry must use the same immutable image")
    record = reservation(ledger, state, profile)
    # Reserve before any remote action. Interrupted writes fail closed on the next run.
    write(run_dir / "budget.json", ledger)
    if not retry and state["phase"] != "idle":
        write(run_dir / f"state-{state['reservation']}.json", state)
    state = {key: state[key] for key in ("app_name", "run_key", "source_commit")} | {
        "phase": "reserved",
        "reservation": record["id"],
        "profile": profile,
        "image_lookup_attempts": state.get("image_lookup_attempts", 0) + 1 if retry else 1,
        "runtime_image": image_id,
        "created_utc": utc(),
    }
    write(run_dir / "state.json", state)
    settings = PROFILES[profile]
    try:
        app = modal.App.lookup(state["app_name"], create_if_missing=True)
        state.update(app_id=app.app_id, phase="image_lookup")
        write(run_dir / "state.json", state)
        runtime_image = modal.Image.from_id(image_id)
        # In Modal 1.4.0 this resolves only ImageFromId. There is no image recipe,
        # ImageGetOrCreate call or builder in this execution path.
        runtime_image.build(app)
        if runtime_image.object_id != image_id:
            raise RuntimeError("Image lookup returned a different image")
    except Exception:
        state.update(phase="image_lookup_failed", failed_utc=utc())
        record["status"] = "IMAGE_LOOKUP_FAILED_NO_SANDBOX"
        write(run_dir / "state.json", state)
        write(run_dir / "budget.json", ledger)
        raise
    state.update(
        phase="sandbox_creation_started",
        image=IMAGE,
        runtime_image=runtime_image.object_id,
        existing_image_confirmed_utc=utc(),
    )
    record["runtime_image"] = image_id
    write(run_dir / "state.json", state)
    # A failure after this point is uncertain: never automatically issue create again.
    sandbox = modal.Sandbox.create(
        app=app,
        name=record["id"],
        image=runtime_image,
        gpu=settings["gpu"],
        cpu=(settings["cpu_request"], settings["cpu_limit"]),
        memory=(settings["memory_request_gib"] * 1024, settings["memory_limit_gib"] * 1024),
        timeout=settings["lifetime"],
        idle_timeout=600,
        env={
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "WANDB_MODE": "disabled",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "PYTHONNOUSERSITE": "1",
            "OMNI_SOURCE_COMMIT": state["source_commit"],
        },
    )
    state.update(phase="running", sandbox_id=sandbox.object_id)
    record["status"] = "RUNNING"
    write(run_dir / "state.json", state)
    write(run_dir / "budget.json", ledger)
    print(json.dumps(state, indent=2))


def get(run_dir, allow_terminated=False):
    state = read(run_dir / "state.json")
    if state["phase"] == "terminated" and not allow_terminated:
        raise RuntimeError("Sandbox has terminated")
    if state["phase"] not in ("sandbox_creation_started", "running", "terminated"):
        raise RuntimeError("No Sandbox was created; inspect local state")
    modal = sdk()
    if "sandbox_id" in state:
        return modal.Sandbox.from_id(state["sandbox_id"])
    # Recover only the exact named Sandbox from this launch, never create a replacement.
    sandbox = modal.Sandbox.from_name(state["app_name"], state["reservation"])
    state.update(sandbox_id=sandbox.object_id, phase="running")
    write(run_dir / "state.json", state)
    return sandbox


def transfer(run_dir, count):
    if count < 0 or count > FILE_LIMIT:
        raise RuntimeError("Single transfer exceeds 50 MiB; remote evidence is retained")
    state, ledger = read(run_dir / "state.json"), read(run_dir / "budget.json")
    validate_budget(ledger)
    entry = next(item for item in ledger["entries"] if item["id"] == state["reservation"])
    if entry["transfer_reserved_bytes"] + count > TRANSFER_LIMIT:
        raise RuntimeError("Reservation transfer limit of 1 GiB would be exceeded")
    entry["transfer_reserved_bytes"] += count
    write(run_dir / "budget.json", ledger)


def upload(run_dir, local, remote):
    if not remote.startswith("/"):
        raise ValueError("Remote destination must be an absolute container path")
    content = Path(local).read_bytes()
    sandbox = get(run_dir)
    transfer(run_dir, len(content))
    sandbox.filesystem.write_bytes(content, remote)
    print(
        json.dumps(
            {"upload": remote, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    )


def execute(run_dir, script, label, seconds):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", label) or not 1 <= seconds <= 900:
        raise ValueError("Use a safe unique label and a timeout from 1 to 900 seconds")
    evidence = run_dir / "evidence"
    evidence.mkdir(exist_ok=True)
    if any((evidence / (label + suffix)).exists() for suffix in (".sh", ".json", ".log")):
        raise RuntimeError("Use a new label; existing evidence cannot be overwritten")
    sandbox = get(run_dir)
    content = Path(script).read_bytes()
    transfer(run_dir, len(content))
    (evidence / (label + ".sh")).write_bytes(content)
    remote_script, remote_log = "/tmp/" + label + ".sh", "/tmp/" + label + ".log"
    sandbox.filesystem.write_bytes(content, remote_script)
    start = time.monotonic()
    result = {
        "label": label,
        "utc": utc(),
        "source_commit": read(run_dir / "state.json")["source_commit"],
        "script_sha256": hashlib.sha256(content).hexdigest(),
        "sandbox_id": sandbox.object_id,
        "timeout_seconds": seconds,
        "returncode": None,
        "log_complete": False,
    }
    try:
        process = sandbox.exec(
            "bash",
            "-lc",
            f"timeout --signal=TERM --kill-after=10s {seconds}s bash {remote_script} > {remote_log} 2>&1",
            timeout=seconds + 20,
        )
        result["returncode"] = process.wait()
        size = sandbox.exec("wc", "-c", remote_log, timeout=20)
        count = int(size.stdout.read().split()[0])
        if size.wait() != 0:
            raise RuntimeError("Cannot determine log transfer size")
        transfer(run_dir, count)
        log = sandbox.filesystem.read_bytes(remote_log)
        (evidence / (label + ".log")).write_bytes(log)
        result.update(log_complete=True, log_sha256=hashlib.sha256(log).hexdigest())
        print(log[-12000:].decode(errors="replace"), flush=True)
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        write(evidence / (label + ".json"), result)
    print(json.dumps(result), flush=True)
    return result["returncode"]


def stop(run_dir):
    state = read(run_dir / "state.json")
    if state["phase"] in ("idle", "image_build_failed", "image_lookup_failed"):
        print("No GPU Sandbox has been created.")
        return
    sandbox = get(run_dir, allow_terminated=True)
    sandbox.terminate(wait=True)
    final_poll = sandbox.poll()
    if final_poll is None:
        raise RuntimeError("Termination is not confirmed; reservation and active state retained")
    state = read(run_dir / "state.json")
    state.update(phase="terminated", terminated_utc=utc(), final_poll=final_poll)
    write(run_dir / "state.json", state)
    ledger = read(run_dir / "budget.json")
    entry = next(item for item in ledger["entries"] if item["id"] == state["reservation"])
    entry.update(status="STOPPED_RESERVATION_RETAINED", stopped_utc=utc())
    write(run_dir / "budget.json", ledger)
    print(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="One external directory shared by all task controllers; never commit it",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser(
        "init", help="Create local zero-spend templates; no authentication or cloud calls"
    )
    init.add_argument("--commit", required=True)
    init.add_argument("--app-name", default="pith-omni-validation")
    launch = sub.add_parser(
        "create", help="Explicitly reserve budget and allocate from an existing image"
    )
    launch.add_argument("profile", choices=PROFILES)
    launch.add_argument(
        "--image-id", required=True, help="Already successfully built immutable image ID"
    )
    put = sub.add_parser("upload")
    put.add_argument("local", type=Path)
    put.add_argument("remote")
    run = sub.add_parser("exec")
    run.add_argument("script", type=Path)
    run.add_argument("label")
    run.add_argument("seconds", type=int)
    sub.add_parser("stop")
    sub.add_parser("status", help="Inspect local state and this Sandbox's remote terminal status")
    sub.add_parser("inspect", help="Read local state and budget only")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.action == "init":
        initialize(run_dir, args.commit, args.app_name)
        return 0
    with lock(run_dir):
        if args.action == "create":
            create(run_dir, args.profile, args.image_id)
        elif args.action == "upload":
            upload(run_dir, args.local, args.remote)
        elif args.action == "exec":
            return execute(run_dir, args.script, args.label, args.seconds)
        elif args.action == "stop":
            stop(run_dir)
        else:
            state, ledger = read(run_dir / "state.json"), read(run_dir / "budget.json")
            validate_budget(ledger)
            result = {"state": state, "budget": ledger}
            if args.action == "status" and state["phase"] in (
                "sandbox_creation_started",
                "running",
                "terminated",
            ):
                result["poll"] = get(run_dir, allow_terminated=True).poll()
            print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
