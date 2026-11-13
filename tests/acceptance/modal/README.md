# Reproducing Omni acceptance on Modal

This directory packages the task-scoped controller and the dependency recipe used
for Omni validation. It does not run automatically, deploy a service, authenticate,
or start a builder. `create` requires the immutable ID of an **already successfully
built** image, supplied at runtime. No account identifiers or previous spending
records are included.

Use the existing authorized Modal account and a Python 3.12 control environment
with `modal==1.4.0` from `controller-requirements.txt`. Do not copy credentials into
the repository, image, scripts, logs, or run directory. If existing authentication
is unavailable, stop and arrange access separately.

## Image provenance and build limitation

The recipe consists of `modal_build_cpu.sh`, `runtime-requirements.txt`, and
`media-requirements.txt`. Its registry base is pinned in `modal_runner.IMAGE`.
Python is 3.12; torch is `2.13.0+cu130`, torchvision is `0.28.0+cu130`, Transformers
is `5.17.0`, and DeepGEMM is pinned to commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`. The build script restores `/tmp` mode
1777 before apt, runs `pip check`, verifies the torch/torchvision versions, and
requires CPU-only construction. The original build command used:

```sh
timeout --signal=TERM --kill-after=10s 600 bash /tmp/modal_build_cpu.sh
```

Keep the successful CPU build log and complete installed-package inventory with
the private run evidence. A script timeout does **not** bound registry resolution,
image setup, or all builder resource charges. Modal 1.4.0 `Image.run_commands`
exposes no CPU/RAM limit, and the SDK and public documentation do not establish a
usable complete builder cost ceiling. Accordingly this controller has **no**
`from_registry`, `run_commands`, or `run_function` execution path. It cannot create
a replacement image if the supplied ID is unavailable.

`Image.from_id(id).build(app)` is used solely to resolve that existing image before
requesting a GPU. In the pinned SDK its loader calls `ImageFromId`, not
`ImageGetOrCreate`; it does not execute the recipe. The API method name `build`
does not imply a new build for this object. Never substitute a recipe-backed Image
in that path. See the [image guide](https://modal.com/docs/guide/images) and
[resource semantics](https://modal.com/docs/guide/resources).

## One budget for the whole task

Use **one coordinator, one run directory, and one ledger across every component
and retry**, including work performed by other agents. Do not initialize a new
ledger for a continuation of an existing task. Initialize only a genuinely new
budget, or first reconcile the existing shared task ledger into this schema
without dropping any past reservations or charges.

The template contains a USD 300 ceiling, USD 100 repair reserve, and zero entries.
The 2026-10-20 rate snapshot is illustrative: verify all current GPU, CPU, RAM,
transfer and storage prices, applicable account terms, and any other related fees
before every launch date. Update `rates_usd_per_second`, `egress_usd_per_gib`, and
`prices_verified_utc_date` in the **local** `budget.json` only after this review.
Do not assume free credits. Reference
[pricing](https://modal.com/pricing),
[Sandbox billing](https://modal.com/docs/guide/sandbox-resources), and
[network egress](https://modal.com/docs/guide/network-egress-billing).
If charges cannot reliably fit the remaining shared budget, stop.

| Profile | GPU | CPU request / limit | RAM request / limit (GiB) | Hard lifetime | Reservation |
|---|---|---:|---:|---:|---:|
| `single` | 1 × H100 | 2 / 4 | 16 / 32 | 1,800 s | $10 |
| `distributed` | 4 × H100 | 8 / 16 | 64 / 128 | 7,200 s | $65 |
| `folded` | 8 × H100 | 16 / 32 | 64 / 128 | 1,800 s | $25 |
| `full` | 1 × H200 | 8 / 16 | 128 / 256 | 3,600 s | $35 |

All four reservations total $135. Each reservation exceeds the lifetime estimate
for GPU + requested CPU limit + RAM limit by at least $5 for legacy build charges,
storage, transfers, and uncertainty. This is a conservative planning allowance,
**not an account billing cap**. CPU limits are throttling limits; actual billing
still requires reconciliation. No volume, snapshot, premium region, or additional
paid function is requested. Confirm any existing image storage charges separately.
The controller never automatically releases a reservation after `stop`.

The local exclusive lock serializes actions, and an active or uncertain launch
blocks the next `create`. A failed existing-image lookup can be retried at most
three times with the same profile, image and reservation. A known terminal
legacy build failure can retain its reservation when switching to an existing
image; the controller never retries the build itself. A lost response after
Sandbox creation begins is **uncertain**, not evidence of failure. `status` or
`stop` resolves only the exact Sandbox name recorded for that reservation. Do not
launch a replacement until its state is reconciled.

These are task-controller constraints, not enforcement against other users or
independent run directories. Never launch another controller for this budget.
After an interrupted controller leaves a lock, first check its process and exact
remote resource state before removing the stale local lock.

## Initialize, explicitly launch, collect evidence, stop

The following Bash example runs from a clean checkout of the final commit. Keep
`RUN_DIR` outside Git. A fresh local initialization has no remote side effects:

```sh
PACKAGE=tests/acceptance/modal
RUN_DIR="$(mktemp -d)"
SOURCE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
python "$PACKAGE/modal_runner.py" --help
python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" init --commit "$SOURCE_COMMIT"
python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" inspect
```

Review/update the local budget as described above. Set `OMNI_RUNTIME_IMAGE_ID` to
the immutable image ID from the successful CPU build evidence. The placeholder
below deliberately supplies no account-specific value:

```sh
: "${OMNI_RUNTIME_IMAGE_ID:?Set the existing validated image ID}"
python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" create single \
  --image-id "$OMNI_RUNTIME_IMAGE_ID"
trap 'python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" stop' EXIT
git bundle create "$RUN_DIR/source.bundle" HEAD
python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" upload \
  "$RUN_DIR/source.bundle" /tmp/omni-source.bundle
python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" exec \
  "$PACKAGE/bootstrap.sh" bootstrap 120
```

`bootstrap.sh` checks out the exact recorded commit, verifies a clean tree, and
records GPU type/count/UUID, driver, topology, Python/CUDA versions and installed
packages. The base image digest and dependency recipe alone do not lock apt
repository contents or Modal's Python patch version; the successful image ID and
runtime manifest are the identity of an actual run.

For each required gate, supply a reviewed script that starts with:

```sh
set -euo pipefail
cd /work/repo
test "$(git rev-parse HEAD)" = "$OMNI_SOURCE_COMMIT"
test -z "$(git status --porcelain)"
export PYTHONPATH=/work/repo
# Invoke the required committed acceptance gate and print its result JSON.
```

Then invoke `exec gate.sh UNIQUE_LABEL SECONDS`. Every invocation has a 1–900 s
timeout with a further 10 s kill grace; the Sandbox also has its fixed lifetime
and a 600 s idle timeout. Run the four full-size text/image/audio/video gates
sequentially on `full`, with no more than 600 s per gate. Longer distributed suites
must use separate bounded commands on the same Sandbox. Tests must not create
additional paid resources. Use the final acceptance matrix and committed scripts
in `tests/acceptance/` to choose gates and interpret their results.

The controller saves the exact script, script hash, return code, source commit,
runtime identifier and complete log/hash under `RUN_DIR/evidence`. Labels cannot
overwrite earlier evidence. A transfer is capped at 50 MiB, with a cumulative
1 GiB upload/download allowance per reservation; larger logs fail retrieval and
remain remote. Keep test result JSON in the log or retrieve it within this
allowance before stopping. These caps do not meter arbitrary network activity
inside a supplied script; review downloads/egress before running it.

```sh
python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" status
python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" stop
python "$PACKAGE/modal_runner.py" --run-dir "$RUN_DIR" inspect
```

`stop` reports termination only after `poll()` is terminal. Stop your Sandbox
after success, failure or timeout; confirm termination and retain its reservation
until invoices/usage are reconciled. Preserve evidence locally. Never commit the
run directory or delete unrelated account resources.

## Local contract check

```sh
python -m pytest -q tests/acceptance/modal/test_runner_contract.py
```

These tests mock remote operations and exercise budget arithmetic, concurrency,
failure states, transfer limits, and confirmed termination. They neither run a
GPU nor establish model correctness. Bootstrap success is also not acceptance:
numerical/gradient thresholds, distributed topology, conversion, checkpoint
resume and integration gates still require their complete final-commit evidence.
