---
name: analyze-nsys-profile
description: Query a captured PithTrain Nsight Systems profile to measure compute/communication overlap, locate exposed comm by DualPipeV stage, and inspect per-rank stream behavior. Use when the user asks to "analyze an nsys profile", "check overlap quality", "find exposed comm", "which stage is the bottleneck", or any question that starts from an existing `.nsys-rep` file. Assumes the trace was already captured (see capture-nsys-profile); provides query primitives the agent composes for the specific question being asked.
---

# Analyze Nsys Profile

A passive query toolkit for PithTrain nsys traces. The agent asks a specific question; the skill provides primitives that answer it fast and correctly. **The skill does not produce an unsolicited full report.** It expects the agent to compose the right query for the question being asked.

## Prerequisites

- A captured `.nsys-rep` exists (default location: `workspace/capture-nsys-profile/pithtrain_node*.nsys-rep`).
- The repo venv is active: `source .venv/bin/activate`.
- `nsys` CLI on `PATH` (for the one-time SQLite export).

## Step 1 — Export the trace to SQLite

```bash
nsys export --type=sqlite --force-overwrite=true --output=workspace/capture-nsys-profile/pithtrain_node0.sqlite workspace/capture-nsys-profile/pithtrain_node0.nsys-rep
```

All subsequent queries hit the SQLite, not the raw `.nsys-rep`.

## Step 2 — Shared preparation (always run first)

Three primitives establish *who*, *when*, and *what* — every downstream analysis depends on the data they surface. Run (or at least understand the output of) all three before reaching for the analysis scripts below.

| Question | Primitive |
|---|---|
| What ranks are in this trace? What's the per-rank setup? | `show_setup.py` |
| What's the steady-state analysis window for each rank? | `find_window.py` |
| Which streams are compute / comm, and what's each comm stream's purpose? | `classify_streams.py` |

Pipeline: `show_setup` → `find_window` → `classify_streams`. show_setup gives you the mapping `pid ↔ rank ↔ mesh coordinates`; find_window picks the median DualPipeV chunk per rank (deterministic across re-runs, so before/after comparisons are valid); classify_streams identifies which CUDA streams in that window are compute vs comm, and labels the comm streams' purpose (`ep_a2a`, `cp_ring`, `pp_p2p`).

## Step 3 — Measure overlap per DualPipeV stage

```bash
python .claude/skills/analyze-nsys-profile/scripts/compute_overlap.py workspace/capture-nsys-profile/pithtrain_node0.sqlite
```

Emits one row per `(rank, stage)` with columns: `pid | stage | exposed_ns | overlap | overlap_min | overlap_max`. The `overlap` column is the time-weighted hidden fraction across the stage's comm kernels; `overlap_min` / `overlap_max` are the extremes of the per-kernel overlap percentage and surface whether the stage is uniformly bad or bimodal.

See [references/examples.md](references/examples.md) for recipes that compose this with the Step 2 primitives.

## Critical conventions

Before composing a custom SQL query, read [references/conventions.md](references/conventions.md). Highlights:

- **`pid`** (Linux PID) is the per-rank join key, extracted exactly as the nsys docs prescribe: `globalPid / 0x1000000 % 0x1000000` (kernel rows) == `globalTid / 0x1000000 % 0x1000000` (NVTX rows). Single SQLite per node → PIDs unique within a trace.
- Always filter **`start >= 0`** — pre-`cudaProfilerStart` NCCL init ranges have negative timestamps.
- **Per-rank setup label** is the first in-window NVTX event per rank: `rank=N; pp=R/S dp=R/S cp=R/S ep=R/S; mbs=M seq=Q`.
- **Chunk anchor** for steady state: the median-indexed `forward chunk X (phaseY) backward chunk Z (phaseW)` NVTX range emitted by DualPipeV. Match with `LIKE 'forward chunk%backward chunk%'` to disambiguate from per-stage forward markers.
- **Compute-vs-comm classification**: a kernel is communication if its short name starts with `nccl`, otherwise compute. A stream is a comm stream iff every one of its kernels is NCCL.
- **Comm-stream purpose** is discovered from the PithTrain stage-NVTX (`layer*.stageN_*`) enclosing each kernel at its CPU-side **launch time** — every kernel must agree on the label (unanimity), otherwise `mixed`.

## Non-fragile classification rules

Avoid these heuristics — they break across configs:

- "Stream with > N kernels of type X is comm" (N depends on layer count, chunks, seq length).
- "Kernel duration > T µs means data movement" (long duration can also be a straggler wait).
- "Stream with avg µs < threshold is EP" (depends on token volume per rank).

Use these instead:

- One-sided purity check for compute-vs-comm streams.
- NVTX-context labeling for stream purpose: look up the innermost PithTrain stage range enclosing each kernel and require unanimous agreement. Implemented in `classify_streams.py`.

## Worked examples

See [references/examples.md](references/examples.md) for recipe-style answers to:

- How well is each EP phase overlapped with compute?
- Which EP phase has the worst overlap?
- Are the PP stages balanced?
- Which (rank, stage) carries the most exposed comm?

## Output guidance

- Scripts emit a fixed-width table to stdout. One column per record field; agents read it directly.
- Cross-script joins are by `pid` — every script's table includes pid as the per-rank identifier; downstream rows compose against `show_setup`'s mapping `pid ↔ rank ↔ setup`.
- When reporting to the human user, summarize as plain prose with a small table extracted from the relevant columns.
- Always cite the analysis window. An overlap percentage with no window is meaningless.

## Gotchas (surfaced by prior agent runs)

- **`classify_streams.py` only reports streams active in the analysis window**, not every stream that exists in the trace. A rank typically has 6-8 streams overall but only 2-3 inside a single steady-state chunk. This is intentional — analyzing a small window does not need the inactive streams.
- **PP P2P kernels rarely appear in a single chunk window** — they fire between chunks. Widen the window (`--start NS --end NS` on the analysis script) if you specifically want to see the PP P2P comm stream.
- **`compute_overlap.py`'s percent cells include a trailing `%`** (`58.4%`, not `0.584`). Sort/compare numerically by stripping the `%` first. Absolute time columns (`exposed_ns`) are bare integer nanoseconds.
- **CPU launch time vs GPU execution time** — for any NVTX-context lookup on a kernel, use `kernel["launch_start"]` (CPU-side `cudaLaunchKernel` time) rather than `kernel["start"]` (GPU-side execution time). All scripts already do this; if you write an ad-hoc query, call `common.innermost_nvtx` on launch_start values.
- **Comm-stream purpose uses unanimity, not majority** — every kernel on the stream must agree on its enclosing-NVTX category, otherwise the label is `mixed`. A single mis-categorized kernel surfaces as `mixed` instead of silently being out-voted.

## Common Issues

### `no such table: NVTX_EVENTS`

The `.nsys-rep` has not been exported yet. Run the `nsys export` command from Step 1.

### PP P2P comm is missing from the overlap output

By design. `compute_overlap.py` buckets kernels by their enclosing PithTrain stage NVTX (`stage1_*` through `stage5_*`); PP P2P kernels live inside the `pipeline send/recv` wrapper, which is not a stage marker, so they are filtered out. Widen the window with `--start NS --end NS` if you need to investigate them — they typically fire between chunks, not inside.

### Negative timestamps on NVTX events

NCCL init opened these ranges before `cudaProfilerStart`. Filter `WHERE start >= 0` to scope to the profiled window.
