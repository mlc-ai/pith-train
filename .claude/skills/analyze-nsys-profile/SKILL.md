---
name: analyze-nsys-profile
description: Query a captured PithTrain Nsight Systems profile to answer specific questions about kernel timing, compute/communication overlap, per-rank behavior, and pipeline structure. Use when the user asks to "analyze an nsys profile", "check overlap quality", "find exposed comm", "compare ranks", "investigate a long NCCL kernel", or any question that begins from an existing `.nsys-rep` file. Tool-first skill — assumes the trace was already captured (see capture-nsys-profile) and provides query primitives the agent composes to answer the specific question being asked.
---

# Analyze Nsys Profile

A passive query toolkit for PithTrain nsys traces. The agent asks a specific question; the skill provides primitives that answer it fast and correctly. **The skill does not produce an unsolicited full report.** It expects the agent to compose the right query for the question being asked.

## Prerequisites

- A captured `.nsys-rep` exists (default location: `workspace/capture-nsys-profile/pithtrain_node*.nsys-rep`).
- The repo venv is active: `source .venv/bin/activate`.
- `nsys` CLI on `PATH` (for the one-time SQLite export).

## Step 1 — Export the trace to SQLite

```bash
nsys export --type=sqlite --force-overwrite=true \
    --output=workspace/capture-nsys-profile/pithtrain_node0.sqlite \
    workspace/capture-nsys-profile/pithtrain_node0.nsys-rep
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

## Step 3 — Run the analysis script for the question

| Question | Primitive |
|---|---|
| How well does comm overlap with compute in the steady-state window? | `compute_overlap.py` |
| Which DualPipeV stage has the worst exposure or the largest idle gaps? | `summarize_stages.py --view exposed`/`gaps` |
| Is this specific NCCL kernel data movement or a straggler wait? | `compare_kernel.py` |

All scripts live under `.claude/skills/analyze-nsys-profile/scripts/` and take the SQLite path as the first positional argument.

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
- "Kernel duration > T µs means data movement" (long duration can be straggler wait — see `compare_kernel.py`).
- "Stream with avg µs < threshold is EP" (depends on token volume per rank).

Use these instead:

- One-sided purity check for compute-vs-comm streams.
- NVTX-context labeling for stream purpose: sample a few kernels per stream, look up the innermost PithTrain stage range, take the mode. Implemented in `classify_streams.py`.
- Cross-rank duration comparison for the bubble-vs-data question. Implemented in `compare_kernel.py`.

## Worked examples

See [references/examples.md](references/examples.md) for recipe-style answers to:

- How much EP all-to-all is being overlapped with compute?
- Which EP phase (dispatch-f/b, combine-f/b) has the worst overlap?
- Which stage has the largest idle gaps between kernels?
- Is rank N's long NCCL kernel real comm or a pipeline bubble?
- Are the PP stages balanced?

## When to dispatch sub-agents

A common pattern: a primary analysis surfaces K interesting kernels (e.g. the top-K longest exposed comm kernels from `compute_overlap.py`). Each kernel can be investigated independently and concurrently. Use the prompt template in [references/subagent-template.md](references/subagent-template.md); each sub-agent runs `compare_kernel.py` on its assigned kernel and reports back with a bubble-vs-data verdict.

## Output guidance

- Scripts emit a fixed-width table to stdout. One column per record field; agents read it directly.
- Cross-script joins are by `pid` — every script's table includes pid as the per-rank identifier; downstream rows compose against `show_setup`'s mapping `pid ↔ rank ↔ setup`.
- When reporting to the human user, summarize as plain prose with a small table extracted from the relevant columns.
- Always cite the analysis window. An overlap percentage with no window is meaningless.

## Gotchas (surfaced by prior agent runs)

- **`classify_streams.py` only reports streams active in the analysis window**, not every stream that exists in the trace. A rank typically has 6-8 streams overall but only 2-3 inside a single steady-state chunk. This is intentional — analyzing a small window does not need the inactive streams.
- **PP P2P kernels rarely appear in a single chunk window** — they fire between chunks. Widen the window (`--start NS --end NS` on the analysis script) if you specifically want to see the PP P2P comm stream.
- **`hidden_pct` and `exposed_pct` are both 0-100** (not 0-1 fractions). Both `compute_overlap.py` and `summarize_stages.py` follow this convention.
- **CPU launch time vs GPU execution time** — for any NVTX-context lookup on a kernel, use `kernel["launch_start"]` (CPU-side `cudaLaunchKernel` time) rather than `kernel["start"]` (GPU-side execution time). All scripts already do this; if you write an ad-hoc query, call `common.innermost_nvtx` on launch_start values.
- **Comm-stream purpose uses unanimity, not majority** — every kernel on the stream must agree on its enclosing-NVTX category, otherwise the label is `mixed`. A single mis-categorized kernel surfaces as `mixed` instead of silently being out-voted.

## Common Issues

### `no such table: NVTX_EVENTS`

The `.nsys-rep` has not been exported yet. Run the `nsys export` command from Step 1.

### Overlap headline looks wrong (e.g. PP P2P inflating "comm time")

You probably summed across all streams without classifying. PP P2P recv kernels block on remote sends — they are pipeline-bubble waits, not overlap candidates. Run `classify_streams.py` first to identify which streams carry EP a2a, then run `compute_overlap.py` per comm stream.

### Negative timestamps on NVTX events

NCCL init opened these ranges before `cudaProfilerStart`. Filter `WHERE start >= 0` to scope to the profiled window.
