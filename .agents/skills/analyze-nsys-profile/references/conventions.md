# PithTrain nsys trace conventions

Everything in this file is empirically verified against a representative capture. When in doubt, query and confirm before reasoning from these notes.

## Per-rank setup label (first NVTX event per rank)

`pithtrain/tasks/pretrain_lm.py` emits an `nvtx.range_push` immediately after `cudaProfilerStart`. Format (three semicolon-separated groups: identity, topology, training config):

```
rank=N; pp=R/S dp=R/S cp=R/S ep=R/S; mbs=M seq=Q
```

Read with this SQL:

```sql
WITH first_event AS (
    SELECT globalTid / 0x1000000 % 0x1000000 AS pid, MIN(start) AS start
    FROM NVTX_EVENTS WHERE start >= 0 GROUP BY pid
)
SELECT e.globalTid / 0x1000000 % 0x1000000 AS pid,
       COALESCE(e.text, s.value) AS setup
FROM NVTX_EVENTS e
JOIN first_event f
  ON e.globalTid / 0x1000000 % 0x1000000 = f.pid AND e.start = f.start
LEFT JOIN StringIds s ON e.textId = s.id;
```

The global rank is included explicitly as the `rank=N` token, so no axis arithmetic is needed.

## Chunk anchor — DualPipeV's overlapped-F/B NVTX

DualPipeV emits per-rank chunk-level NVTX ranges named:

```
forward chunk X (phaseY) backward chunk Z (phaseW)
```

where (X, Y, Z, W) are chunk indices and phase ids. **There are ~59 such ranges per rank in a 32-microbatch-per-stage run**, each ~50-80 ms long. The owning rank is identified by the row's `pid` (no rank prefix in the name).

Within one chunk you'll find ~5-7 layers' worth of activity in both forward and backward directions (DualPipeV interleaves F and B).

**Analysis window picking rule:** for each rank, filter NVTX ranges where `name LIKE 'forward chunk%backward chunk%'` (the substring match rules out per-stage forward markers like `forward chunk 0 (phase0)`) and the row's pid matches, sort by start time, take the **median-indexed** one. This is steady state — far from the pipeline warmup at the start and drain at the end. Implemented in `find_window.extract_window()`.

Cross-rank: median chunks across ranks land within ~6 ms of each other on the wallclock, so they're suitable for cross-rank correlation.

## Layer-stage NVTX

PithTrain wraps each of the 5 DualPipeV stages per layer per micro-batch:

```
layer<NN>.stage1_f, layer<NN>.stage1_b      # attention + routing
layer<NN>.stage2_f, layer<NN>.stage2_b      # all-to-all DISPATCH
layer<NN>.stage3_f, layer<NN>.stage3_b      # expert MLP
layer<NN>.stage3_w                           # delayed weight-grad
layer<NN>.stage4_f, layer<NN>.stage4_b      # all-to-all COMBINE
layer<NN>.stage5_f, layer<NN>.stage5_b      # weighted aggregate + residual
```

There are also fused ranges at stage boundaries like `layer<NN>_stage5_b_layer<MM>_stage1_b` (DualPipeV merges adjacent backward stages across layers). The helper `stage_of(nvtx_name)` in `scripts/common.py` canonicalizes both plain and fused names to a `stage<N>_<f|b|w>` key; use it instead of writing your own regex. For fused ranges it returns the **last** stage marker — so a `stage5_*_stage1_*` fusion is attributed to `stage1_*`, the comm-launching side (matters once CP ring attention adds stage1 comm).

**Direction (`_f` vs `_b`) maps to the corresponding op in that direction:**

- `stage2_f` = dispatch forward (all-to-all sending tokens to experts on forward path)
- `stage2_b` = dispatch backward (all-to-all on backward grad path — itself a collective)
- `stage4_f` = combine forward (all-to-all gathering expert outputs on forward path)
- `stage4_b` = combine backward (all-to-all on backward grad path)

So every `_b` stage emits a real NCCL collective; backward is not just "compute backward" — it carries its own all-to-all on the dispatch/combine streams.

**Use these to label stream purpose** — see [stream-purpose-labeling](#stream-purpose-labeling).

**CPU NVTX vs async kernel execution time:** NVTX is recorded on the CPU thread; a kernel launched on an async stream may execute on the GPU *after* the CPU has moved past the NVTX range that scheduled it. To find which NVTX range scheduled a kernel, use the kernel's **`launch_start`** (CPU-side `cudaLaunchKernel` time, returned by `kernels_in_window`), not its `start` (GPU execution time). The script helpers handle this automatically.

## Stream classification (compute vs communication)

Verified empirically across all 8 ranks in PP2-EP4-CP1: per rank there are 6 pure-comm streams + 2-3 compute streams (one of which has both compute and metadata NCCL).

**Per-kernel rule**: a kernel is communication if its short name (from `StringIds`) starts with `nccl`, otherwise compute. **Per-stream rule**: a stream is a compute stream if any of its kernels is non-NCCL; otherwise it's a comm stream (NCCL only). NCCL kernels on a compute stream are intentional in-order barriers (metadata exchanges) — they cannot overlap with compute on the same stream and should be reported as "barrier" time, never as "exposed comm". Implemented in `common.streams_in_window()`.

## Communication stream purpose

A stream's kind (compute/comm) doesn't tell you its role. For comm streams, use NVTX context for the purpose:

1. Pull every kernel on the comm stream in the window via `common.kernels_in_window`.
2. For each kernel, look up the innermost enclosing NVTX range at its **CPU `launch_start`** (not GPU start) via `common.innermost_nvtx`. The setup label and NCCL-emitted NVTX are filtered out.
3. Categorize each enclosing range via `classify_streams.PURPOSE_MARKERS` substring matches. **Every kernel must agree on the same category** — otherwise the label is `mixed`. Implemented in `classify_streams.classify_stream()`.

| Enclosing-NVTX needles | Purpose |
|---|---|
| `stage2_f`, `stage2_b`, `stage4_f`, `stage4_b` | **ep_a2a** (EP dispatch / combine) |
| `stage1_f`, `stage1_b` | **cp_ring** (CP ring attention send/recv when cp_size > 1) |
| `pipeline send/recv` (explicit DualPipeV wrapper) | **pp_p2p** (pipeline send/recv from `batch_isend_irecv`) |
| no match / no enclosing NVTX | **unknown** |
| disagreement across kernels | **mixed** |

The substring matching naturally handles DualPipeV's fused stage5+stage1 ranges (e.g., `layer05_stage5_f_layer06_stage1_f`) — only the stage1 portion launches comm, so the `stage1_*` needle catches them correctly.

Compute streams aren't purpose-labelled: PithTrain shares one stream across all stages, so any per-kernel stage label would just reflect sampling noise across attn / mlp / aggregate kernels.

## Common pitfalls

- **Don't classify by kernel name alone.** `ncclDevKernel_SendRecv` is used by NCCL for both EP all-to-all (implemented internally as point-to-point) AND pipeline-parallel P2P. The difference is stream identity, which you derive from NVTX context.
- **Don't infer "this kernel is real data movement" from its duration.** Long NCCL kernels can be either real work or straggler waits — disambiguate by comparing the kernel's duration across ranks (the straggler is the one ahead/behind the cluster).
- **Don't sum overlap across the whole step.** Warmup, drain, and optimizer steps dilute the signal. Always use the median-chunk window.
- **Don't ignore the barrier comm.** NCCL on the compute stream is non-zero (~200-300 ms per rank in a representative capture). It's structurally serial with compute and should be reported separately from overlap-eligible comm.
