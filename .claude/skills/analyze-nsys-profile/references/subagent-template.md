# Sub-agent prompt template

When the primary analysis surfaces K interesting NCCL kernels (e.g. from `compute_overlap.py --exposed-top N`), each one can be investigated independently. Dispatch K sub-agents concurrently using the template below — fill in the bracketed fields from the primary analysis output.

```
You are investigating ONE NCCL kernel from a PithTrain nsys profile to
determine whether its duration reflects real data movement or straggler wait.

Trace SQLite path: [PATH]

Kernel of interest:
- on global rank [R]
- on stream id [S]
- starts at [T] ns (relative to capture origin)
- duration [D] ns
- enclosing NVTX range: [NVTX_NAME]

Procedure:
1. Run:
   python .claude/skills/analyze-nsys-profile/scripts/compare_kernel.py [PATH] \
       --rank [R] --kernel-start-ns [T]
2. Read the `verdict` field and the per-rank `partners` table.
3. Identify the longest-duration partner ranks (waiters) and the shortest
   (likely the straggler).
4. If the verdict is STRAGGLER, briefly explain which rank is slow and why
   that might be (uneven layer assignment? heavier compute upstream?). Use
   `compute_overlap.py --rank <slow rank>` for a quick sanity check.

Output: plain prose, under 200 words. Include the verdict, the slowest /
fastest partner rank, and any recommendation (no action needed if bubble;
point to the slow rank if imbalance; flag if real heavy comm).
```

## When to dispatch

- 3+ exposed kernels with similar `enclosing_nvtx` — investigate the top 3 in parallel.
- A single outlier with much higher exposed time than its peers — investigate that one.
- Whole-rank imbalance (one rank has 2x exposed comm of others) — investigate that rank's top 3 kernels.

## When NOT to dispatch

- Total exposed comm < a millisecond — not worth the sub-agent latency.
- PP P2P stream with universally 0 % overlap — that is the pipeline bubble, expected.
- Already-known straggler patterns — no new information.
