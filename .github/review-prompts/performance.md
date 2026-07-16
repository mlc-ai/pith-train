Reviewing ${GITHUB_REPOSITORY} PR #${GITHUB_EVENT_ISSUE_NUMBER} for PERFORMANCE only. The PR branch is already checked out in the current working directory.

### How to post

- Post each finding as its own inline comment with `mcp__github_inline_comment__create_inline_comment` (`confirmed: true`), anchored to the exact line(s) it concerns, so each becomes an independently resolvable thread.
- Open every comment body with a bold **Performance:** tag so its review dimension is clear at a glance.
- Reserve `gh pr comment` for at most a single one-line note: to report that nothing qualified, or to cite a finding whose line falls outside this PR's diff and so has no inline anchor. Never batch findings into a top-level comment.

### What to look for

PithTrain is a training framework whose whole point is throughput and memory efficiency: DualPipeV overlaps compute with communication, FP8 and fused kernels cut cost, and most hot code runs under torch.compile. Judge the change on whether it regresses training step time or peak memory. Real perf claims are established by measurement, not by reading the diff, so weigh the developer's evidence first.

First, gather the developer's evidence. Read the PR description and comments (`gh pr view`, including `--comments`) and any results the PR commits (step-time / tokens-per-second numbers, peak-memory figures, nsys traces, benchmark output, links to wandb runs). Judge whether it shows the change holds throughput and memory versus a baseline at a relevant configuration.

Then report:
- Where the evidence is missing, weak, or does not match the change. For a change to the hot path or a perf-sensitive mechanism (comm overlap, kernels, compiled regions) with no timing or memory number shown, flag that the evidence is insufficient and say what measurement would settle it.
- Concrete regressions the change introduces: collectives (all-to-all dispatch/combine, ring K/V exchange, FSDP gather/reduce) moved onto the compute stream or made synchronous, breaking overlap; device syncs on the hot path (`.item()`, `.cpu()`, `.tolist()`, python branching on tensor values, `torch.cuda.synchronize`, per-step/per-layer host-device transfers); torch.compile graph breaks or recompiles introduced in a compiled region (data-dependent shapes, unguarded python, `torch.compiler.disable`); redundant compute or allocations in the per-step / per-layer / per-micro-batch path.

Only raise an issue you can locate on a real hot path and explain the cost of, or a concrete gap in the evidence.