---
name: launch-with-slurm
description: Reference for launching jobs inside a SLURM allocation via srun (single-node or multi-node). Use whenever $SLURM_JOB_ID is set and work needs to run on the allocated compute — from direct user requests ("run on the cluster", "launch on slurm", "train across N nodes", "dispatch the job") OR from within another skill's workflow (e.g., validate-correctness running validation on the allocation, add-new-model reaching pp=2/ep=2). Covers how to read the allocation context from $SLURM_JOB_ID via scontrol, the srun flags that matter (-W 0, -N, -o, --open-mode, --nodelist), and gotchas like the executable bit requirement and distributed-aware output redirection.
---

# Launch with SLURM

Reference for dispatching work inside a SLURM allocation via `srun`. Most of the time the agent is invoked from inside an existing allocation — the compute is already reserved, and the job is to use it well. Applies to both single-node and multi-node allocations.

Default to `srun` over raw `torchrun`/`bash` whenever `$SLURM_JOB_ID` is set. `srun` propagates env vars, handles distributed-aware I/O, and manages signals across ranks correctly — even on a single node with multiple GPUs.

## Step 1: Read the Allocation

If `$SLURM_JOB_ID` is set, the compute environment is already encoded — don't guess, ask SLURM:

```bash
scontrol show job $SLURM_JOB_ID
```

Key fields to extract:

| Field | Example | What it tells you |
|---|---|---|
| `AllocTRES` | `cpu=208,mem=1860368M,node=1,billing=208,gres/gpu=8` | Node count, GPUs per node, CPUs, memory |
| `NodeList` | `orchard-flame-5` or `orchard-flame-[3-6]` | Which hosts; on most clusters `ssh <name>` gives direct access |

For a quick remaining-time check, use `squeue` directly — it returns `D-HH:MM:SS` without needing to parse timestamps:

```bash
squeue -h -j $SLURM_JOB_ID -o %L
```

Before launching anything long-running, compare this against the estimated runtime. If the budget is too tight, surface this to the user instead of launching and getting killed mid-run.

If `$SLURM_JOB_ID` is not set, you're not inside an allocation — fall back to direct `torchrun` and do not attempt to invoke `srun`.

## Step 2: Build the srun Command

### Flags that matter

- **`-N <n>`** — number of nodes to dispatch to. In most training runs this matches the pipeline-parallel degree (PP), but not always: the full parallelism plan and GPUs-per-node determine total nodes (e.g., PP=1 with EP=16 on 8-GPU nodes still needs 2 nodes).
- **`-W 0`** — wait indefinitely for stragglers after the first task exits. The default behavior terminates remaining tasks shortly after the first one ends, which kills workers that are still cleanly shutting down. Always use `-W 0` for training and evaluation runs.
- **`-o <file>`** — stdout redirection. Use this instead of piping through `tee`. On multi-node, `tee`ing srun output collapses concurrent writes from all ranks. `-o` is distributed-aware — srun collects output from every rank into the single specified file, preserving the one-command-one-log abstraction. By convention, PithTrain runs log under `logging/<descriptive-name>.log`.
- **`--open-mode=append`** vs **`--open-mode=truncate`** — for resumed training, `append` preserves history across restarts. Use `truncate` for fresh runs where overwriting is intended.
- **`--nodelist=<hosts>`** — restrict dispatch to specific nodes. Useful for debugging at a smaller scale (e.g., 4 nodes allocated, but debug with 2 specific ones).

## References

- [srun options](https://slurm.schedmd.com/srun.html#SECTION_OPTIONS) — full list of flags
- [srun environment variables](https://slurm.schedmd.com/srun.html#SECTION_INPUT-ENVIRONMENT-VARIABLES) — `SLURM_*` variables available inside scripts launched by srun
