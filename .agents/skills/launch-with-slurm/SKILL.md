---
name: launch-with-slurm
description: Reference for launching jobs inside a SLURM allocation via srun (single-node or multi-node). Use whenever work needs to run on allocated compute — from direct user requests ("run on the cluster", "use my running job", "on my allocation", "launch on slurm", "train across N nodes", "dispatch the job") OR from within another skill's workflow (e.g., validate-correctness running validation on the allocation, add-new-model reaching pp=2/ep=2). Covers finding the target job ($SLURM_JOB_ID or squeue, including array-job ID resolution), reading the allocation with scontrol, the srun flags that matter (--jobid, -W 0, -N, -o, --open-mode, --nodelist, --overlap), and gotchas like GPU sharing with live steps and distributed-aware output redirection.
---

# Launch with SLURM

`srun --jobid=<jobid>` attaches a step to a running allocation regardless of where it is invoked — login node, `ssh`'d compute node, or inside the job itself. The scheduler draws nodes from the allocation pool and may pick a node other than the invoking one. Default to it over raw `torchrun`/`bash`: `srun` propagates env vars, handles distributed-aware I/O, and manages signals across ranks correctly — even on a single node with multiple GPUs. The `examples/*/launch.sh` scripts read `SLURM_NNODES`/`SLURM_NODEID`, which `srun` sets inside a step, so they work unchanged.

## Step 1: Find the Job ID

Two sources, one answer — the numeric JobId of the job (e.g. `2193449`):

- **`$SLURM_JOB_ID` is set** — you are inside the job, or in an `ssh` session on an allocated node that inherits the job environment. Use it as-is; it is already the numeric JobId.
- **Otherwise** — list the user's jobs and pick a RUNNING one: by name if the user named it, ask if several are ambiguous or none is RUNNING, skip PENDING (no compute yet):

```bash
squeue -u $USER -o "%.10i %.9T %.6D %.24N %.24j %.10L"
```

**Array jobs: resolve to the numeric JobId.** `squeue` reports elements as `<master>_<task>` (e.g. `2193448_1`), but `--jobid` wants the element's own numeric JobId — given `2193448_1`, it strips the `_1`, resolves to the array master, and fails with "Job is pending execution" when the master has pending elements:

```bash
scontrol show job 2193448_1 | grep -oP '^JobId=\K[0-9]+'    # → 2193449
```

Then read the allocation — don't guess, ask SLURM:

```bash
scontrol show job $JOBID
```

Key fields to extract:

| Field | Example | What it tells you |
|---|---|---|
| `AllocTRES` | `cpu=208,mem=1860368M,node=1,billing=208,gres/gpu=8` | Node count, GPUs per node, CPUs, memory |
| `NodeList` | `orchard-flame-5` or `orchard-flame-[3-6]` | Which hosts; on most clusters `ssh <name>` gives direct access |

For a quick remaining-time check, use `squeue` directly — it returns `D-HH:MM:SS` without needing to parse timestamps:

```bash
squeue -h -j $JOBID -o %L
```

Before launching anything long-running, compare this against the estimated runtime. If the budget is too tight, surface this to the user instead of launching and getting killed mid-run.

## Step 2: Build the srun Command

- **`--jobid=<jobid>`** — anchor the step to the allocation. Required when `$SLURM_JOB_ID` is unset or holds a different job; redundant but harmless when it already holds the target.
- **`-N <n>`** — number of nodes to dispatch to. In most training runs this matches PP, but the full parallelism plan and GPUs-per-node determine total nodes (e.g., PP=1 with EP=16 on 8-GPU nodes still needs 2). `-N1` borrows one node of a multi-node allocation for probes or single-node tests.
- **`-W 0`** — wait indefinitely for stragglers after the first task exits. The default behavior terminates remaining tasks shortly after the first one ends, which kills workers that are still cleanly shutting down. Always use `-W 0` for training and evaluation runs.
- **`-o <file>`** — stdout redirection. Use this instead of piping through `tee`. On multi-node, `tee`ing srun output collapses concurrent writes from all ranks. `-o` is distributed-aware — srun collects output from every rank into the single specified file, preserving the one-command-one-log abstraction. By convention, PithTrain runs log under `logging/<descriptive-name>.log`.
- **`--open-mode=append`** vs **`--open-mode=truncate`** — for resumed training, `append` preserves history across restarts. Use `truncate` for fresh runs where overwriting is intended.
- **`--nodelist=<hosts>`** — restrict dispatch to specific nodes. Useful for debugging at a smaller scale (e.g., 4 nodes allocated, but debug with 2 specific ones).
- **`--overlap`** — share the allocation's CPUs, memory, and GPUs with the job's other steps. Without it, a step reserves the whole allocation, so any later step silently pends until it finishes — pass `--overlap` whenever launching alongside a running step.
- **`--gres=gpu:<k>`** — request a subset of the job's GPUs for the step. By default a step sees every GPU the job holds on the node, so `torchrun --nproc-per-node=gpu` just works. Sharing means contention: probe `nvidia-smi` utilization before launching heavy work onto a job whose GPUs are already busy; if it's saturated, surface this to the user instead of piling on.

`srun` execs the command directly, not through a shell — invoke scripts as `bash <script>` or ensure the `+x` bit.

## References

- [srun options](https://slurm.schedmd.com/srun.html#SECTION_OPTIONS) — full list of flags
- [srun environment variables](https://slurm.schedmd.com/srun.html#SECTION_INPUT-ENVIRONMENT-VARIABLES) — `SLURM_*` variables available inside scripts launched by srun
