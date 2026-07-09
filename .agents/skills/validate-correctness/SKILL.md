---
name: validate-correctness
description: Validates that code changes do not break training correctness by comparing per-step loss curves between a base branch and the current feature branch. Use when user asks to "validate correctness", "check if changes break training", "compare loss curves", "run a regression test", or "verify my changes are correct". The user specifies which model to validate and at which parallelism mesh (PP/EP/CP) — do not infer this from git diff.
---

# Validate Correctness

Runs the same short training on the base branch and the feature branch, then diffs per-step metrics. Correctness (cross-entropy, load-balance) is gated by tolerance; performance (step-time, tok/s, memory) is reported but not gated.

The user picks the **model** and the **mesh** — e.g. "validate deepseek-v2-lite at pp=2 ep=2". If vague, ask.

## Prerequisites

- Activate `.venv` in the repo root: `source .venv/bin/activate`.
- Benchmark inputs for the target model — run **setup-benchmark-inputs** first if `workspace/checkpoints/<model>/torch-dcp/step-00000000` is missing.
- `world_size >= PP * CP * EP` with `DP >= 1`.

## Step 1: Worktree for the base branch

Base is usually `main` — confirm with the user. Multi-node needs the worktree on a shared FS (`$HOME`, not `/tmp`).

```bash
BASE_BRANCH=main
REPOROOT=$(git rev-parse --show-toplevel)
WORKTREE=$(mktemp -d -p $HOME pithtrain-base.XXXX)
git worktree add $WORKTREE $BASE_BRANCH
ln -sfn $REPOROOT/.venv $WORKTREE/.venv
ln -sfn $REPOROOT/workspace $WORKTREE/workspace
mkdir -p $REPOROOT/logging
```

## Step 2: Run both branches

Under SLURM, wrap with `srun` (see **launch-with-slurm**). Log paths must differ — `workspace/` is shared between the two runs.

```bash
cd $WORKTREE && srun -N <nodes> -W 0 -o logging/validate-<model>-base.log .agents/skills/validate-correctness/scripts/launch_validate.sh --model <model> --pipeline-parallel-size <PP> --expert-parallel-size <EP>
cd $REPOROOT && srun -N <nodes> -W 0 -o logging/validate-<model>-feat.log .agents/skills/validate-correctness/scripts/launch_validate.sh --model <model> --pipeline-parallel-size <PP> --expert-parallel-size <EP>
```

Optional flags: `--context-parallel-size` (default 1), `--sequence-length` (default 2048), `--max-steps` (default 25). Global batch size is fixed at 1024.

## Step 3: Compare

```bash
python3 .agents/skills/validate-correctness/scripts/compare.py logging/validate-<model>-base.log logging/validate-<model>-feat.log
```

Tune `--tolerance` (default `5e-3`) if FP8/flash-attention non-determinism causes spurious misses. Exit 0 = correctness PASS, 1 = FAIL.

## Step 4: Clean up

```bash
git worktree remove $WORKTREE
```
