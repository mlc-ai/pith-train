---
name: validate-correctness
description: Validates that code changes do not break training correctness by comparing loss deltas against a base-vs-base run-to-run envelope. Use when user asks to "validate correctness", "check if changes break training", "compare loss curves", "run a regression test", or "verify my changes are correct". The user specifies which model to validate and at which parallelism mesh (PP/EP/CP) — do not infer this from git diff.
---

# Validate Correctness

Runs the same short training three times — the base branch twice, the feature branch once — and judges the base-vs-feature difference against the base-vs-base difference.

The user picks the **model** and the **mesh**, e.g. "validate deepseek-v2-lite at pp=2 ep=2". If vague, ask. Throughout, `<mesh>` is a compact tag like `pp2ep8cp1`, and the three runs are `base0`, `base1` and `feat0`.

## Method

Two runs of identical code do not agree past step 1. FlashAttention's backward is not reproducible even under `torch.use_deterministic_algorithms(True)`, which reaches neither it nor the Triton kernels; measured on this codebase the forward is bit-reproducible while gradient norms differ ~8% between two identical runs.

So a base-vs-feature delta means nothing on its own. `base0` and `base1` are identical code, so their difference is the run-to-run floor, and the feature difference is read as a ratio against it. A fixed tolerance cannot substitute: measured drift spans 3.6x across meshes, and `--tolerance 5e-3` both rejected a change that four meshes later showed to be indistinguishable and rejected identical code on two of four meshes.

## Prerequisites

- Activate `.venv` in the repo root: `source .venv/bin/activate`.
- A tokenized corpus for the model — run **setup-benchmark-inputs** if `workspace/datasets/dclm-baseline/toktxt/<tokenizer>` is missing. No checkpoint is needed; runs start from fresh weights.
- `world_size >= PP * CP * EP` with `DP >= 1`.
- **A clean tree.** The arms are switched with `git checkout`, so commit the feature work first — pushed or not. Uncommitted changes either follow you onto the base branch or block the checkout.

Runs must be **strictly sequential**, since one tree serves both arms. Never queue all three and let the scheduler interleave them: a checkout landing while a run is pending or in flight silently executes the wrong branch. Each run must finish before the next starts.

## Step 1: Create the three run files

Copy the template to `base0`, fill in every `<placeholder>`, then copy that filled file to `base1` and `feat0`. Filling once and copying afterwards makes the three identical by construction rather than by discipline. An unfilled copy is a syntax error, since the mesh sizes are unquoted; the quoted ones parse, so a forgotten `<wandb-entity>` surfaces later at `wandb.init` instead.

```bash
W=$(git rev-parse --show-toplevel)/workspace
cp .agents/skills/validate-correctness/templates/validate.py $W/<model>-<mesh>-base0.py
```

Fill the placeholders in that file, then:

```bash
cp $W/<model>-<mesh>-base0.py $W/<model>-<mesh>-base1.py
cp $W/<model>-<mesh>-base0.py $W/<model>-<mesh>-feat0.py
```

Each run takes its wandb name from its own filename, so nothing inside the three files differs. Confirm that, because it is the whole basis of the comparison:

```bash
diff $W/<model>-<mesh>-base0.py $W/<model>-<mesh>-base1.py
diff $W/<model>-<mesh>-base0.py $W/<model>-<mesh>-feat0.py
```

Both must print nothing. `workspace/` and `logging/` are gitignored, so these files and the logs survive every checkout below.

## Step 2: Run the base arm twice

```bash
FEATURE=$(git branch --show-current)
git checkout main
```

Then, on a single machine with no scheduler:

```bash
L=.agents/skills/validate-correctness/scripts/launch.sh
G=logging/<model>-<mesh>

bash $L $W/<model>-<mesh>-base0.py 2>&1 | tee $G-base0.log
bash $L $W/<model>-<mesh>-base1.py 2>&1 | tee $G-base1.log
```

Under SLURM, wrap each in `srun` — see **launch-with-slurm** for the flags that matter:

```bash
srun -N <nodes> -W 0 -o $G-base0.log bash $L $W/<model>-<mesh>-base0.py
srun -N <nodes> -W 0 -o $G-base1.log bash $L $W/<model>-<mesh>-base1.py
```

## Step 3: Run the feature arm

```bash
git checkout $FEATURE
bash $L $W/<model>-<mesh>-feat0.py 2>&1 | tee $G-feat0.log
```

## Step 4: Compare

```bash
python3 .agents/skills/validate-correctness/scripts/compare.py $G-base0.log $G-base1.log $G-feat0.log
```

The verdict is the ratio of mean |delta| for base-vs-feature over base-vs-base:

- **below 3x** — PASS, indistinguishable from run-to-run drift.
- **3x to 5x** — INVESTIGATE. Add a `base2` run: one envelope is a point estimate, and a 2.92x reading on one mesh was contradicted by 0.25x, 0.59x and 1.14x on three others.
- **above 5x** — FAIL.

The step-1 row is reported, not gated. It precedes any optimizer update, so its floor is normally zero and its signal is exactly the numerical difference the change makes to the forward — nonzero means the forward moved, which is expected for a reordered reduction or a swapped kernel and unexpected otherwise. Judge it yourself; a real forward regression also shows up in the 32-step ratio.

## Constraints

`max_steps * global_batch_size` must not exceed the corpus sample count, which scales inversely with sequence length — one tokenized DCLM shard yields ~18,000 samples at 4096 and roughly twice that at 2048. A run that does not fit raises at startup, but only after the JIT warmup.

`sequence_length % (2 * cp_size) == 0` for the zigzag split, and `global_batch / (dp * ep) >= 2 * pp`.

Keep sequence length and global batch identical across meshes, so meshes stay comparable to each other and not only within themselves.

## The fixture

`templates/validate.py` sets 32 steps and a linear warmup from 1e-6 to 1e-5 with no decay. Change these before any run, or not at all: runs are only comparable if they agree on them.

Warmup is load-bearing rather than cosmetic. From fresh weights a constant LR spikes hard in the first steps — on qwen3-30b-a3b at pp=2 ep=8, 12.33 to 19.90 with a pre-clip gradient norm of 1388 at 1e-4 — and lowering the LR tenfold barely helps, because AdamW's second moment is near zero early so the effective step `lr/(sqrt(v)+eps)` saturates whatever `lr` is. With warmup the same run descends monotonically and peaks at a gradient norm of 49, which is what makes a ~0.01 envelope measurable at all.

Correctness uses real routing, so the template leaves `training.benchmark = False`. Never turn it on here: force-balanced routing overwrites the router's top-k with a round-robin over token index, so the routing path goes untested and `load-balance-loss` pins at exactly 1.000000 — a gate that can never fail.

## Performance

A separate pass. Set `training.benchmark = True` and run at least 3 alternating reps per branch.

Compare run medians, not steps. Within a run step times are near-duplicates, while two runs of identical code sit at different offsets — measured here, ~0.2s of within-run spread against up to 0.5s between runs. Absolute numbers will differ elsewhere; what carries over is that the between-run offset dominates the within-run spread, so pooling steps manufactures significance. That offset is fixed at process launch and persisted across runs on matched hardware with no thermal ramp. Discard the first run on freshly acquired or long-idle hardware; it was the single outlier in a 10-run set here.

Resolving power follows from the measured run-to-run spread, not from the rep count alone — at 5 reps per branch it came to about 1.2% here. Compute it from your own base runs; anything below it is invisible end to end and needs a microbenchmark.

## Multi-node notes

If all three runs are on one machine this section does not apply.

Prefer all three on the same nodes, but do not require it. If they differ, the base-vs-base envelope absorbs node variation too, which makes the test more conservative: a within-envelope verdict stays trustworthy, and only an outside-envelope verdict needs a same-node re-run before attributing it to the code.

The three runs are independent, so one that fails can be re-run on its own and the others stay valid — check out the matching branch first. Re-running the whole set instead makes it only as durable as its unluckiest run.
