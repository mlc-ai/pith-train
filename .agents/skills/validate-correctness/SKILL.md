---
name: validate-correctness
description: Validates that code changes do not break training correctness by comparing loss deltas against a base-vs-base run-to-run envelope. Use when user asks to "validate correctness", "check if changes break training", "compare loss curves", "run a regression test", or "verify my changes are correct". For throughput use validate-performance instead. The user specifies which model to validate, at which parallelism mesh (PP/EP/CP), and at which sequence length — do not infer any of it from git diff.
---

# Validate Correctness

Runs the same short training three times — the base branch twice, the feature branch once — and judges the base-vs-feature difference against the base-vs-base difference.

The user picks the **model**, the **mesh** and the **sequence length**, e.g. "validate deepseek-v2-lite at pp=2 ep=2 seq=4096". Sequence length is filled per run, defaulting to 2048 when the user expresses no preference. Dtype is bf16 unless asked otherwise, and is the one dimension carried in the template as a real value. If the model or the mesh is vague, ask.

Runs live under `<model>` as a directory, named by a `<tag>` that spells out every dimension: `pp2-dp1-cp1-ep8-seq2048-bf16`. Include the defaults, so a log identifies itself. DP is not a config field — it is inferred from the world size and appears in the tag for identification only. The three runs are `base0`, `base1` and `feat0`, and the wandb group is `correctness/<model>`.

## Method

Two runs of identical code do not agree past step 1. FlashAttention's backward is not reproducible even under `torch.use_deterministic_algorithms(True)`, which reaches neither it nor the Triton kernels; measured on this codebase the forward is bit-reproducible while gradient norms differ ~8% between two identical runs.

So a base-vs-feature delta means nothing on its own. `base0` and `base1` are identical code, so their difference is the run-to-run floor, and the feature difference is read as a ratio against it. A fixed tolerance cannot substitute: measured drift spans 3.6x across meshes, and `--tolerance 5e-3` both rejected a change that four meshes later showed to be indistinguishable and rejected identical code on two of four meshes.

**The logged losses are rank 0's samples, not the global batch.** `train_step` reduces the cross-entropy over CP but not over DP, so the log prints the mean over the samples of DP rank 0. Both arms must therefore hand rank 0 the same samples: keep the mesh (PP, CP, EP, hence DP), `micro_batch_size` and `global_batch_size` identical. A change that alters any of them, e.g. `micro_batch_size` 1 against 2, moves rank 0 to a different subset of the same global batch, and the cross-entropy rows fail by construction (a measured 362x ratio, with a step-1 forward signal 35x its floor, from a change whose outputs are bit-identical per sample). The global `gradient-norm` row, reduced over every rank, is the comparison to read in that case. `load-balance-loss` also moves with `micro_batch_size` under `global-batch` balancing, whose statistics are gathered per micro-batch slot.

## Prerequisites

- Activate `.venv` in the repo root: `source .venv/bin/activate`.
- A tokenized corpus for the model — run **setup-benchmark-inputs** if `workspace/datasets/dclm-baseline/toktxt/<tokenizer>` is missing. No checkpoint is needed; runs start from fresh weights.
- `world_size % PP == 0` with `CP` and `EP` each dividing `world_size / PP`.
- **A clean tree.** The arms are switched with `git checkout`, so commit the feature work first — pushed or not. Uncommitted changes either follow you onto the base branch or block the checkout.

Runs must be **strictly sequential**, since one tree serves both arms. Never queue all three and let the scheduler interleave them: a checkout landing while a run is pending or in flight silently executes the wrong branch. Each run must finish before the next starts.

`<tokenizer>` is not `<model>`, and `<moe-load-balance-type>` differs per model:

| `<model>` | `<tokenizer>` | `<moe-load-balance-type>` |
|---|---|---|
| `qwen3-30b-a3b` | `qwen3` | `global-batch` |
| `qwen3.5-35b-a3b` | `qwen3.5` | `global-batch` |
| `deepseek-v2-lite` | `deepseek-v2` | `sequence` |
| `gpt-oss-20b`, `gpt-oss-120b` | `gpt-oss` | `global-batch` |

## Step 1: Create the three run files

Copy the template to `base0`, fill in every `<placeholder>`, then copy that filled file to `base1` and `feat0`. Filling once and copying afterwards makes the three identical by construction rather than by discipline. An unfilled copy fails unevenly, so scan the file before launching: the numeric fills — the mesh sizes, `<sequence-length>` and `<global-batch-size>` — are unquoted, so a leftover there is a `SyntaxError` on every rank, while the quoted ones — `<model>`, `<tokenizer>`, `<moe-load-balance-type>`, `<wandb-project>` — parse and surface far later, a forgotten `<wandb-project>` only at `wandb.init`.

```bash
R=$(git rev-parse --show-toplevel)
W=$R/workspace/validate-correctness/<model>
mkdir -p $W $R/logging/validate-correctness/<model>
cp $R/.agents/skills/validate-correctness/templates/validate.py $W/<tag>-base0.py
cp $R/.agents/skills/validate-correctness/scripts/launch.sh $W/launch.sh
git rev-parse --abbrev-ref HEAD > $W/FEATURE
```

The launcher is copied into `workspace/` because the checkouts below switch branches, and `.agents/` is tracked.

Fill the placeholders in that file, then:

```bash
cp $W/<tag>-base0.py $W/<tag>-base1.py
cp $W/<tag>-base0.py $W/<tag>-feat0.py
```

Each run takes its wandb name from its own filename, so nothing inside the three files differs. Confirm that, because it is the whole basis of the comparison:

```bash
diff $W/<tag>-base0.py $W/<tag>-base1.py
diff $W/<tag>-base0.py $W/<tag>-feat0.py
```

Both must print nothing. `workspace/` and `*.log` are both gitignored, so the run files and the logs survive every checkout below without ever making the tree dirty.

## Step 2: Run the base arm twice

Each fence re-derives its own paths, because shell variables do not survive between commands.

```bash
R=$(git rev-parse --show-toplevel); W=$R/workspace/validate-correctness/<model>
G=$R/logging/validate-correctness/<model>/<tag>
git checkout main
bash $W/launch.sh $W/<tag>-base0.py 2>&1 | tee $G-base0.log
bash $W/launch.sh $W/<tag>-base1.py 2>&1 | tee $G-base1.log
```

Under SLURM, wrap each in `srun` — see **launch-with-slurm** for the flags that matter:

```bash
srun -N <nodes> -W 0 -o $G-base0.log bash $W/launch.sh $W/<tag>-base0.py
srun -N <nodes> -W 0 -o $G-base1.log bash $W/launch.sh $W/<tag>-base1.py
```

## Step 3: Run the feature arm

```bash
R=$(git rev-parse --show-toplevel); W=$R/workspace/validate-correctness/<model>
G=$R/logging/validate-correctness/<model>/<tag>
FEATURE=$(cat $W/FEATURE)
git checkout $FEATURE
bash $W/launch.sh $W/<tag>-feat0.py 2>&1 | tee $G-feat0.log
```

## Step 4: Compare

```bash
R=$(git rev-parse --show-toplevel); G=$R/logging/validate-correctness/<model>/<tag>
python3 $R/.agents/skills/validate-correctness/scripts/compare.py $G-base0.log $G-base1.log $G-feat0.log
```

The verdict is the ratio of mean |delta| for base-vs-feature over base-vs-base:

- **below 3x** — PASS, indistinguishable from run-to-run drift.
- **3x to 5x** — INVESTIGATE. Add a `base2` run: one envelope is a point estimate, and a 2.92x reading on one mesh was contradicted by 0.25x, 0.59x and 1.14x on three others.
- **above 5x** — FAIL.

The step-1 row is reported, not gated. It precedes any optimizer update, so its floor is normally zero and its signal is exactly the numerical difference the change makes to the forward — nonzero means the forward moved, which is expected for a reordered reduction or a swapped kernel and unexpected otherwise. Judge it yourself; a real forward regression also shows up in the 32-step ratio.

## Constraints

`max_steps * global_batch_size` must not exceed the corpus sample count, which scales inversely with sequence length — one tokenized DCLM shard yields ~18,000 samples at 4096 and roughly twice that at 2048. A run that does not fit raises at startup, but only after the JIT warmup.

`sequence_length % (2 * cp_size) == 0` for the zigzag split, and `global_batch / dp >= 2 * pp` (where `dp = world_size / (pp * cp)`).

Keep sequence length and global batch identical across meshes, so meshes stay comparable to each other and not only within themselves.

## The fixture

`templates/validate.py` sets 32 steps and a linear warmup from 1e-6 to 1e-5 with no decay. Change these before any run, or not at all: runs are only comparable if they agree on them.

Warmup is load-bearing rather than cosmetic. From fresh weights a constant LR spikes hard in the first steps — on qwen3-30b-a3b at pp=2 ep=8, 12.33 to 19.90 with a pre-clip gradient norm of 1388 at 1e-4 — and lowering the LR tenfold barely helps, because AdamW's second moment is near zero early so the effective step `lr/(sqrt(v)+eps)` saturates whatever `lr` is. With warmup the same run descends monotonically and peaks at a gradient norm of 49, which is what makes a ~0.01 envelope measurable at all.

Correctness uses real routing, so the template leaves `training.benchmark = False`. Never turn it on here: benchmark mode trains on synthetic tokens and replays the router's top-k from drawn expert loads, so the routing path goes untested and the loss carries no signal. Throughput measurement wants it on, which is why it lives in **validate-performance**.

## Multi-node notes

If all three runs are on one machine this section does not apply.

Prefer all three on the same nodes, but do not require it. If they differ, the base-vs-base envelope absorbs node variation too, which makes the test more conservative: a within-envelope verdict stays trustworthy, and only an outside-envelope verdict needs a same-node re-run before attributing it to the code.

The three runs are independent, so one that fails can be re-run on its own and the others stay valid — check out the matching branch first. Re-running the whole set instead makes it only as durable as its unluckiest run.
