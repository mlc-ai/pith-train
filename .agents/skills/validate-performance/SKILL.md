---
name: validate-performance
description: Measures the throughput difference between two branches in benchmark mode (synthetic data, measured routing imbalance). Use when the user asks to "benchmark the performance", "measure throughput", "check for a slowdown", "check throughput did not regress", or "compare step time". The user specifies the model and the mesh; sequence length defaults to 2048 and dtype to bf16 — do not infer any of it from git diff.
---

# Validate Performance

Runs the same short training on the base branch and on the feature branch in benchmark mode, then reports the step-time difference. It measures; it does not gate. For loss correctness see **validate-correctness**.

The user picks the **model** and the **mesh**, e.g. "benchmark qwen3-30b-a3b at pp=2 ep=8". Sequence length defaults to 2048 and dtype to bf16; take either from the user when given. If the model or the mesh is vague, ask.

Runs live under `<model>` as a directory, named by a `<tag>` that spells out every dimension: `pp2-dp1-cp1-ep8-seq2048-bf16`. Include the defaults, so a log identifies itself. DP is not a config field — it is inferred from the world size and appears in the tag for identification only.

The wandb run is grouped as `performance/<model>`, so several meshes for one model land in one group and stay directly comparable, while the correctness pass groups separately and can never be mistaken for it.

## Prerequisites

- Activate `.venv` in the repo root: `source .venv/bin/activate`.
- No corpus and no checkpoint: `training.benchmark` trains on synthetic tokens.
- `world_size % PP == 0` with `CP` and `EP` each dividing `world_size / PP`, `sequence_length % (2 * cp_size) == 0`, and `global_batch / dp >= 2 * pp` (where `dp = world_size / (pp * cp)`).
- Keep sequence length, dtype and global batch identical across the two arms, and across meshes you intend to compare to each other.
- **A clean tree.** The arms are switched with `git checkout`, so commit the feature work first.

The two runs must be **strictly sequential**. One tree serves both arms, so a checkout landing while a run is pending or in flight silently benchmarks the wrong branch.

`<moe-load-balance-type>` differs per model:

| `<model>` | `<moe-load-balance-type>` |
|---|---|
| `qwen3-30b-a3b` | `global-batch` |
| `qwen3.5-35b-a3b` | `global-batch` |
| `deepseek-v2-lite` | `sequence` |
| `gpt-oss-20b`, `gpt-oss-120b` | `global-batch` |

## Step 1: Create the two run files

```bash
R=$(git rev-parse --show-toplevel)
W=$R/workspace/validate-performance/<model>
mkdir -p $W $R/logging/validate-performance/<model>
cp $R/.agents/skills/validate-performance/templates/validate.py $W/<tag>-base.py
cp $R/.agents/skills/validate-performance/scripts/launch.sh $W/launch.sh
git rev-parse --abbrev-ref HEAD > $W/FEATURE
```

The launcher is copied into `workspace/` because the checkouts below switch branches, and `.agents/` is tracked.

The template carries the defaults as real values, so only a non-default request needs editing: `training.sequence_length = 2048` and `training.fp8 = False`. The `<tag>` suffixes name them — `-fp8` is `training.fp8 = True`, and `fp8` is the only dtype knob there is.

Fill every `<placeholder>` in that file, then copy it so the two arms are identical by construction:

```bash
cp $W/<tag>-base.py $W/<tag>-feat.py
diff $W/<tag>-base.py $W/<tag>-feat.py
```

The diff must print nothing. Each run takes its wandb name from its own filename, so nothing inside the two files differs. `workspace/` and `*.log` are both gitignored, so the run files and the logs survive every checkout below without ever making the tree dirty.

## Step 2: Run both arms

Each fence re-derives its own paths, because shell variables do not survive between commands.

```bash
R=$(git rev-parse --show-toplevel); W=$R/workspace/validate-performance/<model>
G=$R/logging/validate-performance/<model>/<tag>
git checkout main
bash $W/launch.sh $W/<tag>-base.py 2>&1 | tee $G-base.log
```

```bash
R=$(git rev-parse --show-toplevel); W=$R/workspace/validate-performance/<model>
G=$R/logging/validate-performance/<model>/<tag>
FEATURE=$(cat $W/FEATURE)
git checkout $FEATURE
bash $W/launch.sh $W/<tag>-feat.py 2>&1 | tee $G-feat.log
```

Under SLURM, wrap each `bash $W/launch.sh …` in `srun -N <nodes> -W 0 -o <log>` — see **launch-with-slurm**.

## Step 3: Report

```bash
R=$(git rev-parse --show-toplevel); G=$R/logging/validate-performance/<model>/<tag>
python3 $R/.agents/skills/validate-performance/scripts/compare.py $G-base.log $G-feat.log
```

It prints median step time, tokens per second and peak memory for each arm, plus the percentage difference. Quote those numbers; do not convert them into a pass or fail.

## Why the fixture looks like this

`templates/validate.py` sets `training.benchmark = True`: synthetic tokens instead of a corpus, and the router's top-k replaced by the routing imbalance measured in a real training run (`pithtrain.modules.imbalance`), seeded by layer and step so both branches route alike. It also makes the loss meaningless, which is why correctness lives in a separate skill.

`training.max_steps = 8`, because step time settles within a few steps once compilation and kernel tuning are done; the routing draws vary a little from step to step, the same on both branches.

`compare.py` drops step 1: it carries the JIT warmup and the `torch.compile` trace, which dwarf the steady-state step time.

It compares medians of whole runs and never pools individual steps. Within a run step times are near-duplicates, while two runs of identical code sit at different offsets, so pooling counts correlated measurements as independent and reports significance for that offset.
