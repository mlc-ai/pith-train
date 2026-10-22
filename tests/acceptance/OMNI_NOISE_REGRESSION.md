# Locked Week3 strict/HF bridge noise regression

> Historical implementation/evidence snapshot. See
> [current Omni acceptance entry points and status](OMNI_ACCEPTANCE_STATUS.md)
> for the integrated implementation and Modal CLI workflow. Old Flame/Slurm
> package, authorization and queue statements below describe their original
> snapshot, not the current execution. Historical NOT_RUN/PASS results are
> not automatically inherited by a newer commit.

This supersedes the normal-training-only harness at 1e21492. That version set
save_interval=None and strict_resume=False, so it did not execute the strict or
converter increment. Its 45 CPU tests were valid structural tests, not evidence
that the missing paths had run. Neither version has a GPU numerical result yet.

## Exact comparison

- base0/base1 production: **a0b14f7abbf4f84c7a9fbb76eb5a759a63f65010**.
- feat0 production: **d237e4c3d5ce9e38cf80e16a38dbd8f2210d0e53**.
- The separate harness revision is recorded in the delivery manifest. Its
  runtime checks reject any replacement of either production SHA.
- Both roots remain clean and read-only during this experiment. No git checkout,
  installation, upload, submission, retry or requeue happens in this driver.

The base is the already integrated PR100/Omni converter merge. The exact range
contains seven changed production files: checkpoint.py, distributed_checkpoint.py,
training.py, pretrain_lm.py, converter _core.py, strict_omni.py, and the Omni
converter CLI. The original ordinary converter predates this base. This range
must never be presented as the complete PR101, PR100, or initial Week3 converter
acceptance. `OMNI_NOISE_SCOPE.json` records the concrete diff and identical-file /
numerical-function audit; `production-range.diff` is in the local delivery.

## What is equal and what intentionally differs

All three arms use the exact existing tiny native Omni Thinker text config,
verified PR100 text bundle, first 256 consumed samples, seed1234, sequence32,
global batch8, microbatch1, AdamW, BF16, real routing, global-batch load balancing
at1e-3, and 32 updates with linear warmup from1e-6 to1e-5. No input/model weights
are downloaded. No SDPA adapter, forced-balanced routing or synthetic training
data is permitted. Model and data source files are byte-identical across the
range. The numerical train_step body is AST-identical after excluding only its
checkpoint block; optimizer, scheduler, FSDP, initialization and setup code are
also AST-identical. Candidate model_class_for_config additionally normalizes
config through the existing text resolver; that is part of the recorded diff.

| Property | base0 / base1 | feat0 |
| --- | --- | --- |
| Production code | locked baseline | locked candidate |
| Native optimizer updates | continuous1..32 | 1..16 then17..32 |
| Checkpoint controls | saving disabled; no restore | strict=True; save_interval16 |
| Midpoint | none | step16 strict -> thinker. fused HF -> strict container |
| Restart | none within the arm | fresh torchrun process, exact step16 restore |
| End | 256 committed samples | 256 committed samples; native final step32 save also executes |

Thus the **numerical workload and update path** match, but the feature's
checkpoint/restart intervention intentionally differs. It would be incorrect
to describe all configuration fields or execution paths as identical.

Every rank compares the complete step16 runtime digest before and after
restoration: canonical model tensors/dtypes, optimizer state, scheduler, data
cursor and Python/NumPy/CPU/CUDA RNG. The digest includes full array/tensor bytes,
not truncated repr output. The converter itself checks unchanged canonical
weight dtype/raw-byte hashes and the numerical resume contract. First/second
halves must contain exactly steps1..16 and17..32; the combined feature log must
contain exactly32. All arms must match initial weights, actual batch hashes,
model/data/settings and runtime. The feature cannot bypass the bridge or claim
a different process restored it by reusing the producer PID.

## Numerical gate and remaining independent gates

The five (WORLD,PP,EP) profiles remain (1,1,1), (2,1,1), (2,2,1), (2,1,2),
(4,2,2), with CP=HSDP=1. Each runs base0 -> base1 -> feat0 sequentially. Each
loss metric uses the unchanged project comparator's 32-step mean absolute
base0/base1 noise envelope and base0/feat0 signal: ratio<3 PASS, 3<=ratio<5
INVESTIGATE, ratio>=5 FAIL. Zero envelope requires exactly zero signal. Step1
forward difference is additionally reported without a separate gate. Missing,
nonfinite, duplicate, short or skipped evidence cannot pass. INVESTIGATE exits2,
and no extra base2 arm is scheduled automatically.

This experiment covers the positive long-horizon strict/HF/resume path and its
normal-training regression. It does not cover malformed/edited companion
negatives, uneven/empty expert shards (the GPU fixture has four experts), split
HF layout, automatic pretrain_lm.launch resume glue (it calls the native setup,
train_step and checkpoint APIs directly), full30B numerical execution, media,
CP>1, HSDP>1, Muon or changing topology. Those remain separate gates. Existing
CPU negative/shard tests and frozen job155302 are complementary evidence; a
PASS here cannot substitute for them or turn their pending/failed results into
PASS. No user PR review until all required gates actually pass.

## CPU preparation

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 "$python" \
  "$harness/tests/acceptance/omni_noise_regression.py" prepare \
  --base-repo "$locked_base" --feature-repo "$locked_candidate" \
  --dataset "$verified_data" --output "$new_evidence"
```

Both clean source SHAs are checked before output creation. CPU preparation
verifies real bundle integrity, processor/model vocabulary, 388 available
samples versus256 required, all five meshes and native configuration fields
using AST plus the shared builder. Existing production operator imports query
CUDA, so the CPU phase does not import or mock the GPU-bound training modules.
Native GPU imports and numerical execution remain NOT RUN until separately
scheduled. The same pinned offline runtime applies to every arm.

## One separately authorized allocation, at most4 GPU-hours

Request one node, **4 H100, 32 CPUs, 128 GiB host RAM, 01:00:00 walltime,
no requeue**, and64 GiB local scratch for checkpoint/companion artifacts. This
is a bounded proposal, not measured runtime or disk usage. Reuse the existing
verified data and seven offline processor/config files; the local incremental
bundle/manifest reports exact additional transfer bytes. Do not alter155302.

The intended coverage is all five profiles:15 logical arms,480 native updates,
20 fresh torchrun processes, and10 CPU converter invocations. Each logical arm
has a600-second cap (the feature cap includes both native halves and both CPU
conversions); all arms share3300 seconds. Termination allows at most15 seconds;
the outer wrapper is3330+15 seconds, within the one-hour allocation. CPU phases
still consume allocated GPU-hours. Because per-arm ceilings exceed the global
budget in sum, completion within4 GPU-hours is unmeasured: budget exhaustion
fails the experiment and leaves the rest NOT RUN, without reducing steps,
loosening gates or launching a second job.

```bash
export PITHTRAIN_RUN_OMNI_GPU_TESTS=1
srun --jobid="$SLURM_JOB_ID" -N1 -n1 -c32 --gres=gpu:4 -W 0 \
  --open-mode=truncate -o "$new_step_log" \
  bash "$harness/tests/acceptance/run_omni_noise_h100.sh" \
  "$python" "$new_evidence/plan.json"
```
