# Production fixed-topology strict resume: implementation and validation boundary

> Historical implementation/evidence snapshot. See
> [current Omni acceptance entry points and status](OMNI_ACCEPTANCE_STATUS.md)
> for the integrated implementation and Modal CLI workflow. Old Flame/Slurm
> package, authorization and queue statements below describe their original
> snapshot, not the current execution. Historical NOT_RUN/PASS results are
> not automatically inherited by a newer commit.

This branch extends frozen `86475821dc59e6d2f6c9f5f3d23ebb8056a8fd7f` in
`/home/kldxbjz/code/pith-train-week3-distributed-resume`, branch
`feat/omni-distributed-strict-resume`. It does not modify that frozen branch,
the first Flame package, FA4, the model, or the real PR100 loader.

## Implemented production path

Set `cfg.training.strict_resume = True`. Production `pretrain_lm.train_step`
passes this option to the existing public `save_checkpoint`; `launch` passes
it to public `load_checkpoint`. `find_checkpoint` recognizes the final strict
completion marker. Existing default behavior and weight initialization remain
unchanged; loading a strict checkpoint without the strict flag is rejected.

The implementation supports **same topology, same code/config, AdamW with
StepLR/LambdaLR and PR100 prepared data**. It operates on actual rank-local
DTensor shards; no `full_tensor()` or model all-gather is used for saving.
Model weights use the existing canonical expert mapping and public
localization on restore. Optimizer parameter membership, its local shards,
scheduler, committed data cursor and Python/NumPy/CPU/CUDA RNG remain per rank.
No dummy optimizer step initializes a fresh live optimizer during preflight.

Strict steps are stored under `root/torch-dcp/XXXXXXXX/` as `rank-NNNNN.pt`
files plus `strict-manifest.json`. Each rank stages CPU tensor payloads and
writes a temporary file before rename; the completion manifest is published
only after every rank finishes and reports a checksum. Incomplete saves are
not discovered as completed checkpoints.

All ranks agree on step, data and numerical contract. Before applying any
state they verify the complete file set/checksums, world size, both parallel
meshes, rank coordinates, DTensor placements/global and local shapes/dtypes,
all model keys, exact optimizer membership and moments, update counters,
scheduler state, committed cursor/fingerprint and every RNG component.
Local validation errors are exchanged so all participating ranks reject.
Recoverable application errors trigger rollback on **all** ranks, including
those that already applied the checkpoint. No rank proceeds into training
after a peer reports failure. Process death, CUDA/NCCL failure or transport
loss still terminates the job or reaches the configured collective timeout;
successful in-process rollback is not promised after a broken communicator.

The common `rewrap_dtensor_experts` now supplies the original global shape
and stride to `DTensor.from_local`, for both weights and optimizer moments.
Previously inferring global shape from an uneven local shard could produce
the wrong global expert count; the CPU test uses three experts split 2/1.

## CPU evidence and what it means

`tests/test_distributed_strict_checkpoint.py` launches real two-process Gloo
groups and real DTensors, with separate producer and restore processes. It
checks DP sharding, PP stage names and EP expert ownership layouts. A small
CPU model and explicitly labeled cursor fixture isolate the checkpoint
protocol; they are **not** a native model or real-loader acceptance substitute.

Cases include exact state and next-update equality, missing rank/completion,
checksum truncation, missing/unexpected model keys, shape/dtype/topology errors,
optimizer order, missing RNG, wrong cursor and bool-valued headers/manifests;
every rank must reject without changing its live state. Additional cases
cover one-rank apply failure with rollback, an uncommitted cursor and missing
optimizer state before save. Existing lifecycle tests now exercise strict
option propagation through both production save and launch/load paths.
Existing converter and single-process strict negative tests remain intact.

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. HF_HUB_OFFLINE=1 python -m pytest \
  tests/test_distributed_strict_checkpoint.py tests/test_pretrain_data_lifecycle.py \
  tests/test_qwen3_omni_checkpoint.py tests/test_strict_checkpoint.py \
  tests/test_omni_model_registration.py -q
```

## Prepared real multi-GPU acceptance — not executed here

`tests/acceptance/omni_distributed_resume.py` uses the actual Omni config,
production `setup_dataset`, `setup_training`, DualPipeV/FSDP and `train_step`.
It computes real native forward/backward, gradient scaling/clipping, optimizer
and scheduler updates. A transparent auditor records the returned real
microbatches without changing them. No SDPA replacement, FA4 patch, synthetic
gradient or fake loader is used in this entrance.

Fixed config: text only, V152064/hidden128/four layers/four experts, BF16
compute, S32, GB8, MB1, CP1; AdamW lr1e-5, warmup from1e-6 for 25% of three
steps. Producer saves step2 through the production automatic save branch,
records the post-save boundary and continues step3. A **new torchrun** loads
step2 into zeroed fresh parameters and empty optimizer state, verifies every
rank's exact boundary, then compares the exact next batch and full state after
the real third update. Restore writes subsequent checkpoints to a new root.

For each `(WORLD,PP,EP)` use a different output directory:
`(1,1,1)`, `(2,1,1)`, `(2,2,1)`, `(2,1,2)`, `(4,2,2)`.
World1 is only a baseline, not evidence of physical sharding. GB8 does not
satisfy the current pipeline microbatch requirement for world8/PP2; the CLI
rejects that setting instead of silently changing the workload.

```bash
# CPU-only preparation; official processor/config must already be cached.
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=. python \
  tests/acceptance/omni_distributed_resume.py --mode prepare \
  --dataset /real/omni-training-small --output /new/evidence \
  --world-size 4 --pp 2 --ep 2

# Only by the coordinator inside its existing bounded allocation (e.g. Modal):
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  timeout --signal=TERM --kill-after=15s 180s python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 \
  tests/acceptance/omni_distributed_resume.py --mode produce \
  --dataset /real/omni-training-small --output /new/evidence \
  --world-size 4 --pp 2 --ep 2
# Repeat as a separate command/process with --mode restore.
# Then run separate --mode negative commands with --negative-case:
# missing-rank, missing-cuda-rng, shape, topology, optimizer-order.
# Each case clones only the step2 checkpoint into a new evidence subdirectory.
```

Require zero exit code, reports from **every** rank and exact matches. Retain
the first failing run. Native FA4 nondeterminism may break the exact next
update criterion; that remains a failure requiring diagnosis, not an automatic
tolerance change. The original native/HF logit/CE/gradient gates are unchanged.

## Concrete remaining gaps

- No H100/real FSDP/NCCL/PP/EP execution has run for this branch. Its real
  multi-GPU acceptance is unverified; do not report distributed Week3 acceptance
  complete from the CPU protocol tests.
- CUDA RNG failure cases, actual FSDP load hooks/collective behavior, GPU OOM
  rollback and PP2EP2 state/update consistency need target execution. The GPU
  entrance includes positive continuation and five separately selectable
  one-rank GPU negative cases; none of these GPU modes has been run.
- This first implementation intentionally rejects changed topology, Muon,
  HSDP replicas>1 and CP>1. Those remain unimplemented; the current supported
  scope and native acceptance are HSDP1/CP1 with PP/EP/FSDP layouts.
- The strict container is a new **resume format**, not ordinary globally
  reshardable DCP. The standalone HF converter does not read this container
  directly at this document's frozen baseline. The follow-up in
  [WEEK3_STRICT_CONVERTER.md](WEEK3_STRICT_CONVERTER.md) implements direct
  export plus strict HF resume companions; ordinary DCP behavior is unchanged.
- Rank-local CPU staging and rollback copies increase host RAM and storage
  usage. Only tiny verification is in scope; no 30B payload or numeric run.
- Three-step continuation is not the repository's 32-step base0/base1/feature
  noise-envelope regression. That full training regression remains outstanding.

This document's original publication pause was part of its historical handoff.
The current integrated change is prepared for a separate Draft under the
coordinator; publication does not imply GPU acceptance. These CLI examples
do not allocate resources or merge any branch.
