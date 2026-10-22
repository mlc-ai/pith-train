# Strict resume CPU baseline and PP/EP integration boundary

The converter candidate remains `ace46ec37678909ad067867898fc20a7112ff0b6`
on `feat/omni-text-checkpoint`, with its earlier **67 CPU passes** and
**6 GPU passes using an explicit SDPA Math adapter / HF eager reference**.
This independent follow-up starts from that commit on `feat/omni-strict-resume`
in `/home/kldxbjz/code/pith-train-week3-resume`.

## Small independent implementation

`pithtrain/modules/strict_checkpoint.py` adds opt-in `StrictResumeState`,
`save_strict_checkpoint`, and `load_strict_checkpoint`. It reuses
`checkpoint.py` for model and optimizer canonical/localized mapping. No loader,
sampler, native model, attention operator, existing converter, or production
training entrypoint is modified.

Strict state requires model, optimizer, scheduler, the existing PR100
`Stateful` data object, completed update count/global batch size, optimizer and
scheduler classes, and Python/NumPy/PyTorch CPU RNG. CUDA RNG is required when
the model is on CUDA. Missing state is never treated as weight initialization.
The existing weight-only initialization path is retained and separately tested.

DCP metadata is checked for exact keys, tensor shapes and dtypes before payload
reads. Payloads load into a detached staging copy so semantic failures cannot
overwrite live parameters. Nested state schemas, scheduler counts, optimizer
group membership/order, data fingerprint, committed cursor and RNG validity are
checked before applying the checkpoint. Saving an uncommitted/wrong data boundary
or overwriting an existing checkpoint is rejected. The checkpoint still uses
`root/torch-dcp/<completed-step>` and `app.model` canonical names.
PyTorch's lazy optimizer-template initialization is isolated on a copied
model/optimizer graph; rejection cannot initialize a pristine live optimizer.

New strict checkpoints additionally contain `app.resume` with the continuation
contract and RNG state. Historical permissive checkpoints and converter outputs
without that contract must use explicit initialization; they do not silently
qualify as strict restart checkpoints. The new format is an opt-in baseline,
not a migration of existing checkpoints.

Current scope is one process, one ordinary CPU or CUDA device, PP=EP=CP=1.
Distributed process groups with more than one rank, non-unit mesh contexts,
and DTensor/FSDP models explicitly raise `NotImplementedError`. CUDA state
handling is implemented but **unrun in this follow-up**. The staging copy costs
additional model/optimizer memory; no full-checkpoint memory or numerical
acceptance is claimed. Private DCP flattening is reused to match the installed
PyTorch planner schema; this was tested on PyTorch 2.13.0+cu130.

## CPU evidence

Final combined run on 2026-10-08: **107 passed, 6 skipped, 85 warnings in
35.24 seconds**, exit 0. This is 67 unchanged converter CPU cases plus 40
recovery cases. The six skips are the unchanged, explicitly unscheduled GPU
tests. Warnings are expected single-process DCP notices. Ruff and
`git diff --check` passed. The original converter tree remained clean at
`ace46ec37678909ad067867898fc20a7112ff0b6` after verification.

`tests/test_strict_checkpoint.py` contains 40 tests:

- Two fresh-process comparisons: released split + AdamW, fused + momentum SGD.
  A real HF Thinker eager CPU model is initialized through HF files -> converter
  -> canonical DCP -> native-shaped checkpoint proxy. Each arm performs 3
  updates; the restart arm restores the boundary after update 2. Comparison is
  exact (`rtol=atol=0`) for the restored boundary, next sample IDs, tokens,
  labels, masks, loss, all model parameters, optimizer buffers, scheduler and
  Python/NumPy/PyTorch RNG. The CPU fixture includes a stochastic test loss
  input and uses the supported zero-attention-dropout model configuration.
- 27 corrupted-checkpoint cases reject missing components/RNG, scheduler
  truncation/extra entries, model missing/unexpected/shape/dtype errors,
  optimizer missing/shape/membership/order errors, wrong data fingerprint or
  cursor, wrong step/batch size/class and malformed RNG. These check that live
  model, optimizer, scheduler, cursor and RNG stay unchanged on failure.
- Checkpoint-boundary and occupied-output preservation checks, legacy
  weight-only compatibility, Python cached-Gaussian RNG restoration, and
  preservation of an uninitialized live optimizer when a checkpoint is rejected.
- Three explicit PP/EP/CP strict-resume rejection tests.
- Four CPU model/optimizer mapping tests for PP/EP dimensions (1,1), (1,2),
  (2,1), (2,2): individual global expert IDs, both local `module.N` prefixes,
  global layer IDs and AdamW slots round-trip exactly across synthetic rank
  ownership. These are **in-memory mapping tests**, with explicit fixture layer
  placement; they do not execute DualPipeV's partitioner, collectives, FSDP,
  native CUDA backward or a distributed optimizer update.

The test data object is explicitly a **test-only PR100 Stateful contract
fixture**, backed by the repository's existing `MemmapDataset`. It proves the
recovery mechanism's next-batch contract, not the actual PR100 prepared-data
loader, prefetch/sampling behavior, real vocabulary, or data acceptance report.
The producer observes state **after saving**, matching PR100's
`save_then_runtime_state_v1` ordering. Every process asserts CUDA uninitialized.

CPU command (in the follow-up worktree):

```bash
CUDA_VISIBLE_DEVICES= PITHTRAIN_RUN_OMNI_GPU_TESTS=0 \
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/kldxbjz/code/pith-train-week3-resume \
timeout --kill-after=15s 180s \
/home/kldxbjz/code/pith-pr101-wsl-validation-20261008/venv/bin/python -m pytest \
tests/test_qwen3_omni_checkpoint.py tests/test_strict_checkpoint.py \
tests/test_qwen3_omni_checkpoint_gpu.py -q -s --tb=short --show-capture=no
```

The six original GPU tests retain their existing scheduling opt-in. They are
not scheduled here while PR101 owns the RTX5080; no skip or tolerance was added
to make an implemented test pass. Earlier six-pass GPU evidence belongs to
unchanged `ace46ec` under the documented Math adapter.

## Exact dependencies and next wiring

The parent must pin and combine the actual candidates in a separate integration
checkout. This branch does not assume either PR was merged, import an older data
branch as a replacement, or modify another task's worktree.

1. **PR100:** requested candidate `24529b3` is absent from both this shared Git
   object store and `/home/kldxbjz/code/pith-pr101-wsl-validation-20261008/repo`.
   Its repository path and full commit were requested from the parent. The
   existing `feat/omni-text-data` at `8839b7f` was read only to map interfaces.
   Recheck these interfaces against the actual final PR100 commit:
   `OmniPretrainData.checkpoint_state`, its version/fingerprint/consumed_samples
   schema, `commit_step`, `pretrain_lm.train_step`'s save boundary and
   `pretrain_lm.launch`'s load-before-next-batch boundary.
2. **PR101:** base is `e6d1889c6df5679749a057fd1952360c378f5cd6`; its current
   SM120/FA4 repairs and packed-gradient integrated candidate are owned by the
   other task. Use the parent's exact final commit and verified attention
   backend. Do not rely on the installed FA4 4.0.0b28 bugs being fixed merely
   because the converter passed under SDPA Math.
3. **CPU loader integration:** pass the real `data.checkpoint_state` into
   `StrictResumeState(model, optimizers, schedulers, data_state=...,
   step=completed, global_batch_size=...)`. Save only after optimizer/scheduler
   update and `data.commit_step`; restore before obtaining the next batch.
   Repeat the fresh-process test with the real staged text fixture, immutable
   fingerprint and actual next sample IDs/tokens/labels/masks. Do not add a new
   loader or change the existing observation protocol.
4. **Native GPU/FSDP integration:** reuse
   `tests/test_omni_training_acceptance.py`'s explicit `--restore`, `--expected`,
   `--checkpoint-step`, `REPORT_VERSION`, source hash and
   `save_then_runtime_state_v1` producer ordering. Its inspected implementation
   constructs Qwen3 and uses Qwen3-specific audit wrappers: select native Omni
   and adapt those hooks before claiming Omni acceptance. Keep vocabulary
   152064 for real prepared tokens, BF16, real routing and original global
   batch/loss normalization. Extend the existing state report with required
   CPU/Python/NumPy RNG alongside CUDA RNG, with a deliberate report-version
   change if its schema changes. Resolve DTensor staging/preflight and per-rank
   state first; the current explicit FSDP guard must not simply be removed.
5. **PP/EP single-update and restart:** after single-device native acceptance,
   test PP=2, EP=2 and combined PP+EP using the actual scheduler, collective
   groups and FSDP layouts. A distributed preflight must collectively agree on
   failure before any rank applies state, require each rank's RNG and committed
   data contract, and respect canonical global key coverage without rejecting
   legitimate other-rank keys. Run identical continuous/restart arms and
   missing-rank/state negatives. Measure repeated-run noise for native GPU
   update comparisons without lowering existing acceptance thresholds.

Real PR100 loader integration, native GPU/FSDP strict restart and real PP/EP
updates are **remaining gates**, not CPU substitutes. No GCS, cluster, Notion,
push, PR, comment, or merge action is included.
