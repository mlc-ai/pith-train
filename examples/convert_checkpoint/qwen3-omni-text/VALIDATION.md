# Converter follow-up validation — execution status and resources

Prepared on the local converter branch; no GPU commands have run in this task.

## CPU checks executed

The full converter suite includes strict JSON uniqueness in config, index,
sidecar and raw safetensors headers; complete key/shape checks; metadata dtype
validation before payload reads or allocation; both layouts' cross-expert dtype
consistency; numeric expert ordering past expert 9; explicit gate-before-up
markers; refusal of unknown text/similar media prefixes; preservation of occupied
output paths; generic Qwen and existing GPT-OSS/Qwen3.5 file conversion. CPU
tensor and HF prediction roundtrips remain exact. Full uses meta/header coverage
only, never full-weight numerical validation.

## Prepared tiny GPU converter acceptance

`tests/test_qwen3_omni_checkpoint_gpu.py` has six cases: released split and HF
5.17 fused, each at sequence lengths 32, 64 and 69. All cases require exact
HF -> DCP -> HF tensor equality, exact same-backend before/after predictions,
strict native loading via checkpoint.py, and the existing PR101 native/HF logit
error threshold. Length 64 deliberately remains included to expose the currently
unresolved LSE anomaly. No threshold is relaxed for that failure.

Only after the parent releases the RTX5080, inspect availability immediately
before execution and schedule one process on that device:

```bash
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=<released-device> \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$PWD" \
  timeout 180 python -m pytest tests/test_qwen3_omni_checkpoint_gpu.py -q -s
```

Without the explicit flag these six tests are skipped *before native imports*;
that skip is preparation status, not GPU acceptance. The synthetic config is
2 layers, vocab 256, BF16, CP/PP/EP=1, no download or training dataset. One GPU is
enough. Reserve several GiB for CUDA/kernel overhead (4 GiB planning allowance;
not measured), and record actual peak memory and the FA4/PyTorch versions on run.
The existing native imports currently probe CUDA and cannot be CPU-dry-run.

## Strict training recovery — integration prerequisite, not executed

The reviewed PR101 baseline `e6d1889` lacks PR100's `data_state` argument. Its
model-only load fallback also deliberately tolerates absent optimizer/scheduler
and CUDA RNG. It cannot yet certify strict training restart. Adding a synthetic
parallel loader or marking this path passed would test the wrong contract.

Read-only inspection of the existing `feat/omni-text-data` branch at `8839b7f`
locates the real reuse points:

- `tests/test_omni_training.py::test_checkpoint_restores_model_optimizer_and_next_data`
  already checks CPU model/optimizer/scheduler and next data batch.
- `tests/test_omni_training_acceptance.py` already provides a fresh-process
  `--restore` + `--expected` arm, `--checkpoint-step`, state hashes and ordered
  checkpoint observations. It currently creates an **existing Qwen3** model and
  has Qwen3-specific auditing wrappers; unmodified results do not certify Omni.
- `pithtrain/modules/checkpoint.py` on that branch accepts `data_state` and
  rejects absent data state. Missing CUDA RNG still needs strict-mode rejection.

Before modifying these files, integrate the actual protected PR100 local
`24529b3` changes with PR101 in a new integration checkout, preserving both
original worktrees. This local Git object store does not contain `24529b3`; the
older inspected branch is only a map of entry points, not a substitute head.

Concrete test to add to those existing entry points after integration:

1. Select native `Qwen3OmniMoeThinkerTextModel` and the existing staged text
   train/validation source. Keep real token vocabulary 152064, CP=1/BF16 and
   the same global batch, seed and loss normalization in both arms.
2. Continuous arm performs 3 updates and saves immediately after update 2.
   New-process arm restores that exact boundary and performs update 3. Reuse
   the existing observation-protocol value and report machinery, rather than
   adding another checkpoint/data-position convention.
3. Require exact restored model/optimizer/scheduler state, data cursor, next
   sample IDs/tokens/labels/masks, CPU/CUDA and any used Python/NumPy RNG. Then
   compare next-step loss and every parameter update, with GPU numerical
   tolerances bound to measured repeated-run noise, and assert finite nonzero
   updates. Weight-only initialization is an explicit separate mode.
4. In copies of that tiny checkpoint, individually remove optimizer, scheduler,
   data state and required per-rank RNG; mutate model key/shape, dataset
   fingerprint and sampler cursor; confirm strict resume fails before consuming
   a batch or updating parameters. Removing a scheduler entry must not be
   hidden by `zip` truncation. Do not mutate the accepted checkpoint or dataset.
5. First run single GPU (plan 4–8 GiB free; measure, no guarantee). PP=2, EP=2,
   then combined PP+EP require at least two/four coordinated devices according
   to the selected mesh and repeat the same contract. RTX5080 alone does not
   establish multi-device correctness. No cluster submission is authorized.

Current gate: parent task `01a11a21` owns the RTX5080 while diagnosing FA4
SM120/GQA and length-64 LSE. No inference, backward, memory probe, multi-GPU
launch, full checkpoint download or cluster submission is performed here.
