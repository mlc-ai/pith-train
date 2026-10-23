# Week4 native image training

> Historical implementation/evidence snapshot. See
> [current Omni acceptance entry points and status](OMNI_ACCEPTANCE_STATUS.md)
> for the integrated implementation and Modal CLI workflow. Old Flame/Slurm
> package, authorization and queue statements below describe their original
> snapshot, not the current execution. Historical NOT_RUN/PASS results are
> not automatically inherited by a newer commit.

Local branch `feat/omni-native-vision`, based on frozen Week3
`d237e4c3d5ce9e38cf80e16a38dbd8f2210d0e53`, which contains PR101
`e6d1889c6df5679749a057fd1952360c378f5cd6` and PR100
`24529b3101dd339782d2dc97b4ffff06685a60e7`. Existing frozen branches and
acceptance packages are unchanged. No FA4 modification is part of this work.

## Implemented boundary

`qwen3_omni_vision.py` implements patch Conv3d, learned bilinear position
embeddings, axial RoPE, noncausal per-image attention, vision blocks, final
projection and DeepStack projections with PyTorch. There is no HF modeling
call in the native path. Config objects and the pinned processor are reused;
HF modeling is an independent test reference. Vision uses PyTorch SDPA and
does not change or replace the existing decoder's FA4 implementation.

`qwen3_omni_image.py` adds placeholder replacement, image/text positions and
DeepStack after decoder layers 0..K-1, where the K features come from the
configured ascending vision block indexes. Differentiable DeepStack features
travel in K extra hidden-width lanes in the normal activation tensor, including
PP sends and manual stage backward. There is no module-level feature cache and
no detached/precomputed/frozen visual training path. This correctness-first
implementation costs (K+1) times the decoder hidden activation/P2P width; the
tiny gate uses K=2. Sparse transport is a later performance optimization.

The existing Microbatch `model_context` and `media_inputs` already match these
consumers, so no duplicate data system or unused payload field is added.
`TrainingCfg.native_vision=True` opts into the composite Thinker image model.
The default remains the original text model. Use AdamW in the acceptance
configuration; when a caller uses the framework's combined Muon optimizer,
all visual parameters go to its AdamW partition, including Conv3d weights.

Supported first boundary: image+paired-text records, one unpadded sample per
microbatch, CP=1, trainable native vision/projections/decoder. Mixed text-only
and image DP batches are rejected by the image modality declaration: their
unused-encoder FSDP collective contract was not implemented in this image-only
milestone. The current unified multimodal selector implements a separate
collective contract and must pass its own native GPU gates. Text-only
training keeps its existing default model. Audio/video, packed/padded image
batches, FP8, full released weights and image HF checkpoint conversion are
outside this historical milestone. Audio/video and complete-Thinker conversion
now have separate implementations; their evidence is not inherited from the
image-only tests. See OMNI_ACCEPTANCE_STATUS.md for current limits and the
recorded production image/audio numerical failures.

Reader failures and consumer validation failures propagate through each PP
group and then across the world before model/EP collectives. A CPU 4-rank Gloo
test corrupts only one DP reader's pixels and requires identical rejection on
all PP/DP ranks, without advancing the durable data cursor. This does not
promise recovery after process death or broken NCCL transport.

## Independently acceptable milestones

1. **Native vision and consumer math / CPU**: vision outputs, every parameter
   and input gradient, updates, geometry, placeholder failures and two live
   image graphs; unchanged real PR100 reader with fixed sample audits.
   Implemented and locally tested. This alone is not Thinker end-to-end training.
2. **Single H100 native image training**: independent HF logits/CE/every declared
   gradient for two real images and four optimizer steps, plus existing text
   numerical regression. Code and runner prepared; GPU NOT RUN.
3. **Staged H100 training**: DualPipeV/FSDP WORLD1, DP2, PP2, EP2 and PP2EP2,
   global target counts excluding PP replicas, nonuniform/zero local targets,
   global all-zero rejection, a separate whole-DP-rank-zero scenario, all owned gradients versus independent HF,
   encoder/projection/decoder update witnesses and four real-reader task steps.
   Code and runner prepared; GPU NOT RUN.

The v3 P2 correction preserves the original nonuniform fixture (DP2 target
totals 28/39) and adds independently launched rank-zero scenarios on DP2, EP2
and PP2EP2 (DP2 totals 0/39). CPU audits use the identical fixture builder;
four-rank Gloo and fault-injection tests verify the labels and nonduplicated
counts. GPU native CE/FSDP gradient/update execution is still NOT RUN. See
WEEK4_ACCEPTANCE_MATRIX.md for the exact evidence boundary.

Planning estimate for the expanded native-encoder scope: approximately 8–12
hours for milestone 1, 8–12 for milestone 2, and 8–16 for milestone 3 including
debugging and review preparation (24–40 hours total, not measured labor).
The original 15-hour week should not retain its completion promise. Queueing
and unresolved Week3/FA4 baseline correctness can extend elapsed dates.
Later mixed-modality, checkpoint and full-model gates need a separate estimate.

## CPU execution

Use the existing complete CPU environment; no shared environment was modified:

```bash
cd /home/kldxbjz/code/pith-train-week4-native-vision
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/kldxbjz/code/pith-train-omni/.venv/bin/python
"$PY" -m pytest tests/test_qwen3_omni_vision.py tests/test_omni_image_contract.py \
  tests/test_omni_image_acceptance_evidence.py tests/test_omni_image_rank_zero.py \
  tests/test_omni_model_registration.py tests/test_omni_training.py \
  tests/test_qwen3_omni_data.py tests/test_training_data.py \
  tests/test_training_sequence_length.py tests/test_qwen3_omni_checkpoint.py \
  tests/test_strict_checkpoint.py tests/test_strict_omni_converter.py \
  tests/test_distributed_strict_checkpoint.py tests/test_omni_native_resume_contract.py \
  -q --junitxml=/new/evidence/cpu.xml
"$PY" tests/acceptance/omni_image_training.py --mode cpu \
  --dataset /path/to/verified/omni-training-dev --output /new/evidence/real-images
```

The exact two audited samples are `coco-57870` and `coco-384029`, with 10 and
12 effective text targets using the bounded audit processor settings. `cpu.json`
records image/pixel hashes, sample IDs, text, grids, token IDs, feature counts,
all T/H/W positions, shifted labels and all 57 visual parameter gradient errors
and updates. Weights are deterministic tiny random weights; no pretrained
30B weights were loaded. The CPU log asserts CUDA remains uninitialized.

## Proposed Flame resource and transfer scope

Do not upload, submit, reuse another running allocation, or modify a frozen
package without the user's new explicit resource/transfer authorization.

Proposal: one node, **4 H100**, 32 CPUs, 128 GiB host RAM, 32 GiB new scratch,
**45-minute allocation = at most 3 allocated GPU-hours**. This is an estimate,
not a measured H100 requirement. Eight staged scenarios plus text/image reference gates fit the same hard budget; timeouts fail rather than skip. Runner wall budget is 40 minutes; per GPU gate
is capped at 120 seconds (single GPU) or 180 seconds (multi-GPU), plus at most
15 seconds termination grace. No retry/requeue. Queue time is not execution.
The tiny vision is 3 layers/64 hidden and text is 4 layers/128 hidden with the
real 152064 vocabulary; this reservation does not cover the released full model.

The local delivery contains a Git bundle, one self-contained asset archive,
ASSETS.json and TRANSFER.json with exact bytes/SHA256. The archive contains only
the existing checksum-listed development data bundle (including media needed
by whole-bundle integrity checks) and the seven pinned processor/config files.
It excludes weights, checkpoints, environments, credentials, FA4 patches and
unlisted stale media. Nothing has been transferred remotely.

After authorization, clone the Git bundle on a new path, extract the archive
into a new asset directory, verify all ASSETS.json hashes, and select the
contained `hub` with HUGGINGFACE_HUB_CACHE. Do not extract into the prior package.
The environment must already have the project's CUDA dependencies and media
extra, Transformers 5.17, and pytest. Missing dependencies are failures, not skips.

Inside the approved allocation, the sole coordinated Flame owner can run:

```bash
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export HUGGINGFACE_HUB_CACHE=/new/week4-assets/hub
export PITHTRAIN_RUN_OMNI_GPU_TESTS=1
srun --jobid="$SLURM_JOB_ID" -N1 -n1 -c32 --gres=gpu:4 -W 0 \
  --open-mode=truncate -o /new/week4-step.log \
  bash /new/week4-code/tests/acceptance/run_week4_h100.sh \
  /path/to/approved/python /new/week4-assets/data /new/week4-evidence
```

The runner checks clean source, H100 identity and free memory before CUDA gates;
it rejects skipped/error pytest suites and missing per-rank reports. Failed,
queued, skipped and not-run cases never become PASS. GPU acceptance and user
review remain pending until the required gates actually pass. The existing
decoder FA4 baseline must pass the text gate; this branch will not bypass it.
