# Qwen3-Omni training data example

This entry point configures the existing `pretrain_lm.launch`, including its
optimizer, DualPipeV scheduler and distributed checkpoint path. Prepare the data
with [the unified recipe](../../prepare_omni_data/qwen3-omni-training/README.md).

```bash
torchrun --standalone --nproc-per-node=1 examples/pretrain_lm/qwen3-omni/script.py \
  --dataset workspace/datasets/omni-training --modalities text \
  --model /path/to/compatible-native-model-config \
  --sequence-length 128 --global-batch-size 8 --steps 4 \
  --checkpoint workspace/checkpoints/qwen3-omni --save-interval 2
```

`--model` must name a model registered in PithTrain with a vocabulary of at least
152064 entries. A reduced model must keep that vocabulary while reducing depth,
width or expert count; never clamp token IDs or take them modulo a smaller
vocabulary. This recipe does not download model weights. Native Qwen3-Omni
registration and numerical validation are a separate model change.

## Shared data configuration

Every pretraining example uses `cfg.data`, a `DataCfg` containing the dataset path,
storage format, modalities, sampling weights, epoch size and worker count. There is
no separate `cfg.dataset` or optional `cfg.omni_data` configuration.

```python
cfg.data.dataset = Path("workspace/datasets/omni-training")
cfg.data.format = "prepared_bundle"
cfg.data.modalities = ("text", "image")
cfg.data.sampling_weights = {"text": 1, "image": 2}
```

Other model examples use the defaults `format="token_bin"` and
`modalities=("text",)`, pointing `cfg.data.dataset` directly to their token shards.
This example defaults to `prepared_bundle`; `--data-format token_bin` selects the
same legacy dense reader. Format describes storage; modalities select record kinds.
The preparation recipe's named stages remain preparation presets. For training,
`--modalities text image` explicitly selects the two kinds, and other nonempty
subsets of prepared modalities are allowed. No extra modality is silently enabled.

## Data modalities and model contract

- `--modalities text` with a prepared bundle reads only `tokens/train/*.bin`, using the existing shuffled dense loader
  and CP zigzag layout. Held-out tokens are never included. The corpus must have
  enough complete sequences for `global_batch_size * steps`.
- Select `--modalities text image`, `--modalities text image audio`, or
  `--modalities text image audio video` to reproduce the former cumulative stages.
  Set `--sampling-weights '{"text":1,"image":2}'` for the two-modality example.
  Media epochs use weighted sampling with replacement; `--epoch-samples` must be
  a multiple of global batch size. Micro-batch size is currently 1 and CP must be 1.
- Within each PP group, only rank 0 verifies/reads media and runs the processor.
  It keeps image/video pixels and audio features in `Microbatch.media_inputs`.
  Tokens, shifted labels and position/layout tensors are broadcast to its PP peers;
  `model_context` holds the shared inputs, without encoder payloads. DP selects
  records; PP and EP do not select additional independent samples. CPU workers use
  a separate RNG so reading a batch does not perturb the model's random state.
- A media-capable model must declare `input_modalities` and accept `model_context`
  in `forward`, `forward_prolog` and `forward_posemb`. Its normal forward passes
  this argument to `model_forward`; the overlapped scheduler passes it directly.
  Encoders/feature insertion belong in the model prolog, and multimodal position
  construction belongs in its position method. The pipeline merges `media_inputs`
  into the context only for model stage 0. All later stages, including the other
  V-shaped chunk on PP rank 0, receive only shared context. Input distribution is
  separate from activation P2P. Root FSDP preserves context precision (including
  timing); individual compute modules cast their inputs.
- Existing models default to text-only. Unsupported model/modalities, vocabulary or
  media CP/batch layouts fail before allocating model parameters. Routing media
  tensors is implemented; consuming them with native Omni encoders/positions is
  still model work. Synchronized audio/video and Talker targets remain unsupported.

## Loss and restart behavior

`pretrain_lm` sums actual non-ignored label counts across the DP x CP ranks of one
pipeline stage. It normalizes summed gradients once by that count and logs the
global token-mean loss. PP replicas are not counted twice; a rank can have zero
targets when the global batch has some. A globally empty target batch fails.

The model/optimizer checkpoint now also stores data identity and the count of
samples consumed by completed optimizer steps. Prefetch never advances this
saved count. Relaunch with the same checkpoint directory to resume; increasing
`--steps` is allowed. Changing the data bundle, modalities, sampling, sequence length,
seed or batch sizes rejects the resume. An old checkpoint without data state
cannot silently restart this data stream. Legacy `.bin`-only recipes keep their
existing checkpoint format and loading path. Selecting the same modalities as an
existing recipe stage preserves its version-1 fingerprint, so checkpoints from
the previous `OmniDataCfg.stage` configuration remain compatible. A new subset
includes its selected modalities in the fingerprint; a different subset cannot
resume the same checkpoint.

## Integration check

```bash
torchrun --standalone --nproc-per-node=4 tests/test_omni_training_gpu.py \
  --dataset workspace/datasets/omni-training \
  --output /tmp/omni-training-check --pp 2 --ep 2 --context
```

Use a fresh output directory. The test builds a reduced existing Qwen3 model with
the real vocabulary and takes two optimizer steps. Loading the first checkpoint
must exactly restore model parameters, AdamW state, scheduler state and CUDA RNG.
Replaying the second step must reproduce token IDs, labels and sequence boundaries,
advance the data cursor, and produce finite nonzero gradients and a parameter
update. Its loss is compared using the existing DualPipeV BF16 tolerance
(`rtol=1e-3`, `atol=1e-3`). Post-update parameters are not compared elementwise:
repeated GPU reductions can change gradients even without saving or loading.
This is a data/checkpoint integration check, not a model numerical-regression test.
`--context` uses a test-only consumer to check normal and overlapped context
transport; it does not implement or validate an Omni encoder. CPU coverage is in
`tests/test_omni_training.py`; GPU checks require Hopper/Blackwell hardware.

The training task consumes all sources through `PretrainData`: `get_batch(step, device)`,
`commit_step(step)` after optimization, and `checkpoint_state` for save/restore.
`DensePretrainData` retains the existing token packing, corpus shuffle and DP/CP slicing;
its position comes from the training step, so legacy checkpoints need no data-state entry.
`OmniPretrainData` owns the dense or media reader and checkpoints the committed cursor and
recipe identity. Adding a source with this interface does not require a type check in the
training loop. This interface does not extend media support beyond CP1/microbatch1.

## Target normalization check

```bash
torchrun --standalone --nproc-per-node=4 tests/test_omni_target_gpu.py
```

That independent test checks the actual fused CE and FSDP summed gradients in
FP32 and BF16, with unequal labels, a rank containing only ignored labels and a
globally empty batch. Two separate pipeline-stage groups catch accidental PP
double-counting. Loss, gradients and an SGD update are compared with a global
single-device reference.


The one-off 12-step repeated-base and fresh-process acceptance harness and its
schema-3 reports are retained in the [PR100 validation record](https://github.com/mlc-ai/pith-train/pull/100).
Those reports compare CE, load-balancing loss and gradient-norm drift against two
unchanged baseline runs (ratio < 3); they do not assert elementwise post-update
parameter equivalence. Historical results apply only to the recorded source,
inputs, topology and harness. The ongoing regressions above cover the data and
checkpoint contracts; they do not establish native Omni model correctness.
