# Train from the prepared Omni data bundle

This entry point configures the existing `pretrain_lm.launch`, including its
optimizer, DualPipeV scheduler and distributed checkpoint path. Prepare the data
with [the unified recipe](../../prepare_omni_data/qwen3-omni-training/README.md).

```bash
torchrun --standalone --nproc-per-node=1 examples/pretrain_lm/omni-data/script.py \
  --dataset workspace/datasets/omni-training --stage text \
  --model /path/to/compatible-native-model-config \
  --sequence-length 128 --global-batch-size 8 --steps 4 \
  --checkpoint workspace/checkpoints/omni-data --save-interval 2
```

`--model` must name a model registered in PithTrain with a vocabulary of at least
152064 entries. Reduced depth/width is fine; the 256-token HF comparison fixture
and the unmodified Qwen3 example vocabulary are incompatible. This recipe does
not download model weights. Native Qwen3-Omni is not registered yet.

## Data stages and model contract

- `text` reads only `tokens/train/*.bin`, using the existing shuffled dense loader
  and CP zigzag layout. Held-out tokens are never included. The corpus must have
  enough complete sequences for `global_batch_size * steps`.
- `image`, `audio` and `video` enable the cumulative mixtures in the data recipe.
  Set `--sampling-weights '{"text":1,"image":2}'` for the image stage, for example.
  Media epochs use weighted sampling with replacement; `--epoch-samples` must be
  a multiple of global batch size. Micro-batch size is currently 1 and CP must be 1.
- The task converts processor batches into `Microbatch`: token IDs are positional
  model inputs, shifted labels are objective inputs, and all processor tensors
  travel in `model_context` on the rank's device. DP selects records; PP and EP
  do not select additional independent samples. Workers decode on CPU and use a
  separate RNG so reading a batch does not perturb the model's random state.
- A media-capable model must declare `input_modalities` and accept `model_context`
  in `forward`, `forward_prolog` and `forward_posemb`. Its normal forward passes
  this argument to `model_forward`; the overlapped scheduler passes it directly.
  Encoders/feature insertion belong in the model prolog, and multimodal position
  construction belongs in its position method. Context is available on each PP
  rank, separately from the hidden activations sent over P2P. Root FSDP preserves
  context precision (including timing); individual compute modules cast their inputs.
- Existing models default to text-only. Unsupported model/stage, vocabulary or
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
`--steps` is allowed. Changing the data bundle, stage, sampling, sequence length,
seed or batch sizes rejects the resume. An old checkpoint without data state
cannot silently restart this data stream. Legacy `.bin`-only recipes keep their
existing checkpoint format and loading path.

## Integration check

```bash
torchrun --standalone --nproc-per-node=4 tests/test_omni_training_gpu.py \
  --dataset workspace/datasets/omni-training \
  --output /tmp/omni-training-check --pp 2 --ep 2 --context
```

Use a fresh output directory. The test builds a reduced existing Qwen3 model with
the real vocabulary, updates parameters, saves/restores training plus data state,
and checks that replaying the second step reproduces its batch and parameters.
`--context` uses a test-only consumer to check normal and overlapped context
transport; it does not implement or validate an Omni encoder. CPU coverage is in
`tests/test_omni_training.py`; GPU checks require Hopper/Blackwell hardware.
