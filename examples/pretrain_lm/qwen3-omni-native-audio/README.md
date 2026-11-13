# Native Qwen3-Omni audio/text pretraining

`TrainingCfg.native_audio=True` selects a native PyTorch audio encoder and the
native Thinker text decoder. The encoder follows Transformers 5.17.0
`Qwen3OmniMoeAudioEncoder`, with the same parameter names and shapes. It does not
call or wrap a Transformers model during training.

Use a composite Thinker or full Omni config; a text-only config lacks the audio
dimensions and token IDs. For released Qwen/Qwen3-Omni-30B-A3B-Instruct, pin the
config and processor to `26291f793822fb6be9555850f06dfe95f2d7e695`.

The input contract is a prepared bundle with paired audio and text, microbatch
size 1, CP=1, no sequence padding or packed sequences, and no image/video payload.
Every sample has audio; mixed text-only and audio samples require the combined
modality consumer. Each clip uses its own audio start/end delimiter and exactly
the number of placeholders emitted by the encoder. The released audio geometry
uses `n_window=50`; the inference window must be a positive multiple of 100.
Dropout is zero. `native_audio` and `native_vision` are mutually exclusive.

`script.py --help` describes the training entry point. It consumes the existing
prepared audio bundle reader and checkpoint machinery. PP/EP options expose the
native pipeline, but each requested topology must pass its acceptance gates
before production use.

```bash
python examples/pretrain_lm/qwen3-omni-native-audio/script.py \
  --dataset /path/to/prepared-bundle --model /path/to/composite-config \
  --checkpoint /path/to/new-checkpoints --epoch-samples 32 \
  --global-batch-size 8 --steps 4 --save-interval 2 --pp 1 --ep 1
```

CPU acceptance compares independent HF and native outputs, every parameter
gradient, input gradients and AdamW updates. It also checks chunk boundaries,
multiple clips, masks, delimiters, invalid configs, live microbatch graphs,
model registration and initialization after allocation from meta tensors:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  timeout 180 python -m pytest tests/test_qwen3_omni_audio.py -q
```

The separate processor gate decodes two audio recordings from an existing local
bundle, runs the pinned HF processor, and checks native output, audio/text token
geometry, every gradient and an optimizer update against HF on CPU:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=2 \
  timeout 180 python tests/acceptance/omni_audio_processor.py \
  --dataset /path/to/prepared-bundle --output /path/to/new-cpu-report.json
```

The GPU reference gate uses synthetic mel features, a reduced encoder and
four-layer decoder with the real tokenizer vocabulary. It checks BF16 output
NSE < 1e-3, every parameter/input gradient NSE < 1e-2, cross entropy with
rtol=atol=1e-3, and four optimizer updates across two live microbatches. It requires
H100, Transformers 5.17.0 and the project's pinned CUDA dependencies:

```bash
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  timeout --signal=TERM --kill-after=10s 180s \
  python tests/acceptance/omni_audio_reference.py --output /path/to/new-report.json
```

The JSON records the exact commit, config, dependency versions, hardware,
input hashes, and per-parameter errors, including a failed verdict on exceptions.
This gate does not establish distributed pipeline correctness, real processor
integration, released-weight conversion, or full training checkpoint/resume.
Those are separate required integration gates. No GPU acceptance is implied by
the presence of the runner or by CPU parity.
