# Native Qwen3-Omni Thinker multimodal pretraining

`TrainingCfg.native_multimodal=True` selects one model with trainable text,
visual/DeepStack and audio encoders. It accepts text-only, image, audio and video
microbatches in the same run, including independent media spans in one document
and synchronized video/audio. The decoder is the native staged MoE implementation;
the encoders use native PyTorch. Transformers executes only in independent tests.

The model/config reference is `Qwen/Qwen3-Omni-30B-A3B-Instruct` revision
`26291f793822fb6be9555850f06dfe95f2d7e695`, with Transformers 5.17.0.
The prepared bundle's recipe must enable `batch.use_audio_in_video=true` to decode
audio tracks. This opt-in is fingerprinted by the bundle checksum. A video without
audio remains video-only; a processor microbatch cannot combine videos with and
without audio tracks. Audio is placed by presentation timestamp relative to the
video's first frame. Gaps become explicit silence; content outside the video
timeline is rejected rather than truncated.

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/pretrain_lm/qwen3-omni-native-multimodal/script.py \
  --dataset /path/to/prepared-bundle --model /path/to/full-thinker-config \
  --checkpoint /path/to/checkpoints --epoch-samples 32 \
  --modalities text image audio video --global-batch-size 8 --pp 2 --ep 2
```

To start from HF weights, first convert the **complete Thinker** weights to the
canonical DCP format, then add `--initialize-from /path/to/converted-dcp-root`
(`--initialize-step 0` is the default source checkpoint). The output `--checkpoint`
must contain no existing checkpoint. This loads model tensors only: optimizer,
scheduler, configured-seed RNG, prepared-data cursor and training step start fresh.
The source checkpoint step is an identifier, not the new run's step.

To resume, keep the output `--checkpoint` and omit both initialization arguments.
The recipe still requires strict same-topology model/optimizer/scheduler/RNG/data
recovery. A model-only checkpoint cannot satisfy resume; specifying initialization
alongside an existing output checkpoint fails before training setup. A strict
rank-local checkpoint must be exported to canonical model weights before using
it as a new run's initialization source.

Current supported layout: BF16, CP=1, one unpadded document per microbatch, AdamW
with strict same-topology resume. PP/EP remain the existing staged decoder paths.
FP8, packing, padding and CP>1 fail explicitly. Talker and speech output are
outside this pretraining model.

Every first-stage microbatch calls the visual encoder once and the audio encoder
once, in that order. Images and videos share one concatenated visual call.
Missing encoders receive minimal valid inputs; zero dependencies produce exact
zero gradients and enforce audio→visual→embedding backward order. This preserves
the FSDP collective schedule when data ranks choose different modalities.
DeepStack features travel in the microbatch activation; no feature cache is kept
on the model.

CPU tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_qwen3_omni_multimodal.py tests/test_qwen3_omni_av.py
```

They compare exact HF T/H/W positions, every encoder gradient, both collective
orders at the module boundary, missing-encoder zero gradients, live microbatch
graphs, optimizer save/restore, real AV decoding/processing, and invalid inputs.
These tests are CPU component evidence, not GPU distributed acceptance. GPU
results must name the tested commit and topology, and include every parameter
gradient plus strict checkpoint restore and next-update evidence.
