# Qwen3-Omni Thinker text pretraining

This implementation covers the Thinker text backbone from
`Qwen/Qwen3-Omni-30B-A3B-Instruct`, revision
`26291f793822fb6be9555850f06dfe95f2d7e695`. It uses native PithTrain attention,
grouped GEMM, FSDP and DualPipeV; Hugging Face supplies the test reference only.

## Configurations and training

- `tests/configs/qwen3_omni_text/tiny.json` reduces depth, width and expert count.
  It retains the real 152064-token vocabulary, so the prepared Omni text corpus
  can be consumed directly. Four layers also exercise all four PP=2 V-chunks.
- `config.json` and `tests/configs/qwen3_omni_text/full.json` retain the released
  Thinker text architecture. A full configuration is not evidence of a full-size
  training run. Select a suitable memory/parallelism layout before running it.

The small prepared dataset contains enough text for this four-step example:

```bash
torchrun --standalone --nproc-per-node=1 \
  examples/pretrain_lm/qwen3-omni-thinker-text/script.py \
  --model tests/configs/qwen3_omni_text/tiny.json \
  --dataset /tmp/omni-training/tokens/train \
  --checkpoints /tmp/omni-checkpoints
```

Both configs train from scratch. The released checkpoint stores individual
`gate_proj`/`up_proj` expert tensors; Transformers 5.17 exposes fused
`gate_up_proj` tensors at runtime. Native experts follow that runtime layout,
`[experts, out, in]`. A released-checkpoint converter and weight roundtrip are
still needed before training from downloaded weights. The HF test copies every
runtime parameter explicitly and checks complete name/shape coverage.

## Validation

```bash
python -m pytest tests/test_qwen3_omni_reference.py -q -s
torchrun --standalone --nproc-per-node=1 tests/test_dualpipev.py \
  --model tests/configs/qwen3_omni_text/tiny.json --pp-size 1 --ep-size 1
```

The HF test compares native logits, next-token CE and every parameter gradient,
including fused experts. It covers ordinary MoE, a dense layer with an
unnormalized router and attention bias, and packed document boundaries. A
separate native/HF rotary comparison uses distinct temporal/height/width axes.

Logits use normalized squared error below 1e-3; CE uses rtol=atol=1e-3;
per-parameter gradient error must stay below the existing DualPipeV bound 1e-2.
Every gradient must exist and be finite; no parameter-name skips are used in
the HF test. GPU results must validate these bounds before merge.

The pipeline ladder is PP/EP 1/1, 2/1, 1/2, 2/2, plus CP=2 and packed/ragged
microbatches. Training loss, optimizer updates, checkpoint recovery and real
tokenized data need end-to-end runs in addition to the model comparisons.

## Current boundary

BF16 text training only. Audio/vision encoders, feature insertion, DeepStack,
multimodal position construction and Talker output are not implemented. The
input capability remains explicitly text-only. RoPE supports the released
default mode; other RoPE modes, nonzero dropout, tied embeddings and FP8 fail
explicitly rather than silently selecting different math. Packed CP is not
supported. No full-size, multi-node, long-training or pretrained-generation
result is claimed by this first implementation.
