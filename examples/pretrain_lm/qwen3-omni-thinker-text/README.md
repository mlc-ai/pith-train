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

## Execution policy

Normal training uses fused token dispatch and compiled stage 1 normalization
and stage 5. The router keeps its eager BF16 rounding boundary. The CE
normalization and grouped-linear gradient fixes apply to both policies.

For deterministic diagnostics, set `cfg.training.deterministic = True` before
calling `launch(cfg)`, and start the process with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. This selects stable dispatch and deterministic
FA4/PyTorch behavior; it does not disable compiled stages or select another
model. The default is `False`. Determinism is scoped to the tested execution
path; it does not promise bitwise HF agreement or uninterrupted/resumed training
equivalence across environments.

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
the HF test. GPU results must validate these bounds on the actual default fast path. The
existing eager `reference_forward` is the pipeline oracle; HF independently
checks that reference. These are separate comparisons with their own bounds.

The pipeline ladder is PP/EP 1/1, 2/1, 1/2, 2/2, plus CP=2 and packed/ragged
microbatches. Training loss, optimizer updates, checkpoint recovery and real
tokenized data need end-to-end runs in addition to the model comparisons.

The 32-step base/base/feature noise ratio is recorded separately as a trajectory
and reproducibility diagnostic. Its original thresholds are unchanged. On
`f43fb74b`, the repeated eager baseline has a zero noise envelope and the
compiled fast path fails that ratio; this is not relabeled as a pass. Finite
loss/gradient checks, input and learning-rate correspondence, complete parameter
gradient/update comparisons, and required distributed tests remain correctness
requirements. No performance improvement or full-size convergence is claimed.

## Current boundary

This PR covers BF16 text training. Audio/vision encoders, feature insertion,
DeepStack, multimodal position construction and Talker output belong to separate
Omni component PRs and are outside this backbone's validation scope. The
input capability remains explicitly text-only. RoPE supports the released
default mode; other RoPE modes, nonzero dropout, tied embeddings and FP8 fail
explicitly rather than silently selecting different math. Packed CP is not
supported. No full-size, multi-node, long-training or pretrained-generation
result is claimed by this first implementation.
