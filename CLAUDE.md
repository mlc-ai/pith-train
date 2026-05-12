# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PithTrain** (package name: `pithtrain`) — a lightweight training framework for Mixture-of-Experts (MoE) language models. The core innovation is **DualPipeV**, an overlapped forward-backward pipeline parallelism technique. Requires Hopper (SM90) or Blackwell (SM100) GPUs. Python >= 3.12.

## Setup

```bash
uv venv && uv sync
```

## Common Commands

### Linting & Formatting (Ruff)

```bash
ruff check --fix pithtrain/          # lint with auto-fix
ruff format pithtrain/               # format
pre-commit run --all-files       # run all pre-commit hooks
```

Style: 100-char line length, double quotes, `py312` target. Rules: E, F, I, W (ignoring E501, E731). First-party import: `pithtrain`.

### Testing (pytest)

```bash
# Single-GPU unit tests (kernels, ops, layer protocol)
pytest tests/test_fp8_quantize_kernels.py -v
pytest tests/test_deepgemm_fp8_linear_correctness.py -v
pytest tests/test_grouped_linear_correctness.py -v
pytest tests/test_ep_dedup_dispatch.py -v
pytest tests/test_silu_mul.py tests/test_clamped_swiglu.py tests/test_indexed_bias_add.py -v
pytest tests/test_ring_attention.py tests/test_layer_partition.py -v
pytest tests/operators/mla/test_triton.py -v

# Single test function
pytest tests/test_fp8_quantize_kernels.py::test_name -v

# Multi-GPU integration test — boots DualPipeV with FSDP, ~4 GPUs, pp=2 ep=2
bash tests/test_fsdp.sh

# Inference tests must also go through DualPipeV (.step(forward_only=True))
pytest tests/test_gpt_oss_single_gpu.py -v
```

### Benchmarks

```bash
python3 -m benchmarks.operators.fp8.test_deepgemm
python3 -m benchmarks.operators.mla.test_triton
```

### Training & Data Prep

The `examples/` tree drives end-to-end workflows; each subdir has a `launch.sh` that auto-detects single-node vs. SLURM and forwards to a `<config>/script.py`.

```bash
# Tokenize a pretraining corpus (per-tokenizer; rerun when switching models)
bash examples/build_tokenized_corpus/launch.sh dclm-qwen3

# Pretrain (qwen3-30b-a3b | deepseek-v2-lite | gpt-oss-20b | gpt-oss-120b)
bash examples/pretrain_language_model/launch.sh qwen3-30b-a3b

# Convert a training checkpoint to / from HuggingFace
bash examples/convert_checkpoint/launch.sh qwen3-30b-a3b
```

### Memory Estimation

```bash
python -m tools.memory_estimator --help   # peak-memory simulator for a given parallelism mesh
```

## Architecture

### DualPipeV Pipeline (`pithtrain/dualpipe/`)

The core pipeline runs two model replicas with overlapped forward-backward execution across micro-batches. Each transformer layer is split into 5 stages:

1. **Attention** — LayerNorm + Attention + LayerNorm + Expert routing
2. **Dispatch** — All-to-all send tokens to assigned experts (async on comm stream)
3. **MLP** — Expert/MLP computation
4. **Combine** — All-to-all gather expert outputs (async on comm stream)
5. **Aggregate** — Weighted expert output + residual connection

Key files:
- `dualpipev.py` — Main scheduler: `DualPipeV.step()` orchestrates overlapped F/B across modules. Supports `forward_only=True` for inference.
- `overlap.py` — `overlapped_forward_backward()` interleaved loop for one pair of micro-batches
- `execution.py` — Stage implementations (`stage1_f`, `stage1_b`, etc.) and `ExecutionCtx`
- `modeling.py` — `decoder_layer_forward/backward` autograd wrappers, dispatch/combine helpers
- `layer_partition.py` — Distributes decoder layers across pipeline stages; edge stages (which hold `embed_tokens` / `norm`+`lm_head`) get fewer layers to balance memory.
- `comm.py` — P2P communication setup between pipeline ranks
- `utils.py` — `FP8WeightCacheControl` (cache quantized weights across micro-batches), `WeightGradStore` (deferred wgrad for zero-bubble scheduling)

### FP8 Training

`ModelImplMode.fp8_training` in `pithtrain/layers/factory.py` selects the linear-layer backend (currently `"deep-gemm"` or BF16 fallback). The DeepGEMM path (`pithtrain/layers/deepgemm_fp8_linear.py`) uses 128-element block scaling with E8M0 scale format, backed by custom Triton quantization kernels in `pithtrain/operators/deepgemm_fp8_quantize.py`. The BF16 grouped linear layer is in `pithtrain/layers/group_linear.py`.

### Distributed Parallelism (`pithtrain/modules/distributed.py`)

Four dimensions: Pipeline Parallel (PP), Expert Parallel (EP), Context Parallel (CP, ring attention), Data Parallel (DP via FSDP2 `fully_shard`). Configured through `DistributedCfg`. `pithtrain/modules/shutdown.py` installs fast-fail cleanup + a 180s NCCL heartbeat so a failed rank does not make peers wait on the watchdog.

### Model Layer Protocol (`pithtrain/models/interface.py`)

Models implement `ModelProtocol` with layers that expose `forward_attn`, `forward_mlp`, `forward_aggregate` — matching the 5-stage split. Supported models: DeepSeek-V2-Lite (`deepseek_v2_lite.py`), Qwen3-30B-A3B (`qwen3_30b_a3b.py`), GPT-OSS 20B/120B (`gpt_oss.py`).

### Optimized Operators (`pithtrain/operators/`)

- **MLA** (`mla/`) — Multi-head Latent Attention (Triton + TileLang)
- **Ring Attention** (`ring_attention/`) — `standard.py` and `mla.py` for context parallelism
- **FlashAttention v4** (`flash_attn_v4.py`) — Wrapper around the FA4 kernel
- **AllToAll** (`all_to_all.py`) — Differentiable collective wrapper
- **EP Dispatch** (`ep_dispatch.py`) — Fused Triton kernels and orchestration for expert-parallel token dispatch with deduplication
- **Token Scatter** (`token_scatter.py`) — Triton scatter kernels for grouping tokens by expert ahead of grouped GEMM
- **FP8 Quantization** (`deepgemm_fp8_quantize.py`) — Fused Triton kernels for DeepGEMM-style FP8 quantization
- **Fused activations / heads** — `silu_mul.py`, `clamped_swiglu.py`, `indexed_bias_add.py`, `cross_entropy.py`

Each operator ships a PyTorch reference impl for correctness testing.

### Training Orchestration (`pithtrain/tasks/pretrain_language_model.py`)

`PretrainLanguageModelCfg` composes `DistributedCfg`, `TrainingCfg`, and `LoggingCfg`. The training loop uses context managers (`distributed_context`, `training_context`, `logging_context`) to set up the full environment.

### Checkpointing (`pithtrain/modules/checkpoint.py`)

Handles checkpoint save/load with resharding between canonical (disk) format and localized (runtime) format. Canonical uses individual expert indices; localized stacks experts per EP rank with DualPipeV prefixes.

## Testing Gotchas

- `F.grouped_mm` may write NaN to padding rows (beyond `grouped_mm_offs[-1]`). Always truncate to `[:actual_M]` before comparing outputs.
- FP8 quantization tests use normalized squared-error (`calc_diff`), threshold typically `< 1e-3`.
- Tests skip gracefully when `deep_gemm` is not installed or CUDA is unavailable.
- Multi-GPU tests require `torchrun` (see `tests/test_fsdp.sh`).

## Config Base Classes

`pithtrain/config.py` defines `SlottedDefault` — all config/context dataclasses inherit from this. Uses `__init_subclass__` to enforce `slots=True` and provides default initialization.

## Agent Skills

This repo ships agent-native workflows under `.claude/skills/` (e.g. `add-new-model`, `capture-nsys-profile`, `validate-correctness`, `estimate-memory`, `setup-benchmark-inputs`, `add-memory-prints`, `launch-with-slurm`). When the user's request matches one of those, invoke the skill rather than re-deriving the workflow from scratch.
