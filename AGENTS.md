# AGENTS.md

This file provides guidance to coding agents (Claude Code and other assistants) when working with code in this repository. It is the canonical, vendor-neutral instruction file; `CLAUDE.md` simply imports it via `@AGENTS.md`.

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
pytest tests/operators/test_ring_attention.py -v

# Single test function
pytest tests/test_fp8_quantize_kernels.py::test_name -v

# Multi-GPU integration test — boots DualPipeV with FSDP, ~4 GPUs, pp=2 ep=2
bash tests/test_dualpipev.sh
```

### Benchmarks

```bash
python3 -m benchmarks.operators.fp8.test_deepgemm
# ring attention: multi-GPU launcher (torchrun), scenario = <model>-cp<N>-s<N>k
bash benchmarks/operators/bench_ring_attention.sh qwen3-30b-a3b-cp4-s32k
```

### Training & Data Prep

The `examples/` tree drives end-to-end workflows; each subdir has a `launch.sh` that auto-detects single-node vs. SLURM and forwards to a `<config>/script.py`.

```bash
# Tokenize a pretraining corpus (per-tokenizer; rerun when switching models)
bash examples/tokenize_corpus/launch.sh dclm-qwen3

# Pretrain (qwen3-30b-a3b | deepseek-v2-lite | gpt-oss-20b | gpt-oss-120b)
bash examples/pretrain_lm/launch.sh qwen3-30b-a3b

# Convert a training checkpoint to / from HuggingFace
bash examples/convert_checkpoint/launch.sh qwen3-30b-a3b
```

### Memory Estimation

```bash
python -m tools.memory_estimator --help   # peak-memory simulator for a given parallelism mesh
```

## Architecture

### DualPipeV Pipeline (`pithtrain/dualpipe/`)

The core pipeline assigns each rank two model chunks in a V-shape (the model is cut into `2 * pp_size` chunks; rank `r` holds chunks `r` and `2*pp_size-1-r`) and overlaps forward and backward execution across micro-batches. Each transformer layer is split into 5 stages:

1. **Attention** — LayerNorm + Attention + LayerNorm + Expert routing
2. **Dispatch** — All-to-all send tokens to assigned experts (async on comm stream)
3. **MLP** — Expert/MLP computation
4. **Combine** — All-to-all gather expert outputs (async on comm stream)
5. **Aggregate** — Weighted expert output + residual connection

Stages 1, 3, and 5 are the layer's compute entry points — `forward_stage1`, `forward_stage3`, `forward_stage5` on `LayerProtocol` (`pithtrain/models/interface.py`); stages 2 and 4 are all-to-all communication driven by the execution machinery (`pithtrain/dualpipe/execution.py`), not layer methods.

Key files:
- `dualpipev.py` — Main scheduler: `DualPipeV.step()` orchestrates overlapped F/B across modules (supports `forward_only=True` for inference), plus `layer_partition()`, which distributes decoder layers across pipeline stages — edge stages (which hold `embed_tokens` / `norm`+`lm_head`) get fewer layers to balance memory.
- `overlap.py` — `overlapped_forward_backward()` interleaved loop for one pair of micro-batches
- `execution.py` — Stage implementations (`stage1_f`, `stage1_b`, etc.), `ExecutionCtx`, and the dispatch/combine helpers
- `comm.py` — P2P communication setup between pipeline ranks
- `utils.py` — `FP8WeightCacheControl` (cache quantized weights across micro-batches), `WeightGradStore` (deferred wgrad for zero-bubble scheduling)

### FP8 Training

`TrainingCfg.fp8` selects the linear-layer backend: at training setup it binds `training.Linear` / `training.GroupedLinear` to the FP8 or BF16 classes. The DeepGEMM path (`pithtrain/operators/linear.py`) uses 128-element block scaling with E8M0 scale format, backed by custom Triton quantization kernels in `pithtrain/operators/deepgemm_quantize.py`. The BF16 grouped linear layer is in `pithtrain/operators/grouped_linear.py`.

### Distributed Parallelism (`pithtrain/modules/distributed.py`)

Four dimensions: Pipeline Parallel (PP), Expert Parallel (EP), Context Parallel (CP, ring attention), Data Parallel (DP via FSDP2 `fully_shard`). PP/CP/EP are configured through `DistributedCfg`; DP is inferred from the world size. The `(PP, DP, CP, EP)` device mesh and process groups are built in `setup_device_mesh`/`setup_default_process_group`. The same module installs a fail-fast excepthook plus an NCCL heartbeat timeout (driven by `DistributedCfg.timeout`, default 15 min) so a failed rank does not make peers wait on the watchdog.

### Model Layer Protocol (`pithtrain/models/interface.py`)

Models implement `ModelProtocol` with layers that expose `forward_stage1`, `forward_stage3`, `forward_stage5` — matching the 5-stage split. Supported models: DeepSeek-V2-Lite (`deepseek_v2.py`), Qwen3 MoE (`qwen3_moe.py`), GPT-OSS 20B/120B (`gpt_oss.py`).

### Tensor Layouts & Sequence Packing

The pipeline is **BSHD** end to end: hidden states are `(B, S, hidden)` through embeddings, norms, attention, MoE, residuals, the P2P activations between pipeline stages, and the loss. Nothing outside attention is aware of packing.

**Packed / variable-length (SFT) training** enforces `micro_batch_size == 1`. A micro-batch is `(1, S, hidden)` where `S` is the total token count and several documents are concatenated along the `S` axis; a per-sample `cu_seqlens` (int32 cumulative document offsets `[0, …, S]`) marks the boundaries. So the residual stream is semantically `TD` (`T = S` tokens × `hidden`) wearing a batch-of-one BSHD coat — kept 3-D on purpose so (a) every BSHD-assuming module (norm, MoE, embed, loss, P2P) is untouched and (b) DualPipeV can still micro-batch by scattering on `batch_dim=0` (the `B=1` is that seam).

**THD is confined to the attention kernel.** Only attention reshapes `(1, S, H, D) → (S, H, D)` (`squeeze(0)`) to hand FlashAttention-4's varlen entry (`operators/flash_attn_v4.py:flash_attn_varlen_func`) a flat `(total_tokens, H, D)` tensor plus `cu_seqlens`, then `unsqueeze(0)` back. Attention dispatches on `cu_seqlens is None` (dense causal) vs set (block-diagonal varlen). `max_seqlen` is passed as `S` — a compile-safe upper bound; FA4 schedules work off `cu_seqlens`, so over-allocating it is free and avoids a per-call device sync.

**Positions**: `forward_posemb(S, cu_seqlens)` builds per-document-reset positions when packed. Because RoPE is relative and the mask is block-diagonal, at `cp_size == 1` this reset is bit-identical to contiguous global positions — it becomes load-bearing only under context parallelism (deferred). **Loss**: boundary/prompt tokens are masked with `-100` (no separate loss-mask tensor) and the loss is token-weighted (the criterion returns the summed loss; the step divides by the global non-ignored token count).

`cu_seqlens` is broadcast along the PP group in `DualPipeV.step` and then threaded through the pipeline as an explicit forward argument (a sibling of `rotary_posemb`, not a module attribute), consumed only by attention and the per-document position reset in `forward_posemb`. The model, engine, and loss support variable-length packing for qwen3, deepseek-v2-lite, and gpt-oss, validated with synthetic `cu_seqlens` via `PACKED_SEQLEN=1 bash tests/test_dualpipev.sh <config>`; qwen3.5 (Gated DeltaNet) is pre-training only and asserts `cu_seqlens is None`. The packed **data pipeline** — turning a tokenized corpus into `cu_seqlens`-bearing samples, and choosing whole-document packing — is a deferred follow-up; no training entrypoint sets `cu_seqlens` yet, so pretraining is the only live data path.

### Optimized Operators (`pithtrain/operators/`)

- **Ring Attention** (`ring_attention.py`) — zigzag ring attention for context parallelism (standard and MLA-aware variants)
- **FlashAttention v4** (`flash_attn_v4.py`) — Wrapper around the FA4 kernel
- **MLA** — Multi-head Latent Attention is implemented inside the DeepSeek model (`models/deepseek_v2.py`), with MLA-aware ring attention in `ring_attention.py`
- **AllToAll** (`all_to_all.py`) — Differentiable collective wrapper
- **EP Dispatch** (`ep_dispatch.py`) — Fused Triton kernels and orchestration for expert-parallel token dispatch with deduplication
- **Token Scatter** (`token_scatter.py`) — Triton scatter kernels for grouping tokens by expert ahead of grouped GEMM
- **FP8 Quantization** (`deepgemm_quantize.py`) — Fused Triton kernels for DeepGEMM-style FP8 quantization
- **Fused activations / heads** — `silu_mul.py`, `clamped_swiglu.py`, `indexed_bias_add.py`, `cross_entropy.py`

Each operator ships a PyTorch reference impl for correctness testing.

### Training Orchestration (`pithtrain/tasks/pretrain_lm.py`)

`PretrainLMCfg` composes `DistributedCfg`, `TrainingCfg`, and `LoggingCfg`. `launch(cfg)` calls `setup_logging`, `setup_distributed`, and `setup_training` in turn, each of which populates its runtime context (see Runtime Contexts below).

### Task Module Convention (`pithtrain/tasks/`)

Each task module (`pretrain_lm`, `tokenize_corpus`, `convert_checkpoint`) exposes a `launch(cfg)` entry point plus a task-level `<Task>Cfg` (e.g. `PretrainLMCfg`, `TokenizeCorpusCfg`). Runtime state is not carried on a per-task context object; it lives in the process-global context modules (see Runtime Contexts below). Two rules keep this consistent:

- **Configs/contexts keep unique, descriptive names** (`PretrainLMCfg`, not bare `Cfg`) so they stay one-shot greppable — this repo is agent-native and `grep PretrainLMCfg` must locate every use. Import them by symbol: `from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch`.
- **`launch` is the generic verb** every task shares. In a single-task file import it directly; in a file that drives several tasks, import the *modules* and qualify the call (`tokenize_corpus.launch(...)`, `convert_checkpoint.launch(...)`) to avoid collisions. The composable building blocks (`DistributedCfg`, `TrainingCfg`, `LoggingCfg`) live in `pithtrain/modules/` and are always referenced by their descriptive names.

### Checkpointing (`pithtrain/modules/checkpoint.py`)

Handles checkpoint save/load with resharding between canonical (disk) format and localized (runtime) format. Canonical uses individual expert indices; localized stacks experts per EP rank with DualPipeV prefixes.

## Testing Gotchas

- `F.grouped_mm` may write NaN to padding rows (beyond `grouped_mm_offs[-1]`). Always truncate to `[:actual_M]` before comparing outputs.
- FP8 quantization tests use normalized squared-error (`calc_diff`), threshold typically `< 1e-3`.
- Multi-GPU tests require `torchrun` (see `tests/test_dualpipev.sh`).

## Config Base Classes

`pithtrain/config.py` defines `SlottedDefault` — all config/context dataclasses inherit from this. Subclasses are declared `@dataclass(init=False, slots=True)`; `SlottedDefault.__init__` auto-applies every field's default (leaving required fields unset), and `to_json_dict()` returns a JSON-serializable representation.

## Runtime Contexts (`pithtrain/contexts/`)

Process-global runtime state, one module per concern (`distributed`, `training`, `logging`). Each holds values that are set once at startup by the matching `setup_*` function and constant thereafter, so code reads them in-line instead of threading them through every constructor and call. Import the module and read fields off it (`distributed.ep_group`), never the bare names — a field does not exist until its `setup_*` runs, so `from ... import ep_group` fails at import and reading it before setup raises `AttributeError`.

## Agent Skills

This repository ships agent-native workflows under `.agents/skills/`, loaded automatically. Invoke a skill whenever a request matches one instead of re-deriving the workflow from scratch.

Skills are living documents, and you should keep them sharp as you work. When you notice any of the following, surface it for review rather than editing silently or derailing the task in progress:

- **Fix** a skill you followed that was stale, wrong, or incomplete.
- **Refine** a skill that worked but whose guidance could be sharper.
- **Propose** a new skill when you did repeatable, skill-worthy work that none covered.
