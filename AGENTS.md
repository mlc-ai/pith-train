# AGENTS.md

This file provides guidance to coding agents (Claude Code and other assistants) when working with code in this repository. It is the canonical, vendor-neutral instruction file; `CLAUDE.md` simply imports it via `@AGENTS.md`.

## Project Overview

**PithTrain** (package name: `pithtrain`) — a lightweight training framework for Mixture-of-Experts (MoE) language models. The core innovation is **DualPipeV**, an overlapped forward-backward pipeline parallelism technique. Requires Hopper (SM90) or Blackwell (SM100) GPUs. Python >= 3.12.

## Setup

```bash
uv sync
```

## Common Commands

### Linting & Formatting (Ruff)

```bash
ruff check --fix pithtrain/          # lint with auto-fix
ruff format pithtrain/               # format
pre-commit run --all-files       # run all pre-commit hooks
```

Style: 100-char line length, double quotes, `py312` target. Rules: E, F, I, W (ignoring E501, E731). First-party import: `pithtrain`. `pithtrain/kernels/` is not reformatted (`ruff format` and its pre-commit hook skip it) and its import blocks are not re-sorted (`I001` ignored there), so every statement without a comment (calls, signatures, imports, tables) stays on one line, with no line-length limit; multi-line PTX strings stay as written. Consecutive independent assignments of the same kind (same called function or operator, or one value split by different operators such as `tidx % 32, tidx // 32`; no right-hand side reading an earlier target) are written as one parallel assignment, `pid_row, pid_col = tl.program_id(0), tl.program_id(1)`, with no blank line between such groups. The rest of the lint rules still apply there.

### Testing (pytest)

```bash
# Kernel verifiers: every name in pithtrain.kernels.__all__ against PyTorch (one GPU)
pytest tests/kernels -v

# Single-GPU operator and layer tests
pytest tests/test_deepgemm_fp8_linear_correctness.py -v
pytest tests/test_grouped_linear_correctness.py -v
pytest tests/test_ep_dedup_dispatch.py -v
pytest tests/test_silu_mul.py tests/test_clamped_swiglu.py tests/test_indexed_bias_add.py tests/test_log_prob.py -v
pytest tests/operators -v              # operator-level tests (ring attention, CP sequence)
pytest tests/test_import_layers.py -v  # import layering, kernel naming, kernels interface and verifier coverage (CPU only, AST)

# Single test function
pytest tests/kernels/test_fp8_gemm.py::test_name -v

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

### DualPipeV Pipeline (`pithtrain/pipeline/`)

The core pipeline assigns each rank two model chunks in a V-shape (the model is cut into `2 * pp_size` chunks; rank `r` holds chunks `r` and `2*pp_size-1-r`) and overlaps forward and backward execution across micro-batches. Each transformer layer is split into 5 stages:

1. **Attention** — LayerNorm + Attention + LayerNorm + Expert routing
2. **Dispatch** — All-to-all send tokens to assigned experts (async on comm stream)
3. **MLP** — Expert/MLP computation
4. **Combine** — All-to-all gather expert outputs (async on comm stream)
5. **Aggregate** — Weighted expert output + residual connection

Stages 1, 3, and 5 are the layer's compute entry points — `forward_stage1`, `forward_stage3`, `forward_stage5` on `LayerProtocol` (`pithtrain/models/interface.py`); stages 2 and 4 are all-to-all communication driven by the execution machinery (`pithtrain/pipeline/execution.py`), not layer methods.

Key files:
- `dualpipev.py` — Main scheduler and P2P between pipeline ranks: `DualPipeV.step()` orchestrates overlapped F/B across modules (forward-only when called under `torch.no_grad()`, e.g. for inference), `overlapped_forward_backward()` is the interleaved loop for one pair of micro-batches, plus `layer_partition()`, which distributes decoder layers across pipeline stages — edge stages (which hold `embed_tokens` / `norm`+`lm_head`) get fewer layers to balance memory.
- `execution.py` — Stage implementations (`stage1_f`, `stage1_b`, etc.), `ExecutionCtx`, and the dispatch/combine helpers; it drives `WeightGradStore` (deferred wgrad for zero-bubble scheduling, `pithtrain/contexts/weight_grad_store.py`)

### FP8 Training

`TrainingCfg.fp8` selects the linear-layer backend: at training setup it binds `training.Linear` / `training.GroupedLinear` to the FP8 or BF16 classes. The DeepGEMM path (`pithtrain/operators/linear.py`, `pithtrain/operators/grouped_linear.py`) uses 128-element block scaling with E8M0 scale format. The quantization kernels and the DeepGEMM recipes (`fp8_quant_*`, `fp8_gemm_nt`, `fp8_gemm_fwd`, `fp8_gemm_bwd`, `fp8_grouped_gemm`, `k_grouped_fp8_gemm`, `fp8_grouped_wgrad_empty`) are imported from `pithtrain.kernels`, which picks the SM90 or SM100 implementation itself (see Kernels below). The FP8 expert ops (activation fused into the quantization) are in `pithtrain/operators/fp8_experts.py`, taken when `fp8_fused_experts_supported()`; elsewhere FP8 experts take the unfused `FP8GroupedLinear` path. The BF16 grouped linear layer is in `pithtrain/operators/grouped_linear.py`, and where `bf16_experts_supported(num_local_experts)` says yes the BF16 experts take the fused CuTe-DSL path of `pithtrain/operators/bf16_experts.py`.

### Distributed Parallelism (`pithtrain/modules/distributed.py`)

**MoE parallel folding** ([arXiv:2504.14960](https://arxiv.org/abs/2504.14960) section 3.2): attention and the experts get independent parallelism over the same ranks. `setup_device_mesh` builds two device meshes, `attn_mesh` as `(pp, dp, cp)` and `expt_mesh` as `(pp, dp, ep)`, which factor the `world_size // pp` ranks of one pipeline stage two different ways, so `dp * cp == ep * expt_dp == world_size // pp`. PP is the one axis the two views share, and it stays outermost in each. PP/CP/EP are configured through `DistributedCfg`; both DP degrees are derived.

Properties worth remembering: `cp` and `ep` each need only divide the stage size, not each other, so EP may span CP; both the CP group and the EP group are contiguous rank blocks, so each stays on NVLink independently; and **`dp_rank` alone decides which data a rank loads**, while `ep_rank` names only the experts a rank hosts. FSDP shards the expert parameters (the routed expert weights) over `expt_mesh["dp"]` and the attention parameters (everything else) over the flattened `attn_mesh["dp", "cp"]`, with a plain-sum gradient reduction on both, so the single divide by the global token count in the training step is the only normalization. `DistributedCfg.hsdp_replica` above 1 splits both of those groups into that many replicas, sharding within one and all-reducing across them. A process group is published only for axes that a collective runs over directly (`pp`, `cp`, `ep`); both `dp` axes reduce through FSDP, which takes a mesh.

`setup_default_process_group` also installs a fail-fast excepthook and passes `DistributedCfg.timeout` (default 15 min) as the collective timeout, so a failed rank does not make peers wait. Both belong to creating the world, so a process that inherits one from its host keeps whatever that host chose.

**Two CP sequence layouts (`pithtrain/operators/cp_sequence.py`).** Softmax attention shards the sequence **zigzag** (rank `r` holds blocks `r` and `2 * cp_size - r - 1` of `2 * cp_size`) to balance causal work; a linear-attention recurrence instead needs **contiguous** (blocks `2r` and `2r + 1`) so that rank order is sequence order. Hybrid models (Qwen3.5) convert with one all-to-all at each boundary between the two kinds of layer, hoisted to the ends of each run of linear-attention layers. The invariant: **hidden states are zigzag at model boundaries and at every attention layer, contiguous within a run of linear-attention layers**; norms, MoE and residuals are per-token and layout-blind. `zigzag_spans(cp_rank, cp_size, seq_len)` is the single definition of the zigzag partition, returning the two global token ranges a rank holds, and both the loader and `forward_posemb` call it. Restating that arithmetic elsewhere mistrains silently rather than raising. A packed sample is sharded the same way one document at a time, the caller padding each document at its end to a multiple of `2 * cp_size` tokens (pad labels `-100`), and `zigzag_varlen_index(cp_rank, cp_size, cu_seqlens, seq_len)` is that partition's single definition, the global index of every token a rank holds: `forward_posemb` calls it, and a packed loader and a routing-replay store must call it too.

Replayed routes follow the loader's zigzag order. Each linear-attention layer converts them to contiguous order through `Qwen35MoeDecoderLayer.resolve_replay`.

### Model Layer Protocol (`pithtrain/models/interface.py`)

Models implement `ModelProtocol` with layers that expose `forward_stage1`, `forward_stage3`, `forward_stage5` — matching the 5-stage split. Supported models: DeepSeek-V2-Lite (`deepseek_v2.py`), Qwen3 MoE (`qwen3_moe.py`), GPT-OSS 20B/120B (`gpt_oss.py`).

### Tensor Layouts & Sequence Packing

The pipeline is **BSHD** end to end: hidden states are `(B, S, hidden)` through embeddings, norms, attention, MoE, residuals, the P2P activations between pipeline stages, and the loss. Nothing outside attention is aware of packing.

**Packed / variable-length (SFT) training** enforces `micro_batch_size == 1`. A micro-batch is `(1, S, hidden)` where `S` is the total token count and several documents are concatenated along the `S` axis; a per-sample `cu_seqlens` (int32 cumulative document offsets `[0, …, S]`) marks the boundaries. So the residual stream is semantically `TD` (`T = S` tokens × `hidden`) wearing a batch-of-one BSHD coat — kept 3-D on purpose so (a) every BSHD-assuming module (norm, MoE, embed, loss, P2P) is untouched and (b) the caller can still split the batch into micro-batches along dim 0 (the `B=1` is that seam).

**THD is confined to the attention kernel.** Only attention reshapes `(1, S, H, D) → (S, H, D)` (`squeeze(0)`) to hand FlashAttention-4's varlen entry (`operators/flash_attn_v4.py:flash_attn_varlen_func`) a flat `(total_tokens, H, D)` tensor plus `cu_seqlens`, then `unsqueeze(0)` back. Attention dispatches on `cu_seqlens is None` (dense causal) vs set (block-diagonal varlen), and under context parallelism the same squeeze feeds `ring_attention_varlen_func`, the zigzag ring made block-diagonal by document. `max_seqlen` is passed as `S` — a compile-safe upper bound; FA4 schedules work off `cu_seqlens`, so over-allocating it is free and avoids a per-call device sync.

**Positions**: `forward_posemb(S, cu_seqlens)` builds per-document-reset positions when packed; under context parallelism it takes each token's global index from `zigzag_varlen_index`, so positions follow the tokens wherever the partition puts them. Because RoPE is relative and the mask is block-diagonal, the reset changes what attention computes only by rounding; it keeps positions within the longest document rather than the packed length, as an inference engine counts them. **Loss**: boundary/prompt tokens are masked with `-100` (no separate loss-mask tensor) and the loss is token-weighted (the objective returns the loss summed over the micro-batch's tokens; the step divides by the global non-ignored token count).

`cu_seqlens` rides on each `Microbatch`, which every pipeline rank receives, and is threaded through the pipeline as an explicit forward argument (a sibling of `rotary_posemb`, not a module attribute), consumed only by attention and the per-document position reset in `forward_posemb`. It always holds the boundaries of the whole packed sample, also under context parallelism, where the rest of the micro-batch holds this rank's share of every document. The packed **data pipeline** — turning a tokenized corpus into `cu_seqlens`-bearing samples, and choosing whole-document packing — is a deferred follow-up; no training entrypoint sets `cu_seqlens` yet, so pretraining is the only live data path.

### Operators (`pithtrain/operators/`)

Only what training code calls directly: autograd functions and custom ops with their functional entry points, `nn.Module` layers, and collective wrappers. The kernels they launch live in `pithtrain/kernels/` (below).

- **Linear layers** (`linear.py`, `grouped_linear.py`) — BF16 / FP8 dense and grouped (expert) linear layers, with fp32 main_grad accumulation of the weight gradient
- **Expert ops** — `fp8_experts.py` (FP8 SwiGLU, fused gate_up SwiGLU and gpt-oss clamped SwiGLU experts as single autograd nodes, plus `fp8_scatter_quant`, the fused stage-3 input), `bf16_experts.py` (SM90-only fused BF16 CuTe experts; models import it lazily after `bf16_experts_supported()`), `fp8_swiglu_mlp.py` (fused FP8 dense SwiGLU MLP), `main_grad.py` (fp32 main_grad of raw expert parameters)
- **Ring Attention** (`ring_attention.py`) — zigzag ring attention for context parallelism (standard, packed and MLA-aware variants)
- **CP Sequence** (`cp_sequence.py`) — defines both CP sequence layouts, exposing the zigzag partition as `zigzag_spans` (and, one packed document at a time, as `zigzag_varlen_index`), converts the hidden stream between them, and fetches the conv state a depthwise convolution needs, the tokens before a shard, from the previous rank
- **FlashAttention v4** (`flash_attn_v4.py`) — custom ops around the FA4 kernel (host-side launch memoization: `memoized_launch`)
- **MLA** — Multi-head Latent Attention is implemented inside the DeepSeek model (`models/deepseek_v2.py`), with MLA-aware ring attention in `ring_attention.py`
- **Linear attention** — `gated_delta_rule.py` (Qwen3.5 Gated DeltaNet, including the dense path through FLA's kernels with pinned configs), `causal_conv1d.py` (channels-last causal depthwise conv)
- **AllToAll** (`all_to_all.py`) — Differentiable collective wrapper
- **EP Dispatch** (`ep_dispatch.py`) — orchestration of expert-parallel token dispatch with deduplication (index math: `ep_dispatch_send_idx`, `ep_dispatch_recv_idx`)
- **Token Scatter** (`token_scatter.py`) — scatter of tokens by expert ahead of the grouped GEMM, padded index gather, per-row group ids
- **Fused activations / heads** — `silu_mul.py`, `clamped_swiglu.py`, `indexed_bias_add.py`, `cross_entropy.py`, `log_prob.py`

Operators import kernels only as `from pithtrain.kernels import <name>` and never branch on the architecture: where a fused path may be missing they call a capability query (`fp8_fused_experts_supported`, `bf16_experts_supported`, `gated_delta_rule_pinned_supported`). Operators keep a PyTorch reference for correctness testing (operator-level tests in `tests/operators/` and `tests/`); every kernel they call is verified against a pure-PyTorch reference in `tests/kernels/test_<feature>.py`.

### Kernels (`pithtrain/kernels/`)

Raw kernel code: `@triton.jit` and CuTe-DSL kernels, their launchers, tile / config tables, scale layouts and the helper math they share. The layering is `kernels → contexts → operators → {modules, pipeline}`: `kernels/` imports only third-party code and other `kernels/` modules (never `contexts/`, `operators/`, `modules/`, `models/` or `pipeline/`, lazy imports included). `tests/test_import_layers.py` checks this and the naming rules statically.

**A black-box library used only through `__all__`.** The whole interface is `__all__` in `pithtrain/kernels/__init__.py`, which holds nothing but plain `from pithtrain.kernels.<module> import <names>` lines and `__all__` listing exactly those names, built with `__all__.append(...)` calls grouped under a comment per feature area. Importing the package never loads cutlass, QuACK or FA4: every module it imports keeps them inside the functions that run them (the pickers import their `_sm90` modules in the SM90 branch; the SM90-only BF16 experts are reached through the plain wrappers `bf16_expert_gemm.py` / `bf16_expert_slots.py`, whose functions import `bf16_expert_*_sm90` on first call), and `tests/test_import_layers.py` checks it. Outside the package the only allowed import is `from pithtrain.kernels import <name in __all__>`, never a submodule (`pithtrain.kernels.fp8_gemm`, a picker, a plain file or an `_smXX` file); `tests/test_import_layers.py` enforces it, and that nothing outside `kernels/` branches on the architecture. The library owns the SM90 / SM100 choice: it exports no architecture query (`ARCH_MAJOR`, `is_sm90`, ... stay inside `arch.py`), and where a caller needs to know whether a fused path exists it calls a capability query named for the feature: `fp8_fused_experts_supported()`, `bf16_experts_supported(num_groups)`, `gated_delta_rule_pinned_supported(q, k, v, g, beta)`. Optional hints return None where the device needs none (`fp8_group_indices`, `fp8_shard_amax_multi`, `fp8_scales_multi`).

Every exported name has a docstring stating its contract (shapes, dtypes, contiguity, output and scale layout, in-place behaviour, SM90-only or not) and a pure-PyTorch verifier in `tests/kernels/test_<feature>.py` (`<feature>` = the stem of the module `__init__.py` imports the name from, without `_sm90` / `_sm100`). A verifier imports from pithtrain only `from pithtrain.kernels import ...`, writes its reference in plain PyTorch in the test, and covers the feature's forward and backward entry points across ragged, empty and non-multiple-of-128 shapes and the four models' sizes. FP8 quantization is compared exactly (the same e4m3 bytes and E8M0 scale values as the recipe: per 128-element block `scale` = the power of two at or above `max(amax, 1e-4) * fp32(1 / 448)` (at least 2^-126), `e4m3_satfinite(x * (1 / scale))` round to nearest even); GEMMs and other float outputs use `calc_diff` or tight tolerances. SM90-only features skip on their capability query. `tests/test_import_layers.py` checks that every name in `__all__` is exercised in `tests/kernels/`. Adding a kernel means: import it in `__init__.py` and list it in `__all__`, write its contract docstring, add its verifier.

Launch-config tuning (`kernels/tuning.py`): a kernel family with several valid launch configs is a `Tunable` (today the BF16 expert GEMMs), whose launches ask `tunable.pick(key, rows)` for a config: one table lookup once tuned. `pretrain_lm.launch` calls `tune_kernels()` on every rank before the first step (installing the choices cached for this setup, so a repeated setup runs tuned from step 1) and after it; a key not in the cache runs its fixed default in the first step, the same on every rank, and the second call pools the keys all ranks used, splits the timing of the candidates across the ranks on synthetic inputs (dropping any whose results differ from the default's), pools the timings and installs the same choice on every rank, checked by a hash. Choices are cached in a file every node sees (`~/.cache/pithtrain/tuning`, or `PITHTRAIN_TUNING_DIR`), named by the GPU and a fingerprint of the library versions, the candidates and the kernel module's source, so tuning is paid once per setup and redone when a kernel changes. `tune_kernels` has no verifier of its own: `tests/kernels/test_bf16_expert_gemm.py` runs a round and checks the GEMMs again (`VERIFIER_OF` in `tests/test_import_layers.py`).

Scale-layout contract: every 1x128 FP8 scale has logical shape `(rows, ceil(K / 128))` fp32 with unspecified strides (MN-major on SM90, the layout DeepGEMM's SM90 GEMMs read without a transpose, `fp8_scale_layout.py`; row-major on SM100). Callers pass scales to this library's GEMMs as returned and never read their strides (`act_scale_like` gives the device's layout for fakes). 128x128 weight scales are row-major on both.

SM100 gap: no Blackwell hardware is available yet, so `tests/kernels` verifies only the path an H200 runs. The `_sm100` modules as SM100 compiles them (native E8M0 scales, row-major scales into DeepGEMM), `fp8cast_rowwise_colwise`, the SM100 branches of `fp8_gemm.py` and DeepGEMM's SM100 entry points (psum-layout m-grouped, TN k-grouped) are untested, as are the SM100 placements in atomic order (`scatter_quant`, `ep_dispatch`). On Blackwell, start from `pytest tests/kernels`: cases that assert the H200 path (capability queries true, K-major grouped layouts, stable row order) need SM100 counterparts. Do not add an architecture override to test `_sm100` code on Hopper.

Naming (SM90 and SM100 are the only supported targets):

1. `kernels/` is flat (no subfolders). Its `__init__.py` holds only the re-export imports and `__all__`; `operators/__init__.py` is empty.
2. A feature with an SM90-specific kernel has exactly two files: `<feature>_sm90.py` (the CuTe-DSL implementation) and `<feature>_sm100.py` (the Triton implementation, kept until a Blackwell kernel replaces it).
3. `<feature>_sm90.py` imports its `_sm100` sibling only as a documented fallback for inputs it does not cover (dtype, contiguity, shape). Named cross-feature edges: the fused SwiGLU / clamped-SwiGLU / scatter quantizations fall back to the plain activation or scatter kernels plus `fp8_rowwise_kmajor_sm90`, which keeps today's SM90 kernels on that path.
4. Kernels both architectures run unchanged are plain `<feature>.py` (`silu_mul`, `clamped_swiglu`, `cross_entropy`, `segment_sum`, `group_bias_add`, `token_scatter`, `causal_conv1d`, `fp8_rowwise_colwise`, `fp8_gemm`, `flash_attn_memoize`, ...).
5. Only `kernels/arch.py` checks the device capability (`ARCH_MAJOR`, `NATIVE_E8M0`, `SM_COUNT`). Each feature has one picker, `kernels/<feature>.py`, that dispatches with `match ARCH_MAJOR:`: `case 9:` runs `_sm90`, `case 10:` runs `_sm100`, and `case _:` raises `NotImplementedError` naming the compute capability (a new architecture gets its own `case`, e.g. `case 11:` for SM110); `__init__.py` re-exports the picker's contract names. `operators/`, `models/`, `modules/`, tests, benchmarks and tools import those names from the package (`from pithtrain.kernels import ...`), never a picker module, a plain file or an `_smXX` file.
6. Shared helpers only SM90 code uses carry the suffix: `cute_ptx_sm90.py` (inline-PTX `dsl_user_op`s), `cute_compile_sm90.py` (fake tensors / stream, `TORCH2CUTE`), `fp8_e8m0_scale_sm90.py` (E8M0 scale math). Arch-neutral shared leaves stay unsuffixed: `arch.py`, `host_transfer.py` (pinned host buffers), `fp8_scale_layout.py`, `fp8_scale.py` (the Triton E8M0 scale).
7. Tests mirror the feature names without suffix: `tests/kernels/test_<feature>.py`, one per feature `__init__.py` imports from, verified against PyTorch through the interface (never against an `_sm100` sibling); an SM90-only feature's file skips on its capability query.
8. A feature stays one file per architecture, forward and backward together. Split it into `<feature>_fwd_sm90.py` / `<feature>_bwd_sm90.py` only when one direction becomes its own pipeline or the file passes about 1.5k lines.
9. No `_name` defined anywhere in `kernels/` (module level, classes, methods, attributes, locals, parameters, import aliases; reads of a library's own `_` hooks such as QuACK's are fine): every kernel module lists what it exports in a literal `__all__`, and a kernel module imports from a sibling only names in that sibling's `__all__` (`tests/test_import_layers.py` checks both). Docstrings are plain text wrapped at 100 columns, with the opening and closing `"""` each on its own line (one-line docstrings too).

A picker exports the contract name (`fp8_quant_grad`) and the implementations the same name with the file's suffix (`fp8_quant_grad_sm90`, `fp8_quant_grad_sm100`), so each is unique to grep; only contract names can be in `__all__`, and a contract the callers no longer need stays internal (`fp8cast_rowwise_kmajor` is reached through `fp8_quant_grouped`).

SM90-only features have no `_sm100` file, because SM100 runs no kernel there: `group_indices_sm90` (behind `fp8_group_indices`, None off SM90), `bf16_expert_gemm_sm90` and `bf16_expert_slots_sm90` (gate: `bf16_experts_supported`; their exported names carry a `bf16_expert_` prefix), `gated_delta_rule_pinned_sm90` (gate: `gated_delta_rule_pinned_supported`). No operator imports an `_sm90` file: `operators/bf16_experts.py` and `operators/gated_delta_rule.py` import the package names after the gate says yes.

Every `_sm90` module that loads cutlass is imported lazily, inside the SM90 branch of its picker or inside the wrapper function of an SM90-only feature: QuACK patches CuTe-DSL globally when imported and cutlass is expensive to import, so importing `pithtrain.kernels` or a picker on SM100 never imports either.

Bit-exactness contracts: each `_sm90` kernel produces exactly the FP8 bytes and E8M0 scale values of the PyTorch implementation of the recipe (the SM90 branch of the E8M0 scale, `cvt.rn.satfinite` e4m3), which its `tests/kernels` verifier checks; its `_sm100` sibling computes the same values and stays the documented fallback for inputs the SM90 kernel does not cover. Scale strides differ by architecture as the scale-layout contract above says; each picker's docstring says so.

### Training Orchestration (`pithtrain/tasks/pretrain_lm.py`)

`PretrainLMCfg` composes `DistributedCfg`, `TrainingCfg` and `LoggingCfg`, plus a `dataset` path of its own. `launch(cfg)` calls `setup_logging`, `setup_distributed` and `setup_training` in turn, each of which populates its runtime context (see Runtime Contexts below). The corpus and the step counter are neither context nor shared config: only a task that reads a corpus has any use for them, so `setup_dataset` lives in `pretrain_lm.py`, returns the dataset, and `launch` owns it alongside the step counter.

The FSDP2 hooks that are neither ops nor kernels live in `pithtrain/modules/`: `fsdp_forward.py` (lean forward hooks), `fp8_param_gather.py` (FP8 parameter all-gather, including gpt-oss's raw expert weights) and `wgrad_accumulation.py` (fp32 main_grad accumulation of the weight gradients, and of raw expert parameters).

### Task Module Convention (`pithtrain/tasks/`)

Each task module (`pretrain_lm`, `tokenize_corpus`, `convert_checkpoint`) exposes a `launch(cfg)` entry point plus a task-level `<Task>Cfg` (e.g. `PretrainLMCfg`, `TokenizeCorpusCfg`). Runtime state is not carried on a per-task context object; it lives in the process-global context modules (see Runtime Contexts below). Two rules keep this consistent:

- **Configs/contexts keep unique, descriptive names** (`PretrainLMCfg`, not bare `Cfg`) so they stay one-shot greppable — this repo is agent-native and `grep PretrainLMCfg` must locate every use. Import them by symbol: `from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch`.
- **`launch` is the generic verb** every task shares. In a single-task file import it directly; in a file that drives several tasks, import the *modules* and qualify the call (`tokenize_corpus.launch(...)`, `convert_checkpoint.launch(...)`) to avoid collisions. The composable building blocks (`DistributedCfg`, `TrainingCfg`, `LoggingCfg`) live in `pithtrain/modules/` and are always referenced by their descriptive names.

### Checkpointing (`pithtrain/modules/checkpoint.py`)

Handles checkpoint save/load with resharding between canonical (disk) format and localized (runtime) format. Canonical uses individual expert indices; localized stacks experts per EP rank with DualPipeV prefixes.

## Testing Gotchas

- `F.grouped_mm` may write NaN to padding rows (beyond `grouped_mm_offs[-1]`). Always truncate to `[:actual_M]` before comparing outputs.
- FP8 GEMM and FP8 layer tests use normalized squared-error (`calc_diff`), threshold typically `< 1e-3`; FP8 quantization verifiers compare bytes and scale values exactly, and scales by logical value only (strides are unspecified).
- Multi-GPU tests require `torchrun` (see `tests/test_dualpipev.sh`).
- Kernel verifiers call only `pithtrain.kernels` names and compare against a PyTorch reference written in the test, never against a picker's `_sm100` sibling (the sibling is a fallback, not the specification). Do not add an architecture override to run `_sm100` code on Hopper.

## Config Base Classes

`pithtrain/config.py` defines `SlottedDefault` — all config/context dataclasses inherit from this. Subclasses are declared `@dataclass(init=False, slots=True)`; `SlottedDefault.__init__` auto-applies every field's default (leaving required fields unset), and `to_json_dict()` returns a JSON-serializable representation.

## Runtime Contexts (`pithtrain/contexts/`)

Process-global runtime state, one module per concern (`distributed`, `training`, `logging`). Each holds values that are set once at startup by the matching `setup_*` function and constant thereafter, so code reads them in-line instead of threading them through every constructor and call. Import the module and read fields off it (`distributed.ep_group`), never the bare names — a field does not exist until its `setup_*` runs, so `from ... import ep_group` fails at import and reading it before setup raises `AttributeError`.

Two per-step stores are the exception to "constant after setup": `weight_grad_store.py` (`WeightGradStore`, the deferred weight-gradient closures of the zero-bubble W phase) and `fp8_weight_cache.py` (`FP8WeightCacheControl`, the version counter that lets FP8 layers reuse a quantized weight across the micro-batches of a step). `DualPipeV.step` drives both; import them by class name.

## Agent Skills

This repository ships agent-native workflows under `.agents/skills/`, loaded automatically. Invoke a skill whenever a request matches one instead of re-deriving the workflow from scratch.

Skills are living documents, and you should keep them sharp as you work. When you notice any of the following, surface it for review rather than editing silently or derailing the task in progress:

- **Fix** a skill you followed that was stale, wrong, or incomplete.
- **Refine** a skill that worked but whose guidance could be sharper.
- **Propose** a new skill when you did repeatable, skill-worthy work that none covered.
