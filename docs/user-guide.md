# PithTrain User Guide

This guide is for **users training models with PithTrain**. The
[README](../README.md) has the minimal commands to get a run going; this guide
fills in the details around them: what hardware you need, the models available,
how to configure and scale a run, how to read the output, and how to recover
when something goes wrong.

If you want to understand *how the framework works internally* (to modify it),
see [`architecture.md`](architecture.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md)
instead.

---

## Requirements

- **GPU:** NVIDIA Hopper (SM90, e.g. H100) or Blackwell (SM100, e.g. B200).
  Other architectures are not supported.
- **CUDA:** >= 13.0.
- **Python:** ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/).
- **Multiple GPUs:** the example configs assume one 8-GPU node, and the
  framework is designed for multi-GPU training. You can lower the parallelism
  degrees for fewer GPUs, but the smaller meshes are not the tested defaults.

Install (users):

```bash
git clone https://github.com/mlc-ai/pith-train.git && cd pith-train
uv venv
uv pip install .
```

---

## Supported models

| Model | Total / active params | Example dir | Default mesh | GPUs |
|---|---|---|---|---|
| DeepSeek-V2-Lite | ~16B / ~2.4B | `deepseek-v2-lite` | `pp=1, ep=8` | 8 (1 node) |
| Qwen3-30B-A3B | ~30B / ~3B | `qwen3-30b-a3b` | `pp=1, ep=8` | 8 (1 node, H200/B200) |
| GPT-OSS-20B | ~21B / ~3.6B | `gpt-oss-20b` | `pp=1, ep=8` | 8 (1 node) |
| GPT-OSS-120B | ~117B / ~5B | `gpt-oss-120b` | `pp=4, ep=8` | 32 (4 nodes) |

Example dirs live under `examples/pretrain_lm/<dir>/`. The default
meshes are starting points — see [Scaling](#scaling-a-run) to change them.
DeepSeek-V2-Lite and GPT-OSS-20B fit any single 8-GPU node. **Qwen3-30B-A3B at
the default `pp=1, ep=8` needs a high-memory single node — 8×H200 (141 GB) or
8×B200 (180 GB); it does not fit 8×H100 (80 GB), where you should instead use
two nodes with `pp=2, ep=8` (16 GPUs).** **GPT-OSS-120B requires multiple nodes.**

---

## End-to-end workflow

The four commands in the README map onto these stages. Each example dir is
self-contained: a `script.py` (the run config) and a `config.json` (the model
architecture).

**1. Tokenize the corpus.** Tokenization is per-tokenizer, so you run it once
per model family. Output lands in `workspace/datasets/dclm-baseline/toktxt/<model>/`.

```bash
bash examples/tokenize_corpus/launch.sh dclm-qwen3   # see examples/tokenize_corpus/ for other tokenizers
```

**2. Configure.** Edit `examples/pretrain_lm/<model>/script.py` for
parallelism, batch size, learning rate, etc. (see [Configuring a run](#configuring-a-run)).
The model architecture is in the sibling `config.json`.

**3. Train.** The launcher auto-detects single-node vs. SLURM and forwards to
`torchrun`. Logs are written to `logging/pretrain_lm/`.

```bash
bash examples/pretrain_lm/launch.sh qwen3-30b-a3b
```

Training **resumes automatically** from the latest checkpoint in
`save_location` if one exists, so re-running the same command continues a run.

**4. Export.** Convert a training checkpoint to HuggingFace format for
evaluation or inference (the same tool also imports HF checkpoints for continued
pretraining):

```bash
bash examples/convert_checkpoint/launch.sh qwen3-30b-a3b
```

---

## Configuring a run

A run is configured by editing `script.py` — there are no command-line flags to
memorize; the file *is* the config. The knobs that matter most:

| Field | Meaning |
|---|---|
| `distributed.pipeline_parallel_size` (PP) | Pipeline stages across ranks. |
| `distributed.expert_parallel_size` (EP) | MoE experts distributed across ranks. |
| `distributed.context_parallel_size` (CP) | Shards the sequence dimension (long context). |
| `distributed.sharding_strategy` | `"fsdp"` (lowest memory) or `"hsdp"` (replicate across DP). |
| `training.micro_batch_size` | Sequences per micro-batch (per rank). |
| `training.global_batch_size` | Total sequences per step; gradient-accumulated over micro-batches. |
| `training.sequence_length` | Tokens per sequence. |
| `training.max_lr` / `min_lr` / `warmup_steps` | Cosine LR schedule (linear warmup → cosine decay). |
| `training.max_steps` | Total optimizer steps. |
| `training.fp8_training` | `"disabled"` (BF16) or `"deep-gemm"` (FP8). See [FP8 training](#fp8-training). |
| `training.moe_load_balance_type` / `moe_load_balance_coef` | MoE load-balance loss (`"global-batch"`, `"sequence"`, `"micro-batch"`); coefficient `0` disables. |
| `training.save_interval` / `save_location` | Checkpoint cadence and directory. |
| `logging.wandb` | Optional Weights & Biases logging (set entity/project, or comment out). |

**Data-parallel (DP) is not set directly** — it is inferred:
`dp = total_gpus / (pp × cp × ep)`.

**Profiling a few steps.** To capture an Nsight Systems trace, set
`training.nsys_start` and `training.nsys_stop`: the CUDA profiler runs from the
start of `nsys_start` up to (but not including) `nsys_stop`, so
`nsys_start=N, nsys_stop=N+1` profiles a single step `N`. Both default to `None`
(disabled). Analogous `training.memory_profile_start` / `memory_profile_stop`
fields drive the CUDA memory profiler.

---

## Scaling a run

The one hard constraint: **`pp × cp × ep` must divide your total GPU count**;
whatever is left over becomes DP. Some worked examples on an 8-GPU node:

| Goal | Mesh |
|---|---|
| Single node, max expert sharding | `pp=1, ep=8` → `dp=1` |
| Single node, some data parallelism | `pp=1, ep=4` → `dp=2` |
| Two nodes (16 GPUs), pipeline + experts | `pp=2, ep=8` → `dp=1` |
| Long sequences | raise `cp` (e.g. `cp=2`), which shards the sequence via ring attention |

**Multi-node (SLURM).** The same launcher works under `srun` — it reads
`SLURM_*` env vars to build the `torchrun` rendezvous automatically:

```bash
srun -W 0 examples/pretrain_lm/launch.sh qwen3-30b-a3b
```

**Sizing memory before you launch.** Use the estimator to check a mesh fits
before spending GPU time:

```bash
python -m tools.memory_estimator --help
```

---

## Reading the training output

Rank 0 prints one line per step:

```
step 00000123/00004096 | step-time 1.234 sec | cross-entropy-loss 7.8901 | load-balance-loss 1.012345 | learning-rate 3.000000e-04 | gradient-norm 0.9876 | tokens-per-second 1,234,567 | peak-gpu-memory 62.34 GB
```

- **cross-entropy-loss** — the training loss; should trend down.
- **load-balance-loss** — MoE expert balance; **1.0 is perfect balance** (this
  is the metric with the coefficient divided out, matching Megatron's
  convention). Much larger than 1.0 means experts are imbalanced.
- **tokens-per-second** — throughput (`global_batch_size × sequence_length / step-time`).
- **peak-gpu-memory** — max allocated this step; watch this when tuning the mesh.

If `logging.wandb` is configured, the same metrics are logged to Weights &
Biases.

---

## Checkpoints

- **Resuming** is automatic: a run loads the latest
  `<save_location>/torch-dcp/step-XXXXXXXX` on startup.
- **Reshardable:** checkpoints are stored in a parallelism-independent format, so
  you can resume the same run under a *different* PP/EP/DP layout (e.g. start on
  one node, continue on two).
- **Export to HuggingFace** with `convert_checkpoint` for downstream
  evaluation/inference with standard tooling.
- **Import from HuggingFace** with the same tool to start from released weights
  (continued pretraining). Imported checkpoints carry no optimizer state, which
  the loader handles.

---

## FP8 training

Set `training.fp8_training = "deep-gemm"` to train in FP8 (128-element block
scaling via DeepGEMM; Hopper and Blackwell). This requires the `deep_gemm`
package to be installed. Leave it `"disabled"` for BF16, which has no extra
dependency. FP8 reduces memory and can improve throughput; validate loss parity
against BF16 for your setup before committing to a long run.

---

## Troubleshooting / FAQ

**"Dataset is too small for this run."** Your run needs `max_steps ×
global_batch_size` samples but the tokenized corpus has fewer. Tokenize more
DCLM shards, or lower `max_steps` / `global_batch_size`.

**`world_size not divisible by pp × cp × ep`.** Adjust the mesh so the product
divides your GPU count (see [Scaling](#scaling-a-run)).

**Out of memory.** `micro_batch_size` is already 1 in the examples; from there,
increase `ep` (or `pp`, or add nodes for more DP), shorten `sequence_length`, or
enable FP8. Run `tools.memory_estimator` to find a mesh that fits.

**`deep_gemm` import error.** The FP8 path needs DeepGEMM installed. Either
install it or set `training.fp8_training = "disabled"`.

**A run hangs after one rank fails.** PithTrain installs a fail-fast excepthook
and an NCCL heartbeat (driven by `distributed.timeout`, default 15 min) so a
crashed rank brings the job down instead of leaving peers to hang. On multi-node
runs, raise `distributed.timeout` if legitimate collectives are slower than the
heartbeat.

**I switched models and tokenization looks wrong.** Re-run the tokenization step
— it is per-tokenizer, and each model reads from its own
`workspace/datasets/dclm-baseline/toktxt/<model>/` directory.
