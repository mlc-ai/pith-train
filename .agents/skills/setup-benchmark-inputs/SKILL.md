---
name: setup-benchmark-inputs
description: Set up the minimal set of artifacts (tokenized DCLM corpus shard + released HuggingFace checkpoint converted to DCP) required to benchmark, profile, or regression-test a MoE model in PithTrain. Use when the user asks to "prepare benchmark inputs", "set up the benchmark workspace", "download the DCLM shard", "fetch and convert the released checkpoint", "tokenize DCLM for DeepSeek/Qwen3", or when a downstream skill (capture-nsys-profile, validate-correctness, or any short canonical run) needs its workspace pre-populated. Produce `workspace/datasets/dclm-baseline/toktxt/<model>` and `workspace/checkpoints/<model>/torch-dcp/step-00000000`. Idempotent and safe to re-run.
---

# Setup Benchmark Inputs

Setup the minimal artifacts needed to benchmark, profile, or regression-test a MoE model in PithTrain: a single DCLM corpus shard tokenized for the target model, and the released HuggingFace checkpoint converted to DCP format. Each step is idempotent (skips if its output already exists).

## Prerequisites

- **Python environment**: activate `.venv` in the repo root (`source .venv/bin/activate`).

## Usage

```bash
mkdir -p logging

# Single-node (DeepSeek-V2-Lite)
bash .agents/skills/setup-benchmark-inputs/scripts/launch_setup.sh --model deepseek-v2-lite 2>&1 | tee logging/setup-deepseek-v2-lite.log

# Multi-node via SLURM (Qwen3-30B-A3B)
srun -W 0 -o logging/setup-qwen3-30b-a3b.log .agents/skills/setup-benchmark-inputs/scripts/launch_setup.sh --model qwen3-30b-a3b
```
