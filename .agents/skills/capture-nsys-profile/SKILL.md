---
name: capture-nsys-profile
description: Capture a Nsight Systems (.nsys-rep) profile of a short PithTrain run for performance analysis. Use when the user asks to "capture an nsys profile", "profile training", or "grab an nsys trace", or wants to inspect kernel timelines / pipeline behavior / all-to-all overheads. Adaptive over pipeline-parallel (PP), expert-parallel (EP), context-parallel (CP), and sequence length; size the global batch so the pipeline reaches steady state without producing a multi-GB .nsys-rep. Run 5 warmup steps + 1 profiled step from a released checkpoint.
---

# Capture Nsys Profile

Capture a single-step Nsight Systems (`.nsys-rep`) trace of PithTrain training, loaded from a released HuggingFace checkpoint so MoE load balancing is representative. The profile configuration (PP / EP / CP / sequence length) is specified at launch time; micro-batch size is hardcoded to 1.

## Prerequisites

- **Python environment**: activate `.venv` in the repo root (`source .venv/bin/activate`).
- **Nsight Systems CLI**: `nsys --version` must work on every compute node.
- **Hardware**: enough GPUs to satisfy `world_size >= PP * CP * EP` (with at least `DP >= 1`). See Step 2.
- **Benchmark inputs set up**: the **setup-benchmark-inputs** skill has run for the target model (produces the tokenized corpus and DCP checkpoint capture loads).

## Step 1: Choose parallelism

The user typically specifies PP and EP (and sometimes CP / sequence length). Confirm the numbers before launching:

- `--model`: one of the supported models
- `--pipeline-parallel-size`: PP
- `--expert-parallel-size`: EP
- `--context-parallel-size`: CP (default 1)
- `--sequence-length`: sequence length in tokens (default 2048)

If the user is vague ("just profile DeepSeek-V2-Lite"), ask for PP and EP before launching. Different parallelism splits surface different bottlenecks, so the right config depends on what they want to see.

## Step 2: Determine node count

Target DP=1 (smallest world that satisfies the parallelism plan). Assuming 8 GPUs per node:

| Config | PP * CP * EP | Min nodes (8 GPUs/node) |
|---|---|---|
| `pp=2 cp=1 ep=2` | 4 | 1 (half of an 8-GPU node) |
| `pp=2 cp=1 ep=8` | 16 | 2 |
| `pp=4 cp=1 ep=8` | 32 | 4 |

### If running under SLURM

If `$SLURM_JOB_ID` is set, use the **launch-with-slurm** skill to read the allocation's node count and compare it to the minimum above. If the allocation is short, surface that to the user instead of launching.

## Step 3: Capture the profile

```bash
# Single-node, minimum GPUs (DeepSeek-V2-Lite, pp=2 ep=2)
bash .agents/skills/capture-nsys-profile/scripts/launch_capture.sh --model deepseek-v2-lite --pipeline-parallel-size 2 --expert-parallel-size 2

# Multi-node via SLURM (Qwen3-30B-A3B, pp=2 ep=8 -> 2 nodes)
srun -N 2 -W 0 .agents/skills/capture-nsys-profile/scripts/launch_capture.sh --model qwen3-30b-a3b --pipeline-parallel-size 2 --expert-parallel-size 8

# Custom sequence length (Qwen3-30B-A3B, pp=4 ep=8 -> 4 nodes, seq=4096)
srun -N 4 -W 0 .agents/skills/capture-nsys-profile/scripts/launch_capture.sh --model qwen3-30b-a3b --pipeline-parallel-size 4 --expert-parallel-size 8 --sequence-length 4096
```

## Output

Each node produces one `.nsys-rep` at `workspace/capture-nsys-profile/pithtrain_node<N>.nsys-rep`, containing traces for all ranks on that node (nsys attaches to `torchrun`'s child processes). Analysis (GUI inspection, `nsys stats`, etc.) is out of scope for this skill; that belongs to a separate analyze-nsys-profile skill.

## Common Issues

### `WORLD_SIZE is not divisible by pp*cp*ep`

The allocation doesn't have enough GPUs for the requested parallelism. Stop; tell the user their allocation is short (report current `world_size` and required `pp * cp * ep`) and ask whether to reduce PP/EP/CP or request more nodes. Do not silently adjust on their behalf.

### No `.nsys-rep` produced after the run

Check the per-node log for nsys errors. Common causes: `nsys` not on PATH inside the srun step, or the output directory not writable.
