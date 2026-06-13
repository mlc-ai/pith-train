---
name: estimate-memory
description: Estimate peak GPU memory for a DualPipeV training run. Use when the user asks to "estimate memory", "will this fit in memory", "how much GPU memory", "check if this OOMs", "memory for training X on Y GPUs", or mentions memory planning for a training configuration. Translates natural-language descriptions of hardware, model, and training setup into the exact CLI arguments for `python -m tools.memory_estimator`.
---

# Memory Estimation Skill

Estimates peak GPU memory usage for DualPipeV MoE training runs using an analytical simulator. Takes natural-language descriptions and translates them into the correct CLI invocation.

## How It Works

The memory estimator at `tools/memory_estimator/` simulates the full DualPipeV 8-step pipeline schedule, tracking activations, autograd saved tensors, gradients, optimizer states, communication buffers, and non-PyTorch overhead (CUDA context, NCCL, torch.compile) at every event boundary. It reports peak memory, a per-component breakdown, and whether the config fits in GPU memory.

## Step 1: Gather Parameters

Extract these parameters from the user's message. If any required parameters are missing, **do not guess or fill in defaults** — list ALL required parameters with a brief description of each and ask the user to provide the missing ones before proceeding. If the user provided no parameters at all (e.g., just `/estimate-memory`), show the full list of required parameters.

### Required parameters

| Parameter | CLI flag | How to determine |
|---|---|---|
| Model config | `--model` | Path to a HuggingFace-style `config.json`. Available models: `examples/pretrain_lm/qwen3-30b-a3b/config.json`, `examples/pretrain_lm/deepseek-v2-lite/config.json`. If the user names a model, find its config under `examples/`. |
| PP size | `--pp-size` | Pipeline parallel degree. If the user says "2-way pipeline", use 2. |
| EP size | `--ep-size` | Expert parallel degree. |
| CP size | `--cp-size` | Context parallel degree. Use 1 if not using context parallelism. |
| Total GPUs | `--total-gpus` | Total GPU count. E.g., "4x8 H100" = 32. "16 GPUs" = 16. FSDP (dp) dimension is derived: `dp = total_gpus / (pp * ep * cp)`. |
| Micro batch size | `--micro-batch-size` | Per-GPU micro batch size. |
| Global batch size | `--global-batch-size` | Global batch size across all GPUs. |
| Sequence length | `--sequence-length` | Token count per sequence. |
| GPU type/memory | `--gpu-memory-gb` | GPU memory capacity in GB. See common hardware table below. |

### Optional parameters (with defaults)

| Parameter | CLI flag | Default | Notes |
|---|---|---|---|
| FP8 training | `--fp8-training` | `disabled` | `deep-gemm` if user mentions FP8. **Caveat:** FP8 training *increases* memory (additional quantized weight and input tensor caches), but this overhead is not yet modeled. When FP8 is enabled, warn the user that the estimate is a lower bound — real usage will be higher. |
| PP rank | `--pp-rank` | -1 (scan all) | -1 finds worst case automatically. |
| EP imbalance | `--ep-imbalance` | 1.0 | Increase for skewed expert routing. |
| Fragmentation | `--fragmentation` | 0.10 | CUDA allocator overhead. |

### CP accuracy caveat

**Important:** When `cp_size > 1`, the estimator reduces the per-rank sequence length (`S = seq_len / cp_size`) and adjusts FSDP sharding, but it does **not** model ring attention communication buffers or any changes to the autograd saved tensors from ring attention. CP memory estimates have not been validated against real measurements. When reporting results with CP > 1, warn the user: "Note: CP > 1 estimates are approximate — ring attention memory overhead is not fully modeled and has not been validated against real measurements."

### Deriving dp_size

The tool computes `dp_size = total_gpus / (pp_size * cp_size * ep_size)`. Verify this is an integer. If not, the config is invalid — tell the user.

### Common hardware specs

| Hardware | `--gpu-memory-gb` |
|---|---|
| H100 SXM | 80 |
| H200 SXM | 141 |
| B200 | 192 |

### Constraint: num_chunks >= pp_size * 2

The tool validates that `num_chunks = global_batch_size / (micro_batch_size * dp_size * ep_size) >= pp_size * 2`. If this fails, suggest increasing global_batch_size or decreasing micro_batch_size.

## Step 2: Run the Estimator

Build the command and **always show the exact command to the user** before running it, so they can copy-paste it for manual runs.

```bash
python -m tools.memory_estimator \
    --model <config_path> \
    --pp-size <N> --ep-size <N> --cp-size <N> --total-gpus <N> \
    --micro-batch-size <N> --global-batch-size <N> --sequence-length <N> \
    --gpu-memory-gb <N>
```

Add `--detail` if the user asks for a detailed breakdown.
Add `--timeline` if the user wants to see the schedule progression.

## Step 3: Interpret Results

The output has four sections:

1. **Static Memory** — parameters, FSDP shards, optimizer states. Constant during training.
2. **Peak Dynamic Memory** — activations, autograd, gradients, comm buffers at the peak event. This is the memory high-water mark.
3. **Grand Total** — model tensors + non-PyTorch overhead + fragmentation.
4. **Suggestions** — actionable advice if memory is tight.

Status interpretation:
- **OK** — fits with >5% headroom.
- **TIGHT** — fits but <5% headroom. May OOM under non-uniform routing or memory spikes.
- **OOM** — estimated peak exceeds GPU capacity.

## Examples

### Example 1: Missing parameters

User: "How much memory does Qwen3-30B-A3B need on 32 H100s with pp=4, ep=8?"

Response: "I need a few more details to run the estimate:
- **CP size** — context parallel degree (1 if not using CP)
- **Micro batch size** — per-GPU micro batch size
- **Global batch size** — total batch size across all GPUs
- **Sequence length** — tokens per sequence

Could you provide these?"

### Example 2: Complete query

User: "Qwen3-30B-A3B, 32 H100s, pp=4, ep=8, cp=1, micro_bs=1, gbs=1024, seq_len=4096"

```bash
python -m tools.memory_estimator \
    --model examples/pretrain_lm/qwen3-30b-a3b/config.json \
    --pp-size 4 --ep-size 8 --cp-size 1 --total-gpus 32 \
    --micro-batch-size 1 --global-batch-size 1024 --sequence-length 4096 \
    --gpu-memory-gb 80
```

### Example 3: How many nodes do I need?

User: "Qwen3-235B-A22B, ep=8, cp=1, fsdp=1, micro_bs=1, gbs=1024, seq_len=2048, pp size is the number of B200 nodes, how many B200 nodes do I need?"

This is a search problem. Each B200 node has 8 GPUs (192 GB each). With ep=8, cp=1, fsdp=1, total_gpus = pp * 8. Try pp=2, pp=4, pp=8, etc. until the estimate shows OK status. Run the estimator for each pp size and report the minimum that fits.

Note: this requires a config.json for the model. If the model doesn't have one under `examples/`, tell the user you need the model's HuggingFace config.json.

### Example 4: Comparing configs

User: "Will training fit if I increase sequence length to 8192?"

Run the estimator twice — once with `--sequence-length 4096` and once with `--sequence-length 8192` — and compare the peak memory and headroom.

### Example 5: Exploring parallelism

User: "What's the best parallelism config for 16 GPUs?"

Try multiple configs (e.g., pp=2/ep=8, pp=4/ep=4, pp=2/ep=4/dp=2) and compare peak memory. Report which has the most headroom.
