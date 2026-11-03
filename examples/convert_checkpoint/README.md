# Convert Checkpoint

Convert checkpoints between HuggingFace (safetensors) and PyTorch Distributed Checkpoint (DCP) formats.

## Quick Start

```bash
bash examples/convert_checkpoint/launch.sh qwen3-30b-a3b
bash examples/convert_checkpoint/launch.sh deepseek-v2-lite
```

## Available Models

| Model | Operations |
|---|---|
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `hf2dcp`, `dcp2hf` |
| [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | `hf2dcp`, `dcp2hf` |

Each model directory contains a `script.py` that downloads the model and runs both conversions. Edit `script.py` to customize.

`hf2dcp` streams weights in chunks, including GPT-OSS MXFP4 dequantization.
`max_chunk_size` defaults to 64 MiB and limits each tensor chunk. DCP files are
packed up to `max_shard_size`, which defaults to 8 GiB and must be at least 4 KiB
for import. Canonical tensor names and shapes are preserved. `dcp2hf` still loads
the full model into memory.

## Checkpoint Layout

The DCP checkpoint is saved to `workspace/checkpoints/<model>/torch-dcp/XXXXXXXX`, matching the layout used by the training task. An imported HuggingFace checkpoint at `00000000` is directly loadable by [pretrain_lm](../pretrain_lm/) without any extra steps.

After training, export any step back to HuggingFace format by pointing `dcp2hf` at the corresponding `XXXXXXXX` directory.
