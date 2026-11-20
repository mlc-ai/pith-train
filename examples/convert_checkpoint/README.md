# Convert Checkpoint

Download a released HuggingFace checkpoint, and export trained PyTorch Distributed Checkpoint (DCP) checkpoints back to HuggingFace (safetensors).

## Quick Start

```bash
bash examples/convert_checkpoint/launch.sh qwen3-30b-a3b
bash examples/convert_checkpoint/launch.sh deepseek-v2-lite
```

## Available Models

| Model | Download | Export |
|---|---|---|
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | yes | yes |
| [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | yes | yes |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | yes | yes |

Each model directory contains a `script.py` that downloads the model and exports the newest checkpoint. Edit `script.py` to customize.

## Checkpoint Layout

The released weights are downloaded to `workspace/checkpoints/<model>/hf-import`. Set `training.hf_import_path` to that directory in [pretrain_lm](../pretrain_lm/) to start training from them; they are streamed straight into the sharded model, with no conversion step.

Training saves DCP checkpoints to `workspace/checkpoints/<model>/torch-dcp/XXXXXXXX`. The script exports the newest one to `workspace/checkpoints/<model>/hf-export`; point `load_path` at another `XXXXXXXX` directory to export a different step.
