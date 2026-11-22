# Pretrain Language Model

Pretrain a Mixture-of-Experts (MoE) language model with pipeline parallelism, expert parallelism, and FSDP.

## Step 1: Prepare Data

Tokenize the training corpus (one-time step):

```bash
bash examples/tokenize_corpus/launch.sh dclm-qwen3
bash examples/tokenize_corpus/launch.sh dclm-qwen3.5
```

See [tokenize_corpus](../tokenize_corpus/) for details.

## Step 2: Launch Training

```bash
bash examples/pretrain_lm/launch.sh qwen3-30b-a3b
bash examples/pretrain_lm/launch.sh qwen3.5-35b-a3b
```

The launch script auto-detects available GPUs and works with both single-node and multi-node (SLURM) setups. Run the multi-node models under `srun` with one task per node:

```bash
srun -N 2 --ntasks-per-node=1 -W 0 examples/pretrain_lm/launch.sh qwen3.5-27b
```

Training starts from random weights. To start from released weights instead, download them with [convert_checkpoint](../convert_checkpoint/) and set `training.hf_import_path` to `workspace/checkpoints/<model>/hf-import` in `script.py`.

## Available Models

Every example pretrains at 4K context (`sequence_length = 4096`, global batch 1024 sequences, about 4M tokens per step) and is sized for 8-GPU H200 nodes (141 GB per GPU).

| Model | H200 nodes | Mesh | Micro-batch |
|---|---|---|---|
| [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) | 1 | 1-way pipeline x 2-way expert | 2 |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 1 | 2-way pipeline x 4-way expert | 2 |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 1 | 1-way pipeline x 8-way expert | 1 |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) (dense) | 2 | 8-way pipeline | 1 |
| [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) | 4 | 4-way pipeline x 8-way expert | 2 |

Remaining GPUs are used as data-parallel (FSDP) ranks. Each model directory contains a `script.py` (training hyperparameters) and a `config.json` (model architecture). Edit `script.py` to customize; on GPUs with less memory, lower the micro-batch or add nodes, and check the mesh with `python -m tools.memory_estimator` first.

## Output

Checkpoints are saved to `workspace/checkpoints/<model>/` at regular intervals. Training resumes from the latest checkpoint automatically.
