# Pretrain Language Model

Pretrain a Mixture-of-Experts (MoE) language model with pipeline parallelism, expert parallelism, and FSDP.

## Step 1: Prepare Data

Tokenize the training corpus (one-time step):

```bash
bash examples/tokenize_corpus/launch.sh dclm-qwen3
bash examples/tokenize_corpus/launch.sh dclm-deepseek-v2
```

See [tokenize_corpus](../tokenize_corpus/) for details.

## Step 2: Launch Training

```bash
bash examples/pretrain_lm/launch.sh qwen3-30b-a3b
bash examples/pretrain_lm/launch.sh deepseek-v2-lite
```

The launch script auto-detects available GPUs and works with both single-node and multi-node (SLURM) setups.

## Available Models

| Model | Default Parallelism |
|---|---|
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 2-way pipeline x 8-way expert |
| [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | 2-way pipeline x 2-way expert |

Remaining GPUs are used as data-parallel (FSDP) ranks. Each model directory contains a `script.py` (training hyperparameters) and a `config.json` (model architecture). Edit `script.py` to customize.

## Output

Checkpoints are saved to `workspace/checkpoints/<model>/` at regular intervals. Training resumes from the latest checkpoint automatically.
