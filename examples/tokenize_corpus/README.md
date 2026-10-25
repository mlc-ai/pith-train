# Build Tokenized Corpus

Download and tokenize a training corpus. This is a one-time data preparation step before pretraining.

## Quick Start

```bash
bash examples/tokenize_corpus/launch.sh dclm-qwen3
bash examples/tokenize_corpus/launch.sh dclm-deepseek-v2
bash examples/tokenize_corpus/launch.sh nemotron-cc-v2-qwen3
bash examples/tokenize_corpus/launch.sh nemotron-cc-v2-qwen3.5
bash examples/tokenize_corpus/launch.sh nemotron-cc-v2-deepseek-v2
bash examples/tokenize_corpus/launch.sh nemotron-cc-v2-gpt-oss
```

DCLM scripts download one shard of [DCLM Baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) and tokenize it with the corresponding model's tokenizer.

The Nemotron-CC v2 examples download the first 40 Parquet files from `High-Quality` in [Nemotron-CC v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) and tokenize them with the corresponding model's tokenizer. Edit `allow_patterns` in the script to select different files.

Once finished, the tokenized dataset is ready for use in [pretrain_lm](../pretrain_lm/).
