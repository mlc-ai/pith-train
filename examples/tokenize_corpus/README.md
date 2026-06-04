# Build Tokenized Corpus

Download and tokenize a training corpus. This is a one-time data preparation step before pretraining.

## Quick Start

```bash
bash examples/tokenize_corpus/launch.sh dclm-qwen3
bash examples/tokenize_corpus/launch.sh dclm-deepseek-v2
```

Each script downloads one shard of [DCLM Baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) and tokenizes it with the corresponding model's tokenizer.

Once finished, the tokenized dataset is ready for use in [pretrain_lm](../pretrain_lm/).
