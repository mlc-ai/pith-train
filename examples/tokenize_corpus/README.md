# Build Tokenized Corpus

Download and tokenize a training corpus. This is a one-time data preparation step before pretraining.

## Quick Start

```bash
bash examples/tokenize_corpus/launch.sh dclm-qwen3
bash examples/tokenize_corpus/launch.sh dclm-deepseek-v2
bash examples/tokenize_corpus/launch.sh nemotron-cc-v1-qwen3
bash examples/tokenize_corpus/launch.sh nemotron-cc-v2-qwen3
```

DCLM scripts download one shard of [DCLM Baseline 1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) and tokenize it with the corresponding model's tokenizer.

The Nemotron-CC v1 example downloads a small `quality=high/kind=actual/kind2=actual` sample by default. Set `NEMOTRON_CC_PARTITIONS=all` and `NEMOTRON_CC_MAX_FILES=0` to download all published partitions, or set `NEMOTRON_CC_PARTITIONS` to a comma-separated list of partition paths.

The Nemotron-CC v2 example downloads one gated Hugging Face Parquet file from `High-Quality` by default. Set `NEMOTRON_CC_V2_SUBSETS=all` and `NEMOTRON_CC_V2_MAX_FILES_PER_SUBSET=0` to download all published v2 subsets, or set `NEMOTRON_CC_V2_SUBSETS` to a comma-separated list of subsets.

Once finished, the tokenized dataset is ready for use in [pretrain_lm](../pretrain_lm/).
