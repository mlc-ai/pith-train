#!/bin/bash
# Download and tokenize a corpus for language model pretraining.
#
# Usage:
#   bash examples/tokenize_corpus/launch.sh dclm-qwen3
#   bash examples/tokenize_corpus/launch.sh dclm-deepseek-v2
#
# For multi-node tokenization with SLURM:
#   srun -W 0 examples/tokenize_corpus/launch.sh dclm-qwen3

set -euo pipefail
export PYTHONUNBUFFERED=1

# Launch the tokenization.
SCRIPT=examples/tokenize_corpus/$1/script.py
OUTPUT=logging/tokenize_corpus/${1}_node${SLURM_NODEID:-0}.log

mkdir -p $(dirname $OUTPUT) && exec > >(tee $OUTPUT) 2>&1
python3 $SCRIPT
