#!/bin/bash
# Convert checkpoints between HuggingFace and DCP formats.
#
# Usage:
#   bash examples/convert_checkpoint/launch.sh qwen3-30b-a3b
#   bash examples/convert_checkpoint/launch.sh deepseek-v2-lite

set -euo pipefail
export PYTHONUNBUFFERED=1

# Launch the conversion.
SCRIPT=examples/convert_checkpoint/$1/script.py
OUTPUT=logging/convert_checkpoint/${1}.log

mkdir -p $(dirname $OUTPUT) && exec > >(tee $OUTPUT) 2>&1
python3 $SCRIPT
