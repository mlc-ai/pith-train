<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/img/pithtrain-logo-text-dark.png">
    <img alt="PithTrain" src="docs/_static/img/pithtrain-logo-text-light.png" width="480">
  </picture>
</p>

<h3 align="center">Compact and Agent-Native MoE Training System</h3>

<p align="center">
  <a href="https://blog.mlc.ai/2026/06/01/pithtrain-compact-agent-native-moe-training-system">Blog</a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2605.31463">Paper</a>
</p>

Production MoE training frameworks deliver peak throughput and broad model coverage, but evolving them for new architectures or system optimizations remains expensive. Meanwhile, the design patterns that make these stacks work for humans, such as plugin systems, registry-based indirection, and heavy compiled extensions, are harder for AI coding agents to navigate.

PithTrain is an MoE training framework designed agent-native from the start: ~11K lines of Python, minimal implicit indirection, with shipped agent skills for recurring tasks. It delivers production-grade performance, including 4D parallelism, compute-communication overlap, and FP8 training, in a codebase compact enough that an agent (or a human) can read it end-to-end.

## Installation

NVIDIA Hopper (SM90) or Blackwell (SM100) GPUs are required. CUDA >= 13.0 and Python >= 3.12 are required. We use [uv](https://docs.astral.sh/uv/) to manage project dependencies.

```bash
git clone https://github.com/mlc-ai/pith-train.git && cd pith-train
uv venv  # skip if you already have a virtual environment
```

**For users:**

```bash
uv pip install .
```

**For developers:**

```bash
uv sync
```

## Getting Started

Pretrain Qwen3-30B-A3B from scratch. Datasets and checkpoints are stored in the `workspace` folder by default. Other models like DeepSeek-V2-Lite follow the same steps. See [`examples`](examples) for available configurations.

**1. Prepare the dataset**

```bash
bash examples/tokenize_corpus/launch.sh dclm-qwen3
```

Download and tokenize the DCLM pretraining corpus into mmap-friendly packed sequences. Each model uses its own tokenizer, so switching to a different model requires running this step again.

**2. Configure training**

Edit [`examples/pretrain_lm/qwen3-30b-a3b/script.py`](examples/pretrain_lm/qwen3-30b-a3b/script.py) to adjust parallelism, batch size, learning rate, and other hyperparameters. The model architecture is defined in the accompanying [`config.json`](examples/pretrain_lm/qwen3-30b-a3b/config.json).

**3. Launch training**

```bash
bash examples/pretrain_lm/launch.sh qwen3-30b-a3b
```

The launch script auto-detects GPUs and supports both single-node and multi-node (SLURM) setups. Training resumes from the latest checkpoint automatically, and checkpoints are reshardable across different parallelism.

**4. Export checkpoint**

```bash
bash examples/convert_checkpoint/launch.sh qwen3-30b-a3b
```

Convert a training checkpoint to standard Hugging Face format for evaluation or inference. The same tool also supports importing Hugging Face checkpoints for continued pretraining.

For hardware requirements, supported models, scaling and multi-node runs, troubleshooting, etc., see the [User Guide](docs/user-guide.md).

## Architecture

<p align="center">
  <img src="docs/_static/img/PithTrain-arch.svg" width="100%">
</p>

PithTrain is structured in three layers:

- **Application** — Training loop for pretraining, SFT, and more.
- **Engine** — The bulk of PithTrain, composed of five modules:
  - *Model* — Protocol interface with implementations for Qwen, DeepSeek, and GPT-OSS architectures.
  - *Building Blocks* — FP8 linear and quantization, ring attention, expert dispatch and deduplication, etc.
  - *Pipeline Engine* — DualPipeV scheduler with 5-stage overlapped forward-backward execution and P2P communication.
  - *Distributed Training* — Pipeline, data, context, and expert parallelism (PP x FSDP x CP x EP).
  - *Training Infrastructure* — `torch.compile`, optimizer and LR scheduling, checkpointing, logging, etc.
- **Operator** — PyTorch (basic ops, NCCL), operator libraries (DeepGEMM, FlashAttention), and Python DSLs (Triton).

For a developer-level tour of the system — the 5-stage overlapped pipeline, the model protocol, the 4D device mesh, etc. — see [`docs/architecture.md`](docs/architecture.md).

## Contributing

Contributions are welcome. [`CONTRIBUTING.md`](CONTRIBUTING.md) covers the development setup, testing and correctness-validation workflow, design principles, extension recipes, etc.

## Acknowledgement

PithTrain is developed by contributors from CMU. It is built on top of DeepSeek's [DualPipe](https://github.com/deepseek-ai/DualPipe), which provides the original pipeline parallelism schedule and examples. We thank the [CMU Foundation and Language Model (FLAME) Center](https://www.cmu.edu/flame/) for providing the compute resources to develop PithTrain. We also acknowledge the support of DGX B200 from NVIDIA.

## Citation

If you find PithTrain useful in your research, please consider citing:

```bibtex
@misc{pithtrain2026,
  title={PithTrain: A Compact and Agent-Native MoE Training System},
  author={Ruihang Lai and Hao Kang and Haozhan Tang and Akaash R. Parthasarathy and Zichun Yu and Junru Shao and Todd C. Mowry and Chenyan Xiong and Tianqi Chen},
  year={2026},
  eprint={2605.31463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.31463},
}
```

## License

PithTrain is released under the [Apache 2.0 License](LICENSE).
