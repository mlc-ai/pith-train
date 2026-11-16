# End-to-end benchmarks

Short training runs of the supported models on H200 at 1, 2 and 4 nodes, each in the configuration
a practitioner would train with: the fastest layout measured that fits. A case is one training
script; its log reports step time, tokens per second and peak memory per step, so a later build
(a dependency bump, a kernel, an agent-evolved change) can be run on the same cases and compared.

The cases come in two families, one case per model and node count where the model fits:

- **Long context**, 32K tokens with context parallelism: DeepSeek-V2-Lite, Qwen3-30B-A3B,
  Qwen3.5-35B-A3B.
- **Pre-training**, 4K tokens without CP: the same three plus GPT-OSS-20B, and GPT-OSS-120B and
  Qwen3.5-122B-A10B on 4 nodes.

## Layout

```
benchmarks/end2end/<model>-seq<N>-<dtype>/h200-<nodes>n<gpus>g/pp<P>-dp<D>-cp<C>-ep<E>[-mbs<M>]/script.py
```

`-mbs<M>` names a micro-batch other than 1. `dp` is the attention DP, `world / (pp * cp)`. The
experts factor the same `world / pp` ranks of a stage as `ep x expert-dp`
(`expert-dp = world / (pp * ep)`), so `cp` and `ep` are chosen independently: EP may span CP, and
either may be larger. The path names the config the script builds.

Every case shares the fixture in `common.py`, set by `training.benchmark = True`:

- **Synthetic data.** `SyntheticDataset`: seeded token ids drawn on the GPU for each step,
  Zipf-distributed over the vocabulary. No corpus to tokenize and no file read on the step path,
  so disk load cannot move the step time. The samples depend only on the step, not on the mesh.
- **Realistic routing.** The router's expert choice is replayed with the expert load of a real
  training run. Per layer and step each expert's share is a log-normal draw with spread
  `pithtrain.modules.imbalance.SIGMA = 0.2` (measured 0.212 on Qwen3-30B-A3B), seeded by layer and
  step, so every run of a case routes alike. Experts sit on EP ranks in contiguous blocks, so the
  per-rank imbalance follows from the EP degree: max/mean 1.04, 1.07, 1.13, 1.23 at EP 4, 8, 16,
  32 for 128 experts (the run measured 1.037, 1.073, 1.130, 1.219). `SIGMA = 0` routes
  force-balanced instead, every expert an equal load.
- **Fixed work per GPU.** 262,144 tokens per GPU per step (`TOKENS_PER_GPU`: 64 sequences of 32K
  or 512 of 4K per node), so every rank runs at least 8 micro-batches a step and DualPipeV has
  micro-batches to overlap. Muon at a constant 3e-4, 12 steps from fresh weights. Nothing is
  checkpointed or sent to wandb.

## Memory

Each case leaves at least 5 GiB of the H200 free: its peak device memory, reported per step as
`peak-device-memory`, is at most 134.8 GiB. That is the memory PyTorch's caching allocator
reserves (`peak-reserved-memory`, which includes what fragmentation holds) plus what the process
holds outside it (CUDA context, NCCL buffers), not just the tensors allocated
(`peak-gpu-memory`). Under realistic routing the expert token counts change every layer and step,
so the allocator fragments: a case can reserve tens of GiB more than it allocates. A 12-step run
may understate how far fragmentation grows over a long run.

## Add a case

Copy the nearest existing case directory and change its layout: the directory name and the mesh
the script sets (`pp`, `cp`, `ep`, and the micro-batch where it is not 1) must agree.

## Run

One case at a time, from the repo root with `.venv` active:

```bash
torchrun --nproc-per-node=8 benchmarks/end2end/<case>/script.py
```

On several nodes, launch it the same way on every node through the `launch-with-slurm` skill. The
first 6 steps are warmup (the `torch.compile` trace, the JIT warmup and kernel tuning); time a case
by the median step time of steps 7 to 12. Each step's log line reports `step-time`,
`tokens-per-second`, `cross-entropy-loss`, `load-balance-loss`, `gradient-norm` and the three peak
memories above. Synthetic tokens make the loss no convergence signal, but from the same seed two
correct builds follow the same curve. For a precise loss-parity verdict on one mesh with real
routing and real data, use the `validate-correctness` skill.
