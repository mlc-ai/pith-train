# End-to-end benchmarks

Short training runs of the supported models on H200 at 1, 2 and 4 nodes, each in the configuration
a practitioner would train with: the fastest measured layout that fits. Each case records step
time, tokens per second per GPU, peak memory and the loss curve in a `result.json`, so a later
build (a dependency bump, a kernel, an agent-evolved change) is diffed against a recorded baseline
case by case, for speed and for correctness.

The cases ([`default.txt`](default.txt), 22 in BF16) come in two families, one case per model
and node count where the model fits:

- **Long context**, 32K tokens with context parallelism: DeepSeek-V2-Lite, Qwen3-30B-A3B,
  Qwen3.5-35B-A3B.
- **Pre-training**, 4K tokens without CP and a micro-batch above 1 where it fits: the same three
  plus GPT-OSS-20B, and GPT-OSS-120B on 4 nodes. Qwen3.5-122B-A10B fits nowhere under the memory
  cap (its 4-node pp2 run peaks at 122 GB), so it has no case.

Each case is the fastest layout measured (tokens per second per GPU) with a peak under 115 GB, 82%
of an H200. A fit search set CP (32K) or the micro-batch (4K) at the largest EP each deployment
allows, then a tuning pass measured the multi-node configs with EP inside a node (ep8, expert data
parallelism across nodes) and with pp2. Keeping EP inside a node won almost everywhere: by 10-75%
at 4K and 4-51% at 32K, so most multi-node cases run ep8, and some pp2. Every layout tried, with its
numbers, is in `workspace/benchmarks/end2end/TUNING.md` of the run that chose them.

## Layout

```
benchmarks/end2end/<model>-seq<N>-<dtype>/h200-<nodes>n<gpus>g/pp<P>-dp<D>-cp<C>-ep<E>[-mbs<M>]/script.py
```

`-mbs<M>` names a micro-batch other than 1.

`dp` is the attention DP, `world / (pp * cp)`. The experts factor the same `world / pp` ranks of a
stage as `ep x expert-dp` (`expert-dp = world / (pp * ep)`), so `cp` and `ep` are chosen
independently: EP may span CP, and either may be larger. The path is the contract;
`tests/test_end2end_cases.py` checks that every script builds the config its path names.

Every case shares the fixture in `common.py`:

- **Synthetic data** (`training.benchmark = True`): `SyntheticDataset`, seeded token ids drawn on
  the GPU for each step, Zipf-distributed over the vocabulary. No corpus to tokenize, no file read
  on the step path, so disk load cannot move the step time. The samples depend only on the step,
  not on the mesh, so every mesh of one node count and sequence length trains on the same data.
- **Realistic routing** (the same flag): the router's expert choice is replayed with the expert
  load of a real training run. Per layer and step each expert's share is a log-normal draw with
  spread `pithtrain.modules.imbalance.SIGMA = 0.2` (measured 0.212 on Qwen3-30B-A3B), seeded by
  layer and step, so every run of a case routes alike. Experts sit on EP ranks in contiguous
  blocks, so the per-rank imbalance follows from the EP degree: max/mean 1.04, 1.07, 1.13, 1.23
  at EP 4, 8, 16, 32 for 128 experts (the run measured 1.037, 1.073, 1.130, 1.219). `SIGMA = 0`
  routes force-balanced instead, every expert an equal load.
- **Fixed work per GPU.** 262,144 tokens per GPU per step (`TOKENS_PER_GPU`: 64 sequences of 32K
  or 512 of 4K per node), so every rank runs at least 8 micro-batches a step and DualPipeV has
  micro-batches to overlap. Muon at a constant 3e-4, 12 steps from fresh weights. Nothing is
  checkpointed or sent to wandb.

The first 6 steps are warmup (the `torch.compile` trace, the JIT warmup and kernel tuning), so
timing is the median of steps 7 to 12. After a change to the step counts, `summarize.py
--remeasure <result.json> ...` recomputes stored results from their per-step records.

## Run

From the repo root with `.venv` active, one `srun` step per node count (a step runs only the cases
whose `h200-<nodes>n8g` matches its node count):

```bash
srun -N 1 -W 0 bash benchmarks/end2end/launch.sh                                    # every 1-node case
srun -N 2 -W 0 bash benchmarks/end2end/launch.sh                                    # every 2-node case
srun -N 4 -W 0 bash benchmarks/end2end/launch.sh benchmarks/end2end/qwen3-30b-a3b-seq32768-bf16
```

Results go to `workspace/benchmarks/end2end/<commit>/<case>/` (`BENCH_OUTPUT` overrides it). A case
that already has an ok `result.json` is skipped, so rerunning a preempted set resumes it; a case that
fails is retried once.

## Compare

```bash
python3 -m benchmarks.end2end.compare benchmarks/end2end/baseline.json workspace/benchmarks/end2end/<commit>
python3 -m benchmarks.end2end.compare <results> --filter seq4096 h200-4n8g      # one set, as a table
python3 -m benchmarks.end2end.collect workspace/benchmarks/end2end/<commit> -o benchmarks/end2end/baseline.json
```

Speed is reported, not gated: over 37 cases run twice on the same main build (the earlier 4K
fixture), step time differed by -2.5% to +1.6%. Loss is flagged: synthetic tokens make the loss no convergence signal, but from
the same seed two correct builds follow the same curve, so it is a step-by-step correctness
signal. In those 37 pairs the first-step loss agreed within 0.006, and the largest gap at any step
was 0.023 in BF16 and 0.069 in FP8 (median 0.003 and 0.021). `compare.py` flags a first-step loss
more than 0.01 apart (the first step precedes any update, so only a forward change moves it) or
any step more than 0.1 apart. Each case also checks that its losses are finite and that
the loss fell over the run. For a precise loss-parity verdict
on one mesh with real routing and real data, use the `validate-correctness` skill.
