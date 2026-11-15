# End-to-end benchmarks

Short training runs of every supported model at the parallel layouts that MoE parallel folding
allows, on H200. Each case records step time, tokens per second per GPU, peak memory and the loss
curve in a `result.json`, so a later build (a dependency bump, a kernel, an agent-evolved change)
is diffed against a recorded baseline case by case, for speed and for correctness.

## Layout

```
benchmarks/end2end/<model>-seq<N>-<dtype>/h200-<nodes>n<gpus>g/pp<P>-dp<D>-cp<C>-ep<E>/script.py
```

`dp` is the attention DP, `world / (pp * cp)`. The experts factor the same `world / pp` ranks of a
stage as `ep x expert-dp` (`expert-dp = world / (pp * ep)`), so `cp` and `ep` are chosen
independently: EP may span CP, and either may be larger. The path is the contract;
`tests/test_end2end_cases.py` checks that every script builds the config its path names.

Every case shares the fixture in `common.py`:

- **Synthetic data.** `cfg.dataset = None` selects `SyntheticDataset`: seeded token ids drawn on
  the GPU for each step, Zipf-distributed over the vocabulary. No corpus to tokenize, no file read
  on the step path, so disk load cannot move the step time. The samples depend only on the step,
  not on the mesh, so every mesh of one node count trains on the same data.
- **Force-balanced routing** (`training.benchmark = True`): every expert gets an equal load, which
  removes the largest source of step-time variance and makes step time comparable across meshes
  and builds. It also pins `load-balance-loss` at 1.0, which the summary checks.
- **Fixed shape.** 4 samples per GPU (`global_batch = 4 * world`), micro-batch 1, Muon at a
  constant 3e-4, 20 steps from fresh weights. Nothing is checkpointed or sent to wandb.

The first two steps carry the `torch.compile` trace, the JIT warmup and kernel tuning, so timing
is the median of steps 3 to 20.

## Run

From the repo root with `.venv` active, one `srun` step per node count (a step runs only the cases
whose `h200-<nodes>n8g` matches its node count):

```bash
srun -N 1 -W 0 bash benchmarks/end2end/launch.sh benchmarks/end2end                      # every 1-node case
srun -N 4 -W 0 bash benchmarks/end2end/launch.sh benchmarks/end2end/qwen3-30b-a3b-seq4096-fp8
srun -N 2 -W 0 bash benchmarks/end2end/launch.sh benchmarks/end2end/gpt-oss-120b-seq4096-bf16/h200-2n8g/pp2-dp8-cp1-ep8
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

Speed is reported, not gated: two runs of identical code differ by about 1% in step time. Loss is
flagged: synthetic tokens make the loss no convergence signal, but from the same seed two correct
builds follow the same curve, so it is a step-by-step correctness signal. Two runs of identical
code agree within 0.001 at every step (DeepSeek-V2-Lite, 1x8, BF16); `compare.py` flags a
first-step loss more than 0.01 apart (the first step precedes any update, so only a forward change
moves it) or any step more than 0.05 apart. Each case also checks that its losses are finite, that
the loss fell over the run, and that routing was force-balanced. For a precise loss-parity verdict
on one mesh with real routing and real data, use the `validate-correctness` skill.
