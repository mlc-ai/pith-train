# End-to-end benchmarks

Short training runs of the supported models on H200 at 1, 2 and 4 nodes, each in the fastest layout
measured that fits: 32K tokens with context parallelism, and 4K without. Each case is one
self-contained script; its log reports step time, tokens per second and peak memory, so a later
build can be run on the same cases and compared.

```
benchmarks/end2end/<model>-seq<N>-<dtype>/h200-<nodes>n<gpus>g/pp<P>-dp<D>-cp<C>-ep<E>[-mbs<M>]/script.py
```

`training.benchmark = True` trains on synthetic tokens and replays a realistic expert load (see
`pithtrain/modules/imbalance.py`). A case fits if at least 5 GiB of the device stays free.

## Run

```bash
torchrun --nproc-per-node=8 benchmarks/end2end/<case>/script.py
```

Multi-node through the `launch-with-slurm` skill. Steps 1 to 6 are warmup; time a case by the median
step time of steps 7 to 12. To add a case, copy the nearest case directory and change its layout.
