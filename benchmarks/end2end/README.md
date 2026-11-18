# End-to-end benchmarks

Short training runs of the supported models on H200 at 1, 2 and 4 nodes, as people would run and
tune them: 32K tokens with context parallelism, and 4K without. Each case is the fastest layout
found that fits, in one self-contained script; its log reports step time, tokens per second and
peak memory, so a later build can be run on the same cases and compared.

```
benchmarks/end2end/<model>-seq<N>-<dtype>/h200-<nodes>n<gpus>g/pp<P>-dp<D>-cp<C>-ep<E>-mbs<M>/script.py
```

`training.benchmark = True` trains on synthetic tokens and replays a realistic expert load (see
`pithtrain/modules/imbalance.py`). A case fits if at least 5 GiB of the device stays free.

## Run

```bash
torchrun --nproc-per-node=8 benchmarks/end2end/<case>/script.py
```

Multi-node through the `launch-with-slurm` skill. Steps 1 to 6 are warmup; time a case by the median
step time of steps 7 to 12. To add a case, copy the nearest case directory and change its layout.

## Results

Measured in November 2026, BF16 on H200. Throughput is the median of steps 7 to 12; memory is peak
allocated. MFU is throughput × model FLOPs per token / 989.4 TFLOP/s (H200 dense BF16), with FLOPs
per token from Megatron's `num_floating_point_operations`.

### 4K

| Model | Nodes | Case | Throughput (tok/s/gpu) | MFU | Memory (GiB) |
|---|---|---|---|---|---|
| deepseek-v2-lite | 1 | `pp4-dp2-cp1-ep1-mbs4` | 24,958 | 41.4% | 114.9 |
| deepseek-v2-lite | 2 | `pp2-dp8-cp1-ep2-mbs4` | 24,882 | 41.3% | 94.8 |
| deepseek-v2-lite | 4 | `pp2-dp16-cp1-ep8-mbs4` | 23,364 | 38.8% | 92.5 |
| gpt-oss-20b | 1 | `pp1-dp8-cp1-ep2-mbs2` | 18,924 | 43.9% | 109.3 |
| gpt-oss-20b | 2 | `pp1-dp16-cp1-ep8-mbs4` | 18,640 | 43.2% | 102.6 |
| gpt-oss-20b | 4 | `pp1-dp32-cp1-ep8-mbs4` | 18,947 | 43.9% | 97.6 |
| gpt-oss-120b | 4 | `pp2-dp16-cp1-ep16-mbs2` | 10,347 | 34.2% | 98.6 |
| qwen3-30b-a3b | 1 | `pp2-dp4-cp1-ep4-mbs2` | 15,339 | 35.8% | 99.8 |
| qwen3-30b-a3b | 2 | `pp2-dp8-cp1-ep4-mbs2` | 14,928 | 34.8% | 85.0 |
| qwen3-30b-a3b | 4 | `pp1-dp32-cp1-ep8-mbs2` | 14,322 | 33.4% | 90.7 |
| qwen3.5-35b-a3b | 1 | `pp1-dp8-cp1-ep8-mbs1` | 14,075 | 27.1% | 97.7 |
| qwen3.5-35b-a3b | 2 | `pp2-dp8-cp1-ep8-mbs2` | 14,898 | 28.7% | 93.5 |
| qwen3.5-35b-a3b | 4 | `pp2-dp16-cp1-ep8-mbs2` | 14,620 | 28.2% | 85.3 |

### 32K

| Model | Nodes | Case | Throughput (tok/s/gpu) | MFU | Memory (GiB) |
|---|---|---|---|---|---|
| deepseek-v2-lite | 1 | `pp4-dp1-cp2-ep1-mbs1` | 13,224 | 37.8% | 111.3 |
| deepseek-v2-lite | 2 | `pp2-dp4-cp2-ep2-mbs1` | 13,280 | 38.0% | 92.1 |
| deepseek-v2-lite | 4 | `pp4-dp4-cp2-ep1-mbs1` | 13,242 | 37.9% | 99.9 |
| qwen3-30b-a3b | 1 | `pp2-dp1-cp4-ep4-mbs1` | 6,946 | 39.9% | 99.8 |
| qwen3-30b-a3b | 2 | `pp2-dp2-cp4-ep4-mbs1` | 6,816 | 39.2% | 85.0 |
| qwen3-30b-a3b | 4 | `pp1-dp8-cp4-ep8-mbs1` | 6,772 | 38.9% | 90.8 |
| qwen3.5-35b-a3b | 1 | `pp1-dp1-cp8-ep8-mbs1` | 8,911 | 23.5% | 97.8 |
| qwen3.5-35b-a3b | 2 | `pp2-dp2-cp4-ep4-mbs1` | 10,296 | 27.2% | 99.3 |
| qwen3.5-35b-a3b | 4 | `pp2-dp4-cp4-ep8-mbs1` | 10,021 | 26.4% | 85.8 |
