# End-to-end benchmarks

## Layout

```
benchmarks/end2end/<model>-seq<N>-<dtype>/h200-<nodes>n<gpus>g/pp<P>-dp<D>-cp<C>-ep<E>/script.py
```

`P`, `C` and `E` are the pipeline, context and expert parallel sizes, and `D` is the attention data
parallel size, the GPU count divided by P × C.

## Results

Measured in November 2026, BF16 on H200. Throughput is the median of steps 7 to 12; memory is peak
allocated. MFU is throughput × model FLOPs per token / 989.4 TFLOP/s (H200 dense BF16), with FLOPs
per token from Megatron's `num_floating_point_operations`.

### 4K

| Model | Nodes | Case | Micro-batch | Throughput (tok/s/gpu) | MFU | Memory (GiB) |
|---|---|---|---|---|---|---|
| deepseek-v2-lite | 1 | `pp4-dp2-cp1-ep1` | 4 | 24,165 | 40.1% | 114.8 |
| deepseek-v2-lite | 2 | `pp2-dp8-cp1-ep2` | 4 | 24,256 | 40.3% | 94.8 |
| deepseek-v2-lite | 4 | `pp2-dp16-cp1-ep8` | 4 | 22,870 | 38.0% | 92.4 |
| gpt-oss-20b | 1 | `pp1-dp8-cp1-ep2` | 2 | 18,038 | 41.8% | 110.5 |
| gpt-oss-20b | 2 | `pp1-dp16-cp1-ep8` | 4 | 18,054 | 41.8% | 102.5 |
| gpt-oss-20b | 4 | `pp1-dp32-cp1-ep8` | 4 | 18,043 | 41.8% | 97.5 |
| gpt-oss-120b | 4 | `pp4-dp8-cp1-ep8` | 2 | 10,667 | 35.2% | 98.6 |
| qwen3-30b-a3b | 1 | `pp2-dp4-cp1-ep4` | 2 | 14,731 | 34.4% | 100.3 |
| qwen3-30b-a3b | 2 | `pp2-dp8-cp1-ep4` | 2 | 14,713 | 34.3% | 85.5 |
| qwen3-30b-a3b | 4 | `pp1-dp32-cp1-ep8` | 2 | 14,317 | 33.4% | 90.7 |
| qwen3.5-35b-a3b | 1 | `pp1-dp8-cp1-ep8` | 1 | 14,358 | 27.6% | 99.4 |
| qwen3.5-35b-a3b | 2 | `pp2-dp8-cp1-ep8` | 2 | 14,907 | 28.7% | 94.4 |
| qwen3.5-35b-a3b | 4 | `pp2-dp16-cp1-ep8` | 2 | 14,475 | 27.9% | 86.2 |

### 32K

| Model | Nodes | Case | Micro-batch | Throughput (tok/s/gpu) | MFU | Memory (GiB) |
|---|---|---|---|---|---|---|
| deepseek-v2-lite | 1 | `pp4-dp1-cp2-ep1` | 1 | 13,028 | 37.2% | 111.2 |
| deepseek-v2-lite | 2 | `pp2-dp4-cp2-ep2` | 1 | 13,096 | 37.5% | 92.0 |
| deepseek-v2-lite | 4 | `pp4-dp4-cp2-ep1` | 1 | 13,112 | 37.5% | 99.8 |
| qwen3-30b-a3b | 1 | `pp2-dp1-cp4-ep4` | 1 | 6,841 | 39.3% | 100.4 |
| qwen3-30b-a3b | 2 | `pp2-dp2-cp4-ep4` | 1 | 6,810 | 39.2% | 85.6 |
| qwen3-30b-a3b | 4 | `pp1-dp8-cp4-ep8` | 1 | 6,797 | 39.0% | 90.7 |
| qwen3.5-35b-a3b | 1 | `pp1-dp1-cp8-ep8` | 1 | 8,974 | 23.7% | 99.5 |
| qwen3.5-35b-a3b | 2 | `pp2-dp2-cp4-ep4` | 1 | 10,224 | 27.0% | 100.2 |
| qwen3.5-35b-a3b | 4 | `pp2-dp4-cp4-ep8` | 1 | 10,019 | 26.4% | 86.5 |
