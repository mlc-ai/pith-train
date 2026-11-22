# Kernel tests: the shape spec

Every kernel in `pithtrain/kernels/` is verified and benchmarked at the launch shapes training
produces, and only there. The shapes are not listed by hand: `cases.py` generates them at test
collection from each model's `config.json` and the mesh rules below, so a new model, mesh or kernel
gets its cases by construction. This page is the spec as people read it; `test_spec.py` fails
(smoke tier) when its per-model tables differ from what `cases.py` generates.

SM90 only (the bounds are calibrated on the H200): the SM100 kernels are out of scope here.

## Rules

| rule | value |
|---|---|
| models | every `examples/pretrain_lm/<model>/config.json` (one extractor per `model_type` in `cases.py`) |
| contexts and CP | 4k and 8k at CP 1; 16k at CP 2 and 4; 32k at CP 4 and 8; 64k at CP 8 and 16 (CP > 1 only where context / CP >= 4k) |
| micro-batch | 1, 2, 4 at CP 1; 1 under CP; tokens per GPU = mbs x context / CP |
| EP | 1, 2, 4, 8, 16 where it divides the expert count; dense models EP 1 |
| gpt-oss | no ring attention (window and sinks), so CP 1 only: contexts 4k and 8k |
| expert routing | balanced; log-normal (sigma 0.2, the measured spread); skewed (one expert at 4x the mean, one empty); stress (all of an EP group's tokens on one expert, at the largest EP per T); boundary (below) |
| boundary routing | the SM90 BF16 expert GEMMs pick a launch config per bucket of rows per expert, [2^b, 2^(b+1)) for b in 3 .. 17 (`pithtrain/kernels/tuning.py`); for every bucket the realistic routings reach at each EP the fused path runs (G = E / EP <= 64), every expert at 2^b and at 2^(b+1) - 1 rows |
| not in the spec | shipped scripts (`benchmarks/end2end`, `examples`) are not read; varlen ring attention (pretraining does not pack: `pretrain_lm` passes `cu_seqlens=None`); EP all-to-all and end-to-end runs |

How a mesh becomes kernel shapes (MoE parallel folding, `pithtrain/modules/distributed.py`): no
tensor parallelism; attention runs on `B = mbs` sequences of `S_local = context / CP`; the experts of
a rank see `T = mbs x S_local` tokens from each of the EP ranks of its group, about `T x top_k` rows
over `G = E / EP` local experts; pipeline parallelism only decides which layers a rank holds.

## What each mesh is tested for

| family | kernels | from each mesh | test |
|---|---|---|---|
| `moe_experts` | FP8 grouped GEMMs (DeepGEMM) and quantizations, fused SwiGLU (+ quant), BF16 fused experts, token scatter, dispatch indices, segment sums | `(T, EP)`: `G`, rows per local expert under each routing | `test_moe_experts.py` |
| `dense_gemm` | `fp8_quant_act/grad/weight`, `fp8_gemm_fwd/bwd` | every `training.Linear` at `M = T` | `test_dense_gemm.py` |
| `weight_quant` | FP8 weight quantization, parameter all-gather packing | every dense weight; expert weights at `G = E / EP` | `test_weight_quant.py` |
| `attention` | FlashAttention-4 | CP 1: dense causal `(B, context)`; CP > 1: the three zigzag ring-step launches at `S_local`; packed varlen at CP 1 | `test_attention.py` |
| `ring_attention` | the ring operators on CP GPUs | every CP > 1 mesh (CP 16 on two nodes) | `test_ring_attention.py` (multigpu) |
| `gated_delta_rule` | pinned GDN kernels; the CP operator on CP GPUs | Qwen3.5 meshes, batch folded into the heads | `test_gated_delta_rule.py` (CP > 1 multigpu) |
| `causal_conv1d` | `causal_conv1d_*` | Qwen3.5 meshes `(B, S_local)` | `test_causal_conv1d.py` |
| `cross_entropy` | the CrossEntropy op (`log_prob_*`), `cross_entropy_fwd_bwd` | `rows = T`, the full vocab | `test_cross_entropy.py` |
| `swiglu_dense` | `silu_mul_*`; DeepSeek-V2's fused FP8 MLP | dense and shared-expert MLPs at `M = T` | `test_swiglu_dense.py` |
| `pointwise_layers` | none: Inductor-compiled RMSNorm, RoPE, router | `(T, mbs)` | `test_pointwise_layers.py` |

A case is a dict with an `id` (`<family>/<model>/<label>`, e.g.
`moe_experts/qwen3-30b-a3b/T8k-ep4-lognormal`), its `tier`, its `axes` (the shape), the `meshes`
that launch it and, where known, its `weight` (FLOPs of one launch and the layers launching it per
micro-batch). Expert routing is materialized in the case (`group_sizes`), so the tests and the
benchmark run the same rows per expert.

## Tiers

| tier | what | run |
|---|---|---|
| `smoke` | one or two cases per (family, model), and the spec's own checks (`test_spec.py`) | `pytest tests/kernels` (default) |
| `spec` | every shape the meshes produce, and the boundary cases | `pytest tests/kernels -m "smoke or spec"` |
| `stress` | past balanced training: all of an EP group's tokens on one expert | `pytest tests/kernels -m stress` |
| `multigpu` | ring attention and GDN under CP, on CP GPUs (in no other tier) | `pytest tests/kernels -m multigpu -k "not cp8 and not cp16"` (4 GPUs), `-k cp8` (8), `-k cp16` (two nodes) |

## Running

Split a tier across the 8 GPUs of a node with `--shard I/N`:

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i pytest tests/kernels -m "smoke or spec or stress" --shard $i/8 &
done; wait
```

A `multigpu` test that skips for lack of GPUs fails instead (`tests/conftest.py`;
`--allow-multigpu-skip` to opt out). Each spawns its ranks through `tests/utilities.launch` on
`MASTER_PORT` (default 15213), so give concurrent runs on one node distinct ports. CP 16 runs one
pytest per node, eight ranks each, joined by the launcher's environment:

```bash
srun -N 2 --ntasks-per-node 1 --gres gpu:8 bash -c 'WORLD_SIZE=16 LOCAL_WORLD_SIZE=8 \
  NODE_RANK=$SLURM_NODEID MASTER_ADDR=<first node> pytest tests/kernels -m multigpu -k cp16'
```

## References and tolerances

Every reference is plain PyTorch in the test, never the kernel under test or another pithtrain
kernel: FP8 GEMMs against the fp32 product of the exactly dequantized operands (TF32 off), FP8
quantizations byte-exact against the E8M0 recipe (`fp8_reference.py`), BF16 expert GEMMs against
float64, attention and ring attention against fp32 attention with a hand-written backward
(`attention_reference.py`), GDN against the fp64 chunked recurrence (`gdn_reference.py`), cross
entropy, SwiGLU, RMSNorm and RoPE against float64.

A bound is analytic where the arithmetic gives one (bit-exact; within one bf16 rounding of the exact
value; half a bf16 ulp plus an accumulation term proportional to the sum of absolute products), and
otherwise about 3x the worst error measured on the H200 over the smoke, spec and stress tiers. Each
test module's docstring lists what it measured; `PITHTRAIN_SPEC_ERRORS`, `PITHTRAIN_SPEC_MEASURE`
and `PITHTRAIN_SPEC_ERROR_LOG` (named in each module) record the errors again. A bound that has to
grow to pass a new case is a finding, not a fix.

## Benchmarks

`benchmarks/kernels/bench_spec.py` times the same cases (CUDA events around single launches, L2
flushed, median / p10 / p90 per op), and with `--base REF` measures a change: the cases with the
pithtrain of the base ref (checked out as a worktree; `--base-dir DIR` for a checkout) and of the
working tree, on one GPU, interleaved op by op over several rounds, so both sides see the same GPU
and no baseline is stored. It reports per op base / head time and speedup, and per op name the
work-weighted speedup (work: FLOPs, or for memory-bound ops the elements read, times the layers
launching the case), and exits 1 when an op that ran at the base fails or an op name is more than
5% slower work-weighted. `--base-dir .` times the tree against itself, the noise floor: up to 10%
either way on single ops of about 1 GB (buffer placement), so read the weighted speedup.

```bash
python -m benchmarks.kernels.bench_spec --base upstream/main --family moe_experts --out ab.json
```

## Kernel changes and kernel-design agents

A change to `pithtrain/kernels/` reads this page for its shapes, and its PR posts the results of a
GPU run of `pytest tests/kernels -m "smoke or spec or stress"` (plus `-m multigpu` when CP code
changes) and of `bench_spec --base upstream/main` over the families it touches. An agent designing a
kernel takes its cases from `cases.py`, never from a shape it picks, must pass `pytest tests/kernels
-m "smoke or spec"` and the `--base` A/B against main, and may add cases (a new rule or edge case in
`cases.py` or its test) but never remove a spec case. Shape specialization is allowed only to shapes
in the spec, so a specialized route is always a verified one.

## Changing the spec

- New model: add `examples/pretrain_lm/<model>/config.json`; a new `model_type` needs an extractor in
  `cases.py`. Paste `python -m tests.kernels.cases --readme` into the Models section below and run
  `pytest tests/kernels -m "smoke or spec" -k <model>`.
- New kernel: list it in a family of `cases.FAMILIES` (or `SHAPE_INDEPENDENT`), and call it from that
  family's `test_<family>.py`, directly or through the operator the test drives.
- New mesh rule: change `cases.py` and this page together.

`test_spec.py` fails on each of these until it is done.

## Models

Generated by `python -m tests.kernels.cases --readme`; do not edit by hand.

### deepseek-v2-lite

| shape | value |
|---|---|
| model_type | `deepseek_v2` |
| hidden / vocab / layers | 2048 / 102400 / 27 |
| attention (27 layers) | 16 q / 16 kv heads, head dim 192 (v 128), MLA kv rank 512, rope dim 64 |
| experts (26 layers) | 64 experts, top 6, inter 1408, gate / up separate, swiglu |
| dense mlp (1 layer) | SwiGLU, inter 10944 |
| shared experts (26 layers) | SwiGLU, inter 2816 |
| dense projections (N x K) | q_proj 3072x2048, kv_a_proj_with_mqa 576x2048, kv_b_proj 4096x512, o_proj 2048x2048, dense_gate_up_proj 10944x2048, dense_down_proj 2048x10944, shared_gate_up_proj 2816x2048, shared_down_proj 2048x2816 |

| context | CP | mbs | tokens/GPU | EP |
|---|---|---|---|---|
| 4k | 1 | 1 | 4k | 1, 2, 4, 8, 16 |
| 4k | 1 | 2 | 8k | 1, 2, 4, 8, 16 |
| 4k | 1 | 4 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 1 | 8k | 1, 2, 4, 8, 16 |
| 8k | 1 | 2 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 4 | 32k | 1, 2, 4, 8, 16 |
| 16k | 2 | 1 | 8k | 1, 2, 4, 8, 16 |
| 16k | 4 | 1 | 4k | 1, 2, 4, 8, 16 |
| 32k | 4 | 1 | 8k | 1, 2, 4, 8, 16 |
| 32k | 8 | 1 | 4k | 1, 2, 4, 8, 16 |
| 64k | 8 | 1 | 8k | 1, 2, 4, 8, 16 |
| 64k | 16 | 1 | 4k | 1, 2, 4, 8, 16 |

Cases: dense_gemm 32, weight_quant 18, moe_experts 104, attention 14, ring_attention 6, cross_entropy 4, swiglu_dense 8, pointwise_layers 6.

### gpt-oss-120b

| shape | value |
|---|---|
| model_type | `gpt_oss` |
| hidden / vocab / layers | 2880 / 201088 / 36 |
| attention (36 layers) | 64 q / 8 kv heads, head dim 64, sliding window 128 on half the layers, sinks |
| experts (36 layers) | 128 experts, top 4, inter 2880, gate / up interleaved, clamped swiglu (alpha 1.702, limit 7), biases |
| dense projections (N x K) | q_proj 4096x2880, k_proj 512x2880, v_proj 512x2880, o_proj 2880x4096 |

| context | CP | mbs | tokens/GPU | EP |
|---|---|---|---|---|
| 4k | 1 | 1 | 4k | 1, 2, 4, 8, 16 |
| 4k | 1 | 2 | 8k | 1, 2, 4, 8, 16 |
| 4k | 1 | 4 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 1 | 8k | 1, 2, 4, 8, 16 |
| 8k | 1 | 2 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 4 | 32k | 1, 2, 4, 8, 16 |

Cases: dense_gemm 16, weight_quant 14, moe_experts 96, attention 16, cross_entropy 4, pointwise_layers 6.

### gpt-oss-20b

| shape | value |
|---|---|
| model_type | `gpt_oss` |
| hidden / vocab / layers | 2880 / 201088 / 24 |
| attention (24 layers) | 64 q / 8 kv heads, head dim 64, sliding window 128 on half the layers, sinks |
| experts (24 layers) | 32 experts, top 4, inter 2880, gate / up interleaved, clamped swiglu (alpha 1.702, limit 7), biases |
| dense projections (N x K) | q_proj 4096x2880, k_proj 512x2880, v_proj 512x2880, o_proj 2880x4096 |

| context | CP | mbs | tokens/GPU | EP |
|---|---|---|---|---|
| 4k | 1 | 1 | 4k | 1, 2, 4, 8, 16 |
| 4k | 1 | 2 | 8k | 1, 2, 4, 8, 16 |
| 4k | 1 | 4 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 1 | 8k | 1, 2, 4, 8, 16 |
| 8k | 1 | 2 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 4 | 32k | 1, 2, 4, 8, 16 |

Cases: dense_gemm 16, weight_quant 14, moe_experts 104, attention 16, cross_entropy 4, pointwise_layers 6.

### qwen3-30b-a3b

| shape | value |
|---|---|
| model_type | `qwen3_moe` |
| hidden / vocab / layers | 2048 / 151936 / 48 |
| attention (48 layers) | 32 q / 4 kv heads, head dim 128 |
| experts (48 layers) | 128 experts, top 8, inter 768, gate / up separate, swiglu |
| dense projections (N x K) | q_proj 4096x2048, k_proj 512x2048, v_proj 512x2048, o_proj 2048x4096 |

| context | CP | mbs | tokens/GPU | EP |
|---|---|---|---|---|
| 4k | 1 | 1 | 4k | 1, 2, 4, 8, 16 |
| 4k | 1 | 2 | 8k | 1, 2, 4, 8, 16 |
| 4k | 1 | 4 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 1 | 8k | 1, 2, 4, 8, 16 |
| 8k | 1 | 2 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 4 | 32k | 1, 2, 4, 8, 16 |
| 16k | 2 | 1 | 8k | 1, 2, 4, 8, 16 |
| 16k | 4 | 1 | 4k | 1, 2, 4, 8, 16 |
| 32k | 4 | 1 | 8k | 1, 2, 4, 8, 16 |
| 32k | 8 | 1 | 4k | 1, 2, 4, 8, 16 |
| 64k | 8 | 1 | 8k | 1, 2, 4, 8, 16 |
| 64k | 16 | 1 | 4k | 1, 2, 4, 8, 16 |

Cases: dense_gemm 16, weight_quant 14, moe_experts 96, attention 14, ring_attention 6, cross_entropy 4, pointwise_layers 6.

### qwen3.5-27b

| shape | value |
|---|---|
| model_type | `qwen3_5_text` |
| hidden / vocab / layers | 5120 / 248320 / 64 |
| attention (16 layers) | 24 q / 4 kv heads, head dim 256 |
| Gated DeltaNet (48 layers) | 16 key / 48 value heads, dims 128 / 128, conv width 4 over 10240 channels |
| mlp (64 layers) | SwiGLU, inter 17408 |
| dense projections (N x K) | q_proj 12288x5120, k_proj 1024x5120, v_proj 1024x5120, o_proj 5120x6144, in_proj_qkv 10240x5120, in_proj_z 6144x5120, out_proj 5120x6144, gate_up_proj 17408x5120, down_proj 5120x17408 |

| context | CP | mbs | tokens/GPU | EP |
|---|---|---|---|---|
| 4k | 1 | 1 | 4k | 1 |
| 4k | 1 | 2 | 8k | 1 |
| 4k | 1 | 4 | 16k | 1 |
| 8k | 1 | 1 | 8k | 1 |
| 8k | 1 | 2 | 16k | 1 |
| 8k | 1 | 4 | 32k | 1 |
| 16k | 2 | 1 | 8k | 1 |
| 16k | 4 | 1 | 4k | 1 |
| 32k | 4 | 1 | 8k | 1 |
| 32k | 8 | 1 | 4k | 1 |
| 64k | 8 | 1 | 8k | 1 |
| 64k | 16 | 1 | 4k | 1 |

Cases: dense_gemm 36, weight_quant 9, attention 14, ring_attention 6, gated_delta_rule 12, causal_conv1d 12, cross_entropy 4, swiglu_dense 4, pointwise_layers 6.

### qwen3.5-35b-a3b

| shape | value |
|---|---|
| model_type | `qwen3_5_moe_text` |
| hidden / vocab / layers | 2048 / 248320 / 40 |
| attention (10 layers) | 16 q / 2 kv heads, head dim 256 |
| Gated DeltaNet (30 layers) | 16 key / 32 value heads, dims 128 / 128, conv width 4 over 8192 channels |
| experts (40 layers) | 256 experts, top 8, inter 512, gate / up halves, swiglu |
| shared expert (40 layers) | SwiGLU, inter 512 |
| dense projections (N x K) | q_proj 8192x2048, k_proj 512x2048, v_proj 512x2048, o_proj 2048x4096, in_proj_qkv 8192x2048, in_proj_z 4096x2048, out_proj 2048x4096, shared_gate_up_proj 512x2048, shared_down_proj 2048x512 |

| context | CP | mbs | tokens/GPU | EP |
|---|---|---|---|---|
| 4k | 1 | 1 | 4k | 1, 2, 4, 8, 16 |
| 4k | 1 | 2 | 8k | 1, 2, 4, 8, 16 |
| 4k | 1 | 4 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 1 | 8k | 1, 2, 4, 8, 16 |
| 8k | 1 | 2 | 16k | 1, 2, 4, 8, 16 |
| 8k | 1 | 4 | 32k | 1, 2, 4, 8, 16 |
| 16k | 2 | 1 | 8k | 1, 2, 4, 8, 16 |
| 16k | 4 | 1 | 4k | 1, 2, 4, 8, 16 |
| 32k | 4 | 1 | 8k | 1, 2, 4, 8, 16 |
| 32k | 8 | 1 | 4k | 1, 2, 4, 8, 16 |
| 64k | 8 | 1 | 8k | 1, 2, 4, 8, 16 |
| 64k | 16 | 1 | 4k | 1, 2, 4, 8, 16 |

Cases: dense_gemm 36, weight_quant 19, moe_experts 88, attention 14, ring_attention 6, gated_delta_rule 12, causal_conv1d 12, cross_entropy 4, swiglu_dense 4, pointwise_layers 6.
