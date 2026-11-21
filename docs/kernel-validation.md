# Kernel validation: the shape spec, its tests and its benchmarks

Kernels in `pithtrain/kernels/` are tuned and verified against one checked-in list of the launch
shapes training produces, `tools/kernel_cases/kernel_cases.json`. Tests, benchmarks and
kernel-design agents read their cases from it; none of them chooses shapes of its own. This page is
the design: what the spec holds, how the cases are derived, the tiers, the references and
tolerances, and how each consumer uses it.

## 1. Why

Before the spec, every test hand-wrote its `parametrize` list and the only kernel benchmark used
DeepSeek-V3 shapes. Nothing tied a kernel's verified shapes to what a training mesh launches, so:

- routes the pickers take at production sizes had no correctness test (the SM100 FP8 grouped GEMM
  switches route at 512 rows per expert; every test stayed under 235);
- attention was checked only up to 1024 tokens per rank and only against FA4 itself;
- a model could be added (qwen3.5-27b) with no kernel case at all;
- kernel-design agents (TIRx-harness Quick Start) took "a representative input" from whatever the
  tests happened to contain, then validated against the same shapes.

## 2. The spec

```
python -m tools.kernel_cases            # exit 1 if kernel_cases.json is stale
python -m tools.kernel_cases --write    # regenerate it
python -m tools.kernel_cases --summary  # case counts per family / model / tier
python -m tools.kernel_cases --export DIR [--family F] [--tier T]   # JSONL for external harnesses
```

`tools/kernel_cases/` derives it from two inputs, both in the repo:

1. **Models**: every `examples/pretrain_lm/<model>/config.json`, read by one extractor per
   `model_type` (`models.py`) written against the model files. An unknown `model_type` fails
   generation, so a new config cannot land without cases.
2. **Meshes** (`meshes.py`):
   - the matrix: tokens per GPU `T` in {4k, 8k, 16k, 32k} x micro-batch `mbs` in {1, 2, 4} x CP
     in {1, 2, 4, 8} (CP 1 only for gpt-oss: ring attention has no window or sinks) x EP in
     {1, 2, 4, 8, 16, 32} with `E % ep == 0`; the sequence is `seq = T * cp / mbs`, kept between 1k
     and 128k with zigzag blocks of whole 128-token tiles;
   - every shipped mesh: `benchmarks/end2end/*/*/*/script.py` and `examples/pretrain_lm/*/script.py`,
     read statically, so a new benchmark or example lands in the spec on regeneration.

How a mesh maps onto kernel shapes (MoE parallel folding, `pithtrain/modules/distributed.py`):
there is no TP; attention sees `B = mbs` sequences of `S_local = seq / cp`; the experts of a rank see
`T = mbs * S_local` tokens from each of the `ep` ranks of its EP group, about `T * top_k` rows over
`G = E / ep` local experts; PP only decides which layers a rank holds. So each family depends on
few axes, and the ~1.5k cases are the deduplicated set over all mesh points:

| family | kernels (pithtrain.kernels) | axes of a case | from |
|---|---|---|---|
| `moe_experts` | FP8 grouped GEMMs and quantizations, fused SwiGLU (+quant), BF16 fused experts, scatter / dispatch / segment sum | `T, ep, G, hidden, inter, act, bias, routing, group_sizes` | (T, ep) per model; routing balanced, log-normal (sigma 0.2, the measured spread), skewed (one expert at 4x the mean, one empty), stress (all EP-group tokens on one expert), boundary (both sides of every SM100 route threshold) |
| `dense_gemm` | `fp8_quant_act/grad/weight`, `fp8_gemm_fwd/bwd/nt` | `M = T, N, K` per projection | every `training.Linear` of the model at each T |
| `weight_quant` | `fp8_quant_weight*`, FP8 param-gather kernels | `G, N, K` | dense projections; expert weights per EP |
| `attention` | FA4 (`memoized_launch` around it) | `kind, B, Sq, Skv, Hq, Hkv, D, Dv, causal, window, sinks[, cu_seqlens]` | CP 1: dense causal `(B, S, S)`; CP > 1: the three zigzag ring launches, `ring_diag (S_local, S_local)` causal, `ring_front (S_local, S_local/2)` with a strided k / v view, `ring_back (S_local/2, S_local)` with a strided q view; varlen at CP 1 |
| `ring_attention` | the ring operators on `cp` GPUs | `cp, B, S_local, heads, mla` | every CP > 1 mesh |
| `gated_delta_rule` | pinned GDN kernels (CP 1), CP pieces (CP > 1) | `cp, mbs, T_local, H = 16 mbs, HV = (32 or 48) mbs` | Qwen3.5 meshes (batch folded into heads) |
| `causal_conv1d` | `causal_conv1d_*` | `cp, B, T_local, C, width` | Qwen3.5 meshes |
| `cross_entropy` | `cross_entropy_fwd_bwd`, `log_prob_*` | `rows = T, vocab` | each T (last pipeline stage) |
| `swiglu_dense` | `silu_mul_*` | `M = T, inter` | dense / shared-expert MLPs |
| `pointwise_layers` | none (Inductor) | `T, hidden, heads, E, top_k` | listed for completeness; no spec test yet |

Expert routing is materialized in the spec (`group_sizes`), not re-drawn by consumers, so a test, a
benchmark and an agent run the same per-expert row counts. The FP8 path pads each group to a
multiple of 128 rows; `fp8_rows` is that total.

Each case is one line of JSON:

```json
{"id":"moe_experts/qwen3-30b-a3b/013","family":"moe_experts","model":"qwen3-30b-a3b","tier":"smoke",
 "axes":{"T":4096,"ep":8,"G":16,"rows":32768,"group_sizes":[...],...},
 "meshes":["T4k-mbs1-cp1-ep8","benchmarks/end2end/..."],"weight":{"flops_fwd":...,"layers":48}}
```

- `id` is `<family>/<model>/<index>`; `meshes` lists every mesh that launches the shape, so a
  case is traceable to the runs that need it;
- `weight` is FLOPs per micro-batch of one launch and the number of layers launching it, for
  weighting an aggregate score by share of step time;
- the header records each config's sha256, the matrix, the shipped meshes and the routing
  constants, and per family its kernels, reference and tolerance.

## 3. Tiers

Cases carry a tier; the spec-driven tests (`tests/spec/`) mark each param with it:

| tier | what | how to run |
|---|---|---|
| `smoke` | one or two cases per (family, model), at T = 4k | default: `pytest tests/spec` |
| `spec` | every case the meshes produce, plus the route-boundary cases | `pytest tests/spec -m "smoke or spec"` |
| `stress` | past balanced training: all EP-group tokens on one expert | `pytest tests/spec -m stress` |
| `multigpu` | ring attention and GDN under CP, on `cp` GPUs of one node (ring attention keeps its CP 8 cases in `test_ring_attention_cp8`, so `-k "not cp8"` fits 4 GPUs there) | `pytest tests/spec -m multigpu` |

`pyproject.toml` deselects `spec`, `stress` and `multigpu` by default (unmarked tests are
unaffected). A `multigpu` test that skips for lack of GPUs fails instead (`tests/conftest.py`;
`--allow-multigpu-skip` to opt out), so a CP run cannot silently test nothing. `--shard I/N` splits a
selection across N processes, one per GPU:

```bash
CUDA_VISIBLE_DEVICES=$i pytest tests/spec -m "smoke or spec" --shard $i/8
```

## 4. References and tolerances

Every reference is plain PyTorch written in the test, never another pithtrain kernel and never the
kernel under test:

- FP8 GEMMs: fp32 matmul (TF32 off) of the exactly dequantized FP8 operands, so the only error left
  is accumulation order and the bf16 output rounding; quantizations bit-exact against the E8M0
  recipe;
- BF16 grouped GEMMs: float64, half a bf16 ulp plus 1e-5 of the sum of absolute products;
- attention: fp32 attention computed in query blocks with a hand-written fp32 backward and a random
  `dout` (not FA4, not an all-ones gradient); ring attention against the same reference over the
  whole gathered sequence;
- GDN: the fp64 chunked recurrence; conv1d: fp32 `F.conv1d`; cross entropy: float64 log-softmax
  through the autograd op with a `grad_output` that is not 1.

Bounds are calibrated, not guessed: each test module's docstring lists the largest error measured on
the H200 over the whole spec tier and sets its bound at about 3x that, the method
`tests/kernels/test_gated_delta_rule_pinned.py` already used. A bound that has to grow to pass a new
case is a finding, not a fix.

A test that exposes a kernel bug keeps the case and marks it `xfail(strict=False)` with a reason
naming the bug; the kernel is fixed in its own change.

Bounds as calibrated on the H200 (worst measured over smoke + spec + stress, then the bound):

| family | output | worst measured | bound |
|---|---|---|---|
| moe_experts (FP8) | GEMM bf16 out, per element: coef of sum \|ab\| / calc_diff | 1.4e-4 / 1.39e-6 | 2^-11 / 5e-6 |
| | k-grouped wgrad fp32: coef / calc_diff | 5.3e-4 / 7.8e-9 | 2^-9 / 3e-8 |
| | fused activation quant | 0.94 of e4m3 + bf16 rounding | 1.0 |
| moe_experts (BF16) | outputs / main grads coef | 2.3e-6 / 8.0e-5 | 1e-5 / 2.5e-4 |
| dense_gemm | y, dx, dw coef / calc_diff vs exact dequant / vs bf16 truth | 9.9e-5 / 1.39e-6 / 7.15e-4 | 3e-4 / 3e-6 / 2e-3 |
| weight_quant | bytes and scales | exact | exact |
| attention | out / dq, dk, dv / dsink (rel L2); lse (abs) | 2.3e-3 / 2.5e-3 / 3.1e-3; 3.8e-6 | 7e-3 / 7.5e-3 / 1e-2; 2e-5 |
| ring_attention | out / grads / MLA latent grads (rel L2) | 3.1e-3 / 3.0e-3 / 4.1e-3 | 1e-2 / 9e-3 / 1.25e-2 |
| gated_delta_rule | calc_diff o .. dbeta; max-abs / ref max | <= 1.1e-5; 8.2e-3 | 2e-5 .. 1e-4; 2e-2 |
| causal_conv1d | y, dx, dw (fraction of one bf16 rounding + fp32 slack) | 0.996 | 1.0 |
| cross_entropy | loss rel; grad (fraction of one rounding); log_prob abs | 1.9e-7; 0.996; 1.9e-6 | 1e-6; 1.0; 1e-5 |
| swiglu_dense | out, dg, du (fraction of one rounding) | 0.9998 | 1.0 |

Set `PITHTRAIN_SPEC_ERRORS` / `PITHTRAIN_SPEC_MEASURE` / `PITHTRAIN_SPEC_ERROR_LOG` (named in each
module) to record the errors again before changing a bound, e.g. on SM100.

Findings the spec tests exposed (xfail until fixed):

- `CrossEntropy.backward` rounds `grad_output` to bf16 before scaling (`inp.mul_(grad_output.to(inp.dtype))`),
  2.65x the one-rounding bound at `grad_output = 1/3` and a systematic 0.2% scale error
  (all `cross_entropy` operator cases);
- SM90 `fp8_quant_grouped` and the fused activation quantizations raise once rows x channels
  reaches 2^31 (int32 indexing; 11 `stress` cases, e.g. gpt-oss-20b at T = 16k, EP 32);
- SM90 `fp8_pack_transpose` raises for outputs of 2^31 bytes or more, so gpt-oss-120b `gate_up` at
  EP 1 cannot be FP8 param-gathered (`weight_quant/gpt-oss-120b/004`).

Not a bug but measured: `FP8Linear` has no fp32 `main_grad`, so each micro-batch's dense FP8 weight
gradient is rounded to bf16 before it is summed (relative L2 1.7e-3 per term;
`test_spec_dense_gemm.py::test_wgrad_bf16_rounding` pins it).

Running the multi-GPU tiers: each case spawns its ranks through `tests/utilities.launch`, which uses
`MASTER_PORT` 15213 unless set, so give concurrent runs on one node distinct ports. Under `srun -n N`,
`-c` is per task.

## 5. Benchmarks

`benchmarks/kernels/bench_spec.py` times the same cases:

```bash
python -m benchmarks.kernels.bench_spec --tier smoke --out results.json
python -m benchmarks.kernels.bench_spec --tier smoke --compare benchmarks/kernels/baselines/h200.json
```

It writes per case and op the median, p10 and p90 time (CUDA events, L2 flushed between reps),
TFLOP/s, and the GPU, SM count, library versions and git sha. `--compare` applies the
`docs/kernel-styles.md` section 9 rule (median within 5% + 3 us) per op and exits 1 on a
regression. Baselines are stored per GPU under `benchmarks/kernels/baselines/`; regenerate one with
nothing else running on the node.

## 6. Kernel-design agents and TIRx-harness

An agent tuning or writing a PithTrain kernel takes its cases from the spec, never from
nearby tests:

```bash
python -m tools.kernel_cases --export /tmp/cases --family moe_experts --tier spec
```

writes `moe_experts.jsonl`, one case per line with the family's kernels, reference, tolerance and
the config sha256 it came from. The intended TIRx-harness integration (not part of this repo):

- an importer turns a family's JSONL into a pinned task (`definition.json` + `workload.jsonl`),
  using `weight` for the aggregate score instead of an unweighted mean;
- Quick-Start skills (`tirx-debug-kernel`, `tirx-profile-kernel`) load these cases instead of
  "a representative input";
- an agent may add edge cases of its own but may not remove or replace spec cases, and a kernel
  is accepted only if every `spec` case of its family passes the PithTrain test and does not
  regress in `bench_spec --compare`.

Shape specialization stays allowed, but only to shapes in the spec, so a specialized route is
always a verified one.

## 7. Changing things

- **New model**: add `examples/pretrain_lm/<model>/config.json`; if its `model_type` is new, add an
  extractor in `tools/kernel_cases/models.py`. Run `python -m tools.kernel_cases --write`, review
  the new cases, run `pytest tests/spec -m "smoke or spec"`.
- **New mesh or benchmark**: add the script; regenerate.
- **New kernel**: list it in a family in `tools/kernel_cases/cases.py` (or in
  `SHAPE_INDEPENDENT`), and make that family's `tests/spec/test_spec_<family>.py` drive it.
- `tests/test_kernel_cases.py` (CPU only) fails on each of these until it is done: stale spec,
  a config or shipped mesh without cases, a kernel in no family, a family without a spec test, an
  SM100 route threshold with cases on one side only.

## 8. Arch coverage

The same cases run on SM90 and SM100. On an H200 the SM100 TIRx kernels never run, so the routes
below are verified only by a B200 run of the same tiers:

- `fp8_grouped_gemm` / `k_grouped_fp8_gemm` (TIRx `fp8_grouped_gemm_sm100`, all routes: w320r, 1-CTA,
  swap, w320, rotating-256, scale-tail; k-grouped short / long);
- every `bf16_expert_*` SM100 route (swap-AB, materialized gate_up, far, 240-col tiles), the slots
  rescan kernel;
- the SM100 quantizers, fused SwiGLU / scatter quantization (wide and tail paths), segment sum,
  group bias add, token scatter, EP dispatch, cross entropy / log-prob, causal conv1d, the pinned GDN
  kernels and the `chunk_gdr_cp_*` stages.
