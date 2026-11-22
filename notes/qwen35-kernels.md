# Qwen3.5-27B dense SM90 kernel inventory

Scope is the text tower in PithTrain at `ef2561e`, on H200 at sequence length 4,096. The one-node PP8 baseline uses CP1, EP1, no TP and microbatch size 1. The layer-level runs also test microbatch size 2. The 64 layers contain 48 Gated DeltaNet blocks and 16 gated-attention blocks. PithTrain’s FP8 path means `TrainingCfg.fp8=True` with its native DualPipeV/FSDP2 hooks and E4M3 operands. The step and block profiles use synthetic tokens. Weight, activation, and output-gradient quantization have different scaling recipes. The profile environment was CUDA 13.2, PyTorch 2.15.0+cu132, FlashAttention-4 4.0.0b28 and DeepGEMM 2.8.2 at the repository pin. [Model configuration](https://github.com/haok1402/pith-train/blob/ef2561e/examples/pretrain_lm/qwen3.5-27b/config.json).

## Shapes

Let `B` be microbatch size and `T=4096B`. GEMM forward is `[T,K] @ [N,K].T`. Its input gradient is `[T,N] @ [N,K]` and its weight gradient is `[T,N].T @ [T,K]`. For a DeepGEMM NT call the logical `(M,N,K)` triples are `(T,N,K)`, `(T,K,N)`, and `(N,K,T)`.

| Dense projection | Weight [N,K] | BF16 recipe | FP8 recipe |
|---|---|---|---|
| DeltaNet fused QKV | [10240,5120] | BF16 | FP8 |
| DeltaNet output-gate Z | [6144,5120] | BF16 | FP8 |
| DeltaNet decay A and write-strength B | [48,5120] each | One [96,5120] GEMM on the kernel branch. Baseline uses two | Two BF16 GEMMs |
| DeltaNet output | [5120,6144] | BF16 | FP8 |
| Full-attention Q and output gate | [12288,5120] | BF16 | FP8 |
| Full-attention K and V | [1024,5120] each | BF16 | FP8 |
| Full-attention output | [5120,6144] | BF16 | FP8 |
| SwiGLU gate and up | [17408,5120] each | BF16 | FP8 |
| SwiGLU down | [5120,17408] | BF16 | FP8 |
| LM head | [248320,5120] | BF16 | BF16 |
| Embedding | [248320,5120] | BF16 lookup | BF16 lookup |

DeltaNet uses 16 key heads and 48 value heads of dimension 128. The pinned kernel folds the batch into heads. The recurrence uses BF16 Q/K/V, BF16 beta and FP32 decay. Its Q/K have shape `[1,4096,16B,128]`, V and output are `[1,4096,48B,128]`, decay and write strength are `[1,4096,48B]`, and the stored chunk WY coefficient matrix has shape `[1,4096,48B,64]`. Before the recurrence, the causal depthwise convolution takes `[B,4096,10240]` activations and `[10240,1,4]` weights at offset 0. Full attention takes Q `[B,4096,24,256]` and K/V `[B,4096,4,256]`, with causal masking and rotary width 64. SwiGLU acts on `[B,4096,17408]`. Residual and final normalization use `[B,4096,5120]`; full-attention Q/K RMSNorm reduces the 256-element head dimension; DeltaNet output RMSNorm uses `[48T,128]` after flattening heads. The head loss consumes `[T,248320]` logits and `[T]` int64 targets. Its SM90 row indexing is 64-bit.

## BF16 launches

Source and test paths are relative to PithTrain. Raw trace and launcher paths are relative to the shared experiment workspace.

| CUDA implementation | Dense call site and workload | Existing correctness test or benchmark |
|---|---|---|
| `pithtrain/kernels/causal_conv1d_sm90.py` | `Qwen35MoeGatedDeltaNet.conv_qkv` → `operators/causal_conv1d.py` → `ConvFwdTma` and `ConvBwdTma` at this shape. Selective checkpointing recomputes forward. See the 10,240-channel convolution shape above. | `tests/kernels/test_causal_conv1d.py`, `tests/test_gated_deltanet_packed.py` |
| `pithtrain/kernels/gated_delta_rule_pinned_fwd_sm90.py` | `Qwen35MoeGatedDeltaNet.block` → `operators/gated_delta_rule.py:gated_delta_rule` → `pithtrain::gated_delta_rule_dense_fwd`. `WyRepresentation` and `ChunkStateOutput` at the 16:48 grouped shape above. | `tests/kernels/test_gated_delta_rule_pinned.py`, `tests/operators/test_gated_delta_rule_dense.py` |
| `pithtrain/kernels/gated_delta_rule_pinned_bwd_sm90.py` | Autograd of the same dense GDR op. `BwdPrologue`, `StateRecompute`, `StateGradient` and `ChunkGradients` rebuild the forward states and compute input and gate gradients. | Same pinned and dense GDR tests. The exact 16:48 grouping is covered by our benchmark and production profile. |
| `pithtrain/kernels/silu_mul_sm90.py` | `Qwen35MLP.forward` → `operators/silu_mul.py` → fused BF16 SiLU multiply and backward on the two `[B,4096,17408]` projections. PithTrain's default dense stage 3 is eager. Inductor uses the tensor formula in a fully compiled MLP. | `tests/kernels/test_silu_mul.py`, `tests/test_silu_mul.py`, benchmark case `silu` |
| `pithtrain/kernels/cross_entropy_sm90.py` | `tasks/pretrain_lm.py:objective` → `operators/cross_entropy.py`. `CountNonIgnored`, `CrossEntropy` and `MeanLoss` compute cluster-row softmax and CE, overwriting contiguous logits with the BF16 mean-loss gradient. Autograd scales the saved gradient with an ATen multiply. | `tests/kernels/test_cross_entropy.py`, benchmark cases `head`, `head-native`, `head-chunk` |
| FlashAttention-4, external `flash_attn.cute` SM90 | `Qwen35MoeAttention.block` → `operators/flash_attn_v4.py:flash_attn_func` → dense causal BSHD GQA forward/backward. The launcher memoization in `kernels/flash_attn_memoize.py` is host-side. | `tests/kernels/test_flash_attn_memoize.py`, `tests/test_dualpipev.py` dense 27B fixture, benchmark cases `fa4` and `attention-shared` |
| cuBLAS / cuBLASLt | `BF16Linear` and the two or fused gate projections. FSDP2/PP BF16 weight gradients use `operators/linear.py:wgrad_accumulate_` with BF16 inputs and an FP32 addmm accumulator. The LM head uses the same path. Muon also issues BF16 GEMMs as part of five Newton–Schulz iterations on eligible weight matrices. | `tests/test_dualpipev.py`, `tests/test_muon.py`, `tests/test_muon_fsdp.py`, benchmark case `gemms` |
| Inductor-generated Triton and PyTorch CUDA | The compiled dense stage 1 and stage 5 cover RMSNorm, residuals, partial RoPE, L2 normalization of DeltaNet Q/K, SiLU after convolution, decay, beta, and output gates. Additional ATen CUDA includes final norm, embedding backward, loss gradient scaling, AdamW and optimizer pointwise operations. | `tests/test_dualpipev.py`, benchmark cases `delta-shared`, `attention-shared`, `norm-compiled`, `embedding` |
| NCCL system kernels | PP>1 transfers activations and gradients through the DualPipeV P2P path, typically `[B,4096,5120]` BF16. DP>1 adds FSDP all-gathers and gradient reduce-scatters. FP8 FSDP gathers E4M3 projection shards and reduces scale maxima during preparation. The one-H200 traces exercise local weight preparation and omit multi-rank transport. | `tests/test_dualpipev.py`, `tests/test_muon_fsdp.py` |

## FP8 launches

The following list is for dense FP8 as shipped. FP8 projections produce BF16 outputs. The Gated DeltaNet convolution and recurrence, FA4, SwiGLU, embedding, BF16 LM head, and cross-entropy run the same CUDA implementations and shapes listed for BF16. The hidden-branch Muon orthogonalization remains BF16. FP8Linear weight gradients are produced in BF16 by DeepGEMM and the pipeline accumulates those in FP32 through side gradients. The LM head still uses the BF16 addmm accumulation path. Dense full attention adds selective checkpointing under FP8, and the checkpoint policy can recompute prequantized projection GEMMs while keeping the custom attention/recurrence results.

| Additional CUDA implementation | FP8 call site and 27B workload | Existing correctness test or benchmark |
|---|---|---|
| `pithtrain/kernels/fp8_rowwise_blockwise_transpose_sm90.py` | `FP8Linear.quantize_input` and `fp8_gemm_fwd` → `fp8_quant_act`. Input `[T,K]` BF16 with K in {5120, 6144, 17408}. Produces E4M3 input with 1×128 scales and E4M3 transposed input with 128×128 tile scales for wgrad. A quantized input is shared across Q/K/V, QKV/Z, or the MLP gate/up pair. | `tests/kernels/test_fp8_rowwise_blockwise_transpose.py`, `tests/test_deepgemm_fp8_linear_correctness.py` |
| `pithtrain/kernels/fp8_rowwise_transpose_sm90.py` | `FP8Linear` backward → `fp8_gemm_bwd` → `fp8_quant_grad`. Input `[T,N]` BF16 with N in {1024, 5120, 6144, 10240, 12288, 17408}. Produces both rowwise and transposed rowwise E4M3 operands for the two backward GEMMs. | `tests/kernels/test_fp8_rowwise_transpose.py`, `tests/test_deepgemm_fp8_linear_correctness.py` |
| `pithtrain/kernels/fp8_weight_quant_sm90.py` | The installed FP8 gather hooks call multi-weight shard amax and scale builders during `precompute_fp8_param_scales`, shard cast before all-gather, then pack and transpose once after gather. Operates on `[1,N_local,K]` parameter shards and `[1,ceil(N/128),ceil(K/128)]` scale tiles. Ungathered `FP8Linear` instead calls the same file's `fp8_quant_weight_sm90` on complete `[N,K]` weights and caches outside Inductor. | `tests/kernels/test_fp8_weight_quant.py`, `tests/test_deepgemm_fp8_linear_correctness.py`, `tests/test_dualpipev.py` |
| DeepGEMM, external | `operators/linear.py` → `kernels/fp8_gemm.py:fp8_gemm_nt` → `deep_gemm.fp8_fp4_gemm_nt` using two E4M3 operands. Forward, dgrad and wgrad triples are given in the shape table. The operator uses regular dense GEMM from a JIT binary. | `tests/kernels/test_fp8_gemm.py`, `tests/test_deepgemm_fp8_linear_correctness.py`, `benchmarks/operators/fp8/test_deepgemm.py` |

## Shared inlined or host-side code

| File | Where it enters | Timing and verifier |
|---|---|---|
| `cute_compile_sm90.py` | Host-side fake CuTe tensors, streams and compilation helpers for the SM90 CuTe kernels. | Host-side only. Coverage follows the caller `tests/kernels` and `tests/test_import_layers.py`. |
| `cute_ptx_sm90.py` | Inlined PTX helpers used by the convolution, pinned GDR, cross-entropy and SM90 quantization families. | Instructions are covered by the parent kernel time and its verifier. |
| `row_slice_sm90.py` | Cross-entropy's CTA row slicing, memory access and staging. | Inlined under `cross_entropy_sm90.py`; verified by `tests/kernels/test_cross_entropy.py`. |
| `fp8_e8m0_scale_sm90.py` | Inlined FP8 power-of-two scale and E4M3 conversion math. Hopper stores those scale values in FP32. | Counted in the FP8 quantization launches. Verified by the three FP8 quantizer tests and `tests/kernels/test_fp8_scale_layout.py`. |
| CuTeDSL, QuACK `sm90_utils`, `flash_attn.cute.ampere_helpers` | Python DSL and inlined SM90 helpers used in the pinned GDR and other CUDA sources. | Covered by the parent CUDA kernel and its verifier. |

## Other SM90 files and their call paths

| Files | Reason |
|---|---|
| `bf16_expert_gemm_sm90.py`, `bf16_expert_slots_sm90.py`, `ep_dispatch_sm90.py`, `token_scatter_sm90.py`, `group_indices_sm90.py`, `group_bias_add_sm90.py`, `segment_sum_sm90.py` | MoE routing, expert dispatch, expert slots and combine. |
| `fp8_scatter_quant_sm90.py`, `fp8_rowwise_kmajor_sm90.py` | Scattered or grouped expert FP8 paths. Dgrad/wgrad of dense FP8 use the single rowwise-transpose kernel above. |
| `clamped_swiglu_sm90.py`, `fp8_clamped_swiglu_quant_sm90.py` | Clamped GPT-OSS-style activation, distinct from Qwen3.5's ordinary SwiGLU. |
| `fp8_silu_mul_quant_sm90.py` | The dense Qwen3.5 MLP calls BF16 `silu_mul` then quantizes the down-projection input. The fused SwiGLU quantizer serves other model paths. |
| `gated_delta_rule_cp_sm90.py` | The distributed GDR path is reached at CP>1. CP1 uses the pinned forward and backward above. |
| `log_prob_sm90.py` | The pretraining objective calls cross-entropy. |

FLA's external Triton chunk kernels and FlashAttention-3 are alternatives measured in job `1263444`. These were standalone benchmarks. The training configuration uses PithTrain's CuTe implementation of the FLA-style chunk-64 gated delta rule. FlashAttention-3 came from Hugging Face Kernels v1. This branch retains FA4 in the dense attention call site. [FLA chunk implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py), [FA3 provider](https://huggingface.co/kernels-community/flash-attn3).

## Measured BF16 blocks and head

Job 1263562 profiled training forward plus backward on one H200, sequence 4,096, microbatch 1, with FP32 main-gradient accumulation. The measured GPU spans were 15.21 ms for DeltaNet, 14.98 ms for attention and 44.29 ms for the final norm, head and loss. The weighted column uses 48 DeltaNet spans, 16 attention spans, one head span and the separately timed 0.93 ms embedding. This estimates the kernel mix of one full-width, full-depth microbatch before optimizer, communication and pipeline scheduling.

| CUDA family | DeltaNet ms | Attention ms | Head ms | Weighted 27B compute share |
|---|---:|---:|---:|---:|
| Causal convolution | 0.172 | — | — | 0.81% |
| Pinned GDR forward | 0.322 | — | — | 1.52% |
| Pinned GDR backward | 0.731 | — | — | 3.46% |
| Fused SiLU multiply | 0.257 | 0.260 | — | 1.63% |
| Fused cross-entropy launches | — | — | 1.056 | 0.10% |
| FlashAttention-4 | — | 2.158 | — | 3.40% |
| cuBLAS training GEMMs | 12.091 | 11.709 | 39.492 | 79.53% |
| Inductor fused pointwise and reductions | 0.753 | 0.495 | — | 4.34% |
| Other ATen CUDA | 0.055 | 0.054 | 3.389 | 0.68% |
| Embedding, separate microbenchmark | — | — | — | 0.09% |
| Gaps between recorded GPU kernels | — | — | — | 4.43% |

The raw reports are logs/kernels-{delta-shared,attention-shared,head}-1263562.nsys-rep. The block times include the MLP. The BF16 head reports 1.056 ms for the CE cluster-row kernel and its two reductions. The remaining ATen row includes both final RMSNorm and the loss-gradient multiply.

The same job timed microbatch 2 at 32.10 ms for DeltaNet, 32.84 ms for attention, 100.03 ms for the unchunked head and 115.45 ms for the 1,024-token chunked head. The two head loss values agreed to 2e-6. The data favor the existing in-place fused CE path at both measured microbatch sizes.

On the exact attention geometry at microbatch 1, FA3 took 1.539 ms, FA4 2.203 ms and PyTorch Flash SDPA 3.115 ms. The FLA 0.6.0 Triton comparator took 1.656 ms with scale=1, while the selected CuTe DeltaNet took 1.182 ms at its standard 1/sqrt(128) scale. Those two timing results compare launch paths. Their output magnitudes use different scales.

## Measured BF16 and FP8 training-step shares

Job 1263636 ran one hybrid cycle on one H200 using the stock dense DualPipeV/FSDP2 training entry point. It kept the 27B widths, all embedding and output weights, 4K sequences, microbatch 1, two microbatches per step and Muon plus AdamW. The table covers step 3 after two warmups. The shorter model gives the head and optimizer more weight than the 64-layer, eight-GPU run.

Each share is summed GPU kernel time divided by the outer Nsight training-step NVTX duration. That duration was 685.95 ms for BF16 and 660.43 ms for FP8. PithTrain reported 685 ms and 659 ms, respectively. CUDA copies and launch gaps contribute to the wall time.

| CUDA family | BF16 launches | BF16 ms | BF16 share | FP8 launches | FP8 ms | FP8 share |
|---|---:|---:|---:|---:|---:|---:|
| Causal convolution | 18 | 1.089 | 0.16% | 18 | 1.082 | 0.16% |
| Pinned GDR forward | 12 | 2.013 | 0.29% | 12 | 2.018 | 0.31% |
| Pinned GDR backward | 24 | 5.293 | 0.77% | 24 | 5.354 | 0.81% |
| Fused SiLU multiply | 16 | 2.083 | 0.30% | 16 | 2.106 | 0.32% |
| Fused cross-entropy launches | 6 | 2.320 | 0.34% | 6 | 2.308 | 0.35% |
| FlashAttention-4 | 12 | 5.043 | 0.74% | 12 | 5.200 | 0.79% |
| FP8 activation quantization | — | — | — | 32 | 1.246 | 0.19% |
| FP8 output-gradient quantization | — | — | — | 50 | 2.318 | 0.35% |
| FP8 FSDP weight preparation | — | — | — | 52 | 4.505 | 0.68% |
| DeepGEMM | — | — | — | 168 | 59.315 | 8.98% |
| cuBLAS | 675 | 437.304 | 63.75% | 549 | 335.234 | 50.76% |
| Inductor | 204 | 5.836 | 0.85% | 212 | 6.376 | 0.97% |
| Other ATen CUDA | 1808 | 195.538 | 28.51% | 1858 | 201.895 | 30.57% |

Of the cuBLAS duration, 250.05 ms in BF16 and 251.82 ms in FP8 were launched after the last model backward NVTX range. Those launches are the optimizer GEMMs. That accounts for 36.5% and 38.1% of the respective step spans.

The FP8 trace recorded 168 regular DeepGEMM launches. There are 25 FP8 projections across the four layers, each with forward, dgrad and wgrad, plus nine recomputed projection forwards per microbatch. It recorded 32 activation quantization launches and 50 output-gradient quantization launches.

| FP8 weight kernel | Launches per step | GPU time |
|---|---:|---:|
| ShardTile | 25 | 1.889 ms |
| ShardAmaxMulti | 1 | 1.326 ms |
| PackTranspose | 25 | 1.280 ms |
| ScalesMulti | 1 | 0.010 ms |

The captures are logs/kernel-step-{bf16,fp8}-1263636.nsys-rep. Both SQLite exports passed integrity checks and contained the full NVTX range for step 3. Slurm recorded a kill signal at the process restart after both traces and both sets of four step logs were written. The eight-GPU baseline job 1263388 hit an OOM after its warmup step.
