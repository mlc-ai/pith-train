# SM100 kernels on the standalone TIRx compiler (tvm_tirx)

Branch `haok/tirx-standalone-sm100` (from `haok/tirx-evolved-sm100` @ bf9d16f). pith-train's 21 SM100
TIRx kernel modules move from apache-tvm 0.27 to the standalone TIRx compiler, so pith-train drops
the full `apache-tvm` dependency. The evolved schedules are unchanged; only the API changed.

**Verdict (in progress).** <!--VERDICT-->Every kernel builds on tvm_tirx with no new intrinsics or
hand-written PTX. On a B200, every kernel verified so far produces outputs bitwise identical to the
apache-tvm 0.27 build on its frontier and held-out workloads (token_scatter is nondeterministic on
both stacks and passes its check), and pith-train's `tests/kernels` gives the same result on both
stacks. The rows marked "not yet verified" are still running. The emitted SASS differs in scalar
code nearly everywhere (one upstream pipeline gap above all: no CSE inside kernel bodies) and, in
the bf16 expert GEMM, in nvcc's unrolling; speed was not measured cleanly enough to quote (other
users' jobs shared the only usable GPU), see the codegen section.<!--/VERDICT-->

## Stacks

| | old | new |
|---|---|---|
| compiler | apache-tvm 0.27.0.post1 (package `tvm`) | apache-tvm-tirx (package `tvm_tirx`): pinned **0.1.0a6** (PyPI); every kernel also built and compared on **0.1.0b0** (apache/tvm-tirx@7c19669, built from source, no wheel published) |
| FFI | apache-tvm-ffi 0.1.14.post0 | apache-tvm-ffi 0.1.15.post2 (+ apache-tvm-ffi-orcjit 0.1.3, a dependency of apache-tvm-tirx) |
| tirx-lite | tirx-kernels 0.1.2.post1 (= mlc-ai/TIRx-kernels@8e4e8fb) | mlc-ai/TIRx-kernels@716d8a3 (main, after the tvm_tirx migration #6-#14); PyPI has no release with the new API, so it is a git source |
| nvcc | CUDA 13.2, `TVM_CUDA_COMPILE_MODE=nvcc`, `-gencode arch=compute_100a,code=sm_100a` fatbin | CUDA 13.2, nvcc chosen in each module's `TARGET` (`backend_config`), sm_100a **cubin** |

Same nvcc flags otherwise (`-O3`, `--use_fast_math`, the `-U__CUDA_NO_*` set, ptxas
`-v --warn-on-local-memory-usage --register-usage-level=10`); gated_delta_rule_cp keeps fast math
off through its `TARGET`'s `nvcc` list (it used `TVM_CUDA_NVCC_NO_FAST_MATH` before).

## Dependencies

`pyproject.toml`:

```diff
-    "apache-tvm==0.27.0.post1",
-    "apache-tvm-ffi==0.1.14.post0",
+    "apache-tvm-ffi==0.1.15.post2",
+    "apache-tvm-tirx==0.1.0a6",
 ...
-    "tirx-kernels==0.1.2.post1",
+    "tirx-kernels",
 ...
+tirx-kernels = { git = "https://github.com/mlc-ai/TIRx-kernels", rev = "716d8a3cfc8cb5810f79f2515a39d400c1264340" }
```

`uv.lock`: apache-tvm removed; apache-tvm-ffi 0.1.14.post0 -> 0.1.15.post2; apache-tvm-ffi-orcjit
0.1.3 and apache-tvm-tirx 0.1.0a6 added; tirx-kernels 0.1.2.post1 -> 0.1.2.post2.dev11+g716d8a3cf.
No other package moved (flash-attn-4, quack and cutlass-dsl accept tvm-ffi 0.1.15).

Size of the TVM-related packages (x86_64, cp312):

| | wheels (download) | installed |
|---|---|---|
| old: apache-tvm 0.27.0.post1 + apache-tvm-ffi 0.1.14.post0 + tirx-kernels 0.1.2.post1 | 95.9 + 3.1 + 2.4 = **101.4 MiB** | 287.1 + 10.2 + 11.8 = **309.1 MiB** |
| new (pinned a6): apache-tvm-tirx 0.1.0a6 + apache-tvm-ffi 0.1.15.post2 + orcjit 0.1.3 + tirx-kernels@716d8a3 | 43.7 + 3.2 + 15.2 + ~2.5 = **64.6 MiB** | 106.0 + 10.5 + 36.3 + 12.1 = **164.9 MiB** |
| new (b0, local build) | 65.5 + 3.2 + 15.2 + ~2.5 MiB | 190.2 + 10.5 + 36.3 + 12.1 = 249.1 MiB |

The rest of the environment (torch 2.14 + its CUDA wheels, several GB) is unchanged, so the venv
shrinks by ~144 MiB (pinned a6).

Compile time: the new stack compiles the same kernels 2-7x slower on the CPU (gated_delta_rule_cp
first call 41 s -> 82 s of a harness run; fp8_weight_quant's 21 builds 17 s -> 114 s), which is a
one-time cost per process per kernel.

## What upstream changed (TIRx-kernels #6-#14 and tvm_tirx), and what the port needed

Beyond import names (`tvm` -> `tvm_tirx`, `tvm.backend.cuda.cpp.descriptors` ->
`tvm_tirx.backend.cuda.cpp.descriptors`):

- **Kernel entry.** `@txl.kernel(warps=, arch=, grid=, min_blocks_per_sm=, host_prelude=)` ->
  `@txl.kernel()` plus a `txl.device_entry(launch=txl.cuda.LaunchConfig(grid=, block=, cluster=,
  preferred_cluster=, cooperative=, programmatic_stream_serialization=), kernel_attrs=
  txl.cuda.KernelAttributes(min_blocks_per_sm=, required_block_size=))` statement. Grid lambdas
  over `p["name"]` become expressions of the bound parameters. Host preludes (tensor-map encoding)
  are ordinary code before the entry. `func.with_attr("tirx.launch_tags" /
  "tirx.kernel_launch_params", ...)` (clusters, PDL, cooperative launch, dynamic shared memory)
  become `LaunchConfig` fields; dynamic shared memory is sized automatically; clusters above 8 are
  allowed by the launcher.
- **Types.** `txl.gptr[dtype]` / `txl.gptr(dtype, shape=lambda p: ...)` -> `txl.Tensor(shape,
  dtype)` with `txl.Var` extents shared with the scalar parameters they equal. The host now
  **checks** tensor shapes: two parameters given the same Tensor type object (hence the same
  symbolic extent) must have equal lengths. dtype tokens (`txl.bf16`, `txl.f32`, ...) are
  `PrimType`s, not strings (`"float32" == txl.f32` but they hash differently, so dicts keyed on one
  miss the other); scalar annotations use them (`n: txl.int32`; string annotations work but are
  F821 to ruff).
- **Control flow / builders.** `If/Then/Else/While` -> `if_/then_/else_/while_`; `alloc_buffer`
  -> `alloc_tensor`; `SW128B` etc. -> `SwizzleMode.SWIZZLE_128B_ATOM` etc.;
  `tvm_warp_shuffle_up` -> `gpu_warp_shuffle_up`; `txl.reinterpret(dtype, x)` ->
  `txl.reinterpret(x, ty=dtype)` (**the old order still parses but types the result Missing**,
  which fails later or drops a statement); `txl.Cast(dtype, x)` keeps its order on a6 and b0
  (upstream kernels at 716d8a3 already write `Cast(x, dtype)`, which a6 and b0 reject).
- **Cluster rank.** `txl.cta_id_in_cluster([C])` -> the cluster launch plus a rank read.
  Upstream's `txl.cuda.cluster_cta_id("x")` reads `%cluster_ctaid.x` (ptxas lowers it with a
  division); the old code read `%cluster_ctarank`. Equal for 1-D clusters; `txl.cuda.mov_sreg(32,
  "cluster_ctarank")` keeps the old SASS (used in causal_conv1d, gated_delta_rule_cp,
  cross_entropy).
- **Compilation.** `.compile()` / `tvm.compile(..., tir_pipeline="tirx")` ->
  `.compile(TARGET)` / `tvm_tirx.tirx.compile(..., pipeline="tirx")`. A CUDA target without an
  arch makes tvm_tirx query the device, so every module passes an explicit `TARGET` (sm_100a).
  The compiler is chosen by the target's `backend_config`, not `TVM_CUDA_COMPILE_MODE` (default
  NVRTC). tvm_tirx's default nvcc output is a fatbin built with `-arch sm_100a`, which also emits
  compute_100 PTX, and ptxas rejects sm_100a-only instructions there (tcgen05,
  `cvt...satfinite.ue8m0x2`, ...): `TARGET` asks for a cubin (sm_100a SASS only, as the old
  `-gencode` build).
- **Two behaviour changes, neutralized in the port:**
  - `txl.cuda.wait_until` is registered without a return type (`target_builtin.cc`), so the call
    is typed Missing and silently **not emitted**. ep_dispatch's grid spin and the clamped
    backward's d_bias exit protocol call it with `ty=txl.void`. Upstream kernels at 716d8a3 call
    it without (`kda_backward_packed`, `ported/deepep/combine.py`, ...), and are likely affected.
  - Float `txl.max` / `txl.min` now emit NaN-propagating selects (`codegen_cuda.cc`
    `PrintMinMaxNanPreservingImpl`); apache-tvm emitted CUDA `max` / `min` (fmaxf / fminf, which
    drop a NaN). cross_entropy, log_prob, fp8_scatter_quant and fp8_silu_mul_quant call
    `old_max` / `old_min` (fmaxf / fminf for float32) so NaN handling stays the old one.
- No kernel needed a new intrinsic or hand-written PTX.

## Codegen, per kernel (CPU only: tvm_tirx and apache-tvm, same nvcc 13.2)

Each module's cached builders were traced and compiled for every configuration training uses (the
four models' shapes, each route and fallback; ~517 device groups in all) under both stacks, and the
cubins compared with `cuobjdump`. Classes per device group: **identical** SASS; **cosmetic** (same
SASS once symbol names and constant-bank offsets are ignored); **reordered** (same opcode counts and
resources, different order); **scalar** (counts differ only in scalar / address / predicate
instructions, registers may move); **family** (a tracked instruction family changes: tcgen05, TMA,
mbarrier, global or shared ld/st, BAR, MUFU, NaN min/max). "b0+CSE" is a diagnostic build of b0 whose
common-subexpression pass also enters device regions (below); it isolates that one pipeline change.


<!--CODEGEN-->
| module | device groups | vs b0: identical / cosmetic / reordered / scalar / family | vs b0+CSE | family changes (b0) | registers / stack changes (b0) |
|---|---|---|---|---|---|
| group_bias_add_sm100 | 5 | 0 / 0 / 0 / 5 / 0 | 5 / 0 / 0 / 0 / 0 | none | 2 groups, REG -6..-6 |
| causal_conv1d_sm100 | 16 | 10 / 2 / 0 / 4 / 0 | 10 / 6 / 0 / 0 / 0 | none | 4 groups, REG +1..+2 |
| gated_delta_rule_cp_sm100 | 14 | 4 / 0 / 3 / 7 / 0 | 6 / 0 / 1 / 7 / 0 | none | 1 groups, REG +2..+2 |
| gated_delta_rule_pinned_sm100 | 21 | 14 / 0 / 0 / 7 / 0 | 14 / 0 / 1 / 6 / 0 | none | REG same; STACK 16->32 |
| fp8_grouped_gemm_sm100 | 17 | 1 / 0 / 2 / 14 / 0 | 1 / 1 / 1 / 14 / 0 | none | 2 groups, REG -4..-4 |
| bf16_expert_gemm_sm100 | 68 | 1 / 0 / 0 / 37 / 30 | 1 / 0 / 0 / 37 / 30 | tma +12..+36, mbarrier -12..+164, gmem -4, smem -44..+164 | 25 groups, REG -14..+15; STACK 0->8, 0->8 |
| bf16_expert_slots_sm100 | 32 | 0 / 0 / 1 / 22 / 9 | 4 / 0 / 1 / 18 / 9 | mufu +1, gmem +8, smem +1..+16 | 23 groups, REG -3..+14; STACK 0->24 |
| token_scatter_sm100 | 10 | 4 / 0 / 0 / 6 / 0 | 4 / 0 / 0 / 6 / 0 | none | REG same |
| segment_sum_sm100 | 37 | 0 / 14 / 0 / 23 / 0 | 0 / 14 / 0 / 23 / 0 | none | 23 groups, REG -10..+130 |
| fp8_clamped_swiglu_quant_sm100 | 32 | 8 / 2 / 0 / 16 / 6 | 8 / 2 / 0 / 16 / 6 | gmem +8..+16 | 11 groups, REG -5..+26; STACK 8->0, 16->0 |
| fp8_scatter_quant_sm100 | 14 | 3 / 0 / 0 / 11 / 0 | 3 / 0 / 0 / 11 / 0 | none | 1 groups, REG +1..+1 |
| fp8_silu_mul_quant_sm100 | 29 | 7 / 0 / 13 / 9 / 0 | 7 / 0 / 13 / 9 / 0 | none | 3 groups, REG -2..+8 |
| fp8_weight_quant_sm100 | 21 | 4 / 5 / 0 / 12 / 0 | 4 / 5 / 0 / 12 / 0 | none | 4 groups, REG -12..-2 |
| fp8_rowwise_blockwise_transpose_sm100 | 5 | 4 / 1 / 0 / 0 / 0 | 4 / 1 / 0 / 0 / 0 | none | REG same |
| fp8_rowwise_transpose_sm100 | 5 | 0 / 0 / 3 / 2 / 0 | 2 / 0 / 3 / 0 / 0 | none | 1 groups, REG -2..-2 |
| fp8_rowwise_colwise_sm100 | 9 | 4 / 0 / 0 / 5 / 0 | 6 / 0 / 0 / 3 / 0 | none | 1 groups, REG +1..+1 |
| cross_entropy_sm100 | 33 | 3 / 6 / 0 / 0 / 24 | 3 / 6 / 0 / 0 / 24 | smem +1 | REG same |
| log_prob_sm100 | 8 | 8 / 0 / 0 / 0 / 0 | 8 / 0 / 0 / 0 / 0 | none | REG same |
| ep_dispatch_sm100 | 44 | 8 / 12 / 0 / 24 / 0 | 8 / 12 / 0 / 24 / 0 | none | REG same |
| clamped_swiglu_sm100 | 12 | 12 / 0 / 0 / 0 / 0 | 12 / 0 / 0 / 0 / 0 | none | REG same |
| silu_mul_sm100 | 24 | 24 / 0 / 0 / 0 / 0 | 24 / 0 / 0 / 0 / 0 | none | REG same |
<!--/CODEGEN-->

The new pipeline differs from apache-tvm 0.27 in ways that touch nearly every kernel's scalar code;
none changes what memory, tensor-core, barrier or TMA operations a kernel issues:

1. **No CSE inside kernel bodies.** tvm-tirx's `CommonSubexprElim` treats every region other than
   `launch_thread` as an optimization boundary (`src/tirx/transform/common_subexpr_elim.cc` ~L615,
   ~L838), and the kernel body is a `tirx.device_scope` region at that point, so nothing in it is
   CSE'd; apache-tvm CSE'd it (the `cse_v*` temporaries, some hoisted to host-computed kernel
   arguments). Patching the pass to enter `device_entry` / `device_scope` regions ("b0+CSE")
   restores the old SASS exactly for group_bias_add, log_prob, causal_conv1d's plain kernels and
   several gated_delta_rule / fp8_rowwise_colwise / fp8_rowwise_transpose groups.
2. **`blockIdx.x` has no known range** (old: an IterVar over [0, grid)), so `bx // c` / `bx % c`
   get sign fix-ups (`(bx + 2147483652) / 12 - 178956971`).
3. **Loop variables are not known to be non-negative**, leaving `0 <= v` guards (gated_delta_rule
   fpass: STACK 16 -> 32 B).
4. **Launch-only parameters leave the kernel signature** (a scalar used only in the grid is no
   longer passed), shifting constant-bank offsets.
5. **Shared-memory pool views are typed pointers per buffer** instead of offsets off one base:
   same addresses, but nvcc no longer merges some neighbouring loads (cross_entropy: one `LDS.64`
   becomes two `LDS`).
6. **Branch facts are not propagated** into the branch (clamped quant fallback: a dead store branch
   stays, +8-16 STG that never execute), and some constants fold that did not.
7. **nvcc unrolling flips** in the bf16 expert GEMM: with the same loop bodies modulo naming,
   nvcc 13.2 fully unrolls some producer loops it unrolled 4x before (one kernel: 12 -> 48
   `cp.async.bulk.tensor` in PTX), so TMA / mbarrier / shared ld-st counts and registers move.
   Not traced to a single IR difference.

## GPU correctness and speed, per kernel

<!--GPU-->
| module | harness task | frontier (old vs new) | held-out (old vs new) | task check, new stack |
|---|---|---|---|---|
| group_bias_add_sm100 | pith_group_bias_add | 35/35 bitwise equal (2 runs) | 36/36 bitwise equal (2 runs) | 35/35 + 36/36 |
| causal_conv1d_sm100 | pith_causal_conv1d | 35/35 bitwise equal (2 runs) | 35/35 bitwise equal (2 runs) | 35/35 + 35/35 |
| gated_delta_rule_cp_sm100 | pith_gated_delta_rule_cp | 46/46 bitwise equal (2 runs) | 40/40 bitwise equal (2 runs) | 46/46 + 40/40 |
| gated_delta_rule_pinned_sm100 | pith_gated_delta_rule | 20/20 bitwise equal (2 runs) | 24/24 bitwise equal (1 run) | 20/20 + 24/24 |
| fp8_grouped_gemm_sm100 | pith_fp8_grouped_gemm | 66/66 bitwise equal (2 runs) | 66/66 bitwise equal (2 runs) | 66/66 + 66/66 |
| bf16_expert_gemm_sm100 | pith_bf16_expert_gemm | 78/78 bitwise equal (2 runs) | 76/76 bitwise equal (2 runs) | 78/78 + 76/76 |
| bf16_expert_slots_sm100 | pith_bf16_expert_slots | 74/74 bitwise equal (2 runs) | 68/68 bitwise equal (2 runs) | 74/74 + 68/68 |
| token_scatter_sm100 | pith_token_scatter | not bitwise comparable; task check 34/34 new, 34/34 old | not bitwise comparable; task check 37/37 new, 37/37 old | 34/34 + 37/37 |
| segment_sum_sm100 | pith_segment_sum | 30/30 bitwise equal (2 runs) | 31/31 bitwise equal (2 runs) | 30/30 + 31/31 |
| fp8_clamped_swiglu_quant_sm100 | pith_fp8_clamped_swiglu_quant | 79/79 bitwise equal (2 runs) | 82/82 bitwise equal (2 runs) | 79/79 + 82/82 |
| fp8_scatter_quant_sm100 | pith_fp8_scatter_quant | **not yet verified** | **not yet verified** | - |
| fp8_silu_mul_quant_sm100 | pith_fp8_silu_mul_quant | **not yet verified** | **not yet verified** | - |
| fp8_weight_quant_sm100 | pith_fp8_weight_quant | **not yet verified** | **not yet verified** | - |
| fp8_rowwise_blockwise_transpose_sm100 | pith_fp8_quant_act | **not yet verified** | **not yet verified** | - |
| fp8_rowwise_transpose_sm100 | pith_fp8_quant_grad | **not yet verified** | **not yet verified** | - |
| fp8_rowwise_colwise_sm100 | pith_fp8_quant_grouped | **not yet verified** | **not yet verified** | - |
| cross_entropy_sm100 | pith_cross_entropy | 25/25 bitwise equal (2 runs) | **not yet verified** | 25/25 |
| log_prob_sm100 | pith_log_prob | 128/128 bitwise equal (2 runs) | 116/116 bitwise equal (2 runs) | 128/128 + 116/116 |
| ep_dispatch_sm100 | pith_ep_dispatch | **not yet verified** | **not yet verified** | - |
| clamped_swiglu_sm100 | pith_clamped_swiglu | 64/64 bitwise equal (2 runs) | 66/66 bitwise equal (2 runs) | 64/64 + 66/66 |
| silu_mul_sm100 | pith_silu_mul | 92/92 bitwise equal (2 runs) | 92/92 bitwise equal (2 runs) | 92/92 + 92/92 |
<!--/GPU-->

## pith-train tests under the new stack

<!--TESTS-->
Run on a B200 (catalyst GPU 5, picked empty at launch and pinned by UUID; KCoral for some), the old
tree (bf9d16f, apache-tvm 0.27) and the new tree (tvm_tirx 0.1.0a6) side by side, same torch 2.14.
DeepGEMM does not build against torch 2.14 in this environment and FA4 is not installed, so a stub
`deep_gemm` that raises when called stands in (the `fp8_gemm_nt` / DeepGEMM-backed tests error
on both stacks) and `tests/kernels/test_flash_attn_memoize.py` is skipped.

| suite | old stack | new stack |
|---|---|---|
| `tests/kernels` (all features, both directions) | 1744 passed, 27 failed, 20 skipped (11.0 min) | 1744 passed, 27 failed, 20 skipped (24.7 min) |
| failures, identical on both | 21 `test_fp8_gemm` (DeepGEMM stub); `test_clamped_swiglu::test_matches_autograd[bf16]`; `test_cross_entropy::test_small_gradient_entries_relative[*-fp16]` (3); `test_log_prob::test_masked_logits` (2) | same 27 tests |
| operators + layer tests (`tests/operators/test_cp_sequence.py`, `tests/test_log_prob.py`, `tests/test_clamped_swiglu.py`, `tests/test_silu_mul.py`, `tests/test_indexed_bias_add.py`) | <!--OLDOPS--> | 90 passed, 13 failed, 14 skipped: `test_packed_positions` (3, need FA4) and `tests/test_clamped_swiglu.py::test_clamped_swiglu_forward_backward` (10) |

The 6 non-DeepGEMM `tests/kernels` failures fail on the old tree too (the first B200 run of these
suites): they are not from the port, and are left for the kernels' owners. The new stack's suite is
slower only because it compiles every kernel 2-7x slower on the first call.
<!--/TESTS-->

## Blockers and caveats

- TIRx-harness and KCoral still pin apache-tvm 0.27, so kernels evolved there are written against the old API.
- **GPU environment.** catalyst GPU 6 cannot create a CUDA context (`cuDevicePrimaryCtxRetain`
  returns CUDA_ERROR_UNKNOWN, `nvidia-smi -q` shows a pending item); only GPU 5 was used. Runs on
  which another user's process appeared have their timing discarded (correctness kept).
- **Shared venvs damaged during this work** (not by the port): `/tmp/haok/harness-venv` lost
  `torch/_utils.py` and parts of triton, `/tmp/haok/tirx-runs/cross_entropy/.venv` lost numpy files,
  and `/tmp/haok/uv-cache` lost archive files (e.g. apache-tvm-ffi 0.1.14.post0's METADATA). The
  checks here use fresh venvs (`/tmp/haok/tirx-old-venv`, `/tmp/haok/tirx-sa-venv`) and a private
  uv cache.
- **Upstream TIRx-kernels tests:** 4 of `tests/test_tirx_lite_api.py` fail at 716d8a3 on a6 and b0
  alike (`alloc_tensor` / `decl_tensor` with `elem_offset` or strides, which #13 removed from the
  library but not from the tests). No pith kernel uses those options.
- **tirx-kernels from git:** PyPI's newest tirx-kernels (0.1.2.post1) still has the apache-tvm
  API, so `pyproject.toml` takes mlc-ai/TIRx-kernels@716d8a3 as a git source until a release.
- **apache-tvm-tirx 0.1.0b0** has no wheel; a6 (pinned) and b0 emit byte-identical CUDA for every
  device group except 20 of fp8_clamped_swiglu_quant's 32, where b0 drops a branch fact a6 uses
  (`threadIdx.x % 32` vs `% 16` inside `if (threadIdx.x % 32 < 16)`: equal values).
- **token_scatter** cannot be compared bitwise (old vs old already differs); its task check passes
  on both stacks for every workload.
- **cross_entropy held-out:** the harness's own comparison runs out of memory on the held-out
  vocabularies (15 GiB allocation with 172 GiB in use) on the old stack as well; retried with one
  correctness run.
<!--CAVEATS-->
