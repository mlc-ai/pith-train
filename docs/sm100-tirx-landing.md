# SM100 TIRx kernels: landing status (draft, 2026-11-01)

Branch `haok/tirx-evolved-sm100`, based on `boston/haok-tirx-evolved` (162a06c). Every one of the 21
finished TIRx-harness SM100 runs has its current best frontier member in `pithtrain/kernels/`. All
times are each run's official KCoral numbers (sum over the task's timed rows, ours vs the previous
SM100 path: pith-train's Triton / DeepGEMM / FLA).

**Draft: B200 end-to-end testing is blocked (Boston is busy).** Nothing on this branch has run on a
GPU in pith-train: no `tests/kernels` on a B200, no training run. The kernels themselves were
verified through KCoral against their harness tasks. The CPU checks pass: ruff check, ruff format,
`tests/test_import_layers.py`, an import of every touched module (with the arch probe stubbed out),
and compiling the touched TIRx builders with nvcc on the CPU.

## Per run

| run | member (run commit) | ours vs baseline (us) | commit | status |
|---|---|---|---|---|
| act | selfprod-persistent (1ef3160) | 226.1 vs 289.4, 1.28x | (earlier, be0dd13 / 95d98e7) | already current |
| grad | oneshot-regload-tile (f103d83) | 217.9 vs 379.1, 1.74x | (earlier, 776ca6a) | already current |
| grouped | regtile-v8-l2pf (158986c) | 780.5 vs 1397.0, 1.79x | (earlier, 930ac5f) | already current |
| weight | tile-pdl (24e4366) | 4102 vs 5452, 1.33x | (earlier, 4842690 / acbb3de) | already current |
| silu | persistent-cpasync (23cc854) | 2126.9 vs 8709.9, 4.10x | (earlier, e7efc0c) | already current |
| gdr | superseg-scan (346c9e3) | 936.0 vs 4421.7, 4.72x | 4495973 | updated |
| ep_dispatch | single-launch-start-gated (1be128e) | 788.6 vs 2384.8, 3.02x | dd7d334 | new SM100 path |
| clamped | tma-ring-fused-dbias (256df46) | 1669.9 vs 14016.2, 8.39x | 6121bc8 | updated (new member) |
| scatter | coop-route-colorder (81a96a7) | 992 vs 6252, 6.30x | 4861adc | updated, see gaps |
| group_bias_add | rowtile-packed-direct-retire (4a1608d) | 587.1 vs 671.3, 1.14x | 91904d3 | updated (new member) |
| segment_sum | compact-swizzle-tagreduce (3bc9af8) | 181.9 vs 640.2, 3.52x | 1ee0280 | updated (new member) |
| token_scatter | host-plan-tma-ring (00207e0) | 179.5 vs 561.6, 3.13x | 51b953d | updated (new member) |
| slots | tma-gather (6e1b68f) | 972 vs 10401, 10.70x | 86a35c3 | updated |
| bf16gemm | cluster2-cpasync-tmareduce (59570dc) | 15702 vs 267991, 17.07x | a7d74c4 | updated |
| fp8gemm | pair2cta-hybrid (db8dbbc) | 8021 vs 11612, 1.45x | 1b2ddd0 | updated |
| clamped_swiglu | flat-f32x2 (e4d9461) | 336.6 vs 660.6, 1.96x | 0b89bb1 | new SM100 path |
| silu_mul | packed-f32x2-stream (6ce8ceb) | 487.8 vs 493.1, 1.01x | 4ff4e4a | new SM100 path |
| cross_entropy | cluster-resident-tma (fce2df2) | 10880 vs 19448, 1.79x | da69e8a | new SM100 path |
| causal_conv1d | tma-ring-cluster-dw (f48a3ab) | 305.2 vs 793.6, 2.60x | d8b6cd8 | new SM100 path |
| log_prob | hardware-redux-max (c12d48b) | 25669 vs 37896, 1.48x | 4c8530e | new SM100 path; kernels moved out of operators/ |
| gated_delta_rule_cp | tma-intra-bulk-state-layout (fe5e80c) | 4551 vs 12948 (FLA CP), 2.84x | e683c32 | new SM100 path |

For a new SM100 path, the plain Triton file became a picker plus `<feature>_sm90.py` (the Triton
code, moved verbatim) and `<feature>_sm100.py` (the TIRx member). Every input the Triton version
accepted still works: inputs a member does not cover take its generic route or a PyTorch path with
the same results.

## Triton on SM100

- No SM100 branch of a picker, and no `_sm100` file, runs a pith-train Triton kernel any more. The
  Triton code is kept for SM90 in `*_sm90.py`, `fp8_blockwise_transpose.py`,
  `fp8_rowwise_colwise.py` and `fp8_scale.py`.
- Exception: `fp8_rowwise_kmajor_sm100` is still Triton (it goes through `fp8_rowwise_colwise.py`).
  It is reachable only through the `fp8cast_rowwise_kmajor` picker, whose one caller
  (`fp8_quant_grouped`) calls it in the SM90 branch, so SM100 never runs it.
- FLA's Triton kernels are still reached on SM100 in three cases:
  - the dense (no-CP) Gated DeltaNet, for inputs `gated_delta_rule_pinned_supported` rejects;
  - the CP Gated DeltaNet with batch > 1 (the batch folds into heads, so the head count is not 32);
  - other head counts, head dims or dtypes.

  `gated_delta_rule_cp_supported` admits only Qwen3.5's geometry: bf16 `(1, S, 32, 128)`, fp32 g,
  bf16 beta.
- Not pith-train Triton, but worth knowing: the `@torch.compile` regions in the models
  (deepseek_v2, gpt_oss, qwen35_moe, ring_attention) let Inductor generate Triton kernels on SM100.

## Gaps to resolve before this leaves draft

- **B200 run of `tests/kernels` and `tests/operators`.** Never done for this branch. New or changed
  verifiers: `test_log_prob.py`, `test_gated_delta_rule_cp.py`, the narrowed
  `test_fp8_scatter_quant.py`, and two H=32 cases in `test_cp_sequence.py` whose `atol=1e-2` is a
  guess, not a measurement.
- **scatter is no longer deterministic on SM100.** Since run commit b4304e3 the member orders each
  CTA's rows within a group with an atomic add. `reverse_shuffle_idxs` and the per-channel FP8 can
  change from call to call, so the weight-gradient input is not bit-reproducible. The rowwise FP8,
  `grouped_mm_offs` and `ks` are unchanged. The verifier's stable-order and repeat checks now run
  only on the SM90 layout. If SM100 must stay deterministic, revert 4861adc to keep the 3925531
  landing (1178 us).
- **token_scatter, gpt-oss rows.** gpt-oss's 5760-byte rows (H=2880) with host counts take a ring
  setup no KCoral row exercised; add an H=2880 host-count case to `test_token_scatter.py`.
- **segment_sum and CUDA graphs.** Its scratch is valid only alongside a per-call tag passed in from
  the host, so the kernel is not safe under CUDA-graph replay. pith-train captures no graphs today.
- **bf16gemm wgrads skip the usual wait for the previous kernel (PDL).** They are safe unless a
  kernel from another module that releases its dependents early produces their inputs right before
  them. None does today.
- **log_prob.** It runs as `torch.library` custom ops inside `torch.compile` on SM100, and its
  row-max / log-sum statistics are base-2 there (internal to the architecture's own backward).
  `test_log_prob_under_torch_compile` has not run on a B200.
- **Race and sanitizer status, per each run's index.json.**
  - gdr: the previous landing's 3 racecheck findings are reported fixed, but the saved mn / prefill
    racecheck JSONs predate the fix.
  - segment_sum: the race check is incomplete on the ring wrap-around case (the run attributes this
    to a checker limitation).
  - compute-sanitizer deferred: slots, bf16gemm, silu_mul, group_bias_add.
  - no compute-sanitizer verdict: clamped's in-launch d_bias.
  - cross_entropy: compute-sanitizer last ran on the previous version.
- **The scatter rowwise scale is now MN-major.** `fp8_grouped_gemm` copies it to row-major on
  SM100, a small copy (clamped already pays the same).
