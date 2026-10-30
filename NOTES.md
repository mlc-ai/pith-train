# pt-env notes
- Job 246136 (1 B200, boston-2-9). Use `srun --jobid=246136 -n 1 ...` (job has 8 tasks; without -n 1 commands run 8x).
- Tasks: 1 env, 2 pytest tests/kernels on B200, 3 SM100-STATUS.md (untracked).
- Scripts/logs in .pt-env/ (git-excluded via .git/info/exclude); login /tmp not visible on node.
- Task1 DONE: env = .venv->/scratch/haok/haok-pith-train/.venv, `CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH UV_CACHE_DIR=/scratch/haok/uv-cache uv sync` inside job (-c 24). Verified via .pt-env/verify.py.
- Task2: run per file with .pt-env/run_kernels.sh (logs .pt-env/klogs/). Findings so far:
  - test_ep_dispatch: test_dispatch_models etc FAIL (check_stable on SM100 atomic order? verify), and
    test_ep_dispatch_recv_idx[8-256] HANGS in Triton compile (static_range EP*EPR unroll in ep_dispatch_sm100.expert_and_expand_idx_kernel). Rerun file with -k deselect.
- Task3: bench script .pt-env/bench_sm100.py (T arg), writes .pt-env/bench_sm100_T<T>.json
- GPU steps: srun --gres errors 'Invalid generic resource' while GPU busy; use .pt-env/gpu_step.sh (retries).
- 03:37 job 246136 preempted during nogate scatter_quant; rerun nogate scatter_quant + bench_extra after RUNNING.
- ALL 3 TASKS DONE (04:0x). SM100-STATUS.md written (untracked). Verifier plugins in .pt-env/conftest_*.py
  (nogate, anyorder, clonemeta, sm100layout, noorder); bench .pt-env/bench_sm100.py + bench_extra.py.
- Deleted ~110 empty (corrupt, post-preemption) dirs in ~/.deep_gemm/cache via rmdir.

## Task B (2026-10-29 06:00-): deliverable branch haok/tirx-evolved (local commits only, no push, no Co-Authored-By)
- Branch from 74e8974. pyproject: apache-tvm==0.27.0.post1, apache-tvm-ffi==0.1.14.post0, tirx-kernels==0.1.2.post1,
  constraint kcoral==0.1.0. uv.lock is gitignored in this repo -> must `git add -f uv.lock`.
- Env for steps: .pt-env/env.sh (CUDA 13.3 nvcc, TVM_CUDA_COMPILE_MODE=nvcc); GPU wrapper .pt-env/gpu.sh;
  combined run .pt-env/run_all.sh (TAG, FILES, BENCH env) -> .pt-env/klogs/<TAG>/summary.txt.
- Landing a frontier solution: .pt-env/land_rename.py (strip _names) + land_docstrings.py, add __all__ + provenance.
- Stage 2 landed (uncommitted until r1 passes): silu (persistent-cpasync), scatter (cluster-sort-stream),
  clamped (tile128-segsum); fp8_fused_experts_supported True on SM100; new fp8_channels_as_rows (dense MLP wgrad
  operand, fixes fp8_swiglu_mlp on SM100); 4 gated verifiers layout-aware + inf/NaN cases (accept SM90-emulated or
  native E8M0 in inf blocks); test_bf16_expert_support/test_ep_dispatch use fp8_group_indices None as "no CuTe";
  fp8_weight_quant tests loop when *_multi is None; ep_dispatch_recv_idx_sm100 rewritten (prefix sum + bsearch).
- Bench: .pt-env/bench_landed.py (new vs unfused path vs old _sm100 from bench_sm100_T4096.json).
- Stage 3 queue: act, grad, grouped, weight (weight implements *_multi?), gdr when > FLA.
- 06:50 re-landed newer bests via .pt-env/land.py: silu persistent-cpasync@06a8f5d (3.815x), scatter
  cluster-sort-balanced@4f06168 (5.725x), clamped tma-ring-2group@1dfb931 (7.197x). Post-fixes: silu "int32"
  annotations -> txl.i32; dead locals removed (ruff F841) in clamped. run r1 queued (tests + check_fused_experts + bench).

## 08:40 status (commits on haok/tirx-evolved, not pushed)
- a0b9bbc deps; fa9f473 ep_dispatch recv rewrite + SM100 test gates; 4333b93 silu/clamped/scatter TIRx
  kernels + fused path on + verifiers; d64e853 gpt-oss fused down-bias fix (group_indices None on SM100,
  found by .pt-env/check_fused_experts.py); 294b5db silu re-landed at 1cde617 (4.185x).
- .pt-env/gpu.sh now uses the team's flock /tmp/haok-b200.lock queue (the old retry loop starved for an
  hour); the gpu-priority flag is BANNED (coordinator 2026-10-29), queue normally.
- check_fused_experts: all fused vs unfused bit-identical or within tolerance (all 4 models + dense MLP).
- bench_landed (pith-train microbench, new vs unfused vs old _sm100): logs .pt-env/klogs/bench_landed.log,
  bench_clamped.log.
- Next: stage 3 landing of act, grad, grouped, weight, gdr, fp8gemm, bf16gemm, slots frontiers.

## final-versions batch (coordinator: silu/grouped/grad final, will not change)
- silu persistent-cpasync 1cde617 (294b5db), grad tilecount-dispatch-hybrid 9c7c73d (60dfcd2; Triton
  fallback moved into fp8_rowwise_transpose_sm90 as fp8_quant_grad_triton), grouped regtile-v8-l2pf
  158986c (930ac5f; new fp8_rowwise_colwise_sm100, fp8_gemm case 10).
- B200: test_fp8_rowwise_transpose + test_fp8_gemm 399 passed 7 skipped; test_import_layers ok;
  check_fused_experts FAILURES none. bench: .pt-env/bench_quant.py (klogs/bench_quant.log): grad
  1.29-1.67x, grouped 1.72-1.78x vs Triton; total 1.669x.
- Remaining: act, weight, gdr (cudnn-segrec: check deps), fp8gemm, bf16gemm, slots once final.

## current-best batch (coordinator: land act, weight, fp8gemm, bf16gemm, slots, scatter, clamped, gdr)
- Commits: be0dd13 act e9efe2b; a0df2d6 scatter 8efa8c6 + clamped 6525b1e (race fix); d49c7e6 fp8gemm
  74e5452; cfce5e6 bf16gemm 3e76cbd + slots d51189b, bf16_experts_supported on SM100; 9cbbeba gdr 48a7601
  (3.23x), new picker gated_delta_rule_pinned.py, pyproject F821/F841 exemption for the generated file.
- weight HELD (audit: dropped use_torch_stream; evo-weight must restore it). Landing saved as
  .pt-env/weight_landing.patch + weight_landing_fp8_blockwise_transpose.py (custom_op for torch.compile,
  x.detach(), Triton moved to plain fp8_blockwise_transpose.py for SM90; tested 360/360 on B200).
  Note: dtype names in gptr annotations must be txl.f32/bf16/u8/i64/f8e4m3 (txl.float32 is a ctor).
- Audit (~/audit/AUDIT.md): every landed kernel is "pending race check" until its index.json entry
  records a racecheck+synccheck verdict and benchmarked-as-committed bytes. None has them yet (13:45).
- B200 verifiers: blockwise_transpose+scatter+clamped 156/156; fp8_gemm 171 + 7 skip; bf16_expert_gemm 74,
  slots 47, support 6 + 4 skip; gdr 9 + 9 skip (FLA bit-exact gated); import_layers 2; check_fused none;
  check_stream all on current stream; check_bf16_experts: grads differ ~1e-5 calc_diff (bf16 rounding).
- scatter frontier now lists 7.423x for cluster-sort-balanced (newer than 8efa8c6): not landed.

## GPU queue rules (coordinator, 2026-10-29)
- Timeouts go INSIDE the step (`gpu.sh timeout 600 <cmd>`), never around gpu.sh: an outer one counts queue wait. run_timeboxed.sh already does this (its timeout is inside the srun'd bash -c).
- compute-sanitizer: one or two small representative shapes (M ~ few hundred to 1k), --kernel-name filters, racecheck + synccheck batched in one step, < 2-3 min.
- Sanitization moved to the very end (coordinator): keep landing newest frontiers marked "pending race check"; don't queue sanitizer runs.
- 0c: re-landed scatter cluster-sort-balanced@7c77fc9 (7.423x) + clamped tma-ring-2group@0647acc (7.515x): verifiers 90/90, check_fused_experts none failed, ruff + test_import_layers clean. Weight still held (frontier tile-split@87cdf58 has no use_torch_stream).
- Queue rule (coordinator): at most ONE GPU step of mine queued/running at a time; batch work into it; CPU work (ruff, compiles) via ssh boston-2-9 / cpu steps meanwhile.
- Landed weight tile-npamax@1864d1d (1.654x) as 4842690: picker part of weight_landing.patch + fp8_blockwise_transpose.py; compile_v wraps launches in tvm_ffi.use_torch_stream() (frontier lacks it); gptr string dtypes -> txl short dtypes. test_fp8_weight_quant 362/362, check_stream all ok, ruff/import_layers clean. Pending race check.
- Scatter tstart: frontier 7c77fc9 writes tstart[g] only for g < num_groups but pass B reads all gb; landed file writes every entry (line 262) — confirmed fixed.

## Full frontier landing from ~/tirx-frontiers (2026-10-29 evening)
- land.py now takes TIRX_RUNS=/home/haok/tirx-frontiers (commit from frontiers/HEADS.txt, provenance shows ours_us vs sota_us); fixup.py converts string dtypes.
- Picked the lowest official ours_us per run. Commits: 95d98e7 776ca6a 8fdbb83 82f9141 fe130dd acbb3de 87c8886 333150b e082b23 5108e69.
- grouped regtile-v8-l2pf unchanged (byte-identical to 930ac5f's).
- USER RULE: no ssh to the nodes; all compute (ruff, pytest, builds) only via srun --jobid=<new 4xB200 job> --overlap. If the job is PENDING: stop, tell the user, wait.
- NOT yet linted (ruff unavailable on login): weight acbb3de, fp8gemm 87c8886, bf16gemm 333150b, slots e082b23, gdr 5108e69. First thing in the job: ruff check + test_import_layers, fix lint, then tests.
- b723719: m_grouped_fp8_gemm_nt docstring. The reviewer's claim was wrong: on SM100, fp8_grouped_gemm case 10 already calls fp8_grouped_gemm_sm100 (m-grouped fwd and dgrad); m_grouped_fp8_gemm_nt (DeepGEMM) is reached only from case 9.
- Triton kernels still on the SM100 path: ep_dispatch (EP>1), cross_entropy (always), causal_conv1d (qwen3.5), silu_mul (deepseek dense MLP incl. inside fp8_swiglu_mlp, qwen3.5 shared expert), group_bias_add + segment_sum (gpt-oss FP8 fused bias). Not reached on SM100 when the fused paths apply: fp8_rowwise_kmajor_sm100, token_scatter scatter_by_expert, clamped_swiglu.

## Finish line (user, 2026-10-29): ZERO Triton in pithtrain/
No `import triton` / @triton.jit anywhere in pithtrain/, including fp8_scale.py and operators/log_prob.py.
Delete each Triton file only once BOTH its SM90 replacement (merge from branch haok/kda-sm90, built on ifm)
and its SM100 TIRx replacement (tasks being written on catalyst) have landed and pass tests.
Triton today (file: #@triton.jit), and what replaces it:
- causal_conv1d.py 4, clamped_swiglu.py 2, cross_entropy.py 1, group_bias_add.py 1, segment_sum.py 5,
  silu_mul.py 2, token_scatter.py 4 (plain, both archs): need SM90 + SM100 kernels -> become pickers + _sm90/_sm100.
- ep_dispatch_sm100.py 5 (SM100 side of ep_dispatch): TIRx winner.
- fp8_rowwise_colwise.py 2 (shared Triton kernel launched by fp8_rowwise_kmajor_sm100, SM90's fallback only).
- fp8_rowwise_transpose_sm90.py 1 (SM90's Triton fp8_quant_grad fallback), fp8_blockwise_transpose.py 2 (SM90 weight transpose
  under torch.compile): SM90 replacements from kda-sm90.
- fp8_scale.py 1 (Triton E8M0 helper; goes when its last Triton user goes).
- gated_delta_rule_pinned_sm90.py: launches FLA's Triton kernels with pinned configs (imports triton, no @jit) —
  needs an SM90 non-Triton GDN (kda-sm90?) to get to zero; FLA itself stays a dependency unless replaced too. Raise with user.
- operators/log_prob.py 2: needs a kernel in kernels/ (SM90 + SM100) and the operator importing it.
Order: finish current landing + GPU tests (job 246928) -> route SM100 to TIRx winners -> merge kda-sm90 -> delete Triton files, tests each step.
- Job 246928 tests (boston-3-9): tests/kernels 1698 passed, 20 expected skips, 16 scale-layout failures -> fixed in 7dd03b6 (act/grad SM100 scales column-major, act_scale_like matches); rerun: scale_layout+act 104, grad 228, fp8_gemm 171/7 skip, linear tests 24, fused experts none failed, ruff clean.
- 2026-10-29 late: no FLA exception (SM90 GDN gets CuTe incl. CP path from kda-sm90). Job 246928 = baselines, DON'T USE.
  Prepared SM100 routing for the TIRx Triton replacements: plan + per-feature table in .pt-env/ROUTING.md;
  new tool .pt-env/split_feature.py (plain -> picker + _sm90), dry-run checked on all 7 plain features.
  No new frontiers yet (checked ~/tirx-frontiers and ~/TIRx-harness/kda_flow_runs).
- 2026-10-30: bf16_expert_gemm_sm100 pair2 bug (one TMA descriptor over w_gate..w_up via address diff as stride; ~5.7 TB apart in training -> globalStrides >= 2^40 crash). Waiting for bf16gemm evolver's fixed frontier in ~/tirx-frontiers (watcher on index.json/solution.py/HEADS.txt); then land with land.py (exports in NOTES), fixups (I->inter, kt2, ld2/cur dead locals), commit pending race check. Job 246928 = baselines, don't use.
- 2026-10-30: landed bf16gemm far-pair fix c2b333c as 3601f9f; 121 bf16 tests passed on 1 B200 (246928, user-approved short step). land.py now tolerates extra words in HEADS lines.
- 2026-10-30: baselines (haok-5b) moved its feat worktree to 3601f9f; PR #133 mesh study BF16 feat arms in 246928 exercise the far-pair path; they report crashes back.
- 2026-10-30: job 246928 cancelled (Slurm hold). New Boston job 247101 is baselines'; don't use it unless baselines hands it over. No GPU job of mine.
- 2026-10-30: landed clamped tma-ring-2group (0830a33) as 544e52b (sched counter keyed per stream), segment_sum tma2d-ring (e356c72) as 8f91336 (split: picker + segment_sum_sm90 + _sm100; SM90_CROSS_EDGES entry). 80f1af3 restores string dtype tables in act/grad/weight _sm100 that fixup.py had broken (regex fixed: (?<![.\w])). ruff + import_layers clean (CPU step in 247101). GPU tests pending: .pt-env/job_pending.sh on 1 B200 in 247101 once baselines hands a GPU over (asked haok-5b).
- 2026-10-30: baselines will pause its runner in ~45 min and message me; nothing in 247101 (not even CPU steps) until then; tell haok-5b when job_pending.sh is done so they resume.
- 2026-10-30: landed group_bias_add flat-vec-l2tail (0a55a8c) as 4fcd671 (split + PyTorch fallback guard in _sm100). Unlinted yet: job_pending.sh now runs ruff/import_layers first + test_group_bias_add.
- 2026-10-30: GPU window (247101, 1 B200): lint ok after dropping unused tvm import (amended into f63baf5); import_layers 2/2; 781 kernel+operator tests passed; check_stream ok; check_fused FAILURES none (rerun with DG_JIT_CACHE_DIR=/scratch/haok/dg-cache-ptenv: default DeepGEMM cache gave 'Corrupted JIT cache directory'). Released window to haok-5b.
- 2026-10-30: env.sh now sets DG_JIT_CACHE_DIR=/scratch/haok/dg-cache-ptenv (~/.deep_gemm corrupted). Baselines' mesh study finished; they wait on the user re repeats.
- 2026-10-30: far-pair fix 3601f9f confirmed in training by baselines: qwen3-30b-a3b / deepseek-v2-lite / gpt-oss-20b BF16 feat at PP2/EP2/DP2 and PP1/EP4/DP4 (seq 4096) 12/12 steps, losses match main to 4 dp; qwen3 (far-pair case) 1.58x / 1.31x. qwen3.5-35b-a3b cells OOM at step 2 on main too (capacity). Logs ~/haok-pith-train-main/logging/pr133-meshes/.
- 2026-10-30: coordinator task: map baselines' deepseek PP2/EP2 FP8 slow kernels (feat 0.67x) to our frontiers/routing + check whether shapes are outside each run's workload/held-out. Prep: Boston TIRx-harness tasks (f357847, 2026-10-29, may lag catalyst) workload table in scratchpad workloads.txt. pith_fp8_grouped_gemm has NO deepseek-v2-lite workload (qwen3.5 N1024/K2048..., gpt-oss 2880/5760); deepseek EP2: 32 local experts, gate_up N=2816 K=2048, down N=2048 K=1408. Held-out sets not on Boston. Asked haok-5b for kernel table + nsys path.
- 2026-10-30: silu persistent-cpasync (23cc854) landed as e7efc0c, untested (job 247101 preempted/PENDING); job_pending.sh now = silu lint+verifiers+fused/stream checks.
- 2026-10-30 deepseek PP2/EP2 FP8 profile (baselines: 0.67x was host noise; repeats 1.14x median; feat non-NCCL 456->393 ms/step/rank). Ours: kgrouped 41.2 ms (vs DeepGEMM 51.4, 1.25x), mgrouped 37.0 (vs 42.3, 1.14x) - fp8gemm workload has NO deepseek shapes (outside). kern_kernel 16.1 = fp8_silu_mul_quant_sm100 (routed N=1408 in workload; shared-expert bwd N=2816 single group via fp8_swiglu_mlp NOT in workload). quant_kernel 3.3 = fp8_rowwise_colwise_sm100 (vs Triton 25.2). fused_kernel 3.8 = fp8_scatter_quant (deepseek ep2 in workload). Remaining Triton: silu_mul_fwd (shared-expert fwd), dedup_* (ep_dispatch). DeepGEMM UE8M0 packing ~12 ms/step (main ~18).
- 2026-10-30: landed gdr superseg-scan fc3183a as a6e1a06 (prof_* dropped, names *_pinned_sm100; 3 CP racecheck findings still open), token_scatter coop-grouped-tma-ring 0386099 as fe96a47 (split; state keyed per stream; cooperative launch noted). c09: .pt-env/launch_bench.py (host_us/gpu_us per public entry, main tree vs ours + noop floors), .pt-env/job_c09.sh (GPU0 tests gdr/token_scatter/silu e7efc0c, GPU1 bench). Background waiter blofmj9xz runs it in job 247224 (2 GPUs, -c16) once RUNNING; then tell coordinator 'GPU done' (with baselines). Slurm: TaskPlugin task/cgroup,task/affinity; 247224 has 32 CPUs.
- 2026-10-30: baselines PP1/EP4 FP8 profiles: no slower TIRx kernel; gpt-oss compute 1136->785 ms, deepseek 833->713; NCCL waits grow and are uneven across ranks (host-side). Launches/step/rank fewer on feat (gpt-oss 46.2k->39.0k, deepseek 61.0k->52.2k). Baselines done with 247224.
- 2026-10-30 job 247224 (2 GPUs, ~10 min): 179 passed / 10 expected skips (gdr pinned, token_scatter, scatter_quant, silu quant, silu_mul, ep_dedup, gated_deltanet_packed); fused + stream checks ok; lint ok after E741 rename (separate commit). c09 bench results in .pt-env/klogs/c09/bench_{main,feat}.json: tvm-ffi bare call 6.5us ~ Triton 6.8; use_torch_stream adds ~5us/launch; single-launch quantizers +11-13us/call vs main; fused ops much less host time; net per step deepseek PP1/EP4 ~ -80 ms host on feat. Job CPUs of a 2-CPU step: 4,116 (NUMA 0).
- 2026-10-30: c09 fix committed 756e55b (launch_fixup.py: drop use_torch_stream x32, builders return entry_function; weight eager skips custom op). Needs GPU: .pt-env/job_c09b.sh (lint, bench main/ours x2, full tests/kernels on 2 GPUs, fused/stream(15 entries)/bf16 checks, operator tests). Job 247224: baselines reruns running.
- 2026-10-30: baselines gives 247224 gap in ~30-35 min (msg 'GPU free'); send them 'bench done' after bench phase (they then run a 2-CPU sqlite step), 'done' at end.
- 2026-10-30: baselines 'before' host idle (nsys, PP1/EP4 FP8, pinned NUMA1): gpt-oss main 974 vs feat 1196 ms idle/step (+222), deepseek 1546 vs 1566. Step CPU set mixes NUMA0+1 (4-7,12-15,116-119,124-127 + 64-71,176-183); PR133 bind_cpu_affinity no-op. 'After' = their rep.sh nsyspin once 756e55b is in their feat worktree (theirs to move).
- 2026-10-30: c09 fix 756e55b verified in 247224: lint ok; tests/kernels 1714 passed/20 expected skips; check_fused none; check_stream 15/15 side stream; check_bf16 max 1.44e-5 (same as before); operators 42 passed. Bench (same window, host us vs main): act -1.2, grad +1.7, weight -2.6, weight_grouped -5.7, grouped -8.7, group_bias_add -0.6; fused much lower. Told baselines done.
- 2026-10-30: c09 'after' (baselines, pinned PP1/EP4 FP8, 756e55b): gpt-oss idle vs main +222 -> -61 ms/step, 12-step 1.07x -> 1.25x (2.201 vs 1.756 s); deepseek +20 -> -131 ms, 1.02x -> 1.19x (2.338 vs 1.957 s).

## STATE AT SHUTDOWN (2026-10-30)
- Running: nothing. No srun steps of mine, no background waiters. Jobs: 247224 (shared with baselines, idle when I stopped); 247101/246928/246136 not mine to use.
- Branch haok/tirx-evolved (local only, never pushed), HEAD = c09 fix 756e55b plus this NOTES commit. All landings "pending race check" (sanitizer pass deferred to the very end). gdr a6e1a06: 3 known CP racecheck findings (mn x2, prefill x1) still open.
- Landed + GPU-tested this session: bf16gemm far-pair 3601f9f, clamped 544e52b, segment_sum route 8f91336, group_bias_add route f63baf5, silu e7efc0c, gdr a6e1a06 (+E741 66d0372), token_scatter route fe96a47, dtype-table repair 80f1af3, c09 host-cost fix 756e55b (verified: 1714 kernel tests pass, stream check 15/15, bench within 2 us of main; baselines e2e: gpt-oss 1.25x, deepseek 1.19x pinned PP1/EP4 FP8).
- Half-done: nothing mid-edit. Untracked helpers in .pt-env/ (land.py, fixup.py, split_feature.py, launch_fixup.py, launch_bench.py, job_*.sh, ROUTING.md, ref/ = copied tvm_ffi/tirx/tvm sources for reading).
- Next steps: (1) land new frontiers from ~/tirx-frontiers as they appear (land.py -> fixup.py -> launch_fixup.py -> lint -> commit "pending race check"; plain features via split_feature.py + SM90_CROSS_EDGES entry), remaining Triton replacements per .pt-env/ROUTING.md: silu_mul, clamped_swiglu, causal_conv1d, cross_entropy, ep_dispatch, log_prob, gated_delta_rule_cp; (2) merge haok/kda-sm90 SM90 replacements, then delete Triton files once both sides pass; (3) GPU work only as srun --overlap steps in a job the coordinator gives, coordinated with baselines; (4) suggestions not done: fp8gemm/silu workloads lack deepseek shapes (EP2/4/8, shared-expert gu_bwd N=2816); act/grad quantizers could emit UE8M0 directly to skip DeepGEMM repacking (~12 ms/step).
