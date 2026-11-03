# SM90 context-parallel gated delta rule: model-level loss envelope (cleanup task 1)

R3 kept SM90 context parallelism on FLA because the CuTe CP route measured a step-0 loss max abs diff of
0.00895 at CP_SIZE=2 FP8=1, over a cap of 0.0080 that had been set from non-CP runs. Base (FLA CP) itself
measured 0.00906 at that setting, so the cap did not describe CP2 FP8. This calibration replaced the cap with a
CP-specific envelope from base runs, registered before any CuTe run. FLA has since been removed from pith-train.

## Setup

- Test: tests/test_dualpipev.py with the qwen3.5-35b-a3b fixture (8 layers, 32 experts), pp 2, ep 2, cp 2,
  mbs 3, 20 chunks, 8 H200 GPUs on one node.
- Harness: a wrapper (workspace/kda-sm90/cpenv/measure.py) with three changes to the test. SEED replaces both
  manual_seed(1234) calls. The two loss checks are recorded instead of raised. Every parameter's gradient
  diff/eps is printed, and the gradient check itself still asserts. On the CuTe arm it also counts
  cute_cp_fwd / cute_cp_bwd calls per rank; a run that shows no calls is invalid.
- Arms:
  - base: haok/kda-sm90 93ad34c (FLA's CP path);
  - cute: kda/sm90-cutecp f0080a2, which is 93ad34c plus kda/sm90-gdr e093895's CuTe CP route.
  - The route committed here is that code.
- Metrics, per run:
  - P: max over pp-rank-0 ranks and both steps of |pipelined outputs - objective outputs|, R3's AC-10 metric
    (P0 is the step-0 value);
  - R: max |CP-averaged loss - single-device reference|;
  - G: max over ranks and parameters of grad diff / the test's eps.
- Calibration: base, FP8 = 0 and 1, seeds 1234 / 1 / 2 / 3, each seed on two different nodes (16 runs).
- Threshold per setting and metric: T = max(base max, base mean + 3 sd).
- Gate, on the same 16 cute specs: PASS iff
  1. every cute run is <= T;
  2. the cute median P is <= the base max P;
  3. FP8=0 cute runs exit 0 (FP8=1 too, unless a base FP8=1 run failed);
  4. the operator CP tests pass.
  - No reruns.

## Envelope (from base; written before any cute run)

| setting | T_P | T_P0 | T_R | T_G | base max P |
|---|---|---|---|---|---|
| FP8=0 | 0.00211629 | 0.00208908 | 0.000975671 | 0.617852 | 0.00184631 |
| FP8=1 | 0.0129091 | 0.0117965 | 0.00628728 | 1.1337 | 0.0112457 |

base FP8=1 seed 3 exits 1 on both of its nodes: the test's own gradient check fails on
1.layers.6.linear_attn.norm.weight, 0.106 against eps 0.1. So item 3 requires exit 0 only at FP8=0.

## Result: PASS

- Item 1: every cute run is inside T.
  - FP8=0 cute max: P 0.0018158, P0 0.0018158, R 0.00095367 (97.7% of T_R), G 0.6041 (97.8% of T_G).
  - FP8=1 cute max: P 0.0111084, P0 0.0110016, R 0.00507355, G 1.0419.
- Item 2: median P is 0.00154877 for FP8=0 and 0.00891113 for FP8=1, both <= base max P.
- Item 3: all 8 FP8=0 cute runs exit 0. Cute FP8=1 seed 3 exits 1 on the same parameter as base (G 1.042 and
  1.031, against base's 1.034 and 1.063).
- Item 4: tests/operators/test_gated_delta_rule_cp.py plus test_cp_sequence.py on f0080a2, 4 GPUs: 19 passed,
  2 skipped (the skipped cases need 8 GPUs).
- Means, cute against base:
  - FP8=1 P 0.00939 against 0.00941, R 0.00358 against 0.00368;
  - FP8=0 P 0.00157 against 0.00162, R 0.000774 against 0.000772.
- Excluded from the count:
  - the first cute FP8=1 seed-3 run on fs-mbz-gpu-793, which printed no call counts because the counter ran in
    atexit, which a raising rank skips; it is invalid per the registration, and its metrics were inside T anyway;
  - two third copies of a spec; only the first two to finish count.

Known limit, carried from R3: this model-level check has no power against a missing backward merge at this
shape, since a negative control with the merge disabled passed. The operator-level CP tests carry that check.

## Runs (parse.py; hosts fs-mbz-gpu-*)

### base
```
base_fp80 | runs=8 | P max=0.00184631 mean=0.00162411 sd=0.000164 T=0.00211629 | P0 max=0.00176239 mean=0.00150204 sd=0.000196 T=0.00208908 | R max=0.00087738 mean=0.000771521 sd=6.8e-05 T=0.000975671 | G max=0.602 mean=0.58875 sd=0.0097 T=0.617852
base_fp80_s1 fs-mbz-gpu-571 P=0.00153351 P0=0.00141144 R=0.00073242 G=0.5874 (1.layers.7.mlp.experts.gate_up_proj.weight) n=(8, 8, 532) cute=['False']
base_fp80_s1 fs-mbz-gpu-935 P=0.00169373 P0=0.00131226 R=0.00078583 G=0.5875 (1.layers.7.mlp.experts.gate_up_proj.weight) n=(8, 8, 532) cute=['False']
base_fp80_s1234 fs-mbz-gpu-571 P=0.00151062 P0=0.00137329 R=0.00071716 G=0.6020 (1.layers.7.mlp.experts.gate_up_proj.weight) n=(8, 8, 532) cute=['False']
base_fp80_s1234 fs-mbz-gpu-600 P=0.00132751 P0=0.00125122 R=0.00076294 G=0.6020 (1.layers.7.mlp.experts.gate_up_proj.weight) n=(8, 8, 532) cute=['False']
base_fp80_s2 fs-mbz-gpu-138 P=0.00184631 P0=0.00169373 R=0.00087738 G=0.5898 (1.layers.7.mlp.experts.down_proj.weight) n=(8, 8, 532) cute=['False']
base_fp80_s2 fs-mbz-gpu-600 P=0.00176239 P0=0.00176239 R=0.00086975 G=0.5884 (1.layers.7.mlp.experts.down_proj.weight) n=(8, 8, 532) cute=['False']
base_fp80_s3 fs-mbz-gpu-047 P=0.00161743 P0=0.00151062 R=0.00070953 G=0.5769 (1.layers.7.mlp.experts.gate_up_proj.weight) n=(8, 8, 532) cute=['False']
base_fp80_s3 fs-mbz-gpu-935 P=0.00170135 P0=0.00170135 R=0.00071716 G=0.5760 (1.layers.7.mlp.experts.gate_up_proj.weight) n=(8, 8, 532) cute=['False']
base_fp81 | runs=8 | P max=0.0112457 mean=0.00941086 sd=0.00117 T=0.0129091 | P0 max=0.0102005 mean=0.00893116 sd=0.000955 T=0.0117965 | R max=0.00513458 mean=0.00368309 sd=0.000868 T=0.00628728 | G max=1.0625 mean=0.973764 sd=0.0533 T=1.1337
base_fp81_s1 fs-mbz-gpu-047 P=0.01124573 P0=0.01020050 R=0.00324249 G=0.9402 (1.layers.6.linear_attn.in_proj_a.weight) n=(8, 8, 532) cute=['False']
base_fp81_s1 fs-mbz-gpu-935 P=0.01098633 P0=0.00964355 R=0.00335693 G=0.9372 (1.layers.6.linear_attn.in_proj_a.weight) n=(8, 8, 532) cute=['False']
base_fp81_s1234 fs-mbz-gpu-287 P=0.00921631 P0=0.00921631 R=0.00496674 G=0.9878 (1.layers.6.linear_attn.in_proj_a.weight) n=(8, 8, 532) cute=['False']
base_fp81_s1234 fs-mbz-gpu-600 P=0.00823212 P0=0.00823212 R=0.00513458 G=0.9858 (1.layers.6.linear_attn.in_proj_a.weight) n=(8, 8, 532) cute=['False']
base_fp81_s2 fs-mbz-gpu-571 P=0.00870514 P0=0.00870514 R=0.00292969 G=0.9234 (1.layers.6.linear_attn.in_proj_a.weight) n=(8, 8, 532) cute=['False']
base_fp81_s2 fs-mbz-gpu-600 P=0.00981903 P0=0.00981903 R=0.00346375 G=0.9188 (1.layers.6.linear_attn.in_proj_a.weight) n=(8, 8, 532) cute=['False']
base_fp81_s3 fs-mbz-gpu-571 P=0.00872040 P0=0.00824738 R=0.00295258 G=1.0343 (1.layers.6.linear_attn.norm.weight) n=(8, 8, 532) cute=['False']
base_fp81_s3 fs-mbz-gpu-935 P=0.00836182 P0=0.00738525 R=0.00341797 G=1.0625 (1.layers.6.linear_attn.norm.weight) n=(8, 8, 532) cute=['False']
```
### cute (counted)
```
cute_fp80_s1     gpu-793      rc=0 P=0.00141907 P0=0.00136566 R=0.00081635 G=0.5842 OK
cute_fp80_s1     gpu-912      rc=0 P=0.00152588 P0=0.00152588 R=0.0006485 G=0.5889 OK
cute_fp80_s1234  gpu-314      rc=0 P=0.00149536 P0=0.00149536 R=0.00086975 G=0.6041 OK
cute_fp80_s1234  gpu-634      rc=0 P=0.00139618 P0=0.00112915 R=0.00067902 G=0.6032 OK
cute_fp80_s2     gpu-252      rc=0 P=0.0018158 P0=0.0018158 R=0.00095367 G=0.5869 OK
cute_fp80_s2     gpu-634      rc=0 P=0.00165558 P0=0.00151825 R=0.00079346 G=0.5906 OK
cute_fp80_s3     gpu-252      rc=0 P=0.00157166 P0=0.00157166 R=0.00076294 G=0.5812 OK
cute_fp80_s3     gpu-793      rc=0 P=0.00165558 P0=0.00165558 R=0.00067139 G=0.5772 OK
cute_fp81_s1     gpu-793      rc=0 P=0.0111084 P0=0.0103149 R=0.0029068 G=0.938 OK
cute_fp81_s1     gpu-936      rc=0 P=0.0110626 P0=0.0105057 R=0.00302124 G=0.9404 OK
cute_fp81_s1234  gpu-138      rc=0 P=0.00914764 P0=0.00914764 R=0.00507355 G=0.9842 OK
cute_fp81_s1234  gpu-166      rc=0 P=0.00867462 P0=0.00867462 R=0.00497437 G=0.9815 OK
cute_fp81_s2     gpu-314      rc=0 P=0.0079422 P0=0.00785065 R=0.0032196 G=0.919 OK
cute_fp81_s2     gpu-868      rc=0 P=0.0110016 P0=0.0110016 R=0.00293732 G=0.9223 OK
cute_fp81_s3     gpu-793      rc=1 P=0.00811768 P0=0.00769806 R=0.00339508 G=1.0419 OK
cute_fp81_s3     gpu-912      rc=1 P=0.00805664 P0=0.00775909 R=0.00311279 G=1.031 OK
```
