<!-- Copied from kda/sm90-gdr 3cd44b7 kda-task/runs/r3/final/prec_summary.md (outputs there); merged as c75a394. -->

# Precision of the CuTe CP kernels as merged into the library (not routed)

Reviewer wrap-up item (c), 2026-11-02: AC-4 rows rerun on the shipped code with the existing scripts
(kda-task/runs/r3/dec2/dec2.py, dec2_ext.py), same seeds. A record, not a gate; DEC-28 (AC-4 = FAIL) stands.
Per the reviewer's (b) ruling, CP stays on FLA on the merged tip; these kernels are in the library, called by no operator.

Tree identity:
- runs used the gdr worktree; launch-time HEAD d51ba399 (code), branch tip e093895 (kda-task only on top)
- `git diff d51ba39 e093895 -- pithtrain tests`: empty (0 bytes)
- `git diff 5e8f94e d51ba39 -- pithtrain tests | sha256sum`: 37cb3c5d6fcd00f9f91e34a2577939c1b2e3d64abd64892c833d7aa2549bbf79
- node fs-mbz-gpu-388, job 2576852 (step .6), 20:10-20:16Z; per-item rc in done/prec_* (all 0)

Verdict lines (files: dec2_w2.out, dec2_w4.out, ext_a_w2.out, ext_a_w4.out, ext_b.out):
- dec2_w2: {"W": 2, "verdict": "FAIL", "failures": ["mild seed 2 dbeta: calc_diff 6.333e-06 vs FLA 6.332e-06, max-abs 1.267e-02 vs 1.108e-02", "power control: the degraded variant stayed within the bounds (the gate has no power here)"], "power": false}
- dec2_w4: {"W": 4, "verdict": "FAIL", "failures": ["power control: the degraded variant stayed within the bounds (the gate has no power here)"], "power": false}
- ext_a_w2: {"W": 2, "part": "a", "what": "verdict", "verdict": "FAIL", "failures": ["mild seed 6 dbeta"]}
- ext_a_w4: {"W": 4, "part": "a", "what": "verdict", "verdict": "FAIL", "failures": ["model seed 4 dg"]}
- ext_b: {"W": 2, "part": "b", "what": "verdict", "verdict": "FAIL", "calc_diff_every_seed": false, "no_max_abs_ratio_above_1_10": false, "median_max_abs_ratio": 1.0, "fla_ge_ours_seeds": 9}

Reproducibility against the earlier R3 outputs (kda-task/runs/r3/dec2/*.out, sha256 of the JSON rows):
- dec2_w2, dec2_w4 and ext_b: identical to the earlier runs.
- ext_a_w2 and ext_a_w4: every value of ours (the first element of each calc_diff / max_abs pair) is identical. Only the
  FLA side (the second element) of some dg rows differs, in the last digits (e.g. W2 model seed 3 dg FLA calc_diff
  4.37771e-6 earlier vs 4.37755e-6 now): FLA's dg is not bit-reproducible from run to run. The verdicts and failing rows
  are the same as in 2283f8b.
- W2 power is "true" by the letter only: DEC-28.1 records it as vacuous in substance.
