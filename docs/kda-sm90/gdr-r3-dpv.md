<!-- Copied from kda/sm90-gdr c7e757f kda-task/runs/r3/final/dpv_summary.md (logs there); merged as c75a394. -->

# AC-10 dpv loss checks on the shipped code (wrap-up item (b))

test_dualpipev qwen3.5-35b-a3b (pp 2, ep 2, mbs 3), each run a fresh 8-rank torchrun (kda-task/runs/r3/dpv/dpv_r3.sh),
per-item rc in final/done/dpv_*. Feat runs: the gdr worktree (d51ba39 code; `git diff d51ba39 e093895 -- pithtrain tests`
empty; `git diff 5e8f94e d51ba39 -- pithtrain tests | sha256sum` = 37cb3c5d6fcd00f9f91e34a2577939c1b2e3d64abd64892c833d7aa2549bbf79).
Base runs: workspace/kda-sm90/base-5e8f94e. The merged tree's non-CP code equals e093895's: merge_tree_check.md.
Rule (plan-r3 AC-10 / DEC-17): FP8=0 passes test_dualpipev's loss and grad assertions; FP8=1 feat step-0 loss max abs
diff within the cap 0.0080 and the base envelope 0.00554-0.00687, against a same-node base pair.

| run | job / node | result |
|---|---|---|
| feat_cp2_fp80 (CuTe CP) | 2576852 / fs-mbz-gpu-159 | PASS: loss matches steps 0, 1; all gradients match |
| feat_cp2_fp81 (CuTe CP) | 2576852 / fs-mbz-gpu-159 | FAIL: step-0 max abs 0.00895 / 0.00693 / 0.00623 (rank 0/1/2) > cap 0.0080 |
| base_cp2_fp81 (FLA CP, record only) | 2394207 / fs-mbz-gpu-223 | step-0 max abs 0.00906 / 0.00732 / 0.00675 / 0.00598 (rank 0-3); not a same-node pair |
| feat_cp1_fp80 (non-CP fold, pinned) | 2398674 / fs-mbz-gpu-006 | PASS: loss matches steps 0, 1; all gradients match |
| feat_cp1_fp81 (non-CP fold, pinned) | 2398674 / fs-mbz-gpu-006 | PASS: step-0 max abs 0.00612 / 0.00522 / 0.00515 (rank 0/2/3) <= cap, inside envelope, <= base |
| base_cp1_fp81 (FLA, same node) | 2398674 / fs-mbz-gpu-006 | step-0 max abs 0.00632 / 0.00583 (rank 0/2) |

Reviewer rulings (2026-11-02): CP2 FP8=1 is an AC-10 FAIL, so CP stays on FLA on the merged tip (the gdr_cp_* kernels
stay in the library, not routed). The non-CP batch fold is kept as feat_cp1_fp80 / fp81 pass. Finding, not a rule
change: base's FLA CP path is also above the non-CP cap at CP2 FP8. FP8=1 runs stop at test_dualpipev's 5e-3 step-0
assert in both arms, as in R2, so they print no gradient line. The packed pair and neg_no_bwd_merge run after this,
for the record only.

## Record-only runs (job 2398674, fs-mbz-gpu-006, 20:39-20:44Z)

| run | result |
|---|---|
| base_cp2_packed (5e8f94e) | stops at the forward_posemb assertion `cu_seqlens is None or distributed.cp_size == 1` (qwen35_moe.py:604), as DEC-23 expects |
| feat_cp2_packed | the same assertion, as DEC-23 expects; packed + CP is not runnable at model level on either tree |
| neg_no_bwd_merge (feat, CuTe CP, backward merge disabled) | did NOT fail: "Loss matches" at steps 0 and 1, and "All gradients match the reference". The linear_attn.A_log largest grad diff rises from 0.0086-0.0096 (feat_cp2_fp80, ranks 4-7) to 0.0221-0.0232, which is still inside test_dualpipev's tolerance |

Finding (record only, no merge effect since CP is not routed to CuTe): at dpv's CP2 shape (T 64 per rank, mbs 3)
test_dualpipev's gradient check has no power against a missing backward merge. The plan's AC-10 negative test
("fails the CP_SIZE=2 grad assertions") is therefore not met. The op-level AC-3 / AC-5 handoff tests remain the
evidence that the merge is needed. A future CP routing would need a model-level negative with power, for example a
longer T per rank or a tighter A_log tolerance.
