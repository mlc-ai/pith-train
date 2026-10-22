# Week3 strict checkpoint ↔ HF integration

> Historical implementation/evidence snapshot. See
> [current Omni acceptance entry points and status](OMNI_ACCEPTANCE_STATUS.md)
> for the integrated implementation and Modal CLI workflow. Old Flame/Slurm
> package, authorization and queue statements below describe their original
> snapshot, not the current execution. Historical NOT_RUN/PASS results are
> not automatically inherited by a newer commit.

Local follow-up to frozen `0b35cb755f9e8ade910c3863e6a053fe7cd77464`.
Worktree `/home/kldxbjz/code/pith-train-week3-strict-converter`, branch
`feat/omni-strict-converter-integration`. Earlier worktrees, PR100's
`24529b3101dd339782d2dc97b4ffff06685a60e7`, PR101 validation, the frozen first
Flame package and FA4 are untouched. No upload, submission or publication.

## Public conversion interface

```bash
# Both calls execute on CPU, without a process group.
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python \
  examples/convert_checkpoint/qwen3-omni-text/script.py dcp2hf \
  /run/checkpoints/torch-dcp/00000002 /new/hf

CUDA_VISIBLE_DEVICES= PYTHONPATH=. python \
  examples/convert_checkpoint/qwen3-omni-text/script.py hf2dcp \
  /new/hf /new/restored/torch-dcp/00000002 --restore-resume
```

Export autodetects `strict-manifest.json` and recovers the exact text config
from the saved numerical contract. An optional `--text-config` must agree.
`--prefix thinker.` (default) or `--prefix ''`, and `--expert-layout fused`
(default) or `split`, retain the existing explicit HF namespace/layout rules.
The output contains normal text-only HF safetensors/config and a
`pithtrain-resume/` companion. The companion preserves per-rank training
state and model descriptors/digests, without a second copy of model weights.

`--restore-resume` requires that companion. It reconstructs the original
model shards from the HF weights and verifies every original shard's dtype
and digest before writing a completed strict container. Editing HF weights
invalidates this strict resume path: stale optimizer/data/RNG cannot silently
accompany changed weights. Ordinary `hf2dcp` remains the existing model-only
initialization path. The destination for strict import is the **step directory**;
pass its parent checkpoint root to production `load_checkpoint(..., strict=True)`.

The bridge verifies the exact manifest/rank file set and hashes, complete text
key/shape coverage, original PP/EP ownership, supported FSDP dim-0 shards,
replica consistency, AdamW moments/group membership and counters, scheduler
state, committed cursor, and all RNG components. CUDA RNG is preserved as
opaque validated byte storage on CPU; only actual CUDA loading validates its
generator representation. Incomplete containers and invalid bundles fail
before creating the destination. Destinations must be empty. An interrupted
write is not transactional cleanup: preserve/remove the incomplete new output
before retrying; only the final completion manifest marks a valid container.

No topology migration or source-code identity rewrite occurs. An old checkpoint
keeps its old code/config identity, so production strict loading still requires
the original matching runtime. Scope remains CP1/HSDP1, AdamW and
StepLR/LambdaLR; Muon and changed topology are unsupported. Audio/video
encoders and DeepStack remain outside this text-only work.

## CPU evidence

`tests/test_strict_omni_converter.py` generates actual strict containers through
public checkpoint save, converts through the public converter/CLI, and reloads
through public strict load in fresh Gloo processes. It covers `(world,PP,EP)`
`(1,1,1)`, `(2,1,1)`, `(2,2,1)`, `(2,1,2)`, `(4,2,2)`, including uneven 3-expert
2/1 DP shards, FP32/BF16, StepLR/LambdaLR, both HF prefixes/layouts.

All tensors and full recovered payloads must match exactly. HF eager logits
and predictions must also match exactly. The CPU model uses real HF text
modules with native-shaped parameter names, explicit DTensors, a test cursor,
and synthetic gradients to isolate serialization. These are **not** native
PP/EP computation or real PR100 loader acceptance. Native continuation remains
the separately prepared H100 gate below.

Negative cases cover missing/completed rank files, checksums, extra files,
missing/unexpected model state, shape/dtype/nonfinite tensors, absent/malformed
optimizer moments/groups/order, malformed scheduler/cursor/RNG/topology/config,
changed HF weights, absent companions, and explicit config mismatch. CLI
tests require no explicit config for strict export and protect existing output.

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. HF_HUB_OFFLINE=1 python -m pytest \
  tests/test_strict_omni_converter.py tests/test_qwen3_omni_checkpoint.py \
  tests/test_strict_checkpoint.py tests/test_distributed_strict_checkpoint.py \
  tests/test_pretrain_data_lifecycle.py tests/test_omni_model_registration.py \
  tests/test_omni_native_resume_contract.py -q
```

The large released model remains **key/shape/header coverage only**. No 30B
payload was downloaded or numerically validated. The bridge stages rank state,
assembled weights and conversion copies in host RAM; it is validated at tiny
scale and makes no large-checkpoint memory/performance claim.

## Next H100 round: prepared, not executed

Only the existing sole Flame owner may run this from its separately coordinated
allocation, with at least four H100s, the existing pinned environment and the
real `omni-training-small` recipe. This command does not submit or upload:

```bash
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 bash tests/acceptance/run_week3_h100.sh \
  /existing/venv/bin/python /real/omni-training-small /new/week3-evidence
```

The runner checks the target, executes the unchanged native converter and
HF-reference tests, then runs production prepare → produce → CPU strict-to-HF
→ CPU HF-to-strict → **fresh-process native restore/update** for all five mesh
profiles above. The restore driver accepts `--restore-root` to load the
converted strict container. Five separate combined PP2EP2 negative jobs mutate
one rank: missing file, missing CUDA RNG, shape, topology, optimizer order.
The final collector requires all **42 per-rank reports** to say `PASSED`.

Planning allowance: **25–45 minutes** including cold compilation and process
startup; this is an estimate, not measured H100 evidence. Each single-GPU gate
has a 120-second cap, each multi-GPU mode 180 seconds, and CPU conversion/
preparation 120 seconds. The complete runner has a 55-minute budget, preserving
five minutes within a one-hour allocation for cleanup/evidence. First failure
or budget exhaustion terminates the runner; subsequent gates are **NOT RUN**.
The sum of every per-gate maximum can exceed the total budget; do not infer
that all gates are guaranteed to finish in one hour or extend the limits.

Pass criteria stay unchanged: zero exit codes; exact conversion tensors and
native pre/post-conversion predictions; native/HF logit normalized squared
error <1e-3 and existing CE/gradient assertions in the pinned reference test;
exact strict boundary, batch, optimizer/scheduler/data/RNG and next-update
state on **every rank**; all ranks reject each corruption without live-state
mutation. Keep the first failure. No SDPA substitution, FA4 patch or tolerance
relaxation is permitted by this handoff.

Remaining gates: H100 execution of the above, CUDA RNG/runtime FSDP load hooks,
NCCL failure behavior, and the repository's separate 32-step base0/base1/feature
noise-envelope regression. Three-step exact continuation does not replace the
32-step gate (<3 noise ratio pass, 3–5 investigate, >5 fail; zero noise requires
exactness). Cross-topology/Muon/CP>1/HSDP>1 remain unimplemented. PR100 already
supplies the data entry; this branch adds no loader. External plan/page updates
remain paused with publication and should be made only by the parent when resumed.


## Recorded local validation

On 2026-10-08, the combined CPU regression completed **179 passed** in
137.71 seconds (85 expected single-process DCP warnings). After the final
outer-config quantization guard and excluded-key reporting review, the focused
import/CLI suite completed **10 passed, 29 deselected** in 19.84 seconds.
There are 39 bridge cases in the final tree; 38 were in the combined run and
the added outer-quantization case passed in the focused run. Ruff, shell
syntax and `git diff --check` passed. No GPU acceptance was executed.
