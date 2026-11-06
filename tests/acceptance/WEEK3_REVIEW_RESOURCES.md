# Focused strict-resume review and next-round resources

> Historical implementation/evidence snapshot. See
> [current Omni acceptance entry points and status](OMNI_ACCEPTANCE_STATUS.md)
> for the integrated implementation and Modal CLI workflow. Old Flame/Slurm
> package, authorization and queue statements below describe their original
> snapshot, not the current execution. Historical NOT_RUN/PASS results are
> not automatically inherited by a newer commit.

This is a local follow-up to frozen `38674d1f1af6d2e2bc96ba70dc6eb2e33784d51f`.
Branch `fix/omni-strict-review`, worktree
`/home/kldxbjz/code/pith-train-week3-strict-review`. No upload or job submission.
The previous branch, package and first-round Slurm job remain unchanged.

## Exact resource envelope

The unified runner's largest case is WORLD4 / PP2 / EP2, so it needs **four
H100 GPUs on one node**, not eight. Proposed next-round allocation:

| Setting | Value |
| --- | --- |
| Nodes / tasks | 1 / 1 (torchrun launches up to four local workers) |
| GPUs | 4 × H100; single node |
| Requested walltime | 01:00:00 |
| Requested allocation ceiling | **4 GPU-hours** (4 × 1 hour) |
| CPUs / host RAM | 32 CPUs / 128 GiB (reservation proposal, not a measured minimum) |
| Working storage reserve | 64 GiB for temporary checkpoints and evidence |
| Requeue | Disabled; no automatic retry |
| Runner budget | 3300 seconds + at most 15 seconds for forced termination |
| Gate limits | single GPU 120 s, multi-GPU 180 s, CPU prepare/conversion 120 s |

The runner budget corresponds to at most **3.6834 allocated GPU-hours** with
four GPUs, excluding outer job startup/cleanup; the one-hour job envelope is
4 GPU-hours. Idle GPUs during CPU stages still count as allocated resources.
Actual consumption must be calculated from Slurm allocation/accounting after
the run. If the cluster allocates eight GPUs despite requesting four, a full
hour would instead cost eight GPU-hours; do not silently increase the request.

The estimate remains 25–45 minutes, not measured H100 evidence. Per-gate caps
sum to more than 55 minutes, so exhausting the global budget stops with failure
and leaves subsequent cases NOT RUN. Queue time is not execution time. This
new round is separate from the earlier approved eight-H100 first-round job;
its allocation and frozen package must not be changed to run this code.

Only the sole coordinated Flame owner may execute after scheduling is approved.
Following the repository `launch-with-slurm` workflow, inside the selected
four-GPU allocation use the following step (paths are the newly staged ones):

```bash
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export HUGGINGFACE_HUB_CACHE=/new/week3-assets/hub
export PITHTRAIN_RUN_OMNI_GPU_TESTS=1
srun --jobid="$SLURM_JOB_ID" -N1 -n1 -c32 --gres=gpu:4 -W 0 \
  --open-mode=truncate -o /new/week3-step.log \
  bash /new/week3-code/tests/acceptance/run_week3_h100.sh \
  /project/flame/zichuwu/orchard/omni-training-env/bin/python \
  /new/week3-assets/data /new/week3-evidence
```

The runner now disables ambient pytest plugins/arguments. Existing numerical
thresholds and all 42 per-rank report requirements remain unchanged. This is
still three-step continuation, not the separate 32-step noise regression.

## Exact transfer scope and reuse

Prepare/upload only these four local files after approval:

1. `week3-reviewed.bundle`: the reviewed branch and its reachable Git source
   history. It preserves the actual commit/tree for the driver's `git` identity
   checks; no working tree caches, venv, dataset, checkpoints or unrelated refs.
2. `week3-assets-delta.tar.gz`: **70 missing bundle files, 24,196,180 bytes**,
   plus **seven processor/config files, 4,477,208 bytes** before compression.
3. `ASSETS.json`: exact archive hash and every reused/uploaded asset's path,
   size and SHA256; required by the assembly helper bundled in the source.
4. `TRANSFER.json`: final code commit, archive sizes/hashes, resources and
   validation evidence summary. Exact final transfer bytes are recorded here.

No 30B weight payload, local checkpoint, private environment/configuration,
credential, FA4/SDPA adapter, dependency installation or original source-cache
directory is in this transfer. The seven pinned files are config.json,
generation_config.json, preprocessor_config.json, tokenizer_config.json,
chat_template.json, merges.txt and vocab.json from revision
`26291f793822fb6be9555850f06dfe95f2d7e695`.

Read-only remote inspection confirmed these four files already exist and match
the local source at
`/project/flame/zichuwu/pr101-flame-e6d1889-9a1c81ae4086/package/data`:

| Reusable file | Bytes |
| --- | ---: |
| tokens/train/00000.bin | 50,052 |
| recipe.json | 2,578 |
| bundle.json | 4,676 |
| checksums.sha256 | 9,998 |
| Total reused | **67,304** |

That partial data directory cannot directly pass the existing `verify_bundle`:
the manifest covers 73 files, plus the checksum file itself makes **74 files /
24,263,484 bytes** in a complete verified bundle. Earlier “73 files” accounting
counted manifest entries. Two stale local media files outside the checksum
manifest are deliberately excluded. The missing media/validation files are
needed only to preserve full bundle integrity checks; training remains text
only and the loader/bundle manifests are unchanged.

No complete matching bundle was found in the two known PR100 package directories
inspected. The default remote cache lacked this pinned processor snapshot.
An owner may reuse another complete copy only after checking **every manifest
hash**, not just directory names or bundle.json. The provided delta plus pinned
config cache removes that dependency on undiscovered cache locations.

On Flame, create new paths; never unpack into the first-round package:

```bash
git clone --single-branch --branch fix/omni-strict-review \
  /new/uploads/week3-reviewed.bundle /new/week3-code
python3 /new/week3-code/tests/acceptance/assemble_week3_assets.py \
  --archive /new/uploads/week3-assets-delta.tar.gz \
  --manifest /new/uploads/ASSETS.json \
  --reuse /project/flame/zichuwu/pr101-flame-e6d1889-9a1c81ae4086/package/data \
  --output /new/week3-assets
```

Assembly hashes the archive and reused files before writing, then verifies
every resulting file. Copying occurs only into the new directory. The cache
is selected through the child process environment, without global settings.

## Review findings and resume semantics

**Only unmodified, exact model-weight roundtrips may restore training state.**
HF→strict requires the original companion and checks each original shard's
dtype and raw-byte SHA256 before creating the destination. Tests now change
one ULP in dense, gate and down weights in both split/fused formats. Every
changed version must fail, leave no strict output and preserve the companion.
Default HF→DCP remains model-only initialization and writes no optimizer state.
There is no option to bypass the digest guard or reuse stale moments with
edited weights. Format hashes provide integrity, not cryptographic provenance
against someone deliberately rewriting both weights and all companion hashes.

Confirmed local fixes:

- A missing live cursor fingerprint on one rank raised before collective error
  exchange. The negative reproduced as an uncaught KeyError and terminated its
  peer. Contract argument construction now runs inside `_phase`, so every rank
  receives the same rejection before mutation.
- Expert offsets previously used balanced division rather than PyTorch's actual
  Shard(0) chunk layout. Five experts across four ranks require local sizes
  2/2/1/0 and offsets 0/2/4/5. The common mapping now uses PyTorch's method and
  validates the local size, for weight and optimizer unstacking.
- Empty expert shards contain no canonical expert entries. Reconstruction now
  supplies only the original zero-element local tensor; it cannot fill a
  missing nonempty weight from live state. CPU coverage includes one expert
  across two ranks and five experts across four ranks.

Optimizer moments stay per-rank, with the original model descriptor/mesh,
parameter membership and update counters; no repartition or reseeding occurs.
Roundtrip fixtures now use nonuniform gradients so moments differ by parameter
and local slice. One-rank moment shape/dtype/mesh/nonfinite corruptions must
reject on every rank without mutation, alongside existing application rollback
and other corruption tests. This validates CPU Gloo/DTensor control paths only.

CUDA/NCCL/FSDP native execution remains NOT RUN. Fail-stop process death or
broken NCCL transport cannot promise an in-process rollback and remains bounded
by configured timeouts. Original code/config identity is preserved, so a strict
checkpoint from the previous commit is not silently relabeled for this code;
the next H100 producer and restorer must both run the reviewed commit.

## Local validation record

The final CPU regression passed **188 tests** in 182.05 seconds (91 expected
single-process DCP warnings). This includes 47 bridge cases, seven real CPU
roundtrip layouts, and three two-process collective corruption/rollback
profiles. The pre-fix malformed-cursor test failed with rank-1 KeyError in
13.01 seconds, confirming the error was not collectively propagated.
No GPU mode or scheduler command was executed.
