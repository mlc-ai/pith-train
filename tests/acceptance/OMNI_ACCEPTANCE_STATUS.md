# Omni acceptance: current entry points and historical evidence

The current scope is Qwen3-Omni Thinker multimodal pretraining with text
prediction, against `Qwen/Qwen3-Omni-30B-A3B-Instruct` revision
`26291f793822fb6be9555850f06dfe95f2d7e695` and Transformers 5.17.0.
Talker/code2wav, SFT and RL are outside this scope.

Implementation, CPU contracts and native GPU acceptance are separate states.
The Week3/Week4 documents preserve earlier implementation and execution
snapshots; their old branch names, frozen packages, pending jobs and resource
proposals do not describe the current integrated checkout. The old Flame jobs
are terminal historical attempts, not queued work or passing acceptance.

## Current implementation and limits

- Production strict checkpoint save/load is wired through `pretrain_lm`.
  `omni_distributed_resume.py` exercises the native DualPipeV/FSDP path.
  `omni_native_resume.py` remains a narrower, unsharded native-model gate.
- Native image, audio, video and unified AV/mixed consumers are implemented
  behind explicit selectors. The unified consumer provides the modality-absent
  collective contract; the earlier image-only declaration remains narrower.
- Complete-Thinker conversion is exposed by
  `examples/convert_checkpoint/qwen3-omni-thinker/script.py`. Text-only conversion
  remains separate. A full Omni checkpoint requires explicit Thinker-only
  selection; Talker/code2wav keys are accounted for rather than silently loaded.
- Strict continuation requires the same supported topology/code/config, CP1,
  HSDP1 and supported AdamW state. Native media CP>1, changed-topology strict
  restore, Muon strict restore and quantized complete-Thinker conversion are
  unsupported. No FP8 or multi-node acceptance is inferred.
- `omni_full_size.py` is a separate resource-planned gate. Tiny numerical,
  header/schema and meta-model checks do not establish full-size correctness.

## Recorded snapshot, not a final completion claim

At commit `ffd6010c70f3b95b05c42b5bbc47fd62e64e502f`, the CPU suite completed
906 tests with no failures or skips; all eight pre-commit hooks passed.
The H100 text HF/reference plus text conversion rerun passed 11 tests.
The coordinated Stage A run reported dense-single, packed-ragged-single,
dense-dp2 and dense-ep2 passing; the other topology cases remain separate gates.
These results belong to that commit and do not automatically validate a later
head or the new media paths.

Actual production media failures remain unresolved in this snapshot:

- Image reference initially failed gradient NSE 0.018093418160302157. The
  compiled staged retry at `d7b11be88c1042f173a4749e246e1ef56e028396` failed
  `model.layers.3.mlp.experts.gate_up_proj` NSE 0.028651398568639238.
- Six standalone audio encoder GPU cases passed, but the combined audio/text
  gate at the same commit failed `audio_tower.layers.0.self_attn.k_proj.bias`
  gradient NSE 0.8366303237450347.

Both require NSE <0.01. No gradient is excluded and no threshold is relaxed.
An eager-only diagnostic, a CPU prepare, a parsed CLI or a successful allocation
does not replace these failed production gates. Video/AV/mixed GPU acceptance,
complete native checkpoint continuation, all five noise profiles and full-size
numerical execution are not established by this snapshot.

Retain the exact executed command, clean commit, runtime/hardware manifest,
data hashes, full stdout, JUnit and per-rank reports. Record PASS, FAIL,
INVESTIGATE, SKIPPED, NOT_RUN and UNSUPPORTED separately. A later head needs its
own source binding and evidence reconciliation; no test totals are additive.

## Modal or another existing allocation

The current coordinated execution uses Modal. These Python CLIs do not create
resources or require Slurm; run them only inside the coordinator's existing
bounded allocation. Keep original legacy shell guards intact and do not set
fake Slurm variables to bypass them. Use one shared budget ledger, sequential
cases and existing opt-in gates; stop the coordinator-owned compute afterward.

For native text topology, pass all selected names after one `--case`:

```bash
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PITHTRAIN_RUN_OMNI_GPU_TESTS=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m tests.acceptance.omni_topology_runner \
  --expected-commit FULL_CHECKED_COMMIT \
  --case dense-single packed-ragged-single dense-dp2 dense-ep2 \
  --out /new/evidence/text-topology
```

The runner enforces per-case deadlines and captures all canonical gradients.
`PASS_SELECTED` is not `PASS_ALL_13`. The folded PP2/EP2/CP2 case needs eight
GPUs; it cannot be supplied by a four-GPU allocation.

For the production native image path, use a fresh directory for each scenario:

```bash
timeout --signal=TERM --kill-after=10s 180s \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/acceptance/omni_image_training.py --mode staged \
  --dataset /verified/data --output /new/evidence/image-pp2-ep2 \
  --pp 2 --ep 2 --scenario nonuniform
```

`omni_audio_training.py`, `omni_video_training.py` and
`omni_multimodal_training.py` are separate modality gates with their own CLI
contracts. Do not substitute an image pass for them. The unified runner uses
positional `prepare`/`staged` modes; audio/video use `--mode`.
Use both nonuniform and rank-zero scenarios where required, with unchanged CE
and all-gradient assertions and actual component update witnesses.

The unsharded native gate accepts `preflight`, `update`, `produce`, `restore`
and `negative` modes. The distributed text gate accepts `prepare`, `produce`,
`restore` and `negative`. Run producer and restore in separate torchrun
processes, and run every required corruption negative separately.
See [DISTRIBUTED_STRICT_RESUME.md](DISTRIBUTED_STRICT_RESUME.md) for the state
contract and [OMNI_NATIVE_MEDIA_RESUME.md](OMNI_NATIVE_MEDIA_RESUME.md) for the
audio/combined media consumer contract. Exact restored boundary and next
update remain mandatory; a same-process reader restore is narrower evidence.

## Final-commit noise experiment

The historical `omni_noise_regression.py` keeps its original a0b14f7/d237e4c
commit locks and workload hash. Use `omni_final_noise.py` for an explicitly
reviewed newer feature SHA; do not edit the historical guards.

The reviewed baseline is `d9c791d0c91978499ea3c4b823d46ed61fee014c`, a descendant of
`be0b2cb59c865e0fecc05ae7b9a7ebef968bec11` with the same two stable expert-row ordering
commits applied to the feature and PR101. It retains current upstream,
reviewed #100/#101 fixes and the shared text stage5 helper. The common
ordering fix is excluded from this feature-vs-baseline noise comparison;
it requires its own kernel/reference and strict-resume validation.
The adapter checks identical numerical/model/data source and declares the
checkpoint increment's scope. It does not compare the entirety of #100/#101,
new modalities or full-size training.

```bash
CUDA_VISIBLE_DEVICES= python -m tests.acceptance.omni_final_noise bind \
  --base-repo /reviewed/baseline --feature-repo "$PWD" \
  --expected-feature-commit FULL_CHECKED_COMMIT \
  --output /new/evidence/noise-binding.json
CUDA_VISIBLE_DEVICES= python -m tests.acceptance.omni_final_noise prepare \
  --binding /new/evidence/noise-binding.json --dataset /verified/data \
  --output /new/evidence/noise-prepared
python -m tests.acceptance.omni_final_noise run-arm \
  --plan /new/evidence/noise-prepared/plan.json --profile 1,1,1 --arm base0
```

For each of `(1,1,1)`, `(2,1,1)`, `(2,2,1)`, `(2,1,2)`, `(4,2,2)`, run
base0, base1 and feat0 sequentially, then `compare --profile all`. Every arm
retains 32 steps and real routing. Feature saves at step16, traverses strict ->
HF -> strict and resumes in a fresh process. Each arm has 600 seconds; all
arms share 3300 seconds. Original ratios are <3 PASS, [3,5) INVESTIGATE and
>=5 FAIL. Binding and CPU tests are not numerical execution.
