# Omni acceptance: current entry points and historical evidence

The target is complete Qwen3-Omni component and training/generation-chain support, against `Qwen/Qwen3-Omni-30B-A3B-Instruct` revision
`26291f793822fb6be9555850f06dfe95f2d7e695` and Transformers 5.17.0.
The scope now includes Thinker, native audio/vision, Talker and its residual code
predictor, Code2Wav, multimodal input processing, conversion, speech training,
SFT and project-relevant RL integration. It does not include Qwen2.5-Omni or
other Omni families. An implemented interface is not end-to-end acceptance.

The later `da5c47f4d92193e3d38a3e398b9f188d84e7f9d7` complete CPU
artifacts record 1924 passes in 85 files, with the original executor exit still
unknown; exact collection and file hashes were reconciled without rerunning the
suite. At `ed95505a688feaae52c9f27bdedf3842db9805bc`, 132 affected CPU cases
passed with native exit 0. Its actual 4-H100 run passed independent FP64 attention
but failed speech step 1: eight expert gradients were missing and two audio
key-bias NSE values exceeded 0.01. Outputs, loss and input gradients passing did
not override those failures; subsequent 19 gates were not run. The instance was
confirmed stopped at 2026-10-21T06:03:49.575919Z.

The successor repairs eager expert autograd propagation and the official single
BF16 codebook reduction in the reference, plus verified bundle identities,
separate pretrained-weight initialization and one-pass image decoder allocation.
The latter changes initialization RNG; final-source evidence must be refreshed.
Targeted CPU regressions are development evidence, not a new GPU pass. PR receipts
retain exact source identities and failed runs; test totals are not additive.

At `023e5b549c5052f7559942fac2971d012a78f590`, the actual four-H100
speech step-1 comparison checked all 351 parameter gradients: 349 PASS, 2 FAIL,
0 missing. The eight previously missing expert gradients passed. The remaining
`audio_tower.layers.0/1.self_attn.k_proj.bias` NSE values were 0.5059281896 and
0.5779254232 against the unchanged strict <0.01 threshold. Small absolute errors
do not turn those results into passes. All 20 forward, loss and six floating
input-gradient checks passed, as did independent FP64 attention. The execution
inventory was 5 PASS (including prerequisites), 1 FAIL, 22 NOT_RUN; it is not a
count of all model requirements. Subsequent continuation, generation,
posttraining and codec-policy GPU gates were not run. Released/full-size
acceptance remains NOT_RUN. The instance was confirmed stopped at
2026-10-21T07:35:26.326781Z. The sealed evidence archive SHA-256 is
`0314269241987240301049cc353ca68356839fbc30e0b5c403591b00704db770`.
This later evidence does not replace the failed historical-source receipts above.

The preserved a8a1302d25140e5f75e9f9fb83d46fdcd891fb59 CPU receipt has
1813 PASS / 0 FAIL / 0 SKIP and eight passing hooks. It predates the scheduler,
RL model/forward contract and official full-generation-entry corrections.
The first corrected source, 548f6a811efe09bb3fc97aba27d28c707b977c8e,
completed 1836 CPU tests: 1805 PASS, 31 FAIL, 0 SKIP. Its failures expose the
strict-conversion scheduler-contract integration and an uninitialized official
CPU generation fixture. That failed receipt is retained; the repairs require a
new source freeze. Old CPU/GPU results are not transferred to the successor. No native GPU pass is claimed for a8a1302.

| Component or path | Implementation | Acceptance still required |
| --- | --- | --- |
| Thinker MoE, T/H/W mRoPE, text loss and routing | Native staged model | Final-source numerical/gradient/topology gates |
| Audio encoder, vision/DeepStack, image/audio/video/AV insertion | Native trainable components and preprocessing | Each media, gradients and integration |
| Talker and 16-codebook residual predictor | Native differentiable model | Final-source output/gradient/update and generation |
| Code2Wav | Native differentiable decoder, batch/chunk decode | Payload conversion, waveform gradients and cross-chunk parity |
| Full conversion | qwen3_omni_full.py, including Talker/Code2Wav | Actual released payload; 28010-key header checks are not payload validation |
| Speech codec/text/waveform training | omni_speech, local replicas and FSDP2 | Native numerical and next-update checkpoint gates |
| SFT | Task/data/objective and native training entry | Final-source integration and resume |
| GRPO text policy | Real native optimizer loop plus owned local HF worker, fixed reference, exact-answer reward and CPU/Gloo DP/PP/EP publication | Production-provider native integration, original independent numerical gates and released/full-size execution |
| Codec policy RL (PR #124) | Single-device frozen-Thinker context, trainable Talker/residual predictor and discrete sequence/EOS reward | Native GPU numerical/update/resume gates and released/full-size execution |
| Full text-to-speech generation | Native controller and complete HF reference entry | Natural nonempty output for each media, EOS/caps/multi-turn |
| Strict continuation | Data/media/processor and model/forward/scheduler contracts | Final-source positive and pre-mutation negative gates |

Open implementation boundaries must not be relabelled as passing: speech
training is CP=EP=PP=1; Thinker topology support does not imply Talker topology
support. Speech FSDP training does not establish FSDP generation. GRPO optimizes
text tokens. `examples/omni_rl/local-hf/script.py` now connects actual native
training to one owned local HF policy/reference worker, configurable local
prompt/media input and exact-answer reward. Explicit CPU/Gloo transport supports
native DP/PP/EP at CP1/HSDP1; it is not an external serving service. CPU HF/Adam
continuation and Gloo shard/partition transport tests are distinct evidence.
`omni_hf_provider.py` supplies the bounded w4/PP2/EP2 production CLI integration
and strict fresh-process update gate; GPU execution is still NOT_RUN. The old
`omni_posttraining.py` independent numerical oracle remains separately required.
Released/full-size provider memory and execution are not established. The
separate codec-policy path in PR #124 supports DP=CP=EP=PP=1 and one unpadded,
frozen-Thinker context per microbatch. Talker projections/trunk/heads and the
residual predictor are trainable; Code2Wav is outside the discrete policy gradient.
Its sequence/EOS reward is a protocol reward, not perceptual or ASR quality.
Raw-media end-to-end, multirank/FSDP codec RL, released/full-size execution and
external reward/serving backends are not established; multirank/FSDP is rejected.
Codec-policy GPU acceptance remains NOT_RUN. Generation
has no serializable KV/in-flight stream resume. Training checkpoint recovery
does not validate interrupted generation. Padding/packing, every precision or
optimizer, and changed-topology strict restore require their own explicit
support rather than inference from the basic path.

The mandatory final matrix and resource plan must distinguish these code gaps,
harness defects, infrastructure restrictions and model numerical failures.
Full-size FP32-master Adam training cannot fit four 80GB cards: ideal four-way
Thinker master+Adam shards alone use about 85.31 GiB/card (113.74 GiB including
FP32 gradients), before activations and other components. Do not admit that
configuration or promise all gates fit one two-hour allocation.

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
- `omni_released_continuation.py` adds an explicitly bounded Thinker-only
  released-payload prepare, public initialization, update/save and fresh-process
  next-update entry. It preserves all-parameter checks and complete state/data
  digests; its tiny CPU/Gloo protocol tests do not establish released/native GPU
  acceptance. Full speech and full-size HF provider resource gaps remain separate.
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

The reviewed baseline is `5a50a7dc950ea2ca6d4066879ed6005aed62dab0`, the exact
`BASE_SHA` enforced by `omni_final_noise.py`. Its single parent is the preserved
`1611d085a7c244bb01a34caf8b58d19aef9eeeff`. Only grouped-linear autograd,
decoder construction and prepared-bundle identity repairs are shared into that
derived baseline; all 33 common files still require equality against the feature.
The adapter records the exact parent, tree, patch and three repaired file hashes.
No old baseline arm is reused, and numerical equivalence to unrepaired 1611 is
not claimed. These repairs also require their separate original GPU gates.
The obsolete d9c791d baseline is not accepted. Preserve all source-comparison checks. The common
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
