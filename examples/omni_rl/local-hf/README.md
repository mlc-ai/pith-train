# Local HF rollout for native Omni GRPO

This recipe connects the existing native DualPipeV/FSDP policy adapter to a real
local Transformers 5.17.0 Thinker rollout and a fixed reference policy. It reads
actual local prompts and media, samples from the newly installed policy, computes
old/reference selected-token log probabilities, evaluates a deterministic answer
reward, and updates the native policy. It does not start a service or download
weights. The original protocol remains usable with other explicitly implemented
providers.

The bridge supports **native DP/PP/EP with CP1 and HSDP1**, microbatch size 1,
an even local GRPO group of at least `2 * PP`, and BF16 native training. One HF
worker runs on global rank zero. Weights use explicit CPU/Gloo tensor transport
in chunks of at most 16 MiB (default 4 MiB), never NCCL object payloads. Duplicate
DP replicas select one owner. DP ranks get distinct prompts; PP followers receive
identical tokens, positions, probabilities and rewards, with media features only
on PP0. CPU tests cover HF training and separate two-rank Gloo DP/PP transport.
The Gloo test uses canonical shard/partition fixtures, not CUDA DualPipeV.
Native GPU training through this recipe is **NOT_RUN** until its acceptance gate
executes. The CLI constructs the real native trainer and strict checkpoint path.

## Inputs and memory admission

Provide an existing local model config, HF safetensors checkpoint and local
processor/tokenizer directory. Full released payloads are not downloaded or
claimed tested. `--scope text` explicitly selects only the Thinker text weights;
`--scope thinker` retains both encoders and the complete Thinker state. Talker,
RVQ/audio-policy RL, external reward judges, and verl/Slime are outside this
provider. The frozen reference is the supplied checkpoint, never the current
policy after an update.

Each JSONL record has exactly these fields:

```json
{"id":"sum-1","text":"What is one plus one? Return only the answer.","media":[],"answer":"two"}
{"id":"image-1","text":"Name the color. Return one word.","media":[{"type":"image","path":"red.png"}],"answer":"red"}
```

Media paths must remain inside the manifest directory. Image, audio, silent
video and synchronized audio/video use the existing Omni reader/collator and
its limits (`--media-options '{"use_audio_in_video":true}'` enables preserved
video audio). Text-only mode rejects media. No prompt truncation, silent media
drop, or implicit data cycling occurs. One record supplies one complete group
per DP coordinate and request; provide at least `requests * (WORLD_SIZE / PP)`
records on every rank. The supplied four-row example is for a small single-DP
run; explicitly extend it before running the distributed example below.

The declared prompt format is **document-prefix-v1**: EOS document boundary,
media prefix in listed order, then the given prompt text. It deliberately does
not guess a conversational chat template. A future conversational formatter
must use a distinct bound format contract. Raw manifest/media bytes, processor
assets, actual processed tensors, answer text, model/position config and code,
reward normalization and tokenizer, sampling settings, and fixed reference
weights are bound into restore validation.

HF policy and reference retain FP32 weights so installation can be read back
exactly. The native CLI requires `--hf-compute-dtype bfloat16` autocast. The
library also supports explicit FP32 for CPU HF tests. Both models default to CPU.
`--hf-device cuda:0` puts both on the leader's native training GPU,
which generally has a much larger memory cost. There is **no automatic offload
or memory-fit guarantee**. `--weight-limit-mib` bounds one canonical state;
the unchanged 512 MiB default rejects released 30B weights before allocation.
Above that bound, an explicitly raised canonical limit also requires
`--host-weight-limit-mib`, and CUDA placement requires
`--device-weight-limit-mib`. These are provider **weight-residency budgets**,
not declarations that the entire native training job fits.

The pinned released Thinker configuration has 126,876,821,952 FP32 parameter
bytes (118.163 GiB). The native recipe's conservative synchronous-installation
phase contains the parent's reference/publication (2 copies), two worker HF
models, and one incoming worker payload: **5 copies, 590.816 GiB on CPU**.
Consumed reference/request payloads and their IPC files are released before the
worker acknowledges completion; idle provider weights therefore need 4 copies.
The separate single-process `LocalHFThinkerRollout` convenience adapter retains a
defensive parent reference copy and requires a 6-copy host budget instead.
Native parameters/Adam/checkpoint state, model buffers, activations, allocator
fragmentation, temporary tensor contiguity copies, and the OS need additional
memory outside these budgets. Native canonical conversion can also copy expert
local tensors. No released peak RSS or throughput has been measured.

Installation copies at most 4 MiB at a time directly into strict canonical views
of the live HF parameters. Readback hashes the actual model in the same bounded
chunks, retaining the original byte-for-byte digest encoding. It neither builds
a full fused-expert export nor clones an entire model for readback. The worker
keeps at most one full weight IPC file at a time, plus bounded request/results
and serialization metadata; allow more than 118.163 GiB of temporary disk for
the released model, separately from its input files and training checkpoints.
No checkpoint, source input, or unrelated file is deleted by this cleanup.

A metadata-only **weight-budget** example for the native CPU-worker recipe is
`--scope thinker --hf-device cpu --weight-limit-mib 131072
--host-weight-limit-mib 655360 --plan`, along with the required local paths.
The 640 GiB declared host weight budget is not a full-job resource reservation:
native training/checkpoint and runtime memory must still be planned separately.
A 256 GiB host cannot pass this released-model admission. Existing worker and
Gloo deadlines still apply; slow full-model IO is not implicitly given retries
or a longer timeout.

With `--hf-device cuda:0`, both worker models still retain FP32 parameters:
**236.326 GiB of device weights alone**, alongside native training on that GPU.
A declared 141 GiB device budget is rejected. This implementation does not
provide sharded HF serving. Two all-BF16 device models would use about 118.163 GiB
of weights, but would change embedding/residual/RMSNorm dtype behavior and fail
the current exact FP32 installed-weight readback. BF16 autocast is not that
storage contract. Such a backend needs a distinct numerical/installation
contract and independent validation; it is not silently enabled here.

Publication validates native live `config/image_config`, RMS, RoPE, head and
routing attributes, forward implementation and compute policy against explicit
HF correspondence rules before installing weights. Worker acknowledgements bind
actual HF config/code/compute and read-back weights. This is not a substitute for
the independent numerical oracle; identical implementations are not claimed.

## Run

From the repository, inspect a local configuration without CUDA/model allocation:

```bash
python examples/omni_rl/local-hf/script.py --plan \
  --model /local/tiny-model --reference-hf /local/tiny-model \
  --hf-prefix '' --scope text --processor /local/tokenizer \
  --prompts examples/omni_rl/local-hf/prompts.jsonl --checkpoint /local/run
```

Run bounded native training on already assigned GPUs:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  timeout --kill-after=10s 600s \
  torchrun --standalone --nproc_per_node=4 examples/omni_rl/local-hf/script.py \
  --model /local/tiny-model --reference-hf /local/tiny-model \
  --hf-prefix '' --scope text --processor /local/tokenizer \
  --prompts examples/omni_rl/local-hf/prompts.jsonl --checkpoint /local/run \
  --pp 2 --ep 2 --requests 4 --updates 4 --group-size 4 --sequence-length 256 --new-tokens 32 \
  --worker-timeout 120 --hf-device cpu --hf-compute-dtype bfloat16
```

The checkpoint directory must be local. To resume, reconstruct with identical
arguments and append `--resume-request N`, where N is the saved completed-request
count. Provider RNG/cursor, reward cursor and all semantic fingerprints are
validated before model/optimizer state is applied by the existing strict task
checkpoint loader. Each later synchronization installs the restored native
weights again; no stale worker cache is accepted. Mid-request/KV-cache resume is
not supported. Failed transactions require a fresh trainer and the last committed
checkpoint; there are no automatic retries.
Use `--no-save` for replay verification without overwriting later checkpoints.

The worker is an owned spawned process. Startup, installation and generation
each have the explicit deadline. On timeout/failure it is terminated, then killed
if needed; unverified cleanup fails closed. Files are local temporary IPC only.
The outer timeout bounds native training and bounded input preparation as well.
Neither timeout is a cloud billing cap or an allocator.

Sampling uses the full categorical distribution without top-k, top-p or a hidden
temperature. Old/reference log probabilities are unmodified model probabilities,
aligned to next-token targets exactly once. EOS ends the response mask; remaining
fixed-length causal EOS inputs are outside the objective. An unpaired generated
media delimiter in complete-Thinker mode rejects that transaction instead of
silently altering the sampling distribution. This limitation is explicit.

The built-in reward is exact decoded answer equality, with configurable stripping
and case folding. Prompt/padding tokens are never rewarded. This is suitable for
verifiable text-answer tasks, including answers conditioned on media; it makes
no claim of conversational quality, reward improvement or audio-policy learning.

## Focused CPU validation

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_omni_hf_rollout.py \
  tests/test_omni_hf_transport.py tests/test_omni_hf_semantics.py \
  tests/test_omni_hf_provider_acceptance.py -q
```

The tests use actual tiny HF sampling, selected-logprob checks, a real Adam
update, a changed native-format publication digest, a fixed reference, fresh
provider/trainer recovery and bitwise next-update equality. Complete Thinker
media cases exercise real HF encoders and preserve all context/feature tensors.
Negative cases cover source/weights/decoder/config/cursor/RNG drift, omitted
media, byte bounds and actual worker process termination. These CPU results do
not establish GPU/native topology parity, full-size memory fit, or trained-model
quality.

## Actual native provider acceptance

`tests.acceptance.omni_hf_provider` uses this exact production CLI/worker/reward,
one w4/PP2/EP2 CP1/HSDP1 profile, real local tiny HF tensors and real tokenized
prompts. Prepare samples one bounded initial group per DP coordinate and freezes
its answer label, requiring nonconstant actual `ExactAnswerReward` scores. It
does not adapt answers after training or claim task quality. Produce requires
nonzero actual gradients and independently changed installed policy digests,
saves checkpoint 2 and completes update 3. A separate process restores checkpoint
2 and requires bitwise boundary/next-update model, Adam, scheduler, data and RNG
equality, identical reports, and actual worker installation/readback. The original
`omni_posttraining` numerical oracle remains separate and unchanged.

Use a fresh heavy state directory outside the summary/log collection root.
Replace SHA and hardware with exact values. Caps are abort bounds, not measured
completion times; no command allocates cloud resources.

```bash
python -m tests.acceptance.omni_hf_provider plan \
  --expected-commit "$SHA" --hardware "$GPU_NAME" \
  --workspace /local/provider-state --output /local/provider-reports
timeout --kill-after=10s 120s python -m tests.acceptance.omni_hf_provider prepare \
  --expected-commit "$SHA" --hardware "$GPU_NAME" \
  --workspace /local/provider-state --output /local/provider-reports
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout --kill-after=10s 600s \
  torchrun --standalone --nproc_per_node=4 -m tests.acceptance.omni_hf_provider produce \
  --expected-commit "$SHA" --hardware "$GPU_NAME" \
  --workspace /local/provider-state --output /local/provider-reports
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout --kill-after=10s 300s \
  torchrun --standalone --nproc_per_node=4 -m tests.acceptance.omni_hf_provider restore \
  --expected-commit "$SHA" --hardware "$GPU_NAME" \
  --workspace /local/provider-state --output /local/provider-reports
```

Collect only summary JSON and bounded logs. Prepared tensors, checkpoints and
full comparison `.pt` files stay under `--workspace`. All four ranks must have
successful produce and restore receipts, with unchanged source verified again
before PASS. CPU prepare, a partial report, or a process starting is not GPU PASS.
Native media RL numerical parity, released payloads and full-size feasibility
remain distinct validation requirements.
