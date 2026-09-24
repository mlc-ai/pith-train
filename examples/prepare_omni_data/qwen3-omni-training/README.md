# Qwen3-Omni training data preparation

This recipe prepares real train/validation corpora, not hand-written fixtures.
Defaults are deliberately small: **16 train + 4 validation records per modality**
(80 records with all four modalities). Increase the limits in `config.json` to
prepare more data through the same path. No model weights or GPU are required.

## Sources and splits

| Modality | Source | Train / validation policy |
| --- | --- | --- |
| Text | [DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | Stable SHA-256 split of normalized documents; 10% held out before selection |
| Image | [COCO Karpathy](https://huggingface.co/datasets/yerevann/coco-karpathy) | Published train / validation splits, first original caption per image |
| Audio | [LibriSpeech](https://www.openslr.org/12) via [OpenSLR on HF](https://huggingface.co/datasets/openslr/librispeech_asr) | train-clean-100 / dev-clean, original transcripts |
| Video | [MSVD](https://huggingface.co/datasets/VLM2Vec/MSVD) | Published train / validation splits, first original caption per clip |

The config pins every dataset revision and the official Omni processor revision.
Test splits are never selected. Shared image IDs, audio speakers and original
video IDs cannot cross the selected train/validation boundary. Duplicate sample
IDs and duplicate media bytes are rejected. DCLM's holdout is by exact normalized
document, not a semantic near-duplicate filter. This is a development training
corpus, not a claim to reproduce Qwen's original data mixture or evaluate general
model quality. Upstream source/asset terms continue to apply.

COCO images come from its public S3 bucket over HTTPS. Captions/transcripts are
the source annotations; the preparation code does not invent replacement text.
The MSVD adapter is for visual captioning. Videos with audio tracks are rejected;
audio/video synchronization and Talker targets require a later data contract.

## Prepare only what the model supports

From an installed PithTrain environment with the `omni-data` extra:

```bash
python examples/prepare_omni_data/qwen3-omni-training/script.py prepare \
  --stage video --output workspace/datasets/omni-training
```

Stages are cumulative, defined in the config:

| Stage | Enabled inputs |
| --- | --- |
| `text` | Text |
| `image` | Text + image |
| `audio` | Text + image + audio |
| `video` | Text + image + audio + visual video |

Use `--stage text` or another earlier stage to avoid fetching later modalities.
A full prepared bundle can also serve earlier-stage loaders without preparing
another copy. Loading an unavailable stage fails; media is never silently removed
from a sample to make it compatible. The defaults do not enable synchronized
audio-in-video merely because `video` is enabled.

## Scale up without rewriting the loader

Copy `config.json` and change:

- `samples_per_modality.train` / `.validation`: accepted records per modality.
- `modality_sample_limits`: optional per-modality train/validation overrides.
  For example, `{"text": {"train": 10000}, "video": {"train": 500}}` keeps the
  default validation count while preparing different amounts of text and video.
  This prevents a small source from limiting the other modalities.
- `max_scan_per_modality`: scanning budget; raise it along with sample counts.
- Source `files`: add pinned shard paths or globs. Audio already lists all
  train-clean-100 shards; text starts with one DCLM shard.
- `records_per_shard`: maximum JSONL records (and text documents) per output shard.
- `sampling_weights`: relative probabilities of the enabled training modalities.
- `batch`: sequence length, image size, audio duration and video sampling limits.
  Optional `max_text_chars` defaults to 20000 for preparation.

Pass the copied config with `--config` and choose a **new output directory**.
Changing limits/stages in an existing output is rejected rather than mixing old
and new files. Selection is deterministic source order up to the requested count;
the runtime sampler supplies shuffling/mixture. Small subsets are not statistically
representative of the full sources (e.g. early LibriSpeech records share a speaker).

Expansion has source and implementation limits. The configured MSVD source has
1200 training clips and 100 validation clips before our filters. Increasing a
quota beyond the available usable records fails; a larger/new-format source
needs a source adapter in `records`, not just a new repository name. The current
corpus covers captioning/transcription, not OCR, multilingual speech, sound-event
understanding or aligned audio/video. Source selection for those tasks is future
work alongside their model paths.

This path supports incremental development-scale expansion. Preparation is
single-process and publication uses one archive per bundle; JSONL indices and
deduplication state still grow with sample count. Multi-terabyte production
preparation, distributed streaming/staging and throughput have not been validated.

Source data is read incrementally. Parquet downloads are limited to needed row
groups, which may contain more records than requested. DCLM caches a bounded
document prefix per selected shard. Source cache files have hashes and are reused
after interruption. `--cache` chooses their location. To rebuild without network:

```bash
HF_HUB_OFFLINE=1 python examples/prepare_omni_data/qwen3-omni-training/script.py prepare \
  --offline --stage video --output workspace/datasets/omni-training-rebuilt
```

An interrupted output has no usable completion marker/checksum pair. Rerunning
reuses checked source/media files and deterministically rebuilds manifests; it
does not append duplicate samples. Offline mode fails on an uncached source.
Expected media decode/length failures go to `rejected.jsonl`, then preparation
continues until the quota is met. Network/cache-integrity failures stop the run.
Exhausting the scan budget raises an error, never a false success with fewer data.

## Outputs and runtime use

```text
bundle.json              # completed catalog, pins, counts, package versions
recipe.json              # effective recipe and enabled stage
checksums.sha256         # catalog, manifests, tokens, rejection log and accepted media
rejected.jsonl           # skipped IDs and reasons
train/<modality>-*.jsonl
validation/<modality>-*.jsonl
media/<modality>/<sha256>.*
tokens/train/*.bin       # existing dense text loader format
tokens/validation/*.bin  # held-out text, in a separate directory
```

The JSONL records retain source IDs, grouping, split, media hash and validated
input/target token counts. Media paths are relative to the bundle root, so moving
the whole bundle does not break them. Preparation runs the actual Omni processor
on **every accepted record**, rejecting overlong expanded sequences rather than
truncating media tokens. It does not persist huge precomputed pixel/audio tensors.

Text `.bin` output reuses `Worker`/`Writer` and the pinned tokenizer's EOS. Point
existing text pretraining at **`tokens/train` only**; never the parent containing
both train and validation. Dense `.bin` retains the existing cross-document
next-token behavior. Multimodal JSONL uses separate padded sequences with media
targets masked out. These are deliberately different data layouts.
Both use the real Omni tokenizer: a reduced training model must retain its
152064-entry vocabulary. A 256-entry synthetic comparison fixture cannot consume
these IDs; changing model depth/width does not require changing the tokenizer.

Each JSONL record has an ID, paired text and ordered media, for example:

```json
{"id":"image-1","text":"A description of the picture.","media":[{"type":"image","path":"media/image/example.jpg"}]}
```

Empty `media` denotes text-only; a sample may contain multiple media items. The
collator returns `Qwen3OmniBatch` with sample IDs, HF-named model inputs, shifted
labels and media timing/shape information. Sequences are a tokenizer EOS boundary,
media prefix, paired text, and final EOS. Only paired text/EOS are loss targets;
media wrappers, placeholders and padding have labels `-100`. For an HF reference,
pass model inputs without these labels and compute CE externally to avoid a second
label shift. The native model must handle padding and construct multimodal positions
from the retained grids and timing.

`create_omni_dataloader` in `pithtrain/modules/qwen3_omni_data.py` reads a bundle,
selects a stage/split and returns `Qwen3OmniBatch` values. It checks catalog and
selected manifest hashes. JSONL shards are indexed by byte offset; the runtime
keeps offsets/indices, not all document text, in memory.

Training sampling is **weighted with replacement**. It is deterministic for
seed, epoch and global sample position. `rank`/`world_size` mean **data-parallel**
rank/size, not EP or PP. `start_sample` is the globally consumed sample count;
record that count after completed steps, not a prefetched iterator position.
Each DP rank receives different draw positions; replacement can intentionally
repeat a record. Set `num_samples` to a multiple of data world size. Validation
visits each selected record once and currently uses one data rank.

```bash
python examples/prepare_omni_data/qwen3-omni-training/script.py check \
  --data workspace/datasets/omni-training \
  --report workspace/omni-training-batches.json
```

This command verifies bundle hashes and exercises train/validation loaders; it
does not train a model. The [training entry point](../../pretrain_lm/omni-data/README.md)
now connects these stages to `pretrain_lm`, including device transfer, Microbatch
context in normal/overlapped DualPipeV execution, global valid-target normalization
and checkpointed consumption. Text retains the existing dense `.bin` loader.
Media requires micro-batch size 1, CP=1 and a registered model that declares support
for the selected modalities. Native Omni encoders/positions are not implemented;
existing text-only models reject media stages before model allocation.

For the existing **text** distributed batch path, use the same bundle's train-only
token export (requires GPUs and the small processor files cached by preparation):

```bash
torchrun --standalone --nproc-per-node=1 tests/test_pretrain_data.py \
  --dataset workspace/datasets/omni-training
```

The checker accepts `--sequence-length`, `--global-batch-size`, `--micro-batch-size`,
`--steps`, and `--pp/--cp/--ep`. This only verifies data partitioning, not an Omni
model update. The former text-only smoke and five-fixture download recipes have
been consolidated here; unit-test media fixtures stay inside the tests.

## Orchard storage

Prepare a small bundle in WSL; keep durable versions in the user's GCS bucket.
The source cache is reproducible working data and is not included in publication.
Archives contain only checksum-listed files, have deterministic metadata, and
use their SHA-256 as the GCS object name. Publishing never overwrites an object.

```bash
python examples/prepare_omni_data/qwen3-omni-training/script.py publish \
  --data workspace/datasets/omni-training \
  --prefix gs://YOUR_BUCKET/pith-train/datasets/omni-training
```

Inside an **allocated compute job**, stage the returned `.tar` URI to its local
SSD. Run staging once per node before starting that node's training processes:

```bash
test -n "${SLURM_JOB_ID:-}"  # do not stage corpora on the login node
python examples/prepare_omni_data/qwen3-omni-training/script.py stage \
  --uri gs://YOUR_BUCKET/pith-train/datasets/omni-training/ARCHIVE_SHA256.tar \
  --output /tmp/omni-data-$SLURM_JOB_ID
```

Staging checks the archive SHA-256, safely extracts regular files, then checks
every payload hash before exposing the final directory. Existing destinations
are never overwritten. Keep source caches and larger preparation output on
allocated-node `/tmp` too; do not put corpora in `/home` or shared `/project`.
GCS publishing/staging uses the already-authenticated `gcloud` CLI. The repo
contains no account identity or credentials. Pure media transfer/verification
does not require a model download.
