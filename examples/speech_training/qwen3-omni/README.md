# Qwen3-Omni chat and speech supervision

This reader prepares local conversations for native Thinker SFT and native
Thinker → Talker/code predictor teacher forcing. It uses the official processor
for input images, audio and video. Output speech codes are a separate, explicitly
identified external codec dataset. Input audio features are never used as codes.

The pinned reference is `Qwen/Qwen3-Omni-30B-A3B-Instruct`, revision
`26291f793822fb6be9555850f06dfe95f2d7e695`, with Transformers 5.17.0. The preparation
code downloads nothing. Cache the processor metadata separately before use.

## Local JSONL

Each line has a unique `id`, `messages`, and an optional `codec` manifest. Paths
must point to files under the JSONL directory. Text-only messages use strings;
user messages may use a list of `text`, `image`, `audio`, or `video` items.

```json
{"id":"sample-1","messages":[{"role":"system","content":"Answer concisely."},{"role":"user","content":[{"type":"image","path":"image.png"},{"type":"text","text":"What is shown?"}]},{"role":"assistant","content":"A red square."}]}
```

The supported template is the official ChatML role structure. The final message
must be assistant text. Only user messages may contain media, which preserves the
Talker user-media position contract. Tools, media in system/assistant messages,
and literal reserved control tokens in text fail explicitly. This adapter emits
one unpadded conversation per microbatch; it does not truncate expanded media or
pack conversations. Existing audio resampling and video timing limits are reused.

Assistant spans are proved against the **final processor-expanded tokens**:
prefixes are taken from the complete rendered template, processed with the same
preceding media, and compared token by token. This retains the official template's
context-dependent thinking text. It does not rely on an absent assistant-mask tag
or assume that tokenizing text before media expansion preserves offsets.

The reader removes only the verified template newline after the final `im_end`,
then shifts once. `input_ids` excludes that final `im_end`; `labels` predicts it.
User/system/header tokens have labels `-100`. Earlier and final assistant text is
supervised. `prompt_lengths` points immediately after the final three-token
assistant header. Neither the SFT objective nor the speech task should shift again.

## External codec labels

For speech training, add the following `codec` object to the same sample (replace
all example identifiers/hashes with the actual encoder and archive provenance):

```json
{
  "path": "codes/sample-1.npz",
  "sha256": "<actual archive SHA256>",
  "sample_id": "sample-1",
  "assistant_index": 2,
  "transcript": "A red square.",
  "source": {"model_id": "<external encoder>", "revision": "<immutable revision>"},
  "num_codebooks": 16,
  "codebook_size": 2048,
  "num_frames": 3,
  "alignment": {"sample_rate": 24000, "frame_hop_samples": 1920, "start_sample": 0, "num_samples": 5700}
}
```

Write NPZ with `codes` as **int64 `[time,16]`** and `codebook_mask` as **bool with
the same shape**. Valid IDs are 0…2047; EOS is supplied by teacher forcing and must
not be stored as a code. Every active frame must contain all 16 known, valid code
IDs, even where `codebook_mask` is false: the mask controls loss only, and those
codes remain inputs to later codebooks and subsequent frames. Unknown/negative
IDs are rejected rather than replaced with fabricated zeros. `num_frames=0`
with `num_samples=0` represents immediate
EOS. The collator returns `codes[B,T,16]`, `code_lengths[B]` and
`codebook_mask[B,T,16]`; code2wav's decoding API uses `[B,16,T]` instead.

Optional `waveform` is finite float32 `[num_samples]`, already cropped to the
declared source segment. `start_sample` refers to its location in the original
waveform. This source alignment is not a claim that every external encoder's
padded frame count decodes to precisely the source length. Any waveform loss and
target/decoder length policy must be configured explicitly by the trainer.

NPZ uses `allow_pickle=False`, a pinned SHA256, a 64 MiB compressed and unpacked
limit, exact required arrays, and explicit shape/dtype/range checks. Do not put
raw audio, feature tensors, Python objects, or downloaded URLs in `codes`.
`ExternalRVQEncoder` defines a preparation protocol; no encoder is silently
selected or invoked while training.

The [Qwen3-Omni report](https://arxiv.org/html/2509.17765v1) describes RVQ speech
codes, Talker training and the causal waveform decoder. The pinned checkpoint and
inspected public recipe do not provide a codec encoder or a concrete code2wav
training loss. This reader therefore does not label waveform L1 or an invented
encoder as the official training recipe.

## Use with the native tasks

```python
from transformers import AutoConfig, Qwen3OmniMoeProcessor
from pithtrain.modules.qwen3_omni_speech_data import (
    Qwen3OmniSpeechDataset, Qwen3OmniSpeechCollator,
    Qwen3OmniSpeechSFTData, Qwen3OmniSpeechData,
)

name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
options = dict(revision="26291f793822fb6be9555850f06dfe95f2d7e695", local_files_only=True)
processor = Qwen3OmniMoeProcessor.from_pretrained(name, **options)
config = AutoConfig.from_pretrained(name, **options)
dataset = Qwen3OmniSpeechDataset("data/train.jsonl")
collator = Qwen3OmniSpeechCollator(processor, config.thinker_config, max_length=4096)

# Native SFT: initialize the training/distributed contexts, then use actual PP.
from pithtrain.contexts import distributed

# DualPipeV needs 2 * PP chunks when PP > 1; its PP=1 minimum is one.
# Two chunks per rank is a valid choice for every supported PP size.
text_data = Qwen3OmniSpeechSFTData(
    dataset, collator, microbatches_per_scan=2 * distributed.pp_size,
)
# Set SFTLMCfg.max_batches <= text_data.total_scans for this finite pass,
# then call pithtrain.tasks.sft_lm.launch(cfg, text_data, ...).
# launch owns get_batch and commit_update; do not pre-consume scan zero.
# Each returned Microbatch retains shifted labels, model_context and stage-0 media_inputs.

# Pass to pithtrain.tasks.omni_speech.launch with an initialized native speech model.
# This reader requires the explicit codec manifest in every selected sample.
speech_data = Qwen3OmniSpeechData(dataset, collator)
speech_batches = speech_data.get_batch(scan_index=0, device="cpu")
batch = speech_batches[0]
# model(**{key: batch[key] for key in (
#     "input_ids", "model_context", "codes", "code_lengths", "prompt_lengths",
#     "codebook_mask", "position_kwargs",
# )})
```

SFT groups `microbatches_per_scan` consecutive conversations into one scan. The
default is one, suitable for PP=1; for PP>1 configure at least `2 * PP`, matching
the native `DualPipeV.min_chunks` schedule guard. Each microbatch remains B=1 and
has its own labels and media context. The finite manifest length must be divisible
by the group size: an incomplete final group raises during reader construction,
before training. It is never silently dropped, repeated or padded. Split or extend
the manifest explicitly if necessary. `total_scans` is the number of complete
groups, and requesting a scan after it raises exhaustion without advancing state.
The speech adapter retains its original single-conversation scan contract.

Both adapters consume explicit zero-based `scan_index`, independently of update
count. An entire group's preparation and device transfer must succeed before one
`scanned_batches` increment; call
`commit_update(scan_index=..., update_index=...)` only after a successful optimizer
update. A globally zero-target scan has no commit but is still scanned. Duplicate
commits, changed manifests, changed group sizes and inconsistent resume counters
are rejected. Cursor checkpoints include `microbatches_per_scan`; older cursor
records lacking that field require an explicit migration and are not guessed.
Checkpoint `data.state_dict()` **together with** task state, model, optimizer,
scheduler and RNG; load both data and task counters before the next scan. The
reader does not infer scan position from successful updates or redo skipped scans.
Datasets are finite, ordered and immutable during a run. Distributed callers must
select identical samples for model-parallel peers and explicitly partition data
across data replicas; this reader does not invent a rank/sharding policy.

CPU validation:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m pytest tests/test_omni_speech_data.py -q
```

Processor-dependent tests use the actual cached official tokenizer and media
expansion; without that cache they report a skip. Tiny CPU data checks do not
establish GPU training, released-weight quality or distributed acceptance.
