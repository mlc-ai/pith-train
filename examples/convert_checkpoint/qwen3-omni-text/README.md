# Omni Thinker text checkpoint conversion

This CPU converter covers the text decoder and LM head at PR101 `e6d1889`.
It does not implement media encoders, DeepStack, Talker, optimizer/RNG/data
recovery, or a distributed update. There are no downloads in the entrypoint.

```bash
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD"
python examples/convert_checkpoint/qwen3-omni-text/script.py \
  hf2dcp /path/to/local/hf /path/to/output/torch-dcp/00000000 --text-only
python examples/convert_checkpoint/qwen3-omni-text/script.py \
  dcp2hf /path/to/output/torch-dcp/00000000 /path/to/hf-text-export \
  --expert-layout fused
```

Destinations must be empty. A full Omni or standalone Thinker checkpoint needs
`--text-only` to exclude the known `thinker.audio_tower.*`, `thinker.visual.*`,
`talker.*`, `code2wav.*` namespaces (or standalone `audio_tower.*`/`visual.*`).
Every excluded key is logged and retained in `qwen3_omni_text.json` beside DCP.
Unknown keys, including unknown keys inside the text namespace, always fail.
The converter validates the index against every shard header, then checks all
text keys/shapes before reading text tensor data. It never reads media tensors.

The exact default namespace is `thinker.model.*` plus `thinker.lm_head.weight`.
Use `--prefix ''` for a standalone `model.*` plus `lm_head.weight` checkpoint.
Prefixes are explicit; arbitrary prefixes and mixed representations are errors.

| Representation | Expert weights |
| --- | --- |
| Official released HF (`--expert-layout split`) | `experts.E.gate_proj.weight [I,H]`, `up_proj.weight [I,H]`, `down_proj.weight [H,I]` |
| Transformers 5.17 HF (`--expert-layout fused`) | `experts.gate_up_proj [E,2I,H]`, `experts.down_proj [E,H,I]` |
| Native runtime | Same fused Parameters, with no `.weight` suffix |
| PithTrain canonical DCP | `experts.E.gate_up_proj [2I,H]`, `experts.E.down_proj [H,I]` |

Gate rows precede up rows. There is no transpose. Ordinary attention, Q/K norms,
router, dense MLP, embedding, final norm and LM-head tensors preserve their
values and dtype. Mixed gate/up or grouped expert dtypes fail instead of promotion.
Split gate/up tensors may reside in different shards. Tied embeddings and
quantized weights are unsupported. Missing, unexpected and wrong-shape keys
are reported together; there is no permissive `strict=False` conversion.

The DCP contains `app.model` only and is a weight initialization checkpoint,
not a training resume checkpoint. It uses the existing checkpoint.py canonical
keys and localized expert stacking. A native training save does not automatically
carry the conversion sidecar. Export it with the **matching training config**:

```bash
python examples/convert_checkpoint/qwen3-omni-text/script.py \
  dcp2hf /path/to/native/torch-dcp/00000002 /path/to/hf-text-export \
  --text-config tests/configs/qwen3_omni_text/tiny.json
```

An unidentified fused-expert DCP fails without the sidecar or explicit config.
Dense-only canonical keys are indistinguishable from other model families, so
the explicit config is essential for native dense-only Omni export as well. The
Omni CLI enforces this for every native DCP without a sidecar.
When both config and sidecar are present, differing semantics (including RoPE
and router settings) fail. Exported `config.json` is the text config, and the
export sidecar explicitly labels the scope as `thinker_text_only`.

The export is a **text weight artifact**, not a complete multimodal
`AutoModel.from_pretrained` directory. Load its text state and LM head strictly,
as the tests do with `Qwen3OmniMoeThinkerTextModel` plus an untied linear head.
Do not load it into a full Thinker and accept randomly initialized media towers.
Tokenizers/processors and other source assets are not copied.

## Validation and limits

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  python -m pytest tests/test_qwen3_omni_checkpoint.py -q
```

Tests use actual HF 5.17 eager attention/experts on CPU with asymmetric synthetic
dimensions (37-token vocabulary). They serialize HF -> DCP -> HF, exercise the
real checkpoint.py localization/canonicalization functions, require **zero**
tensor and prediction error, and cover both prefixes, split/fused, BF16/FP32,
dense layers, attention bias and unnormalized routing. A HF module with native's
parameter structure exercises runtime key packing; native CUDA forward/backward
is not run or simulated. Negative tests cover dropped/extra/reshaped parameters,
expert IDs, prefixes, dtype mismatches, mixed layouts, shard/index corruption,
duplicate tensors and serialized JSON keys (including raw safetensors headers),
media exclusion, config conflicts and nonempty destinations. Header/DCP dtype
checks precede tensor reads/allocations; both export layouts reject inconsistent
expert dtypes. Occupied destinations fail before payload loading.
Generic Qwen plus existing GPT-OSS/Qwen3.5 file-conversion tests guard dispatcher
and shared-writer compatibility. The expanded CPU suite has 67 passing cases.
See [validation preparation](VALIDATION.md) for six explicitly gated GPU cases
and concrete PR100 strict-recovery prerequisites; those GPU/resume checks have
not been executed.

The full-size test constructs the official HF text architecture on **meta** and
checks 531 fused HF keys <-> 12,723 canonical keys <-> 18,867 released split keys.
This is shape coverage, not full numerical execution.

On 2026-10-08, a separate read-only audit checked all 15 official safetensors
headers at revision `26291f793822fb6be9555850f06dfe95f2d7e695`. All 18,867 text
keys/shapes matched, with 30,532,646,912 BF16 text parameters. The 9,143 remaining
keys belong to excluded components. It read 3,662,112 header bytes plus the JSON
config/index, **zero tensor payload bytes**. Local hashes and results are in
`workspace/omni-converter-audit/coverage.json`; this scratch evidence is not an
offline test dependency.

Full text weights alone are about 56.9 GiB in BF16. Conversion currently holds
CPU dictionaries and allocates fused/stacked copies, so peak RAM is greater than
one weight copy; budget multiple copies and measure on a suitable host before
attempting a real full conversion. Each persistent HF or DCP text copy is also
about 56.9 GiB before format overhead. No full allocation, download, prediction,
optimizer update or throughput result is claimed here.

Sources: [official pinned config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/26291f793822fb6be9555850f06dfe95f2d7e695/config.json),
[official index](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/26291f793822fb6be9555850f06dfe95f2d7e695/model.safetensors.index.json),
[Transformers 5.17 source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py).

## Week 3 integration after this baseline

1. Integrate PR100's existing configurable train/validation recipe, staged loader,
   sampler/context and checkpoint wiring with the validated PR101 baseline.
   PR100 implemented the scalable data entrypoint ahead of Week 3. **Do not
   rewrite the loader or restore a separate 128-document recipe.** Preserve the
   local PR100 `24529b3` work and the PR101 validation checkout. Resolve their
   actual commit graph during integration, not by overwriting either checkout.
2. First close the packed-gradient investigation running in task `01a11a21`: an
   isolated FA4 SM120/GQA dK/dV offset fix passed one packed comparison, but a
   separate length-64 LSE anomaly remains under diagnosis. This is not blanket
   model-path acceptance; the original repo/dependencies have not been modified.
   On coordinated GPU resources compare CP=1/BF16 sequential reference with
   PP=1/EP=1, PP=2, EP=2, then PP+EP, using the same global batch. Check every
   expert's parameters/gradients, outputs, loss and one optimizer update. CPU
   checkpoint layout tests do not establish distributed arithmetic correctness.
3. Build strict resume on PR100's public training/data path: compare uninterrupted
   step 3 against two steps + save + a new-process restart + step 3. Require the
   next sample IDs/tokens/labels, model, optimizer, scheduler, sampler/data cursor,
   and required RNG states. Missing resume components must fail, whereas explicit
   HF weight initialization is a separate mode. Do not count the existing
   checkpoint.py model-only fallback as proof of strict recovery.
4. Record GPU/backend tolerances and repeated-run noise independently from this
   exact CPU tensor roundtrip. Keep unrun topologies and full numerical validation
   pending. Media encoders and DeepStack remain later-week work.

Suggested local correction for the Notion weekly-plan/index and Week 3 source:
“PR100 已提前实现可扩展数据入口、train/validation、staging 与公共恢复接线。
Week 3 复用这些实现，接原生 Omni 并验证 PP/EP 单次更新与严格恢复；不重写 loader。
本次文本 converter CPU/tiny 往返已独立验收；full 仅 key/shape 覆盖，分布式与完整恢复待验收。”
The Notion pages were read but not edited under this task's local-only boundary.
