# Complete Qwen3-Omni Thinker conversion

This explicit CPU converter preserves the text decoder, image/video vision
encoder and DeepStack mergers, and audio encoder/projectors. It targets the
Qwen3-Omni Thinker architecture and Transformers 5.17 fused expert layout.
The existing `qwen3-omni-text` command remains text-only.

```bash
python examples/convert_checkpoint/qwen3-omni-thinker/script.py hf2dcp HF DCP
python examples/convert_checkpoint/qwen3-omni-thinker/script.py dcp2hf DCP HF_EXPORT
```

Both standalone Thinker keys (`--prefix ''`) and full Omni Thinker namespaces
(`--prefix thinker.`, the default) are supported. Imports accept released split
or live fused experts. Exports default to fused; `--expert-layout split`
selects the released indexed Linear layout. Every tensor key, shape and floating
dtype is validated before weights are read. Missing/extra media weights fail.

A full Omni snapshot also contains Talker/code2wav weights. `--thinker-only`
explicitly excludes those components and records every omitted key in
`qwen3_omni_thinker.json`. Unknown components are never silently ignored.
This does not convert Talker/code2wav.

Native DCP export needs either the conversion sidecar or
`--thinker-config PATH/config.json`; native strict checkpoints already carry
the complete configuration. Text-only and incomplete single-encoder checkpoints
are rejected in this complete-Thinker mode.

Strict checkpoints support the same fixed PP/EP/DP topology and CP=1 contract
as the existing strict text bridge. Export writes a `pithtrain-resume` companion
with optimizer/scheduler/RNG/data state and model-shard digests. Reimport with
`--restore-resume` reconstructs the original strict shards only if every HF
weight is unchanged:

```bash
python examples/convert_checkpoint/qwen3-omni-thinker/script.py dcp2hf STRICT HF_EXPORT
python examples/convert_checkpoint/qwen3-omni-thinker/script.py hf2dcp HF_EXPORT RESTORED --restore-resume
```

The CPU tests cover all-weight exact file roundtrips, strict HF loading and
fresh-process state restoration on tiny models. They do not establish full-size
released-weight numerical quality or GPU training correctness. Conversion reads
the selected weights into CPU memory; plan RAM for the full Thinker plus fused
expert output copies. Quantized checkpoints are rejected.
