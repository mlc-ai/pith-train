# Complete Qwen3-Omni checkpoint conversion

This local CPU converter supports the exact Qwen3-Omni Thinker, Talker (including
both projections, shared experts, and residual-code predictor), and code2wav
weight trees. It neither downloads weights nor launches training. Select a scope
explicitly; an `all` checkpoint must contain every component.

```bash
PYTHONPATH=. python examples/convert_checkpoint/qwen3-omni-full/script.py \
  hf2dcp /local/omni-hf /local/omni-dcp --components all
PYTHONPATH=. python examples/convert_checkpoint/qwen3-omni-full/script.py \
  dcp2hf /local/omni-dcp /local/omni-export --components all --expert-layout fused
```

The destinations must be empty. Native model-only DCP export requires `--config`
unless `qwen3_omni_components.json` provides the exact original configuration.
The sidecar preserves the original JSON, selected scope, expert layouts and every
excluded key. Whole-model export preserves the full configuration, including
generation token IDs. A single selected component exports its component config
and keeps the original full config in the sidecar.
HF 5.17's serialized Talker config may omit/empty `model_type` and use
`num_local_experts` instead of `num_experts`. Explicit component selection accepts
that official representation, rejects conflicting aliases, and preserves its JSON.

`--components thinker|talker|code2wav` explicitly selects one complete component
and reports all excluded keys. It never accepts a partial trunk/head as a complete
Talker. Unknown keys, missing weights, wrong shapes, non-floating weights, mixed
expert dtypes, invalid configs, and index/header disagreements fail validation.
Every shard header is checked before any selected tensor payload is read.

Both released indexed Linear experts (`--expert-layout split`) and Transformers
5.17 fused experts (`--expert-layout fused`) are supported. Thinker and Talker
input layouts are identified independently. Gate/up rows are concatenated in
that order without transposition; shared experts and the dense code predictor
retain their original layout. Canonical full checkpoints use the three explicit
`thinker.`, `talker.`, `code2wav.` namespaces and indexed routed experts. The
`native_key_map`, `native_to_canonical`, and `canonical_to_native` helpers specify
the native Thinker text-prefix difference explicitly.

`--namespace component` retains the released component prefix. For a standalone
single-component HF model use `--namespace standalone`. Full `all` exports load
through `Qwen3OmniMoeForConditionalGeneration.from_pretrained`. The HF 5.17
standalone Code2Wav class needs its concrete `Qwen3OmniMoeCode2WavConfig` supplied
explicitly; loading a component-prefixed export additionally needs its public
`key_mapping={r"^code2wav\.": ""}` option. Standalone export needs no key mapping.
Standalone split Talker exports can be read by this converter and converted to
fused before using the HF standalone loader; its split-expert conversion alias
is only registered for the composite Omni class in Transformers 5.17.

This format contains model initialization weights. Optimizer/RNG/data state is
not implied: non-model entries in an ordinary DCP are reported, and strict resume
containers/companions are rejected. The existing text/Thinker strict-resume
commands remain separate. Full-model distributed resume needs the full training
topology and optimizer contract and separate validation.

CPU validation:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \
  python -m pytest tests/test_omni_full_converter.py -q
```

Tests use real tiny HF modules, safetensors, DCP, standard HF loading, exact tensor
and prediction comparisons, and the existing native checkpoint stack/unstack
functions. The released schema test covers all 28,010 keys (19,743 Thinker,
8,037 Talker, 230 code2wav) from public headers at revision
`26291f793822fb6be9555850f06dfe95f2d7e695`. Header coverage does not claim that the
released tensor payload, distributed training, or GPU numerics were validated.
