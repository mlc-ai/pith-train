# Naming, Layout, and Key Conventions

These rules exist because we violated them for a past model and paid the
cost in silent bugs and asymmetric converters. **Apply them on every new
model.** The one-sentence summary:

- **Names and structure**: match HuggingFace exactly.
- **In-memory tensor layout (axis order)**: follow the training-framework
  consensus (TorchTitan + Megatron-LM / TransformerEngine).
- **Canonical keys**: per-expert indexed, no rename from HF, transpose
  only at the `dcp2hf` safetensors boundary.
- **Config values**: thread per-checkpoint knobs through `__init__`;
  only architectural constants stay as literals.
- **Example config**: mirror the upstream shipped `config.json`
  field-by-field, including nested blocks.

## Rule 1 — Mirror HF's names and structure

This has zero performance implication. There is no reason to deviate.

### Class names

Grep HF's `modeling_<model>.py` for `^class ` and use the same spelling:

```bash
# Look in the project's venv
grep "^class " .venv/lib/python*/site-packages/transformers/models/<model>/modeling_<model>.py
```

Typical mapping:

| HF class | Our class (example) |
|----------|---------------------|
| `<Prefix>Attention` | `<Prefix>Attention` |
| `<Prefix>MLP` (the MoE block) | `<Prefix>MLP` |
| `<Prefix>Experts` | `<Prefix>Experts` |
| `<Prefix>TopKRouter` or `<Prefix>Gate` | **exactly** what HF uses |
| `<Prefix>RotaryEmbedding` | `<Prefix>RotaryEmbedding` (even if YaRN internally) |
| `<Prefix>DecoderLayer` | `<Prefix>DecoderLayer` |

**Counter-example (do not do this):** GPT-OSS originally named the class
`GptOssGate` and the attribute `self.mlp.gate`. HF calls them
`GptOssTopKRouter` and `self.mlp.router`. That mismatch required an
asymmetric `hf2dcp` (rename `router → gate`) and broke `dcp2hf`
round-trip. Fix landed later — but only because the original choice
ignored HF.

### Attribute names

Grep HF's `__init__` for `self.<x> = ...`:

```bash
grep -n "self\.[a-zA-Z_][a-zA-Z0-9_]*\s*=" \
    .venv/lib/python*/site-packages/transformers/models/<model>/modeling_<model>.py | head -50
```

The names of submodules become part of the `state_dict` keys. If HF
says `self.router`, the state_dict will have
`layers.0.mlp.router.weight`, and you must match exactly or
`load_state_dict(strict=False)` silently drops the weight.

### Activation math — don't trust `hidden_act` <a id="activation-math"></a>

The `config.hidden_act` string is shorthand, not a spec. A config that
says `"silu"` can be hiding:

- `gate * sigmoid(alpha * gate)` with a non-unit `alpha` (the sigmoid
  approximation of GELU from Hendrycks & Gimpel is `alpha ≈ 1.702`,
  *not* 1.0 — plain SiLU is only recovered at `alpha = 1`).
- A clamp on gate or up before the multiply.
- A bias add inside the expert MLP, before or after the activation.
- An `(up + 1) * glu`-style residual inside the MLP.

All of these must be copied verbatim. A single wrong constant in the
activation trains silently to a higher loss without crashing anything —
it passes phase 3 and phase 4 on random weights, and only surfaces as
gibberish or poor loss when real pretrained weights are loaded.

**Rule:** don't rely on `hidden_act`. Open HF's `<Prefix>MLP.forward`
(and the expert MLP if different) and copy the exact math — coefficients,
clamps, bias adds, residual shapes — into your implementation.

### Fused vs split tensors

If HF stores gate and up as **one fused** tensor `gate_up_proj`, our
model must also have a single `gate_up_proj` parameter. Don't split
into `gate_proj` + `up_proj` — that forces a split+deinterleave in
`hf2dcp` and a re-interleave+merge in `dcp2hf` with high bug risk.

Split the output *at forward time* if you need different post-ops:

```python
gate_up = F.grouped_mm(x, self.gate_up_proj.transpose(-2, -1), offs=offs)
gate = gate_up[:, ::2]   # interleaved
up   = gate_up[:, 1::2]
```

### `.weight` suffix: submodule vs raw Parameter

- `nn.Linear` / `nn.Embedding` / `GroupLinear` → state_dict key ends in
  `.weight`.
- Raw `nn.Parameter` stored on the module itself → state_dict key is
  just the attribute name, NO `.weight`.

GPT-OSS uses raw `nn.Parameter` for experts to get direct control over
the `[E, out, in]` storage layout:

```python
class GptOssExperts(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        # These become state_dict keys WITHOUT a .weight suffix:
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2*intermediate_size, hidden_size))
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(num_experts, 2*intermediate_size))
```

State_dict keys look like `layers.0.mlp.experts.gate_up_proj` — no
`.weight`. Your canonical-key convention (below) must match exactly.

## Rule 2 — Expert weight layout: training-framework consensus

For a given model, check what storage layout the mainstream training
frameworks use *for that model*, and follow them:

```bash
# TorchTitan reference (if it exists for this model)
# github.com/pytorch/torchtitan/blob/main/torchtitan/models/<model>/...

# TransformerEngine GroupedLinear
# github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/grouped_linear.py
```

Both have converged on `[E, out, in]` with a `.transpose(-2, -1)` view
at the `F.grouped_mm` call site:

```python
# Storage
self.gate_up_proj = nn.Parameter(torch.empty(E, 2*intermediate, hidden))   # [E, out, in]
# Forward
out = F.grouped_mm(x, self.gate_up_proj.transpose(-2, -1), offs=offs)      # transposed view
```

Both `GroupLinear` (in `pithtrain/layers/group_linear.py`) and this
raw-Parameter pattern match that convention.

HF's live Parameter is often `[E, in, out]` — the transposed layout —
because HF loads for `F.linear`-style usage. That's fine; the transpose
between `[E, out, in]` (runtime/DCP) and `[E, in, out]` (HF BF16
safetensors) lives in **exactly one place**: a model-specific
converter's `postprocess_canonical` (see `GptOssConverter` in
`pithtrain/tasks/convert_checkpoint/gpt_oss.py`). See `checkpoint.md`.

**Do not pick a different layout without evidence** (a microbench showing
materially faster kernels on our target hardware). In the absence of
evidence, deviating from the consensus is strictly worse: kernel authors
tune for it, and quantized paths (MXFP4, FP8) are usually pre-packed in
the consensus layout.

### Verifying layout correctness after conversion

Norms are invariant under transposition. A norm match does NOT prove
the layout is right. The real check is element-wise:

```python
# After hf2dcp → dcp2hf → transformers.AutoModelForCausalLM.from_pretrained:
rt_state = round_tripped_model.state_dict()
hf_state = hf_model.state_dict()           # load the original HF model fresh

for k in hf_state:
    ours = rt_state[k].float()
    theirs = hf_state[k].float()
    assert ours.shape == theirs.shape, (k, ours.shape, theirs.shape)
    diff = (ours - theirs).abs().max().item()
    assert diff == 0, (k, diff)             # byte-identical is the goal
```

If norms match but element-wise differs, you have a transpose bug
(`.view()` instead of `.transpose()` — see `pitfalls.md`).

## Rule 3 — Canonical DCP keys

The DCP checkpoint format is our internal standard. It differs from HF
in exactly two ways:

1. **Per-expert indexed keys.** HF stacks all experts into one 3-D
   tensor:
   ```
   HF:     model.layers.0.mlp.experts.gate_up_proj           shape [E, ...]
   Ours:   layers.0.mlp.experts.0.gate_up_proj               shape [...]
           layers.0.mlp.experts.1.gate_up_proj
           ...
           layers.0.mlp.experts.E-1.gate_up_proj
   ```
   This lets us filter by EP rank cleanly: rank `r` loads experts
   `[r*per_rank, (r+1)*per_rank)` without re-slicing a huge tensor.

2. **No `model.` prefix.** HF's state_dict keys start with `model.` for
   everything except `lm_head`. Ours drops that prefix in the canonical
   keys and reintroduces it in `dcp2hf`.

Nothing else should differ. In particular:

- **No renaming.** If HF uses `router`, our canonical key uses `router`.
  Do not rename to `gate` in `hf2dcp` and back in `dcp2hf` — the
  asymmetry is bug-prone and our *model code* should already use
  `router` (see Rule 1).
- **No shape transforms beyond the layout transpose at dcp2hf.**

### Converter responsibilities

- `hf2dcp`: `model.layers.0.mlp.experts.gate_up_proj [E, out, in]`
  → split along dim 0 into E per-expert tensors (each `[out, in]`)
  → write under `layers.0.mlp.experts.IDX.gate_up_proj`.
  (Dequant if needed — see `checkpoint.md`.)
- `dcp2hf`: a model-specific converter's `postprocess_canonical`
  stacks per-expert tensors back into `[E, out, in]`, transposes to
  `[E, in, out]` for HF live layout. The generic `dcp2hf` then adds
  `model.` prefix and writes safetensors.

### Round-trip test (non-negotiable)

Do this immediately after writing both directions:

```bash
hf2dcp <hf_snapshot> <dcp_dir>
dcp2hf <dcp_dir> <bf16_snapshot>
# strip quantization_config from bf16_snapshot/config.json first
python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('<bf16_snapshot>', dtype='bfloat16')
# compare m.state_dict() against HF's own BF16 dequant of the original
"
```

See `checkpoint.md` for the full round-trip validation recipe.

## Rule 4 — Thread config values through `__init__` <a id="thread-config"></a>

Any value that appears in HF's `config.json` for the model is a
**per-checkpoint knob**, not an architectural constant — even if every
released checkpoint currently ships the same value. Thread it through
the model init from the config, don't hardcode it as a module-level
literal. Future fine-tunes / derivative releases can and do change
these fields.

### Diff check

Diff the shipped `config.json` against every numeric (and string)
literal in your modeling file. If a literal equals a config field, read
it from the config. If it doesn't, leave a one-line comment naming the
source (paper, reference impl) so the reader knows why it's hardcoded.

```python
# Per-checkpoint knob — threaded from config:
#   Model(config) → DecoderLayer(...) → MLP(...) → Experts(..., swiglu_limit=config.swiglu_limit)
def __init__(self, ..., swiglu_limit: float):
    ...
    self.swiglu_limit = swiglu_limit

# Architectural constant — hardcoded, with a one-line source comment:
SWIGLU_ALPHA = 1.702  # sigmoid approximation of GELU (Hendrycks & Gimpel 2016).
```

Decision rule:

| Is it in `config.json`? | Is it in `<Model>Config.__init__` signature? | Treat as |
|-------------------------|---------------------------------------------|----------|
| Yes | Yes | Per-checkpoint knob — thread through `__init__` |
| No | Yes (optional arg with default) | Per-checkpoint knob — thread through `__init__` |
| No | No | Architectural constant — module literal with source comment |

### Why this matters

"We'll fix it when someone ships a different value" is false economy.
A hardcoded knob breaks silently when a derivative release changes it —
no AttributeError, no shape mismatch, just a higher loss than expected
that only surfaces weeks later in training runs that are hard to bisect.

## Rule 5 — Example configs mirror upstream HF, field-by-field <a id="example-config"></a>

When adding `examples/pretrain_lm/<model>/config.json`, it
must match the upstream HuggingFace `config.json` on every
semantically-meaningful knob — especially the nested blocks
(`rope_scaling`, `quantization_config`, any `*_parameters` dicts, etc.).
A missing field defaults to our code's fallback, which may differ from
HF's fallback, and then weights loaded from the released checkpoint
compute against subtly different numerics.

### The field-by-field diff

```bash
python -c "
import json
hf   = json.load(open('<hf_snapshot>/config.json'))
ours = json.load(open('examples/pretrain_lm/<model>/config.json'))
for k in sorted(set(hf) | set(ours)):
    if hf.get(k) != ours.get(k):
        print(k, 'HF=', hf.get(k), 'ours=', ours.get(k))
"
```

For any deliberate divergence, leave a one-line comment in the model
code explaining why. "We didn't know about this field" is the most
common cause of a loaded checkpoint's initial loss being in the wrong
order of magnitude.

### The three sources of config defaults

"Matches the HF config" is not one check, it's three:

1. **The shipped `config.json`** on the HuggingFace Hub — the released
   artifact's view of the model.
2. **The `<Model>Config.__init__` signature** in
   `transformers/models/<model>/configuration_<model>.py` — the
   defaults HF uses when a field is *absent* from the config.json, and
   when users construct the config programmatically.
3. **The generic fallback path** the modeling code reads through (e.g.
   `modeling_rope_utils.py::_compute_yarn_parameters` does
   `rope_parameters.get("truncate", True)`) — what happens when the
   value isn't in either of the above.

These three can disagree. When they do, defaults for a
**model-specific** class should match the **model-specific** HF default
(source 2), not the generic fallback (source 3). If the modeling code
is generic, match the generic fallback.

### Concrete example — rotary embedding with a YaRN-style `truncate` flag

Some YaRN-style rotary implementations expose a `truncate` flag on
whether to `math.floor`/`math.ceil` the correction-range bounds.

- Shipped `config.json` explicitly sets `rope_scaling.truncate = false`.
- `<Model>Config.__init__` hardcodes `rope_parameters = {..., "truncate": False, ...}`.
- Generic `_compute_yarn_parameters` falls back to `.get("truncate", True)`.

A faithful port:

- The modeling class defaults `truncate=False` (match source 2), so
  programmatic construction matches the published config.
- The example `config.json` carries `rope_scaling.truncate: false`
  explicitly (match source 1), so a loaded config never relies on the
  fallback.
- Correction-range helpers take `truncate` as a parameter and only
  round when it's `True`:
  ```python
  def _yarn_find_correction_range(low_rot, high_rot, dim, base, mpe, truncate):
      low  = _yarn_find_correction_dim(low_rot,  dim, base, mpe)
      high = _yarn_find_correction_dim(high_rot, dim, base, mpe)
      if truncate:
          low, high = math.floor(low), math.ceil(high)
      return max(low, 0), min(high, dim - 1)
  ```

A hardcoded `math.floor` / `math.ceil` is a correctness bug for any
model that ships `truncate=false` — rotary positions drift from the
trained checkpoint's and CE lands in the wrong order of magnitude.

### Concrete checklist for a new model

- [ ] Diff the shipped Hub `config.json` against your example
      `config.json` field-by-field — including every nested block.
- [ ] Open `transformers/models/<model>/configuration_<model>.py` and
      read every default in `__init__`. For every default that differs
      from the generic path the modeling code takes, note which opinion
      you're matching and why.
- [ ] For any layered-default field (rope, dropout, init ranges,
      routing coefficients), make sure the default on your class *and*
      the fallback in any config-reader code are both the model-specific
      opinion. They should not disagree with each other.
- [ ] Sanity-check by constructing your model without any config.json
      (pure programmatic path) and confirm numerics match what HF's
      equivalent programmatic construction would produce.

## Summary table

| Aspect | Follow | Failure mode if ignored |
|--------|--------|------------------------|
| Class names | HF exactly | Reader confusion, bring-up slowdown |
| Attribute names | HF exactly | state_dict key mismatch, silent zero loading |
| Fused vs split tensors | HF exactly | Bug-prone split/merge in converter |
| `.weight` suffix | Match model's actual state_dict() output | Missing keys, silent random init |
| Activation math | Read HF's MLP forward verbatim | Silently higher training loss |
| Per-checkpoint knobs | Thread through `__init__` from config | Derivative checkpoints break silently |
| Example `config.json` | Mirror upstream HF field-by-field, nested blocks included | Rotary / numerics drift vs trained weights |
| Class `__init__` defaults | Match model-specific HF `__init__` default, not generic fallback | Programmatic construction disagrees with loaded config |
| Expert weight in-memory layout | TorchTitan / TE consensus (`[E, out, in]`) | Slower kernels, converter drift |
| Canonical DCP keys | Per-expert indexed, no rename | Asymmetric converter, round-trip failure |
| Where the `[E, out, in] ↔ [E, in, out]` transpose lives | Only at `dcp2hf` | Scrambled weights, inference gibberish |

## Quick diagnostic commands

```bash
# Class name inventory in HF and ours (align them first):
grep "^class " .venv/lib/python*/site-packages/transformers/models/<model>/modeling_<model>.py
grep "^class " pithtrain/models/<model>.py

# Attribute name inventory:
grep -n "self\.[a-zA-Z_][a-zA-Z0-9_]*\s*=" \
    .venv/lib/python*/site-packages/transformers/models/<model>/modeling_<model>.py | head -50

# Actual safetensors shapes (never trust comments):
python -c "
from safetensors import safe_open
with safe_open('<shard>', framework='pt', device='cpu') as f:
    for k in sorted(f.keys()):
        t = f.get_tensor(k)
        print(k, tuple(t.shape), t.dtype)
"
```
