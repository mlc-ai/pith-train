# Known Pitfalls

Every entry here is a real bug that cost time on a previous
integration. Read this file proactively — not just after failure. Most
of these have silent failure modes, so a clean-looking test can hide
one of them.

## NaN-padding in `F.grouped_mm` (affects any MoE with biased experts)

`scatter_for_grouped_gemm` over-allocates output rows to a multiple of
`_GEMM_ALLOC_ALIGNMENT = 1024` for caching-allocator efficiency.
`F.grouped_mm(x, W, offs=...)` writes only rows `[0, offs[-1])`; rows
beyond that are uninitialised and can come back as NaN.

Without biases, NaN rows get discarded by
`outs[reverse_shuffle_idxs]` and nothing observable breaks. With
biases + elementwise activations:

```python
gate = grouped_mm(x, W_gate) + bias[group_ids]
glu  = gate * sigmoid(α * gate)           # NaN rows propagate into bias
up   = grouped_mm(x, W_up) + bias[group_ids]
activated = (up + 1) * glu
```

During backward, `0 * NaN = NaN` poisons gradients. The symptom is
tiny bias gradients (`~2.5e-5` local) vs huge reference gradients
(`~2.4e+1`) — a 6-order-of-magnitude discrepancy, not bf16 noise.

### Fix

Truncate expert input to `sum(ks)` before the matmul:

```python
def forward(self, x, grouped_mm_offs, ks=None, ks_tensor=None):
    if ks is not None:
        actual_m = sum(ks)
        if actual_m < x.shape[0]:
            x = x[:actual_m]
    # ... now the grouped GEMM only touches valid rows ...
```

`ks` is a Python list; `sum(ks)` is a cheap Python int. No GPU sync.

## `.view()` vs `.transpose()` — never use `view` to reorder axes

`.view()` reinterprets flat memory in row-major order. It does NOT
permute axes. `.view(E, B, A)` on a `[E, A, B]` tensor gives you
scrambled data, not swapped axes.

```python
# WRONG — scrambles the memory
flat = _dequantize_mxfp4(blocks, scales)   # [E, out, in]
# flat.view(E, in, out)                     # <-- this does NOT transpose

# RIGHT — actually swaps the axes
flat.transpose(-2, -1).contiguous()        # [E, in, out]
```

### When this bit us

MXFP4 dequant came out as `[E, 2*inter, hidden]`. Original code did
`.view(E, hidden, 2*inter)` to "match" a guess about HF's layout.
Every expert weight was scrambled. FSDP gradient tests (which use
random weights) all passed. Inference produced gibberish. Days lost.

### How to catch it early

After any conversion, compare **element-wise** (not just norms) against
HF's live tensor:

```python
diff = (ours - theirs).abs().max().item()
assert diff == 0
```

Norms are invariant under transposition. Norm-only checks miss this
entire bug class.

## Silent-zero experts (fill_weights miss)

When experts are stored as raw `nn.Parameter` on the module itself
(GPT-OSS pattern — used to control `[E, out, in]` layout directly),
`tests/test_fsdp.py`'s `fill_weights` doesn't reach them unless there
is an `isinstance(module, <Model>Experts)` branch. The parameter stays
at its `torch.empty()` state, which on our system is literally zero
(not random garbage, not NaN — zero).

### Symptom

- Every MoE layer emits
  `[warn] Parameter ... has all-zero gradient, skipping`.
- `mlp(x)` is exactly zero (not "small" — zero).
- Residual stream passes through unchanged.
- The test passes trivially because *nothing* is learning.

### Fix

Add a branch in `fill_weights`:

```python
elif isinstance(module, <Model>Experts):
    nn.init.xavier_uniform_(module.gate_up_proj, gain=1.0)
    nn.init.xavier_uniform_(module.down_proj, gain=1.0)
    # initialise any other raw Parameters on the experts module here
```

Production training (`pithtrain/modules/training.py::init_weights`)
doesn't have this bug — it walks `named_parameters()` and dispatches
by name substring. The FSDP test's `fill_weights` is class-dispatched
and must be updated per model.

### Diagnostic

If you see "all-zero gradient" warnings for every MoE param in every
layer, register a forward hook on `layer.mlp` and print its output
norm. Zero everywhere → you missed fill_weights. Five-minute fix vs
hours of chasing the wrong lead.

## `shard_experts` fallback misclassifies the router

`shard_experts` in `tests/test_fsdp.py` detects the experts module
via `GroupLinear` children. When experts are raw `nn.Parameter`, there's
a fallback. **The fallback must be gated on a distinctive weight name,
not `num_experts` alone.** The router has `num_experts`,
`weight.shape[0] == num_experts`, and often `bias.shape[0] == num_experts`
— all of which would *look* like an experts module to a naive
fallback.

### What happens if the router gets sharded

Every rank ends up with only `num_experts / ep_size` entries in the
routing table, so most expert IDs route to a rank that doesn't own
that expert — routing collapses, outputs become garbage.

### Fix

```python
# inside shard_experts
gu = getattr(model, "gate_up_proj", None)
if isinstance(gu, nn.Parameter):
    num_experts = getattr(model, "num_experts", None)
# — the router doesn't have a gate_up_proj, so it's never matched here
```

Extend the gate condition with your model's distinctive expert-weight
name if it's not `gate_up_proj`.

## Stage-record copy — skipping stages 2 and 4

`Model.forward` builds a fresh `IntermediateTensorsLayer` and copies
each stage record into the pre-allocated slot. Stages 1, 3, 5 have
`.args`, `.ctx`, `.outs`. Stages 2 and 4 have `.ctx` only.

### Wrong

```python
for field in fields(layer_record):
    src_rec = getattr(layer_record, field.name)
    if hasattr(src_rec, 'args'):      # <-- SKIPS STAGE 2 AND STAGE 4
        for rf in fields(src_rec):
            setattr(getattr(dst, field.name), rf.name, getattr(src_rec, rf.name))
```

### Right

```python
for field in fields(layer_record):
    src_rec = getattr(layer_record, field.name)
    dst_rec = getattr(dst, field.name)
    for rf in fields(src_rec):
        setattr(dst_rec, rf.name, getattr(src_rec, rf.name))
```

Iterate **every field of every record** — don't special-case.

### Symptom

Confusing "invalid gradient shape" error in stage 4 backward. The
all-to-all combine context is missing, so the shape of its gradient
input doesn't match.

## Left-padding corrupts causal attention on short prompts

When batching variable-length prompts for autoregressive inference, the
natural thing is to left-pad to the longest. A causal-only model (no
key_padding_mask) then attends to pad tokens from later positions as if
they were real context.

### Symptom

In a mixed batch:
- Long prompts (little or no left-pad) → coherent.
- Short prompts (heavy left-pad) → gibberish.

Deceptive because some prompts "work" — it looks like an intermittent
model bug.

### Fix

Trim every prompt to the length of the **shortest**, not left-pad to
the longest:

```python
enc = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
prompt_len = min(t.shape[0] for t in enc)       # not max
for i, t in enumerate(enc):
    buffer[i, :prompt_len] = t[:prompt_len].to(device)
```

The alternative (implementing a key_padding_mask inside the
causal/sliding block_mask) is intrusive and may not compose cleanly
with other mask_mods. Don't reach for it unless you need
variable-length prompts in production.

## Dynamic seq_len under `fullgraph=True` — don't touch the model

Autoregressive decode grows seq_len by 1 per step. Each new seq_len
retraces the compiled region, which can cost many seconds per step
with flex_attention inside.

### DO NOT do

- Set `dynamic=True` on the model's `@torch.compile` decorator.
- Change the model's compile region to `fullgraph=False`.
- Remove the decorator "just for inference testing".

All of these break the invariant that the modeling code equals the
training path. Inference tests must exercise the training code
verbatim.

### DO do

Static-seq-len decode in the test harness. Allocate a `[batch,
prompt_len + max_new_tokens]` buffer, fill prompt tokens, advance a
cursor, always pass the full buffer to the model. Set p2p shapes
*once* outside the loop:

```python
max_seq_len = prompt_len + max_new_tokens
buffer = torch.full((batch, max_seq_len), pad_id, dtype=torch.long, device=device)
for i, t in enumerate(enc):
    buffer[i, :prompt_len] = t[:prompt_len].to(device)
cursor = prompt_len
set_p2p_tensor_shapes([(1, max_seq_len, hidden_size)])    # once!

for step in range(max_new_tokens):
    loss, outputs = dualpipev.step(buffer if ctx.pp_rank == 0 else None, ...)
    next_tok = outputs[:, cursor - 1, :].float().argmax(dim=-1)
    buffer[:, cursor] = next_tok
    cursor += 1
```

Forward cost per step is higher, but you trade
`O(max_new_tokens)` compiles for one. A multi-minute test becomes tens
of seconds.

## Load canonical checkpoint to CPU, not GPU

`_load_dcp_canonical` reads *every* weight into one dict. That's ~42
GB for a 20B model, ~218–234 GB for 120B. If
`torch.set_default_device("cuda")` is set, the entire canonical lands on
the GPU and OOMs before the model even builds.

### Fix

Always allocate on CPU in the metadata iteration:

```python
for key, meta in metadata.state_dict_metadata.items():
    if key.startswith(prefix):
        sd[key] = torch.empty(meta.size, dtype=meta.properties.dtype, device="cpu")
dcp.load(sd, checkpoint_id=dcp_path, no_dist=True)
```

Filter to this rank's keys, then `.to(device)` only the filtered
result.

## The `.weight` suffix depends on storage style

- `nn.Linear` / `nn.Embedding` / `GroupLinear`: state_dict key ends
  in `.weight`.
- Raw `nn.Parameter` stored on the module itself: state_dict key is
  just the attribute name, **no `.weight`**.

Our canonical DCP keys must match the model's actual `state_dict()`
spelling. If the model uses raw `nn.Parameter` (like GPT-OSS experts),
the canonical key is
`layers.0.mlp.experts.0.gate_up_proj` (no `.weight`).

### Symptom of a mismatch

`load_state_dict(strict=False)` silently reports every expert weight
as "missing" and the model runs on random initialisation. Inference
produces gibberish; training starts from scratch instead of resuming.

### Check

```python
print(sorted(model.state_dict().keys())[:20])
# Compare against one-off print of your canonical keys
```

## Small bias gradients below the bf16 noise floor

`router.bias` (zero-init), layer-norm weights, attention biases —
their gradients per micro-batch can land at `~1e-9` in bf16. bf16's
7-bit mantissa gives ~1% relative precision, which accumulated through
8 layers of backward produces 5–20% per-element direction noise
between reference and 5-stage even when the math is identical.
`calc_diff` then reports a false failure.

### Fix

Scale the labels — not the threshold:

```python
label_scale = 10.0
full_l = label_scale * torch.randn(..., dtype=dtype)
```

MSE gradients are linear in the residual, so 10× label scale moves the
affected gradients from `~1e-9` into `~1e-8`, above the noise floor.
Larger gradients on well-conditioned parameters are unaffected (they
were already out of the noise floor).

See `testing.md` for the "when to trust the threshold" decision.

## Don't relax the threshold, don't skip by name

When a test fails, the temptation is to loosen `eps` or add a
name-based skip. **Don't.** Both hide real regressions:

- Loosened `eps` → future bugs in well-behaved gradients go unnoticed.
- Name skips → the skipped parameter has a permanent blind spot;
  refactors that also affect it silently regress.

Instead, print magnitudes and diagnose. See
`testing.md` §decision-tree.

## HF has some quirks to watch for

- **`quantization_config` in config.json** — if your `dcp2hf` writes
  BF16 safetensors but inherits the source config, transformers will
  still try to interpret the weights as quantized. Strip the key and
  set `torch_dtype: "bfloat16"` before round-trip loading.
- **`score_mod` closures over Parameters** — forces
  `flex_attention` to self-compile, which breaks our
  `fullgraph=True` outer compile. Use the LSE-renorm pattern instead
  (see `compile.md`).
- **`rope_scaling.rope_type == "yarn"` has a mscale concentration
  factor** — `0.1 * log(scaling_factor) + 1.0`. Don't skip it; it's
  not a bug in HF, it's the spec. Matches OpenAI's reference
  implementation for GPT-OSS.

## Also watch for

- **`num_hidden_layers` truncation parallel arrays.** Configs sometimes
  have `layer_types`, `attention_pattern`, or similar per-layer
  arrays. When you truncate `num_hidden_layers` in the single-GPU /
  FSDP tests, truncate these parallel arrays to match, or the model
  will read past the end.

- **Don't commit ad-hoc test scripts.** Model-specific single-GPU /
  inference tests are working-stage tools. They live next to
  `tests/` as untracked files; they are not part of the committed
  test suite.

- **Don't commit `docs/`, `AGENTS.md`, `CLAUDE.md`, `.agents/`, or `.claude/`.** Per user
  preferences. This skill itself lives under `.agents/skills/` — it's
  not committed.
