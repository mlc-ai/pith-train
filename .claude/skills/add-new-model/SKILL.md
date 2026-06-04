---
name: add-new-model
description: Adds support for a new MoE language model to PithTrain. Use when the user asks to "add support for model X", "implement model Y in pithtrain", "port model Z", or otherwise integrate a new MoE architecture. Scope covers the model file, all framework wiring (setup_model, apply_fsdp, test_fsdp), optional checkpoint conversion, and running training + inference tests from pp=1/ep=1 up to pp=2/ep=2.
argument-hint: <hf-id-or-snapshot-path> [model-short-name]
---

# Add a New Model to PithTrain

End-to-end workflow for integrating a new MoE language model. This file is
the entry point: it tells you the phase order, the gates between phases,
and which `reference/*.md` to load before each phase. Do not try to do
everything from memory — load the reference file for the phase you're in.

## Input

One of the following:

- **HuggingFace model ID** (e.g. `"mistralai/Mixtral-8x7B-v0.1"`). Used as the
  `--hf-id` for `snapshot_download`, and as the `from_pretrained` source for
  config and tokenizer.
- **Local snapshot path** (a directory containing `config.json`,
  `model*.safetensors`, `tokenizer*` etc.). Treat it exactly like the HF
  ID case — `AutoConfig.from_pretrained(path)` works on both. The snapshot
  itself came from HF, so online reference material (HF's
  `modeling_<model>.py`, TorchTitan, OpenAI release repo, upstream papers)
  is still fair game and should be consulted.

Optionally a **model short name** (filename stem, e.g. `mixtral_8x7b`). If
the user doesn't give one, derive it from the HF ID by lowercasing and
replacing `/` and `-` with `_`. Confirm with the user before using.

## Hard rules (apply in every phase)

These are non-negotiable. Violating any of them will cost time later —
in most cases that's exactly how past bugs landed.

1. **Mirror HuggingFace, not our existing models.** Class names, attribute
   names, and tensor structure (fused vs split) must match HF's
   `modeling_<model>.py`. *Do not* base the new model on Qwen3 / DeepSeek-V2
   / GPT-OSS and rename — that path produced the GPT-OSS
   `gate`/`router` mismatch. See `reference/conventions.md`.
2. **`fullgraph=True` for all three hot regions.** `_forward_attn_compute`,
   router/gate `compute`, and `forward_aggregate` must each carry
   `@torch.compile(fullgraph=True)`. Never reach for `fullgraph=False`.
   See `reference/compile.md`.
3. **Shared experts live in `forward_attn`, not `forward_aggregate`.** If
   the model has shared experts (e.g. DeepSeek-V2), fold them into the
   residual at the end of `_forward_attn_compute`. See
   `reference/protocol.md`.
4. **Check `nvidia-smi` before every GPU command.** This is a shared
   cluster. Free GPUs can change between commands; don't reuse indices.
   `nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv` and
   pick indices with `memory.used < 1000 MiB`. Do this once per
   invocation, not once at the start of the whole skill.
5. **Test timeouts stay short (120–180s).** Do not set a 10-minute timeout
   and walk away — if a test hasn't progressed in 3 minutes, it's hanging
   (likely torch.compile retrace). Kill and diagnose.
6. **When tests fail, diagnose before relaxing anything.** Print actual
   magnitudes. A 6-order-of-magnitude gradient discrepancy is not bf16
   noise. Never loosen thresholds or add name-based skips as a first move.
7. **Reject unused process-group arguments explicitly.** If a model's
   `__init__` accepts a group it does not actually implement (e.g.
   `cp_group` on a model with no ring-attention path), silently ignoring
   it will produce wrong results when a real group is eventually passed.
   Raise `NotImplementedError` when `group is not None and group.size() > 1`.
   See `reference/protocol.md` §init-requirements.
8. **Thread config values through `__init__`; don't hardcode them.** Any
   value that appears in HF's `config.json` is a per-checkpoint knob —
   read it from the config even if every released checkpoint currently
   ships the same value. Only true architectural constants (paper
   coefficients, spec-defined magic numbers) stay as module-level
   literals, and they get a one-line comment naming the source. See
   `reference/conventions.md` §thread-config.
9. **Never stage `.claude/`, `CLAUDE.md`, or `docs/` in commits.**

## Phase overview

Phases 1–4 are modeling + training correctness. Phases 5–6 are real-weight
inference (only needed if the user wants to generate from trained / released
weights). **Phase 5 is skippable** when the user only cares about training
from scratch.

| Phase | Goal | Gate to next phase |
|-------|------|-------------------|
| 0 | Analyze HF's reference implementation | Have class/attribute/shape/config inventory |
| 1 | Write `pithtrain/models/<model>.py` | Imports cleanly; `reference_forward` runs |
| 2 | Wire into `pithtrain/modules/training.py` + `tests/test_fsdp.py` + example config | Example config mirrors upstream; imports clean |
| 3 | Single-GPU sanity test | `reference_forward == 5-stage path` (rel < 0.8) |
| 4 | FSDP scaling (pp=1/ep=1 → 2/2) | All 4 configs pass |
| 5 | *(If needed)* Checkpoint converter + round-trip | `hf → dcp → hf → transformers.load` succeeds |
| 6 | *(If needed)* Ad-hoc inference test | Coherent text from real weights |

Do not skip ahead. Each phase is a gate: if phase N fails, do not move to
phase N+1. If you're tempted to, stop and read `reference/pitfalls.md`.

---

## Phase 0 — Analyze the HF reference

Before writing any code, inventory what you need to match.

**Load `reference/conventions.md` before starting this phase.** It has
the diagnostic commands (grep class names, grep attribute names,
safetensors shape dump) under §Quick diagnostic commands.

Work through these sources in order:

1. **`modeling_<model>.py`** — class names, attribute names, fused vs
   split projections, special-case features (shared experts, sinks,
   sliding window, YaRN RoPE, clamped SwiGLU, attention biases).
2. **A safetensors shard** — actual expert-weight shapes and dtypes,
   not comments.
3. **TorchTitan / Megatron-LM** reference for this model — the
   training-framework-consensus expert layout (`[E, out, in]` vs
   `[E, in, out]`).
4. **`configuration_<model>.py`** — every default in
   `<Model>Config.__init__`. When model-specific defaults disagree
   with a generic fallback path, match the model-specific default
   (see `reference/conventions.md` §example-config).
5. **HF's MLP / activation forward** — read the math directly. Don't
   trust `config.hidden_act` to tell you the whole activation. See
   `reference/conventions.md` §activation-math.

Record in a scratch doc (not a committed file): class names, attribute
names, expert tensor layout, fused/split projections, per-checkpoint
knobs (thread through `__init__`) vs architectural constants (module
literals with a source comment), process groups the model accepts but
doesn't implement (reject via `NotImplementedError`; see
`reference/protocol.md` §init-requirements), and any special-case
features that map to entries in `reference/pitfalls.md`.

**Gate:** you can articulate exactly which class names, attribute names,
tensor shapes, and config knobs you will wire. If any item is a guess,
go back and `print()` it from the actual data.

---

## Phase 1 — Write `pithtrain/models/<model>.py`

**Load `reference/protocol.md` and `reference/compile.md` before starting.**

1. Start from `templates/model_skeleton.py`. It is a structural outline
   (NOT a copy of Qwen3). Fill in the `TODO` placeholders with the HF-
   derived names and shapes from phase 0.
2. Implement in this order:
   - RotaryEmbedding (mirror HF's)
   - Attention (mirror HF's kernel choice — flash_attn for standard MHA/GQA,
     flex_attention for sinks/sliding)
   - Experts module
   - Router / Gate
   - MLP (the MoE block that wires router + experts)
   - DecoderLayer (the 5-stage split)
   - Model (forward / backward / stage-record copy)
3. Checklist for the decoder layer:
   - [ ] `self.idx = layer_idx`
   - [ ] `self.mlp.ep_size` and `self.mlp.ep_group` exposed (so
         `DecoderLayerMlpProtocol` is satisfied)
   - [ ] `@torch.compile(fullgraph=True)` on `_forward_attn_compute`,
         router/gate `compute`, and `forward_aggregate`
   - [ ] Shared experts (if any) fold into residual at the end of
         `_forward_attn_compute`, *before* the return
   - [ ] `reference_forward` runs eager (no compile) and is numerically
         equivalent to `forward_attn → forward_mlp → forward_aggregate`
   - [ ] `forward_mlp` truncates expert input by `sum(ks)` if the expert
         block has biases or elementwise post-ops (prevents 0*NaN=NaN in
         backward)
   - [ ] `forward_mlp` uses `padded_index_gather` (not raw indexing) for
         both expand and reverse shuffle
4. Checklist for the model class:
   - [ ] Uses `layer_partition` from `pithtrain/dualpipe/layer_partition.py`
   - [ ] `forward` copies *every* stage record (including stages 2 and 4,
         which only have `.ctx`) into the pre-allocated
         `IntermediateTensors.layers[layer_idx]`. Iterate with
         `dataclasses.fields`, don't skip any.
   - [ ] `backward` is a `@staticmethod`, walks layers in reverse, drives
         `decoder_layer_backward`, and runs the prolog backward via
         `run_backward(record.outs, dx)`.

**Gate:** file imports cleanly (`python -c "from pithtrain.models.<model> import <Model>"`).

---

## Phase 2 — Wire into the training framework

No new reference file needed — the changes are small and mechanical.

1. `pithtrain/modules/training.py`:
   - Import the new `<Model>Model` class.
   - Add a branch in `setup_model`:
     ```python
     elif module_config.model_type == "<model_type>":
         ModelClass = <Model>Model
         model_kwargs = {"cp_group": cp_group}  # or {} if no CP support
     ```
   - Add the new class to the `apply_fsdp` `isinstance` assertion tuple.
   - Add the HF ID to the `TrainingCfg.model` `Literal[...]` union (if
     the user wants the HF ID to be an accepted value; config-path
     usage doesn't require this).
2. `tests/test_fsdp.py`:
   - Import the new model + router/gate class (+ Experts class if it
     stores raw `nn.Parameter` expert weights — see
     `reference/pitfalls.md`).
   - Add the new class to the `apply_fsdp` `isinstance` assertion tuple.
   - Add a branch in the `config.model_type` switch in `main`. Slice
     `num_hidden_layers` down to 8 (and any parallel arrays like
     `layer_types`) to keep the test fast.
   - Add a `fill_weights` branch if:
     - The expert module stores raw `nn.Parameter` (not `GroupLinear`).
       Without this, expert weights default to zero and the MoE subtree
       silently produces all-zero outputs — see `reference/pitfalls.md`.
     - The router/gate has new Parameters beyond `weight` (e.g. a
       per-expert `bias`).
   - Verify `shard_experts` can detect the experts module. If using
     raw `nn.Parameter`, the fallback gate on `gate_up_proj` already
     handles it. If the Parameter name is different, extend the fallback
     — gate on the distinctive weight name, *not* on `num_experts` alone
     (the router has `num_experts` too and must not be sharded).
3. Add the model config and HF ID to the `models` list at the bottom of
   `tests/test_fsdp.py`.
4. Write `examples/pretrain_lm/<model>/config.json`. Mirror
   upstream HF's `config.json` field-by-field — including every nested
   block (`rope_scaling`, `quantization_config`, etc.). See
   `reference/conventions.md` §example-config for the diff command and
   the three layered defaults you need to reconcile.

**Gate:** `python -c "import tests.test_fsdp"` imports cleanly AND the
example-config diff is either empty or has a documented reason for each
remaining difference.

---

## Phase 3 — Single-GPU sanity test

**Load `reference/testing.md`.** Tier 1 there is the whole phase:
reference template, tiny-config setup, reference-vs-5-stage comparison,
and the `rel < 0.8` bf16-noise bound. Ad-hoc test file (not committed);
base on `tests/test_gpt_oss_single_gpu.py`. Single GPU, `timeout 180`.

**Gate:** assertion passes with `rel < 0.8`, logits and gradients are
finite.

---

## Phase 4 — FSDP scaling (training correctness)

**Keep `reference/testing.md` loaded.** It owns the ladder: full
`torchrun` commands for pp=1/ep=1 → pp=2/ep=1 → pp=1/ep=2 → pp=2/ep=2,
what each config adds, thresholds, and the failure decision tree.

Run the ladder in that order. After each step, stop and diagnose
before continuing if anything fails. `nvidia-smi` before each run.
Timeouts 120–180s; past that it's hanging (compile retrace or
deadlocked all-to-all) — kill, don't raise.

**Gate:** all 4 configs pass (loss `rtol=atol=1e-3`, per-param
`calc_diff < 1e-2`).

---

## Phase 5 — Checkpoint converter (only if needed)

**Skip this phase entirely** if the user only wants training from scratch
(no real released weights involved). The generic path in
`pithtrain/tasks/convert_checkpoint/_core.py` already handles
un-quantized, un-transposed HF checkpoints — Qwen3 and DeepSeek-V2 work
with no model-specific converter.

**Add a converter only if one of the following applies:**

- The released weights are quantized (MXFP4, GPTQ, AWQ, FP8, etc.).
- The HF live tensor layout differs from your model's in-memory layout
  (e.g. our `[E, out, in]` vs HF's `[E, in, out]`).
- HF's key structure differs from ours (e.g. per-expert indexed vs
  stacked, fused vs split projections). This should be rare if you
  followed phase 0 faithfully — ideally our model mirrors HF's structure
  so the converter is trivial.

**Load `reference/checkpoint.md` before starting this phase.**

1. Create `pithtrain/tasks/convert_checkpoint/<model>.py` with a
   `<Model>Converter` class (see `gpt_oss.py` for the pattern).
   Implement `detect_hf` / `detect_dcp` probes, `hf2dcp`, and
   `postprocess_canonical`.
2. Register the converter instance in
   `pithtrain/tasks/convert_checkpoint/_registry.py` (append to
   `CONVERTERS`).
3. Write an `examples/convert_checkpoint/<model>/script.py` that
   downloads + converts. Mirror
   `examples/convert_checkpoint/gpt-oss-20b/script.py`.
4. Run the round-trip: hf → dcp → hf → `transformers.AutoModelForCausalLM.from_pretrained`.
   Compare `state_dict()` element-wise against HF's own BF16 dequant.
   Expected `max_abs_diff == 0`.

**Gate:** round-trip succeeds, one expert weight compares element-wise
equal (not just norms!) against HF's live tensor.

---

## Phase 6 — Ad-hoc inference test (only if needed)

Only needed if the user wants to verify that real weights produce coherent
text. **This test is not committed** — it's model-specific and lives as a
scratch file.

1. Start from `templates/inference_test.py` — the DualPipeV
   autoregressive harness, parameterized for any `<Model>Model`. Fill
   in the model-class import and HF ID default.
2. Run the same pp/ep scaling ladder as phase 4 (same torchrun form,
   replace `tests/test_fsdp.py` with `tests/test_<model>_inference.py`
   and drop `--model <cfg>`). Each config should print coherent
   continuations.
3. Compare outputs across configurations — they should produce
   identical tokens (within bf16 noise). If a specific config produces
   gibberish, diagnose by layer/stage — do not loosen expectations.

**Gate:** coherent text from real released weights, identical (bf16-noise
equivalent) across pp/ep configurations.

---

## Pre-PR self-review

Three sweeps on the new files before opening the PR — low-noise,
high-signal self-reviews that save a review round-trip:

1. **Function-scope imports.** Only justified for circular imports or
   heavy optional deps. Ruff doesn't flag it; reviewers will. Grep for
   indented `import`/`from` in the new model file; move them to
   module level.
2. **Dangling `docs/`, `CLAUDE.md`, `.claude/` pointers in comments or
   docstrings.** Those paths aren't committed, so any pointer is a
   broken link. Grep the new files and inline the derivation or delete.
3. **Unused parameters for interface compatibility.** Accept them
   (e.g. `cp_group` for protocol parity), then either prefix with `_`
   or `raise NotImplementedError` when `size() > 1` (Hard Rule 7).
   Bare unused params trip pyright/pylance.

## Common failure modes → where to look

| Symptom | First thing to read |
|---------|-------------------|
| Single-GPU `rel > 0.8` | `reference/pitfalls.md` §compile-noise |
| All-zero gradient warnings on MoE params | `reference/pitfalls.md` §fill-weights |
| FSDP loss matches but grads don't | `reference/testing.md` §label-scaling + `reference/pitfalls.md` §nan-padding |
| `RuntimeError: tensor data is not allocated yet` | Wrong reshard settings — check `apply_fsdp` |
| Inference gibberish but FSDP passed | `reference/checkpoint.md` §weight-norm-comparison + `reference/conventions.md` §example-config + §thread-config + §activation-math |
| Wrong results only when a real `cp_group` is passed | `reference/protocol.md` §init-requirements (silent-ignore of unused groups) |
| "invalid gradient shape" in stage 4 backward | `reference/protocol.md` §stage-record-copy |
| `compile-inside-compile` on attention | `reference/compile.md` §flex-unwrap |
| Left-padded prompts give gibberish on short inputs | `reference/pitfalls.md` §trim-to-shortest |

## Reference files

- `reference/protocol.md` — 5-stage protocol, `Model.forward`/`backward`, stage-record copy
- `reference/conventions.md` — naming, tensor layout, canonical keys
- `reference/compile.md` — three `@torch.compile(fullgraph=True)` hot regions, unwrap patterns
- `reference/checkpoint.md` — hf2dcp/dcp2hf recipes, when to add, round-trip validation
- `reference/testing.md` — pp/ep scaling ladder, test_fsdp wiring, label scaling
- `reference/pitfalls.md` — NaN padding, `.view()` vs `.transpose()`, silent-zero experts, etc.

## Templates

- `templates/model_skeleton.py` — structural outline with HF-derived placeholders
- `templates/inference_test.py` — DualPipeV autoregressive harness
