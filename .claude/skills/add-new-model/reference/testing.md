# Testing Strategy

Tests in order: single-GPU sanity → FSDP pp/ep scaling ladder →
(optional) inference ladder. Each tier catches a different failure
class. Never skip tiers.

## Hardware hygiene (apply every command)

Shared cluster. Always check which GPUs are free before running
anything that touches CUDA:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
```

Pick indices whose `memory.used` is under ~1000 MiB. Do this check
**again** before each command, not once at the start of the session —
another user can grab GPUs between your runs. When running multiple
torchruns in sequence, reset the chosen GPU set each time.

## Timeouts (keep them short)

- Single-GPU sanity: 120s
- `test_fsdp.py` at pp=1/ep=1: 120s
- `test_fsdp.py` at pp≥2 or ep≥2: 180s
- Inference autoregressive decode: 180s

If the test exceeds the timeout, it is **hanging**, not slow. The
usual cause is torch.compile retracing on a new seq_len (see
`compile.md` on static-seq-len decode). Kill and diagnose; don't raise
the timeout.

## Tier 1 — Single-GPU sanity

Model-specific, **ad-hoc** (not committed). Base on
`tests/test_gpt_oss_single_gpu.py`.

**What it checks:** can we build the model, run `reference_forward`
to completion on 1 GPU, then run the 5-stage path and see bounded
relative drift.

```python
# tests/test_<model>_single_gpu.py  (ad-hoc; don't commit)
import torch, torch.nn.functional as F
from transformers import AutoConfig
from pithtrain.layers.factory import ModelImplMode
from pithtrain.models.<model> import <Model>Model

torch.manual_seed(1234)
device = torch.device("cuda")
torch.set_default_device(device)

config = AutoConfig.from_pretrained("examples/pretrain_lm/<model>/config.json")
# Truncate layers to keep the test fast.  Also truncate any
# parallel arrays like layer_types to match.
keep = min(config.num_hidden_layers, 4)
if getattr(config, "layer_types", None) is not None:
    config.layer_types = config.layer_types[:keep]
config.num_hidden_layers = keep
config.ep_size = 1

model = <Model>Model(config, num_stages=1, stage_id=0)
# Init any raw-Parameter experts (needed when not using GroupLinear)
for p in model.parameters():
    if p.dim() >= 2:
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
model.to(dtype=torch.bfloat16)
model.train()

bsz, seq_len = 2, 128
input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
labels = torch.randn(bsz, seq_len, config.vocab_size, dtype=torch.bfloat16)

# Reference path
ModelImplMode.use_reference_fwd = True
logits = model(input_ids)
assert torch.isfinite(logits).all()
loss = F.mse_loss(logits.float(), labels.float())
loss.backward()

# 5-stage path
ModelImplMode.use_reference_fwd = False
model.zero_grad()
logits2 = model(input_ids)
F.mse_loss(logits2.float(), labels.float()).backward()

# Comparison — bf16 compile-vs-eager noise bound
ModelImplMode.use_reference_fwd = True
logits_ref = model(input_ids)
diff = (logits_ref.float() - logits2.float()).abs().max().item()
rel = diff / logits_ref.float().abs().max().clamp(min=1e-6).item()
print(f"rel={rel:.4e}")
assert rel < 0.8, f"5-stage diverges from reference: rel={rel}"
```

### Why `rel < 0.8` and not tighter

Random-init bf16 through compile-vs-eager produces per-logit drift
that's ~20–30% of the max logit at 2–4 layers — this is expected. The
accumulated drift comes from:

1. `@torch.compile(fullgraph=True)` on `forward_aggregate` re-fusing
   the weighted sum differently from eager.
2. `F.grouped_mm` tile shapes depending on tokens-per-expert, which
   differs between the reference and 5-stage paths.
3. Fused-vs-split kernels choosing different accumulation orders.

The single-GPU test is an **orders-of-magnitude sanity check**, not a
numerical tolerance. Do not tighten it based on a single passing run.

## Tier 2 — FSDP pp/ep scaling ladder

Run `tests/test_fsdp.py` in 4 configurations, in this order:

| Config | GPUs | What it adds | Catches |
|--------|------|-------------|---------|
| pp=1/ep=1 | 1 | baseline FSDP + DualPipeV scheduler | modeling bugs, NaN, compile drift |
| pp=2/ep=1 | 2 | pipeline P2P send/recv | stage-record copy bugs, backward shape errors |
| pp=1/ep=2 | 2 | all-to-all dispatch + combine + dedup | expert-sharding bugs, bias-slicing bugs |
| pp=2/ep=2 | 4 | full combination | nothing new; regression check |

If step N passes and N+1 fails, the *new* parallelism dimension added
in N+1 is the suspect.

```bash
# Check GPUs
nvidia-smi --query-gpu=index,memory.free --format=csv

# Configuration knobs:
CFG=examples/pretrain_lm/<model>/config.json
RDZV="--rdzv-backend=c10d --rdzv-endpoint=localhost:15213"

# 1. pp=1, ep=1 (one GPU)
CUDA_VISIBLE_DEVICES=<g0> timeout 180 torchrun --nproc-per-node=1 $RDZV \
  tests/test_fsdp.py --pp-size 1 --ep-size 1 --model $CFG

# 2. pp=2, ep=1 (two GPUs)
CUDA_VISIBLE_DEVICES=<g0>,<g1> timeout 180 torchrun --nproc-per-node=2 $RDZV \
  tests/test_fsdp.py --pp-size 2 --ep-size 1 --model $CFG

# 3. pp=1, ep=2 (two GPUs — re-check nvidia-smi, pick fresh)
CUDA_VISIBLE_DEVICES=<g0>,<g1> timeout 180 torchrun --nproc-per-node=2 $RDZV \
  tests/test_fsdp.py --pp-size 1 --ep-size 2 --model $CFG

# 4. pp=2, ep=2 (four GPUs)
CUDA_VISIBLE_DEVICES=<g0>,<g1>,<g2>,<g3> timeout 180 torchrun --nproc-per-node=4 $RDZV \
  tests/test_fsdp.py --pp-size 2 --ep-size 2 --model $CFG
```

### What the test validates

- **Loss match:** `torch.allclose(loss, loss_ref, rtol=1e-3, atol=1e-3)`.
- **Gradient match:** `calc_diff < 1e-2` per parameter, where
  `calc_diff = 1 - 2*(x*y).sum() / (x*x + y*y).sum()` (cosine-ish).

Loss matches, grads don't → issue in backward. Loss doesn't match →
issue in forward.

### What to change in `tests/test_fsdp.py` when adding a new model

1. Add the model config path to the `models` list at the bottom.
2. Import the new `<Model>Model`, router/gate class, and Experts class
   (only if experts are raw `nn.Parameter`).
3. Add the new class to `apply_fsdp`'s `isinstance` assertion tuple.
4. Add a `config.model_type` branch in `main` that sets
   `ModelClass = <Model>Model` and truncates `num_hidden_layers` to 8
   (plus any parallel arrays, like `layer_types` for GPT-OSS).
5. Add `fill_weights` branches (see below).
6. Extend `shard_experts` only if the model uses an unusual raw-Parameter
   name — the existing `gate_up_proj` gate covers GPT-OSS-style experts.

### `fill_weights` branches

`fill_weights` is class-dispatched: it relies on
`isinstance(module, <SomeClass>)` to know what to initialise. The
existing branches are `nn.Linear`, `GroupLinear`, `GptOssExperts`,
`DeepseekV2LiteMoEGate`/`Qwen3MoeGate`/`GptOssTopKRouter`, and
`nn.Embedding`.

Add a branch when:

- **Raw-Parameter experts** — the `GroupLinear` branch won't reach
  them, and they'll stay at their `torch.empty()` state, which on our
  system is zero. Symptom: every MoE layer emits
  `[warn] Parameter ... has all-zero gradient`. Without this branch,
  `mlp(x) == 0` in every layer and the residual stream is unchanged.
  See `pitfalls.md` §silent-zero-experts.

- **New router/gate Parameters** beyond `weight` (e.g. a per-expert
  `bias` like GPT-OSS has).

```python
elif isinstance(module, <Model>Experts):   # raw-Parameter experts
    nn.init.xavier_uniform_(module.gate_up_proj, gain=1.0)
    nn.init.xavier_uniform_(module.down_proj, gain=1.0)
elif isinstance(module, (<...existing gates>, <Model>Router)):
    nn.init.xavier_uniform_(module.weight, gain=1.0)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)
```

### `shard_experts` fallback

`shard_experts` walks the module tree. For each `GroupLinear` child, it
slices by expert and replaces with a smaller module. For raw-Parameter
experts, there's a fallback that detects them by a distinctive weight
name (not just `num_experts`):

```python
gu = getattr(model, "gate_up_proj", None)
if isinstance(gu, nn.Parameter):
    num_experts = getattr(model, "num_experts", None)
```

**Do not gate only on `num_experts` alone** — the router has
`num_experts` too, and sharding it breaks routing (every rank needs
the full per-expert table). Extend the gate with *your model's
distinctive expert-weight Parameter name*.

## The label-scaling gotcha

`test_fsdp.py` scales labels by `label_scale = 10.0`:

```python
full_l = label_scale * torch.randn(
    ep_size * num_chunks * micro_batch_size, sequence_length, vocab_size, dtype=dtype,
)
```

This is deliberate. MSE gradients are linear in the residual. Without
the scale, tiny-gradient parameters (zero-init router biases, layer-norm
weights, attention biases) produce gradients at ~1e-9 in bf16, which is
at the bf16 mantissa noise floor. `calc_diff` then measures rounding,
not algorithmic correctness, and reports false failures.

### If `calc_diff` fails on a tiny bias

**Do not loosen the threshold first.** Print magnitudes:

```python
print(f"{n}: p_grad mag={p_grad.abs().max():.4e}, "
      f"p_ref.grad mag={p_ref.grad.abs().max():.4e}, "
      f"diff={diff:.4e}")
```

- If both magnitudes are `~1e-9` or smaller → raise `label_scale` (or
  switch to cross-entropy loss; MSE on one-hot-like labels also helps).
- If magnitudes differ by `>10x` → it's a real bug. Look at the stage
  where the gradient diverges.

See `lessons §23` for the full diagnosis pattern.

## Tier 3 — Inference ladder (optional)

Only when the user wants real-weight inference. Base on
`tests/test_gpt_oss_inference.py`. Same scaling ladder as training:

```bash
# 1/1 first
CUDA_VISIBLE_DEVICES=<g0> timeout 180 torchrun --nproc-per-node=1 $RDZV \
  tests/test_<model>_inference.py --pp-size 1 --ep-size 1

# Then 2/1, 1/2, 2/2 — same structure as test_fsdp
```

Check: all four configurations produce *coherent* text (human-judged),
and should produce identical tokens within bf16 noise. Divergence between
configurations is a bug, not acceptable.

## When a test fails — the decision tree

```
Test fails
├─ Is this a loss mismatch or a grad mismatch?
│
├── Loss mismatch
│    └─ Forward has a bug.  Diff reference vs 5-stage with
│       `reference_forward` one layer at a time.
│
└── Grad mismatch (loss matches)
     ├─ Print magnitudes (p_grad vs p_ref.grad for the worst param)
     │
     ├── Both magnitudes in bf16 noise (~1e-9 or less)
     │    └─ Raise label_scale.  Probably not a real bug.
     │       See lessons §23.
     │
     ├── Magnitudes differ by >10x
     │    ├─ Gradient is huge vs reference → NaN propagation
     │    │  (forward pad rows → backward 0*NaN=NaN).  Truncate to
     │    │  sum(ks) in expert forward.  See pitfalls.md.
     │    │
     │    ├─ Gradient is much smaller than reference → a path isn't
     │    │  accumulating.  Check stage-record copy.  See protocol.md.
     │    │
     │    └─ All-zero gradients on MoE parameters → fill_weights miss.
     │       See pitfalls.md §silent-zero-experts.
     │
     └── Magnitudes similar, but signs or per-element are off
          └─ Real algorithmic bug.  Bisect by stage (disable one at a
             time via use_reference_fwd).
```

## Don't

- Don't relax `rtol`/`atol` on the loss assertion. Loss is well within
  bf16 tolerance when forward is correct.
- Don't relax `eps = 1e-2` on `calc_diff` without first printing
  magnitudes. The noise floor lives at eps=1e-3 for reasonably-scaled
  gradients; 1e-2 is already generous.
- Don't add `--skip-<param-name>` flags. Either the test passes for
  every parameter or the model has a bug. Name-based skips become
  permanent blind spots.
- Don't set `dynamic=True` on compile to "fix" inference retracing.
  Fix it in the test harness with static-seq-len decode — the model
  code must stay identical to training.
- Don't run at full `num_chunks=20` during iteration. Use `num_chunks = pp_size * 2`
  (the minimum DualPipeV allows) with `sequence_length=32` or 64.
  Restore full workload only for the final regression check.
