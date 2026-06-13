# Checkpoint Conversion (hf2dcp / dcp2hf)

**First: decide whether you need a model-specific converter at all.**
The generic path in `pithtrain/tasks/convert_checkpoint/_core.py`
handles un-quantized, un-transposed HF checkpoints — Qwen3 and
DeepSeek-V2 round-trip through it with no model-specific code.

## When to add a model-specific converter

Add a new `<Model>Converter` class in
`pithtrain/tasks/convert_checkpoint/<model>.py` **only if** one of these
applies:

1. **Quantized released weights** — MXFP4, GPTQ, AWQ, FP8, bitsandbytes,
   etc. Need to dequantize to BF16 before writing DCP.
2. **Layout mismatch** — our in-memory layout (`[E, out, in]`) differs
   from HF's live Parameter layout (`[E, in, out]`). Need a
   `.transpose(-2, -1)` at the `dcp2hf` boundary to give downstream
   consumers HF's native layout.
3. **Structural mismatch** — this should be rare. If you followed the
   conventions (fused `gate_up_proj` stays fused, router named `router`,
   etc.), there is nothing to rename or split in the converter.

If none of those apply: skip this whole file. The generic
`hf2dcp` / `dcp2hf` already works.

## Working on a quantized checkpoint

### Step 1 — Identify the format

Open the safetensors index and print shapes:

```python
import json
from safetensors import safe_open
from pathlib import Path

hf_path = Path("<snapshot>")
with open(hf_path / "model.safetensors.index.json") as f:
    index = json.load(f)
shards = set(index["weight_map"].values())

for shard in sorted(shards):
    with safe_open(str(hf_path / shard), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            print(k, tuple(t.shape), t.dtype)
```

Key suffixes to look for:
- `_blocks`, `_scales` → MXFP4 (GPT-OSS) or MXINT
- `_qweight`, `_qzeros`, `_scales` → GPTQ, AWQ
- `_scale_inv` → FP8

**Never trust comments about the on-disk layout — always verify from
the shape.** The GPT-OSS MXFP4 bug burned days because the original
code trusted a `[E, hidden, 2*inter]` comment while the actual prefix
was `[E, 2*inter, G, 16]` = `[E, 2*inter]` = `[E, out]`.

### Step 2 — Determine the axis convention

Packed tensor shape = `[prefix..., quant_axis_blocks, block_size]`. The
prefix tells you the layout. For MXFP4:

| Tensor | Blocks shape | Prefix = dense shape |
|--------|-------------|---------------------|
| `gate_up_proj_blocks` | `[E, 2*inter, G, 16]` | `[E, 2*inter, hidden]` = `[E, out, in]` |
| `down_proj_blocks` | `[E, hidden, G, 16]` | `[E, hidden, inter]` = `[E, out, in]` |

Both happen to match our runtime `[E, out, in]` layout — no transpose
needed inside `hf2dcp`. The transpose only happens at `dcp2hf` (to
produce HF's live `[E, in, out]` BF16 layout).

### Step 3 — Write the dequant function

Adapt from:

- Megatron-Bridge (https://github.com/NVIDIA/Megatron-Bridge)
- HuggingFace Transformers `src/transformers/quantizers/<format>.py`
- The model's release repo (OpenAI, Qwen, etc.)
- TorchTitan if it has a reference

See `_dequantize_mxfp4` in
`pithtrain/tasks/convert_checkpoint/gpt_oss.py` for the MXFP4 reference.

### Step 4 — Verify against HF's live dequant

Don't just test norms — test element-wise:

```python
from transformers import AutoModelForCausalLM
import torch

# HF's own BF16 dequant of the SAME MXFP4 checkpoint
hf = AutoModelForCausalLM.from_pretrained("<hf_id>", dtype=torch.bfloat16)
hf_live = hf.state_dict()

# One expert slice from our DCP conversion
ours_e0_gate = dcp_state_dict["layers.0.mlp.experts.0.gate_up_proj"]  # [2*inter, hidden]
hf_gate_up   = hf_live["model.layers.0.mlp.experts.gate_up_proj"]     # [E, in, out] = [E, hidden, 2*inter]

# HF's stored as [E, in, out]; our stored as [out, in].  Transpose for comparison.
ours_hf_shape = ours_e0_gate.transpose(-2, -1)   # → [hidden, 2*inter]
theirs = hf_gate_up[0]

print("norm ours:", ours_hf_shape.float().norm().item())
print("norm theirs:", theirs.float().norm().item())
print("max_abs_diff:", (ours_hf_shape.float() - theirs.float()).abs().max().item())
```

Norms matching but element-wise differing means you have a
transpose / interleave bug. **Norms are invariant under transposition;
norm-only checks miss the whole class of layout bugs.**

## Where the code lives

`pithtrain/tasks/convert_checkpoint/` is a package:

```
pithtrain/tasks/convert_checkpoint/
    __init__.py     # public API: ConvertCheckpointCfg, ConvertCheckpointCtx, launch
    _core.py        # generic hf2dcp / dcp2hf + dispatcher + launch
    _registry.py    # ModelConverter Protocol + explicit CONVERTERS list
    gpt_oss.py      # GptOssConverter (MXFP4 dequant + stacked-expert transpose)
```

A model-specific converter is a class with four methods (see
`_registry.ModelConverter`):

```python
class ModelConverter(Protocol):
    name: str
    def detect_hf(self, load_path: Path) -> bool: ...
    def detect_dcp(self, metadata) -> bool: ...
    def hf2dcp(self, load_path: Path, save_path: Path, stdout: Logger) -> None: ...
    def postprocess_canonical(
        self, canonical: Dict[str, torch.Tensor], stdout: Logger
    ) -> Dict[str, torch.Tensor]: ...
```

Generic `hf2dcp` delegates entirely to `converter.hf2dcp` when
`detect_hf` matches. Generic `dcp2hf` always runs the load +
canonicalization + shard-write; the converter only supplies
`postprocess_canonical` (stack + transpose) when `detect_dcp` matches.
If no converter matches, the generic path runs untouched.

## Adding a new converter

1. Create `pithtrain/tasks/convert_checkpoint/<model>.py` with a
   `<Model>Converter` class implementing the four methods.
2. Append an instance to `CONVERTERS` in `_registry.py`:
   ```python
   from .<model> import <Model>Converter
   CONVERTERS: List[ModelConverter] = [GptOssConverter(), <Model>Converter()]
   ```
   Order matters only if two converters could both match — keep the list
   small and unambiguous.

## `hf2dcp` method structure

```python
def hf2dcp(self, load_path: Path, save_path: Path, stdout: Logger) -> None:
    # 1. Load raw safetensors into a dict on CPU
    with open(load_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    raw: Dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(load_path / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)

    # 2. Dequantize quant_type-specific tensors (adjust for your format)
    dequantized: Dict[str, torch.Tensor] = {}
    seen = set()
    for key in sorted(raw.keys()):
        if key.endswith("_blocks"):
            base = key.removesuffix("_blocks")
            scales_key = base + "_scales"
            if scales_key in raw:
                dequantized[base] = _dequantize_<format>(raw[key], raw[scales_key]).contiguous()
                seen.add(key); seen.add(scales_key)

    for key, tensor in raw.items():
        if key not in seen and key not in dequantized:
            dequantized[key] = tensor

    # 3. Split stacked expert tensors into per-expert indexed keys,
    # strip "model." prefix, leave everything else alone.
    model_state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in dequantized.items():
        canon = key.removeprefix("model.")

        if canon.endswith((
            ".mlp.experts.gate_up_proj",
            ".mlp.experts.gate_up_proj_bias",
            ".mlp.experts.down_proj",
            ".mlp.experts.down_proj_bias",
        )):
            for idx in range(tensor.shape[0]):
                expert_key = canon.replace(".experts.", f".experts.{idx}.")
                model_state_dict[expert_key] = tensor[idx].contiguous()
        else:
            model_state_dict[canon] = tensor

    # 4. Write DCP
    save_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
```

`detect_hf` typically reads `load_path/config.json` and checks
`model_type`.

## `postprocess_canonical` structure

The generic `dcp2hf` calls this with the un-prefixed DCP state_dict and
expects back a dict with stacked / transposed keys to match HF's live
layout. What a model-specific converter adds is the **layout transpose**
from our `[E, out, in]` storage to HF's live `[E, in, out]`:

```python
def postprocess_canonical(self, canonical, stdout):
    _WEIGHT_KEYS = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
    # per-expert indexed → stacked
    ...
    for stacked_canon, by_idx in to_stack.items():
        items = sorted(by_idx.items())
        stacked = torch.stack([t for _, t in items])       # [E, out, in]
        if stacked_canon.endswith(_WEIGHT_KEYS):
            stacked = stacked.transpose(-2, -1).contiguous()   # → [E, in, out]
        result[stacked_canon] = stacked
    return result
```

`detect_dcp` inspects the DCP metadata — e.g. `GptOssConverter.detect_dcp`
checks for `gate_up_proj` keys. For non-quantized / non-stacked models
(Qwen3, DeepSeek-V2), there is no converter at all and the generic path
writes the DCP as-is.

## The round-trip validation

This is the only check that catches every class of bug.

### Step 1 — Round-trip

```bash
# Config: hf → dcp → hf
python -c "
from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch
from pathlib import Path

cfg = ConvertCheckpointCfg()
cfg.operation = 'hf2dcp'
cfg.load_path = Path('<hf_snapshot>')
cfg.save_path = Path('<dcp_dir>')
launch(cfg)

cfg = ConvertCheckpointCfg()
cfg.operation = 'dcp2hf'
cfg.load_path = Path('<dcp_dir>')
cfg.save_path = Path('<bf16_snapshot>')
launch(cfg)
"
```

### Step 2 — Strip `quantization_config`

Our `dcp2hf` writes BF16 safetensors but inherits `config.json` from
the source. If the source is quantized, its `quantization_config` block
tells transformers to expect quantized tensors — and the load fails.
Strip it:

```python
import json
from pathlib import Path

snap = Path("<bf16_snapshot>")
cfg = json.loads((snap / "config.json").read_text())
cfg.pop("quantization_config", None)
cfg["torch_dtype"] = "bfloat16"
(snap / "config.json").write_text(json.dumps(cfg, indent=2))
```

### Step 3 — Load and compare

```python
import torch
from transformers import AutoModelForCausalLM

# Our round-tripped BF16
ours = AutoModelForCausalLM.from_pretrained("<bf16_snapshot>", dtype=torch.bfloat16)
# HF's own BF16 dequant of the ORIGINAL quantized snapshot
theirs = AutoModelForCausalLM.from_pretrained("<hf_snapshot>", dtype=torch.bfloat16)

our_sd = ours.state_dict()
their_sd = theirs.state_dict()

assert set(our_sd.keys()) == set(their_sd.keys()), (
    set(our_sd.keys()) ^ set(their_sd.keys())
)

max_diff = 0.0
for k in sorted(our_sd.keys()):
    d = (our_sd[k].float() - their_sd[k].float()).abs().max().item()
    if d > max_diff:
        max_diff = d
        print(f"new worst: {k} diff={d:.4e}")
print(f"overall max_abs_diff = {max_diff:.4e}")
```

**Goal: `max_abs_diff == 0` across all keys.** If it's nonzero, bisect
by key prefix (`embed_tokens.*`, `layers.0.self_attn.*`, `layers.0.mlp.experts.*`, etc.)
and find the first class of tensor that differs.

## Launch scripts

Add `examples/convert_checkpoint/<model>/script.py`:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch

cfg = ConvertCheckpointCfg()
cfg.operation = "hf2dcp"
cfg.load_path = Path("workspace/checkpoints/<model>/hf-import")
cfg.save_path = Path("workspace/checkpoints/<model>/torch-dcp/step-00000000")

if __name__ == "__main__":
    snapshot_download(repo_id="<hf-id>", local_dir=cfg.load_path)
    launch(cfg)

cfg = ConvertCheckpointCfg()
cfg.operation = "dcp2hf"
cfg.load_path = Path("workspace/checkpoints/<model>/torch-dcp/step-00000000")
cfg.save_path = Path("workspace/checkpoints/<model>/hf-export")

if __name__ == "__main__":
    launch(cfg)
```

## Memory: load canonical to CPU

`_load_dcp_canonical` reads every weight into one dict. A 20B-class
model is ~42 GB; 120B is ~218–234 GB. If `torch.set_default_device("cuda")`
is set, that whole canonical dict lands on GPU and OOMs before the
model even builds.

**Always** allocate the canonical on CPU (`device="cpu"` in the
`torch.empty` that backs DCP's read), filter to this rank's keys, then
`.to(device)` only the filtered result. See
`tests/test_gpt_oss_inference.py:_load_dcp_canonical` for the pattern.

## Checklist

- [ ] `print(tensor.shape, tensor.dtype)` for every key in one shard —
      don't trust comments.
- [ ] Dequant function adapted from a reference impl; tested on one
      weight against HF's live dequant.
- [ ] `hf2dcp` splits stacked expert tensors into per-expert indexed
      keys; nothing else renamed.
- [ ] `dcp2hf` (or `postprocess_canonical`) stacks back, transposes
      3-D weights, adds `model.` prefix.
- [ ] Round-trip: strip `quantization_config` from output config,
      element-wise compare to HF's BF16 dequant, `max_abs_diff == 0`.
- [ ] `_load_dcp_canonical` allocates on CPU.
