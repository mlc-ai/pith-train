---
name: add-memory-prints
description: Add detailed memory profiling prints throughout the training framework. Instruments distributed setup, model creation, checkpoint loading, pipeline scheduling, per-layer activations, saved tensor profiling, expert MLP internals, and memory snapshot dumps. Use when user asks to "add memory prints", "instrument memory", "profile memory", "memory breakdown", or "debug memory".
argument-hint: "<model-name> [--ranks <rank-list>] [--detail-layers <layer-indices>]"
---

# Memory Prints Instrumentation

Add comprehensive memory profiling instrumentation to the pithtrain training framework. This skill adds 7 groups of memory prints across 6 files, covering every phase from distributed init through the pipeline loop.

## Arguments

Parse the following from `$ARGUMENTS`:

1. **model** (required): Model name matching a file under `pithtrain/models/` (e.g., `qwen3-moe` maps to `pithtrain/models/qwen3_moe.py`).
2. **--ranks** (optional, default: `{0}`): Comma-separated GPU **global** ranks to print on. E.g., `--ranks 3,14` becomes the Python set `{3, 14}`. If not provided, defaults to `{0}`.
3. **--detail-layers** (optional, default: auto): Comma-separated layer indices for fine-grained per-layer profiling. If omitted, auto-select by reading the model file to find the MoE-vs-dense layer condition, then pick the first MoE layer and the next one (2 layers total).

Throughout this document, `RANKS` means the parsed rank set (e.g., `{3, 14}`), and `DETAIL_LAYERS` means the parsed or auto-selected layer indices tuple (e.g., `(5, 6)`).

## Correctness Guarantee

All edits are observation-only:

- `_lmem()` and `_mem_gb()` call `torch.cuda.synchronize()` + read `memory_allocated()`. No tensor modification.
- `_SavedTensorsProfiler` uses `saved_tensors_hooks` with a pack function that returns the tensor unchanged.
- Expert detail prints refactor `return self.down_proj(g * u, ...)` into `gu = g * u; out = self.down_proj(gu, ...); return out` — functionally identical.
- Pipeline prints add synchronize/print between existing operations. Adds latency, does not change computation order.

## Before Starting

1. **Check for existing prints**: Search for `_mem_gb`, `_lmem`, `_layer_mem_profile`, `_setup_mem`, or `memory_profiling` in the codebase. If found, ask the user whether to update ranks/layers or skip.
2. **Read all target files** (listed in Execution Order below) to find exact insertion points.
3. **Parse `$ARGUMENTS`** for model, ranks, and detail layers.

## Group 1: Helpers

### 1A: Pipeline helpers in `pithtrain/dualpipe/dualpipev.py`

Add right before `class DualPipeV`:

```python
def _mem_gb() -> float:
    """Return current CUDA memory allocated in GiB."""
    return torch.cuda.memory_allocated() / 1024**3


def _mem_detail() -> str:
    """Return allocated, cached-pool, and non-pytorch memory in GiB."""
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    G = 1024**3
    cached = reserved - allocated
    non_pytorch = total - free - reserved
    return f"alloc={allocated / G:.2f} cached={cached / G:.2f} non-pt={non_pytorch / G:.2f}"
```

In `DualPipeV.__init__`, add after `self.comm_stream = ...`:

```python
self.memory_profiling = True  # Set to True to enable per-step memory logging
```

### 1B: Layer-level helpers in `pithtrain/dualpipe/modeling.py`

Add after imports, before any function definitions:

```python
# -- Per-layer activation profiling --
_layer_mem_profile = False
_layer_mem_ranks = RANKS


class _SavedTensorsProfiler:
    """Context manager that logs tensors saved by autograd, distinguishing weights from activations."""

    def __init__(self, layer_idx: int, stage_name: str, weight_data_ptrs: set):
        self._layer_idx = layer_idx
        self._stage_name = stage_name
        self._weight_data_ptrs = weight_data_ptrs
        self._log: list[str] = []
        self._act_bytes = 0
        self._wt_bytes = 0

    def _pack(self, t: torch.Tensor) -> torch.Tensor:
        nbytes = t.nelement() * t.element_size()
        is_wt = t.data_ptr() in self._weight_data_ptrs
        tag = "weight" if is_wt else "activ"
        self._log.append(
            f"    saved ({tag}): {tuple(t.shape)} {t.dtype} ({nbytes / 1024**2:.1f} MB)"
        )
        if is_wt:
            self._wt_bytes += nbytes
        else:
            self._act_bytes += nbytes
        return t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        self._ctx.__exit__(*args)

    def print_summary(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank not in _layer_mem_ranks:
            return
        hdr = f"layer{self._layer_idx} {self._stage_name} saved tensors"
        print(f"[rank={rank}] {hdr}:", flush=True)
        for line in self._log:
            print(f"[rank={rank}] {line}", flush=True)
        print(
            f"[rank={rank}]   activ={self._act_bytes / 1024**2:.1f} MB, "
            f"weight={self._wt_bytes / 1024**2:.1f} MB, "
            f"total={len(self._log)} tensors",
            flush=True,
        )


def _lmem(label: str) -> None:
    """Print memory at a layer-internal checkpoint."""
    if not _layer_mem_profile:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank not in _layer_mem_ranks:
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    print(f"[rank={rank}]   {label}: alloc={alloc:.2f}", flush=True)
```

### 1C: Setup helper in `pithtrain/modules/training.py`

Add before `setup_model`:

```python
def _setup_mem(label: str) -> None:
    """Print CUDA memory at a setup checkpoint."""
    if torch.distributed.get_rank() in RANKS:
        torch.cuda.synchronize()
        G = 1024**3
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        free, total = torch.cuda.mem_get_info()
        cached = reserved - alloc
        non_pytorch = total - free - reserved
        print(
            f"[rank={torch.distributed.get_rank()}] setup_model | {label}: "
            f"alloc={alloc / G:.2f} cached={cached / G:.2f} non-pt={non_pytorch / G:.2f}",
            flush=True,
        )
```

---

## Group 2: Distributed Setup Prints (`pithtrain/modules/distributed.py`)

### In `setup_default_process_group`

**Important**: this runs BEFORE `init_process_group`, so `torch.distributed.get_rank()` is not available. Use `ctx.local_rank` for the rank guard.

After setting `ctx.local_rank`, move `torch.cuda.set_device(ctx.local_rank)` to BEFORE `init_process_group` if it isn't already. Then add a memory probe before and after `init_process_group`:

```python
# Probe: isolate CUDA context cost from NCCL communicator init
torch.cuda.set_device(ctx.local_rank)
torch.cuda.synchronize()
_free0, _total = torch.cuda.mem_get_info()
_non_pt0 = _total - _free0
G = 1024**3

# ... existing init_process_group code ...

torch.cuda.synchronize()
_free1, _ = torch.cuda.mem_get_info()
_non_pt1 = _total - _free1
if ctx.local_rank in RANKS:
    print(
        f"[rank={ctx.rank}] init_process_group | "
        f"cuda_ctx={_non_pt0 / G:.2f} "
        f"after_nccl_world={_non_pt1 / G:.2f} "
        f"nccl_world_cost={(_non_pt1 - _non_pt0) / G:.2f}",
        flush=True,
    )
```

### In `setup_device_mesh`

Before and after `init_device_mesh`:

```python
torch.cuda.synchronize()
_free_before, _total = torch.cuda.mem_get_info()
_non_pt_before = _total - _free_before

# ... existing init_device_mesh code ...

torch.cuda.synchronize()
_free_after, _ = torch.cuda.mem_get_info()
_non_pt_after = _total - _free_after
G = 1024**3
if ctx.local_rank in RANKS:
    print(
        f"[rank={ctx.rank}] init_device_mesh | "
        f"non-pt={_non_pt_after / G:.2f} "
        f"mesh_cost={(_non_pt_after - _non_pt_before) / G:.2f}",
        flush=True,
    )
```

---

## Group 3: Model Setup Prints (`pithtrain/modules/training.py`)

Add `_setup_mem(...)` calls at these 5 points in `setup_model`:

1. Before `modules = []` — `_setup_mem("before model creation")`
2. After creating module[0] — `_setup_mem("after module[0] creation")`
3. After creating module[1] — `_setup_mem("after module[1] creation")`
4. After `init_weights` loop — `_setup_mem("after init_weights")`
5. After `apply_fsdp(...)` — `_setup_mem("after apply_fsdp")`

---

## Group 4: Pipeline-Level Prints (`pithtrain/dualpipe/dualpipev.py`)

This is the most complex group. All insertions go into `DualPipeV.step()`. The code below shows every insertion with its exact anchor point.

**4A: Profiling flag setup** — insert after input scattering (`self.criterion = criterion`), before `# Step 1`:

```python
_profiling = self.memory_profiling and self.rank in RANKS
if _profiling:
    torch.cuda.synchronize()
    _m0 = _mem_gb()
    print(
        f"[rank={self.rank} pp={pp_rank}] Before pipeline: {_m0:.2f} GiB | {_mem_detail()}",
        flush=True,
    )
```

**4B: Step 1 (nF0)** — modify the existing loop body. The original loop is:

```python
for i in range(step_1):
    self._forward_chunk(0)
```

Replace with:

```python
for i in range(step_1):
    if _profiling and i == 1:
        import pithtrain.dualpipe.modeling as _mod
        _mod._layer_mem_profile = True
    self._forward_chunk(0)
    if _profiling and i == 1:
        _mod._layer_mem_profile = False
    if _profiling:
        torch.cuda.synchronize()
        print(
            f"[rank={self.rank} pp={pp_rank}] Step1 F0 i={i}: {_mem_gb():.2f} GiB (+{_mem_gb() - _m0:.2f}) | {_mem_detail()}",
            flush=True,
        )
```

After the loop, before Step 2:

```python
if _profiling:
    torch.cuda.synchronize()
    _m1 = _mem_gb()
    print(
        f"[rank={self.rank} pp={pp_rank}] After Step1 ({step_1} F0): {_m1:.2f} GiB (+{_m1 - _m0:.2f}) | {_mem_detail()}",
        flush=True,
    )
```

**4C: Step 2 (nF0F1)** — modify the loop body. Original:

```python
for i in range(step_2):
    self._forward_chunk(0, recv=False, send=False)
    self._recv_forward(0)
    self._forward_chunk(1, send=(not self.is_last_pp_rank) or (i < step_2 - 1))
    self._send_forward(0)
```

Replace with:

```python
for i in range(step_2):
    self._forward_chunk(0, recv=False, send=False)
    if _profiling:
        torch.cuda.synchronize()
        print(
            f"[rank={self.rank} pp={pp_rank}] Step2 i={i} forward_chunk(0): {_mem_gb():.2f} GiB (+{_mem_gb() - _m0:.2f}) | {_mem_detail()}",
            flush=True,
        )
    self._recv_forward(0)
    if _profiling and i == 0:
        import pithtrain.dualpipe.modeling as _mod
        _mod._layer_mem_profile = True
    self._forward_chunk(1, send=(not self.is_last_pp_rank) or (i < step_2 - 1))
    if _profiling and i == 0:
        _mod._layer_mem_profile = False
    if _profiling:
        torch.cuda.synchronize()
        print(
            f"[rank={self.rank} pp={pp_rank}] Step2 i={i} forward_chunk(1): {_mem_gb():.2f} GiB (+{_mem_gb() - _m0:.2f}) | {_mem_detail()}",
            flush=True,
        )
    self._send_forward(0)
```

After the loop:

```python
if _profiling:
    torch.cuda.synchronize()
    _m2 = _mem_gb()
    print(
        f"[rank={self.rank} pp={pp_rank}] After Step2 ({step_2} F0F1): {_m2:.2f} GiB (+{_m2 - _m0:.2f}) | {_mem_detail()}",
        flush=True,
    )
```

**4D: Step 3 (nB1W1F1)** — modify the loop body. Original:

```python
for i in range(step_3):
    self._backward_chunk(1, enable_zb=True)
    self._recv_forward(1)
    self._weight_chunk()
    self._forward_chunk(1, recv=False)
```

Replace with:

```python
for i in range(step_3):
    if _profiling:
        torch.cuda.synchronize()
        _ms3 = _mem_gb()
        print(
            f"[rank={self.rank} pp={pp_rank}] Step3 i={i} before B1: {_ms3:.2f} GiB | {_mem_detail()}",
            flush=True,
        )
    self._backward_chunk(1, enable_zb=True)
    if _profiling:
        torch.cuda.synchronize()
        print(
            f"[rank={self.rank} pp={pp_rank}] Step3 i={i} after B1: {_mem_gb():.2f} GiB (delta={_mem_gb() - _ms3:+.2f}) | {_mem_detail()}",
            flush=True,
        )
    self._recv_forward(1)
    self._weight_chunk()
    if _profiling:
        torch.cuda.synchronize()
        print(
            f"[rank={self.rank} pp={pp_rank}] Step3 i={i} after W1: {_mem_gb():.2f} GiB (delta={_mem_gb() - _ms3:+.2f}) | {_mem_detail()}",
            flush=True,
        )
    self._forward_chunk(1, recv=False)
    if _profiling:
        torch.cuda.synchronize()
        print(
            f"[rank={self.rank} pp={pp_rank}] Step3 i={i} after F1: {_mem_gb():.2f} GiB (delta={_mem_gb() - _ms3:+.2f}) | {_mem_detail()}",
            flush=True,
        )
```

After the loop:

```python
if _profiling:
    torch.cuda.synchronize()
    _m3 = _mem_gb()
    print(
        f"[rank={self.rank} pp={pp_rank}] After Step3 ({step_3} B1W1F1): {_m3:.2f} GiB (+{_m3 - _m0:.2f}) | {_mem_detail()}",
        flush=True,
    )
```

**4E: Step 4 (nF0B1F1B0)** — after the last `self._forward_backward_chunk(1, 0)` inside the loop (there is one at the end of every iteration), add:

```python
    if _profiling:
        torch.cuda.synchronize()
        print(
            f"[rank={self.rank} pp={pp_rank}] Step4 i={i}: {_mem_gb():.2f} GiB | {_mem_detail()}",
            flush=True,
        )
```

After the loop:

```python
if _profiling:
    torch.cuda.synchronize()
    _m4 = _mem_gb()
    print(
        f"[rank={self.rank} pp={pp_rank}] After Step4 ({step_4} F0B1F1B0): {_m4:.2f} GiB | {_mem_detail()}",
        flush=True,
    )
```

**4F: Steps 5-7** — no prints needed.

**4G: After Step 8** — after `assert WeightGradStore.funcs_queue.empty()`, before `self._commit_and_wait_comm()`:

```python
if _profiling:
    torch.cuda.synchronize()
    _m8 = _mem_gb()
    print(
        f"[rank={self.rank} pp={pp_rank}] After Step8 (end of pipeline): {_m8:.2f} GiB | {_mem_detail()}",
        flush=True,
    )
```

---

## Group 5: Per-Layer Prints (`pithtrain/dualpipe/modeling.py` + model file)

### 5A: In `decoder_layer_forward()` (modeling.py)

After `intermediate_tensors = IntermediateTensorsLayer()`, add:

```python
_do_lmem = _layer_mem_profile and layer.idx in DETAIL_LAYERS
_do_saved = _layer_mem_profile and layer.idx in DETAIL_LAYERS

_wt_ptrs: set = set()
if _do_saved:
    _wt_ptrs = {p.data_ptr() for p in layer.parameters()}
```

Then instrument each stage:

**Stage 1** — before the stage, add `if _do_lmem: _lmem(...)`. Wrap `layer.forward_attn(...)`:

```python
if _do_saved:
    _prof = _SavedTensorsProfiler(layer.idx, "stage1", _wt_ptrs)
    with _prof:
        output = layer.forward_attn(next_hidden_states)
    _prof.print_summary()
else:
    output = layer.forward_attn(next_hidden_states)
```

After stage1 record setup: `if _do_lmem: _lmem(f"layer{layer.idx} after stage1 (forward_attn + gate + dispatch_prep)")`

**Stage 2** — after `nvtx.range_pop()`: `if _do_lmem: _lmem(f"layer{layer.idx} after stage2 (dispatch a2a)")`

**Stage 3** — same wrapping pattern as stage 1, using `_SavedTensorsProfiler(layer.idx, "stage3", _wt_ptrs)` around `layer.forward_mlp(...)`. After: `if _do_lmem: _lmem(f"layer{layer.idx} after stage3 (forward_mlp)")`

**Stage 4** — after `nvtx.range_pop()`: `if _do_lmem: _lmem(f"layer{layer.idx} after stage4 (combine a2a)")`

**Stage 5** — same wrapping pattern around `layer.forward_aggregate(...)` with `_SavedTensorsProfiler(layer.idx, "stage5", _wt_ptrs)`. After: `if _do_lmem: _lmem(f"layer{layer.idx} after stage5 (forward_aggregate)")`

### 5B: In model decoder layer `forward_attn` (model file)

Add at the top:
```python
from pithtrain.dualpipe.modeling import _layer_mem_profile, _lmem
_do = _layer_mem_profile and self.idx in DETAIL_LAYERS
```

Add `_lmem` prints:
- Before `_forward_attn_compute`: `if _do: _lmem(f"  layer{self.idx} attn: before _forward_attn_compute")`
- After `_forward_attn_compute`: `if _do: _lmem(f"  layer{self.idx} attn: after _forward_attn_compute (LN+Attn+LN)")`
- After gate call (MoE layers only): `if _do: _lmem(f"  layer{self.idx} attn: after gate")`
- After `moe_ep_prepare_dispatch`: `if _do: _lmem(f"  layer{self.idx} attn: after dispatch_prep  sorted={tuple(sorted_tokens.shape)}")`

### 5C: In model decoder layer `forward_mlp` (model file)

Add at the top:
```python
from pithtrain.dualpipe.modeling import _layer_mem_profile, _lmem
_do = _layer_mem_profile and self.idx in DETAIL_LAYERS
```

Add `_lmem` prints:
- Before `expand_idx` gather: `if _do: _lmem(f"  layer{self.idx} mlp: before expand_idx  gathered={tuple(gathered_tokens.shape)}")`
- After `expand_idx` gather: `if _do: _lmem(f"  layer{self.idx} mlp: after expand_idx   expanded={tuple(gathered_tokens.shape)}")`
- After `scatter_for_grouped_gemm`: `if _do: _lmem(f"  layer{self.idx} mlp: after scatter      output_tokens={tuple(output_tokens.shape)}")`
- After `self.mlp.experts(...)`: `if _do: _lmem(f"  layer{self.idx} mlp: after experts      outs={tuple(outs.shape)}")`
- After unshuffle: `if _do: _lmem(f"  layer{self.idx} mlp: after unshuffle    outs={tuple(outs.shape)}")`

**Critical**: Also pass `_do_mem=_do` to the experts forward call. Change:
```python
outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
```
to:
```python
outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, _do_mem=_do)
```

### 5D: In model experts class `forward` (model file)

1. Add `_do_mem: bool = False` parameter to the `forward()` signature.
2. Add `from pithtrain.dualpipe.modeling import _lmem` at the top.
3. Add prints guarded by `if _do_mem:`:
   - Before `gate_proj`: shape of input `x`
   - After `gate_proj + activation`: shape of `g`
   - After `up_proj`: shape of `u`
   - After `g * u`: shape of `gu`
   - After `down_proj`: shape of `out`
4. **Refactor the return**: Change `return self.down_proj(g * u, ...)` to:
   ```python
   gu = g * u
   if _do_mem:
       _lmem(f"    experts: after g*u         gu={tuple(gu.shape)}")
   out = self.down_proj(gu, **kwargs)
   if _do_mem:
       _lmem(f"    experts: after down_proj   out={tuple(out.shape)}")
   return out
   ```

### 5E: In model class `forward` (model file)

In the pipeline branch (`intermediate_tensors is not None`), add:
```python
from pithtrain.dualpipe.modeling import _layer_mem_profile, _lmem
```

Then:
- After `embed_tokens` prolog: `if _layer_mem_profile: _lmem("after prolog (embed_tokens)")`
- Before norm (epilog): `if _layer_mem_profile: _lmem("before epilog (norm + lm_head)")`
- After `self.norm(...)`: `if _layer_mem_profile: _lmem("after norm, before lm_head")`
- After `self.lm_head(...)`: `if _layer_mem_profile: _lmem("after lm_head")`

---

## Group 6: Checkpoint Load Prints (`pithtrain/tasks/pretrain_lm.py`)

In `load_checkpoint`, after `dcp.load(...)`:

```python
rank = torch.distributed.get_rank()
if rank in RANKS:
    torch.cuda.synchronize()
    optim_mem = sum(
        (s._local_tensor if isinstance(s, DTensor) else s).nelement()
        * (s._local_tensor if isinstance(s, DTensor) else s).element_size()
        for state in optimizer.state.values()
        for s in state.values()
        if isinstance(s, torch.Tensor)
    )
    G = 1024**3
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    cached = reserved - alloc
    non_pytorch = total - free - reserved
    print(
        f"[rank={rank}] load_ckpt | optimizer state: {optim_mem / G:.2f} GiB "
        f"({len(optimizer.state)} param entries) | "
        f"alloc={alloc / G:.2f} cached={cached / G:.2f} non-pt={non_pytorch / G:.2f}",
        flush=True,
    )
```

Ensure `DTensor` is imported: `from torch.distributed.tensor import DTensor`. Check if it's already imported before adding.

---

## Group 7: Memory Snapshot (`pithtrain/tasks/pretrain_lm.py`)

In `train_step`, instrument the first training step with a full memory timeline snapshot.

**Before the forward/backward call** (after `model.train()`):

```python
_mem_profile = ctx.training.step == 0
_mem_snapshot = _mem_profile and torch.distributed.get_rank() in RANKS
if _mem_profile:
    model.memory_profiling = True
if _mem_snapshot:
    torch.cuda.memory._record_memory_history(max_entries=1048576)
```

**Important**: if the file already has `torch.cuda.memory._record_memory_history` for the configurable profiler (`memory_profile_start`), do NOT conflict with it. The snapshot code above is separate — it runs on step 0 unconditionally, while the configurable profiler starts at a user-specified step.

**Wrap `model.step(...)` in try/except**:

```python
try:
    loss, _ = model.step(
        global_tokens,
        num_chunks=accumulate_steps,
        criterion=criterion,
        labels=(global_labels,),
        return_outputs=False,
    )
except torch.OutOfMemoryError:
    if _mem_snapshot:
        snapshot = torch.cuda.memory._snapshot()
        from pickle import dump
        snapshot_path = f"/tmp/memory_snapshot_rank{torch.distributed.get_rank()}.pickle"
        with open(snapshot_path, "wb") as f:
            dump(snapshot, f)
        print(f"[rank={torch.distributed.get_rank()}] Memory snapshot saved to {snapshot_path}")
        torch.cuda.memory._record_memory_history(enabled=None)
    raise
```

**After the model.step call** (before optimizer step):

```python
if _mem_snapshot:
    snapshot = torch.cuda.memory._snapshot()
    from pickle import dump
    snapshot_path = f"/tmp/memory_snapshot_rank{torch.distributed.get_rank()}_step0.pickle"
    with open(snapshot_path, "wb") as f:
        dump(snapshot, f)
    print(f"[rank={torch.distributed.get_rank()}] Memory snapshot saved to {snapshot_path}")
    torch.cuda.memory._record_memory_history(enabled=None)

if _mem_profile:
    if hasattr(model, "memory_profiling"):
        model.memory_profiling = False
```

---

## Execution Order

1. `pithtrain/dualpipe/modeling.py` — Groups 1B, 5A
2. `pithtrain/dualpipe/dualpipev.py` — Groups 1A, 4
3. `pithtrain/models/<model>.py` — Groups 5B, 5C, 5D, 5E
4. `pithtrain/modules/distributed.py` — Group 2
5. `pithtrain/modules/training.py` — Groups 1C, 3
6. `pithtrain/tasks/pretrain_lm.py` — Groups 6, 7

## After Completing

Run `ruff check --fix` and `ruff format` on all modified files to ensure code style compliance.
