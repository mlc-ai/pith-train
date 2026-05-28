# The 5-Stage DualPipeV Protocol

This reference defines the contract every new model must satisfy. It also
covers the model-level `forward` / `backward` and the critical
stage-record copy pattern. Keep `pithtrain/models/interface.py` and an
existing model (`pithtrain/models/qwen3_moe.py` is cleanest) open
alongside this file while writing.

## The five stages

Every decoder layer is split into 5 stages by the scheduler:

| Stage | Method | Owner | What runs |
|-------|--------|-------|-----------|
| 1 | `forward_attn` | model | LN + Attn + LN + (shared-experts?) + Gate + EP dispatch prep |
| 2 | (framework) | framework | All-to-all dispatch on comm stream |
| 3 | `forward_mlp` | model | Scatter-by-expert + grouped GEMM + unshuffle |
| 4 | (framework) | framework | All-to-all combine on comm stream |
| 5 | `forward_aggregate` | model | Weighted expert sum + residual add |

Plus a non-pipelined `reference_forward` for testing/inference.

## `DecoderLayerProtocol` (see `pithtrain/models/interface.py`)

```python
class DecoderLayerProtocol(Protocol):
    idx: int                   # layer index (used by nvtx range labels)
    mlp: DecoderLayerMlpProtocol   # must expose ep_size + ep_group

    def reference_forward(self, hidden_states) -> Tensor: ...
    def forward_attn(self, hidden_states) -> ForwardAttnOutput: ...
    def forward_mlp(
        self, gathered_tokens, expert_idxs=None, expand_idx=None
    ) -> Tensor: ...
    def forward_aggregate(
        self, moe_outs, moe_local_idxs, topk_weight, residual
    ) -> Tensor: ...
```

`ForwardAttnOutput` is a `NamedTuple` with `sorted_tokens`, `moe_local_idxs`,
`topk_weight`, `output_splits`, `input_splits`, `expert_idxs`, `residual`,
and optionally `expand_idx`, `dedup_input_splits`, `dedup_output_splits`.
For a dense (non-MoE) layer, set the MoE fields to `None` and return
`hidden_states` as `sorted_tokens`.

## Stage 1: `forward_attn` — the glue stage

This is where everything before the expert dispatch happens. The general
shape is:

```python
@torch.compile(fullgraph=True)
def _forward_attn_compute(self, hidden_states):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Position embeddings / block masks are set on the layer before the
    # call (by Model.forward). See qwen3/gpt_oss for the pattern.
    hidden_states = self.self_attn(hidden_states, position_embeddings=..., ...)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    # SHARED EXPERTS (if any) go HERE — fold into residual BEFORE return.
    # This overlaps their compute with the stage-2 all-to-all dispatch
    # of the routed tokens.
    if hasattr(self.mlp, "shared_experts"):
        residual = residual + self.mlp.shared_experts(hidden_states)

    return hidden_states, residual

def forward_attn(self, hidden_states) -> ForwardAttnOutput:
    hidden_states, residual = self._forward_attn_compute(hidden_states)

    # Dense (no routing)
    if not hasattr(self.mlp, "experts"):
        return ForwardAttnOutput(
            hidden_states, None, None, None, None, None, residual,
        )

    # MoE routing + dispatch prep
    topk_ids, topk_weight = self.mlp.router(hidden_states)   # or .gate — match HF
    (sorted_tokens, idxs, expert_idxs, expand_idx,
     dedup_input_splits, dedup_output_splits,
     input_splits, output_splits) = moe_ep_prepare_dispatch(
        hidden_states,
        topk_ids,
        self.mlp.num_experts,
        self.mlp.ep_size,
        self.mlp.experts_per_rank,
        self.mlp.ep_group,
    )
    return ForwardAttnOutput(
        sorted_tokens, idxs, topk_weight,
        output_splits, input_splits, expert_idxs, residual,
        expand_idx, dedup_input_splits, dedup_output_splits,
    )
```

### Attention kernel unwrap pattern

If your attention uses `flex_attention` or ring attention, *those kernels
compile themselves*. Nested compile fails (`compile-inside-compile`).
The narrowly-scoped fix is to unwrap `_forward_attn_compute` **only when
the incompatible kernel is active**. See `compile.md` for full detail;
the short version in Qwen3 / DeepSeek-V2 is:

```python
if self.self_attn.use_ring_attn:
    self._forward_attn_compute = self._forward_attn_compute.__wrapped__.__get__(
        self, type(self)
    )
```

For attention with sinks (GPT-OSS pattern) — don't use `score_mod`
closures that capture learned Parameters. Use `return_aux=AuxRequest(lse=True)`
and renormalize as a post-op. See `compile.md`.

## Stage 3: `forward_mlp` — grouped expert GEMM

```python
def forward_mlp(self, gathered_tokens, expert_idxs=None, expand_idx=None):
    # Dense fallback
    if not hasattr(self.mlp, "experts"):
        assert expert_idxs is None
        return self.mlp(gathered_tokens)

    assert expert_idxs is not None
    # Use padded_index_gather, NOT raw gathered_tokens[expand_idx] — the
    # padded version is safe over the padding rows that scatter allocates.
    if expand_idx is not None:
        gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)

    output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
        scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
    )
    del gathered_tokens
    outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
    outs = padded_index_gather(outs, reverse_shuffle_idxs)   # not outs[reverse_shuffle_idxs]
    return outs
```

**Inside the experts `forward`, truncate to `sum(ks)` before the matmul
if the experts do bias-add or elementwise post-ops.** `F.grouped_mm`
leaves rows beyond `offs[-1]` uninitialised (often NaN), and
`bias[group_ids] + NaN` propagates during backward as `0 * NaN = NaN`,
poisoning bias gradients. See `pitfalls.md`.

```python
def forward(self, x, grouped_mm_offs, ks=None, ks_tensor=None):
    if ks is not None:
        actual_m = sum(ks)
        if actual_m < x.shape[0]:
            x = x[:actual_m]
    # ... rest of grouped GEMM ...
```

## Stage 5: `forward_aggregate` — weighted sum + residual

```python
@torch.compile(fullgraph=True)
def forward_aggregate(self, moe_outs, moe_local_idxs, topk_weight, residual):
    if hasattr(self.mlp, "experts"):
        if self.mlp.ep_size > 1:
            # EP path: moe_local_idxs maps back through dedup
            assert moe_local_idxs is not None
            seq_len, topk = topk_weight.shape
            permuted_probs = topk_weight.view(-1)[moe_local_idxs]
            token_indices = moe_local_idxs // topk
            weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
            hidden_states = moe_outs.new_zeros(seq_len, moe_outs.shape[-1])
            hidden_states.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
            hidden_states = hidden_states.view(*residual.shape)
        else:
            # Non-EP: just weighted sum
            assert moe_local_idxs is None
            final_out = moe_outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)
            hidden_states = final_out.sum(dim=1).to(moe_outs.dtype).view(*residual.shape)
    else:
        assert moe_local_idxs is None and topk_weight is None
        hidden_states = moe_outs

    # Residual here closes the decoder block.  Shared-expert output was
    # already folded into `residual` by _forward_attn_compute, so we do
    # NOT re-add it here.
    return residual + hidden_states
```

## `reference_forward` — the non-pipelined path

Pure eager, no compile, no all-to-all. It's used by:

- the single-GPU sanity test,
- the FSDP correctness test's reference model (ep=1, single stage),
- `Model.forward` when `ModelImplMode.use_reference_fwd = True`.

```python
def reference_forward(self, hidden_states):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, position_embeddings=...)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)        # full MoE inside .mlp.forward
    return residual + hidden_states
```

If the attention has a ring-attention path, `reference_forward` should
**disable it** (set a flag, call the non-ring branch). Tests run on a
single non-CP reference to decouple from CP correctness.

The `self.mlp.forward` called here runs `moe_infer` with `ep_size == 1`
assertion — the reference is always non-EP.

## Model-level `forward` — the stage-record copy

This is the most error-prone part of the whole integration. `Model.forward`
runs every decoder layer, but depending on whether the scheduler has
pre-allocated `IntermediateTensors`, it either returns a plain hidden
state (training reference / eager inference) or records every stage's
args/ctx/outs so backward can find them.

### The gotcha

Stages 2 and 4 (dispatch/combine) have `.ctx` only — no `.args`. Skipping
them "because they don't have `args`" silently drops the all-to-all
context and you get a cryptic "invalid gradient shape" several stages
later.

### The correct pattern

Copy this loop verbatim from Qwen3 / GPT-OSS and change only the model
class reference:

```python
from dataclasses import fields
# ...
layer_idx = 0
if self.embed_tokens is not None:
    intermediate_tensors.prolog.args = PrologArgs()
    intermediate_tensors.prolog.outs = PrologOuts(hidden_states)

for _, layer in self.layers.items():
    ret = decoder_layer_forward(layer, hidden_states)
    if len(ret) == 2:
        hidden_states, layer_record = ret
        dst = intermediate_tensors.layers[layer_idx]
        # Iterate EVERY field of the record (Stage1Record, Stage2Record, ...).
        # Do NOT special-case on hasattr(record, 'args').
        for field in fields(layer_record):
            src_rec = getattr(layer_record, field.name)
            dst_rec = getattr(dst, field.name)
            for rf in fields(src_rec):
                setattr(dst_rec, rf.name, getattr(src_rec, rf.name))
    else:
        hidden_states = ret[0]
        dst = intermediate_tensors.layers[layer_idx]
        for field in fields(dst):
            record = getattr(dst, field.name)
            for rf in fields(record):
                setattr(record, rf.name, None)
    layer_idx += 1

if self.norm is not None:
    assert self.lm_head is not None
    if not ModelImplMode.use_reference_fwd:
        hidden_states = hidden_states.detach().requires_grad_()
    intermediate_tensors.epilog.args = EpilogArgs(hidden_states)
    hidden_states = self.norm(hidden_states)
    hidden_states = self.lm_head(hidden_states)

return hidden_states
```

The "plain forward" branch (no `intermediate_tensors`) is what the
`reference_forward` path falls into when the scheduler isn't involved.

## Model-level `backward`

Static method. Runs the epilog backward first (cross-entropy → grad on
hidden_states), then loops through layers in reverse driving
`decoder_layer_backward`, then runs prolog backward via `run_backward`.
Again, copy from Qwen3 / GPT-OSS verbatim:

```python
@staticmethod
def backward(module, dy, loss, intermediate_tensors):
    assert (dy is None) != (loss is None)

    if loss is not None:
        assert module.norm is not None
        assert module.lm_head is not None
        loss.backward()
        loss.detach_()
        dy = (intermediate_tensors.epilog.args.hidden_states.grad,)
        intermediate_tensors.epilog.args = None
        loss = None
    else:
        assert module.norm is None
        assert module.lm_head is None

    dx = dy
    layers_list = [layer for _, layer in module.layers.items()]
    for layer, intermediate_tensors_layer in zip(
        reversed(layers_list), reversed(intermediate_tensors.layers)
    ):
        dx = (decoder_layer_backward(layer, dx, loss, intermediate_tensors_layer),)

    final_grads = dx
    if module.embed_tokens is not None:
        record = intermediate_tensors.prolog
        run_backward(record.outs, dx)
        for rf in fields(record):
            setattr(record, rf.name, None)
        final_grads = (None,)

    return final_grads
```

## Model.__init__ requirements <a id="init-requirements"></a>

- `self.stage_id`, `self.num_stages` stored for later checks.
- Layers distributed via `layer_partition(config.num_hidden_layers, num_stages)`.
- `self.layers` is an `nn.ModuleDict` keyed by the absolute layer id as a
  string (required by FSDP wrapping and by `init_weights`).
- First stage has `self.embed_tokens`; last stage has `self.norm` and
  `self.lm_head`. All other stages set these to `None`.
- Any "setup" tensors that decoder layers need per-forward (cos/sin
  caches, block masks, sink parameters) are set on each layer from
  `forward` before calling `decoder_layer_forward`. Do not bake them
  into the layer `__init__` (they depend on input seq_len).

### Fail loud on unsupported parallelism dimensions

`setup_model` passes every model a consistent set of process groups
(`cp_group`, any future parallelism groups). A model that doesn't
implement a dimension **must reject** a non-trivial group for that
dimension — silently ignoring the argument will produce wrong results
the first time a real group is passed, and the bug is hard to trace.

Keep the parameter in the signature for interface parity, and reject at
the top of `__init__`:

```python
class <Prefix>Model(nn.Module):
    def __init__(
        self,
        config,
        num_stages: int,
        stage_id: int,
        cp_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        if cp_group is not None and cp_group.size() > 1:
            raise NotImplementedError(
                "<Prefix>Model does not support context parallelism."
            )
        # ... rest of __init__ ...
```

When the dimension *is* implemented (Qwen3, DeepSeek-V2 with ring
attention), the parameter is used normally. When it's not (any model
that doesn't have a ring-attention path), the `NotImplementedError`
converts a silent correctness bug into a loud configuration error.

**Rule:** when a new model is wired into `setup_model`, walk every
process-group argument it accepts and confirm it is either (a)
genuinely used or (b) rejected with `NotImplementedError` when
`size() > 1`. "Unused but accepted" is the hardest class of silent
correctness bug to find.

## Router / Gate contract

The router class must provide:

- `self.num_experts: int` — used by `moe_ep_prepare_dispatch` and by
  the load balance loss init.
- `self.weight: nn.Parameter(shape=(num_experts, hidden_size))` — router
  projection weight.
- `self.load_balance_loss_fn` attribute initialised to `None`. It is set
  externally by `setup_model` in `pithtrain/modules/training.py`. The
  injector pattern is:
  ```python
  if self.training and self.load_balance_loss_fn is not None:
      lb_loss = self.load_balance_loss_fn(scores, topk_idx, num_experts, num_experts_per_tok)
      topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
  ```
- A `@torch.compile(fullgraph=True) compute(...)` that returns
  `(topk_idx, topk_weight, lb_loss)`.
- A `forward(...)` that wraps `compute(...)` and calls
  `MoELoadBalanceLossTracker.add(lb_loss)` when the loss is present.

`setup_model` locates the gate by trying both common attribute names:
```python
gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
```
Either name is fine; match HF's spelling.
