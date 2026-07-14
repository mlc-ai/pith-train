# The 5-Stage DualPipeV Protocol

This reference defines the contract every new model must satisfy. It also covers the model-level `forward` / `reference_forward` and how the pipeline records activations for its manual stage-by-stage backward. Keep `pithtrain/models/interface.py` and an existing model (`pithtrain/models/qwen3_moe.py` is cleanest) open alongside this file while writing.

## The five stages

Every decoder layer is split into 5 stages so the scheduler can interleave different micro-batches and overlap the compute of one with the communication of another:

| Stage | Method | Owner | What runs |
|-------|--------|-------|-----------|
| 1 | `forward_stage1` | model | LN + Attn + LN + (shared experts?) + route + prepare dispatch |
| 2 | (engine) | engine | All-to-all dispatch on comm stream |
| 3 | `forward_stage3` | model | Scatter-by-expert + grouped GEMM + unshuffle |
| 4 | (engine) | engine | All-to-all combine on comm stream |
| 5 | `forward_stage5` | model | Weighted expert sum + residual add |

Stages 2 and 4 are **not** layer methods - the engine (`pithtrain/dualpipe/execution.py`) drives the all-to-all on the comm stream. A layer implements only the three compute stages, plus a non-pipelined `reference_forward` for correctness validation.

## `LayerProtocol` (see `pithtrain/models/interface.py`)

```python
class LayerProtocol(Protocol):
    idx: int              # layer index (used by nvtx range labels)
    mlp: MLPProtocol

    def reference_forward(
        self, hidden_states, rotary_posemb, cu_seqlens=None,
    ) -> Tensor: ...

    def forward_stage1(
        self, hidden_states, rotary_posemb, cu_seqlens=None,
    ) -> Tuple[Tensor, Tensor, RoutingInfo | None]: ...

    def forward_stage3(
        self, gathered_tokens, expert_idxs=None, expand_idx=None,
    ) -> Tensor: ...

    def forward_stage5(
        self, moe_outs, moe_local_idxs, topk_weight, residual,
    ) -> Tensor: ...
```

`forward_stage1` returns `(dispatch_tokens, residual, routing)`, where `routing` is a `RoutingInfo` (from `pithtrain.models.interface`) for an MoE layer or `None` for a dense layer. `RoutingInfo` is a `NamedTuple`:

```python
class RoutingInfo(NamedTuple):
    topk_weight: torch.Tensor
    expert_idxs: torch.Tensor
    moe_local_idxs: Optional[torch.Tensor] = None
    expand_idx: Optional[torch.Tensor] = None
    dispatch_splits: Optional[AllToAllSplits] = None
    combine_splits: Optional[AllToAllSplits] = None
```

`prepare_dispatch` (`pithtrain.operators.ep_dispatch`) builds the `dispatch_tokens` tensor and the `RoutingInfo` for you. For a dense (non-MoE) layer, return `(hidden_states, residual, None)` from `forward_stage1` and run the plain MLP in `forward_stage3`.

## Stage 1: `forward_stage1` - the glue stage

Everything before the expert dispatch happens here. Split the compute-heavy prefix into a `@torch.compile(fullgraph=True)` helper (`forward_stage1_compute`) and keep the dispatch prep - which calls communication-aware Triton - in the outer eager method:

```python
@torch.compile(fullgraph=True)
def forward_stage1_compute(self, hidden_states, rotary_posemb, cu_seqlens=None):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    # SHARED EXPERTS (if any) fold into `residual` HERE, before returning, so
    # their compute overlaps the stage-2 all-to-all dispatch of routed tokens:
    #   residual = residual + self.mlp.shared_experts(hidden_states)
    topk_idx, topk_weight, lb_loss = self.mlp.gate(hidden_states)
    return hidden_states, residual, topk_idx, topk_weight, lb_loss

def forward_stage1(self, hidden_states, rotary_posemb, cu_seqlens=None):
    hidden_states, residual, topk_idx, topk_weight, lb_loss = self.forward_stage1_compute(
        hidden_states, rotary_posemb, cu_seqlens
    )
    if lb_loss is not None:
        MoELoadBalanceLossTracker.add(lb_loss)
    dispatch_tokens, routing = prepare_dispatch(
        hidden_states,
        topk_idx,
        topk_weight,
        self.mlp.num_experts,
        distributed.ep_size,
        self.mlp.experts_per_rank,
        distributed.ep_group,
    )
    return dispatch_tokens, residual, routing
```

`rotary_posemb` is a `(cos, sin)` pair built once per chunk by `Model.forward_posemb` and threaded in as an explicit argument (a sibling of `cu_seqlens`), not stashed on the layer. `cu_seqlens` is `None` for dense causal attention and set for packed / variable-length batches; the layer forwards both straight through to attention.

### Attention kernels and compile

Attention runs FlashAttention v4 (`flash_attn_func` / `flash_attn_varlen_func` from `pithtrain.operators.flash_attn_v4`) or, under context parallelism, `ring_attention_func` (`pithtrain.operators.ring_attention`). These are registered as custom ops that trace cleanly inside `forward_stage1_compute`'s `fullgraph=True` region, so no unwrap is needed. Attention with learned sinks (GPT-OSS) passes the sink parameter straight to the kernel via `learnable_sink=...`
- do **not** wrap it in a `score_mod` closure. See `compile.md`.

## Stage 3: `forward_stage3` - grouped expert GEMM

```python
def forward_stage3(self, gathered_tokens, expert_idxs=None, expand_idx=None):
    # Dense fallback (a layer whose mlp is a plain MLP, not an MoE):
    #   return self.mlp(gathered_tokens)

    if distributed.ep_size > 1:
        # Reconstruct the full token-to-expert mapping the dispatch deduplicated.
        # Use padded_index_gather, NOT raw gathered_tokens[expand_idx] - the padded
        # version is safe over the padding rows that scatter allocates.
        gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)

    output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
        scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
    )
    del gathered_tokens
    outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
    return padded_index_gather(outs, reverse_shuffle_idxs)  # not outs[reverse_shuffle_idxs]
```

**Inside the experts `forward`, truncate to `sum(ks)` before any bias-add or elementwise post-op.** `F.grouped_mm` leaves rows beyond `offs[-1]` uninitialised (often NaN), and `bias[group_ids] + NaN` propagates during backward as `0 * NaN = NaN`, poisoning bias gradients. See `pitfalls.md`.

```python
def forward(self, x, grouped_mm_offs, ks=None, ks_tensor=None):
    if ks is not None:
        actual_m = sum(ks)
        if actual_m < x.shape[0]:
            x = x[:actual_m]
    # ... rest of grouped GEMM ...
```

## Stage 5: `forward_stage5` - weighted sum + residual

```python
@torch.compile(fullgraph=True)
def forward_stage5(self, moe_outs, moe_local_idxs, topk_weight, residual):
    if distributed.ep_size == 1:
        # Non-EP: plain weighted sum over the top-k experts.
        weighted = moe_outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)
        return residual + weighted.sum(dim=1).to(moe_outs.dtype).view(*residual.shape)

    # EP path: moe_local_idxs maps combined tokens back through dedup.
    permuted_probs = topk_weight.view(-1)[moe_local_idxs]
    token_indices = moe_local_idxs // topk_weight.shape[1]
    weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
    aggregated = moe_outs.new_zeros(topk_weight.shape[0], moe_outs.shape[-1])
    aggregated.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
    return residual + aggregated.view(*residual.shape)
```

The residual here closes the decoder block. Shared-expert output was already folded into `residual` by `forward_stage1_compute`, so it is **not** re-added here. For a dense layer, `forward_stage5` receives the MLP output as `moe_outs` (with `moe_local_idxs` and `topk_weight` both `None`) and just returns `residual + moe_outs`.

## `reference_forward` - the non-pipelined path

Pure eager, no all-to-all: the same math as the three stages run under ordinary autograd. It is used for single-GPU correctness validation and by the reference model (ep=1, single stage). The layer version:

```python
def reference_forward(self, hidden_states, rotary_posemb, cu_seqlens=None):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp.reference_forward(hidden_states)  # full MoE inside
    return residual + hidden_states
```

`self.mlp.reference_forward` runs the whole MoE with `ep_size == 1` - the reference is always non-EP.

## Model-level `forward` - recording into the `ChunkRecord`

The model exposes two forward paths.

**`forward`** is the pipelined path. It delegates to `record_forward` (`pithtrain.dualpipe.execution`):

```python
def forward(self, hidden_states, cu_seqlens=None):
    return record_forward(self, hidden_states, self.chunk_record, cu_seqlens)
```

`record_forward` runs `forward_prolog` (first stage only) -> every layer's five stages -> `forward_epilog` (last stage only), stashing each stage's args and outputs into a pre-allocated `ChunkRecord` so the manual pipeline backward can backprop stage-by-stage instead of building one monolithic autograd graph. The scheduler sets `self.chunk_record` before the call.

**`reference_forward`** is the plain-autograd path used for correctness validation:

```python
def reference_forward(self, hidden_states, cu_seqlens=None):
    if self.stage_index == 0:
        hidden_states = self.forward_prolog(hidden_states)
    rotary_posemb = self.forward_posemb(hidden_states.shape[1], cu_seqlens)
    for _, layer in self.layers.items():
        hidden_states = layer.reference_forward(hidden_states, rotary_posemb, cu_seqlens)
    if self.stage_index == self.stage_count - 1:
        hidden_states = self.forward_epilog(hidden_states)
    return hidden_states
```

You implement `forward_prolog`, `forward_epilog`, and `forward_posemb`; the stage recording lives entirely in `record_forward` / `layer_forward`, so a new model never hand-rolls the record copy.

## The `ChunkRecord` structure

`ChunkRecord` (`pithtrain.dualpipe.execution`) holds `prolog`, `epilog`, and a `layers` list of `LayerRecord`. Each `LayerRecord` carries a `stage1 .. stage5` record. The stage-2 / stage-4 records hold only the all-to-all `ctx` (splits + process group) - which is why the engine, not the model, owns dispatch and combine. `create_chunk_record` pre-allocates these once and the scheduler reuses them across micro-batches for zero-allocation stepping.

## Model-level backward

There is no model `backward` method. The engine drives `record_backward` (`pithtrain.dualpipe.execution`), which runs epilog backward (via `loss.backward()` on the last stage), loops the layers in reverse through `layer_backward`, then runs prolog backward. Because it backprops through the tensors `record_forward` saved in the `ChunkRecord`, a new model needs no backward code as long as `forward_stage1` / `forward_stage3` / `forward_stage5` build ordinary autograd graphs.

## Model.__init__ requirements <a id="init-requirements"></a>

The model constructor takes `(config, phase)`. `phase` selects the V-shape role and fixes `stage_count` / `stage_index`: phase `0` is the descending leg (`stage_index = pp_rank`), phase `1` is the ascending leg (`stage_index = stage_count - 1 - pp_rank`), and phase `-1` is the non-pipelined reference (a single stage owns the whole model). Then:

- `self.stage_index`, `self.stage_count` stored for later edge checks.
- `self.chunk_record = None` - the scheduler sets it per forward.
- Layers distributed via `layer_partition(config.num_hidden_layers, stage_count, stage_index)` (import from `pithtrain.dualpipe.dualpipev`), collected into an `nn.ModuleDict` keyed by the absolute layer id as a string (required by FSDP wrapping and by weight init).
- First stage (`stage_index == 0`) has `self.embed_tokens`; last stage (`stage_index == stage_count - 1`) has `self.norm` and `self.lm_head`. All other stages set these to `None`.
- Rotary tables live on a `rotary_emb` submodule; per-forward positions are built in `forward_posemb(S, cu_seqlens)` because they depend on the input seq_len (and on `cu_seqlens` for packed batches). Do not bake them into `__init__`.

### Fail loud on unsupported parallelism dimensions

Models read their parallelism sizes from the `distributed` runtime context (`distributed.ep_size`, `distributed.cp_size`, `distributed.cp_group`, ...). A model that does not implement a dimension **must reject** a non-trivial size for it at the top of `__init__` - silently ignoring it produces wrong results the first time a real group is passed, and the bug is hard to trace.

```python
class <Prefix>Model(nn.Module):
    def __init__(self, config, phase: int):
        super().__init__()
        if distributed.cp_size > 1:
            raise NotImplementedError("<Prefix>Model doesn't support context parallelism.")
        # ... rest of __init__ ...
```

When the dimension *is* implemented (Qwen3, DeepSeek-V2 via ring attention), the context values are consumed normally in `self_attn` and `forward_posemb`. When it is not (GPT-OSS, Qwen3.5), the `NotImplementedError` converts a silent correctness bug into a loud configuration error.

**Rule:** when a new model is wired into `setup_model`, walk every parallelism dimension it could see and confirm it is either (a) genuinely used or (b) rejected when `size() > 1`. "Unused but accepted" is the hardest class of silent correctness bug to find.

## Router / Gate contract

The router class (`gate` on Qwen3 / DeepSeek-V2, `router` on GPT-OSS - match HF's spelling) must provide:

- `self.num_experts: int` - used by `prepare_dispatch` and by the load-balance loss init.
- `self.weight: nn.Parameter(shape=(num_experts, hidden_size))` - router projection weight.
- `self.load_balance_loss_fn` initialised to `None`. It is set externally by `setup_model` in `pithtrain/modules/training.py`. When present, the router computes the load-balance loss and injects its gradient onto `topk_weight`:
  ```python
  if self.load_balance_loss_fn is None:
      return topk_idx, topk_weight, None
  lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.num_experts, self.num_experts_per_tok)
  # Token-weight the injected lb gradient so train_step's 1/num_tokens grad
  # scale leaves it normalized (it bypasses the token-weighted criterion).
  topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss * topk_weight.shape[0])
  return topk_idx, topk_weight, lb_loss
  ```
- `self.router_replay` initialised to `None` (used to force a recorded routing during correctness validation).

The router's `forward` returns `(topk_idx, topk_weight, lb_loss)`. The layer calls `MoELoadBalanceLossTracker.add(lb_loss)` when `lb_loss is not None` (see `forward_stage1` above). `setup_model` locates the router by trying both attribute names:

```python
gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
```
