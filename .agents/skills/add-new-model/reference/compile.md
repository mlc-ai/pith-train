# `@torch.compile(fullgraph=True)` - The Compiled Compute Regions

The per-layer compute stages are wrapped with `@torch.compile(fullgraph=True)`:

1. `forward_stage1_compute` - LN + attention + LN (+ shared experts if any)
2. `forward_stage5` - weighted expert sum + residual add

`forward_stage3` (the grouped expert GEMM + all-to-all glue) is **not** compiled: it calls communication-aware Triton and the dedup gather, which live outside the traced region. GPT-OSS additionally decorates its router `forward` (top-k + softmax + load-balance injection); Qwen3 and DeepSeek-V2 leave the gate eager. Match whichever the reference model you are copying does, and keep the two stage-compute regions compiled.

These are not recommendations. They are enforced by the framework: test failures and performance regressions have previously traced back to missing or weakened compile coverage on these regions.

## Why `fullgraph=True` specifically

`fullgraph=True` *forces* Dynamo to raise on any would-be graph break. `fullgraph=False` is strictly worse for this codebase:

- **Compile boundaries accumulate bf16 rounding drift.** Each sub-graph gets its own compile; crossing back and forth adds rounding that the single-shot fullgraph trace avoids.
- **Cross-region fusion is missed.** The LN -> matmul fusion, the weighted-sum + residual fusion, etc., happen *because* the whole region is one graph.
- **Breakage is hidden.** Without `fullgraph=True` you don't learn that a new attention kernel silently self-compiles until a microbench shows the speedup is gone.

**Never reach for `fullgraph=False` as a workaround.** If a region can't compile fullgraph, unwrap the region entirely (see below) and treat the unwrap as tech debt.

## The compiled regions - boilerplate

### Region 1: `forward_stage1_compute`

```python
@torch.compile(fullgraph=True)
def forward_stage1_compute(self, hidden_states, rotary_posemb, cu_seqlens=None):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, rotary_posemb, cu_seqlens)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    # Shared experts fold into residual here, before returning.
    if hasattr(self.mlp, "shared_experts"):
        residual = residual + self.mlp.shared_experts(hidden_states)

    topk_idx, topk_weight, lb_loss = self.mlp.gate(hidden_states)
    return hidden_states, residual, topk_idx, topk_weight, lb_loss
```

The `MoELoadBalanceLossTracker.add(lb_loss)` call and the dispatch prep stay in the *eager* `forward_stage1` wrapper - see `protocol.md`.

### Region 2 (optional): router / gate `forward`

GPT-OSS compiles its router; if you follow that pattern, the whole method must trace fullgraph, including the load-balance injection:

```python
class <Prefix>TopKRouter(nn.Module):   # or <Prefix>Gate - match HF
    @torch.compile(fullgraph=True)
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states, self.weight, self.bias)  # bias optional
        topk_logits, topk_idx = torch.topk(logits, k=self.num_experts_per_tok, dim=-1, sorted=True)
        topk_weight = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

        if self.load_balance_loss_fn is None:
            return topk_idx, topk_weight, None
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        lb_loss = self.load_balance_loss_fn(
            scores, topk_idx, self.num_experts, self.num_experts_per_tok,
        )
        topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss * topk_weight.shape[0])
        return topk_idx, topk_weight, lb_loss
```

### Region 3: `forward_stage5`

```python
@torch.compile(fullgraph=True)
def forward_stage5(self, moe_outs, moe_local_idxs, topk_weight, residual):
    # ... weighted-sum branches - see protocol.md ...
    return residual + aggregated.view(*residual.shape)
```

## Attention kernels trace cleanly - no unwrap needed

Attention runs FlashAttention v4 (`flash_attn_func` / `flash_attn_varlen_func`) or, under context parallelism, `ring_attention_func`. These are registered as custom ops, so they trace as opaque nodes inside `forward_stage1_compute`'s `fullgraph=True` region: the models decorate that method **unconditionally** and call the kernels inside it, with no conditional unwrap. There is no `flex_attention` in the model path and no `compile-inside-compile` problem to work around.

### If a future kernel *does* self-compile

Some kernels compile themselves (e.g. `flex_attention`). A nested compile fails: `torch._dynamo.exc.Unsupported: compile-inside-compile`. The narrowly-scoped fix is to unwrap the region **only when the incompatible kernel is active** - never a blanket unwrap, and never `fullgraph=False`. `@torch.compile` replaces the method with a wrapper whose original function is at `.__wrapped__`; re-bind it to the instance to run the eager body:

```python
if self.self_attn.uses_self_compiling_kernel:
    self.forward_stage1_compute = self.forward_stage1_compute.__wrapped__.__get__(
        self, type(self),
    )
```

A blanket unwrap breaks performance for every configuration, not just the one with the problematic kernel. If the unwrap covers 100% of the attention compute you lose ~10-30% step-time at short context and ~5-15% at long context. Treat every unwrap as tech debt and record a brief in `docs/` describing what would let us re-land the compile.

## Attention sinks - pass them to the kernel, not a closure

If your attention needs a learned parameter inside the softmax (GPT-OSS attention sinks, some ALiBi variants), pass it straight to the FlashAttention v4 kernel as an argument - GPT-OSS uses `learnable_sink=`:

```python
sinks = self.sinks.to(query_states.dtype)
attn_output = flash_attn_func(
    query_states, key_states, value_states,
    softmax_scale=self.scaling, causal=True,
    window_size=window_size, learnable_sink=sinks,
)
```

**Do not** implement sinks with a `score_mod` closure that captures the Parameter. The closure specialises a self-compiling kernel on the Parameter, and that specialisation is exactly what would force our outer `fullgraph=True` to unwrap. Handing the sink to the kernel as a plain tensor argument keeps `forward_stage1_compute` fully compiled.

## Inference-time compile (the seq-len-grows problem)

At training time, seq_len is constant -> one compile total.

At **autoregressive inference**, seq_len grows by 1 every step. If the test harness passes the current `[batch, prompt_len + step, hidden]` tensor through the model, Dynamo retraces `forward_stage1_compute` every step, and each retrace can take many seconds.

### Do NOT fix this by weakening the model's compile

Don't add `dynamic=True` to the model's `@torch.compile` decorators. The modeling code must stay identical to what training uses. An inference-only compile flag on production code means tests stop exercising the training path.

### Fix it in the test harness with static-seq-len decode

Allocate a `[batch, prompt_len + max_new_tokens]` buffer up front, fill the prompt tokens, advance a cursor each step, and always pass the full-size buffer through the model:

```python
max_seq_len = prompt_len + max_new_tokens
buffer = torch.full((batch, max_seq_len), pad_id, dtype=torch.long, device=device)
for i, t in enumerate(encoded_prompts):
    buffer[i, :prompt_len] = t[:prompt_len].to(device)
cursor = prompt_len
set_p2p_tensor_shapes([(1, max_seq_len, hidden_size)])   # ONCE, outside the loop

for step in range(max_new_tokens):
    loss, outputs = dualpipev.step(buffer if distributed.pp_rank == 0 else None, ...)
    next_tok = outputs[:, cursor - 1, :].float().argmax(dim=-1)   # logit at last real pos
    buffer[:, cursor] = next_tok
    cursor += 1
```

Forward cost per step is higher (you always process `max_seq_len` positions), but you trade O(max_new_tokens) compiles for one. In practice this cuts a multi-minute test to tens of seconds. See `templates/inference_test.py` for the full harness.

## Debugging checklist

| Symptom | Likely cause |
|---------|-------------|
| `Unsupported: compile-inside-compile` | A kernel self-compiles - unwrap `forward_stage1_compute` *conditionally* on that kernel |
| Inference wall-clock balloons after first few tokens | Per-seq-len retrace - fix with static-seq-len decode, not `dynamic=True` |
| Graph breaks reported at each step | `fullgraph=False` is silently catching them - switch to `fullgraph=True` and fix |
| Step time 30% slower than an earlier branch | Someone removed a `@torch.compile` - check `forward_stage1_compute` and `forward_stage5` |
| `calc_diff` passes but worst bias grad looks wrong | Compile drift on small-magnitude params - see `testing.md` on label scaling |
