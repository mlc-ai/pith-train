# `@torch.compile(fullgraph=True)` — The Three Hot Regions

Three per-layer regions must be wrapped with
`@torch.compile(fullgraph=True)`:

1. `_forward_attn_compute` — LN + attention + LN (+ shared experts if any)
2. `router.compute` / `gate.compute` — top-k + softmax + load-balance injection
3. `forward_aggregate` — weighted expert sum + residual add

These are not recommendations. They are enforced by the framework: test
failures and performance regressions have previously traced back to
missing or weakened compile coverage on these regions.

## Why `fullgraph=True` specifically

`fullgraph=True` *forces* Dynamo to raise on any would-be graph break.
`fullgraph=False` is strictly worse for this codebase:

- **Compile boundaries accumulate bf16 rounding drift.** Each sub-graph
  gets its own compile; crossing back and forth adds rounding that the
  single-shot fullgraph trace avoids.
- **Cross-region fusion is missed.** The LN → matmul fusion, the
  weighted-sum + residual fusion, etc., happen *because* the whole region
  is one graph.
- **Breakage is hidden.** Without `fullgraph=True` you don't learn that
  a new attention kernel silently self-compiles until a microbench
  shows the speedup is gone.

**Never reach for `fullgraph=False` as a workaround.** If a region can't
compile fullgraph, unwrap the region entirely (see below) and treat the
unwrap as tech debt.

## The three regions — boilerplate

### Region 1: `_forward_attn_compute`

```python
@torch.compile(fullgraph=True)
def _forward_attn_compute(self, hidden_states):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, position_embeddings=..., ...)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    # Shared experts fold into residual here, before returning.
    if hasattr(self.mlp, "shared_experts"):
        residual = residual + self.mlp.shared_experts(hidden_states)

    return hidden_states, residual
```

### Region 2: router / gate `compute`

```python
class <Prefix>TopKRouter(nn.Module):   # or <Prefix>Gate — match HF
    @torch.compile(fullgraph=True)
    def compute(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        logits = F.linear(hidden_states, self.weight, self.bias)  # bias optional
        topk_logits, topk_idx = torch.topk(logits, k=self.num_experts_per_tok, dim=-1, sorted=True)
        topk_weight = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

        if self.training and self.load_balance_loss_fn is not None:
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            lb_loss = self.load_balance_loss_fn(
                scores, topk_idx, self.num_experts, self.num_experts_per_tok,
            )
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states):
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        return topk_idx, topk_weight
```

### Region 3: `forward_aggregate`

```python
@torch.compile(fullgraph=True)
def forward_aggregate(self, moe_outs, moe_local_idxs, topk_weight, residual):
    # ... weighted sum branches — see protocol.md ...
    return residual + hidden_states
```

## When you MUST unwrap — the attention kernel problem

Some attention kernels compile themselves:
- `flex_attention` (always, when called)
- Ring attention with internal `torch.compile`

Nested compile fails: `torch._dynamo.exc.Unsupported: compile-inside-compile`.

### Narrow, conditional unwrap (the pattern)

Unwrap **only when the incompatible kernel is active**. Qwen3 and
DeepSeek-V2 handle ring attention this way:

```python
class <Prefix>DecoderLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... build sub-modules ...

        # Unwrap only if the incompatible attention kernel is selected.
        if self.self_attn.use_ring_attn:
            self._forward_attn_compute = self._forward_attn_compute.__wrapped__.__get__(
                self, type(self),
            )
```

Two reasons this is narrow:

1. Most runs don't hit the unwrap (context parallel is off).
2. The gate should depend on the *kernel selection*, not on a blanket
   "always unwrap because flex is sometimes used" — that loses the
   compile benefit for configurations that could have it.

Similarly, `setup_model` in `pithtrain/modules/training.py` unwraps the
gate's `compute` when CP is on, because the all-reduce injected by the
global-batch load balance loss breaks fullgraph tracing:

```python
if cp_group is not None:
    gate.compute = gate.compute.__wrapped__.__get__(gate, type(gate))
```

### The `__wrapped__.__get__` idiom

`@torch.compile` replaces the method with a wrapped object. The original
function is at `.__wrapped__`. Re-binding it to the instance is done
with `.__get__(self, type(self))`. After the unwrap, calling
`self._forward_attn_compute(x)` runs the eager Python body.

### Don't unwrap unconditionally

A blanket unwrap breaks performance for all configurations, not just
the one with the problematic kernel. If the unwrap covers 100% of the
attention compute, you've lost ~10–30% step-time at short context and
~5–15% at long context. Treat every unwrap as tech debt; record a brief
in `docs/` describing what would let us re-land the compile (e.g.
"upstream flex_attention fix for nested compile").

## The attention-sinks / learned-bias-in-closure problem

If your attention needs a learned parameter inside the softmax (GPT-OSS
attention sinks, some ALiBi variants), **do not** use a `score_mod`
closure that captures the Parameter. The closure specialises the kernel's
self-compile on the Parameter, and that specialisation is what forces
our outer `fullgraph=True` to unwrap.

### The LSE-renormalisation pattern

Move the Parameter out of the closure and apply its effect as a pure
post-op. For sinks:

```python
# Do NOT do this: closure over self.sinks forces nested compile.
# def score_mod(score, b, h, q, kv): return torch.where(kv == seq_len, self.sinks[h], score)
# attn_output = flex_attention(q, k, v, score_mod=score_mod, ...)

# Do this instead: return the LSE and renorm as a pointwise post-op.
attn_output, aux = flex_attention(
    q, k, v,
    block_mask=block_mask,
    scale=self.scaling,
    enable_gqa=True,
    return_aux=AuxRequest(lse=True),
)
lse = aux.lse
# softmax([scores, sink])_i = softmax(scores)_i * sigmoid(lse − sink)
sink_scale = torch.sigmoid(lse - self.sinks.float().view(1, -1, 1))
attn_output = attn_output * sink_scale.to(attn_output.dtype).unsqueeze(-1)
```

This is mathematically identical to including a virtual KV row at
`v = 0` with per-head score `self.sinks[h]`. The sink doesn't
contribute to the numerator (it has `v = 0`), only to the denominator
(the `exp(sinks[h])` term). The `sigmoid(lse - sinks)` factor folds that
denominator change into the output.

**Before writing the refactor yourself, check upstream.** HuggingFace
Transformers and TorchTitan have both converged on this pattern for
GPT-OSS independently. HF's `integrations/flex_attention.py` has the
literal comment that score_mod cannot implement sinks correctly.

## Inference-time compile (the seq-len-grows problem)

At training time, seq_len is constant → one compile total.

At **autoregressive inference**, seq_len grows by 1 every step. If the
test harness passes the current `[batch, prompt_len + step, hidden]`
tensor to `_forward_attn_compute`, Dynamo retraces every step. With
flex_attention inside, each retrace can take tens of seconds.

### Do NOT fix this by weakening the model's compile

Don't add `dynamic=True` to the model's `@torch.compile` decorator. The
modeling code must stay identical to what training uses — see
`feedback_modeling_code_matches_training`. An inference-only compile
flag on production code means tests stop exercising the training path.

### Fix it in the test harness with static-seq-len decode

Allocate a `[batch, prompt_len + max_new_tokens]` buffer up front, fill
initial prompt tokens, advance a cursor each step, and always pass the
full-size buffer through the model:

```python
max_seq_len = prompt_len + max_new_tokens
buffer = torch.full((batch, max_seq_len), pad_id, dtype=torch.long, device=device)
for i, t in enumerate(encoded_prompts):
    buffer[i, :prompt_len] = t[:prompt_len].to(device)
cursor = prompt_len
set_p2p_tensor_shapes([(1, max_seq_len, hidden_size)])   # ONCE, outside the loop

for step in range(max_new_tokens):
    loss, outputs = dualpipev.step(buffer if ctx.pp_rank == 0 else None, ...)
    next_tok = outputs[:, cursor - 1, :].float().argmax(dim=-1)   # logit at last real pos
    buffer[:, cursor] = next_tok
    cursor += 1
```

Forward cost per step is higher (you always process `max_seq_len`
positions), but you trade O(max_new_tokens) compiles for one. In
practice this cuts a multi-minute test to tens of seconds.

See `templates/inference_test.py` for the full harness.

## Debugging checklist

| Symptom | Likely cause |
|---------|-------------|
| `Unsupported: compile-inside-compile` | Nested `@torch.compile` — unwrap outer region *conditionally* on the kernel that self-compiles |
| Inference wall-clock balloons after first few tokens | Per-seq-len retrace — fix with static-seq-len decode, not `dynamic=True` |
| Graph breaks reported at each step | `fullgraph=False` is silently catching them — switch to `fullgraph=True` and fix |
| Step time 30% slower than an earlier branch | Someone removed a `@torch.compile` — check the three hot regions |
| `calc_diff` passes but worst bias grad looks wrong | Compile drift on small-magnitude params — see `testing.md` on label scaling |
