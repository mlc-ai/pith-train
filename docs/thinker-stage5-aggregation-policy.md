# Private Thinker Stage5 aggregation policy — J

This scientific candidate starts from frozen G
`93d713defdb5d3dc1e289929ad24da36eeba4d83`. It preserves G's ordinary grouped
default and adds an explicitly selected eager-aggregation compatibility policy.
It is private, unbound and unqualified for adoption or whole-model equivalence.
No original tests, thresholds, source bindings or hash constants are changed.
G, F, H, `8ce5792` and `43d0202` remain frozen. No GPU/upload/push occurred here.

## API and scope

```python
cfg.training.omni_stage5_aggregation = "grouped_sum"  # default
# Explicit Thinker-only alternative:
cfg.training.omni_stage5_aggregation = "hf_eager_expert_id_serial"
```

The public config uses a two-value Literal enum, not a bool or an HF projection
backend selector. Invalid strings/bools reject. Setup rejects the serial option
for non-Thinker models. Text and media Thinker layers and SparseMoeBlock capture
the policy once. Standalone native Thinker users initialize
`training.omni_stage5_aggregation` before construction; the captured attribute
has the same name. Native `SparseMoeBlock.reference_forward` honors it with the
same aggregation semantics as the staged path. Talker's existing independent
serial/shared-expert arithmetic is unchanged.

HF5.17's public `experts_implementation` / internal `_experts_implementation`
select a full expert backend. The independently audited full HF TextModel
defaults to `grouped_mm` (subject to capability fallback), while a standalone
Experts module with unset config uses its eager body. Explicit eager executes
the expert loop with rounded weighted index_add; grouped_mm restores products
to original slots and sums K with FP32 accumulation. Backend names alone do not
prove kernel/projection/gather/wgrad equivalence. J's field controls aggregation
only and makes no such broader backend claim.

## Default preservation and optional ownership

`grouped_sum` retains G's BF16 product, original K sum and residual order.
Default Stage1 returns the original six-field RoutingInfo and never invokes the
new metadata factory, clone, argsort or wrapper. Its reference return still uses
the original `_combine_expert_sum`; the rounding witness remains 1.03125.

Only serial mode copies original global router IDs and computes a per-token
argsort. Dispatch/dedup/GEMM rows remain unchanged. After returned original
token/slot positions are restored, output/weight pairs are gathered by global
expert order and products/additions round serially; the same witness is 1.0.
Source global IDs are retained before any use of localized Stage3 expert IDs.
No mutable last-route layer attribute is used.

An optional frozen Stage5RoutingInfo wraps the original route, owns metadata,
delegates its six-field iteration/index/attributes/_asdict, and preserves
metadata through `_replace`. The original RoutingInfo class and three-field
Stage5Args remain unchanged. Every original Stage5 backward tensor-grad lane
remains unchanged. The real gather/autograd graph retains the integer argsort
needed by backward, so Stage5Args needs no metadata lane.

Existing four-argument model Stage5 methods remain unchanged. Only Thinker
adds `forward_stage5_with_metadata`; image forwards its DeepStack lanes in that
optional entry. The execution machinery selects that capability only for the
optional wrapper; missing capability or policy/metadata disagreement rejects.
No unrelated model signatures change.

Runtime routing consumers were audited: they use route attributes and splits,
not RoutingInfo isinstance checks or serialization/pytree transforms. The
optional wrapper remains a pytree leaf/owner; actual compiled Stage5 receives
the tensor metadata separately. Synthetic trusted pickle/deepcopy and legacy
adapter behavior are tested. Routes are ephemeral, not checkpoint payloads.

## Numerical identity and directed CPU results

The captured enum is included in the existing model numerical contract;
absent/grouped/serial identities differ. Actual task checkpoint envelope
validation rejects all six cross-policy combinations. Production distributed
resume includes the enum in TrainingCfg JSON and hashes the new source.
No old binding is enrolled. Legacy standalone StrictResumeState, which does
not bind model numerical semantics generally, remains outside this qualification.

Existing local CPU environment, one intra/inter-op thread, one compile worker,
and a 120-second cap per file were used. Receipts are under
`workspace/stage5-policy-evidence/`:

| Unchanged default test file | Result |
| --- | --- |
| test_qwen3_omni_stage5.py | 13 PASS / original 2 BF16 forced-eager output FAIL |
| test_omni_stage5_compile.py | 7 PASS |
| test_omni_talker_reference_isolation.py | 3 PASS |
| test_omni_thinker_expert_sum.py | 4 PASS |

All five old grouped/image expectations remain passing. The original forced-
eager default failures remain FAIL, not fixed or relabeled by the new option.

New `test_omni_stage5_aggregation_policy.py`: **51 PASS**. This includes 32
explicit serial-mode actual-HF shared-projection boundary cases (24 BF16 and
eight FP32 eager), two additional compiled BF16 actual-HF cases, real EP1
Stage4 followed by reverse-microbatch backward after GC, wrapper collection and
graph-only argsort lifetime, original 6/3 tuple adapters, `_replace`, `_asdict`,
pickle/deepcopy, invalid policy rejection, captured reference behavior, public
setup scope, and cross-policy checkpoint validation.

Projection graphs in those boundary cases belong to HF; native whole-MoE
projection/gather/dgrad/wgrad equivalence is not thereby established. Real EP2,
FSDP/PP, optimizer continuation, whole-model numerics and performance remain
unqualified. The preceding FP32 compiled weight-VJP exact discrepancy is not
fixed: its existing NSE/close diagnostics and all H/8ce/43 failure seals remain
unchanged. J changes neither precision nor SIMD settings to hide it. Optional
metadata and serial aggregation add work; default performance and optional
end-to-end cost have not been measured on GPU.
