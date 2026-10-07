# PithTrain Architecture

A developer-oriented tour of how PithTrain is put together. The goal is that after reading this you can open any file in `pithtrain` and know roughly where you are, what it talks to, and why it exists.

PithTrain is a compact, agent-native MoE training framework. Production frameworks maximize model/feature/hardware coverage; PithTrain deliberately trades that away for a codebase small enough that a coding agent or human being can read end-to-end. The design favors local readability over cross-model reuse: it avoids plugin registries and runtime specs, so what runs at a given call site can usually be found by reading the code rather than tracing indirection. Keep that principle in mind; it explains many structural choices below.

If you are extending the framework (adding a model, operator, or feature), read this first, then see `CONTRIBUTING.md`.

## 1. The three layers

The codebase is organized into three layers:

| Layer | Directory | Responsibility |
|---|---|---|
| Application | `pithtrain/tasks` | End-to-end workflows: pretraining, corpus tokenization, checkpoint conversion. |
| Engine | `pithtrain/{pipeline,models,modules,contexts}` | The bulk of PithTrain: pipeline scheduler, model implementations, and distributed/training infrastructure. |
| Operator | `pithtrain/operators` | Fused Triton / library-backed kernels for compute- and communication-critical paths. |

Everything sits on top of PyTorch (NCCL, FSDP2, DCP, `torch.compile`), with external kernel libraries (DeepGEMM, FlashAttention) and a Python kernel DSL (Triton) at the operator layer.

### Directory map

A high-level map (one representative file noted per area; the directories hold more, and file names drift, so treat these as entry points rather than an inventory):

```
pithtrain/
├── tasks/     # APPLICATION: launchable entry points (pretrain_lm.py)
├── pipeline/  # ENGINE: DualPipeV scheduler + F/B overlap (dualpipev.py, execution.py)
├── models/    # ENGINE: one file per model family (qwen3_moe.py); interface.py is the contract
├── modules/   # ENGINE: distributed + training infra (distributed.py, training.py)
├── contexts/  # ENGINE: runtime state (distributed.pp_group, training.model)
└── operators/ # OPERATOR: fused kernels
```

The sections below drill into the parts that carry the most architecture: the model contract, the pipeline engine, and the parallelism mesh.

## 2. The central abstraction: the 5-stage decoder layer

Everything in the engine is organized around one idea. A transformer decoder layer is split into five stages, cut at the expert-parallel communication boundaries. This split is what lets the pipeline overlap one micro-batch's compute with another's communication.

| # | Stage | What happens | Where it runs |
|---|---|---|---|
| 1 | Attention | LayerNorm → Attention → LayerNorm → expert routing (top-k selection) | compute stream |
| 2 | Dispatch | all-to-all: send each token to the rank holding its expert | comm stream |
| 3 | MLP | expert / MLP computation on the received tokens | compute stream |
| 4 | Combine | all-to-all: gather expert outputs back to the originating rank | comm stream |
| 5 | Aggregate | weighted sum of expert outputs + residual connection | compute stream |

Stages 2 and 4 (the all-to-alls) run on a communication stream, so the scheduler can hide them behind the stage-1/3/5 compute of a different micro-batch.

This split is reflected directly in the model contract in `pithtrain/models/interface.py`. Every model layer implements:

```python
class LayerProtocol(Protocol):
    idx: int
    mlp: MlpProtocol

    def forward_stage1(self, hidden_states, rotary_posemb, cu_seqlens=None) -> tuple[Tensor, Tensor, RoutingInfo | None]: ...
    def forward_stage3(self, gathered_tokens, expert_idxs, expand_idx) -> Tensor: ...
    def forward_stage5(self, moe_outs, moe_local_idxs, topk_weight, residual) -> Tensor: ...
    def reference_forward(self, hidden_states, rotary_posemb) -> Tensor: ...
```

Stages 2 and 4 (dispatch/combine) are framework-owned. The layer doesn't implement them; it hands the scheduler the routing metadata (the `RoutingInfo` returned by `forward_stage1`) and the scheduler drives the all-to-alls. The model-level contract is just:

```python
class ModelProtocol(Protocol):
    stage_index: int
    stage_count: int
    layers: Dict[str, LayerProtocol]

    def forward_prolog(self, hidden_states) -> Tensor: ...
    def forward_epilog(self, hidden_states) -> Tensor: ...
    def forward_posemb(self, S, cu_seqlens=None) -> tuple[Tensor, Tensor]: ...
    def reference_forward(self, hidden_states) -> Tensor: ...
```

See `pithtrain/models/qwen3_moe.py` for a complete, readable implementation of this contract.

## 3. DualPipeV: the pipeline engine

`pithtrain/pipeline` is the heart of the framework, derived from DeepSeek's [DualPipe](https://github.com/deepseek-ai/DualPipe) with the compute-communication overlap added on top.

V-shaped placement. Instead of one contiguous slice of layers per rank, the model is cut into `2 x pp_size` chunks arranged in a "V": rank `r` holds chunk `r` and chunk `2 x pp_size - 1 - r`. That is why `DualPipeV` is built from a pair of modules, and it is what keeps each rank busy on both the forward and backward sweep (reducing the pipeline bubble). When the layers don't divide evenly across the pipeline, the edge chunks get fewer transformer layers, since they also carry the embeddings and the language-model head.

## 4. Distributed parallelism: two views of one rank space

`pithtrain/modules/distributed.py` is the single source of truth for ranks. It builds **two** device meshes over the same world, so that attention and the experts each get the parallelism they want:

| Axis | Controlled by | What it shards | How it works |
|---|---|---|---|
| PP | `pipeline_parallel_size` | model layers | DualPipeV + P2P |
| CP | `context_parallel_size` | the sequence | ring attention (zigzag layout) |
| DP | derived | each batch | [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) |
| EP | `expert_parallel_size` | MoE experts | all-to-all dispatch/combine |
| DP (expert view) | derived | the expert weights | FSDP2, over the expert replicas |

This is **MoE parallel folding** ([arXiv:2504.14960](https://arxiv.org/abs/2504.14960) section 3.2). One pipeline stage owns `world_size // pp` ranks; the attention view factors that block as `(dp, cp)` and the expert view re-factors the very same block as `(dp, ep)`:

```
attn_mesh:  rank = pp_rank * stage  +  dp_rank * cp  +  cp_rank
expt_mesh:  rank = pp_rank * stage  +  dp_rank * ep  +  ep_rank
```

so `dp * cp == ep * expt_dp == world_size // pp`. PP is the one axis the two views share, and it stays outermost in each, which is the whole correctness argument: every rank agrees with itself about the pipeline stage it holds, so the attention half and the expert half of a layer live on the same device. `cp` and `ep` each need only divide the stage size, not each other, so EP may span CP and `ep` ranges over any divisor of `world_size // pp`.

Each view puts its high-traffic axis innermost, so the CP group and the EP group are *both* contiguous rank blocks. They are two labellings of the same rank index, so neither costs the other anything: keeping `cp` and `ep` each within a node keeps the ring K/V exchange and the MoE all-to-all on NVLink, independently of one another.

Which rank reads which data is `dp_rank`, and only `dp_rank`. `ep_rank` names the experts a rank hosts and the all-to-all peers of that rank, and says nothing about data. The transition between the two regions costs no communication: the MoE layer flattens `(B, S, hidden)` into a batch of tokens and routes each one independently, so it does not care whether the tokens on a rank arrived by a batch split (DP) or a sequence split (CP).

What FSDP shards over. There are two parameter classes. The **expt** parameters are the routed expert weights, unique per EP rank and replicated across the `dp` axis of the expert view, so FSDP shards them over that axis. The **attn** parameters are everything else (attention, router, shared experts, embeddings, `norm`, `lm_head`), replicated across the whole stage, so FSDP shards them over the flattened `dp x cp`. That is two `fully_shard` calls per layer, and the FSDP2 module walk stops at an already-sharded submodule, so the two parameter classes land in disjoint reduction groups. (That is `hsdp_replica=1`, the default. Above 1, both replica groups split into that many replicas, and FSDP shards within one and all-reduces across them; raise it when one replica already holds the model.) The per-parameter-class mesh selection is in `apply_fsdp` in `pithtrain/modules/training.py`.

Gradients. The two classes reduce over groups of different size, yet either *sum* is already the true global gradient. The attn group is the whole stage. The expert replica group covers all the data too, because every member already receives the tokens of a whole EP group through the all-to-all. So `apply_fsdp` puts every parameter on a plain-sum reduction (`set_gradient_divide_factor(1.0)`), and the training step applies the one and only normalization, dividing by the global token count. An average, which is the FSDP2 default, would instead leave expert gradients `ep_size` times too large.

## 5. FP8 training

FP8 matmuls are backed by DeepSeek's [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), which does 128-element block-scaled FP8 GEMMs (E8M0 power-of-2 scales on both -- native MXFP8 PTX on Blackwell, emulated on Hopper). PithTrain wraps it in `pithtrain/operators/linear.py` and `pithtrain/operators/grouped_linear.py`, with the block-scaling quantization in custom Triton kernels (`pithtrain/operators/deepgemm_quantize.py`).

A whole model flips between FP8 and BF16 through one flag, `TrainingCfg.fp8`. At training setup (`pithtrain/modules/training.py`) it binds the linear classes the models build with, published on the `training` context:

```python
training.Linear        = FP8Linear        if cfg.fp8 else nn.Linear
training.GroupedLinear = FP8GroupedLinear if cfg.fp8 else GroupedLinear
```

## 6. Checkpointing

`pithtrain/modules/checkpoint.py` bridges two representations, saved via [PyTorch DCP](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html):

- Canonical (on disk): parallelism-independent, with global layer names and each expert stored individually.
- Localized (in memory): what the running model holds, laid out for the current PP/EP layout.

Saving converts localized to canonical, loading converts back. Because the on-disk format is parallelism-independent, a checkpoint is reshardable: you can resume the same run under a different PP/EP/DP layout. The HuggingFace import path produces a model-only checkpoint (no optimizer/scheduler), loaded non-strictly.

## 7. Operators

`pithtrain/operators` holds the performance-critical kernels, many wrapped by [PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) with hand-written autograd. Wrapping a kernel in a custom operator lets `torch.compile` treat it as one opaque node; PithTrain compiles all transformer computation with `torch.compile(fullgraph=True)` except the MoE expert computation, whose per-expert shapes are data-dependent under EP. Full-graph mode is deliberate: it turns a silent graph break into a compile error.

A few of the most important, roughly in order:

1. `ep_dispatch.py`: fused Triton kernels for expert-parallel token dispatch with deduplication; central to MoE routing and the all-to-all overlap.
2. `ring_attention.py`: zigzag, causal-balanced ring attention for context parallelism (standard, packed and MLA-aware variants).
3. `deepgemm_quantize.py`: fused block-scaled FP8 quantization behind the FP8 training path.
4. `token_scatter.py`: groups tokens per expert ahead of the grouped GEMM.

The rest are smaller fused activation, loss, and attention-wrapper kernels.

## 8. Configs and contexts

PithTrain pairs a declarative config with a runtime context, roughly one-to-one. A `*Cfg` holds user-provided, serializable knobs (a `SlottedDefault` subclass in `pithtrain/config.py`), and the top-level `PretrainLMCfg` composes them. Before training, a `setup_*(cfg)` function constructs the matching context from it.

A context is a module under `pithtrain/contexts` holding that state as module-level globals: process groups, the device mesh, the built model, the resolved linear classes. Much of it is set up once and constant thereafter, yet needed deep in the call tree. A CP group, for instance, would otherwise thread from the training loop through the model, the decoder layer, and into attention. So instead of passing such state down every call site, any file reads it directly:

```python
from pithtrain.contexts import distributed
cp_group = distributed.cp_group
```

Follow the same `*Cfg` + `setup_*` + `contexts` shape when adding a subsystem.

## 9. Agent skills

PithTrain is agent-native: recurring framework procedures ship as agent skills under `.agents/skills`, not just as prose. Each is a scoped playbook with explicit prerequisites, so a coding agent runs the procedure rather than re-deriving it. When you add a workflow that will be repeated, ship it as one.
