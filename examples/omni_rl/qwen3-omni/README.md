# Omni GRPO training boundary

This is the minimal M6 choice: **on-policy, token-mean GRPO with a clipped policy surrogate and k3 reference KL**. The original plan did not select an RL algorithm. This implementation selects one; it does not implement DPO, GSPO, all RL algorithms, audio/RVQ policy optimization, or a serving stack.

The trained policy predicts Thinker text tokens, optionally using explicit image/audio/video model context and encoder payloads. The caller owns policy construction, the frozen reference, rollout serving, reward calculation and weight publication. No official weights are downloaded, no rollout service is started, and no extra dependency is installed.

## Objective and alignment

The external adapter supplies aligned `[B,S]` input IDs, sampled target IDs, **old-policy selected-token log probabilities**, **frozen-reference selected-token log probabilities**, and a boolean response-token mask. It performs any teacher-forcing shift exactly once; the training objective does not shift. Prompt and padding tokens have a false mask.

For each complete local completion group, rewards are centered and divided by their population standard deviation plus `advantage_epsilon`. A group must have exactly `group_size >= 2` completions. Splitting a reward group over DP ranks is currently unsupported and rejected. Zero reward variance produces zero advantages; with valid tokens the KL term still trains. It does not imply a globally empty token batch.

For each valid token, the minimized loss is:

```text
ratio = exp(current_logprob - old_logprob)
policy = -min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage)
delta = reference_logprob - current_logprob
kl = exp(delta) - delta - 1
loss_sum = sum(policy + beta * kl)
```

All current-policy gradients remain live; rewards, advantages and old/reference log probabilities are detached. Overflow and nonfinite valid values fail explicitly. This variant averages over valid tokens across the global batch, not equally over variable-length completions.

## Actual training flow

`OmniRLTrainer.train_step()` publishes a policy snapshot, requires acknowledgement of its exact identity, obtains and validates one rollout batch, invokes the reward provider, computes advantages, and executes a real backward/optimizer/scheduler update. It increments policy version only after the update succeeds. `launch()` stops at either `max_requests` or `max_updates`, so skipped batches cannot cause an unbounded scan.

Every request binds the policy version and exported-weight digest, plus SHA256 fingerprints of the actual chat template, processor configuration, position configuration, reference policy and reward configuration. `fingerprint()` hashes explicit canonical JSON; it does not invent or retrieve those configurations. A stale batch or a mismatch is rejected before backward.

`TorchPolicyAdapter` trains a caller-owned model on one CPU or CUDA device and implements an actual model/optimizer/scheduler/RNG/provider checkpoint. Schema 2 preserves the process-global Python, NumPy and Torch CPU RNGs, plus every visible CUDA default generator if CUDA was initialized. Restore validates these states on isolated generators before changing live state, requires matching CUDA initialization/visibility, and rolls back all RNGs with model/optimizer/provider state if applying the checkpoint fails. CPU use never initializes CUDA. Old schema-1 checkpoints contain only Torch CPU RNG and are rejected because they cannot provide this continuation contract. Provider-owned generator instances must still be included in provider state.

The tests use a four-symbol rewarding rollout and AdamW, then prove exact next-update reproduction after checkpoint/resume, including dropout, Python/NumPy stochastic forwards and a stochastic rollout with independent provider generators. These toy tests are executable integration tests, not evidence that Qwen3-Omni or an external service has passed RL validation. Actual CUDA RNG continuation remains **NOT_RUN** in the CPU-only suite.

`PithTrainPolicyAdapter.from_training_context()` binds the already initialized native `training.model`, optimizers and schedulers. Its backward calls the existing `DualPipeV.step(microbatches, objective)` interface. It expects the native FSDP **SUM** gradient reduction, then divides FP32 gradients exactly once by the stage-local DP valid-token count. PP duplicates never enter that denominator; the adapter checks that token groups contain unique DP coordinates at one PP stage and that PP copies agree on token IDs, positions, masks, old/reference log probabilities, rewards and policy identity. The first version supports **CP1 and HSDP replicas=1**; it rejects other context-parallel layouts rather than guessing rollout slicing.

The native constructor also receives the actual `training_cfg`: auxiliary load-balance loss must be off and `benchmark=False` must preserve real routing. Native ranks must share the same padded batch dimensions/microbatch count; masks may contain unequal or zero valid targets. This prevents unequal FSDP collective schedules while retaining token-weighted normalization.

Every rank still executes the engine for a rank-local zero mask when other DP ranks have targets. A globally zero count uniformly skips backward, optimizer and scheduler; request and data cursors advance, while update count and policy version do not. Complete request state is checkpointed even after skips.

The native adapter binds `TaskCheckpointState` directly to production `save_checkpoint/load_checkpoint(strict=True)`. Its version-2 envelope stores request/update/data cursors and policy version separately from model/Adam/scheduler/global RNG; provider and reward states remain rank-local. Checkpoint numbering uses completed **requests**, including skipped requests. The legacy PR100 version-1 data contract remains separate. The local adapter additionally checks checksum, model layout, optimizer identity and Adam moment/step coverage before restore.

## Caller integration

The serving adapter implements `ExternalRollout.synchronize/generate/state_dict/validate_state_dict/load_state_dict`. The reward adapter implements `ExternalReward.score` and the same state methods. State validation must be pure, and restore must support rollback. Any stochastic provider must include its RNG and data position in its state.

`PolicySnapshot.payload` is caller-owned. For native training, `export_policy(version, contract)` must publish actual weights and return their digest plus a reviewed checkpoint/shard descriptor. It must be collective-safe; the task does not silently gather or copy a 30B model. `synchronize()` acknowledges installed weights, not merely an echoed request. Only a request with `rollout_leader=True` may create external generation work. PP followers receive the same batch through the provider's transport. This module checks agreement but does not implement that transport.

Context and media payloads are explicit tuples with one dictionary per microbatch. This avoids guessing how flattened image patches, audio features or video frames should be sliced. All PP ranks receive context; only PP0 receives media payloads. The caller must satisfy the native engine's `2*PP` minimum microbatch count and perform the real processor/position preparation.

The bounded contract limits requests, optimizer updates, returned samples and sequence length. It is not a wall-clock supervisor for arbitrary external Python callbacks: the serving/reward adapter must enforce its own I/O deadlines, and the launcher must retain the existing process timeout/cleanup policy.

After a failed transaction, do not retry or checkpoint the possibly partial state. Construct a fresh trainer/runtime and restore the last committed checkpoint. No automatic paid run, provider retry or external messaging occurs.

An initialized native caller wires the interfaces explicitly:

```python
from pithtrain.tasks.omni_rl import OmniRLTrainer, PithTrainPolicyAdapter

backend = PithTrainPolicyAdapter.from_training_context(
    export_policy=publisher.export_policy,
    training_cfg=pretrain_cfg.training,
)
trainer = OmniRLTrainer(rl_cfg, contract=contract, backend=backend,
                        rollout=rollout_adapter, reward=reward_adapter)
result = trainer.train_step()
backend.save(checkpoint_root, trainer)  # all ranks; request count is the step
# After constructing a fresh compatible runtime/trainer:
backend.load(checkpoint_root, trainer, request_index=saved_request_count)
```

`publisher`, rollout/reward adapters and exact configuration fingerprints in this example are required caller implementations, not preconnected services.

## Validation and remaining integration

Run the CPU suite from the repository:

```bash
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_omni_rl.py -q
```

It covers rewarding rollout → gradient/update → checkpoint/resume, independent clipped-objective arithmetic, strict fingerprints/cursors, context forwarding, and a four-rank Gloo pipeline double. The distributed test exercises PP2 × DP2 counting, a zero-target DP shard, uniform global-zero skip and collective rejection of mismatched PP batches. The pipeline double uses real autograd/optimizer/all-reduce, but is **not** an actual CUDA DualPipeV run.

A separate two-rank CPU integration invokes the actual native strict checkpoint entrypoints with the actual RL state envelope: it saves the first globally empty request with no Adam state, saves a later real update, restores each checkpoint and reproduces the next update exactly. The checkpoint transport and task-state integration are real; the CPU policy remains a toy model.

External real rollout/reward services, verl/Slime integration, GPU/native Qwen3 training, full-size memory/timing, and RL reward improvement are **NOT_INTEGRATED / NOT_RUN**. They require a real configured backend and separate validation; the CPU checks do not establish those results.
