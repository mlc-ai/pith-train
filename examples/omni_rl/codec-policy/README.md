# Local Qwen3-Omni codec-policy GRPO

This is a callable, single-device Talker/RVQ policy task. It reuses
`OmniRLCfg`, `OmniRLTrainer`, the existing clipped GRPO + fixed-reference KL
objective, and the speech task's checksum-protected conditional-Adam checkpoint.
There is no second GRPO trainer or checkpoint format.

The optimized actions are the first codec plus all 15 residual codes, followed
by optional EOS. The Thinker supplies detached, fixed features and is not a
policy-gradient target. Code2Wav is outside the discrete action gradient and is
not instantiated by this recipe. Talker input projections, trunk, first head,
and RVQ predictor remain trainable. This does not optimize generated text.

## Explicit local input

Prepare a trusted `torch.save` bundle containing only tensors and primitive
containers; the recipe uses `torch.load(..., weights_only=True)`:

```python
bundle = {
    "config": native_speech_model.config.to_dict(),
    "talker_state": initial_native_talker.state_dict(),
    "reference_state": fixed_reference_native_talker.state_dict(),
    "prompts": records,
    "upstream_contract": {
        "chat_template_sha256": template_sha256,
        "processor_sha256": processor_sha256,
        "position_config_sha256": position_config_sha256,
        "max_frames": 4,
        "max_prompt_tokens": 1024,
        "speaker": "ethan",
    },
}
```

Both weight maps are native Talker state dictionaries using the same config.
They must already be available locally; the recipe does not download weights,
convert official checkpoints, or choose a reference policy implicitly. The
example constructs FP32 models. Caller-provided digests must identify the
actual upstream template, processor and position settings.

For each already processed speech batch from `Qwen3OmniSpeechData`, extract
features using the actual native Thinker once:

```python
from pithtrain.tasks.omni_codec_rl import freeze_thinker_prompt

payload, source_sha256 = freeze_thinker_prompt(native_speech_model, batch)
record = {
    "identity": unique_prompt_id,
    "payload": payload,
    "source_sha256": source_sha256,
    "target_codes": supplied_codes,      # int64 [T, 16], all real codebooks
    "target_eos": True,                  # T < max_frames; otherwise capped False
}
records.append(record)
```

Extraction preserves caller modes and RNG, returns detached CPU tensors before
Talker projection, and binds actual Thinker weights/config, processed inputs and
extraction implementation. At conditioning, the three floating feature tensors
are cast to the Talker projection dtype without modifying their sealed original
values or source dtype. Each prompt must contain a complete assistant
boundary plus fixed conditioning text. Feature extraction is not a replacement
for Thinker text generation and does not jointly optimize the Thinker. The
pure feature, EOS/cap and target tensors are hashed into provider/reward state;
changing them prevents strict resume. The helper reads and hashes Thinker
weights, so the caller must admit its own model and host-memory requirements.

The bundled reward is **exact supplied codec-sequence and EOS match** (1 or 0).
It exercises an explicit supervised sequence-matching RL task. It is not an
ASR, perceptual speech-quality, speaker-preference or audio-quality reward.
Those goals require a separately specified, reviewed reward implementation.
This codec task currently admits only `ExactCodecReward`, whose committed-request
cursor is part of the strict checkpoint contract; the general text-RL reward
protocol alone is insufficient. Sparse exact-match targets can give all
members of a group the same reward; this is reported faithfully, not repaired
by changing targets or sampling until a favorable group appears.

## Run, pause, resume

From the repository root, with the repository installed in the selected Python
environment:

```bash
PYTHONPATH=. python examples/omni_rl/codec-policy/script.py \
  --input prepared-codec-policy.pt --output codec-step.pt --device cpu \
  --max-updates 4 --stop-after-requests 2

PYTHONPATH=. python examples/omni_rl/codec-policy/script.py \
  --input prepared-codec-policy.pt --output codec-resumed.pt --device cpu \
  --max-updates 4 --stop-after-requests 4 --resume codec-step.pt
```

Keep input, algorithm limits, seed, LR and all identities unchanged when
resuming. `--stop-after-requests` is only an execution pause; the committed
request/update/policy cursors remain part of the checkpoint. One prompt record
is consumed per request, with exactly two independent on-policy completions.
Every committed request is saved atomically. A failed transaction is not a
checkpointable update and must be recovered from the previous committed file.

For programmatic use, assemble `CodecPolicy`, `CodecPolicyAdapter`,
`LocalCodecRollout`, `ExactCodecReward`, `RolloutContract` and `OmniRLCfg`, then call
`omni_codec_rl.launch(cfg, contract=..., backend=..., rollout=..., reward=...)`.
For explicit transactions, use `omni_codec_rl.create_trainer(...)`; it validates
that training, rollout actor and fixed reference have independent parameter
storage, then constructs the existing GRPO trainer. The example exposes the
individual `trainer.train_step()` calls only to save
and pause between committed requests.

## Objective and resume contracts

- Sampling uses the full categorical distribution of first-head valid RVQ IDs
  plus EOS, and all valid residual-head IDs. No hidden temperature, top-k,
  repetition penalty or forced EOS changes the behavior policy.
- Actions are frame-major `[first, residual_1, ..., residual_15]`. EOS is a
  separate first-head action. A frame cap does not manufacture an EOS action.
  Padding has no loss; there is no second label shift.
- Autoregressive old logprobs are checked against teacher-forced probabilities
  before reward/update. Fixed reference probabilities use the same action
  alignment. The original GRPO code computes complete-group advantages and
  divides the summed gradient once by the number of actual action targets.
- An immediate EOS is one real action. Predictor-core parameters may have no
  gradient on an all-EOS request; their Adam presence/step history is preserved
  accurately. No synthetic gradients advance unused parameter counters.
- Model/optimizer/scheduler, per-parameter conditional Adam counters, RNG, modes,
  algorithm and implementation identities, frozen inputs/targets, provider and
  reward state, request/update/policy-version cursors are validated before
  strict local restoration. This local checkpoint is not production DCP.
- Missing required gradients or nonfinite gradients reject the update before
  Adam/scheduler advancement. Numerical failure after an optimizer update marks
  the transaction failed; recovery requires the previous committed checkpoint.

## Supported scope and validation

The initial implementation explicitly supports **one device, DP=CP=EP=PP=1**,
B1 unpadded feature contexts, 16 codebooks, zero attention dropout and bounded
prompt/frame/group sizes. FSDP/DTensor and multi-rank execution are rejected.
The provider checks a configurable per-copy weight bound (default 64 MiB), and
the CLI checks a configurable input-file bound (default 256 MiB). These are not
whole-process memory reservations: policy, rollout actor, fixed reference,
optimizer states and strict checkpoint snapshots coexist. This recipe does
not claim full-size 30B single-device feasibility.

Targeted CPU tests use an actual tiny native Talker and pinned installed
Transformers HF modules: independent HF selected-logprob and all-parameter VJP
parity for EOS/cap paths, nonconstant exact-match reward with nonzero update,
frozen reference/stale identity/mask/cursor negatives, and actual subprocess
resume matching uninterrupted training. Frozen feature test inputs are declared
CPU fixtures; raw-media Thinker extraction is not claimed as end-to-end CPU
validation. GPU, released weights, full-size memory/performance, distributed
codec RL and perceptual-quality evaluation remain **NOT_RUN / unsupported by
this local path**, rather than inherited passes from supervised speech tests.
