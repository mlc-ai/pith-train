# Qwen3-Omni RL and rollout acceptance

This feature depends on the native Qwen3-Omni model/training PR. Its separate production
surface is `omni_rl`, `omni_codec_rl`, `omni_hf_rollout`, `omni_hf_transport`, and
`omni_rollout_data`; checkpoint/data/model contracts are reused from the model PR.

## Implementation

- Native GRPO optimizer loop and strict policy, scheduler, reader and checkpoint identities.
- Codec-policy RL with explicit caller-provided rewards and code/data contracts.
- Owned HF rollout worker with bounded installation, IPC/publication limits and cleanup.
- Optional sequential policy/reference GPU residency without changing parameter precision.

## Existing source-bound evidence

Historical executions are not new GPU runs on this assembled commit. The five production
modules and restored GRPO gate come byte-for-byte from
`1c80c59377043d1274861e9cf8ab1e174666c63f`; the native dependency closure is separately
mapped to the model PR and its text/media compilation boundary changes require CUDA checks.

| Gate | Recorded result and scope |
|---|---|
| Run85 native GRPO/provider | PASS on exact PR123 `5d6314ee3e0bf805357d8627c86146dda3a0ee5a`: tiny PP2/EP2, three updates and fresh-process restore. |
| Run88 codec policy | PASS on the recorded local FP32 fixture, two updates and fresh restore; not distributed FSDP codec-policy acceptance. |
| Run108 released HF interface | PASS on exact `1c80c59`: 30.53B public weights, two requests, generation, old/reference log-probabilities, reward call and sequential residency; no optimizer updates. |
| Run108 cleanup | PASS: worker/child/IPC cleanup, one GPU resident model at a time, instance stopped. |
| Native BF16 publication to released HF | NOT TESTED by Run108; full native training/checkpoint/rollout is not implied by the public HF interface. |

Run108 used Transformers 5.17.0, the released config revision recorded in its sealed receipt,
and original FP32 public weights. Its request timeout was explicitly 1,200 seconds; the
production default remains 300 seconds. Reward values were both zero and do not constitute
task-quality validation. CPU limits, publication size and loader admission remain enforced.

## Remaining boundaries

The native model PR's G2/G3 update/gradient failures remain open and are not made passes by
this split. Full released-model end-to-end RL acceptance remains outstanding. Codec-policy
RL is local; multi-rank FSDP speech/codec RL, arbitrary composite CP/EP/PP and native FSDP
generation are not implemented. Rewards and RVQ/audio tokenization remain caller-defined.
Training resume does not claim restoration of an in-flight generation cache or stream.
