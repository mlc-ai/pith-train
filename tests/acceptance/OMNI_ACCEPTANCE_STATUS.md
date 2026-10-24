# Qwen3-Omni native model and training acceptance

This branch consolidates the native model/training work on the reviewed prepared-data
and text-model foundations (#100 and #101). RL/GRPO and rollout orchestration are
tracked separately in the PithRL migration. The target is Qwen3-Omni, with Transformers
5.17.0 and released model
configuration revision `26291f793822fb6be9555850f06dfe95f2d7e695`.

## Implementation

- Native Thinker text/MoE/mRoPE, vision/DeepStack, audio, video and aligned media inputs.
- Talker and its residual-code predictor, differentiable Code2Wav, bounded speech generation.
- SFT and speech losses/readers, model-only initialization, full checkpoint conversion,
  optimizer/data/RNG recovery and checkpoint-to-independent-generation loading.
- Normal dispatch is fused; stable token layout and deterministic attention are explicit
  options through `TrainingCfg.deterministic` / `strict_resume`. The layout is captured
  when the model is constructed and bound into its checkpoint numerical contract.
- The text decoder retains #101's compiled stage1 normalization and stage5. Media decoder
  stage1, and audio stage5, retain the eager boundaries used by their existing numerical
  evidence. Image stage5 retains its fullgraph DeepStack wrapper. Routing stays outside
  the text stage1 compile boundary; CE, FP32 master gradients and expert-order BF16
  accumulation fixes are preserved in both dispatch modes.

## Evidence and remaining gates

Results below are bound to their tested source. The integrated source
`2003a0e381767a966c951d80130e92ddcd0b6525` has current CUDA evidence from Runs133/134/136;
older component results remain historical evidence, not reruns on that source.
The integrated branch is not fully accepted: G2/G3 numerical failures and the
unrun released-model training gate remain open.

| Capability | Existing evidence | Current limitation |
|---|---|---|
| Data/reader and text foundations | #100/#101 core validation and supported topology gates | New assembly inheritance is checked by source mapping, not a new full GPU run. |
| Released payload conversion | Run49: six CPU gates, 28,010 keys and 15 shards, bitwise round-trip | This is conversion, not full-size training. |
| Released Thinker inference | Run97: four comparisons, 48 layers / 235 tokens, NSE 0 | No released-model optimizer/update acceptance. |
| Released Talker and Code2Wav | Run77 component outputs, NSE 0 | Does not certify the combined speech training path. |
| Sharded waveform training | Run98: 108 parameters, four updates, fresh-process recovery | Only the recorded tiny fixture and supported DP path. |
| Checkpoint-to-generation | Run98: 351 checkpoint parameters / 702 slices; four text tokens, eight codec frames, 16 codebooks, 122 waveform values | Bounded fixture; not full speech EOS/quality coverage. |
| Talker pipeline optimizer parity (G2) | Run133 on integrated source `2003a0e3`: all 87 pre-clip and post-clip gradients passed; 85/87 updates passed, **two updates failed** | Three-step continuation and fresh-process restore remain blocked at step 0. |
| Native speech checkpoint recovery | Run134 on `2003a0e3`: 243 trainable parameters, four text+codec updates and exact fresh-process continuation from step 2 through steps 3/4 | Single-rank unsharded tiny fixture; does not clear sharded/HF G2/G3 parity. |
| Combined speech training parity (G3) | Run136 on integrated source `2003a0e3`: 241/243 gradients and 232/243 updates passed per rank; **two gradient and 11 update failures** | Second scan and fresh-process continuation remain blocked; original thresholds unchanged. |
| Released full training/recovery (G1) | **Not run** | Candidate admission blocked by an OS-denied local install-log inspection; no instance created. |

Run100 is source `67ccad1e40d73c90ccb014d7c77744720e9325e8`. Run106 is source `e3ff7d01f3ea5aa9086902ce5099790a94e70e2e`; the 52 speech dependency files were
byte-equal to `1c80c59377043d1274861e9cf8ab1e174666c63f`. Source equality preserves
the failures as well as the passes. Existing per-parameter reports remain authoritative.

Run133 used integrated source `2003a0e381767a966c951d80130e92ddcd0b6525`, the tiny
first-code-only Talker trunk/head, BF16 compute with FP32 master/gradient/Adam state,
and PP2/EP2/CP1 on four H100 GPUs. The remaining input-norm update NSE values are
0.0151594 and 0.0112954 against the unchanged 0.01 threshold. The 87-parameter
gradient/update reports match the earlier Run129/132 reports exactly. Dense operand
captures reproduce the local norm VJPs, FP32 accumulation and Adam deltas; captured
CE and MoE branch accumulation have different BF16 rounding boundaries. Expert-internal
capture was incomplete, so the partial diagnosis does not establish a complete causal
fix and does not convert either failure into a pass.

Run134 used the same integrated source and existing native speech task on one H100.
It trained text and codec losses over five synthetic media inputs, saved at step 2,
and restored in a different process. Steps 3/4 matched uninterrupted training exactly
for model, optimizer, scheduler, RNG, reader/task state and the next inputs. Original
G2/G3 HF comparison failures remain independent and unchanged. Temporary synthetic
checkpoints were destroyed with the stopped instance after retaining hashes and the
complete recovery comparison evidence.

Run136 ran the unchanged combined speech gate on two H100s, DP2/CP1/EP1/PP1,
using the five-media tiny fixture and the original independent HF prefix oracle.
All original per-parameter and post-update functional reports exactly reproduce
Run106/135. The first-code output differences remain, particularly for text and
image inputs; an aggregate loss or a small absolute gradient does not waive them.
The repaired diagnostic extractor retained both complete 94-parameter Talker
states and source/fixture-bound inputs and outputs for offline attribution. This
successful extraction is not a model acceptance result. Run135's extraction error
and missing diagnostic payload remain recorded separately.

The prior matched-state predictor replay locates a BF16 reduction difference
between the native parallel causal sequence and the independent HF prefix graphs.
No incorrect derivative or justified minimal production repair has been established
by that observation. Current-source post-update attribution remains diagnostic;
original numerical assertions, parameter selection and thresholds are unchanged.

The main integration retains media context and upstream inference-record cleanup.
Its 20 affected CPU checks and all eight pre-commit checks passed. The current CUDA
results above cover the named paths only, not every historical component or scale.

## Supported boundaries

Native media uses BF16 activations and FP32 master/gradient/Adam state, CP1 and explicit
unpacked/unpadded media records. Only documented/tested PP/EP combinations are claimed;
the full composite speech training path does not support arbitrary CP/EP/PP layouts.
Speech training checkpoints can initialize a separate unsharded generation model.
Native FSDP generation and in-flight KV-cache/stream-state checkpoint restoration are
not implemented. Released 30B end-to-end training has not passed. Caller-provided speech
codes/data are required; this PR does not add an RVQ tokenizer or a reward model.

Do not turn failed normalized-error checks into passes by changing denominators,
freezing parameters, forcing routing, or citing a small aggregate loss difference.
The original fast-path strict noise ratios are retained as reproducibility diagnostics;
ordinary numerical correctness and strict repeatability remain distinct contracts.
