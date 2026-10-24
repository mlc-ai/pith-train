# Qwen3-Omni SFT and complete-module speech training

These tasks use caller-created models, optimizers, process groups and readers.
They do not download weights, choose an external codec encoder, or initialize
distributed communication implicitly. They support Qwen3-Omni only.

## Assistant-only Thinker SFT

`pithtrain.tasks.sft_lm.launch` uses the existing initialized
`pithtrain.contexts.training` engine and its optimizers/schedulers. The engine's
`step(microbatches, objective)` must accumulate and reduce **SUM** gradients.
The objective consumes already shifted assistant-only labels, computes local CE
sums, then divides reduced FP32 gradients once by the global target count.
Supply an explicit nonduplicated DP x CP process group for each PP stage; PP and
EP copies must not multiply target counts. The task collectively validates the
initialized runtime PP/DP/CP coordinates and exact group membership before
counting; invalid local labels and mismatched PP-copy totals reject all ranks.

```python
from pithtrain.tasks.sft_lm import SFTLMCfg, launch

cfg = SFTLMCfg()
cfg.max_updates = 100
cfg.max_batches = 150
cfg.moe_load_balance_coef = 0.0

# The caller has initialized training.model/optimizers/schedulers, device,
# dp_cp_group, and a Qwen3OmniSpeechSFTData reader.
state = launch(
    cfg, sft_data, device=device, token_group=dp_cp_group,
    checkpoint_root=checkpoint_root,
    checkpoint_contract=stable_model_data_topology_contract,
    resume_scan=None,  # Set an explicitly selected committed scan to resume.
)
```

The optional native checkpoint path uses strict task schema v2, saving model,
optimizer, scheduler, global RNG, scan/update cursors and the actual reader state.
The explicit contract must identify the caller's model/data/topology configuration;
the task additionally binds its objective, clipping and replica settings. Native
checkpoint directory numbers count committed **scans**, while Adam/scheduler
counters count successful **updates**. Zero-target scans advance the reader/task
scan cursor and can be saved with zero optimizer updates. No model step, weight
decay or scheduler step happens for a globally empty SFT batch. A zero-local rank
still counts and participates in a nonempty global engine step.

A callback-only `checkpoint=` option is also available; its caller is responsible
for persisting all runtime/reader state. It is not a substitute for the native
strict checkpoint path above.

Nonzero `moe_load_balance_coef`, nonzero runtime gate injectors, and unknown
injectors are rejected collectively before forward. Auxiliary objectives require
their own formula, masks and counts. `auxiliary_loss_sum` exposes that separate
sum/count boundary but does not wire the existing MoE gradient injector into SFT.

## Complete-module speech task

`pithtrain.tasks.omni_speech.launch` is an eager autograd task using the native
`Qwen3OmniMoeSpeechModel` interface and `Qwen3OmniSpeechData.get_batch`, which returns
one dictionary per unpadded microbatch. It explicitly selects model forward keys;
metadata and text labels are not blindly passed into the model.

| Mode | Trainable components | Objective |
| --- | --- | --- |
| `codec_only` | Talker, including residual predictor | First-head/EOS and residual codec CE |
| `text_and_codec` | Thinker and Talker | Codec CE plus separately normalized assistant text CE |
| `waveform` | Code2Wav | Required caller-supplied differentiable sum/count objective |

Call `configure_trainable_components` **before** optimizer construction and before
DDP wrapping or FSDP setup. Construct the optimizer after FSDP replaces
parameters with DTensors. The task checks that optimizer membership is exactly the active
components, with no frozen parameters or duplicates. It rejects unsupported
cross-component parameter aliases. Frozen components run in evaluation mode.

```python
import torch
from pithtrain.tasks.omni_speech import (
    OmniSpeechCfg, configure_trainable_components, launch,
)

cfg = OmniSpeechCfg(
    mode="text_and_codec", codec_weighting="combined", text_weight=1.0,
    max_updates=100, max_batches=150,
)
# model is the complete native Qwen3OmniMoeSpeechModel on the explicit device.
parameters = configure_trainable_components(model, cfg.mode)
optimizer = torch.optim.AdamW(parameters, lr=1e-5, foreach=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
state = launch(
    cfg, speech_data, model=model, optimizer=optimizer, scheduler=scheduler,
    device=device, token_group=dp_group, gradient_reduction="sum",
    checkpoint_path=local_checkpoint_path,
    resume_from=None,
)
```

The model supplies aligned `first_logits[B,T+1,V0]`/`first_labels`, with first-head
EOS supervision, and `residual_logits[B,T,G-1,Vr]`/`residual_labels`. No objective
shifts either set again. First and residual vocabularies may differ. Combined
weighting divides summed codec losses by the total global codec target count.
`per_head` gives equal weight to globally nonempty heads, each normalized by its
own global count. Text loss has its own global count and explicit `text_weight`.
Masks and `-100` targets affect counts as well as loss. Externally prepared codec
labels and their provenance are mandatory; there is no fabricated native encoder.

Waveform mode calls only `model.decode_training_codes(codes.transpose(1, 2))`.
Provide `waveform_objective(predicted_waveform, batch) -> TargetLoss(loss_sum,
target_count)` and a stable `cfg.waveform_objective_id`. The count is an int64
scalar and the sum must be differentiable. The caller owns the waveform formula,
length/padding mask and target alignment. This task does not invent an official
Code2Wav training loss or differentiate through discrete codec generation.

Plain models use explicit manual DP reduction. Declare `sum` or `mean`; the task
compensates for mean reduction exactly once. A real DDP wrapper must enable
`find_unused_parameters=True`, use the same DP group, and declare its actual
reducer: standard DDP uses `mean`; `sum` requires a caller-installed SUM hook.
The wrapper path currently allows one codec microbatch per scan. Use manual DP
for waveform training or multiple microbatches. Globally unused manual-DP
parameters retain `grad=None` and do not receive decay. For a global-zero DDP
scan, a zero backward completes the reducer cycle, then original gradients,
buffers, modes and RNG are restored; no optimizer/scheduler update is committed.

## Sharded complete-model speech training

The declared FSDP2 path requires the complete ordered WORLD mesh named `dp`,
with CP=EP=PP=1. `apply_speech_fsdp` shards native units from leaves to root and
registers the Thinker, residual predictor and waveform direct-method entries.
Keep FP32 master parameters before setup; compute parameters are BF16 and
gradient reduction is FP32 **SUM**. The task never repeats that reduction or
multiplies it by world size. It clips using the FP64 sum of local shard squared
norms and rejects nonfinite gradients on every rank before any update.

```python
from torch.distributed.device_mesh import init_device_mesh
from pithtrain.modules.omni_speech_fsdp import apply_speech_fsdp

# Initialize the process group first; create the complete native model with
# FP32 master parameters using the caller's explicit initialization/load plan.
configure_trainable_components(model, cfg.mode)
mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp",))
apply_speech_fsdp(model, mesh, mode=cfg.mode)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5, foreach=False,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
state = launch(
    cfg, speech_data, model=model, optimizer=optimizer, scheduler=scheduler,
    device=device, token_group=mesh.get_group(), gradient_reduction="sum",
    checkpoint_root=checkpoint_root,
    checkpoint_contract=stable_model_data_topology_contract,
    resume_scan=None,
)
```

This first FSDP path requires B=1 per microbatch and the same collective call
plan on all ranks: microbatch count, zero/nonzero codec-frame branch, codebook
count and modality presence must agree. Supervision masks and valid target counts
may differ. Unsupported plans reject collectively before forward. A globally
empty objective still completes zero backward and resharding, then restores
logical gradients, buffers, modes and RNG; it advances only the scan cursor.

FSDP uses native strict task-schema checkpoints via `checkpoint_root` and
`resume_scan`, including actual sharded model/optimizer, scheduler, RNG and
reader/task state. Its explicit conditional AdamW policy records each parameter's
state presence and update counter; absent/less-frequently-used parameters must
not acquire fabricated gradients or counters. Native checkpoint numbers count
committed scans; optimizer/scheduler counters count successful updates. Do not
use the ordinary replica `checkpoint_path`/`resume_from` arguments with FSDP.

## Checkpoint and platform boundaries

Speech checkpoints are an independent, local, `torch.load(weights_only=True)`
format for ordinary complete-module AdamW/SGD. A streaming SHA256/length footer
protects the complete payload; explicit per-parameter state presence and exact
update-counter records also reject missing or rewound conditional optimizer state.
The checksum detects corruption, not a malicious writer able to regenerate it. They are **not** the production
DualPipeV/FSDP strict checkpoint format. Save is atomic per rank. Files are suffixed
with the DP rank in distributed runs. Resume validates exact model names, order,
shapes/dtypes, optimizer membership and policy, scheduler counters, task/data
identity and counters, RNG and fixed topology before mutation. Every rank must
agree on the global saved transaction. Local save-preflight or writer failures
reject all ranks. A failed distributed save may leave individual peer files and
must not be treated as a successful transaction. Apply errors trigger rollback on all ranks.
Use scan boundaries with no pending gradients. Resume does not reshard or remap
parameters. CPU rollback snapshots require host memory proportional to full state.

Current speech topology is **CP=EP=PP=1**. Ordinary DP keeps full replicas;
only models configured by the explicit speech FSDP helper may use the sharded
branch. Arbitrary DTensor/FSDP wrappers are rejected. These APIs do not establish
that a specific full-size initialization, weight load, optimizer configuration or
checkpoint fits the available GPU/host memory; plan and validate those peaks
separately. Nonzero injected MoE auxiliary losses are rejected.

CPU tests exercise independent text/codec objectives, local/global zero targets,
SUM/mean manual and real DDP, three-mode optimizer updates, exact update-2 resume,
strict corruption rejection/rollback, all-rank nonfinite-gradient rejection before
optimizer/scheduler updates, and native SFT Gloo/DTensor checkpoint
integration. Real CPU FSDP task tests also compare two Adam updates against an
independent BF16 functional reference with global head counts and sharded clipping,
plus native strict zero-update and update-2 checkpoint/resume. Those task tests use
a small protocol model and explicitly isolate the native-only batch validator;
native ChatML/media admission has separate tests and native end-to-end acceptance.
They are protocol/mathematics evidence; native GPU numerical parity, full-size
training, multi-GPU FSDP and performance validation remain separate gates.

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_omni_sft_objectives.py \
  tests/test_omni_sft_task.py \
  tests/test_omni_speech_task.py
```


## Independent generation from a native speech checkpoint

`read_speech_generation_state` reads the complete native model from a committed
DP-only speech task-v2 checkpoint. Run it in a separate generation process with
the same model configuration, implementation and numerical attributes. Construct
the complete native `Qwen3OmniMoeSpeechModel` on `meta`, with the same trainable
component settings as the checkpoint, before reading. The caller supplies the
manifest SHA from its completed checkpoint receipt and explicit host/file caps.

```python
import torch
from pithtrain.modules.omni_speech_generation_checkpoint import read_speech_generation_state

# model is a fresh complete native meta model; no FSDP wrapping in this process.
state, receipt = read_speech_generation_state(
    model, committed_checkpoint_directory,
    expected_manifest_sha256=completed_checkpoint_manifest_sha256,
    max_state_bytes=planned_full_fp32_model_bytes,
    max_rank_file_bytes=planned_largest_rank_file_bytes,
)
model.load_state_dict(state, strict=True, assign=True)
del state
model.to(device=device).eval()  # FP32 masters remain owned by this fresh model.

# Match the existing acceptance precision: BF16 compute parameters while
# retaining native buffer dtypes. A method call needs a forward dispatch wrapper.
class Generate(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

call = Generate(model)
with torch.no_grad():
    parameters = {name: p.to(torch.bfloat16) for name, p in call.named_parameters()}
    result = torch.func.functional_call(
        call, (parameters, dict(call.named_buffers())), (input_ids, model_context),
        dict(max_text_tokens=max_new_tokens, max_codec_frames=max_codec_frames,
             do_sample=False), strict=True,
    )
```

The reader validates every rank and all model tensor layouts/checksums before
allocating the owned full CPU state. It checks the files again while assembling;
it never mutates the target, optimizer, gradients or RNG. It uses mmap for rank
payloads, but the caller still needs host capacity for the full FP32 state plus
page-cache/transient overhead and device capacity for the unsharded FP32 model
plus BF16 compute copies and activations. The state cap is not a whole-process memory limit. Loading with `assign=True` is for
a fresh generation model, before constructing any optimizer.

This bridges a training checkpoint to independent generation. It does not resume
training, restore a generation KV cache or continue an in-flight stream. The
reader supports the native complete-model DP mesh with Shard(0), CP=EP=PP=1,
HSDP replicas=1; other layouts reject. The generation implementation still
recomputes prefixes and rejects FSDP-wrapped modules. The CPU regression uses a
real two-rank FSDP/task-v2 checkpoint, exact full-parameter reconstruction and a
real Code2Wav forward; it does not establish full-size or CUDA generation parity.
