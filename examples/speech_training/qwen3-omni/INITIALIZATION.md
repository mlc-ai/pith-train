# Released full-speech initialization

`pithtrain.modules.omni_speech_initialization` connects the full-component
checkpoint converter to the native meta/FSDP speech model. It initializes every
Thinker, Talker/code-predictor and Code2Wav parameter. It does not restore an
optimizer, task cursor or RNG state; use the task's strict checkpoint resume for
that operation.

The supported released source is `Qwen/Qwen3-Omni-30B-A3B-Instruct`, revision
`26291f793822fb6be9555850f06dfe95f2d7e695`. Both APIs use local files only. They do
not download, upload, allocate cloud resources or select a training dataset.

## Prepare standard DCP once

Supply a separately reviewed JSON provenance record with these exact fields:

```json
{
  "kind": "released",
  "model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
  "config_sha256": "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d",
  "files": {
    "config.json": {"bytes": 0, "sha256": "REPLACE_WITH_ACTUAL_HASH"}
  }
}
```

The `files` example is incomplete and deliberately not executable. List **every**
regular file in the input directory, including the exact config, index and 15
shards, with actual byte sizes and SHA256 values. The public revision's config,
28,010 HF keys, shapes, shard ownership and BF16 dtypes are checked. The caller
must establish the file hashes' provenance; deriving hashes from unknown files
does not establish that they came from the release. Tests use the explicit
`kind="test_fixture"`, which is reported as such and is not released acceptance.

```python
import json
from pithtrain.modules.omni_speech_initialization import prepare_speech_initialization

prepared = prepare_speech_initialization(
    "/local/released-snapshot", "/local/new-speech-dcp",
    expected_source=json.load(open("/local/reviewed-source.json")),
    max_tensor_bytes=1 << 30,  # Example per-canonical-tensor admission, not total RAM.
)
# Preserve prepared["manifest_sha256"] outside the new DCP directory.
```

The converter audits all headers before materializing payload. It resolves one
canonical tensor at a time, slicing fused expert stacks before loading them.
The standard PyTorch DCP writer uses one tensor per file to avoid its per-file
tensor cache retaining the model. The existing `app.model` canonical names and
`qwen3_omni_components.json` sidecar remain compatible with the full converter.
An additional sidecar section pins the source and all output DCP file hashes.
The sidecar is written last. A failed preparation leaves an incomplete directory
that must not be reused as a valid checkpoint.

## Initialize each rank before creating an optimizer

Use the existing native distributed/training setup with PP=EP=CP=1. Build all
three native components on `meta` with FP32 master dtype. Configure the task's
active components before wrapping, then materialize only local shards:

```python
import json
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from pithtrain.models.qwen3_omni_multimodal import Qwen3OmniMoeThinkerMultimodalModel
from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration
from pithtrain.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
from pithtrain.models.qwen3_omni_speech import Qwen3OmniMoeSpeechModel
from pithtrain.modules.omni_speech_fsdp import apply_speech_fsdp, materialize_speech_shards
from pithtrain.modules.omni_speech_initialization import initialize_speech_from_dcp
from pithtrain.tasks.omni_speech import configure_trainable_components

# Run after the existing setup_distributed/setup_training; use their CUDA device.
config = Qwen3OmniMoeConfig(**json.load(open("/local/released-snapshot/config.json")))
with torch.device("meta"):
    model = Qwen3OmniMoeSpeechModel(
        config,
        Qwen3OmniMoeThinkerMultimodalModel(config.thinker_config, -1),
        Qwen3OmniMoeTalkerForConditionalGeneration(config.talker_config),
        Qwen3OmniMoeCode2Wav(config.code2wav_config),
    )
configure_trainable_components(model, "text_and_codec")
mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
apply_speech_fsdp(model, mesh, mode="text_and_codec")
materialize_speech_shards(model, device=torch.device("cuda", torch.cuda.current_device()))
receipt = initialize_speech_from_dcp(
    model, "/local/new-speech-dcp",
    expected_manifest_sha256=prepared["manifest_sha256"],  # Same externally pinned value on all ranks.
    rank=dist.get_rank(), world_size=dist.get_world_size(),
    max_local_bytes=20 << 30, max_tensor_bytes=1 << 30,  # Example DP8 staging caps; admit actual resources separately.
)
# Only now construct the optimizer and launch the native speech task.
```

The loader checks config, complete canonical inventory, all native parameters and
buffers, exact rank/world/Shard(0) layout, dtype, shape, file hashes and finite CPU
candidates before the first model copy. Every rank hashes the files in its own actual directory before staging and
again after reading, so equal sidecars on different local mounts cannot hide
different payload bytes. Every rank agrees on failure. The
initialized marker is set only after all ranks finish copying. A failed copy
requires discarding the model. Current native rotary/audio-position/codec-offset
constants are computed from config and are not checkpoint tensors; unknown
buffers, including nonpersistent ones, are rejected instead of fabricated.

CPU staging is each rank's local FP32 state plus a canonical DCP read buffer and
finite-check boolean workspace. All ranks' CPU candidates coexist. GPU local
masters already exist, and training adds its own activations/gradients/optimizer
state. The numeric caps in the example are not a guarantee that a device or host
fits. DCP serialization/filesystem overhead and disk space for both input and
output must also be admitted. Each rank performs two complete DCP hash scans, including on a shared mount;
the implementation does not assume a shared cache or trust rank 0's filesystem.
For DP8 and the released BF16 payload, these scans alone request at least
1,128,314,193,440 bytes (about 1.026 TiB) of aggregate logical reads, plus DCP
serialization/metadata bytes. Local shard loading adds further reads. Hashing
uses a 1 MiB streaming buffer per rank, not a full-file memory allocation.

The caller owns the process group and its timeout. Its collective timeout must
cover the slowest complete hash/staging phase and rank arrival skew. The outer
process deadline must cover both all-rank scans, loading, copying and cleanup.
Do not reuse a previous rank-0-only I/O/time estimate: admit the actual storage,
hash throughput and complete phase budget first. No full-size scan timing has
been measured by the tiny CPU test, and this API does not silently extend an
existing deadline or authorize a cloud allocation.

## Validation boundary

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m pytest tests/test_omni_speech_initialization.py -q
```

The focused tests exercise real tiny full-component conversion, standard DCP,
two-rank CPU/Gloo FSDP loading and exact local readback, including empty shards
and failure before application. The CPU environment without Triton executes the
actual Thinker constructor AST, not its accelerator kernels or forward method.
These tests do not establish released-payload numeric correctness, full-size GPU
training, full-speech fresh-update resume, or HF rollout-provider memory fit.
The separate released-Thinker driver's 512-token video-data admission gap remains.
