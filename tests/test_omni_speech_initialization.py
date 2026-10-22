"""Real tiny full-component conversion and CPU FSDP local initialization.

Thinker constructors are executed from their unchanged class AST because the
CPU test environment has no Triton. No Thinker kernel/forward is substituted or
claimed executed. Native audio/vision/Talker/Code2Wav and FSDP/DCP are real.
"""

import ast
import copy
import io
import json
import math
import shutil
import weakref
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from pithtrain.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
from pithtrain.models.qwen3_omni_speech import Qwen3OmniMoeSpeechModel
from pithtrain.models.qwen3_omni_talker import Qwen3OmniMoeTalkerForConditionalGeneration
from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVideoEncoder, Qwen3OmniMoeVisionEncoder
from pithtrain.modules import omni_speech_initialization as initialize
from pithtrain.modules.omni_speech_fsdp import apply_speech_fsdp, materialize_speech_shards
from pithtrain.tasks.convert_checkpoint import qwen3_omni_full as full
from tests import test_omni_full_converter as fixtures

ROOT = Path(__file__).resolve().parents[1]


def source_identity(path, kind="test_fixture"):
    return dict(
        kind=kind,
        model_id=initialize.MODEL,
        revision=initialize.REVISION,
        config_sha256=initialize._sha(path / "config.json"),
        files=initialize._files(path),
    )


def native_meta(raw):
    """Real native constructors, no full allocated state and no accelerator import."""
    ns = dict(
        torch=torch,
        nn=nn,
        partial=partial,
        math=math,
        distributed=SimpleNamespace(pp_size=1, pp_rank=0, ep_size=1, cp_size=1),
        training=SimpleNamespace(fp8=False),
        Qwen3OmniMoeAudioEncoder=Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeVisionEncoder=Qwen3OmniMoeVisionEncoder,
        Qwen3OmniMoeVideoEncoder=Qwen3OmniMoeVideoEncoder,
    )

    def unavailable(*args, **kwargs):
        raise AssertionError("No Thinker numerical kernel/forward runs in initialization proof")

    ns["model_forward"] = unavailable
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    partition = next(
        n
        for n in ast.parse((ROOT / "pithtrain/pipeline/dualpipev.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "layer_partition"
    )
    nodes = [future, partition]
    for filename, selected in (
        ("pithtrain/operators/linear.py", {"BF16Linear"}),
        ("pithtrain/models/qwen3_omni_moe.py", None),
        ("pithtrain/models/qwen3_omni_image.py", None),
        ("pithtrain/models/qwen3_omni_multimodal.py", None),
    ):
        nodes.extend(
            n
            for n in ast.parse((ROOT / filename).read_text()).body
            if isinstance(n, ast.ClassDef) and (selected is None or n.name in selected)
        )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), __file__, "exec"
        ),
        ns,
    )
    ns["training"].Linear = ns["BF16Linear"]
    cfg = Qwen3OmniMoeConfig(**copy.deepcopy(raw))
    with torch.device("meta"):
        return Qwen3OmniMoeSpeechModel(
            cfg,
            ns["Qwen3OmniMoeThinkerMultimodalModel"](cfg.thinker_config, -1),
            Qwen3OmniMoeTalkerForConditionalGeneration(cfg.talker_config),
            Qwen3OmniMoeCode2Wav(cfg.code2wav_config),
        )


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("full-speech-initialization")
    raw = fixtures.tiny_raw()
    fused = {k: v.clone().bfloat16() for k, v in fixtures.reference(raw).state_dict().items()}
    source = root / "hf"
    fixtures.write_hf(source, raw, fixtures.split_fixture(fused, "split"))
    result = initialize.prepare_speech_initialization(
        source, root / "dcp", expected_source=source_identity(source), max_tensor_bytes=2**20
    )
    return root, raw, fused, result


def test_complete_standard_dcp_and_existing_converter_readback(converted):
    root, raw, fused, result = converted
    cfg = full.configuration(raw, "all")
    canonical, excluded = full.load_canonical(root / "dcp", cfg)
    assert not excluded
    previous, previous_excluded, _ = full.read_hf(root / "hf", cfg)
    assert not previous_excluded
    old_canonical, old_excluded, _ = full.hf_to_canonical(previous, cfg)
    assert not old_excluded
    fixtures.exact(old_canonical, canonical)
    actual = full.canonical_to_hf(canonical, cfg)
    fixtures.exact(actual, fused)
    assert result["source_kind"] == "test_fixture" and not result["full_model_replica"]
    assert result["canonical_tensors"] == len(canonical)
    assert len(list((root / "dcp").glob("*.distcp"))) == len(canonical)


def test_lazy_fused_expert_slices_use_same_canonical_format(converted, tmp_path):
    _, raw, fused, _ = converted
    source = tmp_path / "fused"
    fixtures.write_hf(source, raw, fused)
    initialize.prepare_speech_initialization(
        source, tmp_path / "dcp", expected_source=source_identity(source), max_tensor_bytes=2**20
    )
    canonical, excluded = full.load_canonical(tmp_path / "dcp", full.configuration(raw, "all"))
    assert not excluded
    fixtures.exact(full.canonical_to_hf(canonical, full.configuration(raw, "all")), fused)


def test_lazy_producer_does_not_retain_a_whole_model(converted, tmp_path, monkeypatch):
    root, _, _, _ = converted
    old = initialize._HFSavePlanner.resolve_data
    references, peak = [], []

    def tracked(self, item):
        value = old(self, item)
        references.append(weakref.ref(value))
        peak.append(sum(ref() is not None for ref in references))
        return value

    monkeypatch.setattr(initialize._HFSavePlanner, "resolve_data", tracked)
    initialize.prepare_speech_initialization(
        root / "hf",
        tmp_path / "dcp",
        expected_source=source_identity(root / "hf"),
        max_tensor_bytes=2**20,
    )
    assert len(references) > 100 and max(peak) <= 3


@pytest.mark.parametrize("bad", ["source_sha", "released_config", "tensor_cap"])
def test_prepare_rejects_before_destination_creation(converted, tmp_path, bad):
    root, _, _, _ = converted
    identity = source_identity(root / "hf")
    cap = 2**20
    if bad == "source_sha":
        identity["files"]["part-0.safetensors"]["sha256"] = "0" * 64
    elif bad == "released_config":
        identity["kind"] = "released"
    else:
        cap = 1
    with pytest.raises(ValueError):
        initialize.prepare_speech_initialization(
            root / "hf", tmp_path / "dcp", expected_source=identity, max_tensor_bytes=cap
        )
    assert not (tmp_path / "dcp").exists()


def fsdp_worker(rank, location, root, raw, expected, manifest_sha, nonfinite_sha):
    dist.init_process_group(
        "gloo",
        init_method="file://" + location,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_num_threads(1)
        model = native_meta(raw)
        assert all(p.is_meta for p in model.parameters())
        assert not dict(model.named_buffers()), (
            "Current native constants are computed, not checkpoint tensors"
        )
        for name, value in model.named_parameters():
            value.requires_grad_(name.startswith("code2wav."))
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        apply_speech_fsdp(model, mesh, mode="waveform")
        materialize_speech_shards(model, device="cpu")
        with torch.no_grad():
            for value in model.parameters():
                value.to_local().zero_()
        before_rng = torch.get_rng_state().clone()
        kwargs = dict(
            expected_manifest_sha256=manifest_sha,
            rank=rank,
            world_size=2,
            max_local_bytes=2**24,
            max_tensor_bytes=2**20,
        )
        for changes in (
            {"rank": 0},
            {"expected_manifest_sha256": "0" * 64},
            {"max_local_bytes": 1},
        ):
            # Exactly one rank is invalid; every rank must reject before copying.
            selected = dict(kwargs, **changes) if rank else kwargs
            with pytest.raises(ValueError):
                initialize.initialize_speech_from_dcp(model, Path(root) / "dcp", **selected)
            assert not model._omni_speech_fsdp_plan["weights_initialized"]
            assert all(torch.count_nonzero(p.to_local()) == 0 for p in model.parameters())
        original_layer = model.config.talker_config.accept_hidden_layer
        if rank:
            model.config.talker_config.accept_hidden_layer = original_layer + 1
        with pytest.raises(ValueError, match="configuration differs"):
            initialize.initialize_speech_from_dcp(model, Path(root) / "dcp", **kwargs)
        model.config.talker_config.accept_hidden_layer = original_layer
        with pytest.raises(ValueError, match="Nonfinite local candidate"):
            initialize.initialize_speech_from_dcp(
                model,
                Path(root) / "nonfinite",
                **dict(kwargs, expected_manifest_sha256=nonfinite_sha),
            )
        assert not model._omni_speech_fsdp_plan["weights_initialized"]
        assert all(torch.count_nonzero(p.to_local()) == 0 for p in model.parameters())
        model.register_buffer("unknown_nonpersistent", torch.ones(1), persistent=False)
        with pytest.raises(ValueError, match="unknown constants"):
            initialize.initialize_speech_from_dcp(model, Path(root) / "dcp", **kwargs)
        del model.unknown_nonpersistent
        receipt = initialize.initialize_speech_from_dcp(model, Path(root) / "dcp", **kwargs)
        assert receipt["weights_initialized"] and receipt["buffers"] == 0
        assert receipt["parameters"] == len(expected)
        assert torch.equal(before_rng, torch.get_rng_state())
        assert model._omni_speech_fsdp_plan["initialization"] == receipt
        for name, value in model.named_parameters():
            # torch.chunk omits trailing empty chunks; FSDP retains empty local shards.
            chunks = expected[name].float().chunk(2, dim=0)
            target = chunks[rank] if rank < len(chunks) else expected[name].float()[:0]
            torch.testing.assert_close(value.to_local(), target, rtol=0, atol=0)
        with pytest.raises(ValueError, match="newly materialized"):
            initialize.initialize_speech_from_dcp(model, Path(root) / "dcp", **kwargs)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_actual_full_components_meta_fsdp_local_readback_and_all_rank_negatives(
    converted, tmp_path
):
    root, raw, fused, result = converted
    config = full.configuration(raw, "all")
    mapping = full.native_key_map(config)
    expected = {name: fused[key] for name, key in mapping.items()}
    # Valid file inventory, but a nonfinite payload owned by just one rank: all
    # ranks must reject before applying any candidate, even with a trusted hash.
    canonical, _ = full.load_canonical(root / "dcp", config)
    key = next(k for k in canonical if k.startswith("code2wav."))
    canonical[key].flatten()[0] = float("nan")
    dcp.save(
        {"app": {"model": canonical}},
        storage_writer=dcp.FileSystemWriter(root / "nonfinite", single_file_per_rank=False),
        no_dist=True,
    )
    manifest = json.loads((root / "dcp" / full.SIDECAR).read_text())
    manifest["bounded_initialization"]["checkpoint_files"] = initialize._files(root / "nonfinite")
    (root / "nonfinite" / full.SIDECAR).write_text(json.dumps(manifest))
    nonfinite_sha = initialize._sha(root / "nonfinite" / full.SIDECAR)
    mp.spawn(
        fsdp_worker,
        args=(
            str(tmp_path / "rdzv"),
            str(root),
            raw,
            expected,
            result["manifest_sha256"],
            nonfinite_sha,
        ),
        nprocs=2,
        join=True,
    )


def rank_local_corruption_worker(rank, location, root, raw, manifest_sha, modified_value):
    dist.init_process_group(
        "gloo",
        init_method="file://" + location,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_num_threads(1)
        model = native_meta(raw)
        for name, value in model.named_parameters():
            value.requires_grad_(name.startswith("code2wav."))
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        apply_speech_fsdp(model, mesh, mode="waveform")
        materialize_speech_shards(model, device="cpu")
        with torch.no_grad():
            for value in model.parameters():
                value.to_local().zero_()
        directory = Path(root) / ("changed" if rank else "original")
        error = None
        try:
            initialize.initialize_speech_from_dcp(
                model,
                directory,
                expected_manifest_sha256=manifest_sha,
                rank=rank,
                world_size=2,
                max_local_bytes=2**24,
                max_tensor_bytes=2**20,
            )
        except ValueError as exc:
            error = str(exc)
        initialized = model._omni_speech_fsdp_plan["weights_initialized"]
        local_value = float(model.thinker.embed_tokens.weight.to_local().flatten()[0])
        observation = dict(
            rank=rank,
            rejected=error is not None,
            error=error,
            weights_initialized=initialized,
            untouched=all(
                torch.count_nonzero(p.to_local()).item() == 0 for p in model.parameters()
            ),
            finite_modified_payload_loaded=rank == 1
            and initialized
            and local_value == modified_value,
        )
        (Path(root) / f"observation-rank-{rank}.json").write_text(json.dumps(observation))
        print("RANK_LOCAL_INTEGRITY " + json.dumps(observation), flush=True)
        assert observation["rejected"], observation
        assert not initialized and observation["untouched"], observation
        assert "DCP file preflight failed" in error and "File changed" in error
        # Pre-application rejection is recoverable: use the intact directory on
        # both ranks and exercise both successful all-rank file verification phases.
        receipt = initialize.initialize_speech_from_dcp(
            model,
            Path(root) / "original",
            expected_manifest_sha256=manifest_sha,
            rank=rank,
            world_size=2,
            max_local_bytes=2**24,
            max_tensor_bytes=2**20,
        )
        assert receipt["weights_initialized"]
        observation["valid_retry_initialized"] = True
        (Path(root) / f"observation-rank-{rank}.json").write_text(json.dumps(observation))
    finally:
        dist.destroy_process_group()


def test_rank_local_payload_corruption_rejected_on_every_rank_before_copy(converted, tmp_path):
    root, raw, _, result = converted
    original, changed = tmp_path / "original", tmp_path / "changed"
    shutil.copytree(root / "dcp", original)
    shutil.copytree(original, changed)
    metadata = dcp.FileSystemReader(changed).read_metadata()
    key = "app.model.thinker.embed_tokens.weight"
    info = next(info for index, info in metadata.storage_data.items() if index.fqn == key)
    assert info.offset == 0
    path = changed / info.relative_path
    value = torch.load(path, map_location="cpu", weights_only=True)
    row = math.ceil(value.shape[0] / 2)  # First row owned by rank 1.
    value[row].flatten()[0] = 0.75
    modified_value = float(value[row].flatten()[0])
    stream = io.BytesIO()
    torch.save(value, stream)
    assert len(stream.getvalue()) == path.stat().st_size == info.length
    path.write_bytes(stream.getvalue())
    assert torch.isfinite(torch.load(path, weights_only=True)).all()
    assert initialize._sha(path) != initialize._sha(original / info.relative_path)
    # Metadata, sidecar, and its pinned SHA are byte-identical on both mounts.
    assert (changed / ".metadata").read_bytes() == (original / ".metadata").read_bytes()
    assert (changed / full.SIDECAR).read_bytes() == (original / full.SIDECAR).read_bytes()
    mp.spawn(
        rank_local_corruption_worker,
        args=(
            str(tmp_path / "rdzv"),
            str(tmp_path),
            raw,
            result["manifest_sha256"],
            modified_value,
        ),
        nprocs=2,
        join=True,
    )
