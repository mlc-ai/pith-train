"""Actual strict container CPU I/O; HF modules and cursor are protocol fixtures.

No native GPU forward/backward, real loader, or NCCL acceptance is claimed.
"""

import copy
import json
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from pithtrain.contexts import distributed, training
from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
from pithtrain.modules.distributed_checkpoint import _pack, _runtime
from pithtrain.tasks.convert_checkpoint import dcp2hf, hf2dcp
from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import (
    ConversionError,
    canonical_to_hf,
    hf_to_canonical,
)
from pithtrain.tasks.convert_checkpoint.strict_omni import (
    MANIFEST,
    RESUME_DIR,
    RESUME_MANIFEST,
    file_hash,
    read_payloads,
)
from tests import test_qwen3_omni_checkpoint as fixtures
from tests.test_distributed_strict_checkpoint import Cursor
from tests.test_strict_checkpoint import assert_tree_equal, native_shaped_cpu_model


def configuration(profile):
    config = fixtures.config.__wrapped__()
    if profile in {"ep", "pp-ep", "dp-chunks"}:
        config.num_experts = 4
    if profile == "dp-chunks":
        config.num_experts = 5
    if profile == "dp-empty":
        config.num_experts = config.num_experts_per_tok = 1
    return config


def geometry(profile):
    return {
        "single": (1, 1, 1),
        "dp": (2, 1, 1),
        "dp-chunks": (4, 1, 1),
        "dp-empty": (2, 1, 1),
        "pp": (2, 2, 1),
        "ep": (2, 1, 2),
        "pp-ep": (4, 2, 2),
    }[profile]


def make_runtime(profile, rank):
    world, pp, ep = geometry(profile)
    config = configuration(profile)
    distributed.rank, distributed.world_size = rank, world
    distributed.device = torch.device("cpu")
    distributed.attn_mesh = init_device_mesh(
        "cpu", (pp, world // pp, 1), mesh_dim_names=("pp", "dp", "cp")
    )
    distributed.expt_mesh = init_device_mesh(
        "cpu", (pp, world // pp // ep, ep), mesh_dim_names=("pp", "dp", "ep")
    )
    distributed.pp_size, distributed.ep_size, distributed.cp_size = pp, ep, 1
    distributed.dp_size = world // pp
    distributed.pp_rank = distributed.attn_mesh.get_local_rank("pp")
    distributed.dp_rank = distributed.attn_mesh.get_local_rank("dp")
    distributed.ep_rank = distributed.expt_mesh.get_local_rank("ep")
    distributed.cp_rank = 0
    model = native_shaped_cpu_model(config)
    if profile == "ep":
        model.to(dtype=torch.bfloat16)
    body = model["module"][0]
    if pp > 1:
        for index in list(body.layers._modules):
            if int(index) % pp != distributed.pp_rank:
                del body.layers._modules[index]
        if distributed.pp_rank != 0:
            del body.embed_tokens, body.norm, body.lm_head
    for layer in body.layers:
        layer.mlp.experts_per_rank = config.num_experts // ep
        for name, parameter in list(layer.mlp.experts.named_parameters()):
            start = distributed.ep_rank * layer.mlp.experts_per_rank
            layer.mlp.experts.register_parameter(
                name, nn.Parameter(parameter[start : start + layer.mlp.experts_per_rank].clone())
            )
    if world > 1:
        for name, parameter in list(model.named_parameters()):
            parent, field = name.rsplit(".", 1)
            mesh = (
                distributed.expt_mesh["dp"] if ".experts." in name else distributed.attn_mesh["dp"]
            )
            model.get_submodule(parent).register_parameter(
                field, nn.Parameter(distribute_tensor(parameter.detach(), mesh, [Shard(0)]))
            )
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    sch = (
        torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.8**step)
        if profile == "pp-ep"
        else torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.8)
    )
    training.model, training.optimizers, training.schedulers = model, (opt,), (sch,)
    contract = dict(
        model=config.to_dict(),
        training=dict(global_batch_size=2),
        hsdp_replica=1,
        protocol="CPU-fixture",
    )
    return model, opt, sch, Cursor(), contract


def advance(model, opt, sch, data):
    for parameter in model.parameters():
        parameter.grad = parameter.detach() * 0.1 + torch.ones_like(parameter) * 0.01
    opt.step()
    sch.step()
    opt.zero_grad(set_to_none=True)
    data.consumed += 2


def worker(rank, folder, profile, mode):
    root = Path(folder)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / ('rendezvous-' + mode)}",
        rank=rank,
        world_size=geometry(profile)[0],
        timeout=timedelta(seconds=25),
    )
    try:
        torch.set_num_threads(1)
        model, opt, sch, data, contract = make_runtime(profile, rank)
        if mode == "save":
            advance(model, opt, sch, data)
            save_checkpoint(
                root / "original", 1, data_state=data, strict=True, resume_config=contract
            )
        else:
            assert not opt.state
            load_checkpoint(
                root / "restored", 1, data_state=data, strict=True, resume_config=contract
            )
            values, _ = _runtime(model, (opt,), (sch,), data, torch.device("cpu"))
            expected = torch.load(
                root / f"original/torch-dcp/00000001/rank-{rank:05d}.pt", weights_only=True
            )
            assert_tree_equal(values.pop("rng"), expected["rng"])
            assert_tree_equal(_pack(values), expected["state"])
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def produce(root, profile):
    mp.spawn(worker, args=(str(root), profile, "save"), nprocs=geometry(profile)[0], join=True)
    return root / "original/torch-dcp/00000001"


@pytest.mark.parametrize("profile", ["single", "dp", "dp-chunks", "dp-empty", "pp", "ep", "pp-ep"])
def test_public_strict_hf_strict_roundtrip(tmp_path, profile):
    src = produce(tmp_path, profile)
    original = read_payloads(src)[1]
    for layout, prefix in (("fused", "thinker."), ("split", "")):
        hf, rebuilt = tmp_path / ("hf-" + layout), tmp_path / ("rebuilt-" + layout)
        cfg = fixtures.make_cfg(
            load_path=src, save_path=hf, omni_prefix=prefix, omni_expert_layout=layout
        )
        dcp2hf(cfg, fixtures.LOGGER)
        config = configuration(profile)
        canonical, _ = hf_to_canonical(fixtures.read_export(hf), config, prefix=prefix)
        dtype = torch.bfloat16 if profile == "ep" else torch.float32
        expected = fixtures.model(config, dtype)
        opt = torch.optim.AdamW(expected.parameters(), lr=0.01, foreach=False)
        for parameter in expected.parameters():
            parameter.grad = parameter.detach() * 0.1 + torch.ones_like(parameter) * 0.01
        opt.step()
        exported = canonical_to_hf(canonical, config, prefix="")
        assert_tree_equal(exported, expected.state_dict())
        actual = fixtures.model(config, dtype)
        actual.load_state_dict(exported, strict=True)
        tokens = torch.tensor([[1, 3, 2, 11, 5]])
        with torch.no_grad():
            before = expected["lm_head"](
                expected["model"](input_ids=tokens, use_cache=False).last_hidden_state
            )
            after = actual["lm_head"](
                actual["model"](input_ids=tokens, use_cache=False).last_hidden_state
            )
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        assert torch.equal(after.argmax(-1), before.argmax(-1))
        companion = read_payloads(hf / RESUME_DIR, companion=True)[1]
        assert all(
            set(entry) == {"tensor", "sha256"}
            for payload in companion
            for entry in payload["state"]["model"].values()
        )
        cfg.load_path, cfg.save_path, cfg.omni_restore_resume = hf, rebuilt, True
        hf2dcp(cfg, fixtures.LOGGER)
        assert_tree_equal(read_payloads(rebuilt)[1], original)
    shutil.copytree(rebuilt, tmp_path / "restored/torch-dcp/00000001")
    mp.spawn(worker, args=(str(tmp_path), profile, "load"), nprocs=geometry(profile)[0], join=True)
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("strict-converter-source")
    return produce(root, "single")


FAULTS = (
    "missing-rank",
    "checksum",
    "manifest-bool",
    "missing-manifest",
    "extra-file",
    "missing-model",
    "extra-model",
    "shape",
    "dtype",
    "nonfinite",
    "missing-moment",
    "moment-shape",
    "optimizer-order",
    "extra-optimizer-state",
    "missing-group-field",
    "scheduler-field",
    "scheduler-step",
    "scheduler-nonfinite",
    "cursor",
    "missing-rng",
    "cpu-rng",
    "topology",
    "config",
    "extra-header",
)


def corrupt(folder, fault, companion=False):
    marker = folder / (RESUME_MANIFEST if companion else MANIFEST)
    path = folder / "rank-00000.pt"
    manifest = json.loads(marker.read_text())
    if fault == "missing-rank":
        path.unlink()
        return
    if fault == "checksum":
        path.write_bytes(b"truncated")
        return
    if fault == "manifest-bool":
        manifest["step"] = True
        marker.write_text(json.dumps(manifest))
        return
    if fault == "missing-manifest":
        marker.unlink()
        return
    if fault == "extra-file":
        (folder / "unexpected").touch()
        return
    payload = torch.load(path, weights_only=True)
    state = payload["state"]
    key = next(iter(state["model"]))
    opt = state["optimizer"][0]
    if fault == "missing-model":
        del state["model"][key]
    elif fault == "extra-model":
        state["model"]["extra"] = state["model"][key]
    elif fault == "shape":
        state["model"][key]["value"] = torch.ones(1)
    elif fault == "dtype":
        state["model"][key]["value"] = state["model"][key]["value"].double()
    elif fault == "nonfinite":
        state["model"][key]["value"].flatten()[0] = float("nan")
    elif fault == "missing-moment":
        del opt["state"][0]["exp_avg"]
    elif fault == "moment-shape":
        opt["state"][0]["exp_avg"]["value"] = torch.ones(1)
    elif fault == "optimizer-order":
        opt["param_groups"][0]["params"].reverse()
    elif fault == "extra-optimizer-state":
        opt["state"][999] = copy.deepcopy(opt["state"][0])
    elif fault == "missing-group-field":
        del opt["param_groups"][0]["betas"]
    elif fault == "scheduler-field":
        del state["scheduler"][0]["base_lrs"]
    elif fault == "scheduler-step":
        state["scheduler"][0]["last_epoch"] = 2
    elif fault == "scheduler-nonfinite":
        state["scheduler"][0]["gamma"] = float("nan")
    elif fault == "cursor":
        state["data"]["consumed_samples"] = 1
    elif fault == "missing-rng":
        del payload["rng"]["torch_cpu"]
    elif fault == "cpu-rng":
        payload["rng"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
    elif fault == "topology":
        payload["header"]["topology"]["ep_size"] = 999
    elif fault == "config":
        del payload["header"]["config"]["model"]
    elif fault == "extra-header":
        payload["header"]["extra"] = 1
    else:
        raise AssertionError(fault)
    torch.save(payload, path)
    manifest["files"][path.name] = file_hash(path)
    marker.write_text(json.dumps(manifest))


@pytest.mark.parametrize("fault", FAULTS)
def test_export_rejects_corruption_before_writing(saved, tmp_path, fault):
    src, output = tmp_path / "bad", tmp_path / "output"
    shutil.copytree(saved, src)
    corrupt(src, fault)
    with pytest.raises((ConversionError, FileNotFoundError)):
        dcp2hf(fixtures.make_cfg(load_path=src, save_path=output), fixtures.LOGGER)
    assert not output.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "weights-changed",
        "companion-missing",
        "missing-moment",
        "cursor",
        "missing-rng",
        "checksum",
        "config-diff",
    ],
)
def test_import_rejects_invalid_resume_bundle(saved, tmp_path, fault):
    hf, output = tmp_path / "hf", tmp_path / "output"
    dcp2hf(fixtures.make_cfg(load_path=saved, save_path=hf), fixtures.LOGGER)
    if fault == "weights-changed":
        path = next(hf.glob("*.safetensors"))
        values = load_file(path)
        values[next(iter(values))].flatten()[0] += 0.1
        save_file(values, path)
    elif fault == "companion-missing":
        shutil.rmtree(hf / RESUME_DIR)
    elif fault == "config-diff":
        path = hf / "config.json"
        config = json.loads(path.read_text())
        config["rms_norm_eps"] *= 2
        path.write_text(json.dumps(config))
    else:
        corrupt(hf / RESUME_DIR, fault, companion=True)
    with pytest.raises((ConversionError, FileNotFoundError)):
        hf2dcp(
            fixtures.make_cfg(load_path=hf, save_path=output, omni_restore_resume=True),
            fixtures.LOGGER,
        )
    assert not output.exists()


def test_cli_autodetect_and_restore_flag(saved, tmp_path):
    script = Path(__file__).parents[1] / "examples/convert_checkpoint/qwen3-omni-text/script.py"
    hf, output = tmp_path / "hf", tmp_path / "strict"
    for args in (
        ("dcp2hf", str(saved), str(hf)),
        ("hf2dcp", str(hf), str(output), "--restore-resume"),
    ):
        subprocess.run(
            [sys.executable, str(script), *args], check=True, capture_output=True, text=True
        )
    assert_tree_equal(read_payloads(saved)[1], read_payloads(output)[1])
    sentinel = hf / "config.json"
    original = sentinel.read_bytes()
    with pytest.raises(ConversionError):
        dcp2hf(fixtures.make_cfg(load_path=saved, save_path=hf), fixtures.LOGGER)
    assert sentinel.read_bytes() == original


def test_explicit_config_disagreement_rejected(saved, tmp_path):
    config = configuration("single").to_dict()
    config["rms_norm_eps"] *= 2
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "out"
    with pytest.raises(ConversionError, match="config disagrees"):
        dcp2hf(
            fixtures.make_cfg(load_path=saved, save_path=out, omni_text_config=path),
            fixtures.LOGGER,
        )
    assert not out.exists()


def test_resume_rejects_outer_quantization(saved, tmp_path):
    hf, output = tmp_path / "hf", tmp_path / "output"
    dcp2hf(fixtures.make_cfg(load_path=saved, save_path=hf), fixtures.LOGGER)
    path = hf / "config.json"
    raw = json.loads(path.read_text())
    path.write_text(
        json.dumps(
            dict(
                model_type="qwen3_omni_moe",
                thinker_config=dict(text_config=raw),
                quantization_config={"quant_method": "fp8"},
            )
        )
    )
    with pytest.raises(ConversionError, match="Quantized"):
        hf2dcp(
            fixtures.make_cfg(load_path=hf, save_path=output, omni_restore_resume=True),
            fixtures.LOGGER,
        )
    assert not output.exists()


@pytest.mark.parametrize("layout", ["fused", "split"])
@pytest.mark.parametrize("component", ["dense", "gate", "down"])
def test_changed_weights_cannot_restore_optimizer(saved, tmp_path, layout, component):
    hf, output = tmp_path / "hf", tmp_path / "strict"
    dcp2hf(
        fixtures.make_cfg(load_path=saved, save_path=hf, omni_expert_layout=layout), fixtures.LOGGER
    )
    hashes = {p.name: file_hash(p) for p in (hf / RESUME_DIR).iterdir()}
    path = next(hf.glob("*.safetensors"))
    values = load_file(path)
    suffix = {
        "dense": "q_proj.weight",
        "gate": "gate_up_proj" if layout == "fused" else "gate_proj.weight",
        "down": "down_proj" if layout == "fused" else "down_proj.weight",
    }[component]
    key = next(k for k in values if k.endswith(suffix))
    value = values[key].flatten()
    value[0] = torch.nextafter(value[0], torch.tensor(float("inf")))
    save_file(values, path)
    with pytest.raises(ConversionError, match="HF weights changed"):
        hf2dcp(
            fixtures.make_cfg(load_path=hf, save_path=output, omni_restore_resume=True),
            fixtures.LOGGER,
        )
    assert not output.exists()
    assert hashes == {p.name: file_hash(p) for p in (hf / RESUME_DIR).iterdir()}
    # A deliberate new initialization remains model-only, with no stale moments.
    hf2dcp(fixtures.make_cfg(load_path=hf, save_path=output), fixtures.LOGGER)
    import torch.distributed.checkpoint as dcp

    metadata = dcp.FileSystemReader(output).read_metadata()
    assert all(k.startswith("app.model.") for k in metadata.state_dict_metadata)
    assert not (output / MANIFEST).exists()
