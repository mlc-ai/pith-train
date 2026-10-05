import json
import logging
import weakref
from array import array
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, dcp2hf, hf2dcp
from pithtrain.tasks.convert_checkpoint._streaming import HfCheckpoint, save_dcp
from pithtrain.tasks.convert_checkpoint.gpt_oss import _dequantize_mxfp4

STDOUT = logging.getLogger(__name__)


def _write_hf(path, shards, config=None):
    path.mkdir()
    weight_map = {}
    for i, tensors in enumerate(shards):
        shard = "model-%05d.safetensors" % i
        save_file(tensors, path / shard)
        weight_map.update({key: shard for key in tensors})
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    if config is not None:
        (path / "config.json").write_text(json.dumps(config))


def _convert(load_path, save_path, operation, *, max_chunk_size=24):
    cfg = ConvertCheckpointCfg()
    cfg.load_path = load_path
    cfg.save_path = save_path
    cfg.max_shard_size = 8192 if operation is hf2dcp else 128
    cfg.max_chunk_size = max_chunk_size
    operation(cfg, STDOUT)


def _assert_dcp(path, expected):
    metadata = dcp.FileSystemReader(path).read_metadata()
    assert set(metadata.state_dict_metadata) == {"app.model." + key for key in expected}
    assert metadata.planner_data == {"app.model." + key: ("app", "model", key) for key in expected}
    assert all(file.stat().st_size <= 8192 for file in path.glob("*.distcp"))
    state = {key: torch.empty_like(tensor) for key, tensor in expected.items()}
    dcp.load({"app": {"model": state}}, checkpoint_id=path, no_dist=True)
    for key, tensor in expected.items():
        torch.testing.assert_close(state[key], tensor, rtol=0, atol=0)


def _assert_hf(path, expected):
    actual = {}
    for shard in path.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as f:
            actual.update({key: f.get_tensor(key).clone() for key in f.keys()})
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_generic_round_trip(tmp_path, dtype):
    hf, checkpoint, exported = (tmp_path / name for name in ("hf", "dcp", "export"))
    weights = {
        "model.embed_tokens.weight": torch.arange(24).reshape(6, 4).to(dtype),
        "model.layers.0.mlp.experts.2.up_proj.weight": torch.arange(12).reshape(3, 4).to(dtype),
        "lm_head.weight": torch.arange(24, 48).reshape(6, 4).to(dtype),
        "model.scalar": torch.tensor(7),
        "model.empty": torch.empty(0, 4, dtype=dtype),
        "model.mask": torch.tensor([True, False]),
    }
    items = list(weights.items())
    _write_hf(hf, [dict(items[:2]), dict(items[2:])])
    indexed = HfCheckpoint(hf, STDOUT)
    assert all(tensor.is_meta for tensor in indexed.tensors.values())
    _convert(hf, checkpoint, hf2dcp)
    _assert_dcp(checkpoint, {key.removeprefix("model."): value for key, value in weights.items()})
    _convert(checkpoint, exported, dcp2hf)
    _assert_hf(exported, weights)


@pytest.mark.parametrize("nested_config", [False, True])
def test_qwen35_round_trip(tmp_path, monkeypatch, nested_config):
    hf, checkpoint, exported = (tmp_path / name for name in ("hf", "dcp", "export"))
    prefix = "model.language_model."
    weights = {
        prefix + "layers.0.mlp.experts.gate_up_proj": torch.arange(72).reshape(3, 6, 4).bfloat16(),
        prefix + "layers.0.mlp.experts.down_proj": torch.arange(36).reshape(3, 4, 3).bfloat16(),
        prefix + "layers.0.linear_attn.A_log": torch.arange(4).float(),
        prefix + "norm.weight": torch.arange(4).bfloat16(),
        "lm_head.weight": torch.arange(20).reshape(5, 4).bfloat16(),
    }
    dropped = {key: torch.ones(4) for key in ("model.visual.weight", "mtp.weight", "other")}
    config = {"model_type": "qwen3_5_moe_text"}
    if nested_config:
        config = {"model_type": "qwen3_5_moe", "text_config": config}
    _write_hf(hf, [weights, dropped], config)
    loaded = []
    original_load = HfCheckpoint.load

    def load(self, key, expert=None, slices=()):
        loaded.append((key, expert))
        return original_load(self, key, expert, slices)

    monkeypatch.setattr(HfCheckpoint, "load", load)
    _convert(hf, checkpoint, hf2dcp)
    expected = {}
    for key, tensor in weights.items():
        canon = key.removeprefix(prefix)
        if ".mlp.experts." in canon:
            for idx in range(3):
                expected[canon.replace(".experts.", f".experts.{idx}.") + ".weight"] = tensor[idx]
        else:
            expected[canon] = tensor
    assert all(key not in dropped for key, _ in loaded)
    assert all(idx is not None for key, idx in loaded if ".experts." in key)
    _assert_dcp(checkpoint, expected)
    _convert(checkpoint, exported, dcp2hf)
    _assert_hf(exported, weights)


def _reference_mxfp4(blocks, scales):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    values = torch.stack((lut[(blocks & 15).long()], lut[(blocks >> 4).long()]), dim=-1)
    values = values.flatten(-2) * (2.0 ** (scales.float() - 127)).unsqueeze(-1)
    return values.flatten(-2).bfloat16()


@pytest.mark.parametrize("quantized", [False, True])
def test_gpt_oss_round_trip(tmp_path, monkeypatch, quantized):
    hf, checkpoint, exported = (tmp_path / name for name in ("hf", "dcp", "export"))
    prefix = "model.layers.0.mlp.experts."
    weights = {
        "model.norm.weight": torch.arange(6).bfloat16(),
        "lm_head.weight": torch.arange(30).reshape(5, 6).bfloat16(),
        prefix + "gate_up_proj_bias": torch.arange(12).reshape(3, 4).bfloat16(),
        prefix + "down_proj_bias": torch.arange(18).reshape(3, 6).bfloat16(),
    }
    scales_shard = {}
    expected = dict(weights)
    for name, shape in (("gate_up_proj", (3, 4, 2, 16)), ("down_proj", (3, 6, 1, 16))):
        blocks = torch.arange(torch.Size(shape).numel()).reshape(shape).to(torch.uint8)
        scales = (torch.arange(blocks.numel() // 16) % 5 + 125).reshape(shape[:-1]).to(torch.uint8)
        expected[prefix + name] = _reference_mxfp4(blocks, scales)
        if quantized:
            weights[prefix + name + "_blocks"] = blocks
            scales_shard[prefix + name + "_scales"] = scales
        else:
            weights[prefix + name] = expected[prefix + name]
    _write_hf(hf, [weights, scales_shard] if quantized else [weights], {"model_type": "gpt_oss"})
    loaded = []
    original_load = HfCheckpoint.load

    def load(self, key, expert=None, slices=()):
        loaded.append((key, expert))
        return original_load(self, key, expert, slices)

    monkeypatch.setattr(HfCheckpoint, "load", load)
    _convert(hf, checkpoint, hf2dcp)
    canonical, hf_expected = {}, {}
    for key, tensor in expected.items():
        canon = key.removeprefix("model.")
        if ".mlp.experts." in canon:
            for idx in range(3):
                canonical[canon.replace(".experts.", f".experts.{idx}.")] = tensor[idx]
        else:
            canonical[canon] = tensor
        hf_expected[key] = tensor.transpose(-2, -1) if key.endswith("_proj") else tensor
    assert all(idx is not None for key, idx in loaded if ".experts." in key)
    _assert_dcp(checkpoint, canonical)
    _convert(checkpoint, exported, dcp2hf)
    _assert_hf(exported, hf_expected)


@pytest.mark.parametrize("rows_per_chunk", [1, 3, 32768])
def test_mxfp4_chunks(rows_per_chunk):
    blocks = torch.arange(256).reshape(4, 4, 16).to(torch.uint8)
    scales = torch.arange(120, 136).reshape(4, 4).to(torch.uint8)
    actual = _dequantize_mxfp4(blocks, scales, rows_per_chunk=rows_per_chunk)
    torch.testing.assert_close(actual, _reference_mxfp4(blocks, scales), rtol=0, atol=0)


def test_invalid_mxfp4_shapes(tmp_path):
    hf = tmp_path / "hf"
    _write_hf(
        hf,
        [
            {
                "model.layers.0.mlp.experts.down_proj_blocks": torch.zeros(
                    3, 6, 1, 16, dtype=torch.uint8
                ),
                "model.layers.0.mlp.experts.down_proj_scales": torch.zeros(
                    3, 5, 1, dtype=torch.uint8
                ),
            }
        ],
        {"model_type": "gpt_oss"},
    )
    with pytest.raises(AssertionError, match="does not match"):
        _convert(hf, tmp_path / "dcp", hf2dcp)
    assert not Path(tmp_path, "dcp", ".metadata").exists()


def test_streaming_releases_weights(tmp_path):
    tensors = {f"layers.{i}.weight": torch.empty(8, device="meta") for i in range(8)}
    buffers, loaded = [], []

    def load_tensor(key, slices):
        # frombuffer keeps its owner alive through detach.
        assert sum(ref() is not None for ref in buffers) <= 1
        values = array("f", range(8))
        buffers.append(weakref.ref(values))
        loaded.append(key)
        return torch.frombuffer(values, dtype=torch.float32)

    save_dcp(tensors, load_tensor, tmp_path / "dcp", STDOUT, max_shard_size=8192)
    assert sorted(loaded) == sorted(tensors)
    assert all(ref() is None for ref in buffers)
    assert len(list((tmp_path / "dcp").glob("*.distcp"))) < len(tensors)
    _assert_dcp(tmp_path / "dcp", {key: torch.arange(8).float() for key in tensors})


def _load_sharded(rank, checkpoint, port):
    timeout = timedelta(seconds=60)
    store = torch.distributed.TCPStore("127.0.0.1", port, timeout=timeout)
    torch.distributed.init_process_group(
        "gloo", store=store, rank=rank, world_size=2, timeout=timeout
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        tensor = DTensor.from_local(torch.empty(5, 4), mesh, [Shard(0)])
        dcp.load({"app": {"model": {"embed_tokens.weight": tensor}}}, checkpoint_id=checkpoint)
        expected = torch.arange(40).reshape(10, 4).float()[rank * 5 : (rank + 1) * 5]
        torch.testing.assert_close(tensor.to_local(), expected, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_dcp_reshard(tmp_path):
    hf, checkpoint = tmp_path / "hf", tmp_path / "dcp"
    _write_hf(hf, [{"model.embed_tokens.weight": torch.arange(40).reshape(10, 4).float()}])
    _convert(hf, checkpoint, hf2dcp, max_chunk_size=48)
    store = torch.distributed.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    torch.multiprocessing.spawn(_load_sharded, args=(checkpoint, store.port), nprocs=2)
