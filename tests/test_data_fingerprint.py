"""Static data identity and real DCP resume checks, with CUDA bookkeeping on CPU."""

import copy
import hashlib
import json
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.utils._pytree import tree_flatten

from pithtrain.modules import checkpoint
from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.training_data import PreparedBundleData, TokenBinData
from tests import test_training_data as text_fixtures

text_corpus = text_fixtures.text_corpus


def assert_state_equal(actual, expected):
    actual_leaves, actual_spec = tree_flatten(actual)
    expected_leaves, expected_spec = tree_flatten(expected)
    assert actual_spec == expected_spec
    for left, right in zip(actual_leaves, expected_leaves, strict=True):
        if isinstance(right, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert type(left) is type(right) and left == right


def test_token_contract_covers_source_and_batch_config(text_corpus):
    root, cfg, _, _ = text_corpus
    reader = TokenBinData(root, cfg)
    original = reader.data_fingerprint
    reader.get_batch(1, "cpu")
    assert reader.data_fingerprint == original
    for field, value in dict(
        seed=17, global_batch_size=4, micro_batch_size=2, sequence_length=8
    ).items():
        changed = copy.copy(cfg)
        setattr(changed, field, value)
        assert TokenBinData(root, changed).data_fingerprint != original, field
    extended = copy.copy(cfg)
    extended.max_steps = 4
    assert TokenBinData(root, extended, dp_rank=1, dp_size=2).data_fingerprint == original
    # Inventory changes are detected without reading or hashing all token bytes.
    shard = next(root.glob("*.bin"))
    shard.with_name("extra.bin").write_bytes(shard.read_bytes())
    assert TokenBinData(root, cfg).data_fingerprint != original


@pytest.fixture
def prepared_contract(tmp_path):
    # Fingerprinting needs only verified metadata; decoding these fixture payloads
    # is intentionally outside this test (real media processing has separate coverage).
    from pithtrain.tasks.prepare_omni_data import write_bundle, write_json

    root = tmp_path / "bundle"
    root.mkdir()
    recipe = dict(
        stages={"mixed": ["image", "audio"]}, sampling_weights={"image": 1, "audio": 2}, batch={}
    )
    write_json(root / "recipe.json", dict(recipe=recipe))
    (root / "rejected.jsonl").write_text("")
    files, manifests = ["recipe.json", "rejected.jsonl"], {}
    for kind in ("image", "audio"):
        media = f"{kind}.bin"
        (root / media).write_bytes(kind.encode())
        name = f"{kind}.jsonl"
        (root / name).write_text(
            json.dumps(dict(id=kind, text="paired text", media=[dict(type=kind, path=media)]))
            + "\n"
        )
        files.extend([name, media])
        manifests[kind] = [name]
    write_bundle(
        root,
        dict(
            status="complete",
            recipe=recipe,
            manifests={"train": manifests},
            statistics={"train": {kind: {"samples": 1} for kind in manifests}},
        ),
        files,
    )
    data_cfg = DataCfg()
    data_cfg.dataset, data_cfg.format, data_cfg.modalities = (
        root,
        "prepared_bundle",
        ("image", "audio"),
    )
    training_cfg = SimpleNamespace(
        global_batch_size=2, micro_batch_size=1, sequence_length=64, seed=42, max_steps=3
    )
    return data_cfg, training_cfg, files


def test_prepared_contract_keeps_legacy_identity_and_effective_settings(prepared_contract):
    cfg, tc, _ = prepared_contract
    data = PreparedBundleData(cfg, tc)
    # Independently reproduce the version-1 fingerprint saved by the former stateful reader.
    legacy_identity = dict(
        version=1,
        bundle_sha256=hashlib.sha256((cfg.dataset / "bundle.json").read_bytes()).hexdigest(),
        stage="mixed",
        global_batch_size=2,
        micro_batch_size=1,
        sequence_length=64,
        seed=42,
        weights={"image": 1, "audio": 2},
        epoch_samples=2,
    )
    expected = hashlib.sha256(json.dumps(legacy_identity, sort_keys=True).encode()).hexdigest()
    assert data.data_fingerprint == expected
    cfg.num_workers, cfg.modalities = 3, ("audio", "image")
    cfg.sampling_weights, cfg.epoch_samples = dict(audio=2, image=1), 2
    tc.max_steps = 100
    assert PreparedBundleData(cfg, tc, dp_rank=1, dp_size=2).data_fingerprint == expected
    assert not hasattr(data, "commit_step") and not hasattr(data, "state_dict")


def test_prepared_contract_detects_sampling_and_content_changes(prepared_contract):
    cfg, tc, files = prepared_contract
    expected = PreparedBundleData(cfg, tc).data_fingerprint
    for updates in (
        {"modalities": ("image",)},
        {"sampling_weights": {"image": 2, "audio": 1}},
        {"epoch_samples": 4},
    ):
        changed = copy.deepcopy(cfg)
        for key, value in updates.items():
            setattr(changed, key, value)
        assert PreparedBundleData(changed, tc).data_fingerprint != expected
    for key, value in dict(global_batch_size=4, sequence_length=32, seed=123).items():
        changed = copy.copy(tc)
        setattr(changed, key, value)
        assert PreparedBundleData(cfg, changed).data_fingerprint != expected
    # A legitimately resealed new bundle must still be rejected against an old checkpoint.
    from pithtrain.tasks.prepare_omni_data import write_bundle

    bundle = json.loads((cfg.dataset / "bundle.json").read_text())
    (cfg.dataset / "image.bin").write_bytes(b"new image content")
    write_bundle(cfg.dataset, bundle, files)
    assert PreparedBundleData(cfg, tc).data_fingerprint != expected


@pytest.fixture
def cpu_runtime(monkeypatch):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    for name, value in dict(
        model=model, optimizers=(optimizer,), schedulers=(scheduler,), offloaded=False
    ).items():
        monkeypatch.setattr(checkpoint.training, name, value, raising=False)
    monkeypatch.setattr(
        checkpoint.logging, "stdout", SimpleNamespace(info=lambda *_: None), raising=False
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_, **__: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", torch.set_rng_state)
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    return checkpoint.CheckpointState(model, (optimizer,), (scheduler,))


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_contract_roundtrip_and_reject_before_restore(tmp_path, cpu_runtime, legacy):
    state = cpu_runtime
    expected = copy.deepcopy(state.state_dict())
    location = tmp_path / "torch-dcp/00000001"
    if legacy:
        payload = copy.deepcopy(expected)
        payload["data"] = dict(version=1, fingerprint="same-contract", consumed_samples=128)
        dcp.save({"app": payload}, checkpoint_id=location)
    else:
        checkpoint.save_checkpoint(tmp_path, 1, data_fingerprint="same-contract")
        probe = {"app": {"data_fingerprint": ""}}
        dcp.load(probe, checkpoint_id=location)
        assert probe["app"]["data_fingerprint"] == "same-contract"
    with torch.no_grad():
        state.model.weight.add_(10)
    unchanged = copy.deepcopy(state.state_dict())
    with pytest.raises(ValueError, match="data fingerprint differs"):
        checkpoint.load_checkpoint(tmp_path, 1, data_fingerprint="changed-contract")
    assert_state_equal(state.state_dict(), unchanged)
    checkpoint.load_checkpoint(tmp_path, 1, data_fingerprint="same-contract")
    assert_state_equal(state.state_dict(), expected)


@pytest.mark.parametrize("model_only", [False, True])
def test_checkpoint_without_contract_remains_compatible(tmp_path, cpu_runtime, model_only):
    state = cpu_runtime
    expected = copy.deepcopy(state.state_dict())
    payload = {"model": expected["model"]} if model_only else expected
    dcp.save({"app": payload}, checkpoint_id=tmp_path / "torch-dcp/00000001")
    with torch.no_grad():
        state.model.weight.add_(10)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        checkpoint.load_checkpoint(tmp_path, 1, data_fingerprint="current-contract")
    missing = [warning for warning in seen if "no data fingerprint" in str(warning.message)]
    assert bool(missing) == (not model_only)
    torch.testing.assert_close(state.model.state_dict(), expected["model"], rtol=0, atol=0)
