"""CPU checks of dense batch contents and the shared checkpoint boundary.

Only corpus shuffle placement and its barrier are mocked: the real memmap,
shuffle, DP/CP slicing, batch adapter, and DCP serialization code execute.
"""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.utils._pytree import tree_flatten

from pithtrain.modules.checkpoint import CheckpointState
from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.training_data import DensePretrainData, OmniPretrainData
from pithtrain.operators.cp_sequence import zigzag_spans


@pytest.fixture
def text_corpus(tmp_path, monkeypatch, request):
    sequence_length = getattr(request, "param", 16)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    token_root = tmp_path / "tokens/train"
    token_root.mkdir(parents=True)
    inputs, labels = [], []
    # Distinct shards exercise concat offsets; short tails must stay dropped.
    for shard, count in enumerate((17, 19)):
        tail = max(1, sequence_length // 2 - 1)
        tokens = np.arange(count * sequence_length + tail, dtype=np.uint32) + shard * 10000
        with (token_root / f"{shard}.bin").open("wb") as stream:
            np.save(stream, tokens)
            np.save(stream, np.array([len(tokens)], dtype=np.uint64))
        inputs.append(
            tokens[: count * sequence_length].astype(np.int64).reshape(count, sequence_length)
        )
        labels.append(
            tokens[1 : count * sequence_length + 1].astype(np.int64).reshape(count, sequence_length)
        )
    cfg = SimpleNamespace(
        global_batch_size=8,
        micro_batch_size=1,
        sequence_length=sequence_length,
        seed=431,
        max_steps=3,
    )
    return (
        token_root,
        cfg,
        torch.from_numpy(np.concatenate(inputs)),
        torch.from_numpy(np.concatenate(labels)),
    )


@pytest.fixture
def text_bundle(text_corpus):
    token_root, cfg, _, _ = text_corpus
    root = token_root.parent.parent
    bundle = dict(
        status="complete",
        recipe=dict(stages={"text": ["text"]}, sampling_weights={"text": 1.0}, batch={}),
        manifests={"train": {"text": []}},
        statistics={"train": {"text": {"samples": 2}}},
    )
    (root / "bundle.json").write_text(json.dumps(bundle))
    paths = [root / "bundle.json", *sorted(token_root.glob("*.bin"))]
    (root / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
            for path in paths
        )
    )
    return root, cfg


def prepared_config(root):
    cfg = DataCfg()
    cfg.dataset, cfg.format = root, "prepared_bundle"
    return cfg


def tensors(batches):
    assert all(mb.cu_seqlens is None and mb.model_context is None for mb in batches)
    assert all(
        mb.model_inputs[0].dtype == mb.objective_inputs[0].dtype == torch.long for mb in batches
    )
    return torch.cat([mb.model_inputs[0] for mb in batches]), torch.cat(
        [mb.objective_inputs[0] for mb in batches]
    )


@pytest.mark.parametrize("dp_size,cp_size,mbs", [(1, 1, 1), (2, 1, 2), (2, 2, 2), (4, 2, 1)])
def test_dense_batches_match_global_token_oracle(text_corpus, dp_size, cp_size, mbs):
    root, cfg, inputs, labels = text_corpus
    cfg.micro_batch_size = mbs
    rng = torch.get_rng_state().clone()
    # CPU shuffle is only a test stand-in for GPU placement, not a GPU shuffle claim.
    order = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(cfg.seed))
    for dp_rank in range(dp_size):
        for cp_rank in range(cp_size):
            data = DensePretrainData(
                root, cfg, dp_rank=dp_rank, dp_size=dp_size, cp_rank=cp_rank, cp_size=cp_size
            )
            front, back = zigzag_spans(cp_rank, cp_size, cfg.sequence_length)
            positions = [*front, *back]
            assert data.checkpoint_state is None
            for step in range(cfg.max_steps):
                batch = data.get_batch(step, "cpu")
                assert len(batch) == cfg.global_batch_size // dp_size // mbs
                assert all(mb.model_inputs[0].shape == (mbs, len(positions)) for mb in batch)
                selected = (
                    order[step * 8 : (step + 1) * 8].reshape(-1, dp_size, mbs)[:, dp_rank].flatten()
                )
                actual_x, actual_y = tensors(batch)
                torch.testing.assert_close(actual_x, inputs[selected][:, positions], rtol=0, atol=0)
                torch.testing.assert_close(actual_y, labels[selected][:, positions], rtol=0, atol=0)
                data.commit_step(step)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("text_corpus", [1, 5], indirect=True)
@pytest.mark.parametrize("prepared", [False, True])
def test_cp1_odd_text_keeps_every_token_and_target(text_corpus, text_bundle, prepared):
    token_root, cfg, inputs, labels = text_corpus
    bundle_root, _ = text_bundle
    cfg.micro_batch_size = 2
    dp_size = 2
    # The oracle is the complete original stream, without CP splitting arithmetic.
    order = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(cfg.seed))
    selected = order[: cfg.max_steps * cfg.global_batch_size].reshape(
        cfg.max_steps, -1, dp_size, cfg.micro_batch_size
    )
    for dp_rank in range(dp_size):
        ranks = dict(dp_rank=dp_rank, dp_size=dp_size, cp_rank=0, cp_size=1)

        def make_data():
            if prepared:
                return OmniPretrainData(prepared_config(bundle_root), cfg, **ranks)
            return DensePretrainData(token_root, cfg, **ranks)

        data = make_data()
        for step in range(cfg.max_steps):
            batch = data.get_batch(step, "cpu")
            assert all(
                mb.model_inputs[0].shape == (cfg.micro_batch_size, cfg.sequence_length)
                for mb in batch
            )
            indices = selected[step, :, dp_rank].flatten()
            actual_inputs, actual_labels = tensors(batch)
            torch.testing.assert_close(actual_inputs, inputs[indices], rtol=0, atol=0)
            torch.testing.assert_close(actual_labels, labels[indices], rtol=0, atol=0)
            data.commit_step(step)
            if prepared and step == 0:
                state = data.checkpoint_state.state_dict()
                data = make_data()
                data.checkpoint_state.load_state_dict(state)


def test_prepared_text_uses_same_batches_and_committed_resume(text_bundle):
    root, cfg = text_bundle
    ranks = dict(dp_rank=1, dp_size=2, cp_rank=1, cp_size=2)
    legacy = DensePretrainData(root / "tokens/train", cfg, **ranks)
    data = OmniPretrainData(prepared_config(root), cfg, **ranks)
    assert data.checkpoint_state is data
    torch.testing.assert_close(
        tensors(data.get_batch(0, "cpu")), tensors(legacy.get_batch(0, "cpu")), rtol=0, atol=0
    )
    with pytest.raises(RuntimeError, match="optimizer step"):
        data.checkpoint_state.state_dict()
    with pytest.raises(ValueError, match="not read"):
        data.commit_step(1)
    data.commit_step(0)
    saved = data.checkpoint_state.state_dict()
    assert set(saved) == {"version", "fingerprint", "consumed_samples"}
    assert saved["version"] == 1 and saved["consumed_samples"] == cfg.global_batch_size
    expected = tensors(data.get_batch(1, "cpu"))
    restored = OmniPretrainData(prepared_config(root), cfg, **ranks)
    restored.checkpoint_state.load_state_dict(saved)
    with pytest.raises(ValueError, match="disagree"):
        restored.get_batch(0, "cpu")
    torch.testing.assert_close(tensors(restored.get_batch(1, "cpu")), expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors(legacy.get_batch(1, "cpu")), expected, rtol=0, atol=0)
    restored.commit_step(1)
    with pytest.raises(ValueError, match="not read"):
        restored.commit_step(1)
    changed_cfg = copy.copy(cfg)
    changed_cfg.seed += 1
    changed = OmniPretrainData(prepared_config(root), changed_cfg, **ranks)
    with pytest.raises(ValueError, match="differs"):
        changed.checkpoint_state.load_state_dict(saved)


@pytest.mark.parametrize("prepared", [False, True])
def test_dense_checkpoint_compatibility_and_next_batch(text_bundle, tmp_path, prepared):
    root, cfg = text_bundle

    def source():
        return (
            OmniPretrainData(prepared_config(root), cfg)
            if prepared
            else DensePretrainData(root / "tokens/train", cfg)
        )

    data = source()
    data.get_batch(0, "cpu")
    data.commit_step(0)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    state = CheckpointState(model, (optimizer,), (scheduler,), data_state=data.checkpoint_state)
    expected_state = copy.deepcopy(state.state_dict())
    assert ("data" in expected_state) == prepared
    checkpoint = tmp_path / "checkpoint"
    # Legacy serialization deliberately omits data_state, as older checkpoints did.
    to_save = state if prepared else CheckpointState(model, (optimizer,), (scheduler,))
    dcp.save({"app": to_save}, checkpoint_id=checkpoint)
    restored_data = source()
    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    restored = CheckpointState(
        restored_model,
        (restored_optimizer,),
        (restored_scheduler,),
        data_state=restored_data.checkpoint_state,
    )
    dcp.load({"app": restored}, checkpoint_id=checkpoint)
    # Canonical optimizer FQNs and the data fingerprint are strings, not tensors.
    actual_leaves, actual_spec = tree_flatten(restored.state_dict())
    expected_leaves, expected_spec = tree_flatten(expected_state)
    assert actual_spec == expected_spec
    for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert type(actual) is type(expected) and actual == expected
    torch.testing.assert_close(
        tensors(restored_data.get_batch(1, "cpu")),
        tensors(data.get_batch(1, "cpu")),
        rtol=0,
        atol=0,
    )
    restored_data.commit_step(1)


def test_dense_rejects_missing_and_insufficient_corpus(text_corpus, tmp_path):
    root, cfg, _, _ = text_corpus
    with pytest.raises(ValueError, match="No token shards"):
        DensePretrainData(tmp_path / "absent", cfg)
    cfg.max_steps = 5
    with pytest.raises(AssertionError, match="run needs"):
        DensePretrainData(root, cfg)


def test_pre_datacfg_text_checkpoint_keeps_identity_and_next_batch(text_bundle):
    root, cfg = text_bundle
    # Independently construct the version-1 state written before DataCfg. In this
    # fixture the old recipe has 2 text records, rounded to one global batch of 8.
    identity = dict(
        version=1,
        bundle_sha256=hashlib.sha256((root / "bundle.json").read_bytes()).hexdigest(),
        stage="text",
        global_batch_size=8,
        micro_batch_size=1,
        sequence_length=16,
        seed=431,
        weights={"text": 1.0},
        epoch_samples=8,
    )
    saved = dict(
        version=1,
        fingerprint=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        consumed_samples=8,
    )
    ranks = dict(dp_rank=1, dp_size=2, cp_rank=1, cp_size=2)
    restored = OmniPretrainData(prepared_config(root), cfg, **ranks)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    dense = DensePretrainData(root / "tokens/train", cfg, **ranks)
    torch.testing.assert_close(
        tensors(restored.get_batch(1, "cpu")), tensors(dense.get_batch(1, "cpu")), rtol=0, atol=0
    )
    restored.commit_step(1)
    assert restored.state_dict() == dict(saved, consumed_samples=16)
