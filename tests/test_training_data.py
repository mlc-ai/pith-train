"""CPU checks of token batch contents and replay from the training step.

Only the shuffle owner's rank, placement and barrier are mocked: the real memmap,
shuffle, DP/CP slicing and batch adapter execute.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.training_data import PreparedBundleData, TokenBinData
from pithtrain.operators.cp_sequence import zigzag_spans


@pytest.fixture
def text_corpus(tmp_path, monkeypatch, request):
    sequence_length = getattr(request, "param", 16)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    token_root = tmp_path / "tokens/train"
    token_root.mkdir(parents=True)
    # Simulate the one shuffle writer, then readers on the other ranks. Rewriting an
    # already mapped cache is invalid on Windows, where all rank views share this process.
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda *args, **kwargs: int((token_root / "indices.npy").exists()),
    )
    inputs, labels = [], []
    # Distinct shards exercise concat offsets; short tails must stay dropped.
    for shard, count in enumerate((17, 19)):
        tail = max(1, sequence_length // 2 - 1)
        tokens = np.arange(count * sequence_length + tail, dtype=np.uint32) + shard * 10000
        with (token_root / f"{shard:05d}.bin").open("wb") as stream:
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
    from pithtrain.tasks.prepare_omni_data import write_bundle, write_json

    recipe = dict(stages={"text": ["text"]}, sampling_weights={"text": 1.0}, batch={})
    write_json(root / "recipe.json", dict(recipe=recipe, stage="text", format_version=3))
    (root / "rejected.jsonl").write_text("")
    files, manifests = ["recipe.json", "rejected.jsonl"], []
    for index, tokens in enumerate(sorted(token_root.glob("*.bin"))):
        name = f"train/text-{index:05d}.jsonl"
        (root / name).parent.mkdir(exist_ok=True)
        (root / name).write_text(
            json.dumps(dict(id=str(index), text="Fixture text.", media=[])) + "\n"
        )
        manifests.append(name)
        files.extend([name, tokens.relative_to(root).as_posix()])
    write_bundle(
        root,
        dict(
            status="complete",
            recipe=recipe,
            manifests={"train": {"text": manifests}},
            statistics={"train": {"text": {"samples": 2}}},
        ),
        files,
    )
    return root, cfg


def prepared_config(root):
    cfg = DataCfg()
    cfg.dataset, cfg.format = root, "prepared_bundle"
    return cfg


def tensors(batches):
    assert all(mb.cu_seqlens is None and mb.media_context is None for mb in batches)
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
            data = TokenBinData(
                root, cfg, dp_rank=dp_rank, dp_size=dp_size, cp_rank=cp_rank, cp_size=cp_size
            )
            front, back = zigzag_spans(cp_rank, cp_size, cfg.sequence_length)
            positions = [*front, *back]
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
                return PreparedBundleData(prepared_config(bundle_root), cfg, **ranks)
            return TokenBinData(token_root, cfg, **ranks)

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


def test_prepared_text_replays_by_step(text_bundle):
    root, cfg = text_bundle
    ranks = dict(dp_rank=1, dp_size=2, cp_rank=1, cp_size=2)
    legacy = TokenBinData(root / "tokens/train", cfg, **ranks)
    data = PreparedBundleData(prepared_config(root), cfg, **ranks)
    for step in (1, 0, 1, 2):
        expected = tensors(legacy.get_batch(step, "cpu"))
        torch.testing.assert_close(tensors(data.get_batch(step, "cpu")), expected, rtol=0, atol=0)
        restored = PreparedBundleData(prepared_config(root), cfg, **ranks)
        torch.testing.assert_close(
            tensors(restored.get_batch(step, "cpu")), expected, rtol=0, atol=0
        )
