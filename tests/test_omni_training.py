"""CPU tests for the training data boundary, checkpoint state and rank reductions."""

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn.functional as F

from pithtrain.modules.checkpoint import CheckpointState
from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.training_data import (
    OmniDataCfg,
    OmniPretrainData,
    global_loss_mean,
    global_target_count,
)
from tests import test_qwen3_omni_data as data_fixtures

manifest = data_fixtures.manifest
processor_config = data_fixtures.processor_config


@pytest.fixture
def training_bundle(manifest):
    root = manifest.parent
    recipe_path = (
        Path(__file__).parents[1] / "examples/prepare_omni_data/qwen3-omni-training/config.json"
    )
    recipe = json.loads(recipe_path.read_text())
    files, manifests, statistics = [], {}, {}
    for row in [json.loads(line) for line in manifest.read_text().splitlines()]:
        kind = row["media"][0]["type"] if row["media"] else "text"
        filename = f"train/{kind}.jsonl"
        (root / filename).parent.mkdir(exist_ok=True)
        (root / filename).write_text(json.dumps(row) + "\n")
        files.append(filename)
        files.extend(item["path"] for item in row["media"])
        manifests[kind], statistics[kind] = [filename], {"samples": 1}
    bundle = dict(
        status="complete",
        recipe=recipe,
        manifests={"train": manifests},
        statistics={"train": statistics},
    )
    (root / "bundle.json").write_text(json.dumps(bundle))
    files.append("bundle.json")
    (root / "checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n" for name in files
        )
    )
    return root


def source(root, *, stage="video", rank=0, world=1, seed=1234, cp=1):
    cfg = OmniDataCfg()
    cfg.stage, cfg.epoch_samples = stage, 8
    tc = SimpleNamespace(global_batch_size=4, micro_batch_size=1, sequence_length=1024, seed=seed)
    return OmniPretrainData(root, cfg, tc, dp_rank=rank, dp_size=world, cp_size=cp)


def consume(data, step):
    data.begin_step(step)
    batches = data.media_microbatches("cpu")
    data.commit_step(step)
    return batches


def compare_batches(actual, expected):
    assert [mb.sample_ids for mb in actual] == [mb.sample_ids for mb in expected]
    for left, right in zip(actual, expected, strict=True):
        assert left.model_context.keys() == right.model_context.keys()
        torch.testing.assert_close(left.objective_inputs[0], right.objective_inputs[0])
        for name in left.model_context:
            torch.testing.assert_close(left.model_context[name], right.model_context[name])


def test_training_media_batches_dp_resume_and_epoch(training_bundle, processor_config):
    data = source(training_bundle)
    rng_before = torch.get_rng_state().clone()
    first = consume(data, 0)
    assert torch.equal(torch.get_rng_state(), rng_before), "Data iteration changed model RNG"
    state = data.state_dict()
    expected = consume(data, 1)
    next_epoch = consume(data, 2)
    restored = source(training_bundle)
    restored.load_state_dict(state)
    compare_batches(consume(restored, 1), expected)
    compare_batches(consume(restored, 2), next_epoch)
    ranks = [consume(source(training_bundle, rank=rank, world=2), 0) for rank in range(2)]
    compare_batches([mb for pair in zip(*ranks) for mb in pair], first)
    for micro in first + expected + next_epoch:
        assert micro.model_inputs[0] is micro.model_context["input_ids"]
        assert micro.model_inputs[0].shape[0] == 1
        assert micro.model_inputs[0].shape == micro.objective_inputs[0].shape
        assert all(value.device.type == "cpu" for value in micro.model_context.values())
    with pytest.raises(ValueError, match="differs"):
        source(training_bundle, seed=42).load_state_dict(state)


def test_training_rejects_unsupported_layout_and_uncommitted_save(
    training_bundle, processor_config
):
    with pytest.raises(ValueError, match="CP=1"):
        source(training_bundle, cp=2)
    data = source(training_bundle)
    _, config = processor_config
    with pytest.raises(ValueError, match="does not implement"):
        data.validate_model(type("TextOnly", (), {}), config)
    data.begin_step(0)
    data.media_microbatches("cpu")
    with pytest.raises(RuntimeError, match="optimizer step"):
        data.state_dict()
    with pytest.raises(ValueError, match="disagree"):
        data.begin_step(0)


def test_checkpoint_restores_model_optimizer_and_next_data(
    training_bundle, processor_config, tmp_path
):
    data = source(training_bundle)
    consume(data, 0)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    expected_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint = tmp_path / "checkpoint"
    state = CheckpointState(model, (optimizer,), (scheduler,), data_state=data)
    dcp.save({"app": state}, checkpoint_id=checkpoint)
    expected_batch = consume(data, 1)
    restored_data = source(training_bundle)
    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    restored = CheckpointState(
        restored_model, (restored_optimizer,), (restored_scheduler,), data_state=restored_data
    )
    dcp.load({"app": restored}, checkpoint_id=checkpoint)
    assert restored_data.consumed_samples == 4
    assert restored_optimizer.state and restored_scheduler.last_epoch == scheduler.last_epoch
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name])
    compare_batches(consume(restored_data, 1), expected_batch)


def _reduction_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    # Two separate pipeline stages, each with two data/context ranks. Counting
    # the whole world would double the denominator and fail the update comparison.
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    group = groups[rank // 2]
    local_rank = rank % 2
    model = torch.nn.Linear(2, 3, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.arange(6, dtype=torch.float64).reshape(3, 2) / 10)
    inputs = torch.arange(8, dtype=torch.float64).reshape(4, 2) / 10
    labels = torch.tensor([-100] * 4 if local_rank == 0 else [0, 1, -100, 2])
    micro = Microbatch(model_inputs=(inputs,), cu_seqlens=None, objective_inputs=(labels,))
    count = global_target_count([micro], group)
    assert count.item() == 3
    loss_sum = F.cross_entropy(model(inputs), labels, ignore_index=-100, reduction="sum")
    loss_sum.backward()
    dist.all_reduce(model.weight.grad, group=group)
    model.weight.grad.div_(count)
    mean = global_loss_mean(loss_sum, count, group)
    reference = torch.nn.Linear(2, 3, bias=False).double()
    reference.load_state_dict(model.state_dict())
    expected = F.cross_entropy(reference(inputs), torch.tensor([0, 1, -100, 2]), ignore_index=-100)
    expected.backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    torch.testing.assert_close(mean.double(), expected.detach(), rtol=1e-6, atol=1e-6)
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    torch.optim.SGD(reference.parameters(), lr=0.1).step()
    torch.testing.assert_close(model.weight, reference.weight)
    dist.destroy_process_group()


def test_global_target_normalization_excludes_pipeline_duplicates(tmp_path):
    mp.spawn(_reduction_worker, args=(str(tmp_path / "rendezvous"),), nprocs=4, join=True)


def test_worker_prefetch_does_not_advance_checkpoint(training_bundle, processor_config):
    expected = source(training_bundle)
    first, second = consume(expected, 0), consume(expected, 1)
    data = source(training_bundle)
    data.cfg.num_workers = 1
    compare_batches(consume(data, 0), first)
    saved = data.state_dict()
    assert saved["consumed_samples"] == 4
    restored = source(training_bundle)
    restored.cfg.num_workers = 1
    restored.load_state_dict(saved)
    compare_batches(consume(restored, 1), second)
