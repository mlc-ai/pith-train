"""Real CPU FSDP save -> fresh unsharded native state; not GPU forward acceptance.

Thinker constructors reuse the existing unchanged-AST CPU fixture. No Thinker
kernel is replaced or run. Controlled nonzero gradients test serialization;
the restored native Code2Wav forward is real.
"""

import copy
import json
import math
import shutil
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from pithtrain.modules import distributed_checkpoint as checkpoint
from pithtrain.modules.omni_speech_fsdp import (
    apply_speech_fsdp,
    load_speech_local_shards,
    materialize_speech_shards,
    native_speech_checkpoint_state,
    speech_checkpoint_context,
)
from pithtrain.modules.omni_speech_generation_checkpoint import read_speech_generation_state
from pithtrain.modules.omni_speech_initialization import _sha
from pithtrain.modules.resume_semantics import model_contract
from tests.test_omni_full_converter import tiny_raw
from tests.test_omni_speech_initialization import native_meta


def target(raw):
    model = native_meta(raw)
    # Identify the existing AST fixture honestly as a test implementation. The
    # same classes/source are used by both producer and consumer processes.
    for module in model.modules():
        if type(module).__module__ == "builtins":
            type(module).__module__ = __name__
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("code2wav."))
    return model


def initial(shape, index, offset=0):
    width = math.prod(shape[1:])
    return (
        (torch.arange(math.prod(shape), dtype=torch.float32) + offset * width).reshape(shape) * 1e-6
        + 0.02
        + index * 1e-5
    )


class Cursor:
    def state_dict(self):
        return dict(scanned_batches=1, completed_updates=1, data_cursor=1)

    def validate_state_dict(self, state):
        assert state == self.state_dict()

    def load_state_dict(self, state):
        self.validate_state_dict(state)


def producer(rank, root, raw):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(root / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        model = target(raw)
        apply_speech_fsdp(
            model, init_device_mesh("cpu", (2,), mesh_dim_names=("dp",)), mode="waveform"
        )
        materialize_speech_shards(model, device="cpu")
        shards = {}
        for index, (name, p) in enumerate(model.named_parameters()):
            local = p.to_local()
            _, offset = Shard.local_shard_size_and_offset(p.shape[0], 2, rank)
            shards[name] = initial(local.shape, index, offset)
        load_speech_local_shards(model, shards, rank=rank, world_size=2)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0,
            foreach=False,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, 0.1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        contract = dict(
            hsdp_replica=1, speech_task=dict(schema=2, model_contract=model_contract(model))
        )
        adapter = native_speech_checkpoint_state(Cursor(), contract=contract)
        with speech_checkpoint_context(model, optimizer, scheduler):
            checkpoint.save(root / "checkpoint", 1, data_state=adapter, config=contract)
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("speech-generation-checkpoint")
    raw = tiny_raw()
    mp.spawn(producer, args=(str(root), raw), nprocs=2, join=True)
    location = root / "checkpoint/torch-dcp/00000001"
    model = target(raw)
    expected = {k: initial(v.shape, i) for i, (k, v) in enumerate(model.state_dict().items())}
    model.load_state_dict(expected, assign=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0, foreach=False
    )
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, 0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return location, raw, model


def options(location):
    return dict(
        expected_manifest_sha256=_sha(location / checkpoint.MANIFEST),
        max_state_bytes=2**26,
        max_rank_file_bytes=2**28,
    )


def test_real_two_rank_checkpoint_to_fresh_generation_model(saved):
    location, raw, reference = saved
    fresh = target(raw)
    before_rng = torch.get_rng_state().clone()
    values, receipt = read_speech_generation_state(fresh, location, **options(location))
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(p.is_meta and p.grad is None for p in fresh.parameters())
    assert values.keys() == reference.state_dict().keys()
    for name, value in values.items():
        assert torch.equal(value, reference.state_dict()[name]), name
    fresh.load_state_dict(values, assign=True)
    codes = torch.zeros(1, raw["code2wav_config"]["num_quantizers"], 3, dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(fresh.code2wav(codes), reference.code2wav(codes))
    assert receipt["step"] == receipt["updates"] == 1
    assert receipt["checkpoint_world_size"] == 2
    assert not receipt["restored_training_state"] and not receipt["restored_generation_cache"]


@pytest.mark.parametrize(
    "problem", ["corrupt", "missing_rank", "wrong_rank", "missing_tensor", "config", "state_cap"]
)
def test_rejects_before_touching_target(saved, tmp_path, problem):
    source, raw, _ = saved
    location = tmp_path / "checkpoint"
    shutil.copytree(source, location)
    fresh = target(copy.deepcopy(raw))
    kwargs = options(location)
    rank = location / "rank-00001.pt"
    if problem == "corrupt":
        data = bytearray(rank.read_bytes())
        data[-20] ^= 1
        rank.write_bytes(data)
    elif problem == "missing_rank":
        rank.unlink()
    elif problem in ("wrong_rank", "missing_tensor"):
        payload = torch.load(rank, weights_only=True)
        if problem == "wrong_rank":
            payload["header"]["topology"]["rank"] = 0
        else:
            payload["state"]["model"].pop(next(iter(payload["state"]["model"])))
        torch.save(payload, rank)
        manifest = json.loads((location / checkpoint.MANIFEST).read_text())
        manifest["files"][rank.name] = _sha(rank)
        (location / checkpoint.MANIFEST).write_text(json.dumps(manifest))
        kwargs = options(location)
    elif problem == "config":
        fresh.config.code2wav_config.rope_parameters["rope_theta"] *= 2
    else:
        kwargs["max_state_bytes"] = 1
    before_rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, KeyError)):
        read_speech_generation_state(fresh, location, **kwargs)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(p.is_meta and p.grad is None for p in fresh.parameters())
