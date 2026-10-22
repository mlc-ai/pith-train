"""Speech collective contracts; these CPU checks are not full-size GPU acceptance."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from pithtrain.modules.omni_speech_fsdp import (
    _shard_units,
    require_complete_optimizer_state,
    validate_speech_collective_plan,
)


def batch(*, frames=2, media=False):
    return dict(
        codes=torch.ones(1, frames, 16, dtype=torch.long),
        input_ids=torch.ones(1, 3, dtype=torch.long),
        model_context={"input_features": torch.ones(1, 2)} if media else {},
        codebook_mask=torch.ones(1, frames, 16, dtype=torch.bool),
    )


def test_target_masks_are_not_invocation_plan():
    first, second = batch(), batch()
    second["codebook_mask"].zero_()
    assert validate_speech_collective_plan(
        [first], "codec_only"
    ) == validate_speech_collective_plan([second], "codec_only")


@pytest.mark.parametrize("mode", ["codec_only", "text_and_codec", "waveform"])
def test_supported_modes_require_one_row(mode):
    item = batch()
    item["codes"] = item["codes"].expand(2, -1, -1)
    with pytest.raises(ValueError, match="B=1"):
        validate_speech_collective_plan([item], mode)


def test_predictor_zero_frame_branch_is_explicit():
    assert validate_speech_collective_plan(
        [batch(frames=0)], "codec_only"
    ) != validate_speech_collective_plan([batch()], "codec_only")
    with pytest.raises(ValueError, match="at least one frame"):
        validate_speech_collective_plan([batch(frames=0)], "waveform")


class DirectLayer(nn.Linear):
    def reference_forward(self, x):
        return self.reference_forward_with_hidden(x)[0]

    def reference_forward_with_hidden(self, x):
        return self(x), x


class Thinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_count, self.stage_index = 1, 0
        self.embed_tokens, self.norm, self.lm_head = (
            nn.Embedding(8, 4),
            nn.LayerNorm(4),
            nn.Linear(4, 8),
        )
        self.layers = nn.ModuleDict({"0": DirectLayer(4, 4)})
        self.visual, self.audio_tower = nn.Linear(4, 4), nn.Linear(4, 4)

    def forward_prolog(self, ids):
        return self.embed_tokens(ids)

    def forward_posemb(self, length):
        return None

    def forward_epilog(self, hidden):
        return self.lm_head(self.norm(hidden))


class Trunk(nn.Module):
    def __init__(self, predictor=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.codec_embedding = (
            nn.ModuleList([nn.Embedding(8, 4), nn.Embedding(8, 4)])
            if predictor
            else nn.Embedding(8, 4)
        )


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Trunk(predictor=True)
        self.lm_head = nn.ModuleList([nn.Linear(4, 8), nn.Linear(4, 8)])

    def teacher_force(self, x):
        return x


def shaped_model():
    model = nn.Module()
    model.thinker, model.talker, model.code2wav = Thinker(), nn.Module(), nn.Linear(4, 4)
    model.talker.model = Trunk()
    model.talker.text_projection = nn.Linear(4, 4)
    model.talker.hidden_projection = nn.Linear(4, 4)
    model.talker.codec_head = nn.Linear(4, 8)
    model.talker.code_predictor = Predictor()
    model.decode_training_codes = lambda codes: model.code2wav(codes)
    return model


def test_bottom_up_units_preserve_single_direct_entry_and_repeated_edges():
    model = shaped_model()
    before = {name: id(value) for name, value in model.named_parameters()}
    units = _shard_units(model)
    table = {id(module): methods for module, methods, _ in units}
    assert table[id(model.thinker.layers["0"])] == ("reference_forward_with_hidden",)
    assert table[id(model.talker.code_predictor)] == ("teacher_force",)
    assert id(model.thinker.embed_tokens) in table
    assert id(model.talker.model.codec_embedding) in table
    assert all(id(m) in table for m in model.talker.code_predictor.model.codec_embedding)
    assert units[-1][0] is model
    assert before == {name: id(value) for name, value in model.named_parameters()}


def test_default_strict_optimizer_contract_does_not_fake_unused_updates():
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    optimizer = torch.optim.AdamW(model.parameters())
    require_complete_optimizer_state(model, optimizer, 0)
    model[0](torch.ones(2, 3)).sum().backward()
    optimizer.step()
    with pytest.raises(ValueError, match="Conditional/missing"):
        require_complete_optimizer_state(model, optimizer, 1)
    assert all(p not in optimizer.state for p in model[1].parameters())


def plan_worker(rank, root):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{Path(root) / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        group = dist.group.WORLD
        item = batch()
        if rank:
            item["codebook_mask"].zero_()
        validate_speech_collective_plan([item], "text_and_codec", group)
        for bad in ([], [batch(frames=0)], [batch(media=True)], [batch(), batch()]):
            with pytest.raises(ValueError):
                validate_speech_collective_plan(bad if rank else [batch()], "codec_only", group)
        with pytest.raises(ValueError):
            validate_speech_collective_plan([batch()], "waveform" if rank else "codec_only", group)
    finally:
        dist.destroy_process_group()


def test_real_two_rank_plan_rejection_without_forward_deadlock(tmp_path):
    mp.spawn(plan_worker, args=(str(tmp_path),), nprocs=2, join=True)


def materialization_worker(rank, root):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from pithtrain.modules.omni_speech_fsdp import (
        apply_speech_fsdp,
        load_speech_local_shards,
        materialize_speech_shards,
    )

    dist.init_process_group(
        "gloo",
        init_method=f"file://{Path(root) / 'materialize-rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        with torch.device("meta"):
            model = shaped_model()
            model.register_buffer("materialized_buffer", torch.empty(2))
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("code2wav."))
        original_shapes = {name: p.shape for name, p in model.named_parameters()}
        apply_speech_fsdp(model, mesh, mode="waveform")
        assert all(isinstance(p, DTensor) and p.is_meta for p in model.parameters())
        materialize_speech_shards(model, device="cpu")
        assert model._omni_speech_fsdp_plan["weights_initialized"] is False
        local = {name: p.to_local() for name, p in model.named_parameters()}
        local.update(dict(model.named_buffers()))
        for name, p in model.named_parameters():
            assert p.shape == original_shapes[name]
            assert p.to_local().numel() * 2 == p.numel()
        with torch.no_grad():
            for value in local.values():
                value.zero_()
        shards = {name: torch.full_like(value, rank + 0.25) for name, value in local.items()}
        bad = dict(shards)
        if rank:
            bad.pop(next(iter(bad)))
        with pytest.raises(ValueError, match="Local shard keys differ"):
            load_speech_local_shards(model, bad, rank=rank, world_size=2)
        assert all(torch.count_nonzero(value) == 0 for value in local.values())
        with pytest.raises(ValueError, match="identity mismatch"):
            load_speech_local_shards(model, shards, rank=0, world_size=2)
        assert all(torch.count_nonzero(value) == 0 for value in local.values())
        load_speech_local_shards(model, shards, rank=rank, world_size=2)
        assert model._omni_speech_fsdp_plan["weights_initialized"] is True
        for name, value in local.items():
            torch.testing.assert_close(value, shards[name], rtol=0, atol=0)
        with pytest.raises(ValueError, match="initialization-only"):
            load_speech_local_shards(model, shards, rank=rank, world_size=2)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], foreach=False
        )
        # The actual custom root entry must initialize the common FSDP root.
        loss = model.decode_training_codes(torch.ones(1, 4)).float().sum()
        loss.backward()
        assert all(p.grad is not None for p in model.code2wav.parameters())
        optimizer.step()
        assert all(isinstance(p, DTensor) and not p.is_meta for p in model.parameters())
    finally:
        dist.destroy_process_group()


def test_meta_model_materializes_only_local_shards_and_loads_actual_values(tmp_path):
    mp.spawn(materialization_worker, args=(str(tmp_path),), nprocs=2, join=True)


def batch_validation_worker(rank, root):
    from tests.test_omni_speech_batch_contract import sample

    dist.init_process_group(
        "gloo",
        init_method=f"file://{Path(root) / 'batch-rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model, item = sample()
        # The config-only trap separately proves validator purity; this helper
        # also inspects the FSDP descriptor of actual module instances.
        from types import SimpleNamespace

        model = SimpleNamespace(config=model.config)
        validate_speech_collective_plan([item], "text_and_codec", dist.group.WORLD, model=model)
        if rank:
            item["code_lengths"] = torch.tensor([99])
        with pytest.raises(ValueError, match="Codec lengths"):
            validate_speech_collective_plan([item], "text_and_codec", dist.group.WORLD, model=model)
        assert model.config.talker_config.num_code_groups == 16
    finally:
        dist.destroy_process_group()


def test_native_batch_contract_rejects_one_rank_before_fsdp_forward(tmp_path):
    mp.spawn(batch_validation_worker, args=(str(tmp_path),), nprocs=2, join=True)
