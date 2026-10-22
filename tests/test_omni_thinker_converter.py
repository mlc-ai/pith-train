"""CPU complete-Thinker files and strict-state roundtrips; no CUDA claims."""

import copy
import json
import shutil
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.tensor import Shard, distribute_tensor
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration as HFThinker,
)

from pithtrain.contexts import distributed, training
from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
from pithtrain.modules.distributed_checkpoint import _pack, _runtime
from pithtrain.tasks.convert_checkpoint import dcp2hf, hf2dcp
from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import ConversionError
from pithtrain.tasks.convert_checkpoint.qwen3_omni_thinker import (
    SIDECAR,
    canonical_to_hf,
    hf_to_canonical,
    parameter_shapes,
    read_hf,
    thinker_config,
)
from pithtrain.tasks.convert_checkpoint.strict_omni import RESUME_DIR, read_payloads
from tests import test_qwen3_omni_checkpoint as fixtures
from tests import test_strict_omni_converter as strict
from tests.test_qwen3_omni_vision import vision_config
from tests.test_strict_checkpoint import assert_tree_equal


def configuration(profile="single"):
    text = strict.configuration(profile)
    audio = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=16,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=24,
        d_model=16,
        output_dim=24,
        downsample_hidden_size=4,
        max_source_positions=13,
        n_window=50,
        n_window_infer=200,
        conv_chunksize=2,
    )
    cfg = Qwen3OmniMoeThinkerConfig(
        text_config=text,
        vision_config=vision_config(),
        audio_config=audio,
        image_token_id=10,
        vision_start_token_id=11,
        vision_end_token_id=12,
        audio_token_id=13,
        audio_start_token_id=14,
        audio_end_token_id=15,
        video_token_id=16,
    )
    cfg._attn_implementation = "eager"
    cfg.text_config._experts_implementation = "eager"
    return thinker_config(cfg.to_dict())


def reference(config, dtype=torch.float32):
    config = copy.deepcopy(config)
    config._attn_implementation = "eager"
    config.text_config._experts_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(172)
        model = HFThinker(config).to(dtype=dtype)
        for p in model.parameters():
            torch.nn.init.uniform_(p, -0.1, 0.1)
    return model


def source_state(config, prefix, layout, dtype):
    # Independent actual HF modules, not the converter's inverse.
    state = {prefix + k: v.clone() for k, v in reference(config, dtype).state_dict().items()}
    if layout == "fused":
        return state
    output = {}
    for key, value in state.items():
        if key.endswith(".experts.gate_up_proj"):
            root = key.removesuffix("gate_up_proj")
            size = config.text_config.moe_intermediate_size
            for i in range(config.text_config.num_experts):
                output[f"{root}{i}.gate_proj.weight"] = value[i, :size].clone()
                output[f"{root}{i}.up_proj.weight"] = value[i, size:].clone()
        elif key.endswith(".experts.down_proj"):
            root = key.removesuffix("down_proj")
            for i in range(config.text_config.num_experts):
                output[f"{root}{i}.down_proj.weight"] = value[i].clone()
        else:
            output[key] = value
    return output


@pytest.mark.parametrize("prefix", ["thinker.", ""])
@pytest.mark.parametrize("layout", ["split", "fused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_complete_thinker_file_roundtrip(tmp_path, prefix, layout, dtype):
    config = configuration()
    state = source_state(config, prefix, layout, dtype)
    assert {k: tuple(v.shape) for k, v in state.items()} == parameter_shapes(config, layout, prefix)
    src, dcp, exported = tmp_path / "hf", tmp_path / "dcp", tmp_path / "exported"
    fixtures.write_hf(src, config, state, sharded=True)
    cfg = fixtures.make_cfg(
        load_path=src,
        save_path=dcp,
        omni_multimodal=True,
        omni_prefix=prefix,
        omni_expert_layout=layout,
        max_shard_size=2000,
    )
    hf2dcp(cfg, fixtures.LOGGER)
    assert json.loads((dcp / SIDECAR).read_text())["excluded_keys"] == []
    cfg.load_path, cfg.save_path = dcp, exported
    dcp2hf(cfg, fixtures.LOGGER)
    actual = fixtures.read_export(exported)
    assert_tree_equal(actual, state)
    canonical, excluded = hf_to_canonical(actual, config, prefix=prefix)
    assert not excluded
    fused = canonical_to_hf(canonical, config, prefix="", layout="fused")
    loaded = reference(config, dtype)
    loaded.load_state_dict(fused, strict=True)
    assert_tree_equal(loaded.state_dict(), reference(config, dtype).state_dict())
    if layout == "fused":
        from_disk, info = HFThinker.from_pretrained(
            exported,
            dtype=dtype,
            local_files_only=True,
            output_loading_info=True,
            attn_implementation="eager",
        )
        assert all(
            not info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        )
        assert_tree_equal(from_disk.state_dict(), reference(config, dtype).state_dict())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("fault", ["missing-audio", "missing-vision", "extra", "shape", "dtype"])
def test_complete_schema_rejects_incomplete_or_unknown_weights(tmp_path, fault):
    config = configuration()
    state = source_state(config, "thinker.", "fused", torch.float32)
    if fault == "missing-audio":
        del state["thinker.audio_tower.conv2d1.weight"]
    elif fault == "missing-vision":
        del state["thinker.visual.patch_embed.proj.weight"]
    elif fault == "extra":
        state["thinker.visual.unknown"] = torch.ones(1)
    elif fault == "shape":
        state["thinker.audio_tower.proj2.weight"] = torch.ones(1)
    else:
        state["thinker.visual.pos_embed.weight"] = state["thinker.visual.pos_embed.weight"].int()
    src = tmp_path / "source"
    fixtures.write_hf(src, config, state)
    with pytest.raises(ConversionError):
        read_hf(src, config)
    assert not torch.cuda.is_initialized()


def test_talker_exclusion_is_explicit_and_audited(tmp_path):
    config = configuration()
    state = source_state(config, "thinker.", "fused", torch.float32)
    state["talker.test.weight"] = torch.ones(1)
    state["code2wav.test.weight"] = torch.ones(1)
    src, dst = tmp_path / "src", tmp_path / "dst"
    fixtures.write_hf(src, config, state)
    cfg = fixtures.make_cfg(load_path=src, save_path=dst, omni_multimodal=True)
    with pytest.raises(ConversionError, match="omni_thinker_only"):
        hf2dcp(cfg, fixtures.LOGGER)
    assert not dst.exists()
    cfg.omni_thinker_only = True
    hf2dcp(cfg, fixtures.LOGGER)
    assert json.loads((dst / SIDECAR).read_text())["excluded_keys"] == [
        "code2wav.test.weight",
        "talker.test.weight",
    ]


@pytest.mark.parametrize("fault", ["text-config", "text-option", "scope-confusion", "dimension"])
def test_modes_and_composite_config_fail_closed(tmp_path, fault):
    cfg = fixtures.make_cfg(
        load_path=tmp_path / "src", save_path=tmp_path / "dst", omni_multimodal=True
    )
    cfg.load_path.mkdir()
    raw = configuration().to_dict()
    if fault == "text-config":
        raw = raw["text_config"]
    elif fault == "dimension":
        del raw["vision_config"]["hidden_size"]
    elif fault == "text-option":
        cfg.omni_text_only = True
    else:
        cfg.omni_multimodal = False
        (cfg.load_path / SIDECAR).write_text("{}")
    (cfg.load_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ConversionError):
        hf2dcp(cfg, fixtures.LOGGER)
    assert not cfg.save_path.exists()


def complete_runtime(profile, rank):
    model, _, _, data, contract = strict.make_runtime(profile, rank)
    config = configuration(profile)
    hf = reference(config)
    if distributed.pp_rank == 0:
        model["module"][0].visual = hf.visual
        model["module"][0].audio_tower = hf.audio_tower
        if distributed.world_size > 1:
            for component in (hf.visual, hf.audio_tower):
                for name, parameter in list(component.named_parameters()):
                    parent, field = name.rsplit(".", 1)
                    value = distribute_tensor(
                        parameter.detach(), distributed.attn_mesh["dp"], [Shard(0)]
                    )
                    component.get_submodule(parent).register_parameter(
                        field, torch.nn.Parameter(value)
                    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    training.optimizers, training.schedulers = (optimizer,), (scheduler,)
    contract["model"] = config.to_dict()
    contract["training"]["native_multimodal"] = True
    return model, optimizer, scheduler, data, contract


def resume_worker(rank, folder, profile, mode):
    root = Path(folder)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / ('rdzv-' + mode)}",
        rank=rank,
        world_size=strict.geometry(profile)[0],
        timeout=timedelta(seconds=25),
    )
    try:
        torch.set_num_threads(1)
        model, opt, sch, data, contract = complete_runtime(profile, rank)
        if mode == "save":
            strict.advance(model, opt, sch, data)
            save_checkpoint(
                root / "original", 1, data_state=data, strict=True, resume_config=contract
            )
        else:
            load_checkpoint(
                root / "restored", 1, data_state=data, strict=True, resume_config=contract
            )
            values, _ = _runtime(model, (opt,), (sch,), data, torch.device("cpu"))
            expected = torch.load(
                root / f"original/torch-dcp/00000001/rank-{rank:05d}.pt", weights_only=True
            )
            assert_tree_equal(values.pop("rng"), expected["rng"])
            assert_tree_equal(_pack(values), expected["state"])
            strict.advance(model, opt, sch, data)
            assert data.consumed == 4
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("profile", ["single", "dp", "pp", "ep", "pp-ep"])
def test_complete_strict_hf_strict_and_fresh_process_resume(tmp_path, profile):
    world = strict.geometry(profile)[0]
    mp.spawn(resume_worker, args=(str(tmp_path), profile, "save"), nprocs=world, join=True)
    src = tmp_path / "original/torch-dcp/00000001"
    original = read_payloads(src, multimodal=True)[1]
    hf, restored = tmp_path / "hf", tmp_path / "restored/torch-dcp/00000001"
    cfg = fixtures.make_cfg(load_path=src, save_path=hf, omni_multimodal=True)
    dcp2hf(cfg, fixtures.LOGGER)
    assert (hf / RESUME_DIR).exists()
    cfg.load_path, cfg.save_path, cfg.omni_restore_resume = hf, restored, True
    hf2dcp(cfg, fixtures.LOGGER)
    assert_tree_equal(read_payloads(restored, multimodal=True)[1], original)
    mp.spawn(resume_worker, args=(str(tmp_path), profile, "load"), nprocs=world, join=True)
    # A changed media tensor cannot be coupled to the old optimizer/RNG.
    altered = tmp_path / "altered"
    shutil.copytree(hf, altered)
    index = json.loads((altered / "model.safetensors.index.json").read_text())
    key = "thinker.audio_tower.conv2d1.weight"
    shard = altered / index["weight_map"][key]
    from safetensors.torch import load_file

    state = load_file(shard)
    state[key] += 1
    save_file(state, shard)
    cfg.load_path, cfg.save_path = altered, tmp_path / "invalid"
    with pytest.raises(ConversionError, match="HF weights changed"):
        hf2dcp(cfg, fixtures.LOGGER)
    assert not cfg.save_path.exists()


def test_legacy_released_vision_position_count_uses_hf_default():
    raw = configuration().to_dict()
    del raw["vision_config"]["num_position_embeddings"]
    config = thinker_config(raw)
    assert config.vision_config.num_position_embeddings == 2304
    assert parameter_shapes(config)["visual.pos_embed.weight"] == (2304, 32)
    raw["vision_config"]["num_position_embeddings"] = 0
    with pytest.raises(ConversionError, match="num_position_embeddings"):
        thinker_config(raw)
