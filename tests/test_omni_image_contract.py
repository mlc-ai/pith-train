"""CPU structure and real-reader collective failure gates; no fake model math."""

import ast
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.models.qwen3_omni_image_inputs import validate_image_inputs
from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.model_config import resolve_text_model_config
from pithtrain.modules.training_data import OmniPretrainData
from tests import test_omni_training as fixtures

manifest = fixtures.manifest
processor_config = fixtures.processor_config
training_bundle = fixtures.training_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_native_vision_registration_is_explicit_and_preserves_text_default():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerConfig,
    )

    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "model_class_for_config"
    )
    image, text = object(), object()
    ns = dict(
        Qwen3OmniMoeThinkerImageModel=image,
        Qwen3OmniMoeThinkerTextModel=text,
        resolve_text_model_config=resolve_text_model_config,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "registration", "exec"), ns)
    config = Qwen3OmniMoeThinkerConfig()
    assert ns[function.name](config) is text
    assert ns[function.name](config, native_vision=True) is image
    with pytest.raises(ValueError, match="composite"):
        ns[function.name](config.text_config, native_vision=True)


def test_native_visual_modules_have_no_hf_execution_imports_or_feature_cache():
    for name in ("qwen3_omni_vision.py", "qwen3_omni_image.py", "qwen3_omni_image_inputs.py"):
        tree = ast.parse((ROOT / "pithtrain/models" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("transformers"), name
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute):
                        assert target.attr not in {
                            "image_features",
                            "deepstack_features",
                            "model_context",
                        }, name


def _bad_image_worker(rank, root, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    pp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    cfg = DataCfg()
    cfg.dataset, cfg.format, cfg.modalities = root, "prepared_bundle", ("image",)
    cfg.epoch_samples = 8
    training = SimpleNamespace(
        global_batch_size=4, micro_batch_size=1, sequence_length=1024, seed=1234
    )
    data = OmniPretrainData(
        cfg,
        training,
        dp_rank=rank % 2,
        dp_size=2,
        pp_rank=rank // 2,
        pp_size=2,
        pp_group=pp_groups[rank % 2],
    )

    def validator(ids, inputs, config):
        if rank == 1:
            inputs["pixel_values"][0, 0] = float("nan")
        return validate_image_inputs(ids, inputs, config)

    from pithtrain.tasks.prepare_omni_data import processor_for

    _, config = processor_for(data.recipe, True)
    data.validate_model(
        SimpleNamespace(input_modalities=("image",), validate_media_batch=validator), config
    )
    try:
        data.get_batch(0, "cpu")
    except RuntimeError as error:
        message = str(error)
        assert "Global media batch rejected" in message and "pixel_values" in message
        assert data.consumed_samples == 0
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(dict(status="PASSED", error=message))
        )
    else:
        raise AssertionError("Bad image accepted on a rank")
    finally:
        dist.destroy_process_group()


def test_one_bad_image_rejects_all_pp_and_dp_ranks(training_bundle, processor_config, tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    mp.spawn(
        _bad_image_worker, args=(training_bundle, tmp_path / "rdzv", output), nprocs=4, join=True
    )
    reports = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(4)]
    assert len({row["error"] for row in reports}) == 1
    assert all(row["status"] == "PASSED" for row in reports)


def test_optimizer_keeps_vision_conv_and_projection_in_adamw():
    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_muon_param"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "optimizer", "exec"), ns)
    choose = ns["is_muon_param"]
    assert not choose("module.0.visual.patch_embed.proj.weight", torch.empty(8, 3, 2, 2, 2))
    assert not choose("module.0.visual.blocks.0.attn.qkv.weight", torch.empty(24, 8))
    assert choose("module.0.layers.0.self_attn.q_proj.weight", torch.empty(24, 8))
    assert not choose("module.0.layers.0.mlp.gate.weight", torch.empty(4, 8))


def test_visual_initialization_after_empty_allocation():
    import math

    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init_weights"
    )
    ns = {"torch": torch, "nn": torch.nn, "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "initialization", "exec"), ns)
    model = torch.nn.Module()
    model.visual = torch.nn.Sequential(
        torch.nn.Conv3d(3, 8, 2), torch.nn.LayerNorm(8), torch.nn.Linear(8, 16)
    )
    model.to_empty(device="cpu")
    ns["init_weights"](model, 4)
    for name, value in model.named_parameters():
        assert torch.isfinite(value).all(), name
        if value.ndim == 1:
            torch.testing.assert_close(
                value, torch.zeros_like(value) if name.endswith("bias") else torch.ones_like(value)
            )
        else:
            assert value.std() > 0
