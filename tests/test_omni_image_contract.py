"""CPU structure and real-reader collective failure gates; no fake model math."""

import ast
import json
import math
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.contexts import training as training_context
from pithtrain.models.qwen3_omni_image_inputs import validate_image_inputs
from pithtrain.models.qwen3_omni_stage5_policy import validate_omni_stage5_aggregation
from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.model_config import resolve_text_model_config
from pithtrain.modules.training_data import PreparedBundleData
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
                            "media_context",
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
    ranks = dict(
        dp_rank=rank % 2, dp_size=2, pp_rank=rank // 2, pp_size=2, pp_group=pp_groups[rank % 2]
    )
    data = PreparedBundleData(cfg, training, **ranks)
    corrupt = [rank == 1]

    def validator(ids, inputs, config):
        if corrupt[0]:
            inputs["pixel_values"][0, 0] = float("nan")
        return validate_image_inputs(ids, inputs, config)

    from pithtrain.tasks.prepare_omni_data import processor_for

    _, config = processor_for(data.recipe, True)
    consumer = SimpleNamespace(input_modalities=("image",), validate_media_batch=validator)
    data.validate_model(consumer, config)
    try:
        try:
            data.get_batch(0, "cpu")
        except RuntimeError as error:
            message = str(error)
            assert "Global media batch rejected" in message and "pixel_values" in message
        else:
            raise AssertionError("Bad image accepted on a rank")
        # Nothing durable advanced: once the input is valid, retrying the rejected step
        # on every rank returns exactly what a fresh reader returns for that step.
        corrupt[0] = False
        retry = data.get_batch(0, "cpu")
        fresh = PreparedBundleData(cfg, training, **ranks)
        fresh.validate_model(consumer, config)
        fixtures.compare_batches(retry, fresh.get_batch(0, "cpu"))
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(dict(status="PASSED", error=message))
        )
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


def test_staged_reader_observation_passes_batches_and_records_steps():
    from tests.acceptance.omni_image_staged import ObservedSteps

    batches = [SimpleNamespace(sample_ids=("image-a",)), SimpleNamespace(sample_ids=("image-b",))]
    source = SimpleNamespace(get_batch=lambda step, device: batches, data_fingerprint="image-id")
    data = ObservedSteps(source)
    assert data.get_batch(3, "cpu") is batches
    assert data.data_fingerprint == "image-id"
    assert data.steps == [dict(step=3, sample_ids=[("image-a",), ("image-b",)])]


def test_optimizer_keeps_vision_conv_and_projection_in_adamw():
    from pithtrain.modules.optimizer import is_muon_param as choose

    assert not choose("module.0.visual.patch_embed.proj.weight", torch.empty(8, 3, 2, 2, 2))
    assert not choose("module.0.visual.blocks.0.attn.qkv.weight", torch.empty(24, 8))
    assert choose("module.0.layers.0.self_attn.q_proj.weight", torch.empty(24, 8))
    assert not choose("module.0.layers.0.mlp.gate.weight", torch.empty(4, 8))


def test_visual_initialization_after_empty_allocation():
    import math

    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("init_weights", "weight_generator")
    ]
    ns = {
        "torch": torch,
        "nn": torch.nn,
        "math": math,
        "np": np,
        "distributed": SimpleNamespace(ep_rank=0),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "initialization", "exec"), ns)
    model = torch.nn.Module()
    model.visual = torch.nn.Sequential(
        torch.nn.Conv3d(3, 8, 2), torch.nn.LayerNorm(8), torch.nn.Linear(8, 16)
    )
    model.to_empty(device="cpu")
    # init_weights takes a rank's V-chunks, an nn.Sequential, and names parameters without
    # the chunk index.
    ns["init_weights"](torch.nn.Sequential(model), 4)
    for name, value in model.named_parameters():
        assert torch.isfinite(value).all(), name
        if value.ndim == 1:
            torch.testing.assert_close(
                value, torch.zeros_like(value) if name.endswith("bias") else torch.ones_like(value)
            )
        else:
            assert value.std() > 0


def constructor_namespace():
    """Execute real constructors with real tensors, without GPU-only imports.

    Only class constructors/attributes are extracted; no decoder forward/kernel
    implementation is substituted or claimed executed by these allocation tests.
    Native vision and BF16Linear retain their actual parameter constructors.
    """
    from pithtrain.models.qwen3_omni_vision import Qwen3OmniMoeVisionEncoder

    ns = dict(
        torch=torch,
        nn=torch.nn,
        partial=partial,
        math=math,
        np=np,
        validate_omni_stage5_aggregation=validate_omni_stage5_aggregation,
        distributed=SimpleNamespace(pp_size=2, pp_rank=0, ep_size=1, ep_rank=0, cp_size=1),
        training=SimpleNamespace(
            fp8=False,
            deterministic_token_layout=False,
            omni_stage5_aggregation=training_context.omni_stage5_aggregation,
        ),
        Qwen3OmniMoeVisionEncoder=Qwen3OmniMoeVisionEncoder,
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    partition = [
        n
        for n in ast.parse((ROOT / "pithtrain/pipeline/dualpipev.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in ("layer_sizes", "layer_partition")
    ]
    init = [
        n
        for n in ast.parse((ROOT / "pithtrain/modules/training.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in ("init_weights", "weight_generator")
    ]
    nodes = [future, *partition, *init]
    for filename, selected in (
        ("pithtrain/operators/linear.py", {"BF16Linear"}),
        ("pithtrain/models/qwen3_omni_moe.py", None),
        ("pithtrain/models/qwen3_omni_image.py", None),
    ):
        for cls in ast.parse((ROOT / filename).read_text()).body:
            if not isinstance(cls, ast.ClassDef) or (selected and cls.name not in selected):
                continue
            cls.body = [
                n
                for n in cls.body
                if (
                    isinstance(n, (ast.Assign, ast.AnnAssign))
                    and not (
                        isinstance(n, ast.Assign)
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "forward"
                    )
                )
                or (isinstance(n, ast.FunctionDef) and n.name == "__init__")
            ] or [ast.Pass()]
            nodes.append(cls)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), __file__, "exec"
        ),
        ns,
    )
    ns["training"].Linear = ns["BF16Linear"]
    return ns


def allocation_config(released=False):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerConfig,
    )

    if released:
        raw = json.loads((ROOT / "tests/configs/qwen3_omni_full/released-config.json").read_text())
        return Qwen3OmniMoeThinkerConfig(**raw["thinker_config"])
    from tests.test_qwen3_omni_vision import vision_config

    return Qwen3OmniMoeThinkerConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=40,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            moe_intermediate_size=12,
            num_experts=4,
            num_experts_per_tok=2,
            mlp_only_layers=[0],
            attention_dropout=0.0,
            tie_word_embeddings=False,
            rope_parameters=dict(
                rope_type="default", rope_theta=1000000, mrope_section=[2, 1, 1], interleaved=True
            ),
        ),
        vision_config=vision_config(),
    )


def legacy_image_constructor(ns, config, phase):
    """Independent reproduction of the former two-stack allocation order."""
    model = ns["Qwen3OmniMoeThinkerTextModel"](config.text_config, phase)
    count = len(config.vision_config.deepstack_visual_indexes)
    model.layers = torch.nn.ModuleDict(
        {
            key: ns["Qwen3OmniMoeThinkerImageDecoderLayer"](config.text_config, int(key), count)
            for key in model.layers
        }
    )
    model.visual = (
        ns["Qwen3OmniMoeVisionEncoder"](config.vision_config) if model.stage_index == 0 else None
    )
    return model


@pytest.mark.parametrize("released", [False, True])
@pytest.mark.parametrize("rank,phase", [(0, 0), (1, 0), (0, 1), (1, 1), (0, -1)])
def test_image_constructs_each_selected_decoder_once_and_preserves_state(released, rank, phase):
    ns = constructor_namespace()
    ns["distributed"].pp_rank = rank
    ns["distributed"].ep_size = 4 if released else 1
    config = allocation_config(released)
    base = ns["Qwen3OmniMoeThinkerTextDecoderLayer"]
    original = base.__init__
    allocations = []

    def counted(self, cfg, layer_id):
        original(self, cfg, layer_id)
        allocations.append((type(self).__name__, layer_id))

    base.__init__ = counted
    with torch.device("meta" if released else "cpu"), torch.random.fork_rng(devices=[]):
        model = ns["Qwen3OmniMoeThinkerImageModel"](config, phase)
        expected = list(
            ns["layer_partition"](
                config.text_config.num_hidden_layers, model.stage_count, model.stage_index
            )
        )
        assert allocations == [("Qwen3OmniMoeThinkerImageDecoderLayer", i) for i in expected]
        assert list(model.layers) == [str(i) for i in expected]
        assert all(layer.idx == int(key) for key, layer in model.layers.items())
        assert all(
            layer.deepstack_count == len(config.vision_config.deepstack_visual_indexes)
            for layer in model.layers.values()
        )
        assert model.hidden_size == config.text_config.hidden_size * (1 + model.deepstack_count)
        assert (model.visual is not None) == (model.stage_index == 0)
        allocations.clear()
        old = legacy_image_constructor(ns, config, phase)
        assert len(allocations) == 2 * len(expected), "Regression fixture must reproduce R2"
        assert {name: tuple(value.shape) for name, value in model.state_dict().items()} == {
            name: tuple(value.shape) for name, value in old.state_dict().items()
        }
        model.load_state_dict(old.state_dict(), strict=True)
    assert not torch.cuda.is_initialized()


def test_default_text_factory_preserves_constructor_rng_and_parameters():
    ns = constructor_namespace()
    config = allocation_config().text_config
    cls = ns["Qwen3OmniMoeThinkerTextModel"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(814)
        default = cls(config, -1)
        ns["init_weights"](torch.nn.Sequential(default), config.num_hidden_layers)
        default_rng = torch.get_rng_state()
        torch.manual_seed(814)
        explicit = cls(config, -1, decoder_layer_factory=ns["Qwen3OmniMoeThinkerTextDecoderLayer"])
        ns["init_weights"](torch.nn.Sequential(explicit), config.num_hidden_layers)
        assert torch.equal(torch.get_rng_state(), default_rng)
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)


def test_image_constructor_rng_change_no_longer_reaches_initialized_state():
    """
    The legacy image constructor allocated each decoder twice, so it consumes more of the global
    RNG. init_weights draws every parameter from a stream keyed by the seed and the parameter's
    name, so both constructions initialize to identical states without any reseeding.
    """
    ns = constructor_namespace()
    config = allocation_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(814)
        old = legacy_image_constructor(ns, config, -1)
        ns["init_weights"](torch.nn.Sequential(old), config.text_config.num_hidden_layers)
        old_rng = torch.get_rng_state()
        torch.manual_seed(814)
        current = ns["Qwen3OmniMoeThinkerImageModel"](config, -1)
        ns["init_weights"](torch.nn.Sequential(current), config.text_config.num_hidden_layers)
        assert not torch.equal(torch.get_rng_state(), old_rng)
        for name, value in current.state_dict().items():
            torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)
