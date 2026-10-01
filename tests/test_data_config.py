"""CPU config, entrypoint and explicit-modality compatibility checks.

The actual task config/setup ASTs run without GPU imports. Example launch/model
builders are mocked; loaders, processors and the data configuration remain real.
"""

import ast
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from pithtrain.config import SlottedDefault
from pithtrain.modules.data_config import DataCfg, resolve_bundle_modalities
from pithtrain.modules.training_data import DensePretrainData, OmniPretrainData
from tests import test_training_data as text_fixtures

text_corpus = text_fixtures.text_corpus
text_bundle = text_fixtures.text_bundle
ROOT = Path(__file__).parents[1]


def task_config(**overrides):
    path = ROOT / "pithtrain/tasks/pretrain_lm.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [
        node
        for node in tree.body
        if getattr(node, "name", None) in {"PretrainLMCfg", "setup_dataset"}
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = dict(
        dataclass=dataclass,
        field=field,
        SlottedDefault=SlottedDefault,
        DataCfg=DataCfg,
        DistributedCfg=SimpleNamespace,
        TrainingCfg=SimpleNamespace,
        LoggingCfg=SimpleNamespace,
        DensePretrainData=DensePretrainData,
        OmniPretrainData=OmniPretrainData,
        distributed=SimpleNamespace(
            dp_rank=0, dp_size=1, cp_rank=0, cp_size=1, pp_rank=0, pp_size=1, pp_group=None
        ),
    )
    namespace.update(overrides)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def test_one_data_config_owns_source_and_independent_defaults(tmp_path):
    task = task_config()
    left, right = task.PretrainLMCfg(), task.PretrainLMCfg()
    assert left.data is not right.data
    assert not hasattr(left, "dataset") and not hasattr(left, "omni_data")
    for removed in ("dataset", "omni_data"):
        with pytest.raises(AttributeError, match=removed):
            setattr(left, removed, tmp_path)
    left.data.dataset = tmp_path
    left.data.validate()
    assert left.data.to_json_dict() == dict(
        dataset=str(tmp_path),
        format="token_bin",
        modalities=("text",),
        sampling_weights=None,
        epoch_samples=None,
        num_workers=0,
    )
    with pytest.raises(ValueError, match="data.dataset"):
        right.data.validate()


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"format": "guess"}, "data.format"),
        ({"format": []}, "data.format"),
        ({"dataset": None}, "data.dataset"),
        ({"dataset": ""}, "data.dataset"),
        ({"modalities": ()}, "modalities"),
        ({"modalities": "text"}, "modalities"),
        ({"modalities": ("text", "text")}, "modalities"),
        ({"modalities": ("lidar",)}, "modalities"),
        ({"modalities": (["text"],)}, "modalities"),
        ({"modalities": ("image",)}, "token_bin"),
        ({"num_workers": -1}, "num_workers"),
        ({"num_workers": True}, "num_workers"),
        ({"epoch_samples": 8}, "Dense text"),
        ({"sampling_weights": {"text": 1}}, "Dense text"),
        (
            {
                "format": "prepared_bundle",
                "modalities": ("text", "image"),
                "sampling_weights": {"text": 1},
            },
            "weights",
        ),
        (
            {
                "format": "prepared_bundle",
                "modalities": ("image",),
                "sampling_weights": {"image": float("nan")},
            },
            "weights",
        ),
        (
            {
                "format": "prepared_bundle",
                "modalities": ("image",),
                "sampling_weights": {"image": "1"},
            },
            "weights",
        ),
        (
            {"format": "prepared_bundle", "modalities": ("image",), "epoch_samples": 0},
            "epoch_samples",
        ),
    ],
)
def test_config_rejects_ambiguous_or_unsupported_settings(tmp_path, updates, match):
    cfg = DataCfg()
    cfg.dataset = tmp_path
    for name, value in updates.items():
        setattr(cfg, name, value)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_task_routes_storage_format_with_identical_text_batches(text_bundle, monkeypatch):
    root, training = text_bundle
    # Model-capability checking is exercised separately with real processor config.
    checks = []
    monkeypatch.setattr(OmniPretrainData, "validate_model", lambda self, *args: checks.append(args))
    task = task_config(
        AutoConfig=SimpleNamespace(from_pretrained=lambda path: path),
        model_class_for_config=lambda cfg: "Model",
    )
    cfg = task.PretrainLMCfg()
    cfg.training = training
    cfg.training.model = "model-config"
    cfg.data.dataset = root / "tokens/train"
    dense = task.setup_dataset(cfg)
    assert isinstance(dense, DensePretrainData) and not checks
    cfg.data.dataset, cfg.data.format = root, "prepared_bundle"
    prepared = task.setup_dataset(cfg)
    assert isinstance(prepared, OmniPretrainData) and checks == [("Model", "model-config")]
    torch.testing.assert_close(
        text_fixtures.tensors(prepared.get_batch(0, "cpu")),
        text_fixtures.tensors(dense.get_batch(0, "cpu")),
        rtol=0,
        atol=0,
    )


def test_explicit_subset_and_ambiguous_preset():
    recipe = {"stages": {"text": ["text"], "image": ["text", "image"]}}
    assert resolve_bundle_modalities(recipe, ("image", "text")) == (["text", "image"], "image")
    assert resolve_bundle_modalities(recipe, ("text", "audio")) == (["audio", "text"], None)
    recipe["stages"]["alias"] = ["text", "image"]
    with pytest.raises(ValueError, match="multiple recipe stages"):
        resolve_bundle_modalities(recipe, ("text", "image"))


def example_modules(monkeypatch):
    task = task_config()
    calls = []
    module = ModuleType("pithtrain.tasks.pretrain_lm")
    module.PretrainLMCfg, module.launch = task.PretrainLMCfg, calls.append
    monkeypatch.setitem(sys.modules, module.__name__, module)
    training = ModuleType("pithtrain.modules.training")
    for name in (
        "make_adamw_optimizer",
        "make_muon_optimizer",
        "make_constant_scheduler",
        "make_wsd_scheduler",
    ):
        setattr(training, name, lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, training.__name__, training)
    return calls


@pytest.mark.parametrize(
    "model", ["deepseek-v2-lite", "gpt-oss-20b", "gpt-oss-120b", "qwen3-30b-a3b", "qwen3.5-35b-a3b"]
)
def test_existing_examples_use_shared_text_config(monkeypatch, model):
    example_modules(monkeypatch)
    result = runpy.run_path(str(ROOT / f"examples/pretrain_lm/{model}/script.py"))
    cfg = result["cfg"]
    cfg.data.validate()
    assert cfg.data.format == "token_bin" and cfg.data.modalities == ("text",)


@pytest.mark.parametrize(
    "arguments,format,kinds",
    [
        ([], "prepared_bundle", ("text",)),
        (
            [
                "--modalities",
                "text",
                "image",
                "--sampling-weights",
                '{"text":1,"image":2}',
                "--num-workers",
                "1",
            ],
            "prepared_bundle",
            ("text", "image"),
        ),
        (["--data-format", "token_bin"], "token_bin", ("text",)),
    ],
)
def test_qwen_omni_example_cli_builds_one_data_config(
    monkeypatch, tmp_path, arguments, format, kinds
):
    calls = example_modules(monkeypatch)
    path = ROOT / "examples/pretrain_lm/qwen3-omni/script.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--dataset",
            str(tmp_path),
            "--model",
            "model-config",
            "--checkpoint",
            str(tmp_path / "ckpt"),
            *arguments,
        ],
    )
    runpy.run_path(str(path), run_name="__main__")
    assert len(calls) == 1
    cfg = calls[0]
    assert cfg.data.dataset == tmp_path and cfg.data.format == format
    assert cfg.data.modalities == kinds
    cfg.data.validate()


def test_task_passes_pipeline_ownership_to_prepared_provider(tmp_path):
    group = object()
    calls = []

    class Provider:
        def __init__(self, cfg, training, **ranks):
            calls.append(ranks)

        def validate_model(self, model, config):
            assert model == "TestModel" and config == "native-config"

    task = task_config(
        OmniPretrainData=Provider,
        distributed=SimpleNamespace(
            dp_rank=1, dp_size=2, cp_rank=0, cp_size=1, pp_rank=1, pp_size=2, pp_group=group
        ),
        AutoConfig=SimpleNamespace(from_pretrained=lambda _: "native-config"),
        model_class_for_config=lambda _: "TestModel",
    )
    cfg = task.PretrainLMCfg()
    cfg.data.dataset, cfg.data.format = tmp_path, "prepared_bundle"
    cfg.training.model = "model-path"
    assert isinstance(task.setup_dataset(cfg), Provider)
    assert calls == [
        dict(dp_rank=1, dp_size=2, cp_rank=0, cp_size=1, pp_rank=1, pp_size=2, pp_group=group)
    ]
