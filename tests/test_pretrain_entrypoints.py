"""Exercise migrated entrypoints on CPU, including the historical base config.

Scripts and their argument/config construction execute as written. GPU imports,
model builders and launch are stand-ins; the acceptance runner stops at setup.
These checks cannot validate GPU execution or numerical acceptance.
"""

import builtins
import re
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from pithtrain.config import SlottedDefault
from tests.test_data_config import ROOT, example_modules


@dataclass(init=False, slots=True)
class HistoricalConfig(SlottedDefault):
    """Config surface at the PR base, 8b17d15, without importing its GPU modules."""

    distributed: SimpleNamespace = field(default_factory=SimpleNamespace)
    training: SimpleNamespace = field(default_factory=SimpleNamespace)
    logging: SimpleNamespace = field(default_factory=SimpleNamespace)
    dataset: Path


def stub_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    return module


def test_qwen_example_help_needs_no_training_imports(monkeypatch, capsys):
    path = ROOT / "examples/pretrain_lm/qwen3-omni/script.py"
    real_import = builtins.__import__

    def no_training_import(name, *args, **kwargs):
        assert not name.startswith("pithtrain"), "--help must precede training imports"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_training_import)
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(path), run_name="__main__")
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--dataset" in help_text and "--modalities" in help_text
    assert "--data-format" in help_text


@pytest.mark.parametrize(
    "arguments,match",
    [
        (["--modalities", "text", "text"], "modalities"),
        (["--data-format", "token_bin", "--modalities", "image"], "token_bin"),
        (
            ["--modalities", "text", "image", "--sampling-weights", '{"text":1}'],
            "weights",
        ),
        (["--num-workers", "-1"], "num_workers"),
    ],
)
def test_qwen_example_rejects_invalid_data_before_launch(monkeypatch, tmp_path, arguments, match):
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
            str(tmp_path / "checkpoint"),
            *arguments,
        ],
    )
    with pytest.raises(ValueError, match=match):
        runpy.run_path(str(path), run_name="__main__")
    assert calls == []


@pytest.mark.parametrize("historical", [False, True], ids=["current", "pr-base"])
@pytest.mark.parametrize("kind", ["correctness", "performance"])
def test_identical_validation_template_supports_base_and_feature(monkeypatch, historical, kind):
    calls = example_modules(monkeypatch)
    if historical:
        monkeypatch.setattr(
            sys.modules["pithtrain.tasks.pretrain_lm"], "PretrainLMCfg", HistoricalConfig
        )
    stub_module(monkeypatch, "pithtrain.modules.logging", LoggingWandbCfg=SimpleNamespace)
    path = ROOT / f".agents/skills/validate-{kind}/templates/validate.py"
    source = path.read_text()
    for name, value in dict(
        tokenizer="fixture-tokenizer",
        model="qwen3-30b-a3b",
        **{
            "pipeline-parallel-size": "2",
            "expert-parallel-size": "2",
            "context-parallel-size": "1",
            "sequence-length": "128",
            "global-batch-size": "16",
            "moe-load-balance-type": "global-batch",
            "wandb-project": "fixture-validation",
        },
    ).items():
        source = source.replace(f"<{name}>", value)
    assert not re.search(r"<[\w-]+>", source), "Every template parameter must be filled"
    exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
    assert len(calls) == 1
    cfg = calls[0]
    data = cfg if historical else cfg.data
    assert data.dataset == Path("workspace/datasets/dclm-baseline/toktxt/fixture-tokenizer")
    if not historical:
        data.validate()
        assert data.format == "token_bin" and data.modalities == ("text",)
    assert cfg.distributed.pipeline_parallel_size == cfg.distributed.expert_parallel_size == 2
    assert cfg.distributed.context_parallel_size == 1
    assert cfg.training.global_batch_size == 16 and cfg.training.micro_batch_size == 1
    assert cfg.training.max_steps == (32 if kind == "correctness" else 8)
    assert cfg.training.benchmark == (kind == "performance")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "historical,mode",
    [(True, "legacy"), (False, "token_bin"), (False, "text"), (False, "media")],
    ids=["pr-base-legacy", "current-token-bin", "prepared-text", "prepared-media"],
)
def test_acceptance_runner_preserves_data_selection(monkeypatch, tmp_path, historical, mode):
    calls = example_modules(monkeypatch)
    task = sys.modules["pithtrain.tasks.pretrain_lm"]
    if historical:
        monkeypatch.setattr(task, "PretrainLMCfg", HistoricalConfig)
    import pithtrain.tasks

    monkeypatch.setattr(pithtrain.tasks, "pretrain_lm", task, raising=False)

    class ConfigCaptured(Exception):
        pass

    def capture_config(cfg):
        calls.append(cfg)
        raise ConfigCaptured

    def unexpected_setup(*args, **kwargs):
        pytest.fail("This config test must stop before GPU/model/distributed setup")

    stub_module(
        monkeypatch,
        "pithtrain.contexts",
        distributed=SimpleNamespace(),
        logging=SimpleNamespace(),
        training=SimpleNamespace(),
    )
    stub_module(monkeypatch, "pithtrain.models.qwen3_moe", Qwen3MoeModel=object)
    stub_module(
        monkeypatch,
        "pithtrain.modules.checkpoint",
        load_checkpoint=unexpected_setup,
        save_checkpoint=unexpected_setup,
    )
    stub_module(monkeypatch, "pithtrain.modules.distributed", setup_distributed=unexpected_setup)
    stub_module(monkeypatch, "pithtrain.modules.logging", setup_logging=capture_config)
    monkeypatch.setattr(
        sys.modules["pithtrain.modules.training"], "setup_training", unexpected_setup, raising=False
    )
    stub_module(monkeypatch, "pithtrain.pipeline.execution", model_forward=unexpected_setup)
    path = ROOT / "tests/test_omni_training_acceptance.py"
    arguments = (
        ["--data-format", "token_bin"]
        if mode == "token_bin"
        else ([f"--{mode}"] if mode != "text" else [])
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "--dataset",
            str(tmp_path / "bundle"),
            "--output",
            str(tmp_path / "run"),
            "--report",
            str(tmp_path / "report"),
            "--observation-protocol",
            "save_then_runtime_state_v1",
            "--pp",
            "2",
            "--ep",
            "2",
            *arguments,
        ],
    )
    with pytest.raises(ConfigCaptured):
        runpy.run_path(str(path), run_name="__main__")
    assert len(calls) == 1
    cfg = calls[0]
    data = cfg if historical else cfg.data
    expected_root = tmp_path / "bundle"
    assert data.dataset == (
        expected_root / "tokens/train" if mode in {"legacy", "token_bin"} else expected_root
    )
    if not historical:
        data.validate()
        assert data.format == ("token_bin" if mode == "token_bin" else "prepared_bundle")
        assert data.modalities == (
            ("text", "image", "audio", "video") if mode == "media" else ("text",)
        )
    assert cfg.training.max_steps == 12 and cfg.training.global_batch_size == 8
    assert cfg.training.sequence_length == 128 and cfg.training.micro_batch_size == 1
    assert cfg.distributed.pipeline_parallel_size == cfg.distributed.expert_parallel_size == 2
    assert cfg.distributed.context_parallel_size == 1
    assert not (tmp_path / "run").exists()
    assert not torch.cuda.is_initialized()
