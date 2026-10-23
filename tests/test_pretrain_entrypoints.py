"""Exercise migrated entrypoints on CPU, including the historical base config.

Scripts and their argument/config construction execute as written. GPU imports,
model builders and launch are stand-ins.
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
