"""CPU configuration/registration checks; no kernel or fake-model execution."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)

from pithtrain.modules.model_config import resolve_text_model_config


def test_official_omni_config_forms():
    text = Qwen3OmniMoeTextConfig()
    thinker = Qwen3OmniMoeThinkerConfig(text_config=text)
    omni = Qwen3OmniMoeConfig(thinker_config=thinker)
    for config in (text, thinker, omni):
        assert resolve_text_model_config(config) is text


def test_local_tiny_autoconfig(tmp_path):
    raw = json.loads((Path(__file__).parent / "configs/qwen3_omni_text/tiny.json").read_text())
    Qwen3OmniMoeTextConfig(**raw).save_pretrained(tmp_path)
    config = AutoConfig.from_pretrained(tmp_path)
    assert resolve_text_model_config(config).model_type == "qwen3_omni_moe_text"


@pytest.mark.parametrize(
    "model_type, child",
    [("qwen3_omni_moe", "thinker_config"), ("qwen3_omni_moe_thinker", "text_config")],
)
def test_invalid_nested_config_rejected(model_type, child):
    for value in (None, SimpleNamespace(model_type="qwen3_moe")):
        with pytest.raises(ValueError):
            resolve_text_model_config(SimpleNamespace(model_type=model_type, **{child: value}))


def test_other_models_unchanged():
    config = SimpleNamespace(model_type="qwen3_moe")
    assert resolve_text_model_config(config) is config


def test_registration_and_setup_use_shared_resolver():
    # Training imports CUDA operators at module import. Inspect/execute only
    # its pure selection function; sentinels represent class identities, not models.
    path = Path(__file__).parents[1] / "pithtrain/modules/training.py"
    tree = ast.parse(path.read_text())
    resolver = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "model_class_for_config"
    )
    names = (
        "DeepSeekV2Model",
        "GptOssModel",
        "Qwen3MoeModel",
        "Qwen35MoeModel",
        "Qwen3OmniMoeThinkerTextModel",
    )
    namespace = {name: object() for name in names}
    namespace["resolve_text_model_config"] = resolve_text_model_config
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(path), "exec"), namespace)
    assert (
        namespace["model_class_for_config"](SimpleNamespace(model_type="qwen3_omni_moe_text"))
        is namespace["Qwen3OmniMoeThinkerTextModel"]
    )
    thinker = Qwen3OmniMoeThinkerConfig(text_config=Qwen3OmniMoeTextConfig())
    for nested in (thinker, Qwen3OmniMoeConfig(thinker_config=thinker)):
        assert (
            namespace["model_class_for_config"](nested) is namespace["Qwen3OmniMoeThinkerTextModel"]
        )
    with pytest.raises(ValueError, match="Unsupported"):
        namespace["model_class_for_config"](SimpleNamespace(model_type="unknown"))
    setup = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "setup_model")
    calls = [
        n.func.id
        for n in ast.walk(setup)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert calls.count("resolve_text_model_config") == calls.count("model_class_for_config") == 1
