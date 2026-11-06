"""CPU routing/coverage contracts for the unexecuted audio GPU acceptance driver."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder
from tests.acceptance.omni_audio_training import hf_name, parse_args, training_groups
from tests.acceptance.omni_image_evidence import fixed_image_batches, validate_target_layout
from tests.test_qwen3_omni_audio import thinker_config

ROOT = Path(__file__).resolve().parents[1]


def test_production_dataset_routes_explicit_native_audio_flag():
    tree = ast.parse((ROOT / "pithtrain/tasks/pretrain_lm.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "setup_dataset"
    )
    calls = []

    class Data:
        def __init__(self, *args, **kwargs):
            pass

        def validate_model(self, cls, config):
            calls.append((cls, config))

    def choose(config, *, native_vision, native_audio, native_video, native_multimodal):
        assert config == "composite"
        assert native_audio is True and native_vision is False
        assert native_video is False and native_multimodal is False
        return "native-audio-class"

    distributed = SimpleNamespace(
        dp_rank=0, dp_size=1, cp_rank=0, cp_size=1, pp_rank=0, pp_size=1, pp_group=None
    )
    namespace = dict(
        PretrainLMCfg=object,
        PretrainData=object,
        distributed=distributed,
        OmniPretrainData=Data,
        model_class_for_config=choose,
        AutoConfig=SimpleNamespace(from_pretrained=lambda _: "composite"),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "dataset", "exec"), namespace)
    config = SimpleNamespace(
        data=SimpleNamespace(validate=lambda: None, format="prepared_bundle"),
        training=SimpleNamespace(
            model="path",
            native_audio=True,
            native_vision=False,
            native_video=False,
            native_multimodal=False,
        ),
    )
    assert isinstance(namespace["setup_dataset"](config), Data)
    assert calls == [("native-audio-class", "composite")]


def test_module_update_groups_cover_every_audio_parameter_exactly_once():
    config = thinker_config()
    groups = training_groups(config)
    native = Qwen3OmniMoeAudioEncoder(config.audio_config)
    for name, _ in native.named_parameters():
        assert sum(("audio_tower." + name).startswith(prefix) for prefix in groups) == 1, name
    assert len({prefix for prefix in groups if prefix.startswith("audio_tower.")}) == 9
    assert hf_name("audio_tower.conv2d1.weight") == "audio_tower.conv2d1.weight"
    assert hf_name("visual.merger.weight") == "visual.merger.weight"
    assert hf_name("layers.0.mlp.gate.weight") == "model.layers.0.mlp.gate.weight"


@pytest.mark.parametrize("pp,ep", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("scenario", ["nonuniform", "rank-zero"])
def test_requested_cp1_dp2_topologies_preserve_nonuniform_and_rank_zero_targets(
    tmp_path, pp, ep, scenario
):
    args = parse_args(
        [
            "--mode",
            "preflight",
            "--dataset",
            str(tmp_path / "input"),
            "--output",
            str(tmp_path / "output"),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
            "--dp",
            "2",
            "--scenario",
            scenario,
        ]
    )
    batches = [
        SimpleNamespace(labels=torch.arange(247).reshape(1, -1)),
        SimpleNamespace(labels=torch.arange(273).reshape(1, -1)),
    ]
    fixed = fixed_image_batches(batches, args.dp, args.scenario)
    layout = validate_target_layout(fixed, args.dp, args.scenario)
    assert layout["global_target_count"] == sum(layout["rank_target_totals"])
    assert (layout["rank_target_totals"][0] == 0) == (scenario == "rank-zero")


@pytest.mark.parametrize(
    "extra", [["--dp", "1", "--ep", "2"], ["--dp", "1", "--scenario", "rank-zero"]]
)
def test_invalid_topology_or_zero_target_case_is_rejected(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "preflight",
                "--dataset",
                str(tmp_path / "input"),
                "--output",
                str(tmp_path / "output"),
                *extra,
            ]
        )
