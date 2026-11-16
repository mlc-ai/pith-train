"""CPU contracts of the released default-recipe driver (validation row 23).

The GPU run belongs to the driver. These tests run the real recipe script's main()
against stand-ins for the CUDA-only training modules, and check the pure validation
rules the GPU run applies.
"""

import ast
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.acceptance import omni_released_continuation as continuation
from tests.acceptance import omni_released_default_recipe as recipe
from tests.acceptance import omni_released_speech as speech
from tests.acceptance import omni_released_update_support as support

ROOT = Path(__file__).resolve().parents[1]


def stand_ins(monkeypatch):
    """CUDA-free modules with the attributes the recipe sets; launch must be captured."""
    import pithtrain.modules
    import pithtrain.tasks

    training = types.ModuleType("pithtrain.modules.training")

    def make_adamw_optimizer(cfg):
        raise AssertionError("not called")

    def make_constant_scheduler(cfg):
        raise AssertionError("not called")

    training.make_adamw_optimizer = make_adamw_optimizer
    training.make_constant_scheduler = make_constant_scheduler
    pretrain = types.ModuleType("pithtrain.tasks.pretrain_lm")

    class PretrainLMCfg:
        def __init__(self):
            self.initialize_from, self.initialize_step = None, 0
            self.data = SimpleNamespace(
                dataset=None,
                format="token_bin",
                modalities=("text",),
                sampling_weights=None,
                epoch_samples=None,
                num_workers=2,
                validate=lambda: None,
            )
            self.distributed = SimpleNamespace(
                pipeline_parallel_size=1, context_parallel_size=1, expert_parallel_size=1
            )
            # TrainingCfg defaults for every field the recipe leaves alone.
            self.training = SimpleNamespace(deterministic=False, strict_resume=False)

    def launch(cfg):
        raise AssertionError("The recipe launch must be captured, not run")

    pretrain.PretrainLMCfg, pretrain.launch = PretrainLMCfg, launch
    monkeypatch.setitem(sys.modules, "pithtrain.modules.training", training)
    monkeypatch.setitem(sys.modules, "pithtrain.tasks.pretrain_lm", pretrain)
    monkeypatch.setattr(pithtrain.modules, "training", training, raising=False)
    monkeypatch.setattr(pithtrain.tasks, "pretrain_lm", pretrain, raising=False)
    return pretrain


@pytest.mark.parametrize("phase", ["produce", "restore"])
def test_recipe_main_is_captured_with_documented_values(monkeypatch, phase):
    pretrain = stand_ins(monkeypatch)
    initialize = Path("/work/initialization") if phase == "produce" else None
    paths = dict(
        dataset=Path("/data/bundle"), model=Path("/work/model"), checkpoint=Path("/scratch/ckpt")
    )
    argv = recipe.recipe_argv(**paths, epoch_samples=48, initialize_from=initialize)
    saved = list(sys.argv)
    cfg = recipe.recipe_cfg(argv)
    assert sys.argv == saved and pretrain.launch.__name__ == "launch"
    settings = recipe.recipe_settings(cfg)
    expected = recipe.check_recipe(settings, **paths, epoch_samples=48, initialize_from=initialize)
    assert settings == expected
    assert settings["strict_resume"] is True and settings["deterministic_field"] is False
    assert settings["max_steps"] == 4 and recipe.RUN_STEPS == 3 and recipe.SAVE_STEP == 2
    for key, value in (("strict_resume", False), ("lr", 2e-4), ("global_batch_size", 8)):
        with pytest.raises(AssertionError, match="differs"):
            recipe.check_recipe(
                dict(settings, **{key: value}),
                **paths,
                epoch_samples=48,
                initialize_from=initialize,
            )


def test_strict_resume_enables_the_deterministic_policies():
    """The premise of row 23's bitwise criterion, read from the production source."""
    tree = ast.parse((ROOT / "pithtrain/modules/training.py").read_text())
    fields = {}
    setup = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TrainingCfg":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields[item.target.id] = item.value
        if isinstance(node, ast.FunctionDef) and node.name == "setup_training":
            setup = node
    assert ast.literal_eval(fields["deterministic"]) is False
    assert ast.literal_eval(fields["strict_resume"]) is False
    calls = [
        n
        for n in ast.walk(setup)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "configure_deterministic_training"
    ]
    assert len(calls) == 1
    names = {
        ast.literal_eval(n.args[1])
        for n in ast.walk(calls[0])
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "getattr"
    }
    assert names == {"deterministic", "strict_resume"}
    script = (ROOT / "examples/pretrain_lm/qwen3-omni-native-multimodal/script.py").read_text()
    assert "t.strict_resume = args.save_interval, args.checkpoint, True" in script


def test_estimate_matches_the_released_thinker_header():
    plan = recipe.estimate()
    assert plan["parameters"] == speech.PARAMETERS["thinker"] == 31_719_205_488
    assert plan["parameter_keys"] == 531 + 351 + 525
    assert plan["checkpoint_bytes"] == 380_630_465_856
    assert plan["microbatches_per_attention_dp_rank"] == 4 >= 2 * recipe.PP
    single = support.estimate("image")
    assert plan["pp1_sharded_elements"] == single["pp1_sharded_elements"]
    assert plan["pp0_sharded_elements"] > single["pp0_sharded_elements"]
    json.dumps(recipe.plan(), allow_nan=False)


def test_media_presence_and_required_groups():
    tensor = torch.zeros(1)
    batches = [
        SimpleNamespace(media_inputs=None),
        SimpleNamespace(media_inputs={"pixel_values_videos": tensor}),
    ]
    assert recipe.media_presence(batches) == dict(visual=True, audio=False)
    audio = [SimpleNamespace(media_inputs={"input_features": tensor})]
    assert recipe.media_presence(audio) == dict(visual=False, audio=True)
    groups = ("visual.blocks.0.", "audio_tower.conv1.", "model.layers.0.", "lm_head.")
    assert recipe.required_groups(groups, dict(visual=False, audio=True)) == groups[1:]
    assert recipe.required_groups(groups, dict(visual=True, audio=False)) == (
        "visual.blocks.0.",
        "model.layers.0.",
        "lm_head.",
    )


def scan(nonzero=4, finite=True, sha="a"):
    return dict(
        elements=4, dtype="torch.float32", finite=finite, nonzero_elements=nonzero, sha256=sha
    )


def rank_rows(rank, step, *, visual_gradient=0):
    def row(nonzero):
        return dict(
            before=scan(sha="a"),
            after=scan(sha="b"),
            changed=True,
            gradient=scan(nonzero=nonzero),
            optimizer_state=dict(step=float(step), exp_avg=scan(), exp_avg_sq=scan()),
        )

    return dict(
        rank=rank,
        parameters={
            f"model.layers.0.w{rank}": row(4),
            f"visual.blocks.0.w{rank}": row(visual_gradient),
        },
    )


def test_update_rows_apply_every_parameter_rule_for_later_updates():
    groups = ("model.layers.0.", "visual.blocks.0.")
    rows = [rank_rows(0, 3), rank_rows(1, 3)]
    summary, errors = recipe.update_rows(rows, groups, ("model.layers.0.",), 3)
    assert errors == [] and summary["visual.blocks.0."]["gradient_nonzero_elements"] == 0
    assert recipe.update_rows(rows, groups, groups, 3)[1] == [
        "Missing/zero gradient/no actual update component visual.blocks.0."
    ]
    assert any(
        "Adam step is not 2" in e for e in recipe.update_rows(rows, groups, groups[:1], 2)[1]
    )
    broken = [rank_rows(0, 3)]
    broken[0]["parameters"]["model.layers.0.w0"]["gradient"] = None
    assert any("missing/nonfinite" in e for e in recipe.update_rows(broken, groups, (), 3)[1])
    stray = [
        dict(rank=0, parameters={"unknown.w": rank_rows(0, 3)["parameters"]["model.layers.0.w0"]})
    ]
    assert any("ambiguous" in e for e in recipe.update_rows(stray, groups, (), 3)[1])


def test_prepared_identity_binds_receipt_and_unchanged_production_code(tmp_path, monkeypatch):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (tmp_path / "expected").mkdir()
    (tmp_path / "expected/index.json").write_text("{}")
    receipt = dict(
        status="PASS_RELEASED_THINKER_PREPARE_ONLY",
        revision=support.REVISION,
        config_sha256=support.CONFIG_SHA,
        source_commit=head,
        expected_index_sha256=continuation.file_sha(tmp_path / "expected/index.json"),
    )
    continuation.write_json(tmp_path / "prepared.json", receipt)
    args = SimpleNamespace(
        prepared=tmp_path,
        prepared_sha256=continuation.file_sha(tmp_path / "prepared.json"),
        prepared_commit=head,
    )
    assert recipe.prepared_identity(args)["source_commit"] == head
    with pytest.raises(AssertionError, match="checksum"):
        recipe.prepared_identity(SimpleNamespace(**dict(vars(args), prepared_sha256="0" * 64)))
    with pytest.raises(AssertionError, match="identity"):
        recipe.prepared_identity(SimpleNamespace(**dict(vars(args), prepared_commit="1" * 40)))
    monkeypatch.setattr(
        recipe.subprocess, "check_output", lambda *a, **k: "pithtrain/modules/training.py\n"
    )
    with pytest.raises(AssertionError, match="Production code changed"):
        recipe.prepared_identity(args)


def test_cli_requires_complete_absolute_inputs(capsys):
    recipe.main(["plan"])
    assert json.loads(capsys.readouterr().out)["row"] == 23
    base = [
        "produce",
        "--expected-commit",
        "a" * 40,
        "--prepared",
        "/p",
        "--prepared-sha256",
        "b" * 64,
        "--prepared-commit",
        "c" * 40,
        "--dataset",
        "/d",
        "--checkpoint",
        "/scratch/c",
        "--tuning-dir",
        "/scratch/t",
        "--output",
        "/o",
    ]
    for extra in (
        ["--epoch-samples", "40"],
        ["--epoch-samples", "32"],
        ["--epoch-samples", "48", "--producer", "/x"],
    ):
        with pytest.raises(SystemExit):
            recipe.main(base + extra)
    with pytest.raises(SystemExit):
        recipe.main(["restore", *base[1:], "--epoch-samples", "48"])
    relative = [item if item != "/o" else "relative" for item in base]
    with pytest.raises(SystemExit):
        recipe.main(relative + ["--epoch-samples", "48"])
