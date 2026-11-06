"""CPU observation/coverage guards for strict native media resume acceptance."""

from types import SimpleNamespace

import pytest
import torch

from pithtrain.modules.microbatch import Microbatch
from tests.acceptance.omni_media_resume import (
    ObjectiveObservation,
    ObservedMediaData,
    audio_spec,
    parse_args,
    require_components,
    run_gpu,
)
from tests.acceptance.omni_native_resume import equal


def microbatch():
    return Microbatch(
        model_inputs=(torch.tensor([[1, 2]]),),
        objective_inputs=(torch.tensor([[2, 3]]),),
        cu_seqlens=None,
        model_context=dict(
            feature_attention_mask=torch.ones(1, 3), input_ids=torch.tensor([[1, 2]])
        ),
        media_inputs=dict(input_features=torch.ones(1, 4, 3)),
        sample_ids=("real-audio",),
    )


@pytest.mark.parametrize("field", ["media", "layout", "labels", "ids"])
def test_resume_next_input_digest_includes_every_media_and_layout_tensor(field):
    batch = microbatch()
    source = SimpleNamespace(get_batch=lambda step, device: [batch], consumed_samples=0)
    observed = ObservedMediaData(source)
    assert observed.get_batch(0, "cpu")[0] is batch
    saved = observed.batches[0]
    if field == "media":
        batch.media_inputs["input_features"][0, 0, 0] = 2
    elif field == "layout":
        batch.model_context["feature_attention_mask"][0, 0] = 0
    elif field == "labels":
        batch.objective_inputs[0][0, 0] = -100
    else:
        batch.sample_ids = ("different-audio",)
    observed.get_batch(1, "cpu")
    with pytest.raises(AssertionError):
        equal(observed.batches[1], saved)


def test_later_pp_rank_records_context_and_absent_large_payload():
    batch = microbatch()
    batch.media_inputs = None
    observed = ObservedMediaData(SimpleNamespace(get_batch=lambda *args: [batch]))
    observed.get_batch(0, "cpu")
    record = observed.batches[0][0]
    assert record["media_inputs"] is None
    assert record["model_context"]["feature_attention_mask"]["shape"] == [1, 3]


def test_objective_observation_preserves_loss_graph_and_exact_return_values():
    calls = []
    value = torch.tensor(2.0, requires_grad=True)

    def objective(outputs, targets):
        calls.append((outputs, targets))
        loss = outputs[0].square()
        return loss, loss.detach()

    observer = ObjectiveObservation(objective)
    targets = (torch.tensor([1, 2]),)
    loss, detached = observer((value,), targets)
    assert len(calls) == len(observer.outputs) == 1
    assert loss.item() == detached.item() == 4
    loss.backward()
    assert value.grad.item() == 4


@pytest.mark.parametrize("missing", ["audio_tower.", "visual."])
def test_full_thinker_checkpoint_cannot_omit_either_media_tower(missing):
    keys = {"layers.0.weight", "audio_tower.conv.weight", "visual.patch.weight"}
    keys = {key for key in keys if not key.startswith(missing)}
    with pytest.raises(AssertionError, match="Missing checkpoint component"):
        require_components(keys, ("audio_tower.", "visual."))


def test_complete_media_checkpoint_keys_pass_and_empty_keys_fail():
    require_components(
        {"audio_tower.conv.weight", "visual.patch.weight"}, ("audio_tower.", "visual.")
    )
    with pytest.raises(AssertionError, match="Empty"):
        require_components(set(), ("audio_tower.",))


def test_missing_gpu_optin_rejects_before_cuda_access(monkeypatch):
    monkeypatch.delenv("PITHTRAIN_RUN_OMNI_GPU_TESTS", raising=False)
    assert not torch.cuda.is_initialized()
    with pytest.raises(AssertionError, match="opt-in"):
        run_gpu(SimpleNamespace(), audio_spec())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("pp,ep,world", [(1, 1, 2), (1, 2, 2), (2, 1, 4), (2, 2, 4)])
def test_resume_topologies_keep_exact_dp2_cp1(tmp_path, pp, ep, world):
    args = parse_args(
        [
            "--mode",
            "restore",
            "--dataset",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "out"),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
            "--world-size",
            str(world),
        ]
    )
    assert args.world_size // args.pp == 2
    assert args.restore_root is None


@pytest.mark.parametrize(
    "extra",
    [
        ["--world-size", "3", "--pp", "2"],
        ["--world-size", "1", "--ep", "2"],
        ["--world-size", "8", "--pp", "1"],
        ["--world-size", "2", "--negative-case", "shape"],
    ],
)
def test_invalid_resume_topologies_and_negative_modes_rejected(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "produce",
                "--dataset",
                str(tmp_path / "data"),
                "--output",
                str(tmp_path / "out"),
                *extra,
            ]
        )


def test_audio_consumer_contract_pins_source_hash_and_component():
    contract = audio_spec().contract()
    assert contract["modalities"] == ["audio"]
    assert contract["required_components"] == ["audio_tower."]
    assert contract["training_flag"] == "native_audio"
    assert all(len(digest) == 64 for digest in contract["sources"].values())


def test_gradient_diagnostics_do_not_exempt_near_zero_parameters():
    from tests.acceptance.omni_audio_reference import GradientParityError, check_gradients

    actual, expected = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    actual.grad, expected.grad = torch.tensor([2e-12]), torch.tensor([1e-12])
    with pytest.raises(GradientParityError) as failure:
        check_gradients({"any_parameter": actual}, {"any_parameter": expected}, context={"step": 0})
    row = failure.value.diagnostics["any_parameter"]
    assert row["status"] == "FAILED" and row["nse"] > 1e-2
    assert row["actual_l2"] > row["reference_l2"] > 0
    assert row["difference_max_abs"] > 0
    assert failure.value.context == {"step": 0}


def test_gradient_diagnostics_report_all_parameters_and_missing_gradients():
    from tests.acceptance.omni_audio_reference import GradientParityError, check_gradients

    a, b = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    a.grad, b.grad = torch.zeros(1), torch.zeros(1)
    assert check_gradients({"exact": a}, {"exact": b}) == {"exact": 0.0}
    missing = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(GradientParityError) as failure:
        check_gradients({"exact": a, "missing": missing}, {"exact": b, "missing": b})
    assert set(failure.value.diagnostics) == {"exact", "missing"}
    assert failure.value.diagnostics["exact"]["status"] == "PASSED"
    assert failure.value.diagnostics["missing"]["actual_missing"] is True


@pytest.mark.parametrize(
    "component,modalities,flag,prefixes",
    [
        ("audio", ("audio",), "native_audio", ("audio_tower.",)),
        ("video", ("video",), "native_video", ("visual.",)),
        (
            "multimodal",
            ("text", "image", "audio", "video"),
            "native_multimodal",
            ("audio_tower.", "visual."),
        ),
    ],
)
def test_component_cli_and_source_contract(tmp_path, component, modalities, flag, prefixes):
    from tests.acceptance.omni_media_resume import component_spec, reader_config

    args = parse_args(
        [
            "--component",
            component,
            "--mode",
            "prepare",
            "--dataset",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "out"),
            "--world-size",
            "4",
            "--pp",
            "2",
            "--ep",
            "2",
        ]
    )
    spec = component_spec(args.component)
    contract = spec.contract()
    assert spec.modalities == modalities and spec.training_flag == flag
    assert spec.required_components == prefixes
    assert contract["epoch_samples"] == 32 and contract["seed"] == 1234
    assert all(len(value) == 64 for value in contract["sources"].values())
    data, training = reader_config(args.dataset, spec)
    assert data.modalities == modalities and data.epoch_samples == 32
    assert training.seed == 1234 and training.sequence_length == 2048
    assert training.global_batch_size == 8 and training.micro_batch_size == 1
    assert callable(spec.validate_inputs)


def test_named_mixed_fixtures_retain_all_six_classes_without_iteration_loss():
    from tests.acceptance.omni_media_resume import component_spec, fixture_batches

    inputs = {name: object() for name in ("text", "image", "audio", "video", "av", "all")}
    actual = fixture_batches(component_spec("multimodal"), inputs)
    assert actual.keys() == inputs.keys()
    assert all(actual[name] is inputs[name] for name in inputs)


@pytest.mark.parametrize("missing", ("text", "image", "audio", "video", "av", "all"))
def test_mixed_prepare_cannot_omit_any_fixture_class(missing):
    from tests.acceptance.omni_media_resume import component_spec, fixture_batches

    inputs = {name: object() for name in ("text", "image", "audio", "video", "av", "all")}
    del inputs[missing]
    with pytest.raises(AssertionError, match="six input classes"):
        fixture_batches(component_spec("multimodal"), inputs)


def test_single_component_list_fixtures_keep_every_batch():
    from tests.acceptance.omni_media_resume import fixture_batches

    first, second = object(), object()
    assert fixture_batches(audio_spec(), [first, second]) == {"0": first, "1": second}


@pytest.mark.parametrize("missing", ("audio", "visual"))
def test_global_production_step_requires_both_tower_inputs(missing):
    from tests.acceptance.omni_media_resume import require_input_components

    audio = SimpleNamespace(media_inputs={"input_features": torch.ones(1)})
    visual = SimpleNamespace(media_inputs={"pixel_values_videos": torch.ones(1)})
    text = SimpleNamespace(media_inputs=None)
    with pytest.raises(AssertionError, match="No production input"):
        require_input_components(
            [text, visual if missing == "audio" else audio], ("audio_tower.", "visual.")
        )
    # Component coverage is global, not a per-DP-rank or per-microbatch constraint.
    require_input_components([text, audio, visual], ("audio_tower.", "visual."))


@pytest.mark.parametrize("field", ["audio", "image", "video", "av-layout"])
def test_complete_thinker_next_input_observation_detects_any_tower_or_av_change(field):
    batch = microbatch()
    batch.media_inputs.update(pixel_values=torch.ones(1, 4), pixel_values_videos=torch.ones(2, 4))
    batch.model_context["video_second_per_grid"] = torch.tensor([0.5])
    observed = ObservedMediaData(SimpleNamespace(get_batch=lambda *args: [batch]))
    observed.get_batch(0, "cpu")
    if field == "av-layout":
        batch.model_context["video_second_per_grid"][0] = 0.75
    else:
        key = {"audio": "input_features", "image": "pixel_values", "video": "pixel_values_videos"}[
            field
        ]
        batch.media_inputs[key].fill_(2)
    observed.get_batch(1, "cpu")
    with pytest.raises(AssertionError):
        equal(observed.batches[0], observed.batches[1])


def test_component_groups_include_both_towers_every_layer_and_deepstack():
    from tests.acceptance.omni_media_resume import component_spec

    config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=4),
        audio_config=SimpleNamespace(encoder_layers=2),
        vision_config=SimpleNamespace(depth=3, deepstack_visual_indexes=[0, 2]),
    )
    groups = component_spec("multimodal").training_groups(config)
    for name in (
        "audio_tower.conv2d1.weight",
        "audio_tower.conv2d2.weight",
        "audio_tower.conv2d3.weight",
        "audio_tower.conv_out.weight",
        "audio_tower.ln_post.weight",
        "audio_tower.proj1.weight",
        "audio_tower.proj2.weight",
        "visual.patch_embed.weight",
        "visual.pos_embed.weight",
        "visual.merger.weight",
        "visual.merger_list.0.weight",
        "visual.merger_list.1.weight",
        *(f"audio_tower.layers.{i}.weight" for i in range(2)),
        *(f"visual.blocks.{i}.weight" for i in range(3)),
        *(f"model.layers.{i}.weight" for i in range(4)),
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ):
        assert len([group for group in groups if name.startswith(group)]) == 1, name
