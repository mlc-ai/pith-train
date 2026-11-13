"""CPU observation/coverage guards for strict native media resume acceptance."""

import ast
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.modules.microbatch import Microbatch
from tests import test_omni_training as fixtures
from tests.acceptance.omni_media_resume import (
    FINGERPRINT_REJECTION,
    ObjectiveObservation,
    ObservedMediaData,
    audio_spec,
    parse_args,
    phase_marker,
    process_age,
    require_components,
    run_gpu,
    verify_fingerprint_rejection,
)
from tests.acceptance.omni_native_resume import equal

manifest = fixtures.manifest
processor_config = fixtures.processor_config
training_bundle = fixtures.training_bundle


def microbatch():
    return Microbatch(
        model_inputs=(torch.tensor([[1, 2]]),),
        objective_inputs=(torch.tensor([[2, 3]]),),
        cu_seqlens=None,
        media_context=dict(
            feature_attention_mask=torch.ones(1, 3), input_ids=torch.tensor([[1, 2]])
        ),
        media_inputs=dict(input_features=torch.ones(1, 4, 3)),
        sample_ids=("real-audio",),
    )


@pytest.mark.parametrize("field", ["media", "layout", "labels", "ids"])
def test_resume_next_input_digest_includes_every_media_and_layout_tensor(field):
    batch = microbatch()
    source = SimpleNamespace(get_batch=lambda step, device: [batch], data_fingerprint="fixture-id")
    observed = ObservedMediaData(source)
    assert observed.get_batch(0, "cpu")[0] is batch
    # The static identity reaches strict loading unchanged; steps are recorded as read.
    assert observed.data_fingerprint == "fixture-id" and observed.steps_read == [0]
    saved = observed.batches[0]
    if field == "media":
        batch.media_inputs["input_features"][0, 0, 0] = 2
    elif field == "layout":
        batch.media_context["feature_attention_mask"][0, 0] = 0
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
    assert record["media_context"]["feature_attention_mask"]["shape"] == [1, 3]


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


def test_negative_mode_offers_the_shared_corrupted_fingerprint_case(tmp_path):
    common = ["--dataset", str(tmp_path / "data"), "--output", str(tmp_path / "out")]
    args = parse_args(
        ["--mode", "negative", "--negative-case", "fingerprint", "--world-size", "2", *common]
    )
    assert args.negative_case == "fingerprint"


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
    batch.media_context["video_second_per_grid"] = torch.tensor([0.5])
    observed = ObservedMediaData(SimpleNamespace(get_batch=lambda *args: [batch]))
    observed.get_batch(0, "cpu")
    if field == "av-layout":
        batch.media_context["video_second_per_grid"][0] = 0.75
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


def test_phase_marker_prints_one_flushed_json_line_to_stderr(capsys, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    phase_marker("train-step-done", step=2, max_allocated_bytes=7)
    captured = capsys.readouterr()
    assert captured.out == ""
    (line,) = captured.err.splitlines()
    prefix, payload = line.split(" ", 1)
    record = json.loads(payload)
    assert prefix == "OMNI_MEDIA_RESUME_PHASE" and record["phase"] == "train-step-done"
    assert record["step"] == 2 and record["max_allocated_bytes"] == 7
    assert record["rank"] == "1" and record["local_rank"] == "0" and record["pid"] == os.getpid()
    assert record["utc"].endswith("+00:00")
    age = process_age()
    assert age is None or age >= 0


def test_gpu_phases_are_marked_in_execution_order():
    path = Path(__file__).parents[1] / "tests/acceptance/omni_media_resume.py"
    tree = ast.parse(path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gpu")
    calls = [
        call
        for call in ast.walk(run)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "phase_marker"
    ]
    # Source order (ast.walk is breadth-first).
    phases = [call.args[0].value for call in sorted(calls, key=lambda call: call.lineno)]
    for earlier, later in (
        ("run-gpu-start", "cuda-device-set"),
        ("cuda-device-set", "setup-distributed-start"),
        ("setup-distributed-start", "setup-distributed-done"),
        ("setup-distributed-done", "dataset-ready"),
        ("dataset-ready", "training-ready"),
        ("checkpoint-found", "expected-state-saved"),
        ("data-identity-start", "data-identity-verified"),
        ("data-identity-verified", "checkpoint-load-start"),
        ("checkpoint-load-start", "checkpoint-loaded"),
    ):
        assert phases.index(earlier) < phases.index(later)
    for phase in (
        "train-step-start",
        "train-step-done",
        "rank-report-written",
        "final-barrier-done",
    ):
        assert phase in phases


def test_gpu_restore_validates_data_identity_before_the_strict_load():
    """The GPU-only path loads by static fingerprint and never passes a reader cursor."""
    path = Path(__file__).parents[1] / "tests/acceptance/omni_media_resume.py"
    tree = ast.parse(path.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gpu")
    calls = sorted(
        (n for n in ast.walk(run) if isinstance(n, ast.Call)),
        key=lambda call: (call.lineno, call.col_offset),
    )
    names = [ast.unparse(call.func) for call in calls]
    (load,) = [call for call in calls if ast.unparse(call.func) == "load_checkpoint"]
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in load.keywords}
    assert keywords["data_fingerprint"] == "data.data_fingerprint" and keywords["strict"] == "True"
    assert "data_state" not in keywords
    # The producer reads back its committed shard; the restorer does too, then must see a
    # mismatched identity rejected before its real strict load.
    assert names.count("require_stored_fingerprint") == 2
    assert names.index("fingerprint_rejection") < names.index("load_checkpoint")
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"checkpoint_state", "commit_step", "consumed_samples"}


def test_cpu_reader_audit_replays_the_next_step_from_fresh_readers(
    training_bundle, processor_config
):
    from pithtrain.modules.training_data import PreparedBundleData
    from tests.acceptance.omni_media_resume import component_spec, cpu_reader_audit, reader_config

    spec = component_spec("multimodal")
    report = cpu_reader_audit(training_bundle, spec, processor_config[1], 2)
    assert report["status"] == "PASSED" and "consumed_samples" not in report
    assert report["restored_step"] == report["next_step"] == 2
    fresh = PreparedBundleData(*reader_config(training_bundle, spec))
    assert report["data_fingerprint"] == fresh.data_fingerprint
    # The mixed steps differ, so replaying another step cannot pass unnoticed.
    steps = [json.dumps([row["sample_ids"] for row in step]) for step in report["steps"]]
    assert len(set(steps)) == 3


@pytest.mark.parametrize(
    "fault,match",
    [("previous-step", "reader-0-next-input"), ("other-identity", "reader-0-fingerprint")],
)
def test_cpu_reader_audit_rejects_a_fresh_reader_off_the_step_or_identity(
    training_bundle, processor_config, monkeypatch, fault, match
):
    from pithtrain.modules import training_data
    from tests.acceptance.omni_media_resume import component_spec, cpu_reader_audit

    readers = []

    class Faulty(training_data.PreparedBundleData):
        """Readers built after the continuous DP pair carry the injected fault."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            readers.append(self)
            if fault == "other-identity" and len(readers) > 2:
                self.data_fingerprint = "other-data"

        def get_batch(self, step, device):
            lagging = fault == "previous-step" and readers.index(self) >= 2
            return super().get_batch(step - 1 if lagging else step, device)

    monkeypatch.setattr(training_data, "PreparedBundleData", Faulty)
    with pytest.raises(AssertionError, match=match):
        cpu_reader_audit(training_bundle, component_spec("multimodal"), processor_config[1], 2)


def rejection(phase=None, ranks=(0,), error_type=None, reason=None):
    from pithtrain.modules.distributed_checkpoint import DistributedCheckpointError

    expected = FINGERPRINT_REJECTION
    return DistributedCheckpointError(
        (phase or expected["phase"])
        + ": "
        + "; ".join(
            f"rank {rank}: {error_type or expected['error_type']}: {reason or expected['message']}"
            for rank in ranks
        )
    )


def test_fingerprint_rejection_evidence_accepts_the_exact_pre_mutation_reason():
    evidence = verify_fingerprint_rejection(rejection())
    assert evidence["reason_verified"] is True and evidence["message"] == str(rejection())
    for error in (None, RuntimeError(str(rejection()))):
        with pytest.raises(AssertionError, match="Expected a collective"):
            verify_fingerprint_rejection(error)


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(phase="load contract arguments"), "rejection phase"),
        (dict(phase="apply strict checkpoint"), "rejection phase"),
        (dict(reason="Rank shard checksum mismatch"), "rejection reason"),
        (dict(error_type="OSError"), "rejection reason"),
        (dict(ranks=(0, 1)), "rejecting ranks"),
        (dict(ranks=()), "Malformed collective rejection"),
    ],
)
def test_fingerprint_rejection_evidence_rejects_other_failures(change, match):
    with pytest.raises(AssertionError, match=match):
        verify_fingerprint_rejection(rejection(**change))


def _identity_worker(rank, directory):
    """Real strict save/load on two Gloo ranks: a CPU protocol fixture, not GPU acceptance."""
    from pithtrain.modules.distributed_checkpoint import load_checkpoint, save_checkpoint
    from tests.acceptance.omni_media_resume import fingerprint_rejection, require_stored_fingerprint
    from tests.test_distributed_strict_checkpoint import (
        FINGERPRINT,
        advance,
        build,
        initialize_mesh,
        snapshot,
    )

    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        parts = build(*initialize_mesh("fsdp-layout", rank))
        contract = dict(training=dict(global_batch_size=2), protocol="CPU-only")
        checkpoint = root / "checkpoints"
        advance(parts)
        advance(parts)
        save_checkpoint(
            checkpoint, 2, data_fingerprint=FINGERPRINT, strict=True, resume_config=contract
        )
        assert require_stored_fingerprint(checkpoint, rank, FINGERPRINT) == FINGERPRINT
        with pytest.raises(AssertionError, match="data fingerprint differs from this run"):
            require_stored_fingerprint(checkpoint, rank, "other-data")
        boundary = snapshot(parts)
        advance(parts)
        live = snapshot(parts)
        evidence = fingerprint_rejection(
            checkpoint, 2, FINGERPRINT, contract, lambda: snapshot(parts)
        )
        assert evidence["reason_verified"] and evidence["atomicity_verified"]
        equal(snapshot(parts), live, "rejected-identity-left-state-unchanged")
        # Positive control: the matching identity restores the step-2 boundary.
        load_checkpoint(
            checkpoint, 2, data_fingerprint=FINGERPRINT, strict=True, resume_config=contract
        )
        equal(snapshot(parts), boundary, "matching-identity-restored-boundary")
        assert not torch.cuda.is_initialized()
        (root / f"rank-{rank}.json").write_text(
            json.dumps(dict(status="PASS_CPU_PROTOCOL_ONLY", message=evidence["message"]))
        )
    finally:
        dist.destroy_process_group()


def test_strict_checkpoint_holds_and_enforces_the_live_data_fingerprint(tmp_path):
    mp.spawn(_identity_worker, args=(str(tmp_path),), nprocs=2, join=True)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(report["status"] == "PASS_CPU_PROTOCOL_ONLY" for report in reports)
    assert len({report["message"] for report in reports}) == 1
