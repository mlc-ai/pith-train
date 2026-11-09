"""CPU checks for the audio key-bias structural-zero criterion and its calibration."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from tests.acceptance import omni_key_bias_calibration as calibration
from tests.acceptance import omni_structural_zero as zero
from tests.acceptance.omni_image_staged import gradient_rows, record_gradient_rows
from tests.test_qwen3_omni_audio import audio_config

KEY = "audio_tower.layers.0.self_attn.k_proj.bias"
QUERY = "audio_tower.layers.0.self_attn.q_proj.bias"
WEIGHT = "audio_tower.layers.0.self_attn.k_proj.weight"
B = 0.05  # An arbitrary bound for the rule's own logic.
WIDTH = 16


def staged_like():
    """The staged audio gate's tiny encoder (omni_audio_training) on the official framing."""
    return audio_config(
        num_mel_bins=128,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        d_model=32,
        downsample_hidden_size=4,
        output_dim=64,
        conv_chunksize=2,
        n_window=50,
        n_window_infer=800,
        max_source_positions=1500,
    )


def staged_bound():
    return zero.bound("staged", staged_like())


def microbatches(config, kind="probe"):
    lengths = ([300, 100], [201, 17])
    items = [calibration.random_features(config, value, i) for i, value in enumerate(lengths)]
    return [(*items[index % 2], kind, index) for index in range(4)]


def rows(k_actual, k_reference, s_actual=1.0, s_reference=1.0, nse=0.9):
    """A key-bias row and both candidate scale rows, each giving scales s_actual/s_reference."""
    scale = dict(status="PASS", finite=True, difference=1e-6)
    return [
        dict(
            name=KEY,
            status="PASS" if nse < 1e-2 else "FAIL",
            finite=True,
            actual_l2=k_actual,
            reference_l2=k_reference,
            difference=nse,
        ),
        dict(name=QUERY, actual_l2=s_actual, reference_l2=s_reference, shape=[WIDTH], **scale),
        dict(
            name=WEIGHT,
            actual_l2=s_actual * WIDTH**0.5,
            reference_l2=s_reference * WIDTH**0.5,
            shape=[WIDTH, WIDTH],
            **scale,
        ),
    ]


@pytest.mark.parametrize("configuration", ["official", "staged", "multimodal"])
def test_each_bound_is_ten_times_its_recorded_cpu_calibration_maximum(configuration):
    entry = zero.CALIBRATION["configurations"][configuration]
    assert zero.MARGIN == 10 and zero.CALIBRATION["margin"] == zero.MARGIN
    value = zero.bound(configuration)
    assert value == entry["bound"]
    assert value == zero.ceil_two_significant(zero.MARGIN * entry["max_bf16_ratio"])
    # Exact FP64 zeros are the key biases only.
    assert zero.CALIBRATION["fp64"][configuration]["zero_sets_match"]
    assert set(entry["audio_config"]) == set(zero.CALIBRATED_FIELDS)
    assert zero.criterion_contract(configuration)["bound"] == value
    # Every evaluated layout was calibrated, and RoPE on keys is always caught.
    rope = entry["faults"]["rope_on_keys"]
    assert set(rope) == {str(n) for n in entry["evaluated_shards"]}
    assert all(layout["detected"] for layout in rope.values())


def test_full_size_bound_catches_every_injected_fault_in_every_layer():
    faults = zero.CALIBRATION["configurations"]["official"]["faults"]
    assert all(layout["every_layer"] for layouts in faults.values() for layout in layouts.values())


def test_recorded_calibration_matches_the_calibration_script_and_audio_model():
    root = Path(calibration.__file__).resolve().parents[2]
    sources = dict(
        script_sha256=root / "tests/acceptance/omni_key_bias_calibration.py",
        audio_model_sha256=root / "pithtrain/models/qwen3_omni_audio.py",
    )
    for key, source in sources.items():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == zero.CALIBRATION[key], key


def test_bound_refuses_an_uncalibrated_encoder_or_configuration(monkeypatch):
    assert staged_bound() == zero.bound("staged")
    with pytest.raises(RuntimeError, match="recalibrate"):
        zero.bound("staged", audio_config())
    with pytest.raises(RuntimeError, match="No calibrated"):
        zero.bound("image")
    monkeypatch.setattr(zero, "KEY_BIAS_RATIO_BOUNDS", None)
    with pytest.raises(RuntimeError, match="not been calibrated"):
        zero.bound("staged")


def test_shard_layouts_outside_the_calibration_are_refused():
    parameters = {KEY: torch.zeros(32), QUERY: torch.zeros(32)}
    assert zero.require_calibrated_shards("staged", parameters, lambda value: value[:16]) == [2]
    assert zero.require_calibrated_shards("official", parameters, lambda value: value) == [1]
    with pytest.raises(RuntimeError, match="not calibrated"):
        zero.require_calibrated_shards("staged", parameters, lambda value: value[:4])
    with pytest.raises(RuntimeError, match="not calibrated"):
        zero.require_calibrated_shards("official", parameters, lambda value: value[:16])
    with pytest.raises(RuntimeError, match="not calibrated"):
        zero.require_calibrated_shards("staged", {QUERY: torch.zeros(32)}, lambda value: value)


@pytest.mark.parametrize(
    "value,expected",
    [(0.0123, 0.013), (0.01, 0.01), (0.0100001, 0.011), (1.0, 1.0), (9.99, 10.0), (123, 130)],
)
def test_ceil_two_significant(value, expected):
    result = zero.ceil_two_significant(value)
    assert result >= value and result == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_ceil_two_significant_rejects_nonpositive_or_nonfinite(value):
    with pytest.raises(ValueError):
        zero.ceil_two_significant(value)


@pytest.mark.parametrize("statistic", sorted(zero.STATISTIC_DEFINITIONS))
def test_small_residues_pass_whatever_their_nse(statistic, monkeypatch):
    monkeypatch.setattr(zero, "STATISTIC", statistic)
    key, *scales = zero.judge_key_bias_rows(rows(B / 2, B / 3, 2.0, 3.0), 1e-2, B)
    assert key["status"] == "PASS" and key["nse_status"] == "FAIL" and "reason" not in key
    assert key["criterion"] == zero.CRITERION and key["residue_ratio_bound"] == B
    assert key["residue_statistic"] == statistic
    assert key["residue_ratio"] == pytest.approx(dict(actual=B / 4, reference=B / 9))
    assert all("criterion" not in row and row["status"] == "PASS" for row in scales)


def test_the_chosen_statistic_is_the_calibrated_one():
    assert zero.STATISTIC == zero.CALIBRATION["statistic"] in zero.STATISTIC_DEFINITIONS


def test_ratio_at_the_bound_passes():
    (key, *_) = zero.judge_key_bias_rows(rows(B, B, nse=0.0), 1e-2, B)
    assert key["status"] == "PASS" and key["nse_status"] == "PASS"


@pytest.mark.parametrize("side", ["actual", "reference"])
def test_either_side_above_the_bound_fails_even_with_a_passing_nse(side):
    sizes = dict(actual=B / 2, reference=B / 2)
    sizes[side] = B * 1.01
    table = rows(sizes["actual"], sizes["reference"], nse=0.0)
    (key, *_) = zero.judge_key_bias_rows(table, 1e-2, B)
    assert key["status"] == "FAIL" and key["nse_status"] == "PASS"
    assert key["reason"].startswith("key-bias residue ratio above")


@pytest.mark.parametrize(
    "change,states",
    [
        ("no_scale_rows", ("MISSING", "MISSING")),
        ("zero_scale", ("UNDEFINED_ZERO_SCALE", "UNDEFINED_ZERO_SCALE")),
        ("nonfinite_key", ("NONFINITE", "DEFINED")),
        ("missing_reference_norm", ("DEFINED", "MISSING")),
    ],
)
def test_undefined_ratios_fail_closed(change, states):
    table = rows(B / 2, B / 2, nse=0.0)
    if change == "no_scale_rows":
        table = table[:1]
    elif change == "zero_scale":
        for row in table[1:]:
            row.update(actual_l2=0.0, reference_l2=0.0)
    elif change == "nonfinite_key":
        table[0]["actual_l2"] = math.inf
    else:
        del table[0]["reference_l2"]
    key = zero.judge_key_bias_rows(table, 1e-2, B)[0]
    assert key["status"] == "FAIL" and key["reason"].startswith("key-bias residue ratio undefined")
    assert key["residue_ratio_state"] == dict(actual=states[0], reference=states[1])


def test_uncompared_and_other_parameter_rows_are_unchanged():
    missing = dict(name=KEY, status="FAIL", reason="missing gradient", difference=None)
    reference_phase = dict(name=KEY, status="PASS", finite=True, actual_l2=1e-9)
    names = (
        "audio_tower.layers.0.self_attn.k_proj.weight",
        "audio_tower.layers.0.self_attn.v_proj.bias",
        "visual.blocks.0.attn.qkv.bias",
        "model.layers.0.self_attn.k_proj.bias",
    )
    others = [
        dict(name=name, status="FAIL", finite=True, actual_l2=1e-9, reference_l2=1e-9, difference=1)
        for name in names
    ]
    for table in ([missing, *rows(1.0, 1.0)[1:]], [reference_phase], others):
        before = json.loads(json.dumps(table))
        # No compared key-bias row, so no bound is needed either.
        assert zero.judge_key_bias_rows(table, 1e-2, None) == before


def test_a_compared_key_bias_row_requires_a_bound():
    with pytest.raises(RuntimeError, match="calibrated bound"):
        zero.judge_key_bias_rows(rows(0.0, 0.0), 1e-2, None)


def test_staged_rows_record_every_parameter_before_failing(tmp_path):
    generator = torch.Generator().manual_seed(3)
    query, weight = torch.randn(32, generator=generator), torch.randn(32, 32, generator=generator)
    by_name = {
        QUERY: dict(actual_l2=query.norm().item(), shape=[32]),
        WEIGHT: dict(actual_l2=weight.norm().item(), shape=[32, 32]),
    }
    scale = zero.residue_scale(by_name, KEY.removesuffix("k_proj.bias"), "actual")
    residue = scale * B / 4 / 32**0.5
    pairs = [
        (KEY, torch.full((32,), residue), torch.full((32,), -residue)),
        (QUERY, query, query.clone()),
        (WEIGHT, weight, weight.clone()),
        ("model.norm.weight", torch.ones(4), -torch.ones(4)),
        ("lm_head.weight", torch.tensor([1.0, math.nan]), torch.ones(2)),
        ("model.embed_tokens.weight", torch.ones(3), torch.ones(2)),
        ("audio_tower.proj1.bias", None, torch.ones(2)),
    ]
    table = gradient_rows(pairs, B)
    assert [row["name"] for row in table] == [name for name, *_ in pairs]
    key = table[0]
    # Opposite residues: NSE 2, yet both are far below the bound.
    assert key["status"] == "PASS" and key["nse_status"] == "FAIL" and key["difference"] == 2
    assert key["actual_l2"] == pytest.approx(residue * 32**0.5)
    assert key["residue_ratio"]["actual"] == pytest.approx(B / 4)
    assert table[1]["status"] == table[2]["status"] == "PASS" and table[1]["difference"] == 0
    assert [row["status"] for row in table[3:]] == ["FAIL"] * 4
    assert table[3]["difference_max_abs"] == 2 and table[3]["reference_l2"] == 2
    path = tmp_path / "gradient-rows-rank-00000.json"
    with pytest.raises(AssertionError) as failure:
        record_gradient_rows(path, 0, table, "staged")
    assert all(name in str(failure.value) for name, *_ in pairs[3:])
    saved = json.loads(path.read_text())
    assert saved["rank"] == 0 and len(saved["rows"]) == len(pairs)
    assert saved["criterion"]["configuration"] == "staged"
    assert saved["criterion"]["bound"] == zero.bound("staged")
    passing = tmp_path / "gradient-rows-rank-00001.json"
    errors = record_gradient_rows(passing, 1, table[:3])
    assert errors == {KEY: 2.0, QUERY: 0.0, WEIGHT: 0.0} and passing.exists()


@pytest.mark.parametrize("implementation", ["native", "hf"])
def test_fp64_structural_zeros_are_exactly_the_audio_key_biases(implementation):
    config = staged_like()
    model = calibration.prepared(config, implementation, "fp64", 0)
    norms = calibration.gradient_norms(model, implementation, "fp64", "cpu", microbatches(config))
    scan = calibration.structural_scan(norms)
    assert scan["equals_audio_key_bias_set"] and scan["audio_key_bias_parameters"] == 2
    assert scan["largest_zero_relative"] < 1e-12 < 1e-6 < scan["smallest_other_relative"]


@pytest.mark.parametrize("kernels", calibration.KERNELS)
@pytest.mark.parametrize("mode", ["bf16-master-fp32", "bf16"])
@pytest.mark.parametrize("implementation", ["native", "hf"])
def test_bf16_residues_stay_below_the_staged_bound(implementation, mode, kernels):
    config = staged_like()
    for seed in (0, 1):
        model = calibration.prepared(config, implementation, mode, seed)
        batch = microbatches(config)
        norms = calibration.gradient_norms(model, implementation, mode, kernels, batch)
        rows = calibration.layer_ratios(norms)
        assert calibration.case_max(rows, zero.STATISTIC) <= staged_bound()


@pytest.mark.parametrize("fault", calibration.FAULTS)
def test_shift_invariance_faults_fail_the_staged_gate_in_every_shard_layout(fault):
    """A gate fails when any evaluated ratio exceeds the bound.

    On these short random windows every injected fault is caught. On the gates' longer real
    recordings the half-key bias fault is not (CALIBRATION records it).
    """
    config = staged_like()
    model = calibration.prepared(config, "native", "bf16-master-fp32", 0)
    batch = microbatches(config)
    norms = calibration.gradient_norms(
        model, "native", "bf16-master-fp32", "emulated", batch, fault
    )
    rows = calibration.layer_ratios(norms)
    for shards in (1, 2, 4):
        assert calibration.case_max(rows, zero.STATISTIC, (shards,)) > staged_bound()


def test_one_frame_windows_have_undefined_ratios():
    config = staged_like()
    model = calibration.prepared(config, "native", "bf16", 0)
    batch = [(*calibration.random_features(config, [1], 0), "probe", 0)]
    rows = calibration.layer_ratios(
        calibration.gradient_norms(model, "native", "bf16", "cpu", batch)
    )
    assert all(row["query_l2"] == row["key_weight_l2"] == 0 for row in rows)
    assert all(row[name]["1"] == [None] for row in rows for name in calibration.STATISTICS)
    assert calibration.case_max(rows, "query_bias") is None


def test_emulated_kernels_round_like_bf16_kernels_and_restore_the_originals():
    generator = torch.Generator().manual_seed(1)
    a, b = (torch.randn(8, 16, generator=generator).bfloat16() for _ in range(2))
    originals = (torch.nn.functional.linear, torch.nn.functional.conv2d, torch.matmul)
    with calibration.emulated_bf16_kernels(True):
        emulated = torch.matmul(a, b.T)
        mixed = torch.matmul(a.float(), b.T.float())
    assert emulated.dtype == torch.bfloat16 and mixed.dtype == torch.float32
    torch.testing.assert_close(emulated, (a.float() @ b.T.float()).bfloat16(), rtol=0, atol=0)
    assert (torch.nn.functional.linear, torch.nn.functional.conv2d, torch.matmul) == originals
