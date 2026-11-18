"""Structural-zero criterion for audio self-attention key-bias gradients.

Test-only helpers; no production code changes.

Qwen3-Omni audio self-attention (native and HF) adds k_proj.bias to every key a query
attends to and applies no RoPE, key norm or bias-dependent mask, so softmax shift
invariance makes the exact k_proj.bias gradient identically zero. A computed value is
rounding residue whose size and sign depend on the arithmetic path, so the NSE between
two residues compares noise (about 1 when they are independent).

judge_key_bias_rows therefore judges each compared audio key-bias row by its size on each
side separately: the residue ratio must be defined and at most the bound of the gate's
configuration. The ratio's scale (STATISTIC) is the size of the key-gradient rows whose
rounded sum the residue is, measured by the same layer's k_proj.weight gradient norm over
sqrt(input width); the q_proj.bias gradient norm was the other candidate. The NSE verdict
stays as a diagnostic (nse_status). A row that failed for another reason (missing or
wrongly shaped gradient, nonfinite values) keeps its failure.

Each bound comes from omni_key_bias_calibration, a CPU experiment independent of every GPU
result, and was fixed before any GPU run under this criterion: MARGIN times the largest
BF16 ratio over that configuration's calibration grid, as its gate evaluates it (whole
vectors, or FSDP dim-0 shards), rounded up to two significant figures. The same
calibration chose STATISTIC as the candidate with the larger weakest fault-detection
margin. bound() refuses an encoder whose dimensions differ from the calibrated ones, and
require_calibrated_shards an uncalibrated shard layout. The scope is the random-
initialization gates whose biases are zero when gradients are compared: "official" for
omni_full_media, "staged" for the audio run of omni_image_staged and "multimodal" for
omni_multimodal_training. CALIBRATION records each configuration's detection of three
injected invariance faults, including the ones the tiny staged bounds cannot detect. The
criterion was adopted on 2026-11-08 by maintainer decision after the f1 failures; results
judged by NSE are not re-scored.
"""

import math
import re

AUDIO_KEY_BIAS = re.compile(r"^(audio_tower\.layers\.\d+\.self_attn\.)k_proj\.bias$")
CRITERION = "structural_zero_key_bias"
MARGIN = 10
NSE_ROLE = "diagnostic: NSE between rounding residues of a structurally zero gradient"
# Encoder fields a calibration is valid for (attention geometry, windows, depth).
CALIBRATED_FIELDS = (
    "num_mel_bins",
    "encoder_layers",
    "encoder_attention_heads",
    "encoder_ffn_dim",
    "d_model",
    "downsample_hidden_size",
    "conv_chunksize",
    "n_window",
    "n_window_infer",
)
STATISTIC_DEFINITIONS = dict(
    key_weight=(
        "L2(key-bias gradient) * sqrt(input width) / L2(same-layer k_proj.weight gradient), "
        "per side"
    ),
    query_bias="L2(key-bias gradient) / L2(same-layer q_proj.bias gradient), per side",
)
STATISTIC = "key_weight"
KEY_BIAS_RATIO_BOUNDS = {"official": 0.025, "staged": 0.17, "multimodal": 0.078}
# Generated from the calibration report; see omni_key_bias_calibration.
CALIBRATION = {
    "date": "2026-11-09",
    "script": "tests/acceptance/omni_key_bias_calibration.py",
    "script_sha256": "7bf2387623e6ba98e83787339782da8f5db3c10bcdfcd9beb27f3c947b49b784",
    "script_note": "executed as "
    "5cfd13a088fc2915c856ede21d8f9a1f3ce4ed174738b55eb384df509ba5b508; ruff "
    "format was applied during the run, AST identical",
    "audio_model_sha256": "442a73ab363d485a3eba67616f2b543f324012857304b857dc0dd019130d3b62",
    "base_commit": "f1cc8f3189afc1584beb2b93624dce7e09180977",
    "report": "resume-20261026/claude-key-bias-calibration-20261109/calibration-full.json",
    "report_sha256": "aea92c0942e67a8e4665748e78eb683144639c4d1f7cd2d3b88109ba1efec14b",
    "torch": "2.13.0+cpu",
    "transformers": "5.17.0",
    "device": "cpu (x86_64), 10 threads",
    "official_config_sha256": "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d",
    "fixture": "librispeech-374-180298-0000",
    "recordings": ["librispeech-374-180298-0000", "librispeech-374-180298-0001"],
    "margin": 10,
    "statistic": "key_weight",
    "statistic_rule": "the candidate whose weakest gate-level fault detection margin is largest",
    "candidates": {
        "key_weight": {
            "weakest_detection_margin": 0.2879,
            "bounds": {"official": 0.025, "staged": 0.17, "multimodal": 0.078},
        },
        "query_bias": {
            "weakest_detection_margin": 0.1621,
            "bounds": {"official": 0.024, "staged": 0.39, "multimodal": 0.39},
        },
    },
    "fp64": {
        "official": {
            "zero_sets_match": True,
            "largest_zero_relative": 6.88e-19,
            "smallest_other_relative": 5.39e-05,
            "max_ratio": 8.68e-15,
        },
        "staged": {
            "zero_sets_match": True,
            "largest_zero_relative": 3.21e-21,
            "smallest_other_relative": 2.06e-06,
            "max_ratio": 1.76e-15,
        },
        "multimodal": {
            "zero_sets_match": True,
            "largest_zero_relative": 2.06e-20,
            "smallest_other_relative": 8.37e-06,
            "max_ratio": 2.8e-15,
        },
    },
    "configurations": {
        "official": {
            "gate": "omni_full_media",
            "audio_config": {
                "num_mel_bins": 128,
                "encoder_layers": 32,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "d_model": 1280,
                "downsample_hidden_size": 480,
                "conv_chunksize": 500,
                "n_window": 50,
                "n_window_infer": 800,
            },
            "evaluated_shards": [1],
            "cases": 37,
            "ratios": 1184,
            "undefined_ratios": 0,
            "max_bf16_ratio": 0.002468506420079282,
            "median_bf16_ratio": 0.001433,
            "max_by_shards": {"1": 0.002469},
            "max_by_kernels": {"emulated": 0.002469, "cpu": 0.001975},
            "bound": 0.025,
            "faults": {
                "rope_on_keys": {
                    "1": {
                        "max_ratio": 0.6024,
                        "min_ratio": 0.2616,
                        "detected": True,
                        "every_layer": True,
                    }
                },
                "key_layer_norm": {
                    "1": {
                        "max_ratio": 0.2744,
                        "min_ratio": 0.1697,
                        "detected": True,
                        "every_layer": True,
                    }
                },
                "bias_on_half_the_keys": {
                    "1": {
                        "max_ratio": 0.1502,
                        "min_ratio": 0.07103,
                        "detected": True,
                        "every_layer": True,
                    }
                },
            },
            "all_faults_detected": True,
        },
        "staged": {
            "gate": "omni_image_staged (audio)",
            "audio_config": {
                "num_mel_bins": 128,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 64,
                "d_model": 32,
                "downsample_hidden_size": 4,
                "conv_chunksize": 2,
                "n_window": 50,
                "n_window_infer": 800,
            },
            "evaluated_shards": [1, 2, 4],
            "cases": 1120,
            "ratios": 13440,
            "undefined_ratios": 2240,
            "max_bf16_ratio": 0.01627082003044721,
            "median_bf16_ratio": 0.002456,
            "max_by_shards": {"1": 0.005813, "2": 0.008427, "4": 0.01627},
            "max_by_kernels": {"emulated": 0.01627, "cpu": 0.01627},
            "bound": 0.17,
            "faults": {
                "rope_on_keys": {
                    "1": {
                        "max_ratio": 0.7613,
                        "min_ratio": 0.5872,
                        "detected": True,
                        "every_layer": True,
                    },
                    "2": {
                        "max_ratio": 0.7738,
                        "min_ratio": 0.553,
                        "detected": True,
                        "every_layer": True,
                    },
                    "4": {
                        "max_ratio": 1.021,
                        "min_ratio": 0.4549,
                        "detected": True,
                        "every_layer": True,
                    },
                },
                "key_layer_norm": {
                    "1": {
                        "max_ratio": 0.4801,
                        "min_ratio": 0.3787,
                        "detected": True,
                        "every_layer": True,
                    },
                    "2": {
                        "max_ratio": 0.4881,
                        "min_ratio": 0.3477,
                        "detected": True,
                        "every_layer": True,
                    },
                    "4": {
                        "max_ratio": 0.6036,
                        "min_ratio": 0.2415,
                        "detected": True,
                        "every_layer": True,
                    },
                },
                "bias_on_half_the_keys": {
                    "1": {
                        "max_ratio": 0.04895,
                        "min_ratio": 0.03149,
                        "detected": False,
                        "every_layer": False,
                    },
                    "2": {
                        "max_ratio": 0.05896,
                        "min_ratio": 0.01844,
                        "detected": False,
                        "every_layer": False,
                    },
                    "4": {
                        "max_ratio": 0.0841,
                        "min_ratio": 0.0182,
                        "detected": False,
                        "every_layer": False,
                    },
                },
            },
            "all_faults_detected": False,
        },
        "multimodal": {
            "gate": "omni_multimodal_training",
            "audio_config": {
                "num_mel_bins": 128,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 96,
                "d_model": 64,
                "downsample_hidden_size": 8,
                "conv_chunksize": 500,
                "n_window": 50,
                "n_window_infer": 800,
            },
            "evaluated_shards": [1, 2, 4],
            "cases": 1120,
            "ratios": 13440,
            "undefined_ratios": 2240,
            "max_bf16_ratio": 0.007703490674996177,
            "median_bf16_ratio": 0.00209,
            "max_by_shards": {"1": 0.004618, "2": 0.005545, "4": 0.007703},
            "max_by_kernels": {"emulated": 0.007703, "cpu": 0.007608},
            "bound": 0.078,
            "faults": {
                "rope_on_keys": {
                    "1": {
                        "max_ratio": 0.3552,
                        "min_ratio": 0.162,
                        "detected": True,
                        "every_layer": True,
                    },
                    "2": {
                        "max_ratio": 0.3629,
                        "min_ratio": 0.1584,
                        "detected": True,
                        "every_layer": True,
                    },
                    "4": {
                        "max_ratio": 0.5347,
                        "min_ratio": 0.123,
                        "detected": True,
                        "every_layer": True,
                    },
                },
                "key_layer_norm": {
                    "1": {
                        "max_ratio": 0.431,
                        "min_ratio": 0.2271,
                        "detected": True,
                        "every_layer": True,
                    },
                    "2": {
                        "max_ratio": 0.4906,
                        "min_ratio": 0.1828,
                        "detected": True,
                        "every_layer": True,
                    },
                    "4": {
                        "max_ratio": 0.588,
                        "min_ratio": 0.1579,
                        "detected": True,
                        "every_layer": True,
                    },
                },
                "bias_on_half_the_keys": {
                    "1": {
                        "max_ratio": 0.05876,
                        "min_ratio": 0.04513,
                        "detected": False,
                        "every_layer": False,
                    },
                    "2": {
                        "max_ratio": 0.06232,
                        "min_ratio": 0.04216,
                        "detected": False,
                        "every_layer": False,
                    },
                    "4": {
                        "max_ratio": 0.06548,
                        "min_ratio": 0.03627,
                        "detected": False,
                        "every_layer": False,
                    },
                },
            },
            "all_faults_detected": False,
        },
    },
}


def ceil_two_significant(value):
    """Smallest number with two significant decimal figures that is >= value > 0."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError("Expected a positive finite value")
    exponent = math.floor(math.log10(value)) - 1
    digits = math.ceil(value / 10**exponent)
    candidate = digits * 10**exponent
    # Guard the float round trip: the result must never fall below the input.
    return candidate if candidate >= value else (digits + 1) * 10**exponent


def bound(configuration, audio_config=None):
    """The calibrated bound of one gate configuration.

    With audio_config, also require the encoder to match the calibrated dimensions, so a
    changed gate cannot silently reuse a stale bound.
    """
    if KEY_BIAS_RATIO_BOUNDS is None or CALIBRATION is None:
        raise RuntimeError("The key-bias ratio bound has not been calibrated")
    if configuration not in KEY_BIAS_RATIO_BOUNDS:
        raise RuntimeError(f"No calibrated key-bias bound for configuration {configuration!r}")
    if audio_config is not None:
        expected = CALIBRATION["configurations"][configuration]["audio_config"]
        actual = {field: getattr(audio_config, field) for field in CALIBRATED_FIELDS}
        if actual != expected:
            raise RuntimeError(
                f"Audio encoder differs from the calibrated {configuration!r} configuration; "
                f"recalibrate (expected {expected}, got {actual})"
            )
    return KEY_BIAS_RATIO_BOUNDS[configuration]


def require_calibrated_shards(configuration, parameters, local):
    """Refuse a key-bias shard layout (full / local size) the calibration did not cover."""
    counts = {
        parameter.numel() // local(parameter).numel()
        for name, parameter in parameters.items()
        if AUDIO_KEY_BIAS.match(name)
    }
    allowed = set(CALIBRATION["configurations"][configuration]["evaluated_shards"])
    if not counts or not counts <= allowed:
        raise RuntimeError(
            f"Audio key-bias shard layout {sorted(counts)} is not calibrated for "
            f"{configuration!r} (calibrated: {sorted(allowed)})"
        )
    return sorted(counts)


def criterion_contract(configuration=None):
    """The criterion as recorded in gate reports."""
    return dict(
        criterion=CRITERION,
        parameters=AUDIO_KEY_BIAS.pattern,
        statistic=STATISTIC,
        statistic_definition=STATISTIC_DEFINITIONS[STATISTIC],
        configuration=configuration,
        bound=None if configuration is None else bound(configuration),
        bounds=KEY_BIAS_RATIO_BOUNDS,
        margin=MARGIN,
        calibration=CALIBRATION,
        nse=NSE_ROLE,
    )


def residue_ratio(norm, scale):
    """(ratio, state) of a key-bias gradient norm to its scale."""
    if norm is None or scale is None:
        return None, "MISSING"
    if not (math.isfinite(norm) and math.isfinite(scale)):
        return None, "NONFINITE"
    if scale == 0:
        return None, "UNDEFINED_ZERO_SCALE"
    return norm / scale, "DEFINED"


def residue_scale(by_name, prefix, side, statistic=None):
    """One side's scale for the key bias of the layer whose names start with prefix.

    key_weight: the k_proj.weight gradient norm over sqrt(its input width), the size of the
    key-gradient rows whose rounded sum is the residue. query_bias: the q_proj.bias gradient
    norm. Rows may hold local shards; the width is the last dimension either way.
    """
    statistic = statistic or STATISTIC
    if statistic == "query_bias":
        row = by_name.get(prefix + "q_proj.bias")
        return None if row is None else row.get(side + "_l2")
    if statistic != "key_weight":
        raise ValueError(f"Unknown statistic: {statistic}")
    row = by_name.get(prefix + "k_proj.weight")
    norm = None if row is None else row.get(side + "_l2")
    shape = None if row is None else row.get("shape")
    if norm is None or not shape:
        return None
    return norm / math.sqrt(shape[-1])


def judge_key_bias_rows(rows, threshold, bound):
    """Apply the structural-zero criterion in place to compared audio key-bias rows.

    A row is compared when its NSE ("difference") is present, which requires both sides
    to be present, of the right shape and finite; only then was its status decided by NSE
    alone, and only then is that decision replaced. Rows carry name, status, difference,
    actual_l2 and reference_l2, and the scale row also its shape. Other rows are
    unchanged. bound may be None only when no compared audio key-bias row exists.
    """
    by_name = {row["name"]: row for row in rows}
    for row in rows:
        match = AUDIO_KEY_BIAS.match(row["name"])
        if match is None or row.get("difference") is None:
            continue
        if bound is None:
            raise RuntimeError("An audio key-bias row needs a calibrated bound")
        ratios, states = {}, {}
        for side in ("actual", "reference"):
            scale = residue_scale(by_name, match.group(1), side)
            ratios[side], states[side] = residue_ratio(row.get(side + "_l2"), scale)
        defined = all(state == "DEFINED" for state in states.values())
        within = defined and all(value <= bound for value in ratios.values())
        row.update(
            criterion=CRITERION,
            residue_statistic=STATISTIC,
            residue_ratio=ratios,
            residue_ratio_state=states,
            residue_ratio_bound=bound,
            nse_status="PASS" if row["difference"] < threshold else "FAIL",
            nse_role=NSE_ROLE,
            status="PASS" if within else "FAIL",
        )
        if not defined:
            row["reason"] = f"key-bias residue ratio undefined: {states}"
        elif not within:
            row["reason"] = f"key-bias residue ratio above {bound}: {ratios}"
    return rows
