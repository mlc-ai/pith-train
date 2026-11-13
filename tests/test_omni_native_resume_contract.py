"""CPU-only failure checks for native acceptance gates; no loader stand-ins."""

from types import SimpleNamespace

import pytest
import torch

from tests.acceptance.omni_native_resume import (
    compare_named,
    cpu_tree,
    difference,
    equal,
    model_config,
    parse_args,
    setup_device,
)


def test_native_gpu_requires_optin_before_cuda_access(monkeypatch):
    monkeypatch.delenv("PITHTRAIN_RUN_OMNI_GPU_TESTS", raising=False)
    assert not torch.cuda.is_initialized()
    with pytest.raises(AssertionError, match="opt-in"):
        setup_device(SimpleNamespace())
    assert not torch.cuda.is_initialized()


def test_distributed_is_explicitly_unimplemented(monkeypatch):
    monkeypatch.setenv("PITHTRAIN_RUN_OMNI_GPU_TESTS", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(NotImplementedError, match="not implemented"):
        setup_device(SimpleNamespace())
    assert not torch.cuda.is_initialized()


def test_real_token_vocabulary_config_is_retained():
    config = model_config()
    assert config.model_type == "qwen3_omni_moe_text"
    assert config.vocab_size == 152064
    assert config.num_hidden_layers == 4 and config.num_experts == 4


@pytest.mark.parametrize("actual", [{}, {"wrong": torch.ones(2)}, {"weight": None}])
def test_parameter_coverage_never_silently_drops(actual):
    with pytest.raises(AssertionError):
        compare_named(actual, {"weight": torch.ones(2)}, 1e-2, "gradient")


@pytest.mark.parametrize(
    "actual",
    [
        torch.ones(3),
        torch.ones(2, dtype=torch.float64),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
    ],
)
def test_shape_dtype_and_nonfinite_rejected(actual):
    with pytest.raises(AssertionError):
        difference(actual, torch.ones(2))


def test_compare_actual_update_delta_not_large_weight():
    # Absolute weights could conceal the total loss of an optimizer update.
    with pytest.raises(AssertionError, match="optimizer-update"):
        compare_named(
            {"weight": torch.zeros(2)}, {"weight": torch.full((2,), 1e-6)}, 1e-2, "optimizer-update"
        )
    with pytest.raises(AssertionError):
        compare_named(
            {"weight": torch.full((2,), 1e-6)}, {"weight": torch.zeros(2)}, 1e-2, "optimizer-update"
        )


def test_boundary_comparison_is_exact_and_snapshot_detached():
    source = {"model": {"weight": torch.ones(2)}, "rng": (None, [1, 2])}
    saved = cpu_tree(source)
    equal(saved, source)
    source["model"]["weight"][0] += 1e-6
    with pytest.raises(AssertionError):
        equal(source, saved)
    with pytest.raises(AssertionError):
        equal({"model": saved["model"]}, saved)


def test_cli_uses_explicit_real_dataset_and_mode(tmp_path):
    args = parse_args(
        [
            "--mode",
            "restore",
            "--dataset",
            str(tmp_path / "real"),
            "--output",
            str(tmp_path / "evidence"),
        ]
    )
    assert args.mode == "restore" and args.dataset == tmp_path / "real"


def test_empty_coverage_cannot_pass():
    with pytest.raises(AssertionError, match="no parameters"):
        compare_named({}, {}, 1e-2, "gradient")


def test_invalid_existing_output_is_not_written(monkeypatch, tmp_path):
    from tests.acceptance import omni_native_resume as runner

    args = SimpleNamespace(mode="preflight", output=tmp_path, dataset=tmp_path / "missing")
    monkeypatch.setattr(runner, "parse_args", lambda: args)

    def rejected(_):
        raise ValueError("invalid existing output")

    monkeypatch.setattr(runner, "preflight", rejected)
    with pytest.raises(ValueError, match="invalid existing output"):
        runner.main()
    assert list(tmp_path.iterdir()) == []
