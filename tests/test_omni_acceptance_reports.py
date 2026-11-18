"""The regression oracle must compare the same loss across DP/PP layouts."""

import copy
import json

import pytest

from tests.compare_omni_acceptance import load_run, main


def write_run(path, *, legacy, cp=1):
    path.mkdir()
    # Rank0: one target at loss2; rank1: three targets at loss6. The global
    # mean is5, not the unweighted mean4 or the original DP0-only log2.
    for rank in range(4):
        item = dict(
            result="PASSED",
            start=0,
            steps=1,
            legacy=legacy,
            mesh=dict(
                rank=rank,
                pp_rank=rank // 2,
                dp_rank=rank % 2 // cp,
                cp_rank=rank % cp,
                pp_size=2,
                dp_size=2 // cp,
                cp_size=cp,
            ),
            loss_statistics=(
                [dict(loss_sum=2 if rank == 0 else 18, target_count=1 if rank == 0 else 3)]
                if rank < 2
                else []
            ),
            rows=(
                [
                    {
                        "train/step": 0,
                        "train/cross-entropy-loss": (2 if cp == 1 else 4) if legacy else 5,
                    }
                ]
                if rank == 0
                else []
            ),
        )
        (path / f"rank{rank}.json").write_text(json.dumps(item))
    return path


@pytest.mark.parametrize("cp", [1, 2])
def test_global_oracle_preserves_and_validates_different_raw_log_scopes(tmp_path, cp):
    base = write_run(tmp_path / "base", legacy=True, cp=cp)
    feature = write_run(tmp_path / "feature", legacy=False, cp=cp)
    base_row, feature_row = load_run(base)["rows"][0], load_run(feature)["rows"][0]
    assert base_row["train/cross-entropy-loss"] == feature_row["train/cross-entropy-loss"] == 5
    assert base_row["logged/cross-entropy-loss"] == (2 if cp == 1 else 4)
    assert feature_row["logged/cross-entropy-loss"] == 5


@pytest.mark.parametrize(
    "failure", ["wrong_global_log", "nan", "pp_copy", "missing_rank", "zero_targets"]
)
def test_invalid_loss_evidence_is_rejected(tmp_path, failure):
    path = write_run(tmp_path / "run", legacy=False)
    rank0 = path / "rank0.json"
    item = json.loads(rank0.read_text())
    if failure == "wrong_global_log":
        item["rows"][0]["train/cross-entropy-loss"] = 2
    elif failure == "nan":
        item["loss_statistics"][0]["loss_sum"] = float("nan")
    elif failure == "pp_copy":
        file = path / "rank2.json"
        duplicate = json.loads(file.read_text())
        duplicate["loss_statistics"] = item["loss_statistics"]
        file.write_text(json.dumps(duplicate))
    elif failure == "missing_rank":
        (path / "rank1.json").unlink()
    else:
        item["loss_statistics"][0]["target_count"] = 0
        file = path / "rank1.json"
        other = json.loads(file.read_text())
        other["loss_statistics"][0]["target_count"] = 0
        file.write_text(json.dumps(other))
    rank0.write_text(json.dumps(item))
    with pytest.raises(AssertionError):
        load_run(path)


@pytest.mark.parametrize("regression", [False, True])
def test_full_gate_accepts_matching_scope_and_rejects_real_regression(
    tmp_path, monkeypatch, regression
):
    paths = []
    for name, legacy, drift in (
        ("base0", True, 0),
        ("base1", True, 0.01),
        ("feature", False, 0.2 if regression else 0.005),
    ):
        path = write_run(tmp_path / name, legacy=legacy)
        paths.append(path)
        for rank in range(4):
            file = path / f"rank{rank}.json"
            item = json.loads(file.read_text())
            item.update(
                steps=2,
                initial_state={"rank": rank},
                audited_view_hooks=2,
                batches=["same-input-0", "same-input-1"],
            )
            if rank < 2:
                second = copy.deepcopy(item["loss_statistics"][0])
                second["loss_sum"] += drift * second["target_count"]
                item["loss_statistics"].append(second)
            if rank == 0:
                first = item["rows"][0]
                first.update({"train/load-balance-loss": 1, "train/gradient-norm": 2})
                second = {key: value + drift for key, value in first.items()}
                second["train/step"] = 1
                item["rows"].append(second)
            file.write_text(json.dumps(item))
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv", ["compare", *(str(path) for path in paths), "--output", str(output)]
    )
    if regression:
        with pytest.raises(AssertionError, match="Numerical deltas"):
            main()
    else:
        main()
    result = json.loads(output.read_text())
    assert result["result"] == ("INVESTIGATE" if regression else "PASSED")
    assert result["raw_logged_ce"]["base0"][0] == 2
    assert result["raw_logged_ce"]["feature"][0] == 5
