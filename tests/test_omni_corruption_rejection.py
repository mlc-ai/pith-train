"""CPU checks for fault-specific rejection evidence; no native GPU acceptance."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.modules import distributed_checkpoint as checkpoint
from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
from tests.acceptance.omni_distributed_resume import (
    negative_case,
    verify_corruption_rejection,
)
from tests.acceptance.omni_native_resume import equal
from tests.test_distributed_strict_checkpoint import advance, build, initialize_mesh, snapshot

FAULTS = {
    "missing-rank": (
        "read and verify files",
        "DistributedCheckpointError",
        "Missing/unexpected rank files",
    ),
    "missing-cuda-rng": (
        "validate every rank before mutation",
        "StrictResumeError",
        "app.resume.rng: missing=['cuda'], unexpected=[]",
    ),
    "shape": (
        "validate every rank before mutation",
        "DistributedCheckpointError",
        "state.model.weight: local shard shape/dtype mismatch",
    ),
    "topology": (
        "validate every rank before mutation",
        "DistributedCheckpointError",
        "Rank topology, configuration, membership or model layout differs",
    ),
    "optimizer-order": (
        "validate every rank before mutation",
        "DistributedCheckpointError",
        "Optimizer group configuration/membership differs",
    ),
}


def rejection(fault, *, world_size=2, phase=None, error_type=None, reason=None, ranks=None):
    expected_phase, expected_type, expected_reason = FAULTS[fault]
    ranks = (
        ranks
        if ranks is not None
        else (list(range(world_size)) if fault == "missing-rank" else [world_size - 1])
    )
    return checkpoint.DistributedCheckpointError(
        (phase or expected_phase)
        + ": "
        + "; ".join(
            f"rank {rank}: {error_type or expected_type}: {reason or expected_reason}"
            for rank in ranks
        )
    )


def verify(fault, error):
    return verify_corruption_rejection(error, fault=fault, world_size=2, model_key="weight")


@pytest.mark.parametrize("fault", FAULTS)
def test_exact_phase_reason_and_rejecting_ranks_are_recorded(fault):
    error = rejection(fault)
    evidence = verify(fault, error)
    assert evidence["reason_verified"] is True
    assert evidence["fault"] == fault and evidence["message"] == str(error)
    assert evidence["phase"] == FAULTS[fault][0]
    assert evidence["expected"]["error_type"] == FAULTS[fault][1]
    assert evidence["expected"]["message"] == FAULTS[fault][2]
    assert [item["rank"] for item in evidence["rank_errors"]] == evidence["expected"]["ranks"]


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("error_type", ["OutOfMemoryError", "OSError", "TypeError"])
def test_unrelated_phase_wrapped_exceptions_cannot_pass_even_with_expected_message(
    fault, error_type
):
    with pytest.raises(AssertionError, match="Wrong rejection reason"):
        verify(fault, rejection(fault, error_type=error_type))


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize(
    "change,match",
    [
        (dict(phase="load arguments"), "Wrong rejection phase"),
        (dict(phase="apply strict checkpoint"), "Wrong rejection phase"),
        (dict(reason="Rank shard checksum mismatch"), "Wrong rejection reason"),
        (dict(ranks=[0, 0]), "Wrong rejecting ranks"),
        (dict(ranks=[]), "Malformed collective rejection"),
    ],
)
def test_wrong_phase_reason_rank_or_empty_evidence_fails(fault, change, match):
    with pytest.raises(AssertionError, match=match):
        verify(fault, rejection(fault, **change))


@pytest.mark.parametrize("fault", FAULTS)
def test_a_correct_reason_cannot_hide_an_extra_unrelated_rank_failure(fault):
    error = rejection(fault)
    error = checkpoint.DistributedCheckpointError(str(error) + "; rank 0: OSError: disk failed")
    with pytest.raises(AssertionError, match="Wrong rejection reason"):
        verify(fault, error)


@pytest.mark.parametrize("fault", FAULTS)
def test_no_exception_or_a_wrong_outer_type_cannot_pass(fault):
    for error in (None, RuntimeError(str(rejection(fault)))):
        with pytest.raises(AssertionError, match="Expected a collective rejection"):
            verify(fault, error)


def test_shape_reason_must_name_the_tensor_that_was_corrupted():
    with pytest.raises(AssertionError, match="Wrong rejection reason"):
        verify(
            "shape",
            rejection("shape", reason="state.model.other: local shard shape/dtype mismatch"),
        )


def protocol_worker(rank, directory):
    """Exercise the real driver + checkpoint loader with a CPU protocol fixture."""
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
        cursor = parts[-1]
        cfg = SimpleNamespace(
            training=SimpleNamespace(
                model=root / "model",
                to_json_dict=lambda: dict(global_batch_size=2),
            ),
            distributed=SimpleNamespace(hsdp_replica=1),
        )
        contract = checkpoint.resume_contract(cfg)
        advance(parts)
        advance(parts)
        save_checkpoint(
            root / "checkpoints", 2, data_state=cursor, strict=True, resume_config=contract
        )
        before = snapshot(parts)
        # Positive control uses the identical production load path and configuration.
        load_checkpoint(
            root / "checkpoints", 2, data_state=cursor, strict=True, resume_config=contract
        )
        equal(snapshot(parts), before)
        data = SimpleNamespace(checkpoint_state=cursor)
        args = SimpleNamespace(output=root, restore_root=None, world_size=2)
        results = {}
        for fault in FAULTS:
            args.negative_case = fault
            evidence = negative_case(args, cfg, data)
            assert evidence["atomicity_verified"] and evidence["reason_verified"]
            assert evidence["corruption"]["target_rank"] == 1
            results[fault] = evidence
        # Real _phase must wrap these accidental errors, and the acceptance
        # matcher must reject them instead of treating that wrapper as a PASS.
        original_hash = checkpoint._hash
        for exception in (OSError, TypeError, torch.OutOfMemoryError):
            args.output = root / exception.__name__
            args.restore_root = root / "checkpoints"
            args.negative_case = "shape"

            def broken_read(path):
                if rank == 1 and Path(path).suffix == ".pt":
                    raise exception("injected unrelated read failure")
                return original_hash(path)

            with patch.object(checkpoint, "_hash", broken_read):
                with pytest.raises(
                    checkpoint.DistributedCheckpointError,
                    match="verify corruption rejection reason.*Wrong rejection phase",
                ):
                    negative_case(args, cfg, data)
            equal(snapshot(parts), before)
        assert not torch.cuda.is_initialized()
        (root / f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    status="PASS_CPU_PROTOCOL_ONLY",
                    positive_load=True,
                    rejections=results,
                    incidental_failures_rejected=["OSError", "TypeError", "OutOfMemoryError"],
                    cuda_initialized=False,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


def test_real_two_rank_loader_all_five_faults_and_incidental_failures(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model/config.json").write_text('{"fixture":"CPU protocol only"}\n')
    mp.spawn(protocol_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        evidence = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert evidence["status"] == "PASS_CPU_PROTOCOL_ONLY"
        assert set(evidence["rejections"]) == set(FAULTS)
        assert evidence["positive_load"] is True
        assert evidence["cuda_initialized"] is False
