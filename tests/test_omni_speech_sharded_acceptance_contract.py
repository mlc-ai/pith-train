"""CPU contracts and Gloo DTensor collection; these do not establish GPU acceptance."""

import copy
import json
import os
import subprocess
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from tests.acceptance import omni_speech_sharded as gate


def samples():
    return [
        dict(
            codes=torch.arange(64).view(1, 4, 16) % 23,
            labels=torch.tensor([[-100, 1, 2, 3, 4]]),
            media_kind="text",
        )
    ]


@pytest.mark.parametrize("mode", gate.MODES)
def test_plan_keeps_four_updates_and_distinguishes_true_zero(mode):
    report = gate.plan(mode)
    assert report["updates"] == 4
    assert report["topology"] == dict(dp=2, cp=1, ep=1, pp=1)
    assert report["checkpoint"]["save_scan"] == report["checkpoint"]["save_updates"] == 2
    assert report["checkpoint"]["policy"] == "conditional_adamw_v1"
    assert report["checkpoint"]["bounded_load"] is True
    assert report["thresholds"]["gradient_nse_strict_lt"] == 1e-2
    assert report["thresholds"]["actual_update_delta_nse_strict_lt"] == 1e-2
    assert report["caps_seconds"] == dict(prepare=120, produce=600, restore=600, kill_after=10)
    assert report["matrix"]["whole_objective_global_zero"] == (
        "REQUIRED" if mode == "waveform" else "NOT_EXPRESSIBLE_EOS"
    )
    assert report["status"] == "NOT_RUN"


def test_all_names_no_small_gradient_exemption_and_exact_boundary(monkeypatch):
    left = dict(zero=torch.zeros(2), tiny=torch.full((2,), 1e-12), absent=None)
    right = dict(zero=torch.zeros(2), tiny=torch.full((2,), -1e-12), absent=torch.ones(2))
    rows = gate.compare_tensors(left, right)
    assert set(rows) == set(left)
    assert rows["zero"]["status"] == "PASS"
    assert rows["tiny"]["status"] == rows["absent"]["status"] == "FAIL"
    assert rows["tiny"]["nse"] == 2
    monkeypatch.setattr(gate.chain, "nse", lambda a, b: 1e-2)
    assert not gate.passed(gate.compare_tensors({"a": torch.ones(1)}, {"a": torch.ones(1)}))
    assert not gate.passed({})


@pytest.mark.parametrize("fault", ("shape", "dtype", "nan", "missing", "unexpected"))
def test_report_finishes_other_parameters_after_corruption(fault):
    actual, expected = (
        dict(a=torch.ones(2), z=torch.zeros(2)),
        dict(a=torch.ones(2), z=torch.zeros(2)),
    )
    if fault == "shape":
        actual["a"] = torch.ones(1)
    elif fault == "dtype":
        actual["a"] = actual["a"].double()
    elif fault == "nan":
        actual["a"][0] = float("nan")
    elif fault == "missing":
        del actual["a"]
    else:
        actual["extra"] = torch.ones(2)
    rows = gate.compare_tensors(actual, expected)
    assert rows["z"]["status"] == "PASS"
    assert not gate.passed(rows)
    assert set(rows) == set(actual) | set(expected)


@pytest.mark.parametrize("mode", gate.MODES)
def test_schedule_cursor_and_exact_next_input_resume(mode):
    source = samples()
    original = copy.deepcopy(source)
    data = gate.ScheduledData(source, "fixture", mode, 1)
    saved = None
    for scan in range(gate.scans(mode)):
        data.get_batch(scan, torch.device("cpu"))
        if gate.updates_at(mode, scan + 1) > data.completed_updates:
            data.commit_update(scan_index=scan, update_index=data.completed_updates)
        data.validate_state_dict(data.state_dict())
        if scan == 1:
            saved = data.state_dict()
    assert data.completed_updates == 4
    fresh = gate.ScheduledData(source, "fixture", mode, 1)
    fresh.load_state_dict(saved)
    fresh.get_batch(2, torch.device("cpu"))
    assert fresh.observed[2] == data.observed[2]
    gate.chain.exact(source, original)
    wrong = dict(saved, completed_updates=1)
    with pytest.raises(ValueError, match="schedule"):
        fresh.validate_state_dict(wrong)


@pytest.mark.parametrize(
    "field,value",
    (("rank", 0), ("mode", "codec_only"), ("fingerprint", "changed"), ("scanned_batches", True)),
)
def test_data_preflight_is_pure(field, value):
    data = gate.ScheduledData(samples(), "fixture", "waveform", 1)
    before = data.state_dict()
    with pytest.raises(ValueError):
        data.load_state_dict(dict(before, **{field: value}))
    assert data.state_dict() == before


def test_rankzero_globalzero_and_nonuniform_are_actual_count_masks():
    counts = []
    for scan in range(3):
        counts.append(
            [
                int(
                    gate.waveform_objective(
                        torch.ones(1, 9),
                        gate.scheduled_batches(samples(), "waveform", scan, rank, "cpu")[0],
                    ).target_count
                )
                for rank in range(2)
            ]
        )
    assert counts == [[9, 4], [9, 0], [0, 0]]
    for rank in range(2):
        batch = gate.scheduled_batches(samples(), "text_and_codec", 2, rank, "cpu")[0]
        assert (batch["labels"] == -100).all()
        output = SimpleNamespace(
            first_labels=torch.tensor([[1, 2, 3, 4, 99]]),
            residual_labels=torch.ones(1, 4, 15, dtype=torch.long),
        )
        first, _ = gate.independent_labels(output, batch)
        assert first[0, -1] == 99  # Production EOS remains a real target.


def test_empty_waveform_loss_preserves_zero_autograd_edge():
    output = torch.randn(1, 7, requires_grad=True)
    value = gate.waveform_objective(output, dict(waveform_fraction=0))
    assert value.target_count == 0
    value.loss_sum.backward()
    assert torch.equal(output.grad, torch.zeros_like(output))


class ToyReference(nn.Module):
    """An objective aggregation test fixture, never a native GPU replacement."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(16, 1, bias=False)
        nn.init.constant_(self.model.weight, 0.002)

    def decode_training_codes(self, codes):
        return self.model(codes.transpose(1, 2).float()).squeeze(-1)


def test_independent_dp_objective_matches_concatenated_target_math_and_rng():
    from pithtrain.tasks import omni_speech as task

    hf = ToyReference()
    optimizer = torch.optim.AdamW(hf.parameters(), lr=1e-4, weight_decay=0, foreach=False)
    expected = ToyReference()
    expected.load_state_dict(hf.state_dict())
    numerators, denominators = [], []
    for rank in range(2):
        batch = gate.scheduled_batches(samples(), "waveform", 0, rank, "cpu")[0]
        prediction = expected.decode_training_codes(batch["codes"].transpose(1, 2))
        valid = prediction.shape[-1] if rank == 0 else prediction.shape[-1] // 2
        target = torch.sin(torch.arange(valid).float() * 0.1) * 0.1
        numerators.append((prediction[..., :valid] - target).square().sum())
        denominators.append(valid)
    loss = sum(numerators) / sum(denominators)
    loss.backward()
    rng = task._rng_state()
    report = gate.reference_step(hf, optimizer, samples(), "waveform", 0, "cpu")
    gate.chain.exact(rng, task._rng_state())
    assert report["counts"] == [[4, 0], [2, 0]]
    torch.testing.assert_close(report["loss"], loss.detach(), rtol=0, atol=0)
    torch.testing.assert_close(report["pre"]["weight"], expected.model.weight.grad, rtol=0, atol=0)
    assert report["delta"]["weight"].abs().sum() > 0
    assert report["pre"]["weight"].norm() > report["post"]["weight"].norm()


def test_reference_globalzero_does_not_initialize_adam_slots():
    hf = ToyReference()
    optimizer = torch.optim.AdamW(hf.parameters(), lr=1e-4, weight_decay=0, foreach=False)
    report = gate.reference_step(hf, optimizer, samples(), "waveform", 2, "cpu")
    assert not report["updated"] and report["global_counts"] == [0, 0]
    assert not optimizer.state
    assert not report["delta"]["weight"].count_nonzero()


def test_evidence_exclusive_write_and_checksum(tmp_path):
    path = tmp_path / "scan.pt"
    payload = dict(state=torch.arange(4), rng=(1, [2, 3]))
    digest = gate.evidence_save(path, payload)
    gate.chain.exact(gate.evidence_load(path, digest), payload)
    with pytest.raises(FileExistsError):
        gate.evidence_save(path, {})
    with pytest.raises(gate.chain.PreconditionError):
        gate.evidence_load(path, "0" * 64)


def test_producer_cannot_be_self_different_runtime_or_failed(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    args = SimpleNamespace(
        source=dict(commit="a" * 40, tree="b" * 40),
        training_mode="waveform",
        runtime_lock=lock,
        rank=0,
        hardware="CPU-contract-no-GPU",
    )
    good = dict(
        schema=gate.SCHEMA,
        rank=0,
        topology=dict(dp=2, cp=1, ep=1, pp=1),
        runtime=dict(hardware=args.hardware),
        status="PASS",
        mode="produce",
        source=args.source,
        training_mode="waveform",
        fixture_sha256="fixture",
        pid=os.getpid() + 1,
        runtime_lock_sha256=gate.chain.sha(lock),
    )
    gate.validate_producer(good, args, "fixture")
    for key, value in (
        ("pid", os.getpid()),
        ("status", "FAIL"),
        ("runtime_lock_sha256", "bad"),
        ("source", {}),
    ):
        with pytest.raises(gate.chain.PreconditionError):
            gate.validate_producer(dict(good, **{key: value}), args, "fixture")


def _gloo_worker(rank, rendezvous, output):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        parameter = nn.Parameter(distribute_tensor(torch.arange(8).float(), mesh, (Shard(0),)))
        parameter.grad = distribute_tensor(torch.arange(8).float() / 10, mesh, (Shard(0),))
        actual = gate.collect_full(dict(parameter=parameter), gradients=True)
        torch.testing.assert_close(
            actual["parameter"], torch.arange(8).float() / 10, rtol=0, atol=0
        )
        if rank == 1:
            parameter.grad = None
        missing = gate.collect_full(dict(parameter=parameter), gradients=True)
        assert missing == dict(parameter=None)  # Both ranks agree before any tensor collective.
        with pytest.raises(RuntimeError, match="identities"):
            gate.collect_full({"parameter" if rank == 0 else "different": parameter})
        torch.save(dict(status="CPU_GLOO_COLLECTION_ONLY", rank=rank), f"{output}-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_real_two_rank_dtensor_collection_and_missing_peer_preflight(tmp_path):
    mp.spawn(
        _gloo_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "proof")),
        nprocs=2,
        join=True,
    )
    assert len(list(tmp_path.glob("proof-*.pt"))) == 2


def test_plan_cli_is_cpu_only_and_not_acceptance():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.acceptance.omni_speech_sharded",
            "plan",
            "--training-mode",
            "waveform",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
    report = json.loads(result.stdout)
    assert report["status"] == "NOT_RUN" and report["scans"] == 5


def test_single_rank_runtime_rejected_before_any_cuda_probe(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA must not be touched"))
    with pytest.raises(gate.chain.PreconditionError, match="DP2"):
        gate.runtime(SimpleNamespace())


def test_real_adam_evidence_checks_every_counter_and_fp32_moment():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    parameters = dict(model.named_parameters())
    assert not gate.passed(gate.optimizer_evidence(parameters, optimizer, 1))
    for step in range(1, 5):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        assert gate.passed(gate.optimizer_evidence(parameters, optimizer, step))
    optimizer.state[model.bias]["step"] -= 1
    rows = gate.optimizer_evidence(parameters, optimizer, 4)
    assert set(rows) == set(parameters)
    assert rows["weight"]["status"] == "PASS"
    assert rows["bias"]["status"] == "FAIL"
    optimizer.state[model.weight]["exp_avg"][0, 0] = float("nan")
    assert gate.optimizer_evidence(parameters, optimizer, 4)["weight"]["status"] == "FAIL"


def test_nonfinite_optimizer_counter_produces_json_failure_not_lost_report():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    optimizer.state[model.weight]["step"].fill_(float("nan"))
    rows = gate.optimizer_evidence(dict(model.named_parameters()), optimizer, 1)
    assert rows["weight"]["status"] == "FAIL"
    assert rows["bias"]["status"] == "PASS"
    json.dumps(rows, allow_nan=False)
