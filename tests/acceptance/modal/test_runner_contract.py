"""Local controller contracts. These tests never import or call the Modal SDK."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "omni_modal_runner", Path(__file__).with_name("modal_runner.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def initialized(tmp_path, monkeypatch):
    def no_cloud():
        raise AssertionError("Local setup must not use Modal")

    monkeypatch.setattr(runner, "sdk", no_cloud)
    runner.initialize(tmp_path, "a" * 40, "omni-contract-test")
    ledger = runner.read(tmp_path / "budget.json")
    ledger["prices_verified_utc_date"] = runner.utc()[:10]
    runner.write(tmp_path / "budget.json", ledger)
    return tmp_path


def test_initialization_is_local_and_cannot_overwrite(initialized):
    state = runner.read(initialized / "state.json")
    ledger = runner.read(initialized / "budget.json")
    assert state["phase"] == "idle"
    assert "sandbox_id" not in state and "app_id" not in state
    assert ledger["entries"] == [] and ledger["unreserved_remaining"] == 300
    with pytest.raises(RuntimeError, match="overwrite"):
        runner.initialize(initialized, "b" * 40, "different-app")


def test_profiles_share_one_budget_and_preserve_repair_reserve(initialized):
    ledger = runner.read(initialized / "budget.json")
    state = runner.read(initialized / "state.json")
    for profile in ("single", "distributed", "folded", "full"):
        runner.reservation(ledger, state, profile)
    assert ledger["reserved_upper_bound"] == 135
    assert ledger["unreserved_remaining"] == 165
    runner.reservation(ledger, state, "distributed")
    with pytest.raises(RuntimeError, match="repair reserve"):
        runner.reservation(ledger, state, "single")


def test_failed_build_reuses_reservation_and_limits_retry(initialized):
    ledger = runner.read(initialized / "budget.json")
    state = runner.read(initialized / "state.json")
    original = runner.reservation(ledger, state, "single")
    state.update(
        phase="image_build_failed",
        profile="single",
        reservation=original["id"],
        image_build_attempts=1,
    )
    before = copy.deepcopy(ledger)
    assert runner.reservation(ledger, state, "single") is original
    assert ledger == before
    with pytest.raises(RuntimeError, match="same profile"):
        runner.reservation(ledger, state, "full")
    state["image_build_attempts"] = 3
    with pytest.raises(RuntimeError, match="three attempts"):
        runner.reservation(ledger, state, "single")


@pytest.mark.parametrize(
    "phase",
    [
        "reserved",
        "image_building",
        "image_build_uncertain",
        "sandbox_creation_started",
        "running",
    ],
)
def test_active_or_uncertain_launch_cannot_be_recreated(initialized, phase):
    ledger = runner.read(initialized / "budget.json")
    state = runner.read(initialized / "state.json") | {"phase": phase}
    with pytest.raises(RuntimeError, match="active or uncertain"):
        runner.reservation(ledger, state, "single")


@pytest.mark.parametrize(
    "change",
    [
        {"total_budget_usd": 301},
        {"unreserved_remaining": 299},
        {"reserved_upper_bound": -1},
        {"total_budget_usd": float("nan")},
    ],
)
def test_inconsistent_budget_rejected(initialized, change):
    ledger = runner.read(initialized / "budget.json") | change
    with pytest.raises(ValueError):
        runner.validate_budget(ledger)


def test_prices_must_be_reverified_and_allowance_must_fit(initialized):
    ledger = runner.read(initialized / "budget.json")
    state = runner.read(initialized / "state.json")
    ledger["prices_verified_utc_date"] = None
    with pytest.raises(RuntimeError, match="today"):
        runner.reservation(ledger, state, "single")
    ledger["prices_verified_utc_date"] = runner.utc()[:10]
    ledger["rates_usd_per_second"]["h100_each"] = 1
    with pytest.raises(RuntimeError, match="allowance"):
        runner.reservation(ledger, state, "single")


def test_lock_excludes_second_controller(initialized):
    with runner.lock(initialized):
        with pytest.raises(RuntimeError, match="Another controller"):
            with runner.lock(initialized):
                pytest.fail("Second lock must not succeed")
    assert not (initialized / "controller.lock").exists()


@pytest.mark.parametrize(
    "label,seconds", [("../escape", 60), ("x;true", 60), ("ok", 901), ("ok", 0)]
)
def test_execution_rejects_unsafe_input_without_cloud(initialized, label, seconds):
    with pytest.raises(ValueError):
        runner.execute(initialized, Path("unused.sh"), label, seconds)


def test_stop_requires_confirmed_termination_and_retains_reservation(initialized, monkeypatch):
    state, ledger = (
        runner.read(initialized / "state.json"),
        runner.read(initialized / "budget.json"),
    )
    record = runner.reservation(ledger, state, "single")
    state.update(phase="running", reservation=record["id"])
    runner.write(initialized / "state.json", state)
    runner.write(initialized / "budget.json", ledger)
    sandbox = SimpleNamespace(terminate=lambda **kwargs: None, poll=lambda: None)
    monkeypatch.setattr(runner, "get", lambda *args, **kwargs: sandbox)
    with pytest.raises(RuntimeError, match="not confirmed"):
        runner.stop(initialized)
    assert runner.read(initialized / "state.json")["phase"] == "running"
    sandbox.poll = lambda: 0
    runner.stop(initialized)
    assert runner.read(initialized / "state.json")["phase"] == "terminated"
    assert runner.read(initialized / "budget.json")["reserved_upper_bound"] == 10


def test_transfer_allowance_is_cumulative(initialized):
    state, ledger = (
        runner.read(initialized / "state.json"),
        runner.read(initialized / "budget.json"),
    )
    record = runner.reservation(ledger, state, "single")
    state.update(phase="running", reservation=record["id"])
    record["transfer_reserved_bytes"] = runner.TRANSFER_LIMIT - 100
    runner.write(initialized / "state.json", state)
    runner.write(initialized / "budget.json", ledger)
    runner.transfer(initialized, 100)
    with pytest.raises(RuntimeError, match="1 GiB"):
        runner.transfer(initialized, 1)
    with pytest.raises(RuntimeError, match="50 MiB"):
        runner.transfer(initialized, runner.FILE_LIMIT + 1)


def test_tracked_templates_contain_no_runtime_identifiers():
    for name in ("budget.template.json", "state.template.json"):
        value = json.loads(Path(__file__).with_name(name).read_text())
        assert "app_id" not in value and "sandbox_id" not in value


def fake_sdk(events, *, fail_lookup=False, fail_create=False):
    image = SimpleNamespace(object_id="im-ContractOnly")

    def resolve(app):
        events.append("lookup")
        if fail_lookup:
            raise RuntimeError("Lookup unavailable")

    image.build = resolve

    def create(**kwargs):
        events.append("create")
        assert kwargs["image"] is image
        assert kwargs["timeout"] == 1800
        if fail_create:
            raise RuntimeError("Creation result uncertain")
        return SimpleNamespace(object_id="sb-ContractOnly")

    return SimpleNamespace(
        App=SimpleNamespace(
            lookup=lambda *args, **kwargs: SimpleNamespace(app_id="ap-ContractOnly")
        ),
        Image=SimpleNamespace(from_id=lambda image_id: image),
        Sandbox=SimpleNamespace(create=create),
    )


def test_create_resolves_existing_image_before_gpu(initialized, monkeypatch):
    events = []
    monkeypatch.setattr(runner, "sdk", lambda: fake_sdk(events))
    runner.create(initialized, "single", "im-ContractOnly")
    assert events == ["lookup", "create"]
    state = runner.read(initialized / "state.json")
    assert state["phase"] == "running" and state["runtime_image"] == "im-ContractOnly"


def test_image_lookup_failure_never_requests_gpu_and_reuses_reservation(initialized, monkeypatch):
    events = []
    monkeypatch.setattr(runner, "sdk", lambda: fake_sdk(events, fail_lookup=True))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Lookup unavailable"):
            runner.create(initialized, "single", "im-ContractOnly")
    assert events == ["lookup", "lookup"]
    ledger = runner.read(initialized / "budget.json")
    assert len(ledger["entries"]) == 1 and ledger["reserved_upper_bound"] == 10
    assert runner.read(initialized / "state.json")["image_lookup_attempts"] == 2


def test_uncertain_creation_cannot_allocate_another_sandbox(initialized, monkeypatch):
    events = []
    monkeypatch.setattr(runner, "sdk", lambda: fake_sdk(events, fail_create=True))
    with pytest.raises(RuntimeError, match="uncertain"):
        runner.create(initialized, "single", "im-ContractOnly")
    with pytest.raises(RuntimeError, match="active or uncertain"):
        runner.create(initialized, "single", "im-ContractOnly")
    assert events == ["lookup", "create"]
