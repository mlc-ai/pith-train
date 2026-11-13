"""Actual SFT task updates, CPU Gloo uneven ranks, and checkpoint callback resume."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from pithtrain.contexts import distributed, training
from pithtrain.modules.microbatch import Microbatch
from pithtrain.tasks.omni_objectives import global_target_counts
from pithtrain.tasks.sft_lm import SFTLMCfg, SFTState, launch, objective, train_step


class Engine(nn.Module):
    """CPU autograd engine implementing the same sum-gradient model.step boundary."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.projection = nn.Linear(3, 5, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.calls = 0

    def forward(self, features):
        return self.projection(self.dropout(features))

    def step(self, microbatches, objective_fn):
        self.calls += 1
        outputs = []
        for batch in microbatches:
            loss, detached = objective_fn((self(*batch.model_inputs),), batch.objective_inputs)
            loss.backward()
            outputs.append(detached)
        if dist.is_initialized():
            for parameter in self.parameters():
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        return outputs


def context(model):
    training.model = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    training.optimizers, training.schedulers = (optimizer,), (scheduler,)
    return optimizer, scheduler


def cfg(updates=1, scans=1):
    config = SFTLMCfg()
    config.max_updates, config.max_batches, config.max_grad_norm = updates, scans, None
    return config


def batch(features, labels):
    return Microbatch(model_inputs=(features,), cu_seqlens=None, objective_inputs=(labels,))


class Data:
    def __init__(self, batches):
        self.batches, self.scans, self.commits = batches, [], []

    def get_batch(self, scan_index, device):
        self.scans.append(scan_index)
        return self.batches[scan_index]

    def commit_update(self, *, scan_index, update_index):
        self.commits.append((scan_index, update_index))

    def state_dict(self):
        return dict(scans=list(self.scans), commits=list(self.commits))

    def load_state_dict(self, state):
        self.scans, self.commits = list(state["scans"]), list(state["commits"])


@pytest.fixture(autouse=True)
def restore_context():
    names = ("model", "optimizers", "schedulers", "moe_load_balance_coef")
    before = {name: getattr(training, name) for name in names if hasattr(training, name)}
    yield
    for name in names:
        if name in before:
            setattr(training, name, before[name])
        elif hasattr(training, name):
            delattr(training, name)


def test_objective_consumes_aligned_labels_without_another_shift():
    logits = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, 2, 4]])
    loss, detached = objective((logits,), (labels,))
    expected = F.cross_entropy(logits[0, 1:], torch.tensor([2, 4]), reduction="sum")
    torch.testing.assert_close(loss, expected)
    assert not detached.requires_grad


def test_global_zero_skips_every_mutation_but_advances_serializable_scan():
    model = Engine().eval()
    optimizer, scheduler = context(model)
    data = Data([[batch(torch.ones(1, 2, 3, dtype=torch.float64), torch.full((1, 2), -100))]])
    params = copy.deepcopy(model.state_dict())
    rng, schedule = torch.get_rng_state().clone(), copy.deepcopy(scheduler.state_dict())
    saved = []
    state = launch(
        cfg(), data, device=torch.device("cpu"), token_group=None, checkpoint=saved.append
    )
    assert state.state_dict() == {"scanned_batches": 1, "completed_updates": 0}
    assert saved == [state.state_dict()] and data.commits == [] and data.scans == [0]
    assert model.calls == 0 and not model.training and not optimizer.state
    assert scheduler.state_dict() == schedule and torch.equal(torch.get_rng_state(), rng)
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, params[name])


def test_scan_then_update_resume_matches_uninterrupted_training(tmp_path):
    features = torch.arange(18, dtype=torch.float64).reshape(1, 6, 3) / 20
    valid = torch.tensor([[-100, 1, 2, 3, 4, 0]])
    source = [[batch(features, torch.full_like(valid, -100))], [batch(features, valid)]] * 2
    torch.manual_seed(41)
    initial = copy.deepcopy(Engine(dropout=0.25).state_dict())

    def make():
        model = Engine(dropout=0.25)
        model.load_state_dict(initial)
        optimizer, scheduler = context(model)
        return model, optimizer, scheduler, Data(source)

    full, full_opt, full_sched, full_data = make()
    torch.manual_seed(217)
    full_state = launch(cfg(2, 4), full_data, device=torch.device("cpu"), token_group=None)
    expected = copy.deepcopy(full.state_dict())
    expected_opt, expected_sched = (
        copy.deepcopy(full_opt.state_dict()),
        copy.deepcopy(full_sched.state_dict()),
    )
    expected_rng = torch.get_rng_state().clone()

    first, first_opt, first_sched, first_data = make()
    checkpoint_path = tmp_path / "sft.pt"
    checkpoints = []

    def save(cursor):
        checkpoints.append(dict(cursor))
        torch.save(
            dict(
                task=cursor,
                model=first.state_dict(),
                optimizer=first_opt.state_dict(),
                scheduler=first_sched.state_dict(),
                data=first_data.state_dict(),
                rng=torch.get_rng_state(),
            ),
            checkpoint_path,
        )

    torch.manual_seed(217)
    launch(cfg(1, 2), first_data, device=torch.device("cpu"), token_group=None, checkpoint=save)
    assert checkpoints == [
        dict(scanned_batches=1, completed_updates=0),
        dict(scanned_batches=2, completed_updates=1),
    ]
    payload = torch.load(checkpoint_path, weights_only=True)
    resumed, resumed_opt, resumed_sched, resumed_data = make()
    resumed.load_state_dict(payload["model"])
    resumed_opt.load_state_dict(payload["optimizer"])
    resumed_sched.load_state_dict(payload["scheduler"])
    resumed_data.load_state_dict(payload["data"])
    state = SFTState()
    state.load_state_dict(payload["task"])
    torch.set_rng_state(payload["rng"])
    resumed_state = launch(
        cfg(2, 4), resumed_data, device=torch.device("cpu"), token_group=None, state=state
    )
    assert (
        full_state.state_dict()
        == resumed_state.state_dict()
        == dict(scanned_batches=4, completed_updates=2)
    )
    assert resumed_data.scans == full_data.scans == [0, 1, 2, 3]
    assert resumed_data.commits == full_data.commits == [(1, 0), (3, 1)]
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    for key, values in resumed_opt.state_dict()["state"].items():
        assert values["step"].item() == 2
        for name, value in values.items():
            torch.testing.assert_close(value, expected_opt["state"][key][name], rtol=0, atol=0)
    assert resumed_sched.state_dict() == expected_sched
    assert torch.equal(torch.get_rng_state(), expected_rng)


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=25)
    )
    try:
        distributed.rank, distributed.world_size = rank, 2
        distributed.pp_size, distributed.dp_size, distributed.cp_size = 1, 2, 1
        distributed.pp_rank, distributed.dp_rank, distributed.cp_rank = 0, rank, 0
        global_counts = global_target_counts(torch.tensor([rank, 1]), token_group=dist.group.WORLD)
        assert global_counts.tolist() == [1, 2]
        features = torch.arange(600, dtype=torch.float64).reshape(2, 100, 3) / 500
        for counts in [(10, 90), (0, 100), (0, 0)]:
            torch.manual_seed(51)
            model = Engine()
            oracle = copy.deepcopy(model)
            optimizer, scheduler = context(model)
            oracle_opt = torch.optim.AdamW(oracle.parameters(), lr=0.01, weight_decay=0.2)
            labels = (torch.arange(200).reshape(2, 100) % 5).clone()
            for r, count in enumerate(counts):
                labels[r, count:] = -100
            local = [
                batch(features[rank : rank + 1, :30], labels[rank : rank + 1, :30]),
                batch(features[rank : rank + 1, 30:], labels[rank : rank + 1, 30:]),
            ]
            schedule_before = copy.deepcopy(scheduler.state_dict())
            result = train_step(cfg(), local, token_group=dist.group.WORLD)
            assert result.global_targets == sum(counts)
            if sum(counts):
                expected = F.cross_entropy(oracle(features).reshape(-1, 5), labels.flatten())
                expected.backward()
                oracle_opt.step()
                torch.testing.assert_close(
                    result.loss, expected.detach().float(), rtol=1e-6, atol=1e-7
                )
                assert result.updated and model.calls == 1
                assert all(value["step"].item() == 1 for value in optimizer.state.values())
            else:
                assert not result.updated and model.calls == 0 and not optimizer.state
                assert scheduler.state_dict() == schedule_before
            for actual, expected_parameter in zip(
                model.parameters(), oracle.parameters(), strict=True
            ):
                torch.testing.assert_close(actual, expected_parameter, rtol=1e-12, atol=1e-12)
        # A capability rejection discovered on one rank must reject both ranks,
        # including a peer whose local targets are all ignored.
        model = Engine()
        optimizer, scheduler = context(model)
        if rank == 1:
            model.projection.load_balance_loss_fn = SimpleNamespace(lb_coef=0.01)
        with pytest.raises(NotImplementedError, match="auxiliary"):
            train_step(
                cfg(),
                [batch(features[:1], torch.full((1, 100), -100))],
                token_group=dist.group.WORLD,
            )
        assert model.calls == 0 and not optimizer.state
        model = Engine()
        optimizer, scheduler = context(model)
        before = copy.deepcopy(model.state_dict())
        schedule = copy.deepcopy(scheduler.state_dict())
        handle = model.projection.weight.register_hook(
            lambda gradient: gradient * float("nan") if rank == 1 else gradient
        )
        with pytest.raises(FloatingPointError, match="no optimizer"):
            train_step(
                cfg(),
                [batch(features[:1], torch.zeros((1, 100), dtype=torch.long))],
                token_group=dist.group.WORLD,
            )
        handle.remove()
        assert not optimizer.state and scheduler.state_dict() == schedule
        for name, value in model.state_dict().items():
            assert torch.equal(value, before[name])
        model = Engine()
        optimizer, scheduler = context(model)
        local_labels = torch.zeros((1, 100), dtype=torch.long)
        if rank == 1:
            local_labels[0, 0] = -3
        with pytest.raises(ValueError, match="collective admission"):
            train_step(cfg(), [batch(features[:1], local_labels)], token_group=dist.group.WORLD)
        assert not optimizer.state and model.calls == 0
        distributed.pp_size, distributed.dp_size = 2, 1
        distributed.pp_rank, distributed.dp_rank = rank, 0
        with pytest.raises(ValueError, match="one PP stage"):
            train_step(
                cfg(),
                [batch(features[:1], torch.zeros_like(local_labels))],
                token_group=dist.group.WORLD,
            )
        assert not optimizer.state and model.calls == 0
        groups = [dist.new_group([r]) for r in range(2)]
        # A correct stage-local group still requires equal global target totals
        # across PP copies before any stage skips or starts the engine.
        unequal = torch.zeros_like(local_labels)
        if rank == 0:
            unequal.fill_(-100)
        with pytest.raises(ValueError, match="PP copies disagree"):
            train_step(cfg(), [batch(features[:1], unequal)], token_group=groups[rank])
        assert not optimizer.state and model.calls == 0
    finally:
        dist.destroy_process_group()


def test_actual_gloo_uneven_zero_rank_and_global_zero_update_contracts(tmp_path):
    rendezvous = (tmp_path / "gloo").as_uri()
    mp.spawn(distributed_worker, args=(rendezvous,), nprocs=2, join=True)


@pytest.mark.parametrize(
    "state",
    [
        dict(scanned_batches=0, completed_updates=1),
        dict(scanned_batches=True, completed_updates=0),
        dict(scanned_batches=-1, completed_updates=0),
        dict(scanned_batches=1),
    ],
)
def test_invalid_resume_cursors_rejected_before_partial_load(state):
    cursor = SFTState(3, 2)
    with pytest.raises(ValueError):
        cursor.load_state_dict(state)
    assert cursor.state_dict() == dict(scanned_batches=3, completed_updates=2)


@pytest.mark.parametrize("source", ["config", "runtime", "injector", "unknown_injector"])
def test_auxiliary_normalization_rejected_before_forward_even_zero_targets(source):
    model = Engine().eval()
    optimizer, scheduler = context(model)
    config = cfg()
    if source == "config":
        config.moe_load_balance_coef = 0.01
    elif source == "runtime":
        training.moe_load_balance_coef = 0.01
    else:
        model.projection.load_balance_loss_fn = (
            SimpleNamespace(lb_coef=0.01) if source == "injector" else object()
        )
    before, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    schedule = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(NotImplementedError, match="auxiliary"):
        train_step(
            config,
            [batch(torch.ones(1, 2, 3, dtype=torch.float64), torch.full((1, 2), -100))],
            token_group=None,
        )
    assert model.calls == 0 and not model.training and not optimizer.state
    assert scheduler.state_dict() == schedule and torch.equal(torch.get_rng_state(), rng)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_default_local_clipping_matches_independent_update_without_process_group():
    torch.manual_seed(59)
    model = Engine()
    oracle = copy.deepcopy(model)
    optimizer, _ = context(model)
    oracle_opt = torch.optim.AdamW(oracle.parameters(), lr=0.01, weight_decay=0.2)
    features, labels = torch.ones(1, 4, 3, dtype=torch.float64) * 20, torch.tensor([[1, 2, 3, 4]])
    train_step(SFTLMCfg(), [batch(features, labels)], token_group=None)
    F.cross_entropy(oracle(features).flatten(0, 1), labels.flatten()).backward()
    torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
    oracle_opt.step()
    for actual, expected in zip(model.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(value["step"].item() == 1 for value in optimizer.state.values())


class NativeData:
    def __init__(self):
        self.scanned_batches = self.completed_updates = 0

    def get_batch(self, scan_index, device):
        assert scan_index == self.scanned_batches
        self.scanned_batches += 1
        labels = torch.tensor([[-100 if scan_index % 2 == 0 else 1]])
        return [batch(torch.ones(1), labels)]

    def commit_update(self, *, scan_index, update_index):
        assert scan_index == self.scanned_batches - 1 and update_index == self.completed_updates
        self.completed_updates += 1

    def state_dict(self):
        return dict(
            fingerprint="native-sft-cpu",
            scanned_batches=self.scanned_batches,
            completed_updates=self.completed_updates,
        )

    def validate_state_dict(self, state):
        if set(state) != set(self.state_dict()) or state["fingerprint"] != "native-sft-cpu":
            raise ValueError("reader identity")
        SFTState().validate_state_dict(
            {key: state[key] for key in ("scanned_batches", "completed_updates")}
        )

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.scanned_batches, self.completed_updates = (
            state["scanned_batches"],
            state["completed_updates"],
        )


def test_native_sft_composite_preflight_is_pure():
    from pithtrain.tasks.sft_lm import SFTCheckpointState

    data, state = NativeData(), SFTState()
    composite = SFTCheckpointState(state, data)
    before = copy.deepcopy(composite.state_dict())
    bad = copy.deepcopy(before)
    bad["reader_state"]["fingerprint"] = "other"
    with pytest.raises(ValueError):
        composite.load_state_dict(bad)
    assert composite.state_dict() == before
    bad = copy.deepcopy(before)
    bad["data_cursor"] = True
    with pytest.raises(ValueError):
        composite.validate_state_dict(bad)
    assert composite.state_dict() == before


def native_sft_worker(rank, root):
    from pathlib import Path
    from types import MethodType

    from pithtrain.modules import distributed_checkpoint as native
    from pithtrain.modules.task_checkpoint import TaskCheckpointState
    from pithtrain.tasks.sft_lm import SFTCheckpointState
    from tests.acceptance.omni_native_resume import equal
    from tests.test_distributed_strict_checkpoint import build, initialize_mesh

    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(root / "native-sft-rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        meshes = initialize_mesh("fsdp-layout", rank)

        def make_runtime():
            model, opt, sch, _ = build(*meshes)

            def engine_step(self, microbatches, objective_fn):
                outputs = []
                for microbatch in microbatches:
                    # The strict fixture owns true CPU DTensor parameters. The
                    # protocol engine uses every parameter in a real CE graph.
                    scalar = sum(p.square().sum().full_tensor() for p in self.parameters())
                    logits = torch.stack((scalar, -scalar)).reshape(1, 1, 2)
                    loss, detached = objective_fn((logits,), microbatch.objective_inputs)
                    loss.backward()
                    outputs.append(detached)
                return outputs

            model.step = MethodType(engine_step, model)
            return model, opt, sch, NativeData(), SFTState()

        contract = {"model": "native-sft-protocol-v1", "hsdp_replica": 1}

        def snapshot(parts):
            model, opt, sch, data, state = parts
            adapter = TaskCheckpointState(
                SFTCheckpointState(state, data),
                task_kind="sft",
                task_schema=1,
                contract=contract,
                transaction_key="scanned_batches",
                update_key="completed_updates",
                data_key="data_cursor",
            )
            values, _ = native._runtime(model, (opt,), (sch,), adapter, torch.device("cpu"))
            return native._pack(values)

        full = make_runtime()
        torch.manual_seed(900 + rank)
        launch(
            cfg(2, 4),
            full[3],
            device=torch.device("cpu"),
            token_group=dist.group.WORLD,
            state=full[4],
        )
        expected = snapshot(full)
        first = make_runtime()
        torch.manual_seed(900 + rank)
        launch(
            cfg(1, 2),
            first[3],
            device=torch.device("cpu"),
            token_group=dist.group.WORLD,
            state=first[4],
            checkpoint_root=root / "checkpoints",
            checkpoint_contract=contract,
        )
        assert first[4].state_dict() == dict(scanned_batches=2, completed_updates=1)
        resumed = make_runtime()
        launch(
            cfg(2, 4),
            resumed[3],
            device=torch.device("cpu"),
            token_group=dist.group.WORLD,
            state=resumed[4],
            checkpoint_root=root / "checkpoints",
            checkpoint_contract=contract,
            resume_scan=2,
        )
        equal(snapshot(resumed), expected)
        assert resumed[4].completed_updates == 2 and resumed[4].scanned_batches == 4
        assert all(item["step"].item() == 2 for item in resumed[1].state.values())
        assert resumed[2].last_epoch == 2
        zero = make_runtime()
        launch(
            cfg(0, 1),
            zero[3],
            device=torch.device("cpu"),
            token_group=dist.group.WORLD,
            state=zero[4],
            checkpoint_root=root / "checkpoints",
            checkpoint_contract=contract,
            resume_scan=1,
        )
        assert zero[4].state_dict() == dict(scanned_batches=1, completed_updates=0)
        assert not zero[1].state and zero[2].last_epoch == 0
    finally:
        dist.destroy_process_group()


def test_actual_sft_launch_native_gloo_dtensor_checkpoint_skip_and_resume(tmp_path):
    mp.spawn(native_sft_worker, args=(str(tmp_path),), nprocs=2, join=True)


@pytest.mark.parametrize("clipping", [None, 1.0])
def test_nonfinite_sft_gradients_never_update_optimizer_or_scheduler(clipping):
    model = Engine()
    optimizer, scheduler = context(model)
    config = cfg()
    config.max_grad_norm = clipping
    before, schedule = copy.deepcopy(model.state_dict()), copy.deepcopy(scheduler.state_dict())
    handle = model.projection.weight.register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        train_step(
            config,
            [batch(torch.ones(1, 2, 3, dtype=torch.float64), torch.tensor([[1, 2]]))],
            token_group=None,
        )
    handle.remove()
    assert not optimizer.state and scheduler.state_dict() == schedule
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
