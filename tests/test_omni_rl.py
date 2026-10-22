"""CPU rewarding-rollout/update/resume and native objective-adapter contracts.

The local toy is an actual policy/reward loop. The pipeline double tests the
native .step boundary and SUM normalization; it is not a CUDA DualPipeV run.
"""

import copy
import hashlib
import json
import math
import random
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.tasks.omni_rl import (
    OmniRLCfg,
    OmniRLTrainer,
    PithTrainPolicyAdapter,
    PolicyIdentity,
    RolloutBatch,
    RolloutContract,
    TorchPolicyAdapter,
    fingerprint,
    group_relative_advantages,
    grpo_loss_sum,
    launch,
)


def config(**kwargs):
    cfg = OmniRLCfg()
    cfg.max_updates, cfg.max_requests, cfg.max_samples = 3, 4, 4
    cfg.max_sequence_length = 3
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def contract():
    return RolloutContract(
        *(
            fingerprint(v)
            for v in (
                "toy-user-assistant-v1",
                {"tokenizer": "four-symbols"},
                {"position": "absolute"},
                {"reference": "uniform", "vocab": 4},
                {"reward": "count-zero-symbol-v1"},
            )
        )
    )


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.4, -0.2, 0.1, 0.0],
                    [0.1, 0.3, -0.2, 0.0],
                    [0.2, 0.0, 0.1, -0.1],
                    [0.0, -0.1, 0.3, 0.2],
                ]
            )
        )

    def forward(self, ids, **context):
        assert set(context) <= {"positions", "pixel_values"}
        return self.table[ids]


class RewardingRollout:
    def __init__(self, *, zero=False, tamper=None):
        self.calls = 0
        self.zero, self.tamper = zero, tamper

    def synchronize(self, snapshot):
        self.identity, self.weights = snapshot.identity, copy.deepcopy(snapshot.payload)
        return self.identity

    def generate(self, request):
        assert request.identity == self.identity
        generator = torch.Generator().manual_seed(request.seed + request.dp_rank)
        ids = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]])
        logits = self.weights["table"][ids]
        tokens = torch.multinomial(logits.softmax(-1).view(-1, 4), 1, generator=generator).view(
            4, 3
        )
        old = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mask[:, 0] = False
        if self.zero:
            mask.zero_()
        self.calls += 1
        batch = RolloutBatch(
            request.identity,
            request.request_index,
            ids,
            tokens,
            mask,
            old,
            torch.full_like(old, -math.log(4)),
            tuple(f"dp{request.dp_rank}-group{i // 2}" for i in range(4)),
            tuple(f"request{request.request_index}-sample{i}" for i in range(4)),
        )
        if self.tamper is not None:
            self.tamper(batch)
        return batch

    def state_dict(self):
        return {"calls": self.calls, "schema": 1}

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.calls = state["calls"]

    def validate_state_dict(self, state):
        if set(state) != {"calls", "schema"} or state["schema"] != 1:
            raise ValueError("Rollout cursor schema differs")


class SymbolReward:
    def __init__(self):
        self.calls = 0

    def score(self, request, batch):
        self.calls += 1
        return ((batch.sampled_token_ids == 0) & batch.token_mask).float().sum(-1)

    def state_dict(self):
        return {"calls": self.calls, "reward": "count-zero-symbol-v1"}

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.calls = state["calls"]

    def validate_state_dict(self, state):
        if state["reward"] != "count-zero-symbol-v1":
            raise ValueError("Reward version differs")


def make_trainer(cfg=None, *, rollout=None, policy_contract=None):
    model = ToyPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    backend = TorchPolicyAdapter(model, opt, scheduler=scheduler)
    trainer = OmniRLTrainer(
        cfg or config(),
        contract=policy_contract or contract(),
        backend=backend,
        rollout=rollout or RewardingRollout(),
        reward=SymbolReward(),
    )
    return trainer


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            tree_equal(left, right)
    else:
        assert a == b


def test_rewarding_rollout_updates_actual_policy_and_reproduces_after_resume(tmp_path):
    torch.manual_seed(19)
    original = make_trainer()
    before = original.backend.model.table.detach().clone()
    first = original.train_step()
    assert first["status"] == "UPDATED" and first["global_tokens"] == 8
    assert not torch.equal(before, original.backend.model.table)
    assert original.policy_version == original.updates == original.requests == 1
    original.backend.save(tmp_path / "checkpoint", original)
    second = original.train_step()
    expected_state = copy.deepcopy(original.backend.model.state_dict())
    expected_optimizer = copy.deepcopy(original.backend.optimizer.state_dict())
    expected_rng = torch.rand(4)
    restored = make_trainer()
    restored.backend.load(tmp_path / "checkpoint", restored)
    resumed_second = restored.train_step()
    assert second == resumed_second
    tree_equal(expected_state, restored.backend.model.state_dict())
    tree_equal(expected_optimizer, restored.backend.optimizer.state_dict())
    tree_equal(original.state_dict(), restored.state_dict())
    tree_equal(expected_rng, torch.rand(4))
    assert not torch.cuda.is_initialized()


def test_global_zero_advances_only_request_data_cursor_and_is_bounded():
    trainer = make_trainer(
        config(max_updates=2, max_requests=2), rollout=RewardingRollout(zero=True)
    )
    before = copy.deepcopy(trainer.backend.model.state_dict())
    scheduler = copy.deepcopy(trainer.backend.scheduler.state_dict())
    trainer, reports = launch(
        trainer.cfg,
        contract=trainer.contract,
        backend=trainer.backend,
        rollout=trainer.rollout,
        reward=trainer.reward,
        trainer=trainer,
    )
    assert [x["status"] for x in reports] == ["SKIPPED_GLOBAL_ZERO"] * 2
    assert trainer.requests == 2 and trainer.data_cursor == 8
    assert trainer.policy_version == trainer.updates == 0
    assert trainer.backend.optimizer.state == {}
    tree_equal(before, trainer.backend.model.state_dict())
    tree_equal(scheduler, trainer.backend.scheduler.state_dict())
    with pytest.raises(ValueError, match="limit"):
        trainer.train_step()


def test_relative_advantages_and_zero_reward_variance():
    actual = group_relative_advantages(
        torch.tensor([1.0, 3.0, 7.0, 7.0]), ("a", "a", "b", "b"), 2, 1e-6
    )
    torch.testing.assert_close(actual, torch.tensor([-1 / 1.000001, 1 / 1.000001, 0.0, 0.0]))
    with pytest.raises(ValueError, match="spanning DP"):
        group_relative_advantages(torch.tensor([1.0, 3.0]), ("a", "b"), 2, 1e-6)


def test_clipped_grpo_and_kl_match_independent_scalar_formula():
    cfg = config()
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
    sampled = torch.tensor([[0, 1], [0, 1]])
    mask = torch.tensor([[True, False], [True, True]])
    old = torch.tensor([[-1.0, -2.0], [-0.2, -1.0]])
    reference = torch.full_like(old, -math.log(2))
    advantage = torch.tensor([1.0, -1.0])
    actual = grpo_loss_sum(logits, sampled, old, reference, mask, advantage, cfg)
    current = logits.log_softmax(-1)
    expected = torch.zeros(())
    for row in range(2):
        for token in range(2):
            if mask[row, token]:
                value = current[row, token, sampled[row, token]]
                ratio = torch.exp(value - old[row, token])
                delta = reference[row, token] - value
                expected = (
                    expected
                    - torch.minimum(
                        ratio * advantage[row], torch.clamp(ratio, 0.8, 1.2) * advantage[row]
                    )
                    + 0.02 * (torch.exp(delta) - delta - 1)
                )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.equal(logits.grad[0, 1], torch.zeros(2))
    assert torch.isfinite(logits.grad).all()


def test_local_zero_is_differentiable_and_does_not_gather_masked_invalid_ids():
    logits = torch.randn(2, 3, 4, requires_grad=True)
    loss = grpo_loss_sum(
        logits,
        torch.full((2, 3), -100),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.zeros(2),
        config(),
    )
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    "field",
    [
        "chat_template_sha256",
        "processor_sha256",
        "position_config_sha256",
        "reference_policy_sha256",
        "reward_config_sha256",
    ],
)
def test_each_contract_fingerprint_mismatch_rejects_before_update(field):
    def tamper(batch):
        batch.identity = replace(
            batch.identity, contract=replace(batch.identity.contract, **{field: "f" * 64})
        )

    trainer = make_trainer(rollout=RewardingRollout(tamper=tamper))
    before = trainer.backend.model.table.detach().clone()
    with pytest.raises(ValueError, match="Stale/mismatched"):
        trainer.train_step()
    assert torch.equal(before, trainer.backend.model.table)
    assert trainer.backend.optimizer.state == {}
    with pytest.raises(ValueError, match="failed"):
        trainer.state_dict()


@pytest.mark.parametrize(
    "tamper,reason",
    [
        (lambda b: setattr(b, "identity", replace(b.identity, version=3)), "Stale/mismatched"),
        (
            lambda b: setattr(b, "identity", replace(b.identity, weights_sha256="e" * 64)),
            "Stale/mismatched",
        ),
        (lambda b: setattr(b, "request_index", 8), "Stale/mismatched"),
        (lambda b: b.old_logprobs.fill_(float("nan")), "logprob"),
        (lambda b: setattr(b, "sample_ids", ("same",) * 4), "duplicate"),
        (lambda b: setattr(b, "group_ids", ("a", "b", "c", "d")), "Incomplete"),
    ],
)
def test_bad_rollout_cannot_update(tamper, reason):
    trainer = make_trainer(rollout=RewardingRollout(tamper=tamper))
    with pytest.raises(ValueError, match=reason):
        trainer.train_step()
    assert trainer.updates == 0 and trainer.backend.optimizer.state == {}


def test_checkpoint_fingerprint_and_counter_corruption_rejected_before_mutation(tmp_path):
    trainer = make_trainer()
    trainer.train_step()
    trainer.backend.save(tmp_path / "state", trainer)
    other = make_trainer(policy_contract=replace(contract(), processor_sha256="c" * 64))
    before = other.backend.model.table.detach().clone()
    with pytest.raises(ValueError, match="fingerprints"):
        other.backend.load(tmp_path / "state", other)
    assert torch.equal(before, other.backend.model.table)
    corrupted = trainer.state_dict()
    corrupted["policy_version"] = 7
    with pytest.raises(ValueError, match="counters"):
        trainer.load_state_dict(corrupted)
    assert trainer.policy_version == 1
    with (tmp_path / "state/state.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        trainer.backend.load(tmp_path / "state", trainer)


class PipelineDouble(torch.nn.Module):
    def __init__(self, group=None):
        super().__init__()
        self.policy = ToyPolicy()
        self.group = group
        self.calls = 0

    def step(self, microbatches, objective):
        self.calls += 1
        metrics = []
        for mb in microbatches:
            loss, metric = objective((self.policy(*mb.model_inputs),), mb.objective_inputs)
            loss.backward()
            metrics.append(metric)
        if dist.is_initialized():
            for p in self.parameters():
                dist.all_reduce(
                    p.grad, group=self.group
                )  # emulate documented FSDP SUM, not average
        return metrics


def distributed_worker(rank, directory):
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.set_num_threads(1)
        pp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        token_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        dp_rank, pp_rank = rank // 2, rank % 2
        model = PipelineDouble(token_groups[pp_rank])
        opt = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)

        def export(version, cfg, semantics):
            # Use the local policy exporter on a single-rank-shaped double; no
            # actual external service or production sharded-weight export is tested.
            from pithtrain.tasks.omni_rl import PolicySnapshot, _weights_digest

            weights = copy.deepcopy(model.policy.state_dict())
            return PolicySnapshot(
                PolicyIdentity(version, _weights_digest(weights), cfg, semantics["sha256"]),
                weights,
                semantics,
            )

        backend = PithTrainPolicyAdapter(
            model,
            (opt,),
            (sched,),
            device=torch.device("cpu"),
            token_group=token_groups[pp_rank],
            pp_group=pp_groups[dp_rank],
            pp_rank=pp_rank,
            pp_size=2,
            dp_rank=dp_rank,
            export_policy=export,
            clip_grad=lambda maximum: torch.nn.utils.clip_grad_norm_(model.parameters(), maximum),
        )
        trainer = OmniRLTrainer(
            config(),
            contract=contract(),
            backend=backend,
            rollout=RewardingRollout(zero=dp_rank == 0),
            reward=SymbolReward(),
        )
        report = trainer.train_step()
        assert report["global_tokens"] == 8  # not 16 from counting both PP copies
        assert trainer.policy_version == 1 and model.calls == 1
        weights = [None] * 4
        dist.all_gather_object(weights, model.policy.table.detach().tolist())
        assert all(value == weights[0] for value in weights)
        before = model.policy.table.detach().clone()
        scheduler_before = copy.deepcopy(sched.state_dict())
        trainer.rollout.zero = True
        zero = trainer.train_step()
        assert zero["status"] == "SKIPPED_GLOBAL_ZERO"
        assert trainer.requests == 2 and trainer.updates == trainer.policy_version == 1
        assert torch.equal(before, model.policy.table) and sched.state_dict() == scheduler_before
        assert model.calls == 1
        if rank == 1:
            trainer.rollout.tamper = lambda b: b.input_ids.__setitem__((0, 0), 3)
        try:
            trainer.train_step()
        except ValueError as error:
            assert "Pipeline ranks received different" in str(error)
        else:
            raise AssertionError("PP mismatch was not rejected uniformly")
        backend.token_group = dist.group.WORLD
        wrong_group = OmniRLTrainer(
            config(),
            contract=contract(),
            backend=backend,
            rollout=RewardingRollout(),
            reward=SymbolReward(),
        )
        try:
            wrong_group.train_step()
        except ValueError as error:
            assert "Token-count group" in str(error)
        else:
            raise AssertionError("PP-duplicating token group was not rejected")
        assert model.calls == 1
        (root / f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    global_tokens=report["global_tokens"],
                    zero_status=zero["status"],
                    updates=trainer.updates,
                    requests=trainer.requests,
                    pp_mismatch_rejected=True,
                    cuda_initialized=torch.cuda.is_initialized(),
                )
            )
        )
    finally:
        dist.destroy_process_group()


def test_native_adapter_sum_normalization_local_zero_global_zero_pp_consistency(tmp_path):
    mp.spawn(distributed_worker, args=(str(tmp_path),), nprocs=4, join=True)
    rows = [json.loads((tmp_path / f"rank-{i}.json").read_text()) for i in range(4)]
    assert all(row["global_tokens"] == 8 and row["pp_mismatch_rejected"] for row in rows)
    assert all(not row["cuda_initialized"] for row in rows)


def test_native_adapter_requires_explicit_token_group():
    with pytest.raises(ValueError, match="stage-local"):
        PithTrainPolicyAdapter(
            None,
            (),
            (),
            device="cpu",
            token_group=None,
            pp_group=None,
            pp_rank=0,
            pp_size=1,
            dp_rank=0,
            export_policy=None,
            clip_grad=None,
        )


def test_zero_request_checkpoint_preserves_scan_and_future_sampling(tmp_path):
    trainer = make_trainer(rollout=RewardingRollout(zero=True))
    trainer.train_step()
    trainer.backend.save(tmp_path / "zero-checkpoint", trainer)
    restored = make_trainer(rollout=RewardingRollout(zero=True))
    restored.backend.load(tmp_path / "zero-checkpoint", restored)
    assert restored.requests == 1 and restored.updates == restored.policy_version == 0
    assert restored.data_cursor == 4 and restored.backend.optimizer.state == {}
    trainer.rollout.zero = restored.rollout.zero = False
    assert trainer.train_step() == restored.train_step()
    tree_equal(trainer.backend.model.state_dict(), restored.backend.model.state_dict())


def test_optimizer_options_and_provider_state_mismatch_fail_before_mutation(tmp_path):
    trainer = make_trainer()
    trainer.train_step()
    trainer.backend.save(tmp_path / "checkpoint", trainer)
    other = make_trainer()
    other.backend.optimizer_contract[0]["defaults"]["lr"] = 0.5
    before = other.backend.model.table.detach().clone()
    with pytest.raises(ValueError, match="fingerprints"):
        other.backend.load(tmp_path / "checkpoint", other)
    assert torch.equal(before, other.backend.model.table)
    invalid = trainer.state_dict()
    invalid["reward"]["reward"] = "other-reward"
    with pytest.raises(ValueError, match="Reward version"):
        trainer.load_state_dict(invalid)
    assert trainer.updates == trainer.policy_version == 1


@pytest.mark.parametrize("fault", ["step", "moment-shape", "missing-parameter"])
def test_adam_checkpoint_state_contract_rejects_before_mutation(tmp_path, fault):
    trainer = make_trainer()
    trainer.train_step()
    trainer.backend.save(tmp_path / "checkpoint", trainer)
    file = tmp_path / "checkpoint/state.pt"
    payload = torch.load(file, weights_only=True)
    state = payload["optimizer"]["state"]
    key = next(iter(state))
    if fault == "step":
        state[key]["step"] += 1
    elif fault == "moment-shape":
        state[key]["exp_avg"] = torch.zeros(1)
    else:
        del state[key]
    torch.save(payload, file)
    (file.parent / "sha256").write_text(hashlib.sha256(file.read_bytes()).hexdigest())
    other = make_trainer()
    before = other.backend.model.table.detach().clone()
    with pytest.raises(ValueError, match="RL Adam"):
        other.backend.load(file.parent, other)
    assert torch.equal(before, other.backend.model.table)


def native_checkpoint_worker(rank, directory):
    from pithtrain.contexts import distributed, training
    from tests.test_distributed_strict_checkpoint import initialize_mesh

    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'checkpoint-rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.set_num_threads(1)
        initialize_mesh("fsdp-layout", rank)
        token_group = distributed.attn_mesh["dp", "cp"]._flatten().get_group()
        model = PipelineDouble(token_group)
        model.policy = NormalizedPolicy()
        opt = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0, foreach=False)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule_factory(0.9))
        training.model, training.optimizers, training.schedulers = model, (opt,), (sched,)

        def export(version, cfg, semantics):
            from pithtrain.tasks.omni_rl import PolicySnapshot, _weights_digest

            weights = copy.deepcopy(model.policy.state_dict())
            return PolicySnapshot(
                PolicyIdentity(version, _weights_digest(weights), cfg, semantics["sha256"]),
                weights,
                semantics,
            )

        backend = PithTrainPolicyAdapter(
            model,
            (opt,),
            (sched,),
            device=torch.device("cpu"),
            token_group=token_group,
            pp_group=None,
            pp_rank=0,
            pp_size=1,
            dp_rank=rank,
            export_policy=export,
            clip_grad=lambda maximum: torch.nn.utils.clip_grad_norm_(model.parameters(), maximum),
        )
        trainer = OmniRLTrainer(
            config(),
            contract=contract(),
            backend=backend,
            rollout=RewardingRollout(zero=True),
            reward=SymbolReward(),
        )
        assert trainer.train_step()["status"] == "SKIPPED_GLOBAL_ZERO"
        backend.save(root / "native", trainer)
        trainer.rollout.zero = False
        second = trainer.train_step()
        second_model = copy.deepcopy(model.state_dict())
        second_optimizer = copy.deepcopy(opt.state_dict())
        backend.save(root / "native", trainer)
        third = trainer.train_step()
        third_model = copy.deepcopy(model.state_dict())
        third_optimizer = copy.deepcopy(opt.state_dict())
        third_task = trainer.state_dict()
        from pithtrain.modules.distributed_checkpoint import DistributedCheckpointError

        for drift in ("epsilon", "scheduler"):
            if rank == 1:
                if drift == "epsilon":
                    model.policy.norm.eps = 1.0
                else:
                    sched.lr_lambdas = [_schedule_factory(0.1)]
            before = copy.deepcopy(
                (
                    model.state_dict(),
                    opt.state_dict(),
                    sched.state_dict(),
                    trainer.state_dict(),
                    torch.get_rng_state(),
                )
            )
            with pytest.raises(DistributedCheckpointError):
                backend.load(root / "native", trainer, request_index=2)
            tree_equal(
                before,
                (
                    model.state_dict(),
                    opt.state_dict(),
                    sched.state_dict(),
                    trainer.state_dict(),
                    torch.get_rng_state(),
                ),
            )
            model.policy.norm.eps = 1e-5
            sched.lr_lambdas = [_schedule_factory(0.9)]
        backend.load(root / "native", trainer, request_index=2)
        assert trainer.requests == 2 and trainer.updates == trainer.policy_version == 1
        assert trainer.train_step() == third
        tree_equal(model.state_dict(), third_model)
        tree_equal(opt.state_dict(), third_optimizer)
        tree_equal(trainer.state_dict(), third_task)
        backend.load(root / "native", trainer, request_index=1)
        assert trainer.requests == 1 and trainer.updates == trainer.policy_version == 0
        assert not opt.state
        assert trainer.train_step() == second
        tree_equal(model.state_dict(), second_model)
        tree_equal(opt.state_dict(), second_optimizer)
        (root / f"native-rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    status="PASS_CPU_NATIVE_CHECKPOINT_BOUNDARY",
                    zero_update_restore=True,
                    next_update_exact=True,
                    cuda_initialized=torch.cuda.is_initialized(),
                )
            )
        )
    finally:
        dist.destroy_process_group()


def test_native_checkpoint_entrypoints_preserve_rl_request_update_provider_state(tmp_path):
    mp.spawn(native_checkpoint_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        report = json.loads((tmp_path / f"native-rank-{rank}.json").read_text())
        assert report["next_update_exact"] and report["zero_update_restore"]
        assert not report["cuda_initialized"]


def test_explicit_microbatch_media_and_position_context_are_forwarded_without_slicing():
    def add_context(batch):
        batch.model_context = tuple({"positions": torch.arange(3).view(1, 3)} for _ in range(4))
        batch.media_inputs = tuple(
            {"pixel_values": torch.full((2 + i, 5), float(i))} for i in range(4)
        )

    trainer = make_trainer(rollout=RewardingRollout(tamper=add_context))
    seen = []
    model = trainer.backend.model

    def forward(inputs, context):
        seen.append({key: value.clone() for key, value in context.items()})
        return model(*inputs, **context)

    trainer.backend.forward = forward
    assert trainer.train_step()["status"] == "UPDATED"
    assert [row["pixel_values"].shape[0] for row in seen] == [2, 3, 4, 5]
    assert all(torch.equal(row["positions"], torch.arange(3).view(1, 3)) for row in seen)


def test_optimizer_must_cover_exact_policy_parameters():
    model = ToyPolicy()
    other = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="foreign"):
        TorchPolicyAdapter(model, torch.optim.AdamW([other]))


def test_launch_cannot_substitute_existing_trainer_configuration():
    trainer = make_trainer()
    with pytest.raises(ValueError, match="does not match"):
        launch(
            config(max_requests=99),
            contract=trainer.contract,
            backend=trainer.backend,
            rollout=trainer.rollout,
            reward=trainer.reward,
            trainer=trainer,
        )
    assert trainer.requests == trainer.updates == 0


@pytest.mark.parametrize(
    "cp,coefficient,benchmark,reason",
    [
        (2, 0.0, False, "context parallel"),
        (1, 0.1, False, "load-balance"),
        (1, 0.0, True, "real routing"),
    ],
)
def test_native_context_rejects_unsupported_parallelism_and_hidden_objectives(
    monkeypatch, cp, coefficient, benchmark, reason
):
    from types import SimpleNamespace

    from pithtrain.contexts import distributed

    monkeypatch.setattr(distributed, "cp_size", cp, raising=False)
    with pytest.raises(ValueError, match=reason):
        PithTrainPolicyAdapter.from_training_context(
            export_policy=None,
            training_cfg=SimpleNamespace(moe_load_balance_coef=coefficient, benchmark=benchmark),
        )
    assert not torch.cuda.is_initialized()


def test_native_context_rejects_unsupported_hsdp_resume(monkeypatch):
    from types import SimpleNamespace

    from pithtrain.contexts import distributed

    monkeypatch.setattr(distributed, "cp_size", 1, raising=False)
    with pytest.raises(ValueError, match="HSDP"):
        PithTrainPolicyAdapter.from_training_context(
            export_policy=None,
            hsdp_replica=2,
            training_cfg=SimpleNamespace(moe_load_balance_coef=0.0, benchmark=False),
        )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_requests", 0),
        ("group_size", 1),
        ("clip_epsilon", 2),
        ("kl_beta", float("nan")),
        ("max_grad_norm", float("inf")),
    ],
)
def test_invalid_config_rejected(field, value):
    with pytest.raises(ValueError):
        make_trainer(config(**{field: value}))


class StochasticPolicy(ToyPolicy):
    def forward(self, ids, **context):
        value = torch.nn.functional.dropout(super().forward(ids, **context), 0.35, self.training)
        return value * (0.8 + 0.1 * random.random() + 0.1 * np.random.random())


class StochasticRollout(RewardingRollout):
    """Independent provider RNGs belong to provider state, not global RNG state."""

    def __init__(self):
        super().__init__()
        self.python = random.Random(144)
        self.numpy = np.random.RandomState(255)

    def generate(self, request):
        offset = self.python.randrange(10000) + self.numpy.randint(10000)
        offset += random.randrange(10000) + np.random.randint(10000)
        return super().generate(replace(request, seed=request.seed + int(offset)))

    def state_dict(self):
        state = self.numpy.get_state()
        return dict(
            calls=self.calls,
            python=self.python.getstate(),
            numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]),
        )

    def validate_state_dict(self, state):
        assert set(state) == {"calls", "python", "numpy"}
        assert type(state["calls"]) is int and state["calls"] >= 0
        random.Random().setstate(state["python"])
        value = state["numpy"]
        np.random.RandomState().set_state(
            (value[0], np.array(value[1], dtype=np.uint32), *value[2:])
        )

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.calls = state["calls"]
        self.python.setstate(state["python"])
        value = state["numpy"]
        self.numpy.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))


def make_stochastic_trainer():
    model = StochasticPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    return OmniRLTrainer(
        config(),
        contract=contract(),
        backend=TorchPolicyAdapter(
            model, optimizer, scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        ),
        rollout=StochasticRollout(),
        reward=SymbolReward(),
    )


def training_snapshot(trainer):
    from pithtrain.tasks.omni_rl import _local_rng_state

    return copy.deepcopy(
        dict(
            model=trainer.backend.model.state_dict(),
            optimizer=trainer.backend.optimizer.state_dict(),
            scheduler=trainer.backend.scheduler.state_dict(),
            trainer=trainer.state_dict(),
            rng=_local_rng_state(),
        )
    )


def rewrite_checkpoint(path, mutate):
    payload = torch.load(path / "state.pt", weights_only=True)
    mutate(payload)
    torch.save(payload, path / "state.pt")
    (path / "sha256").write_text(hashlib.sha256((path / "state.pt").read_bytes()).hexdigest())


def test_dropout_python_numpy_and_stochastic_provider_exact_next_update(tmp_path):
    random.seed(178)
    np.random.seed(289)
    torch.manual_seed(390)
    # Exercise Python's optional cached Gaussian, which is absent in fresh RNGs.
    random.gauss(0, 1)
    trainer = make_stochastic_trainer()
    assert trainer.train_step()["status"] == "UPDATED"
    before_save = training_snapshot(trainer)
    trainer.backend.save(tmp_path / "checkpoint", trainer)
    tree_equal(before_save, training_snapshot(trainer))
    expected_report = trainer.train_step()
    expected = training_snapshot(trainer)
    random.seed(999)
    np.random.seed(888)
    torch.manual_seed(777)
    restored = make_stochastic_trainer()
    restored.rollout.python.seed(333)
    restored.rollout.numpy.seed(444)
    restored.backend.load(tmp_path / "checkpoint", restored)
    assert restored.train_step() == expected_report
    tree_equal(expected, training_snapshot(restored))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("field", ["torch_cpu", "python", "numpy", "cuda"])
def test_rng_corruption_rejected_before_model_or_provider_mutation(tmp_path, monkeypatch, field):
    trainer = make_stochastic_trainer()
    trainer.train_step()
    path = tmp_path / "checkpoint"
    trainer.backend.save(path, trainer)

    def corrupt(payload):
        payload["rng"][field] = {
            "torch_cpu": torch.zeros(5, dtype=torch.float32),
            "python": (0, (), None),
            "numpy": ("invalid", [], 0, 0, 0.0),
            "cuda": [torch.zeros(5, dtype=torch.uint8)],
        }[field]

    rewrite_checkpoint(path, corrupt)
    before = training_snapshot(trainer)
    monkeypatch.setattr(
        trainer.backend.model,
        "load_state_dict",
        lambda *a, **k: pytest.fail("Model mutated before RNG validation"),
    )
    monkeypatch.setattr(
        trainer.rollout,
        "load_state_dict",
        lambda *a, **k: pytest.fail("Provider mutated before RNG validation"),
    )
    with pytest.raises(ValueError):
        trainer.backend.load(path, trainer)
    tree_equal(before, training_snapshot(trainer))


def test_rng_and_training_state_rollback_after_provider_restore_failure(tmp_path, monkeypatch):
    trainer = make_stochastic_trainer()
    trainer.train_step()
    trainer.backend.save(tmp_path / "checkpoint", trainer)
    trainer.train_step()
    before = training_snapshot(trainer)
    original = trainer.reward.load_state_dict
    calls = 0

    def fail_once_after_mutation(state):
        nonlocal calls
        original(state)
        calls += 1
        random.random()
        np.random.random()
        torch.rand(3)
        if calls == 1:
            raise RuntimeError("Injected reward restore failure")

    monkeypatch.setattr(trainer.reward, "load_state_dict", fail_once_after_mutation)
    with pytest.raises(RuntimeError, match="Injected reward restore failure"):
        trainer.backend.load(tmp_path / "checkpoint", trainer)
    assert calls == 2
    tree_equal(before, training_snapshot(trainer))


def test_incomplete_legacy_rng_checkpoint_is_rejected(tmp_path):
    trainer = make_trainer()
    trainer.backend.save(tmp_path / "checkpoint", trainer)

    def legacy(payload):
        payload["schema"] = 1
        payload["torch_rng"] = payload.pop("rng")["torch_cpu"]

    rewrite_checkpoint(tmp_path / "checkpoint", legacy)
    before = training_snapshot(trainer)
    with pytest.raises(ValueError, match="Unsupported RL checkpoint schema"):
        trainer.backend.load(tmp_path / "checkpoint", trainer)
    tree_equal(before, training_snapshot(trainer))


def test_local_policy_rejects_mixed_device_buffers_before_training():
    model = ToyPolicy()
    model.register_buffer("unsupported", torch.empty(1, device="meta"))
    with pytest.raises(ValueError, match="one CPU or CUDA"):
        TorchPolicyAdapter(model, torch.optim.AdamW(model.parameters()))


def _contract_snapshot(trainer):
    from pithtrain.tasks.omni_rl import _local_rng_state

    backend = trainer.backend
    return copy.deepcopy(
        (
            backend.model.state_dict(),
            backend.optimizer.state_dict(),
            backend.scheduler.state_dict(),
            trainer.state_dict(),
            _local_rng_state(),
            [parameter.grad for parameter in backend.model.parameters()],
        )
    )


def _schedule_factory(factor):
    return lambda step: 1.0 if step < 2 else factor


@pytest.mark.parametrize("drift", ["code", "closure"])
def test_changed_lambda_policy_rejected_before_any_rl_state_mutation(tmp_path, drift):
    original = make_trainer()
    original.backend.scheduler.lr_lambdas = [_schedule_factory(1.0)]
    original.train_step()
    original.backend.save(tmp_path / "policy", original)
    restored = make_trainer()
    restored.backend.scheduler.lr_lambdas = [
        (lambda step: 1.0 if step < 2 else 0.1) if drift == "code" else _schedule_factory(0.1)
    ]
    # PyTorch's own scheduler serialization cannot distinguish these functions.
    assert restored.backend.scheduler.state_dict()["lr_lambdas"] == [None]
    before = _contract_snapshot(restored)
    with pytest.raises(ValueError, match="semantics"):
        restored.backend.load(tmp_path / "policy", restored)
    tree_equal(before, _contract_snapshot(restored))


class NormalizedPolicy(ToyPolicy):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.norm = torch.nn.LayerNorm(4, eps=eps)

    def forward(self, ids, **context):
        return self.norm(super().forward(ids, **context))


def _normalized_trainer(eps):
    model = NormalizedPolicy(eps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule_factory(1.0))
    return OmniRLTrainer(
        config(),
        contract=contract(),
        backend=TorchPolicyAdapter(model, optimizer, scheduler=scheduler),
        rollout=RewardingRollout(),
        reward=SymbolReward(),
    )


def test_rl_same_tensors_changed_epsilon_changes_publication_and_rejects_resume(tmp_path):
    original, restored = _normalized_trainer(1e-5), _normalized_trainer(1.0)
    tree_equal(original.backend.model.state_dict(), restored.backend.model.state_dict())
    left, right = (trainer.backend.snapshot(0, contract()) for trainer in (original, restored))
    assert left.identity.weights_sha256 == right.identity.weights_sha256
    assert left.identity.semantics_sha256 != right.identity.semantics_sha256
    assert left.identity != right.identity
    assert left.semantics != right.semantics
    original.backend.save(tmp_path / "policy", original)
    before = _contract_snapshot(restored)
    with pytest.raises(ValueError, match="semantics"):
        restored.backend.load(tmp_path / "policy", restored)
    tree_equal(before, _contract_snapshot(restored))


def test_forward_adapter_closure_changes_publication_and_resume_contract(tmp_path):
    def forward(model, scale):
        return lambda inputs, context: model(*inputs, **context) * scale

    original, restored = make_trainer(), make_trainer()
    original.backend.forward = forward(original.backend.model, 1.0)
    restored.backend.forward = forward(restored.backend.model, 0.5)
    assert (
        original.backend.snapshot(0, contract()).identity
        != restored.backend.snapshot(0, contract()).identity
    )
    original.backend.save(tmp_path / "policy", original)
    before = _contract_snapshot(restored)
    with pytest.raises(ValueError, match="semantics"):
        restored.backend.load(tmp_path / "policy", restored)
    tree_equal(before, _contract_snapshot(restored))


def test_identical_lambda_factory_resume_next_two_updates_are_bitwise_equal(tmp_path):
    original = make_trainer()
    original.backend.scheduler.lr_lambdas = [_schedule_factory(0.5)]
    original.train_step()
    original.backend.save(tmp_path / "policy", original)
    expected = [original.train_step(), original.train_step()]
    snapshot = _contract_snapshot(original)
    restored = make_trainer()
    restored.backend.scheduler.lr_lambdas = [_schedule_factory(0.5)]
    restored.backend.load(tmp_path / "policy", restored)
    assert [restored.train_step(), restored.train_step()] == expected
    tree_equal(snapshot, _contract_snapshot(restored))
