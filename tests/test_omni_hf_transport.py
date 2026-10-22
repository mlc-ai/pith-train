"""Real CPU/Gloo sharded publication and sole-worker HF DP/PP continuation.

The canonical parameter container is a transport fixture, not a CUDA pipeline.
HF sampling and processor/reward/cursor semantics are real.
"""

import copy
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from pithtrain.modules.omni_hf_transport import (
    DistributedHFThinkerRollout,
    _ownership,
    gather_native_policy,
    initialize_native_policy,
    publish_native_policy,
)
from pithtrain.tasks.omni_hf_rollout import (
    ExactAnswerReward,
    HFCanonicalPolicyAdapter,
    LocalHFRolloutCfg,
    _canonical,
    _new_hf,
)
from pithtrain.tasks.omni_rl import RolloutRequest, _cpu_tree, _weights_digest
from tests.test_omni_hf_rollout import inputs, raw_config
from tests.test_omni_rl import tree_equal


def container(values):
    model = torch.nn.Module()
    for name, value in values.items():
        owner = model
        *parts, leaf = name.split(".")
        for part in parts:
            if part not in owner._modules:
                owner.add_module(part, torch.nn.Module())
            owner = owner._modules[part]
        owner.register_parameter(leaf, torch.nn.Parameter(value))
    return model


def run_transport(rank, world, init, directory, mode):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    policy, config = _new_hf(raw_config(), "text")
    reference = _canonical(_cpu_tree(policy.state_dict()), config, "text")
    adapter = HFCanonicalPolicyAdapter(
        policy.requires_grad_(True), torch.optim.AdamW(policy.parameters())
    )
    semantics = adapter.model_semantics()
    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=world, timeout=timedelta(seconds=120)
    )
    active = restored = None
    try:
        if mode == "dp":
            mesh = init_device_mesh("cpu", (world,))
            values = {
                name: distribute_tensor(value.clone(), mesh, [Shard(0)])
                for name, value in reference.items()
            }
            dp_rank, pp_rank, pp_size = rank, 0, 1
        else:
            values = {
                name: value.clone()
                for index, (name, value) in enumerate(reference.items())
                if index % world == rank
            }
            dp_rank, pp_rank, pp_size = 0, rank, world
        model = container(values)
        opts = LocalHFRolloutCfg(
            max_requests=2, group_size=2 * pp_size, max_sequence_length=10, max_new_tokens=2
        )
        initialize_native_policy(
            model,
            reference if rank == 0 else None,
            config,
            opts,
            group=dist.group.WORLD,
            chunk_bytes=32,
        )
        # All ranks reject the same explicit memory admission before weight transfer.
        from dataclasses import replace

        with pytest.raises(ValueError, match="host weight residency"):
            gather_native_policy(
                model,
                config,
                replace(opts, max_host_weight_bytes=1),
                group=dist.group.WORLD,
                chunk_bytes=32,
            )
        gathered = gather_native_policy(model, config, opts, group=dist.group.WORLD, chunk_bytes=32)
        if rank == 0:
            tree_equal(gathered, reference)
        prompts, tokens = inputs(Path(directory) / f"{mode}-{rank}")
        reward = ExactAnswerReward(prompts, tokens, dp_rank=dp_rank, dp_size=world // pp_size)

        def construct(reward_config=None):
            return DistributedHFThinkerRollout(
                opts,
                group=dist.group.WORLD,
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                pp_size=pp_size,
                model_config=raw_config(),
                prompts=prompts,
                reference=reference if rank == 0 else None,
                eos_token_id=1,
                reward_config=reward.config if reward_config is None else reward_config,
            )

        active = construct()
        snapshot0 = publish_native_policy(
            model, config, opts, dist.group.WORLD, 0, active.contract, semantics
        )
        assert (
            snapshot0.payload is None
            if rank != 0
            else _weights_digest(snapshot0.payload) == _weights_digest(reference)
        )
        active.synchronize(snapshot0)

        def request(snapshot, index):
            return RolloutRequest(
                snapshot.identity,
                index,
                index * opts.group_size,
                opts.seed + index,
                opts.group_size,
                opts.group_size,
                opts.max_sequence_length,
                dp_rank,
                pp_rank,
                pp_rank == 0,
            )

        first = active.generate(request(snapshot0, 0))
        reward.score(request(snapshot0, 0), first)
        state = copy.deepcopy(active.state_dict())
        owners = [None] * world
        dist.all_gather_object(owners, active.worker is not None)
        assert owners == [True] + [False] * (world - 1)
        with torch.no_grad():
            for value in model.parameters():
                value.add_(0.001)
        snapshot1 = publish_native_policy(
            model, config, opts, dist.group.WORLD, 1, active.contract, semantics
        )
        assert snapshot1.identity.weights_sha256 != snapshot0.identity.weights_sha256
        active.synchronize(snapshot1)
        expected = active.generate(request(snapshot1, 1))
        expected_state = copy.deepcopy(active.state_dict())
        active.close()
        restored = construct()
        restored.load_state_dict(state)
        restored.synchronize(snapshot1)
        actual = restored.generate(request(snapshot1, 1))
        for name in (
            "input_ids",
            "sampled_token_ids",
            "token_mask",
            "old_logprobs",
            "reference_logprobs",
        ):
            tree_equal(getattr(actual, name), getattr(expected, name))
        tree_equal(restored.state_dict(), expected_state)
        peers = [None] * world
        dist.all_gather_object(peers, dict(ids=actual.sample_ids, tokens=actual.sampled_token_ids))
        if mode == "pp":
            tree_equal(peers[0], peers[1])
        else:
            assert peers[0]["ids"] != peers[1]["ids"]
        if mode == "dp":
            # Independent reward config deliberately has no same-file source hash.
            # This extends the existing real-Gloo case without another worker run.
            from pithtrain.tasks import omni_hf_rollout as provider

            external_reward = {"kind": "external_reward_fixture"}
            independent = construct(external_reward)
            saved = independent.state_dict()
            assert saved["semantic"]["binding"]["execution"] == provider.execution_identity()
            assert (
                provider.worker_contract(raw_config(), opts)["execution"]
                == provider.execution_identity()
            )
            original = provider.implementation_contract

            def changed_engine(value):
                identity = original(value)
                if value is provider._Engine:
                    identity["source_sha256"] = "0" * 64
                return identity

            with patch.object(provider, "implementation_contract", changed_engine):
                changed = construct(external_reward)
                with pytest.raises(ValueError, match="resume semantics"):
                    changed.load_state_dict(saved)
                assert changed.worker is None
            independent.close()
        assert not torch.cuda.is_initialized()
    finally:
        if active is not None:
            active.close()
        if restored is not None:
            restored.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["dp", "pp"])
def test_two_rank_actual_shards_single_hf_worker_and_fresh_provider_resume(tmp_path, mode):
    mp.spawn(
        run_transport,
        args=(2, f"file://{tmp_path / 'gloo'}", str(tmp_path), mode),
        nprocs=2,
        join=True,
    )


def test_metadata_owner_selection_and_replica_drift_fail_closed():
    first = dict(name="weight", shape=[4], offset=0, count=2, sha256="a")
    second = dict(name="weight", shape=[4], offset=2, count=2, sha256="b")
    assert _ownership([[first], [first, second]], {"weight": (4,)}) == [
        ("weight", 0, 2, 0),
        ("weight", 2, 2, 1),
    ]
    with pytest.raises(ValueError, match="replica weights differ"):
        _ownership([[first], [{**first, "sha256": "different"}, second]], {"weight": (4,)})
    with pytest.raises(ValueError, match="Incomplete"):
        _ownership([[first]], {"weight": (4,)})


def run_slow_phase_control(rank, init):
    """Real Gloo; simulated slow worker control only, not HF/model acceptance."""
    from pithtrain.modules import omni_hf_transport as transport
    from pithtrain.modules.resume_semantics import seal
    from pithtrain.tasks import omni_hf_rollout as provider
    from pithtrain.tasks.omni_rl import PolicyIdentity, PolicySnapshot, RolloutContract

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=2, timeout=timedelta(seconds=5)
    )
    active = None
    try:
        reference = {"weight": torch.ones(1)}
        digest = _weights_digest(reference)
        contract = RolloutContract("0" * 64, "0" * 64, "0" * 64, digest, "0" * 64)
        semantics = seal({"fixture": "control-flow-only"})
        identity = PolicyIdentity(0, digest, contract, semantics["sha256"])
        snapshot = PolicySnapshot(identity, reference if rank == 0 else None, semantics)
        receipt = dict(
            installed_weights_sha256=digest,
            reference_weights_sha256=digest,
            worker_contract={"fixture": "control-flow-only"},
        )
        calls = dict(start=0, install=0, close=0)

        class SlowWorker:
            fail = False

            def __init__(self, *args):
                calls["start"] += 1
                time.sleep(3)

            def call(self, command, payload):
                assert command == "synchronize"
                calls["install"] += 1
                if self.fail:
                    raise TimeoutError("controlled install failure; no retry")
                time.sleep(3)
                return receipt

            def close(self):
                calls["close"] += 1

        active = DistributedHFThinkerRollout.__new__(DistributedHFThinkerRollout)
        active.cfg = LocalHFRolloutCfg(timeout_seconds=4)
        active.group = dist.group.WORLD
        active.prompts = SimpleNamespace(validate=lambda: None)
        active.raw = {}
        active.reference = reference if rank == 0 else None
        active.contract = contract
        active.failed = False
        active.worker = None
        with (
            patch.object(provider, "_WorkerProcess", SlowWorker),
            patch.object(
                transport, "validate_policy_semantics", lambda *args: receipt["worker_contract"]
            ),
        ):
            # Each 3s phase fits the 4s worker budget/5s collective timeout;
            # the former combined 6s phase times out the follower.
            assert active.synchronize(snapshot) == identity
            if rank == 0:
                active.worker.fail = True
            with pytest.raises(ValueError, match="controlled install failure"):
                active.synchronize(snapshot)
            assert active.failed and active.worker is None
        peers = [None, None]
        dist.all_gather_object(peers, calls)
        assert peers == [dict(start=1, install=2, close=1), dict(start=0, install=0, close=0)]
        assert not torch.cuda.is_initialized()
    finally:
        if active is not None:
            active.close()
        dist.destroy_process_group()


def test_startup_and_install_have_separate_gloo_deadlines_and_failure_cleanup(tmp_path):
    mp.spawn(
        run_slow_phase_control,
        args=(f"file://{tmp_path / 'slow-phases'}",),
        nprocs=2,
        join=True,
    )
