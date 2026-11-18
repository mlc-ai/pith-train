"""Versioned task cursors for native transactional SFT/RL checkpoints.

Task implementations own pure validation of their provider/data state. Native
distributed_checkpoint owns the model, optimizer, scheduler and global RNG
transaction; this envelope does not replace that transaction with callbacks.
"""

import hashlib
import json
import math
import re

import torch

TASK_STATE_VERSION = 2
CONDITIONAL_OPTIMIZER_POLICY = "conditional_adamw_v1"
RNG_METADATA = {
    "version": 1,
    "global": "rank-shard-python-numpy-torch-v1",
    "provider": "task-state-cpu-tensors-v1",
}


def _portable(value):
    """Clone only weights_only-safe, finite task state without device collectives."""
    if type(value) is torch.Tensor:
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Task/provider checkpoint tensors must be dense CPU tensors")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError("Nonfinite task/provider checkpoint tensor")
        return value.detach().clone()
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("Task/provider state requires string dictionary keys")
        return {key: _portable(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_portable(item) for item in value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Unsupported or nonfinite task/provider checkpoint value")


class TaskCheckpointState:
    """Bind a concrete task/provider state to native distributed save/load.

    ``transaction_key`` counts committed scans/requests; ``update_key`` counts
    completed optimizer updates. ``data_key`` is a rank-local committed cursor.
    The task must expose state_dict(), pure validate_state_dict(), and
    load_state_dict() which can restore a previously captured state on rollback.
    Provider schema, RNG streams, data identity and policies belong in the
    task's versioned state and the stable rank-independent ``contract``.
    """

    def __init__(
        self,
        task,
        *,
        task_kind,
        task_schema,
        contract,
        transaction_key,
        update_key,
        data_key,
        policy_version_key=None,
        optimizer_policy=None,
        bounded_load=False,
    ):
        if not isinstance(task_kind, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", task_kind):
            raise ValueError("A named task kind is required")
        if type(task_schema) is not int or task_schema < 1 or type(contract) is not dict:
            raise ValueError("Explicit task schema and stable contract are required")
        keys = (transaction_key, update_key, data_key)
        if any(type(key) is not str or not key for key in keys) or len(set(keys)) != 3:
            raise ValueError("Three distinct task cursor keys are required")
        if policy_version_key is not None and (
            type(policy_version_key) is not str or not policy_version_key
        ):
            raise ValueError("Invalid provider policy-version key")
        if any(
            not callable(getattr(task, name, None))
            for name in ("state_dict", "validate_state_dict", "load_state_dict")
        ):
            raise TypeError("Concrete task must expose state, pure validation and restore")
        if optimizer_policy not in (None, CONDITIONAL_OPTIMIZER_POLICY):
            raise ValueError("Unsupported task optimizer checkpoint policy")
        if type(bounded_load) is not bool or (
            bounded_load and optimizer_policy != CONDITIONAL_OPTIMIZER_POLICY
        ):
            raise ValueError("Bounded task load requires the explicit conditional Adam policy")
        self.bounded_load = bounded_load
        self.optimizer_policy = optimizer_policy
        identity = dict(
            task_kind=task_kind,
            task_schema=task_schema,
            contract=contract,
            transaction_key=transaction_key,
            update_key=update_key,
            data_key=data_key,
            policy_version_key=policy_version_key,
        )
        if optimizer_policy is not None:
            identity["optimizer_policy"] = optimizer_policy
        if bounded_load:
            identity["load_policy"] = "local_shard_inplace_v1"
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.task, self.task_kind, self.task_schema = task, task_kind, task_schema
        self.fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        self.transaction_key, self.update_key, self.data_key = keys
        self.policy_version_key = policy_version_key

    def _cursors(self, state):
        result = dict(
            transaction=state[self.transaction_key],
            updates=state[self.update_key],
            data=state[self.data_key],
        )
        if any(type(value) is not int or value < 0 for value in result.values()):
            raise ValueError("Task cursors must be nonnegative integers, not booleans")
        if result["updates"] > result["transaction"]:
            raise ValueError("Task updates exceed committed transactions")
        return result

    def state_dict(self):
        state = _portable(self.task.state_dict())
        value = dict(
            version=TASK_STATE_VERSION,
            task_kind=self.task_kind,
            task_schema=self.task_schema,
            fingerprint=self.fingerprint,
            cursors=self._cursors(state),
            provider_policy_version=None
            if self.policy_version_key is None
            else state[self.policy_version_key],
            rng_metadata=dict(RNG_METADATA),
            state=state,
        )
        if self.optimizer_policy is not None:
            value["optimizer_policy"] = self.optimizer_policy
        if self.bounded_load:
            value["load_policy"] = "local_shard_inplace_v1"
        self.validate_state_dict(value)
        return value

    def validate_state_dict(self, value):
        fields = {
            "version",
            "task_kind",
            "task_schema",
            "fingerprint",
            "cursors",
            "provider_policy_version",
            "rng_metadata",
            "state",
        }
        if self.optimizer_policy is not None:
            fields.add("optimizer_policy")
        if self.bounded_load:
            fields.add("load_policy")
        if type(value) is not dict or set(value) != fields:
            raise ValueError("Incomplete version-2 task checkpoint envelope")
        if (
            type(value["version"]) is not int
            or value["version"] != TASK_STATE_VERSION
            or type(value["task_schema"]) is not int
            or value["task_schema"] != self.task_schema
            or value["task_kind"] != self.task_kind
            or value["fingerprint"] != self.fingerprint
            or value["rng_metadata"] != RNG_METADATA
            or type(value["rng_metadata"].get("version")) is not int
        ):
            raise ValueError("Task schema/contract/fingerprint/RNG metadata mismatch")
        if value.get("optimizer_policy") != self.optimizer_policy:
            raise ValueError("Task optimizer checkpoint policy mismatch")
        if value.get("load_policy") != ("local_shard_inplace_v1" if self.bounded_load else None):
            raise ValueError("Task load policy mismatch")
        state = _portable(value["state"])
        cursors = self._cursors(state)
        if (
            type(value["cursors"]) is not dict
            or set(value["cursors"]) != set(cursors)
            or any(type(value["cursors"][key]) is not int for key in cursors)
            or value["cursors"] != cursors
        ):
            raise ValueError("Envelope and task cursors disagree")
        policy = value["provider_policy_version"]
        if self.policy_version_key is None:
            if policy is not None:
                raise ValueError("Unexpected provider policy version")
        elif (
            type(policy) is not int
            or policy < 0
            or policy != cursors["updates"]
            or type(state[self.policy_version_key]) is not int
            or policy != state[self.policy_version_key]
        ):
            raise ValueError("Provider policy version/update cursor mismatch")
        self.task.validate_state_dict(state)

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        self.task.load_state_dict(_portable(value["state"]))

    def checkpoint_contract(self, value, *, include_cursors):
        self.validate_state_dict(value)
        identity = {
            key: value[key]
            for key in ("version", "task_kind", "task_schema", "fingerprint", "rng_metadata")
        }
        if self.optimizer_policy is not None:
            identity["optimizer_policy"] = self.optimizer_policy
        if self.bounded_load:
            identity["load_policy"] = "local_shard_inplace_v1"
        if include_cursors:
            identity.update(
                transaction=value["cursors"]["transaction"],
                updates=value["cursors"]["updates"],
                provider_policy_version=value["provider_policy_version"],
            )
        return identity
