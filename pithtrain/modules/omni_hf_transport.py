"""CPU/Gloo transport for the existing native CP1/HSDP1 Omni GRPO adapter.

Model payloads move as bounded CPU tensor chunks, never NCCL object collectives.
Canonical metadata chooses one owner for duplicate replicas before data transfer.
"""

import copy
import hashlib
import importlib.metadata
import io
import json
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from pithtrain.modules.resume_semantics import implementation_contract
from pithtrain.tasks.omni_hf_rollout import (
    _check_weights,
    _config,
    _converter,
    execution_identity,
    validate_native_runtime,
    validate_policy_semantics,
    weight_memory_plan,
)
from pithtrain.tasks.omni_hf_rollout import (
    _stream_weights_digest as _weights_digest,
)
from pithtrain.tasks.omni_rl import (
    PolicyIdentity,
    PolicySnapshot,
    RolloutBatch,
    RolloutContract,
    fingerprint,
)


def _gloo(group):
    if (
        not dist.is_initialized()
        or dist.get_backend(group) != "gloo"
        or dist.get_world_size(group) != dist.get_world_size()
    ):
        raise ValueError("Explicit all-rank CPU/Gloo transport group required")


def _agree_error(error, group):
    rows = [None] * dist.get_world_size(group)
    dist.all_gather_object(rows, None if error is None else str(error)[:2048], group=group)
    if any(rows):
        raise ValueError(f"Local HF transport preflight failed: {rows}")


def _broadcast(value, group, limit, *, chunk_bytes=4 << 20):
    """Bounded CPU-only response transport; trusted local ranks, weights_only load."""
    rank = dist.get_rank()
    raw, error = None, None
    if rank == 0:
        try:
            stream = io.BytesIO()
            torch.save(value, stream)
            raw = stream.getvalue()
            if len(raw) > limit:
                raise ValueError("Response exceeds CPU transport byte bound")
        except Exception as exc:
            error = exc
    _agree_error(error, group)
    size = torch.tensor([len(raw) if rank == 0 else 0], dtype=torch.int64)
    dist.broadcast(size, src=0, group=group)
    if not 0 < size.item() <= limit:
        raise ValueError("Invalid bounded response length")
    output = bytearray() if rank != 0 else None
    for begin in range(0, size.item(), chunk_bytes):
        count = min(chunk_bytes, size.item() - begin)
        tensor = (
            torch.frombuffer(bytearray(raw[begin : begin + count]), dtype=torch.uint8)
            if rank == 0
            else torch.empty(count, dtype=torch.uint8)
        )
        dist.broadcast(tensor, src=0, group=group)
        if rank != 0:
            output.extend(tensor.numpy().tobytes())
    return (
        value
        if rank == 0
        else torch.load(io.BytesIO(output), weights_only=True, map_location="cpu")
    )


def _leader(function, group, limit):
    result = None
    if dist.get_rank() == 0:
        try:
            result = (None, function())
        except Exception as error:
            result = (f"{type(error).__name__}: {error}", None)
    error, value = _broadcast(result, group, limit)
    if error is not None:
        raise ValueError(f"Local HF leader failed: {error}")
    return value


def _pieces(model, config, cfg, *, hashes, chunk_elements):
    from pithtrain.modules.checkpoint import to_canonical_model

    weight_memory_plan(config, cfg)
    expected = _converter(cfg.scope).parameter_shapes(config)
    if sum(torch.Size(shape).numel() * 4 for shape in expected.values()) > cfg.max_weight_bytes:
        raise ValueError("Canonical export exceeds reviewed host weight bound")
    canonical = to_canonical_model(dict(model.named_parameters()), model)
    values, rows = {}, []
    for name, value in sorted(canonical.items()):
        if name not in expected or tuple(value.shape) != expected[name]:
            raise ValueError(f"Unexpected native canonical shape/key: {name}")
        offset = 0
        if isinstance(value, DTensor):
            if value.device_mesh.ndim != 1 or value.placements != (Shard(0),):
                raise ValueError("Only native HSDP1 one-dimensional row sharding is supported")
            size, row_offset = Shard.local_shard_size_and_offset(
                value.shape[0], value.device_mesh.size(), value.device_mesh.get_local_rank()
            )
            width = torch.Size(value.shape[1:]).numel()
            offset = row_offset * width
            local = value.to_local().detach()
            if local.numel() != size * width:
                raise ValueError("Native local shard does not match declared DTensor placement")
        else:
            local = value.detach()
        if local.dtype != torch.float32 or not local.is_contiguous():
            raise ValueError("Native publication requires contiguous FP32 master local views")
        flat = local.view(-1)
        digest = hashlib.sha256()
        if hashes:
            for begin in range(0, flat.numel(), chunk_elements):
                chunk = flat[begin : begin + chunk_elements].cpu()
                if not torch.isfinite(chunk).all():
                    raise ValueError("Nonfinite native policy publication")
                digest.update(chunk.numpy().tobytes())
        rows.append(
            dict(
                name=name,
                shape=list(expected[name]),
                offset=offset,
                count=flat.numel(),
                sha256=digest.hexdigest() if hashes else None,
            )
        )
        values[name] = flat
    if len(json.dumps(rows)) > 16 << 20:
        raise ValueError("Canonical shard metadata exceeds 16 MiB per-rank limit")
    return values, rows, expected


def _ownership(rows, expected):
    unique = {}
    for rank, parts in enumerate(rows):
        for row in parts:
            name, offset, count = row["name"], row["offset"], row["count"]
            if (
                name not in expected
                or tuple(row["shape"]) != expected[name]
                or min(offset, count) < 0
            ):
                raise ValueError("Foreign/malformed canonical shard metadata")
            if count == 0:
                continue
            key = (name, offset, count)
            if key in unique and unique[key][1] != row["sha256"]:
                raise ValueError("Duplicate native replica weights differ")
            unique.setdefault(key, (rank, row["sha256"]))
    result = []
    for name, shape in sorted(expected.items()):
        cursor = 0
        parts = sorted(
            (offset, count, owner)
            for (key, offset, count), (owner, _) in unique.items()
            if key == name
        )
        for offset, count, owner in parts:
            if offset != cursor:
                raise ValueError(f"Missing/overlapping canonical shard: {name}")
            result.append((name, offset, count, owner))
            cursor += count
        if cursor != torch.Size(shape).numel():
            raise ValueError(f"Incomplete native canonical coverage: {name}")
    return result


def gather_native_policy(model, config, cfg, *, group, chunk_bytes=4 << 20):
    """Reconstruct only on rank zero; duplicate replicas send no weight payload."""
    _gloo(group)
    if not 4 <= chunk_bytes <= 16 << 20 or chunk_bytes % 4:
        raise ValueError("FP32 transport chunks must be 4-byte aligned and <=16 MiB")
    values, local, expected, error = None, None, None, None
    try:
        values, local, expected = _pieces(
            model, config, cfg, hashes=True, chunk_elements=chunk_bytes // 4
        )
    except Exception as exc:
        error = exc
    _agree_error(error, group)
    all_rows = [None] * dist.get_world_size(group)
    dist.all_gather_object(all_rows, local, group=group)
    owners = _ownership(all_rows, expected)  # Identical metadata on every rank.
    rank = dist.get_rank()
    result = (
        {
            name: torch.empty(shape, dtype=torch.float32, device="cpu")
            for name, shape in expected.items()
        }
        if rank == 0
        else None
    )
    transfer_error = None
    for name, offset, count, owner in owners:
        digest = hashlib.sha256()
        for begin in range(0, count, chunk_bytes // 4):
            size = min(chunk_bytes // 4, count - begin)
            if rank == owner:
                piece = values[name][begin : begin + size].cpu()
                if owner != 0:
                    dist.send(piece, dst=0, group=group)
            if rank == 0:
                if owner != 0:
                    piece = torch.empty(size, dtype=torch.float32)
                    dist.recv(piece, src=owner, group=group)
                result[name].view(-1)[offset + begin : offset + begin + size].copy_(piece)
                digest.update(piece.numpy().tobytes())
        if rank == 0:
            declared = next(
                row["sha256"]
                for row in all_rows[owner]
                if row["name"] == name and row["offset"] == offset and row["count"] == count
            )
            if digest.hexdigest() != declared:
                transfer_error = ValueError("Native publication changed during bounded transfer")
    _agree_error(transfer_error, group)
    return result


def initialize_native_policy(model, reference, config, cfg, *, group, chunk_bytes=4 << 20):
    """Copy leader's fixed reference into exact native local views in bounded chunks.

    This is model initialization only, never optimizer/RNG resume. No nonleader
    materializes a full canonical model. Failed initialization cannot continue.
    """
    _gloo(group)
    rank, values, rows, expected, error = dist.get_rank(), None, None, None, None
    try:
        values, rows, expected = _pieces(
            model, config, cfg, hashes=False, chunk_elements=chunk_bytes // 4
        )
        if rank == 0:
            _check_weights(reference, config, cfg)
    except Exception as exc:
        error = exc
    _agree_error(error, group)
    all_rows = [None] * dist.get_world_size(group)
    dist.all_gather_object(all_rows, rows, group=group)
    _ownership(all_rows, expected)
    lookup = {row["name"]: row for row in rows}
    with torch.no_grad():
        for name, shape in sorted(expected.items()):
            for begin in range(0, torch.Size(shape).numel(), chunk_bytes // 4):
                count = min(chunk_bytes // 4, torch.Size(shape).numel() - begin)
                piece = (
                    reference[name].view(-1)[begin : begin + count].clone()
                    if rank == 0
                    else torch.empty(count, dtype=torch.float32)
                )
                dist.broadcast(piece, src=0, group=group)
                local = lookup.get(name)
                if local is not None:
                    lo, hi = (
                        max(begin, local["offset"]),
                        min(begin + count, local["offset"] + local["count"]),
                    )
                    if hi > lo:
                        view = values[name][lo - local["offset"] : hi - local["offset"]]
                        view.copy_(piece[lo - begin : hi - begin])
                        if not torch.equal(view.cpu(), piece[lo - begin : hi - begin]):
                            error = ValueError("Native initialization readback differs")
                _agree_error(error, group)


def publish_native_policy(model, config, cfg, group, version, contract, semantics):
    canonical = gather_native_policy(model, config, cfg, group=group)
    digest = _leader(lambda: _weights_digest(canonical), group, 1 << 20)
    return PolicySnapshot(
        PolicyIdentity(version, digest, contract, semantics["sha256"]), canonical, semantics
    )


class DistributedHFThinkerRollout:
    """One HF worker on rank zero; one prompt stream/RNG per DP coordinate.

    CP1 and HSDP1 are explicit requirements. PP copies receive identical tokens,
    probability/position metadata and rewards; encoder payload is kept on PP0.
    """

    def __init__(
        self,
        cfg,
        *,
        group,
        dp_rank,
        pp_rank,
        pp_size,
        model_config,
        prompts,
        reference,
        eos_token_id,
        reward_config,
        native_model=None,
    ):
        if importlib.metadata.version("transformers") != "5.17.0":
            raise ValueError("Distributed HF provider requires Transformers 5.17.0")
        _gloo(group)
        cfg.validate()
        self.cfg, self.group, self.prompts = cfg, group, prompts
        self.native_model = native_model
        self.dp_rank, self.pp_rank, self.pp_size = dp_rank, pp_rank, pp_size
        self.dp_size = dist.get_world_size() // pp_size
        coordinates = [None] * dist.get_world_size()
        dist.all_gather_object(coordinates, (dp_rank, pp_rank), group=group)
        if (
            sorted(coordinates) != [(dp, pp) for dp in range(self.dp_size) for pp in range(pp_size)]
            or cfg.group_size < 2 * pp_size
        ):
            raise ValueError(
                "Invalid native DP/PP coordinates or insufficient pipeline microbatches"
            )
        binding = dict(
            cfg=asdict(cfg),
            config=model_config,
            prompts=prompts.sha256,
            eos=eos_token_id,
            reward=fingerprint(reward_config),
            execution=execution_identity(),
        )
        bindings = [None] * dist.get_world_size()
        dist.all_gather_object(bindings, fingerprint(binding), group=group)
        if len(set(bindings)) != 1:
            raise ValueError("Ranks disagree on HF rollout data/config/reward/implementation")

        def construct():
            config = _config(model_config, cfg.scope)
            _check_weights(reference, config, cfg)
            vocab = config.vocab_size if cfg.scope == "text" else config.text_config.vocab_size
            if type(eos_token_id) is not int or not 0 <= eos_token_id < vocab:
                raise ValueError("Explicit tokenizer EOS must fit the policy vocabulary")
            if any(
                p.inputs["input_ids"].min() < 0 or p.inputs["input_ids"].max() >= vocab
                for p in prompts.prompts
            ):
                raise ValueError("Prepared prompt IDs exceed the model vocabulary")
            contract = RolloutContract(
                fingerprint(prompts.identity["provenance"]),
                prompts.sha256,
                fingerprint(
                    dict(
                        config=model_config, input_format="document-prefix-v1", hf_attention="eager"
                    )
                ),
                _weights_digest(reference),
                fingerprint(reward_config),
            )
            return dict(contract=asdict(contract), binding=binding)

        identity = _leader(construct, group, cfg.max_result_bytes)
        self.contract = RolloutContract(**identity["contract"])
        self.raw, self.config, self.reference, self.eos = (
            copy.deepcopy(model_config),
            _config(model_config, cfg.scope),
            reference,
            eos_token_id,
        )
        self.semantic = dict(
            schema=1,
            binding=identity["binding"],
            contract=identity["contract"],
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            dp_size=self.dp_size,
            pp_size=pp_size,
            implementation=implementation_contract(type(self)),
        )
        if cfg.compute_dtype == "bfloat16":
            error = None
            try:
                if native_model is None:
                    raise ValueError("Distributed BF16 provider requires the actual native policy")
                self.semantic["native_runtime"] = validate_native_runtime(
                    native_model, self.raw, cfg
                )
            except Exception as exc:
                error = exc
            _agree_error(error, group)
        self.requests = self.cursor = 0
        self.rng = torch.Generator().manual_seed(cfg.seed + dp_rank * 100003).get_state()
        self.worker = self.identity = self.installation_receipt = None
        self.failed = False

    def synchronize(self, snapshot):
        from pithtrain.modules.resume_semantics import seal
        from pithtrain.tasks.omni_hf_rollout import _WorkerProcess

        error = None
        try:
            self.prompts.validate()
            self._validate_native()
            snapshot.identity.validate()
            if (
                self.failed
                or snapshot.identity.contract != self.contract
                or seal(snapshot.semantics["identity"]) != snapshot.semantics
                or snapshot.identity.semantics_sha256 != snapshot.semantics["sha256"]
            ):
                raise ValueError("Distributed snapshot/provider semantics differ")
            expected_worker = validate_policy_semantics(snapshot.semantics, self.raw, self.cfg)
        except Exception as exc:
            error = exc
        _agree_error(error, self.group)

        def start():
            if (
                _weights_digest(self.reference) != self.contract.reference_policy_sha256
                or _weights_digest(snapshot.payload) != snapshot.identity.weights_sha256
            ):
                raise ValueError("Fixed reference/published policy digest differs")
            if self.worker is None:
                self.worker = _WorkerProcess(self.raw, self.cfg, self.reference)

        try:
            # Followers acknowledge startup before entering the installation wait.
            # Each collective phase covers at most one worker operation deadline.
            _leader(start, self.group, 1 << 20)
            receipt = _leader(
                lambda: self.worker.call("synchronize", snapshot.payload), self.group, 1 << 20
            )
            if receipt != dict(
                installed_weights_sha256=snapshot.identity.weights_sha256,
                reference_weights_sha256=self.contract.reference_policy_sha256,
                worker_contract=expected_worker,
            ):
                raise ValueError("Installed distributed policy/reference differs")
            self.identity, self.installation_receipt = snapshot.identity, receipt
            return self.identity
        except BaseException:
            self.failed = True
            self.close()
            raise

    def generate(self, request):
        error = None
        try:
            self.prompts.validate()
            if (
                self.failed
                or request.identity != self.identity
                or request.request_index != self.requests
                or request.data_cursor != self.cursor
                or request.dp_rank != self.dp_rank
                or request.pp_rank != self.pp_rank
                or request.rollout_leader != (self.pp_rank == 0)
                or request.seed != self.cfg.seed + self.requests
                or request.group_size != self.cfg.group_size
                or request.max_samples < self.cfg.group_size
                or request.max_sequence_length != self.cfg.max_sequence_length
                or self.requests >= self.cfg.max_requests
            ):
                raise ValueError("Distributed rollout request identity/cursor/bounds differ")
        except Exception as exc:
            error = exc
        _agree_error(error, self.group)
        states = [None] * dist.get_world_size()
        dist.all_gather_object(
            states,
            dict(dp=self.dp_rank, request=self.requests, cursor=self.cursor, rng=self.rng),
            group=self.group,
        )
        by_dp = {}
        for row in states:
            previous = by_dp.setdefault(row["dp"], row)
            if (
                row["request"] != self.requests
                or row["cursor"] != self.cursor
                or not torch.equal(row["rng"], previous["rng"])
            ):
                raise ValueError("Pipeline copies have inconsistent provider cursor/RNG")
        try:
            for dp in range(self.dp_size):
                index = self.requests * self.dp_size + dp
                value = _leader(
                    lambda: self.worker.call(
                        "generate",
                        dict(
                            prompt=self.prompts.payload(index),
                            rng=by_dp[dp]["rng"],
                            eos_token_id=self.eos,
                        ),
                    ),
                    self.group,
                    self.cfg.max_result_bytes,
                )
                if dp == self.dp_rank:
                    self.rng = value.pop("rng")
                    name = self.prompts.prompts[index].sample_id
                    if self.pp_rank != 0:
                        value["media_inputs"] = None
                    result = RolloutBatch(
                        request.identity,
                        self.requests,
                        group_ids=(f"dp{dp}:{name}",) * self.cfg.group_size,
                        sample_ids=tuple(
                            f"{self.requests}:dp{dp}:{name}:{i}" for i in range(self.cfg.group_size)
                        ),
                        **value,
                    )
            self.requests += 1
            self.cursor += self.cfg.group_size
            return result
        except BaseException:
            self.failed = True
            self.close()
            raise

    def state_dict(self):
        if self.failed:
            raise ValueError("Cannot checkpoint failed distributed HF provider")
        self.prompts.validate()
        if (
            dist.get_rank() == 0
            and _weights_digest(self.reference) != self.contract.reference_policy_sha256
        ):
            raise ValueError("Fixed distributed HF reference changed")
        return dict(
            schema=1,
            semantic=copy.deepcopy(self.semantic),
            requests=self.requests,
            cursor=self.cursor,
            rng=self.rng.clone(),
        )

    def validate_state_dict(self, state):
        self.prompts.validate()
        self._validate_native()
        if (
            set(state) != {"schema", "semantic", "requests", "cursor", "rng"}
            or state["schema"] != 1
            or state["semantic"] != self.semantic
        ):
            raise ValueError("Distributed HF resume semantics differ")
        if (
            type(state["requests"]) is not int
            or not 0
            <= state["requests"]
            <= min(self.cfg.max_requests, len(self.prompts.prompts) // self.dp_size)
            or type(state["cursor"]) is not int
            or state["cursor"] != state["requests"] * self.cfg.group_size
        ):
            raise ValueError("Distributed HF resume cursor differs")
        if (
            not isinstance(state["rng"], torch.Tensor)
            or state["rng"].dtype != torch.uint8
            or state["rng"].device.type != "cpu"
        ):
            raise ValueError("Distributed HF RNG is invalid")
        torch.Generator().set_state(state["rng"])
        if (
            dist.get_rank() == 0
            and _weights_digest(self.reference) != self.contract.reference_policy_sha256
        ):
            raise ValueError("Fixed distributed HF reference changed")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.close()
        self.requests, self.cursor, self.rng = (
            state["requests"],
            state["cursor"],
            state["rng"].clone(),
        )
        self.identity = self.installation_receipt = None
        self.failed = False

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def _validate_native(self):
        if self.cfg.compute_dtype == "bfloat16":
            actual = validate_native_runtime(self.native_model, self.raw, self.cfg)
            if actual != self.semantic["native_runtime"]:
                raise ValueError("Native runtime model/forward numerical contract changed")
