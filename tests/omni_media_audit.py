"""Test-only observations of real media reads, broadcasts and pipeline calls.

Wrappers forward their original arguments/results; they inspect tensor metadata
and identity without copying, casting or computing on the tensors.
"""

from collections import Counter
from unittest.mock import patch

import torch

PAYLOADS = {"pixel_values", "input_features", "pixel_values_videos"}


def no_tensors(value):
    assert not isinstance(value, torch.Tensor), "A media tensor entered pickled metadata"
    if isinstance(value, dict):
        for item in value.values():
            no_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            no_tensors(item)


def identity(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


class MediaAudit:
    def __init__(self, pp_rank, pp_size, pp_group):
        self.pp_rank, self.pp_size, self.pp_group = pp_rank, pp_size, pp_group
        self.counts, self.dispatch = Counter(), Counter()
        self.in_batch = self.in_object_broadcast = False
        self.specs = None
        self.bound = None
        self.active_dispatch = None

    def wrap_dispatch(self, original, path):
        def wrapped(pipeline, phase, *args, **kwargs):
            assert self.active_dispatch is None
            assert not pipeline.forward_only
            self.active_dispatch = (
                path,
                pipeline.module[phase].stage_index,
                pipeline.current_f_chunk_id[phase],
            )
            before = sum(self.dispatch.values())
            try:
                result = original(pipeline, phase, *args, **kwargs)
                assert sum(self.dispatch.values()) == before + 1, (
                    "Missing or duplicate observed dispatch"
                )
                return result
            finally:
                self.active_dispatch = None

        return wrapped

    def install_dispatch(self, stack, pipeline_module):
        for name, path in (
            ("_forward_compute_chunk", "normal"),
            ("_forward_backward_compute_chunk", "overlap"),
        ):
            original = getattr(pipeline_module.DualPipeV, name)
            stack.enter_context(
                patch.object(pipeline_module.DualPipeV, name, self.wrap_dispatch(original, path))
            )
        original = pipeline_module.overlapped_forward_backward

        def overlap(module, *args, **kwargs):
            self.observe_dispatch(module.stage_index, kwargs["model_context"], "overlap")
            return original(module, *args, **kwargs)

        stack.enter_context(patch.object(pipeline_module, "overlapped_forward_backward", overlap))

    def install(self, stack, data_class, prepare, reader, dist):
        def owner_only(name, original):
            def wrapped(*args, **kwargs):
                assert self.pp_rank == 0, f"Nonowner called {name}"
                self.counts[name] += 1
                return original(*args, **kwargs)

            return wrapped

        for module, name, counter in (
            (prepare, "verify_bundle", "verify_calls"),
            (prepare, "processor_for", "processor_calls"),
            (reader, "create_omni_dataloader", "loader_calls"),
        ):
            stack.enter_context(
                patch.object(module, name, owner_only(counter, getattr(module, name)))
            )

        original_share = data_class._share_metadata

        def share(data, read):
            assert not data.is_text and data.pp_rank == self.pp_rank

            def observed_read():
                assert self.pp_rank == 0, "Nonowner performed a media read"
                self.counts["owner_read_calls"] += 1
                if self.in_batch:
                    self.counts["payload_read_calls"] += 1
                return read()

            return original_share(data, observed_read)

        original_validate = data_class.validate_model

        def validate(data, *args, **kwargs):
            result = original_validate(data, *args, **kwargs)
            self.counts["model_validation_calls"] += 1
            return result

        stack.enter_context(patch.object(data_class, "validate_model", validate))
        original_batch = data_class._next_media_microbatch

        def batch(data, device):
            assert not self.in_batch
            self.in_batch, self.specs = True, None
            try:
                result = original_batch(data, device)
                assert self.pp_size == 1 or self.specs == [], "A shared tensor was not broadcast"
                self.counts["microbatches"] += 1
                return result
            except BaseException:
                self.counts["failed_microbatches"] += 1
                raise
            finally:
                self.in_batch, self.specs = False, None

        original_object = dist.broadcast_object_list

        def object_broadcast(message, *args, **kwargs):
            if self.pp_size == 1 or kwargs.get("group") is not self.pp_group:
                return original_object(message, *args, **kwargs)
            assert kwargs["src"] == dist.get_global_rank(self.pp_group, 0)
            no_tensors(message)
            self.in_object_broadcast = True
            try:
                result = original_object(message, *args, **kwargs)
            finally:
                self.in_object_broadcast = False
            no_tensors(message)
            value, error = message[0]
            self.counts["metadata_headers"] += 1
            if self.in_batch and error is None:
                _, specs = value
                assert isinstance(specs, list) and specs
                names = [name for name, _, _ in specs]
                assert len(set(names)) == len(names) and not PAYLOADS & set(names)
                self.specs = list(specs)
                self.counts["shared_specs"] += len(specs)
            return result

        original_tensor = dist.broadcast

        def tensor_broadcast(tensor, *args, **kwargs):
            if (
                self.in_batch
                and not self.in_object_broadcast
                and kwargs.get("group") is self.pp_group
            ):
                assert kwargs["src"] == dist.get_global_rank(self.pp_group, 0)
                assert self.specs, "Unexpected shared tensor broadcast"
                name, shape, dtype = self.specs.pop(0)
                assert (
                    name not in PAYLOADS
                    and tuple(tensor.shape) == tuple(shape)
                    and tensor.dtype == dtype
                )
                self.counts["tensor_broadcasts"] += 1
            return original_tensor(tensor, *args, **kwargs)

        stack.enter_context(patch.object(data_class, "_share_metadata", share))
        stack.enter_context(patch.object(data_class, "_next_media_microbatch", batch))
        stack.enter_context(patch.object(dist, "broadcast_object_list", object_broadcast))
        stack.enter_context(patch.object(dist, "broadcast", tensor_broadcast))

    def bind_batches(self, batches, stages):
        assert self.bound is None, "Previous pipeline step was not completed"
        assert set(stages) == {self.pp_rank, 2 * self.pp_size - 1 - self.pp_rank}
        self.stages = set(stages)
        self.bound = {}
        self.order = []
        self.seen = Counter()
        for batch in batches:
            key = identity(batch.model_context["input_ids"])
            assert key not in self.bound, "Ambiguous microbatch storage"
            self.bound[key] = batch
            self.order.append(key)

    def observe_dispatch(self, stage, context, path):
        assert path in {"normal", "overlap"} and stage in self.stages
        assert self.bound is not None and isinstance(context, dict)
        key = identity(context["input_ids"])
        assert key in self.bound, "Pipeline used another microbatch's context"
        assert self.active_dispatch is not None and self.active_dispatch[:2] == (path, stage)
        assert key == self.order[self.active_dispatch[2]], "Wrong phase/microbatch association"
        expected = self.bound[key].context_for_stage(stage)
        assert context.keys() == expected.keys(), "Wrong stage payload ownership"
        assert all(
            identity(context[name]) == identity(value) for name, value in expected.items()
        ), "Pipeline changed or exchanged context tensors"
        self.seen[(stage, key)] += 1
        self.dispatch[f"{path}.stage{stage}"] += 1

    def end_step(self):
        assert self.bound is not None
        assert self.seen == Counter(
            {(stage, key): 1 for stage in self.stages for key in self.bound}
        ), "Missing or repeated stage/microbatch dispatch"
        self.bound = None
        self.counts["completed_pipeline_steps"] += 1

    def finish_data(self, *, providers, validations, expected_batches):
        owner = int(self.pp_rank == 0)
        assert self.counts["failed_microbatches"] == 0
        assert self.counts["verify_calls"] == providers * owner
        assert self.counts["processor_calls"] == providers * owner
        assert self.counts["loader_calls"] >= providers * owner
        if not owner:
            assert self.counts["loader_calls"] == 0
        assert self.counts["model_validation_calls"] == validations
        assert self.counts["microbatches"] == expected_batches
        assert self.counts["payload_read_calls"] == expected_batches * owner
        assert (
            self.counts["owner_read_calls"] == (providers + validations + expected_batches) * owner
        )
        assert self.counts["metadata_headers"] == (
            providers + validations + expected_batches if self.pp_size > 1 else 0
        )
        assert self.counts["tensor_broadcasts"] == self.counts["shared_specs"]
        assert (self.counts["tensor_broadcasts"] > 0) == (self.pp_size > 1)
        return {
            name: self.counts[name]
            for name in (
                "verify_calls",
                "processor_calls",
                "loader_calls",
                "owner_read_calls",
                "payload_read_calls",
                "model_validation_calls",
                "metadata_headers",
                "shared_specs",
                "tensor_broadcasts",
                "microbatches",
                "failed_microbatches",
                "completed_pipeline_steps",
            )
        }

    def finish_dispatch(self, *, expected_steps):
        assert self.bound is None and self.counts["completed_pipeline_steps"] == expected_steps
        assert any(key.startswith("normal.") for key in self.dispatch)
        if self.pp_size > 1:
            assert any(key.startswith("overlap.") for key in self.dispatch)
        return dict(self.dispatch)
