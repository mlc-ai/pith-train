"""Real Qwen processor and CPU/Gloo checks of PP media ownership.

No CUDA/NCCL or native encoder execution is covered by these tests.
"""

import builtins
import json
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.qwen3_omni_data import create_omni_dataloader
from pithtrain.modules.training_data import PreparedBundleData
from tests import test_omni_training as fixtures

manifest = fixtures.manifest
training_bundle = fixtures.training_bundle
processor_config = fixtures.processor_config
single_cpu_thread = fixtures.single_cpu_thread
PAYLOADS = {"pixel_values", "pixel_values_videos", "input_features"}
MODES = [("image",), ("audio",), ("video",), ("text", "image", "audio", "video")]


def configs(root, kinds):
    cfg = DataCfg()
    cfg.dataset, cfg.format = root, "prepared_bundle"
    cfg.modalities, cfg.epoch_samples = kinds, 8
    tc = SimpleNamespace(global_batch_size=4, micro_batch_size=1, sequence_length=1024, seed=17)
    return cfg, tc


def assert_batches(batches, expected, owns_media):
    for actual, reference in zip(batches, expected, strict=True):
        assert actual.sample_ids == reference["sample_ids"]
        torch.testing.assert_close(actual.objective_inputs[0], reference["labels"], rtol=0, atol=0)
        all_inputs = reference["inputs"]
        common = {name: value for name, value in all_inputs.items() if name not in PAYLOADS}
        media = {name: value for name, value in all_inputs.items() if name in PAYLOADS}
        torch.testing.assert_close(actual.media_context, common, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.media_inputs or {}, media if owns_media else {}, rtol=0, atol=0
        )
        assert actual.model_inputs[0] is actual.media_context["input_ids"]
        assert not PAYLOADS & actual.context_for_stage(3).keys()
        if owns_media:
            torch.testing.assert_close(actual.context_for_stage(0), all_inputs, rtol=0, atol=0)


def _media_worker(rank, rendezvous, root, reference_file):
    from pithtrain.modules import omni_bundle, qwen3_omni_data

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).resolve().as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=90),
    )
    # PP=2, DP=2: PP sources are global ranks 0 AND 1, not always world rank 0.
    groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    pp_rank, dp_rank = rank // 2, rank % 2
    group = groups[dp_rank]
    kwargs = dict(dp_rank=dp_rank, dp_size=2, pp_rank=pp_rank, pp_size=2, pp_group=group)
    reference = torch.load(reference_file, weights_only=True)
    counters = dict(verify=0, processor=0, loader=0, headers=0, tensors=0)
    original_object_broadcast = dist.broadcast_object_list
    original_tensor_broadcast = dist.broadcast
    pending_specs = []

    def owner_only(name, function):
        def checked(*args, **kw):
            assert pp_rank == 0, f"Non-encoder PP rank called {name}"
            counters[name] += 1
            return function(*args, **kw)

        return checked

    def no_tensors(value):
        assert not isinstance(value, torch.Tensor), "A tensor was pickled into the metadata header"
        if isinstance(value, dict):
            for item in value.values():
                no_tensors(item)
        if isinstance(value, (tuple, list)):
            for item in value:
                no_tensors(item)

    def object_broadcast(message, **kw):
        assert not pending_specs, "Previous batch did not broadcast every shared tensor"
        assert kw["group"] is group and kw["src"] == dp_rank
        no_tensors(message)
        original_object_broadcast(message, **kw)
        no_tensors(message)
        value, error = message[0]
        if error is None and isinstance(value, tuple) and isinstance(value[0], tuple):
            _, specs = value
            assert not PAYLOADS & {name for name, _, _ in specs}
            assert specs and len({name for name, _, _ in specs}) == len(specs)
            pending_specs.extend(specs)
        counters["headers"] += 1

    def tensor_broadcast(tensor, **kw):
        assert kw["group"] is group and kw["src"] == dp_rank
        assert pending_specs, "Unexpected tensor broadcast outside the shared header"
        _, shape, dtype = pending_specs.pop(0)
        assert tuple(tensor.shape) == shape and tensor.dtype == dtype
        counters["tensors"] += 1
        return original_tensor_broadcast(tensor, **kw)

    with ExitStack() as stack:
        for module, name, counter in (
            (omni_bundle, "read_verified_bundle", "verify"),
            (qwen3_omni_data, "processor_for", "processor"),
            (qwen3_omni_data, "create_omni_dataloader", "loader"),
        ):
            stack.enter_context(
                patch.object(module, name, owner_only(counter, getattr(module, name)))
            )
        stack.enter_context(patch.object(dist, "broadcast_object_list", object_broadcast))
        stack.enter_context(patch.object(dist, "broadcast", tensor_broadcast))
        for mode, kinds in enumerate(MODES):
            cfg, tc = configs(root, kinds)
            data = PreparedBundleData(cfg, tc, **kwargs)
            supported = type("TestMediaModel", (), {"input_modalities": set(kinds)})
            data.validate_model(supported, SimpleNamespace(vocab_size=200000))
            rng = torch.get_rng_state().clone()
            first = data.get_batch(0, "cpu")
            assert_batches(first, reference[mode][dp_rank][0], pp_rank == 0)
            if pp_rank != 0:
                assert data._iterator is None and data._processor is None
            for step in (1, 2):
                assert_batches(
                    data.get_batch(step, "cpu"), reference[mode][dp_rank][step], pp_rank == 0
                )
            assert torch.equal(torch.get_rng_state(), rng), "Data distribution changed model RNG"

            restored = PreparedBundleData(cfg, tc, **kwargs)
            for step in (1, 2):
                assert_batches(
                    restored.get_batch(step, "cpu"), reference[mode][dp_rank][step], pp_rank == 0
                )

        cfg, tc = configs(root, ("image",))
        failed = PreparedBundleData(cfg, tc, **kwargs)

        def fail_read(step):
            raise ValueError("expected-reader-failure")

        failed._make_iterator = fail_read
        with pytest.raises(RuntimeError, match="expected-reader-failure"):
            failed.get_batch(0, "cpu")
        with pytest.raises(ValueError, match="process group does not match"):
            PreparedBundleData(cfg, tc, **dict(kwargs, pp_group=dist.group.WORLD))

        # The owner still checks EVERY payload hash. A corrupted media file
        # fails the whole PP group, including peers that never open the file.
        dist.barrier()
        if rank == 0:
            (Path(root) / "image.png").write_bytes(b"corrupt test image")
        dist.barrier()
        with pytest.raises(RuntimeError, match="Dataset hash mismatch"):
            PreparedBundleData(cfg, tc, **kwargs)

    assert not pending_specs
    assert all(counters[name] > 0 for name in ("headers", "tensors"))
    assert all((counters[name] > 0) == (pp_rank == 0) for name in ("verify", "processor", "loader"))
    assert not torch.cuda.is_initialized()
    dist.destroy_process_group()


def test_media_payloads_stay_on_owner_with_real_processor_and_gloo(
    training_bundle,
    processor_config,
    tmp_path,
    single_cpu_thread,
):
    recipe = json.loads((training_bundle / "bundle.json").read_text())["recipe"]
    reference = []
    for kinds in MODES:
        ranks = []
        for dp_rank in range(2):
            steps = []
            for step in range(3):
                loader = create_omni_dataloader(
                    training_bundle,
                    *processor_config,
                    modalities=kinds,
                    split="train",
                    batch_size=1,
                    num_samples=8,
                    weights={k: recipe["sampling_weights"][k] for k in kinds},
                    epoch=step // 2,
                    start_sample=(step * 4) % 8,
                    rank=dp_rank,
                    world_size=2,
                    seed=17,
                    batch_cfg=dict(recipe["batch"], max_length=1024),
                )
                iterator = iter(loader)
                batch = [next(iterator) for _ in range(2)]
                steps.append(
                    [
                        dict(sample_ids=b.sample_ids, inputs=b.model_inputs, labels=b.labels)
                        for b in batch
                    ]
                )
            ranks.append(steps)
        reference.append(ranks)
    path = tmp_path / "reference.pt"
    torch.save(reference, path)
    mp.spawn(
        _media_worker,
        args=(str(tmp_path / "rendezvous"), str(training_bundle), str(path)),
        nprocs=4,
        join=True,
    )


def test_stage_context_keeps_shared_inputs_and_rejects_overwrite():
    tokens, pixels = torch.ones(1, 3, dtype=torch.long), torch.ones(4, 3)
    shared = {"input_ids": tokens, "image_grid_thw": torch.tensor([[1, 2, 2]])}
    micro = Microbatch(
        model_inputs=(tokens,),
        objective_inputs=(tokens,),
        cu_seqlens=None,
        media_context=shared,
        media_inputs={"pixel_values": pixels},
    )
    assert micro.context_for_stage(0)["pixel_values"] is pixels
    assert micro.context_for_stage(1) is shared
    assert micro.context_for_stage(3) is shared
    assert "pixel_values" not in shared
    micro.media_inputs = {"input_ids": pixels}
    with pytest.raises(ValueError, match="overwrite"):
        micro.context_for_stage(0)
    plain = Microbatch(model_inputs=(tokens,), objective_inputs=(tokens,), cu_seqlens=None)
    assert plain.context_for_stage(0) is None and plain.context_for_stage(1) is None


def test_non_reader_model_validation_does_not_import_media_dependencies(
    training_bundle, monkeypatch
):
    cfg, tc = configs(training_bundle, ("image",))
    data = PreparedBundleData(cfg, tc)
    data.pp_rank, data.pp_size, data.pp_group = 1, 2, object()
    monkeypatch.setattr(dist, "get_global_rank", lambda group, rank: 0)

    def broadcast(message, *, src, group):
        assert src == 0 and group is data.pp_group
        message[0] = (200000, None)

    monkeypatch.setattr(dist, "broadcast_object_list", broadcast)
    original_import = builtins.__import__

    def no_media_import(name, *args, **kwargs):
        assert name != "pithtrain.modules.qwen3_omni_data", "Only the reader imports media code"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_media_import)
    model = type("MediaModel", (), {"input_modalities": {"text", "image"}})
    data.validate_model(model, SimpleNamespace(vocab_size=200000))
    assert data._processor is None and data._thinker_config is None
