"""Real Qwen processor and CPU/Gloo checks of PP media ownership.

No CUDA/NCCL or native encoder execution is covered by these tests.
"""

import hashlib
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
from pithtrain.modules.training_data import OmniPretrainData
from tests import test_omni_training as fixtures
from tests.omni_media_audit import MediaAudit
from tests.test_omni_training_acceptance import media_kind

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
        media_kind(actual, 0 if owns_media else 1)
        assert actual.sample_ids == reference["sample_ids"]
        torch.testing.assert_close(actual.objective_inputs[0], reference["labels"], rtol=0, atol=0)
        all_inputs = reference["inputs"]
        common = {name: value for name, value in all_inputs.items() if name not in PAYLOADS}
        media = {name: value for name, value in all_inputs.items() if name in PAYLOADS}
        torch.testing.assert_close(actual.model_context, common, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.media_inputs or {}, media if owns_media else {}, rtol=0, atol=0
        )
        assert actual.model_inputs[0] is actual.model_context["input_ids"]
        assert not PAYLOADS & actual.context_for_stage(3).keys()
        if owns_media:
            torch.testing.assert_close(actual.context_for_stage(0), all_inputs, rtol=0, atol=0)


def _media_worker(rank, rendezvous, root, reference_file):
    from pithtrain.modules import qwen3_omni_data
    from pithtrain.tasks import prepare_omni_data

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
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
        assert kw["group"] is group and kw["src"] == dp_rank
        no_tensors(message)
        original_object_broadcast(message, **kw)
        no_tensors(message)
        value, error = message[0]
        if error is None and isinstance(value, tuple) and isinstance(value[0], tuple):
            _, specs = value
            assert not PAYLOADS & {name for name, _, _ in specs}
        counters["headers"] += 1

    def tensor_broadcast(tensor, **kw):
        assert kw["group"] is group and kw["src"] == dp_rank
        counters["tensors"] += 1
        return original_tensor_broadcast(tensor, **kw)

    with ExitStack() as stack:
        for module, name, counter in (
            (prepare_omni_data, "verify_bundle", "verify"),
            (prepare_omni_data, "processor_for", "processor"),
            (qwen3_omni_data, "create_omni_dataloader", "loader"),
        ):
            stack.enter_context(
                patch.object(module, name, owner_only(counter, getattr(module, name)))
            )
        stack.enter_context(patch.object(dist, "broadcast_object_list", object_broadcast))
        stack.enter_context(patch.object(dist, "broadcast", tensor_broadcast))
        audit = MediaAudit(pp_rank, 2, group)
        audit.install(stack, OmniPretrainData, prepare_omni_data, qwen3_omni_data, dist)
        for mode, kinds in enumerate(MODES):
            cfg, tc = configs(root, kinds)
            data = OmniPretrainData(cfg, tc, **kwargs)
            supported = type("TestMediaModel", (), {"input_modalities": set(kinds)})
            data.validate_model(supported, SimpleNamespace(vocab_size=200000))
            rng = torch.get_rng_state().clone()
            first = data.get_batch(0, "cpu")
            assert_batches(first, reference[mode][dp_rank][0], pp_rank == 0)
            with pytest.raises(RuntimeError, match="optimizer step"):
                data.state_dict()
            data.commit_step(0)
            saved = data.state_dict()
            assert saved["version"] == 1 and saved["consumed_samples"] == 4
            if pp_rank != 0:
                assert data._iterator is None and data._processor is None
            for step in (1, 2):
                assert_batches(
                    data.get_batch(step, "cpu"), reference[mode][dp_rank][step], pp_rank == 0
                )
                data.commit_step(step)
            assert torch.equal(torch.get_rng_state(), rng), "Data distribution changed model RNG"

            restored = OmniPretrainData(cfg, tc, **kwargs)
            restored.load_state_dict(saved)
            for step in (1, 2):
                assert_batches(
                    restored.get_batch(step, "cpu"), reference[mode][dp_rank][step], pp_rank == 0
                )
                restored.commit_step(step)
            assert restored.state_dict() == data.state_dict()

        audit_report = audit.finish_data(
            providers=2 * len(MODES), validations=len(MODES), expected_batches=10 * len(MODES)
        )
        (Path(rendezvous).parent / f"media-audit-rank{rank}.json").write_text(
            json.dumps(audit_report)
        )
        cfg, tc = configs(root, ("image",))
        failed = OmniPretrainData(cfg, tc, **kwargs)

        def fail_read():
            raise ValueError("expected-reader-failure")

        failed._make_iterator = fail_read
        with pytest.raises(RuntimeError, match="expected-reader-failure"):
            failed.get_batch(0, "cpu")
        assert failed.pending_step == 0 and failed.consumed_samples == 0
        with pytest.raises(RuntimeError, match="optimizer step"):
            failed.state_dict()
        with pytest.raises(ValueError, match="process group does not match"):
            OmniPretrainData(cfg, tc, **dict(kwargs, pp_group=dist.group.WORLD))

        # The owner still checks EVERY payload hash. A corrupted media file
        # fails the whole PP group, including peers that never open the file.
        dist.barrier()
        if rank == 0:
            (Path(root) / "image.png").write_bytes(b"corrupt test image")
        dist.barrier()
        with pytest.raises(RuntimeError, match="Dataset hash mismatch"):
            OmniPretrainData(cfg, tc, **kwargs)

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


@pytest.mark.parametrize("rank,size", [(-1, 2), (2, 2), (0, 0), (True, 2)])
def test_media_rejects_invalid_pp_coordinates(training_bundle, rank, size):
    with pytest.raises(ValueError, match="pipeline-parallel"):
        OmniPretrainData(*configs(training_bundle, ("image",)), pp_rank=rank, pp_size=size)


def test_media_rejects_missing_pp_group(training_bundle):
    with pytest.raises(ValueError, match="initialized pipeline"):
        OmniPretrainData(*configs(training_bundle, ("image",)), pp_size=2)


def test_stage_context_keeps_shared_inputs_and_rejects_overwrite():
    tokens, pixels = torch.ones(1, 3, dtype=torch.long), torch.ones(4, 3)
    shared = {"input_ids": tokens, "image_grid_thw": torch.tensor([[1, 2, 2]])}
    micro = Microbatch(
        model_inputs=(tokens,),
        objective_inputs=(tokens,),
        cu_seqlens=None,
        model_context=shared,
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


def test_media_checkpoint_fingerprint_unchanged(training_bundle, processor_config):
    cfg, tc = configs(training_bundle, ("text", "image", "audio", "video"))
    data = OmniPretrainData(cfg, tc)
    recipe = json.loads((training_bundle / "bundle.json").read_text())["recipe"]
    identity = dict(
        version=1,
        bundle_sha256=hashlib.sha256((training_bundle / "bundle.json").read_bytes()).hexdigest(),
        stage="video",
        global_batch_size=4,
        micro_batch_size=1,
        sequence_length=1024,
        seed=17,
        weights={k: recipe["sampling_weights"][k] for k in cfg.modalities},
        epoch_samples=8,
    )
    saved = dict(
        version=1,
        fingerprint=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        consumed_samples=4,
    )
    data.load_state_dict(saved)
    assert data.state_dict() == saved
