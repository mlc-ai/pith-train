"""Cross-rank input failures and reader resume for the native video boundary."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pithtrain.models.qwen3_omni_video_inputs import validate_video_inputs
from pithtrain.modules.data_config import DataCfg
from pithtrain.modules.training_data import OmniPretrainData
from tests import test_omni_training as fixtures

manifest = fixtures.manifest
processor_config = fixtures.processor_config
training_bundle = fixtures.training_bundle


def video_data(root, **ranks):
    cfg = DataCfg()
    cfg.dataset, cfg.format, cfg.modalities = root, "prepared_bundle", ("video",)
    cfg.epoch_samples = 8
    training = SimpleNamespace(
        global_batch_size=4, micro_batch_size=1, sequence_length=1024, seed=1234
    )
    return OmniPretrainData(cfg, training, **ranks)


def _bad_video_worker(rank, root, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    pp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    data = video_data(
        root,
        dp_rank=rank % 2,
        dp_size=2,
        pp_rank=rank // 2,
        pp_size=2,
        pp_group=pp_groups[rank % 2],
    )

    def validator(ids, inputs, config):
        if rank == 1:
            inputs["video_second_per_grid"][0] = float("nan")
        return validate_video_inputs(ids, inputs, config)

    from pithtrain.tasks.prepare_omni_data import processor_for

    _, config = processor_for(data.recipe, True)
    data.validate_model(
        SimpleNamespace(input_modalities=("video",), validate_media_batch=validator), config
    )
    try:
        data.get_batch(0, "cpu")
    except RuntimeError as error:
        message = str(error)
        assert "Global media batch rejected" in message and "video_second_per_grid" in message
        assert data.consumed_samples == 0
        Path(output, f"rank-{rank}.json").write_text(
            json.dumps(dict(status="PASSED", error=message))
        )
    else:
        raise AssertionError("Invalid temporal metadata accepted")
    finally:
        dist.destroy_process_group()


def test_one_bad_video_rejects_all_pp_and_dp_ranks(training_bundle, processor_config, tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    mp.spawn(
        _bad_video_worker, args=(training_bundle, tmp_path / "rdzv", output), nprocs=4, join=True
    )
    reports = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(4)]
    assert len({row["error"] for row in reports}) == 1
    assert all(row["status"] == "PASSED" for row in reports)


def test_video_reader_resume_keeps_pixels_and_temporal_metadata(training_bundle, processor_config):
    _, config = processor_config
    consumer = SimpleNamespace(
        input_modalities=("video",), validate_media_batch=validate_video_inputs
    )
    data = video_data(training_bundle, dp_rank=0, dp_size=1)
    data.validate_model(consumer, config)
    data.get_batch(0, "cpu")
    data.commit_step(0)
    state = data.state_dict()
    expected = data.get_batch(1, "cpu")
    restored = video_data(training_bundle, dp_rank=0, dp_size=1)
    restored.validate_model(consumer, config)
    restored.load_state_dict(state)
    actual = restored.get_batch(1, "cpu")
    fixtures.compare_batches(actual, expected)
    for batch in actual:
        assert "pixel_values_videos" in batch.media_inputs
        assert "video_second_per_grid" in batch.model_context
        validate_video_inputs(batch.model_inputs[0], batch.context_for_stage(0), config)
