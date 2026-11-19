"""CPU contracts of the released speech FSDP driver (validation rows 20-22).

GPU acceptance belongs to the driver itself. These tests check its pure contracts
and run the complete waveform produce/restore path on the tiny full-component
model with two Gloo ranks, including a fresh-process mismatch that must fail.
"""

import copy
import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

from pithtrain.tasks.convert_checkpoint import qwen3_omni_full as full
from tests.acceptance import omni_released_continuation as continuation
from tests.acceptance import omni_released_speech as speech
from tests.acceptance import omni_speech_chain as chain
from tests.acceptance import omni_speech_sharded as sharded

ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "tests/configs/qwen3_omni_full/released-config.json"


def released():
    return Qwen3OmniMoeConfig(**json.loads(RELEASED.read_text()))


def text_sample(config, index, *, modulus=2048, groups=16):
    """A real ChatML layout shaped like omni_speech_chain.make_sample, without media."""
    user = [config.im_start_token_id, config.user_token_id, 198]
    assistant = [config.im_start_token_id, config.assistant_token_id, 198]
    ids = torch.tensor([user + [100 + index, 101, 102] + assistant + [40, 41, 42, 43]])
    prompt = ids.shape[1] - 4
    labels = torch.full_like(ids, -100)
    labels[:, prompt - 1 : -1] = torch.tensor([40, 41, 42, 43])
    generator = torch.Generator().manual_seed(chain.SEED + index)
    return dict(
        input_ids=ids,
        media_context=dict(input_ids=ids.clone(), attention_mask=torch.ones_like(ids)),
        labels=labels,
        codes=torch.randint(modulus, (1, speech.CODE_FRAMES, groups), generator=generator),
        code_lengths=torch.tensor([speech.CODE_FRAMES]),
        prompt_lengths=[prompt],
        position_kwargs=dict(use_audio_in_video=False),
        media_kind=chain.MEDIA[index],
    )


def test_parameter_counts_and_budgets_match_the_pinned_released_headers():
    config = full.configuration(json.loads(RELEASED.read_text()), "all")
    shapes = full.parameter_shapes(config, "split")
    counts = {
        name: sum(math.prod(shape) for key, shape in shapes.items() if key.startswith(name + "."))
        for name in full.COMPONENTS
    }
    assert counts == speech.PARAMETERS
    # resume-20261026/claude-478ffaa-rows20-23-design-20261115/released-sizes-20261115.json
    expected = {
        "text_and_codec": (70_195_612_225, 421_389_689_924, 491_909_327_014),
        "codec_only": (90_467_215_970, 167_636_046_020, 238_155_683_110),
        "waveform": (71_815_736_552, 142_767_406_796, 213_287_043_886),
    }
    for mode, values in expected.items():
        budget = speech.sizes(mode)
        assert values == (
            budget["persistent_bytes_per_gpu"],
            budget["checkpoint_bytes"],
            budget["scratch_peak_bytes"],
        )
        assert budget["scratch_required_bytes"] > budget["scratch_peak_bytes"]


@pytest.mark.parametrize("mode", list(speech.DP))
def test_plan_keeps_the_tiny_gate_schedule(mode):
    plan = speech.plan(mode)
    assert plan["row"] == {"text_and_codec": 20, "codec_only": 21, "waveform": 22}[mode]
    assert plan["topology"] == dict(dp=speech.DP[mode], cp=1, ep=1, pp=1, hsdp_replica=1)
    assert plan["scans"] == sharded.scans(mode)
    assert sharded.updates_at(mode, plan["scans"]) == plan["updates"] == 4
    assert plan["save_scan"] == 2
    json.dumps(plan, allow_nan=False)


def test_two_rank_schedule_is_exactly_the_tiny_gate_schedule():
    config = released()
    samples = [text_sample(config, i, modulus=23) for i in range(len(chain.MEDIA))]
    for mode in speech.DP:
        for scan in range(sharded.scans(mode)):
            for rank in range(2):
                chain.exact(
                    speech.scheduled_batches(samples, mode, scan, rank, 2, 23, "cpu"),
                    sharded.scheduled_batches(samples, mode, scan, rank, "cpu"),
                    f"{mode}.{scan}.{rank}",
                )


def test_eight_rank_schedule_validates_against_the_released_config():
    config = released()
    modulus, groups = speech.code_layout(config)
    assert (modulus, groups) == (2048, 16)
    samples = [text_sample(config, i) for i in range(len(chain.MEDIA))]
    for mode, world in (("text_and_codec", 8), ("codec_only", 2), ("waveform", 2)):
        plans = speech.validate_schedule(samples, config, mode, world)
        assert len(plans) == sharded.scans(mode)
    for scan in range(sharded.scans("waveform")):
        per_rank = [
            speech.scheduled_batches(samples, "waveform", scan, r, 8, modulus, "cpu")
            for r in range(8)
        ]
        for rank, batches in enumerate(per_rank):
            odd = rank % 2
            for batch, sample in zip(batches, samples):
                assert torch.equal(batch["codes"], (sample["codes"] + rank + scan) % modulus)
                assert int(batch["codes"].max()) < modulus
                zero = scan == 2 or (scan == 1 and odd)
                assert batch["waveform_fraction"] == (0 if zero else 2 - odd)
                assert torch.equal(batch["codebook_mask"], per_rank[odd][0]["codebook_mask"])
                if (scan == 1 and odd) or scan == 2:
                    assert (batch["labels"] == -100).all()
    with pytest.raises(ValueError, match="rank/world"):
        speech.scheduled_batches(samples, "waveform", 0, 8, 8, modulus, "cpu")


def test_make_samples_only_widens_the_label_range():
    config = released()
    batches = {
        kind: SimpleNamespace(
            model_inputs=dict(
                input_ids=torch.tensor([[100 + i, 101, 102]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
            )
        )
        for i, kind in enumerate(("text", "image", "audio", "video", "all"))
    }
    samples = speech.make_samples(batches, config)
    for index, (kind, sample) in enumerate(zip(chain.MEDIA, samples)):
        tiny = chain.make_sample(batches["all" if kind == "mixed" else kind], config, index)
        generator = torch.Generator().manual_seed(chain.SEED + index)
        assert torch.equal(sample["codes"], torch.randint(2048, (1, 4, 16), generator=generator))
        chain.exact(
            {k: v for k, v in sample.items() if k != "codes"},
            {k: v for k, v in tiny.items() if k != "codes"},
        )


def test_reader_state_contract_including_the_waveform_zero_scan():
    config = released()
    samples = [text_sample(config, i) for i in range(len(chain.MEDIA))]
    data = speech.ReleasedData(samples, "f" * 64, "waveform", 3, 8, 2048)
    for scan in range(sharded.scans("waveform")):
        data.get_batch(scan, "cpu")
        if sharded.updates_at("waveform", scan + 1) > sharded.updates_at("waveform", scan):
            data.commit_update(scan_index=scan, update_index=data.completed_updates)
        else:
            with pytest.raises(ValueError, match="update mismatch"):
                data.commit_update(scan_index=scan, update_index=data.completed_updates)
    state = data.state_dict()
    assert (state["scanned_batches"], state["completed_updates"], state["last_update_scan"]) == (
        5,
        4,
        4,
    )
    fresh = speech.ReleasedData(samples, "f" * 64, "waveform", 3, 8, 2048)
    fresh.load_state_dict(dict(state, scanned_batches=3, completed_updates=2, last_update_scan=1))
    assert fresh.scanned_batches == 3
    with pytest.raises(ValueError, match="identity"):
        fresh.validate_state_dict(dict(state, world=2))
    with pytest.raises(ValueError, match="schedule"):
        fresh.validate_state_dict(dict(state, completed_updates=3))
    with pytest.raises(ValueError, match="scan mismatch"):
        fresh.get_batch(0, "cpu")


def test_scratch_admission_rejects_memory_backed_or_small_scratch(tmp_path):
    path = tmp_path.resolve()
    disk = "22 1 8:1 / / rw,relatime - xfs /dev/nvme0n1p1 rw"
    tmpfs = f"30 22 0:5 / {path} rw - tmpfs tmpfs rw"
    ok = speech.scratch_admission(path, 1, disk)
    assert (ok["mount"], ok["fstype"], ok["errors"]) == ("/", "xfs", [])
    memory = speech.scratch_admission(path, 1, disk + "\n" + tmpfs)
    assert memory["fstype"] == "tmpfs" and memory["errors"]
    assert speech.scratch_admission(path, 1 << 62, disk)["errors"]
    spaced = path / "with space"
    spaced.mkdir()
    escaped = str(spaced).replace(" ", "\\040")
    assert speech.mount_of(spaced, f"{disk}\n31 22 0:6 / {escaped} rw - tmpfs t rw")[1] == "tmpfs"


def test_snapshot_is_direct_only_for_the_exact_pinned_inventory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = {}
    for name, payload in (("config.json", b"{}"), ("model-00001.safetensors", b"abc")):
        (source / name).write_bytes(payload)
        rows[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (source / "download.json").write_bytes(b"receipt")
    extra = {"download.json": dict(bytes=7, sha256=hashlib.sha256(b"receipt").hexdigest())}
    mode, expected = speech.choose_snapshot(source, rows, extra)
    assert mode == "direct" and set(expected) == set(rows) | {"download.json"}
    (source / "README.md").write_text("not pinned")
    assert speech.choose_snapshot(source, rows, extra) == ("copied", rows)
    (source / "README.md").unlink()
    (source / "link").symlink_to(source / "config.json")
    assert speech.choose_snapshot(source, rows, extra)[0] == "copied"
    target = tmp_path / "copy"
    target.mkdir()
    speech.copy_verified(source / "config.json", target / "config.json", rows["config.json"])
    assert (target / "config.json").read_bytes() == b"{}"
    with pytest.raises(chain.PreconditionError, match="differs"):
        speech.copy_verified(
            source / "config.json", target / "bad.json", rows["model-00001.safetensors"]
        )
    with pytest.raises(FileExistsError):
        speech.copy_verified(source / "config.json", target / "config.json", rows["config.json"])


def test_released_files_bind_the_pinned_metadata_and_receipt(tmp_path, monkeypatch):
    metadata, source = tmp_path / "metadata", tmp_path / "source"
    metadata.mkdir()
    source.mkdir()
    names = [f"model-{i:05d}-of-00015.safetensors" for i in range(1, 16)]
    for name in names:
        (source / name).write_bytes(name.encode())
    (metadata / "config.json").write_text("{}")
    index = dict(weight_map={f"key.{i}": name for i, name in enumerate(names)})
    (metadata / "model.safetensors.index.json").write_text(json.dumps(index))

    def row(path):
        return dict(file=path.name, bytes=path.stat().st_size, sha256=continuation.file_sha(path))

    manifest = dict(
        model=speech.MODEL,
        revision=speech.REVISION,
        files=[row(metadata / "config.json"), row(metadata / "model.safetensors.index.json")],
    )
    (metadata / "manifest.json").write_text(json.dumps(manifest))
    download = dict(
        status="DOWNLOADED_FROM_PINNED_PUBLIC_REVISION_NOT_YET_CONVERTED",
        model=speech.MODEL,
        revision=speech.REVISION,
        files=[
            dict(
                name=n,
                bytes=(source / n).stat().st_size,
                observed_sha256=continuation.file_sha(source / n),
            )
            for n in names
        ],
    )
    (source / "download.json").write_text(json.dumps(download))
    monkeypatch.setattr(speech, "METADATA_SHA", continuation.file_sha(metadata / "manifest.json"))
    monkeypatch.setattr(speech, "CONFIG_SHA", continuation.file_sha(metadata / "config.json"))
    monkeypatch.setattr(speech, "SHARD_BYTES", sum(len(n) for n in names))
    receipt = continuation.file_sha(source / "download.json")
    files, receipt_row = speech.released_files(metadata, source, receipt)
    assert set(files) == {"config.json", "model.safetensors.index.json", *names}
    assert receipt_row["sha256"] == receipt
    with pytest.raises(chain.PreconditionError, match="receipt"):
        speech.released_files(metadata, source, "0" * 64)
    monkeypatch.setattr(speech, "SHARD_BYTES", 1)
    with pytest.raises(chain.PreconditionError, match="bytes"):
        speech.released_files(metadata, source, receipt)


def test_cleanup_removes_only_the_owned_scratch(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    scratch = tmp_path / "scratch"
    marker = speech.make_scratch(scratch, "waveform")
    (scratch / "speech-dcp").mkdir()
    (scratch / "speech-dcp" / "payload").write_bytes(b"12345")
    with pytest.raises(chain.PreconditionError, match="new absolute"):
        speech.make_scratch(scratch, "waveform")
    report = dict(
        schema=speech.SCHEMA,
        status="PASS_RELEASED_SPEECH_DCP_STAGED",
        source=dict(commit="a" * 40),
        training_mode="waveform",
        scratch=str(scratch),
        owner_uuid=marker["uuid"],
    )
    stage = tmp_path / "stage.json"
    continuation.write_json(stage, report)
    args = SimpleNamespace(
        stage=stage,
        stage_sha256=continuation.file_sha(stage),
        expected_commit="a" * 40,
        training_mode="waveform",
        scratch=scratch,
        output=tmp_path / "cleanup",
    )
    monkeypatch.setenv("SLURM_JOB_ID", "another-job")
    with pytest.raises(chain.PreconditionError, match="not owned"):
        speech.cleanup(args)
    assert (scratch / "speech-dcp" / "payload").exists() and not args.output.exists()
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(chain.PreconditionError, match="mode"):
        speech.cleanup(SimpleNamespace(**dict(vars(args), training_mode="codec_only")))
    result = speech.cleanup(args)
    assert result["absent"] and not scratch.exists()
    assert result["removed_bytes"] == 5 + len((json.dumps(marker, sort_keys=True) + "\n").encode())


def test_cli_requires_explicit_inputs():
    with pytest.raises(SystemExit):
        speech.parser().parse_args(["restore", "--training-mode", "waveform"])
    args = speech.parser().parse_args(["plan", "--training-mode", "codec_only"])
    assert args.phase == "plan"


def tiny_stage(root):
    from tests import test_omni_full_converter as fixtures
    from tests import test_omni_speech_initialization as initialization

    raw = fixtures.tiny_raw()
    fused = {k: v.clone().bfloat16() for k, v in fixtures.reference(raw).state_dict().items()}
    source = root / "hf"
    fixtures.write_hf(source, raw, fixtures.split_fixture(fused, "split"))
    scratch = root / "scratch"
    marker = speech.make_scratch(scratch, "waveform")
    fields = speech.stage_dcp(source, initialization.source_identity(source), scratch, False)
    stage = dict(
        fields,
        schema=speech.SCHEMA,
        status="PASS_RELEASED_SPEECH_DCP_STAGED",
        source=dict(commit="0" * 40),
        training_mode="waveform",
        scratch=str(scratch),
        owner_uuid=marker["uuid"],
    )
    return raw, stage


def waveform_samples(config):
    modulus, groups = speech.code_layout(config)
    samples = []
    for index in range(len(chain.MEDIA)):
        generator = torch.Generator().manual_seed(index)
        ids = torch.tensor([[1, 2, 3, 4]])
        samples.append(
            dict(
                input_ids=ids,
                labels=torch.tensor([[-100, 5, 6, -100]]),
                codes=torch.randint(modulus, (1, speech.CODE_FRAMES, groups), generator=generator),
                media_kind=chain.MEDIA[index],
            )
        )
    return samples


def bind_sources(model):
    """native_meta executes real class bodies without a module; name their real files.

    The strict checkpoint contract hashes every module class's source file, which a
    class created by exec() lacks. Only the test's Triton-free constructors need this.
    """
    import ast
    import sys
    import types

    owners = {}
    for filename in (
        "pithtrain/operators/linear.py",
        "pithtrain/models/qwen3_omni_moe.py",
        "pithtrain/models/qwen3_omni_image.py",
        "pithtrain/models/qwen3_omni_multimodal.py",
    ):
        for node in ast.parse((ROOT / filename).read_text()).body:
            if isinstance(node, ast.ClassDef):
                owners.setdefault(node.name, filename)
    for module in model.modules():
        for cls in type(module).__mro__:
            if cls.__module__ == "builtins" and cls.__name__ in owners:
                name = "released_speech_test." + owners[cls.__name__][:-3].replace("/", ".")
                if name not in sys.modules:
                    stub = types.ModuleType(name)
                    stub.__file__ = str(ROOT / owners[cls.__name__])
                    sys.modules[name] = stub
                cls.__module__ = name
    return model


def phase_worker(rank, root, phase, tag, raw, stage, samples, producer, producer_sha):
    from tests.test_omni_speech_initialization import native_meta

    dist.init_process_group(
        "gloo",
        init_method=f"file://{Path(root) / ('rendezvous-' + tag)}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.set_num_threads(1)
        output = Path(root) / tag
        if rank == 0:
            output.mkdir()
        dist.barrier()
        speech.run_phase(
            phase,
            mode="waveform",
            config=Qwen3OmniMoeConfig(**copy.deepcopy(raw)),
            samples=samples,
            fixture_sha="f" * 64,
            stage=stage,
            scratch=Path(stage["scratch"]),
            output=output,
            source=dict(commit="0" * 40),
            runtime={},
            device=torch.device("cpu"),
            build_model=lambda _config: bind_sources(native_meta(raw)),
            seconds=dict(initialization=120, save=120, load=120, update=120, snapshot=120),
            producer=producer,
            producer_sha256=producer_sha,
            contexts=False,
        )
    finally:
        dist.destroy_process_group()


def test_waveform_produce_and_fresh_restore_on_two_gloo_ranks(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    raw, stage = tiny_stage(tmp_path)
    config = Qwen3OmniMoeConfig(**copy.deepcopy(raw))
    samples = waveform_samples(config)
    mp.spawn(
        phase_worker,
        args=(str(tmp_path), "produce", "produce", raw, stage, samples, None, None),
        nprocs=2,
        join=True,
    )
    produce = tmp_path / "produce"
    complete = continuation.read_json(produce / "complete.json")
    assert complete["status"] == "PASS_RELEASED_SPEECH_PRODUCER"
    reports = [continuation.read_json(produce / f"rank-0/scan-{n}.json") for n in range(1, 6)]
    assert [r["expected_update"] for r in reports] == [True, True, False, True, True]
    assert all(r["status"] == "PASS" for r in reports)
    assert (tmp_path / "scratch/checkpoint").is_dir()
    digest = continuation.file_sha(produce / "complete.json")
    mp.spawn(
        phase_worker,
        args=(str(tmp_path), "restore", "restore", raw, stage, samples, str(produce), digest),
        nprocs=2,
        join=True,
    )
    restore = tmp_path / "restore"
    done = continuation.read_json(restore / "complete.json")
    assert done["status"] == "PASS_RELEASED_SPEECH_FRESH_RESTORE"
    for rank in range(2):
        for count in range(2, 6):
            name = f"rank-{rank}/state-{count}/index.json"
            assert continuation.file_sha(restore / name) == continuation.file_sha(produce / name)
    # Different post-boundary inputs: the restored boundary still matches, scan 3 must not.
    changed = [dict(s, codes=(s["codes"] + 1) % speech.code_layout(config)[0]) for s in samples]
    with pytest.raises(Exception):
        mp.spawn(
            phase_worker,
            args=(str(tmp_path), "restore", "changed", raw, stage, changed, str(produce), digest),
            nprocs=2,
            join=True,
        )
    failure = continuation.read_json(tmp_path / "changed/rank-0/failure.json")
    assert failure["status"] == "FAIL" and "input batch 3 differs" in failure["error"]
    assert (tmp_path / "changed/rank-0/state-2/index.json").exists()


def test_seed_process_seeds_every_generator_like_the_launcher(monkeypatch):
    """Each rank sees every DP device; the checkpoint restores only its own CUDA generator."""
    import random

    import numpy as np

    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
    seeded = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded.append)
    states = []
    try:
        for _ in range(2):
            random.random()
            np.random.rand()
            torch.rand(1)
            speech.seed_process()
            states.append(
                (random.getstate(), np.random.get_state()[1].tolist(), torch.get_rng_state())
            )
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
    assert seeded and set(seeded) == {chain.SEED}
    assert states[0][:2] == states[1][:2] and torch.equal(states[0][2], states[1][2])
