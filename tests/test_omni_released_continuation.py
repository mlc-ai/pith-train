"""Real CPU public model initialization and fresh-process strict continuation.

The distributed fixture is deliberately tiny: this verifies driver/checkpoint
mechanics, never released-weight values, native GPU math or full-size resources.
"""

import hashlib
import json
import logging
import random
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from pithtrain.contexts import logging as log_context
from pithtrain.modules.checkpoint import initialize_checkpoint, iter_canonical_parameters
from tests.acceptance import omni_released_continuation as driver
from tests.test_distributed_strict_checkpoint import advance, build, initialize_mesh


def test_complete_typed_streamed_state_and_tampering(tmp_path):
    value = {
        1: torch.arange(32).reshape(8, 4).float(),
        "1": (1, [2]),
        "bf16": torch.tensor([1.5], dtype=torch.bfloat16),
        "rng": np.arange(4, dtype=np.uint32),
    }
    root = tmp_path / "state"
    driver.write_rows(root, driver.state_rows(value))
    driver.compare_rows(driver.state_rows(value), driver.read_rows(root))
    with pytest.raises(AssertionError):
        driver.compare_rows(
            driver.state_rows({"1": value[1], 1: value["1"]}), driver.read_rows(root)
        )
    value[1][0, 0] = 20
    with pytest.raises(AssertionError, match="state mismatch"):
        driver.compare_rows(driver.state_rows(value), driver.read_rows(root))
    chunk = root / "rows-00000.json"
    chunk.write_text("[]")
    with pytest.raises(AssertionError):
        list(driver.read_rows(root))


def test_nested_deadline_keeps_outer_remaining_limit():
    with driver.deadline(2):
        with driver.deadline(1):
            time.sleep(0.01)
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 0 < remaining < 2 and interval == 0
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_independent_split_mapping_reads_every_value(tmp_path):
    source = tmp_path / "weights.bin"
    gate = torch.arange(6).reshape(3, 2).bfloat16()
    up = gate + 10
    source.write_bytes(
        gate.view(torch.uint8).numpy().tobytes() + up.view(torch.uint8).numpy().tobytes()
    )
    key = "layers.0.mlp.experts.2.gate_up_proj"
    names = driver.canonical_parts(key)
    assert names == [
        "thinker.model.layers.0.mlp.experts.2.gate_proj.weight",
        "thinker.model.layers.0.mlp.experts.2.up_proj.weight",
    ]
    headers = {
        n: dict(file=source.name, offset=i * 12, bytes=12, shape=[3, 2])
        for i, n in enumerate(names)
    }
    record = driver.expected_fp32_record(key, (6, 2), headers, tmp_path)
    expected = torch.cat((gate, up)).float().numpy().tobytes()
    assert record["sha256"] == hashlib.sha256(expected).hexdigest()
    with pytest.raises(AssertionError):
        driver.expected_fp32_record(key, (5, 2), headers, tmp_path)


class Data:
    def __init__(self, cursor):
        self.checkpoint_state = cursor


def _worker(rank, directory, phase):
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / ('init-' + phase)}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        torch.set_num_threads(1)
        log_context.stdout = logging.getLogger("public-init-cpu-test")
        parts = build(*initialize_mesh("fsdp-layout", rank))
        model, optimizer, _, cursor = parts
        seed = rank + (400 if phase == "produce" else 700)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        output = root / phase
        if rank == 0:
            output.mkdir()
        dist.barrier()
        if phase == "produce":
            # This is an actual canonical BF16 DCP payload, not a mocked loader.
            canonical = {k: v.detach().cpu().bfloat16() for k, v in iter_canonical_parameters()}
            expected = {
                k: dict(
                    key=k,
                    shape=list(v.shape),
                    dtype="torch.float32",
                    sha256=driver.tensor_digest(v.float())["sha256"],
                )
                for k, v in canonical.items()
            }
            if rank == 0:
                dcp.save(
                    {"app": {"model": canonical}},
                    checkpoint_id=root / "model-only/torch-dcp/00000000",
                    no_dist=True,
                )
            dist.barrier()
            shapes = {k: v.shape for k, v in canonical.items()}
            driver.inspect_initialization_dcp(root / "model-only", shapes)
            with pytest.raises(AssertionError):
                driver.inspect_initialization_dcp(root / "model-only", dict(shapes, surprise=(1,)))
            with torch.no_grad():
                for p in model.parameters():
                    p.to_local().fill_(-6)
            before_rng = torch.get_rng_state().clone()
            initialize_checkpoint(root / "model-only", 0)
            assert not optimizer.state and cursor.consumed == 0
            assert torch.equal(torch.get_rng_state(), before_rng)
            driver.verify_loaded_parameters(expected, output / f"initialized-{rank}")
            bad = dict(expected)
            key = next(iter(bad))
            bad[key] = dict(bad[key], sha256="0" * 64)
            with pytest.raises(AssertionError):
                driver.verify_loaded_parameters(bad, output / f"negative-readback-{rank}")
        data = Data(cursor)
        steps = []

        def train_one(step):
            assert cursor.consumed == 2 * step
            steps.append(step)
            advance(parts)

        driver.continuation(
            phase,
            dict(
                checkpoint=root / "strict",
                save_seconds=40,
                load_seconds=40,
                resume_contract={"training": {"global_batch_size": 2}},
            ),
            data,
            torch.device("cpu"),
            output,
            root / "produce",
            train_one,
        )
        assert steps == ([0, 1] if phase == "produce" else [1])
        assert cursor.consumed == 4 and not torch.cuda.is_initialized()
        (output / f"result-{rank}.json").write_text(json.dumps(dict(status="PASS", steps=steps)))
    finally:
        dist.destroy_process_group()


def test_real_public_init_then_fresh_process_strict_next_update(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path), "produce"), nprocs=2, join=True)
    mp.spawn(_worker, args=(str(tmp_path), "restore"), nprocs=2, join=True)
    for phase in ("produce", "restore"):
        for rank in range(2):
            assert (
                json.loads((tmp_path / phase / f"result-{rank}.json").read_text())["status"]
                == "PASS"
            )


def test_inner_deadline_cannot_extend_outer_timeout():
    with pytest.raises(TimeoutError):
        with driver.deadline(1):
            with driver.deadline(2):
                time.sleep(1.2)
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def test_collective_timeout_covers_serialized_owner_io_budget():
    assert driver.collective_timeout_seconds("produce", 200, 1200, 300) == 1210
    assert driver.collective_timeout_seconds("restore", 200, 300, 1500) == 1510
    assert driver.collective_timeout_seconds("restore", 800, 300, 400) == 810
    with pytest.raises(AssertionError):
        driver.collective_timeout_seconds("restore", 800, 300, 0)


def test_actual_tiny_hf_public_converter_config(tmp_path):
    from tests import test_omni_thinker_converter as tiny

    config = tiny.configuration()
    state = tiny.source_state(config, "thinker.", "split", torch.bfloat16)
    source, converted = tmp_path / "hf", tmp_path / "native"
    tiny.fixtures.write_hf(source, config, state, sharded=True)
    driver.convert_thinker(source, converted)
    result = driver.inspect_initialization_dcp(converted, tiny.parameter_shapes(config))
    assert result["keys"] == len(tiny.parameter_shapes(config))
    assert not torch.cuda.is_initialized()
