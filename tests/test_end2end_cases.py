"""
Every end-to-end benchmark case builds the config its path names, and the mesh is one PithTrain
accepts. The scripts run without their __main__ block, which only builds the config; one GPU is
needed only because importing pithtrain reads the device capability.
"""

import json
import runpy
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("importing pithtrain needs a GPU", allow_module_level=True)

from benchmarks.end2end.common import MODELS  # noqa: E402
from benchmarks.end2end.summarize import parse_case  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(ROOT.glob("benchmarks/end2end/*/*/*/script.py"))


def test_cases_exist():
    assert len(SCRIPTS) > 0


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.parent.relative_to(ROOT)))
def test_case_matches_path(script: Path, monkeypatch):
    monkeypatch.chdir(ROOT)
    case = parse_case(str(script.parent.relative_to(ROOT / "benchmarks/end2end")))
    cfg = runpy.run_path(str(script), run_name="case")["cfg"]
    training, distributed = cfg.training, cfg.distributed
    pp, cp, ep = (
        distributed.pipeline_parallel_size,
        distributed.context_parallel_size,
        distributed.expert_parallel_size,
    )
    world = case["world_size"]
    assert case["gpu"] == "h200" and case["gpus_per_node"] == 8
    assert training.model == MODELS[case["model"]][0]
    assert training.sequence_length == case["sequence_length"]
    assert training.fp8 == (case["dtype"] == "fp8")
    assert training.global_batch_size == case["global_batch_size"]
    assert training.benchmark and cfg.dataset is None
    assert training.save_interval is None and cfg.logging.wandb is None
    assert (pp, cp, ep) == (case["pp"], case["cp"], case["ep"])
    assert world % pp == 0 and (world // pp) % cp == 0 and (world // pp) % ep == 0
    assert case["dp"] == world // (pp * cp)
    assert training.sequence_length % (2 * cp) == 0
    assert training.global_batch_size % (training.micro_batch_size * case["dp"]) == 0
    assert training.global_batch_size // case["dp"] >= 2 * pp
    config = json.loads(Path(training.model, "config.json").read_text())
    config = config.get("text_config", config)
    num_experts = next(
        config[k] for k in ("num_experts", "n_routed_experts", "num_local_experts") if config.get(k)
    )
    assert num_experts % ep == 0
