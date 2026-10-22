"""Prepared single-GPU text checkpoint acceptance; explicit scheduling opt-in.

Run only after the shared GPU is released:
PITHTRAIN_RUN_OMNI_GPU_TESTS=1 python -m pytest tests/test_qwen3_omni_checkpoint_gpu.py -q -s
This is tiny forward/weight validation, not PP/EP updates or strict training resume.
"""

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from pithtrain.contexts import distributed, training
from pithtrain.modules.checkpoint import to_localized_model
from pithtrain.tasks.convert_checkpoint import dcp2hf, hf2dcp
from pithtrain.tasks.convert_checkpoint.qwen3_omni_moe import (
    canonical_to_hf,
    hf_to_canonical,
    load_canonical,
    text_config,
)
from tests.test_qwen3_omni_checkpoint import (
    LOGGER,
    make_cfg,
    model,
    read_export,
    synthetic_hf,
    write_hf,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") != "1",
    reason="GPU acceptance is prepared; requires explicit resource-coordination opt-in",
)


@pytest.mark.parametrize("layout", ["split", "fused"])
@pytest.mark.parametrize("length", [32, 64, 69])
def test_tiny_gpu_checkpoint_predictions(tmp_path, monkeypatch, layout, length):
    assert torch.cuda.is_available(), "An explicitly scheduled CUDA device is required"
    # Native imports probe CUDA. Keep them behind the opt-in, never in collection.
    from pithtrain.models.qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel
    from pithtrain.operators.grouped_linear import GroupedLinear

    raw = json.loads((Path(__file__).parent / "configs/qwen3_omni_text/tiny.json").read_text())
    raw.update(vocab_size=256, num_hidden_layers=2)
    config = text_config(raw)
    source = synthetic_hf(config, layout=layout, dtype=torch.bfloat16)
    src, disk, out = (tmp_path / name for name in ("hf", "dcp", "out"))
    write_hf(src, config, source, sharded=layout == "split")
    cfg = make_cfg(load_path=src, save_path=disk)
    hf2dcp(cfg, LOGGER)
    canonical = load_canonical(disk, config)
    cfg.load_path, cfg.save_path, cfg.omni_expert_layout = disk, out, layout
    dcp2hf(cfg, LOGGER)
    exported = read_export(out)
    assert source.keys() == exported.keys()
    for key in source:
        torch.testing.assert_close(exported[key], source[key], rtol=0, atol=0)
    restored, _ = hf_to_canonical(exported, config)
    for key in canonical:
        torch.testing.assert_close(canonical[key], restored[key], rtol=0, atol=0)

    # Preserve HF's FP32 rotary buffers while making weights BF16, as in PR101.
    hf = model(config)
    for parameter in hf.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    hf_after = copy.deepcopy(hf)
    hf_after.load_state_dict(canonical_to_hf(restored, config, prefix=""), strict=True)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    for key, value in {
        "pp_size": 1,
        "ep_size": 1,
        "cp_size": 1,
        "pp_rank": 0,
        "ep_rank": 0,
        "cp_rank": 0,
        "pp_group": None,
        "ep_group": None,
        "cp_group": None,
        "device": device,
    }.items():
        monkeypatch.setattr(distributed, key, value, raising=False)
    for key, value in {
        "fp8": False,
        "Linear": nn.Linear,
        "GroupedLinear": GroupedLinear,
        "PARAM_DTYPE": torch.bfloat16,
    }.items():
        monkeypatch.setattr(training, key, value, raising=False)
    with torch.device(device):
        before = Qwen3OmniMoeThinkerTextModel(config, phase=-1).to(dtype=torch.bfloat16)
        after = Qwen3OmniMoeThinkerTextModel(config, phase=-1).to(dtype=torch.bfloat16)
    # Independent direct HF -> native names for the reference arm.
    before.load_state_dict(
        {key.removeprefix("model."): value for key, value in hf.state_dict().items()}, strict=True
    )
    after.load_state_dict(to_localized_model(restored, after), strict=True)
    hf, hf_after = hf.to(device), hf_after.to(device)
    tokens = torch.randint(
        0, config.vocab_size, (2, length), generator=torch.Generator().manual_seed(17)
    ).to(device)
    with torch.no_grad():
        original_logits = before.reference_forward(tokens)
        restored_logits = after.reference_forward(tokens)
        hf_logits = hf["lm_head"](hf["model"](input_ids=tokens, use_cache=False).last_hidden_state)
        hf_restored = hf_after["lm_head"](
            hf_after["model"](input_ids=tokens, use_cache=False).last_hidden_state
        )
    assert torch.isfinite(original_logits).all() and torch.isfinite(hf_logits).all()
    torch.testing.assert_close(restored_logits, original_logits, rtol=0, atol=0)
    torch.testing.assert_close(hf_restored, hf_logits, rtol=0, atol=0)
    actual, expected = restored_logits.double(), hf_logits.double()
    error = (
        (actual - expected).square().sum() / (actual.square() + expected.square()).sum()
    ).item()
    assert error < 1e-3, f"Native/HF normalized squared logit error={error}"
    print(f"layout={layout} length={length} roundtrip_max_error=0 native_hf_error={error:.8g}")
