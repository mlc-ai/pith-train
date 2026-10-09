"""Native Thinker text vs pinned-config HF: logits, CE and every parameter gradient.

Run on a Hopper/Blackwell GPU: python -m pytest tests/test_qwen3_omni_reference.py -q
Synthetic tiny weights isolate numerical correctness. They are not released
weights or a full-size execution; full.json retains the released text config.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel as HFTextModel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native kernels require a CUDA Hopper/Blackwell GPU"
)


def difference(actual, expected):
    actual, expected = actual.double(), expected.double()
    denominator = (actual.square() + expected.square()).sum()
    if denominator == 0:
        return 0.0
    return ((actual - expected).square().sum() / denominator).item()


def native_name(name):
    return name if name.startswith("lm_head.") else "model." + name


@pytest.mark.parametrize(
    "variant", ["moe", "dense-and-unnormalized-router", "packed", "odd-length"]
)
def test_native_omni_matches_hf(variant):
    from pithtrain.contexts import distributed, training
    from pithtrain.models.qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel
    from pithtrain.operators.grouped_linear import GroupedLinear

    device, dtype = torch.device("cuda", 0), torch.bfloat16
    torch.cuda.set_device(device)
    distributed.pp_size = distributed.ep_size = distributed.cp_size = 1
    distributed.pp_rank = distributed.ep_rank = distributed.cp_rank = 0
    distributed.pp_group = distributed.ep_group = distributed.cp_group = None
    distributed.device = device
    training.fp8, training.Linear, training.GroupedLinear = False, nn.Linear, GroupedLinear
    path = Path(__file__).parent / "configs/qwen3_omni_text/tiny.json"
    config = Qwen3OmniMoeTextConfig(**json.loads(path.read_text()))
    if variant == "dense-and-unnormalized-router":
        config.mlp_only_layers = [0]
        config.norm_topk_prob = False
        config.attention_bias = True
    hf_config = copy.deepcopy(config)
    hf_config._attn_implementation = "eager"
    # Construct directly in BF16, retaining HF's FP32 RoPE frequency buffers.
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            hf_model = nn.ModuleDict(
                {
                    "model": HFTextModel(hf_config),
                    "lm_head": nn.Linear(config.hidden_size, config.vocab_size, bias=False),
                }
            )
            for name, parameter in hf_model.named_parameters():
                if parameter.ndim > 1:
                    nn.init.normal_(parameter, std=config.initializer_range)
                elif name.endswith("bias"):
                    nn.init.zeros_(parameter)
                else:
                    nn.init.ones_(parameter)
    finally:
        torch.set_default_dtype(original_dtype)
    hf_model = hf_model.to(device)
    with torch.device(device):
        native_model = Qwen3OmniMoeThinkerTextModel(config, phase=-1).to(dtype=dtype)
    expected_params = dict(hf_model.named_parameters())
    actual_params = {native_name(name): value for name, value in native_model.named_parameters()}
    assert actual_params.keys() == expected_params.keys(), "Incomplete native/HF parameter coverage"
    with torch.no_grad():
        for name, value in actual_params.items():
            assert value.shape == expected_params[name].shape, name
            value.copy_(expected_params[name])
    generator = torch.Generator(device="cpu").manual_seed(7)
    batch = 1 if variant in {"packed", "odd-length"} else 2
    # Real media exposed CP1 dropping the last RoPE position at odd lengths.
    length = 69 if variant == "odd-length" else 32
    tokens = torch.randint(
        0, config.vocab_size, (batch, length + 1), generator=generator, device="cpu"
    ).to(device)
    inputs = tokens[:, :-1].contiguous()
    # With batch=1 both slices are already contiguous views of tokens. Boundary
    # masking must not overwrite the next document's input with label ignore=-100.
    labels = tokens[:, 1:].clone()
    cu, kwargs = None, {}
    if variant == "packed":
        cu = torch.tensor([0, 7, 19, 32], device=device, dtype=torch.int32)
        lengths = cu[1:] - cu[:-1]
        document = torch.repeat_interleave(torch.arange(3, device=device), lengths)
        position = torch.arange(32, device=device)
        allow = (document[:, None] == document[None, :]) & (position[:, None] >= position[None, :])
        mask = torch.zeros(32, 32, device=device, dtype=dtype).masked_fill(
            ~allow, torch.finfo(dtype).min
        )
        kwargs["attention_mask"] = mask[None, None]
        positions = position - torch.repeat_interleave(cu[:-1], lengths)
        kwargs["position_ids"] = positions.view(1, 1, -1).expand(4, 1, -1)
        labels[:, cu[1:-1].long() - 1] = -100
    hf_logits = hf_model["lm_head"](
        hf_model["model"](input_ids=inputs, use_cache=False, **kwargs).last_hidden_state
    )
    logits = native_model.reference_forward(inputs, cu)
    expected_loss = F.cross_entropy(hf_logits.float().flatten(0, 1), labels.flatten())
    loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
    expected_loss.backward()
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(hf_logits).all()
    logit_diff = difference(logits, hf_logits)
    assert logit_diff < 1e-3, f"logit normalized squared error={logit_diff}"
    torch.testing.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-3)
    worst = (0.0, "")
    for name, parameter in actual_params.items():
        actual, expected = parameter.grad, expected_params[name].grad
        assert actual is not None and expected is not None, f"Missing gradient: {name}"
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
        error = difference(actual, expected)
        worst = max(worst, (error, name))
        assert error < 1e-2, (
            f"{name}: grad error={error}, native max={actual.abs().max().item()}, HF max={expected.abs().max().item()}"
        )
    print(
        f"{variant}: logits_error={logit_diff:.8g}, CE={loss.item():.8g}, HF_CE={expected_loss.item():.8g}, worst_grad={worst}"
    )


def test_native_interleaved_rotary_matches_hf():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerTextRotaryEmbedding as HFRotary,
    )

    from pithtrain.models.qwen3_omni_moe import Qwen3OmniMoeThinkerTextRotaryEmbedding

    config = Qwen3OmniMoeTextConfig(
        **json.loads((Path(__file__).parent / "configs/qwen3_omni_text/tiny.json").read_text())
    )
    positions = torch.arange(3 * 2 * 17, device="cuda").reshape(3, 2, 17)
    native = Qwen3OmniMoeThinkerTextRotaryEmbedding(config).cuda()
    hf = HFRotary(config).cuda()
    expected = hf(
        torch.empty(2, 17, config.hidden_size, device="cuda", dtype=torch.bfloat16), positions
    )
    torch.testing.assert_close(native(positions, torch.bfloat16), expected, rtol=0, atol=0)
