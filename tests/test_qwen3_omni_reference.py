"""Scaffold for comparing native Qwen3-Omni Thinker text against Hugging Face.

Run: python -m pytest tests/test_qwen3_omni_reference.py -q -rs

The only test is skipped until native construction, weight/gradient mapping and
GPU tolerances are implemented. No HF-only correctness tests run in this file.
HF supplies the expected results for the future native-model comparison.

Start with tiny on one GPU. Keep full.json for a later resource-controlled run;
loading the full fixture is not itself a full-model execution test. Both configs
describe Thinker text and originate from Qwen/Qwen3-Omni-30B-A3B-Instruct revision
26291f793822fb6be9555850f06dfe95f2d7e695 (tiny reduces the model dimensions).
"""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel


def build_native_from_hf(config: Qwen3OmniMoeTextConfig, hf_model: nn.ModuleDict) -> nn.Module:
    """TODO: Set up PP=EP=CP=1 and the native backend, then build Omni at phase=-1.

    Use the HF model's device/dtype and copy its weights, mapping expert layouts and
    parameter names as needed. Require complete weight coverage; matching seeds alone
    does not give the two implementations identical weights.
    """
    raise NotImplementedError("Native Omni construction and HF weight mapping are not implemented")


def native_gradients_in_hf_layout(native_model: nn.Module) -> dict[str, torch.Tensor]:
    """TODO: Map every native parameter gradient back to HF names and tensor layouts."""
    raise NotImplementedError("Native Omni gradient mapping is not implemented")


@pytest.mark.skip(reason="TODO: implement native Omni adapters and validate GPU tolerances")
def test_native_omni_matches_hf():
    """Same weights, inputs and next-token objective; compare both implementations."""
    path = Path(__file__).parent / "configs" / "qwen3_omni_text" / "tiny.json"
    config = Qwen3OmniMoeTextConfig(**json.loads(path.read_text()))
    device, dtype = torch.device("cuda"), torch.bfloat16

    # Construct the HF side from local config with synthetic weights, without downloads.
    hf_config = copy.deepcopy(config)
    hf_config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(0)
        decoder = Qwen3OmniMoeThinkerTextModel(hf_config).float()
        head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)
        nn.init.normal_(head.weight, std=config.initializer_range)
        if config.tie_word_embeddings:
            head.weight = decoder.embed_tokens.weight
    hf_model = nn.ModuleDict({"model": decoder, "lm_head": head}).to(device=device, dtype=dtype)
    native_model = build_native_from_hf(config, hf_model)
    hf_model.train()
    native_model.train()

    # Share one batch and shift labels once, matching the existing pretraining loader.
    generator = torch.Generator(device="cpu").manual_seed(1)
    tokens = torch.randint(0, config.vocab_size, (2, 17), generator=generator, device="cpu")
    tokens = tokens.to(device)
    input_ids, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    # Both sides use dense causal attention, the same positions and no router auxiliary loss.
    hidden = hf_model["model"](
        input_ids=input_ids, use_cache=False, output_router_logits=False, return_dict=True
    ).last_hidden_state
    hf_logits = hf_model["lm_head"](hidden)
    native_logits = native_model.reference_forward(input_ids)
    hf_loss = F.cross_entropy(hf_logits.float().flatten(0, 1), labels.flatten())
    native_loss = F.cross_entropy(native_logits.float().flatten(0, 1), labels.flatten())
    hf_loss.backward()
    native_loss.backward()

    # TODO: Validate and document explicit GPU tolerances for logits, loss and gradients
    # before enabling this test. Defaults below only show the comparison wiring.
    torch.testing.assert_close(native_logits, hf_logits)
    torch.testing.assert_close(native_loss, hf_loss)
    hf_gradients = {name: parameter.grad for name, parameter in hf_model.named_parameters()}
    native_gradients = native_gradients_in_hf_layout(native_model)
    assert native_gradients.keys() == hf_gradients.keys(), "Incomplete gradient mapping"
    for name, expected in hf_gradients.items():
        actual = native_gradients[name]
        assert expected is not None and actual is not None, f"Missing gradient: {name}"
        torch.testing.assert_close(actual, expected, msg=name)


# TODO: After the single-GPU comparison passes, add Omni to tests/test_dualpipev.py
# and tests/test_dualpipev.sh for native-reference-vs-pipeline PP/EP/CP validation.
