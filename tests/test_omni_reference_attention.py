"""CPU acceptance-contract tests; no mocked result is called a CUDA PASS."""

import argparse
import hashlib
import importlib.metadata
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTextConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeVisionEncoder,
)

from pithtrain.contexts import training
from tests.acceptance import omni_reference_attention as contract
from tests.test_qwen3_omni_vision import vision_config


def small_reference():
    text = Qwen3OmniMoeTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000,
            "mrope_section": [2, 3, 3],
        },
    )
    text._attn_implementation = "eager"
    vision = vision_config()
    vision._attn_implementation = "eager"
    owner = nn.Module()
    owner.model = Qwen3OmniMoeThinkerTextModel(text)
    owner.visual = Qwen3OmniMoeVisionEncoder(vision)
    return owner, SimpleNamespace(vision_config=vision)


def test_default_cli_and_legacy_execution_remain_eager_without_cuda(monkeypatch):
    parser = argparse.ArgumentParser()
    contract.add_reference_attention_argument(parser)
    assert parser.parse_args([]).reference_attention == contract.LEGACY
    reference, config = small_reference()
    monkeypatch.setattr(contract, "verify_local_fa4", lambda: pytest.fail("Legacy imported FA4"))
    before = {name: value.clone() for name, value in reference.state_dict().items()}
    report = contract.configure_reference_attention(reference, config)
    assert report["contract"] == contract.LEGACY
    assert report["components"] == {"text": "eager", "vision": "eager"}
    assert contract.reference_attention_kwargs(reference) == {}
    assert all(torch.equal(value, before[name]) for name, value in reference.state_dict().items())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "native_vision,expected", [(None, "sdpa"), ("sdpa", "sdpa"), ("eager", "eager")]
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_matched_uses_each_public_setter_preserves_weights_and_explicit_policy(
    monkeypatch, native_vision, expected, deterministic
):
    reference, config = small_reference()
    # A pure contract simulation: the real CUDA API is deliberately not invoked.
    config = SimpleNamespace(vision_config=SimpleNamespace(_attn_implementation=native_vision))
    calls = []

    def setter(self, name, allow_all_kernels=False):
        calls.append((self, name, allow_all_kernels))
        self.config._attn_implementation = name

    monkeypatch.setattr(PreTrainedModel, "set_attn_implementation", setter)
    monkeypatch.setattr(contract, "verify_local_fa4", lambda: {"scope": "CPU_CONTRACT_SIMULATION"})
    monkeypatch.setattr(training, "deterministic_attention", deterministic)
    before = {name: value.clone() for name, value in reference.state_dict().items()}
    report = contract.configure_reference_attention(reference, config, contract.MATCHED)
    assert [(name, allow) for _, name, allow in calls] == [
        ("flash_attention_4", False),
        (expected, False),
    ]
    assert report["components"] == {"text": "flash_attention_4", "vision": expected}
    assert report["vision_alignment"]["identical_cuda_sdpa_kernel"] == "NOT_ESTABLISHED"
    assert report["vision_alignment"]["actual_cuda_sdpa_kernel"] == "NOT_OBSERVED"
    assert set(report["vision_alignment"]["enabled_sdpa_backends"]) == {
        "flash",
        "memory_efficient",
        "math",
        "cudnn",
    }
    assert contract.reference_attention_kwargs(reference) == {"deterministic": deterministic}
    assert all(torch.equal(value, before[name]) for name, value in reference.state_dict().items())


def test_real_official_vision_setter_changes_backend_without_changing_any_weight():
    reference, _ = small_reference()
    before = {name: value.clone() for name, value in reference.visual.state_dict().items()}
    reference.visual.set_attn_implementation("sdpa")
    assert reference.visual.config._attn_implementation == "sdpa"
    assert all(
        torch.equal(value, before[name]) for name, value in reference.visual.state_dict().items()
    )


@pytest.mark.parametrize("bad", ["kernels-community/flash-attn4", "eager", "flash_attention_2"])
def test_resolved_backend_substitution_cannot_pass(monkeypatch, bad):
    reference, config = small_reference()
    monkeypatch.setattr(contract, "verify_local_fa4", lambda: {})

    def wrong(self, name, allow_all_kernels=False):
        self.config._attn_implementation = bad

    monkeypatch.setattr(PreTrainedModel, "set_attn_implementation", wrong)
    with pytest.raises(RuntimeError, match="backend mismatch"):
        contract.configure_reference_attention(reference, config, contract.MATCHED)


def test_missing_distribution_fails_before_any_loader_or_setter(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []

    def missing(name):
        calls.append(name)
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        contract.verify_local_fa4()
    assert calls == ["transformers"]
    assert not torch.cuda.is_initialized()


def test_online_environment_rejected_before_distribution_or_loader(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(
        contract, "verify_sources", lambda: pytest.fail("Should fail before imports")
    )
    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"):
        contract.verify_local_fa4()


@pytest.mark.parametrize("fault", ["version", "membership", "bytes", "name"])
def test_source_identity_rejects_unreviewed_local_package(monkeypatch, tmp_path, fault):
    path = tmp_path / "reviewed.py"
    path.write_text("reviewed = True\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(contract, "SOURCES", {"package": ("1", {"reviewed.py": digest})})
    distribution = SimpleNamespace(
        metadata={"Name": "wrong" if fault == "name" else "package"},
        version="2" if fault == "version" else "1",
        files=[] if fault == "membership" else [Path("reviewed.py")],
        locate_file=lambda name: tmp_path / name,
    )
    if fault == "bytes":
        path.write_text("reviewed = False\n")
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    with pytest.raises(RuntimeError, match="Unreviewed|distribution member"):
        contract.verify_sources()


def test_matching_distribution_member_and_bytes_are_recorded(monkeypatch, tmp_path):
    path = tmp_path / "reviewed.py"
    path.write_text("x = 1\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(contract, "SOURCES", {"package": ("1", {"reviewed.py": digest})})
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            metadata={"Name": "package"},
            version="1",
            files=[Path("reviewed.py")],
            locate_file=lambda name: tmp_path / name,
        ),
    )
    assert contract.verify_sources()["reviewed.py"] == {"path": str(path), "sha256": digest}


def test_wrong_resolved_function_source_and_api_fail_closed():
    def fake(q, k, v):
        return q

    with pytest.raises(RuntimeError, match="resolved function"):
        contract._verify_function(fake, __file__, ("q",), "flash_attn.cute.interface")
    with pytest.raises(RuntimeError, match="function source"):
        contract._verify_function(fake, "/not/the/source.py", ("q",), __name__)
    with pytest.raises(RuntimeError, match="Unsupported local FA4 API"):
        contract._verify_function(fake, __file__, ("deterministic",), __name__)


def test_unknown_contract_rejected_before_mutation():
    reference, config = small_reference()
    with pytest.raises(ValueError, match="Unknown reference"):
        contract.configure_reference_attention(reference, config, "best-effort")


def simulated_cuda_dependencies(monkeypatch):
    import huggingface_hub.constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub, "is_offline_mode", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.7.1")


def test_unavailable_fa4_never_attempts_official_or_hub_loader(monkeypatch):
    import transformers.modeling_flash_attention_utils as flash_utils
    import transformers.utils

    simulated_cuda_dependencies(monkeypatch)
    monkeypatch.setattr(contract, "verify_sources", lambda: {})
    monkeypatch.setattr(transformers.utils, "is_flash_attn_4_available", lambda: False)
    monkeypatch.setattr(
        flash_utils,
        "lazy_import_flash_attention",
        lambda *a, **k: pytest.fail("Fallback attempted"),
    )
    with pytest.raises(RuntimeError, match="Hub fallback is forbidden"):
        contract.verify_local_fa4()


def test_cached_replaced_fa4_function_rejected_after_official_loader(monkeypatch):
    import transformers.modeling_flash_attention_utils as flash_utils
    import transformers.utils

    simulated_cuda_dependencies(monkeypatch)
    monkeypatch.setattr(transformers.utils, "is_flash_attn_4_available", lambda: True)
    monkeypatch.setattr(
        contract,
        "verify_sources",
        lambda: {"flash_attn/cute/interface.py": {"path": __file__, "sha256": "CPU_SIMULATION"}},
    )

    def dense(q, k, v, softmax_scale=None, causal=False, deterministic=False):
        raise AssertionError("CPU contract must not execute a fake kernel")

    def varlen(
        q,
        k,
        v,
        softmax_scale=None,
        causal=False,
        deterministic=False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
    ):
        raise AssertionError("CPU contract must not execute a fake kernel")

    dense.__module__ = varlen.__module__ = "flash_attn.cute.interface"
    fake = ModuleType("flash_attn.cute")
    fake.flash_attn_func, fake.flash_attn_varlen_func = dense, varlen
    monkeypatch.setitem(sys.modules, "flash_attn.cute", fake)
    monkeypatch.setattr(
        flash_utils,
        "lazy_import_flash_attention",
        lambda *a, **k: ((varlen, varlen, None, None, None), None),
    )
    with pytest.raises(RuntimeError, match="Resolved FA4 differs"):
        contract.verify_local_fa4()
