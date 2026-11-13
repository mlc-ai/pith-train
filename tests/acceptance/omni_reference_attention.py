"""Versioned HF attention reference contracts; no production model changes.

Legacy remains runnable. Matched v2 uses the installed official HF public API,
never a PithTrain callback or Hub fallback; original numerical gates remain.
"""

import hashlib
import importlib
import importlib.metadata
import inspect
import os
import sys
from pathlib import Path

import torch

LEGACY = "legacy-eager-v1"
MATCHED = "native-matched-v2"
CONTRACTS = (LEGACY, MATCHED)

# Reviewed installed primary sources. Changing a pin requires reviewing these
# sources again, not silently accepting the next environment's implementation.
SOURCES = {
    "transformers": (
        "5.17.0",
        {
            "transformers/modeling_utils.py": "38c2bd02ed7af229f54e2f02dae65663be7af29ed2c9498bd1c460cf60fbb62d",
            "transformers/modeling_flash_attention_utils.py": "b5301ec90664cfe088dc5bd91cb4851cc8affb5ac44e2a1c1c422ba20fa99c11",
            "transformers/integrations/flash_attention.py": "a58018813c0aff5c29aa983bc0a82ce5b4b573d7665b8e9a4fb8fca0c157e98c",
            "transformers/integrations/sdpa_attention.py": "53c7229daca9ade4c5df874194448938c1edc925abbc71809f9750dd66381e6f",
            "transformers/masking_utils.py": "50a737f63d8c778a5597fa34ac139049af921f44958205e3e1c29fe2bae77254",
            "transformers/utils/import_utils.py": "e147f99cf68352e3260e9780930501c3becc0c40a5ff04150ff641b724161732",
            "transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py": "1c1fc322dfa547c697576e8ad3c88b817aa382abc9db5d966755be3d9fabf7ce",
        },
    ),
    "flash-attn-4": (
        "4.0.0b28",
        {
            "flash_attn/cute/interface.py": "b0af7198dd03d639f549b70ada9e64f2ab7674936c6a508205190b537d8f45f8",
            "flash_attn/cute/flash_bwd_sm90.py": "2170fcc979bd6843e73fb81133e3af23fc4d73e29b537a41a5e9a3f06492d840",
            "flash_attn/cute/copy_utils.py": "0f563eeeb6654602d1541f7202b7b88386c4dbdbd24ad6b0825834d65e508c09",
        },
    ),
}


def add_reference_attention_argument(parser):
    parser.add_argument("--reference-attention", choices=CONTRACTS, default=LEGACY)


def reference_contract(args):
    # Existing Python callers/tests retain their original eager behavior.
    return getattr(args, "reference_attention", LEGACY)


def verify_sources():
    """Verify distribution membership and reviewed source bytes before imports."""
    result = {}
    for name, (version, expected) in SOURCES.items():
        distribution = importlib.metadata.distribution(name)
        normalized = distribution.metadata["Name"].lower().replace("_", "-")
        if normalized != name or distribution.version != version:
            raise RuntimeError(f"Unreviewed distribution: {name} {distribution.version}")
        members = {str(path) for path in distribution.files or ()}
        for relative, digest in expected.items():
            if relative not in members:
                raise RuntimeError(f"Source is not a distribution member: {relative}")
            path = Path(distribution.locate_file(relative)).resolve()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise RuntimeError(f"Unreviewed source bytes: {relative}")
            result[relative] = {"path": str(path), "sha256": actual}
    return result


def _verify_function(function, path, required, module):
    if not callable(function) or function.__module__ != module:
        raise RuntimeError(f"Unexpected resolved function: {module}")
    if Path(inspect.getsourcefile(function)).resolve() != Path(path).resolve():
        raise RuntimeError(f"Unexpected function source: {module}")
    parameters = inspect.signature(function).parameters
    if not set(required).issubset(parameters):
        raise RuntimeError(f"Unsupported local FA4 API: {sorted(parameters)}")


def verify_local_fa4():
    """Fail before public setters if the direct local/offline path is unavailable."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required before importing HF")
    sources = verify_sources()
    from huggingface_hub import is_offline_mode
    from huggingface_hub.constants import HF_HUB_OFFLINE
    from transformers.utils import is_flash_attn_4_available

    if not HF_HUB_OFFLINE or not is_offline_mode():
        raise RuntimeError("HF offline mode was not active at import time")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("Local FA4 requires CUDA SM90 or newer")
    if torch.__version__ != "2.13.0+cu130" or importlib.metadata.version("triton") != "3.7.1":
        raise RuntimeError("Matched reference requires reviewed Torch 2.13.0+cu130 / Triton 3.7.1")
    if not is_flash_attn_4_available():
        raise RuntimeError("Local FA4 unavailable; Hub fallback is forbidden")
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    from transformers.integrations.flash_attention import flash_attention_forward
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for function, extra in (
        (flash_attn_func, ()),
        (
            flash_attn_varlen_func,
            ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"),
        ),
    ):
        _verify_function(
            function,
            sources["flash_attn/cute/interface.py"]["path"],
            ("q", "k", "v", "softmax_scale", "causal", "deterministic", *extra),
            "flash_attn.cute.interface",
        )
    resolved, _ = lazy_import_flash_attention("flash_attention_4", allow_all_kernels=False)
    if resolved[0] is not flash_attn_func or resolved[1] is not flash_attn_varlen_func:
        raise RuntimeError("Resolved FA4 differs from reviewed local functions")
    if (
        ALL_ATTENTION_FUNCTIONS.get_interface("flash_attention_4", None)
        is not flash_attention_forward
    ):
        raise RuntimeError("HF FA4 interface was replaced")
    if ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None) is not sdpa_attention_forward:
        raise RuntimeError("HF SDPA interface was replaced")
    # Loaded module origins must match the distribution files we verified.
    for relative, identity in sources.items():
        module_name = relative.removesuffix(".py").replace("/", ".")
        if module_name in sys.modules:
            module = importlib.import_module(module_name)
            if Path(module.__file__).resolve() != Path(identity["path"]):
                raise RuntimeError(f"Shadowed local module: {module_name}")
    return {
        "versions": {name: version for name, (version, _) in SOURCES.items()},
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "sources": sources,
        "capability": list(torch.cuda.get_device_capability()),
        "hardware": torch.cuda.get_device_name(),
        "offline": True,
        "hub_fallback": "FORBIDDEN",
        "dense_function": f"{flash_attn_func.__module__}.{flash_attn_func.__name__}",
        "varlen_function": f"{flash_attn_varlen_func.__module__}.{flash_attn_varlen_func.__name__}",
    }


def configure_reference_attention(reference, native_config, contract=LEGACY):
    if contract not in CONTRACTS:
        raise ValueError(f"Unknown reference contract: {contract}")
    text = reference.model
    components = {"text": text}
    if hasattr(reference, "visual"):
        components["vision"] = reference.visual
    if hasattr(reference, "audio_tower"):
        components["audio"] = reference.audio_tower
    requested = {name: "eager" for name in components}
    runtime = None
    deterministic = None
    if contract == MATCHED:
        from pithtrain.contexts import training

        runtime = verify_local_fa4()
        deterministic = training.deterministic_attention
        if not isinstance(deterministic, bool):
            raise RuntimeError("Native attention determinism must be an explicit boolean")
        requested["text"] = "flash_attention_4"
        if text.config.attention_dropout != 0:
            raise RuntimeError("The declared native FA4 contract requires zero attention dropout")
        if "vision" in components:
            requested["vision"] = native_config.vision_config._attn_implementation or "sdpa"
            if requested["vision"] not in ("sdpa", "eager"):
                raise RuntimeError("Unsupported native vision attention configuration")
        from transformers.modeling_utils import PreTrainedModel

        for name, module in components.items():
            if (
                module.set_attn_implementation.__func__
                is not PreTrainedModel.set_attn_implementation
            ):
                raise RuntimeError(f"Nonofficial HF attention setter: {name}")
            module.set_attn_implementation(requested[name], allow_all_kernels=False)
    resolved = {name: module.config._attn_implementation for name, module in components.items()}
    if resolved != requested:
        raise RuntimeError(f"HF backend mismatch: {resolved} != {requested}")
    result = {
        "contract": contract,
        "components": resolved,
        "deterministic": deterministic,
        "backward_policy": {
            "native": deterministic,
            "hf_explicit": deterministic,
            "matched": contract == MATCHED,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "runtime": runtime,
        "independent_model_reference": True,
        "independent_fa4_math_reference": False,
        "historical_eager_results": "PRESERVED_SEPARATELY",
    }
    if "vision" in components:
        result["vision_alignment"] = {
            "scope": "API_AND_CONFIG_MATCH" if contract == MATCHED else "LEGACY_CROSS_BACKEND",
            "native_qkv_rank": 4,
            "hf_qkv_rank": 4,
            "identical_cuda_sdpa_kernel": "NOT_ESTABLISHED",
            "actual_cuda_sdpa_kernel": "NOT_OBSERVED",
            "enabled_sdpa_backends": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            },
            "original_numerical_thresholds": "REQUIRED_UNCHANGED",
        }
    reference.reference_attention = result
    return result


def reference_attention_kwargs(reference):
    report = getattr(reference, "reference_attention", None)
    return (
        {"deterministic": report["deterministic"]}
        if report and report["contract"] == MATCHED
        else {}
    )
