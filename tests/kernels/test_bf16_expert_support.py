"""
Verifies bf16_experts_supported: true on SM100 (the TIRx kernels), and on SM90 with the exact
quack-kernels version the fused BF16 CuTe experts were validated against, for at most 64 local
experts; otherwise false, with at most one warning per device. The query never imports QuACK,
cutlass or tvm.

The version and architecture cases run in a subprocess that patches the probes the query reads
(importlib.metadata.version and torch.cuda.get_device_capability) before importing the library,
so each starts from an empty per-device cache.
"""

import importlib.metadata
import subprocess
import sys
import textwrap

import pytest
import torch

from pithtrain.kernels import bf16_experts_supported, fp8_group_indices

VALIDATED_QUACK = "0.6.5"


def quack_version() -> str | None:
    try:
        return importlib.metadata.version("quack-kernels")
    except importlib.metadata.PackageNotFoundError:
        return None


def h200() -> bool:
    """
    The H200 (SM90) path, told by the library rather than an architecture probe: only SM90's grouped
    GEMM takes the fp8_group_indices hint (None elsewhere).
    """
    offs = torch.tensor([128], dtype=torch.int32, device="cuda")
    return fp8_group_indices(offs, 128) is not None


def run(code: str) -> None:
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


@pytest.mark.skipif(not h200(), reason="needs SM90")
def test_supported_on_h200():
    if quack_version() != VALIDATED_QUACK:
        pytest.skip(f"venv quack-kernels is {quack_version()}, not {VALIDATED_QUACK}")
    assert bf16_experts_supported()
    for num_groups in (1, 8, 16, 32, 64):
        assert bf16_experts_supported(num_groups)


@pytest.mark.parametrize("num_groups", [65, 128, 256])
def test_too_many_groups(num_groups):
    assert not bf16_experts_supported(num_groups)


def test_query_imports_no_kernel_compiler():
    run(
        """
        import sys
        import pithtrain.kernels
        assert not any(m.split(".")[0] in ("quack", "cutlass", "tvm", "tvm_tirx", "tirx_kernels") for m in sys.modules)
        from pithtrain.kernels import bf16_experts_supported
        bf16_experts_supported()
        bf16_experts_supported(16)
        bf16_experts_supported(128)
        bad = sorted(m for m in sys.modules if m.split(".")[0] in ("quack", "cutlass", "tvm", "tvm_tirx", "tirx_kernels"))
        assert not bad, bad
        """
    )


@pytest.mark.skipif(not h200(), reason="needs SM90")
@pytest.mark.parametrize("version", ["0.6.4", "0.7.0", None])
def test_other_quack_version_disables_and_warns_once(version):
    run(
        f"""
        import importlib.metadata, sys, warnings
        real = importlib.metadata.version
        def fake(name):
            if name != "quack-kernels":
                return real(name)
            if {version!r} is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return {version!r}
        importlib.metadata.version = fake
        from pithtrain.kernels import bf16_experts_supported
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert not bf16_experts_supported()
            assert not bf16_experts_supported(16)
        msgs = [str(x.message) for x in w]
        assert len(msgs) == 1 and "quack-kernels" in msgs[0], msgs
        assert not any(m.split(".")[0] in ("quack", "cutlass") for m in sys.modules)
        """
    )


def test_sm100_supported_without_quack():
    run(
        """
        import importlib.metadata, warnings
        import torch
        torch.cuda.get_device_capability = lambda *a, **k: (10, 0)
        real = importlib.metadata.version
        def fake(name):
            if name == "quack-kernels":
                raise importlib.metadata.PackageNotFoundError(name)
            return real(name)
        importlib.metadata.version = fake
        from pithtrain.kernels import bf16_experts_supported
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert bf16_experts_supported()
            assert bf16_experts_supported(64)
            assert not bf16_experts_supported(65)
        assert not w, [str(x.message) for x in w]
        """
    )


def test_other_architecture_disables_and_warns_once():
    run(
        """
        import warnings
        import torch
        torch.cuda.get_device_capability = lambda *a, **k: (12, 0)
        from pithtrain.kernels import bf16_experts_supported
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert not bf16_experts_supported()
            assert not bf16_experts_supported(8)
        msgs = [str(x.message) for x in w]
        assert len(msgs) == 1 and "SM90 or SM100" in msgs[0], msgs
        """
    )
