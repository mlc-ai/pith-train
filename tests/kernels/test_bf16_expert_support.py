"""
Verifies bf16_experts_supported: true only on SM90, with the exact quack-kernels version the fused
BF16 CuTe experts were validated against, and for at most 64 local experts; otherwise false, with
at most one warning per device. The query never imports QuACK or cutlass.

The version and architecture cases run in a subprocess that patches the probes the query reads
(importlib.metadata.version and torch.cuda.get_device_capability) before importing the library,
so each starts from an empty per-device cache.
"""

import importlib.metadata
import subprocess
import sys
import textwrap

import pytest

from pithtrain.kernels import bf16_experts_supported, fp8_fused_experts_supported

VALIDATED_QUACK = "0.6.5"


def quack_version() -> str | None:
    try:
        return importlib.metadata.version("quack-kernels")
    except importlib.metadata.PackageNotFoundError:
        return None


def h200() -> bool:
    """
    The H200 (SM90) path, told by a capability query rather than an architecture probe.
    """
    return fp8_fused_experts_supported()


def run(code: str) -> None:
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


@pytest.mark.skipif(not h200(), reason="needs SM90")
def test_supported_on_h200():
    if quack_version() != VALIDATED_QUACK:
        pytest.skip(f"venv quack-kernels is {quack_version()}, not {VALIDATED_QUACK}")
    assert bf16_experts_supported()
    for num_groups in (1, 8, 16, 32, 64):
        assert bf16_experts_supported(num_groups)


@pytest.mark.skipif(not h200(), reason="needs SM90")
@pytest.mark.parametrize("num_groups", [65, 128, 256])
def test_too_many_groups(num_groups):
    assert not bf16_experts_supported(num_groups)


def test_query_imports_neither_quack_nor_cutlass():
    run(
        """
        import sys
        import pithtrain.kernels
        assert not any(m.split(".")[0] in ("quack", "cutlass") for m in sys.modules)
        from pithtrain.kernels import bf16_experts_supported
        bf16_experts_supported()
        bf16_experts_supported(16)
        bf16_experts_supported(128)
        bad = sorted(m for m in sys.modules if m.split(".")[0] in ("quack", "cutlass"))
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


def test_non_sm90_disables_and_warns_once():
    run(
        """
        import warnings
        import torch
        torch.cuda.get_device_capability = lambda *a, **k: (10, 0)
        from pithtrain.kernels import bf16_experts_supported
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert not bf16_experts_supported()
            assert not bf16_experts_supported(8)
        msgs = [str(x.message) for x in w]
        assert len(msgs) == 1 and "SM90" in msgs[0], msgs
        """
    )
