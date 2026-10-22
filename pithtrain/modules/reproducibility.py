"""Process-local policy for opt-in deterministic training and strict continuation.

This configures PyTorch-managed kernels. Custom FA4/Triton kernels retain their
own explicit policies and still require numerical and fresh-process validation.
"""

import os

import torch

from pithtrain.contexts import training

_CUBLAS_CONFIGS = {":4096:8", ":16:8"}


def configure_deterministic_training(enabled: bool) -> None:
    """Set deterministic policy before model construction; ordinary runs are unchanged.

    For CUDA, launch Python/torchrun with CUBLAS_WORKSPACE_CONFIG=:4096:8
    (or :16:8) before CUDA/cuBLAS initialization. Do not attempt to repair that
    process-start precondition here by changing an already-running environment.
    Unsupported deterministic operations fail rather than silently falling back.
    """
    if not enabled:
        return
    if (
        torch.cuda.is_available()
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in _CUBLAS_CONFIGS
    ):
        raise RuntimeError(
            "Deterministic training on CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "or :16:8 in the launcher before starting Python; the existing "
            "environment is not modified"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_determinism_snapshot() -> dict:
    """Read only a fixed whitelist from the actual reporting rank process."""
    return dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        fa4_deterministic=training.deterministic_attention,
    )
