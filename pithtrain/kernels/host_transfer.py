"""
Pinned host buffers for the asynchronous device-to-host copies of the MoE ops.

They are process-global, not per-thread: safe only because the MoE ops run on one thread at a time
(DualPipeV runs backwards with autograd multithreading disabled). Two threads using them at once
would race and corrupt the read-back.
"""

import torch

__all__ = ["get_pinned_buffer"]

pinned_buffers: dict[tuple[str, torch.dtype, int], torch.Tensor] = {}


def get_pinned_buffer(name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
    """
    The same tensor on every call with the same arguments. Cached because freeing a pinned tensor
    costs a cudaEventRecordWithFlags in PyTorch's allocator.
    """
    key = (name, dtype, numel)
    buf = pinned_buffers.get(key)
    if buf is None:
        buf = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
        pinned_buffers[key] = buf
    return buf
