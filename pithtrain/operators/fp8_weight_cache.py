"""
Cache control for FP8-quantized weights.

Quantizing a weight is per-step work, not per-micro-batch work. This module holds the version
counter that lets every FP8 linear module skip the redundant casts.
"""

import torch.nn as nn


class FP8WeightCacheControl:
    """
    Global version counter for caching FP8-quantized weights across micro-batches.

    Weights do not change between the micro-batches of one training step, so each FP8
    linear module can quantize its weight once and reuse the result. Whoever drives the
    step must call step() before the first micro-batch to invalidate the previous
    step's caches; DualPipeV.step() does this.
    """

    version: int = 0

    @classmethod
    def step(cls):
        """
        Increment version to invalidate all module caches.
        """
        cls.version += 1

    @classmethod
    def clear(cls, *modules: nn.Module) -> None:
        """
        Release all cached FP8 weight tensors from modules to free GPU memory.

        Should be called after the pipeline step completes and before optimizer.step()
        so the memory is available for optimizer temporaries.
        """
        for module in modules:
            for m in module.modules():
                if hasattr(m, "_wq_cache"):
                    m._wq_cache = None
