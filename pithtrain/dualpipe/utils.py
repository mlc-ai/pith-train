"""
Utility classes and functions for DualPipeV pipeline parallelism.

The ``WeightGradStore`` class and the ``run_backward`` function are derived
from ``dualpipe/utils.py`` in DeepSeek's DualPipe project
(https://github.com/deepseek-ai/DualPipe), licensed under the MIT License.
Copyright (c) 2025 DeepSeek. See ``pithtrain/dualpipe/LICENSE`` for the
full license text.

``FP8WeightCacheControl`` is an original addition.
"""

import queue
from typing import Callable, List

import torch
import torch.nn as nn
from torch.autograd import Variable


class FP8WeightCacheControl:
    """Global version counter for caching FP8-quantized weights across micro-batches.

    Within a single DualPipeV.step(), weights don't change between micro-batches
    (optimizer steps only after all micro-batches). This allows each FP8 linear
    module to quantize its weight once and reuse the result for subsequent chunks.

    Call ``step()`` at the start of each ``DualPipeV.step()`` to invalidate stale
    caches from the previous training step.
    """

    version: int = 0

    @classmethod
    def step(cls):
        """Increment version to invalidate all module caches."""
        cls.version += 1

    @classmethod
    def clear(cls, *modules: nn.Module) -> None:
        """Release all cached FP8 weight tensors from modules to free GPU memory.

        Should be called after the pipeline step completes and before
        ``optimizer.step()`` so the memory is available for optimizer temporaries.
        The caches will be regenerated on the next forward pass.
        """
        for module in modules:
            for m in module.modules():
                if hasattr(m, "_wq_cache"):
                    m._wq_cache = None


class WeightGradStore:
    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    pairs = [(t, g) for t, g in zip(tensors, grad_tensors) if t is not None]
    if not pairs:
        return
    tensors, grad_tensors = map(tuple, zip(*pairs))
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    with torch.autograd.set_multithreading_enabled(False):
        Variable._execution_engine.run_backward(tensors, grad_tensors, **kwargs)
