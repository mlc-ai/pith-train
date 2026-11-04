"""
Deferred weight gradients for zero-bubble pipeline scheduling.

Expert layers whose backward defers its weight-gradient GEMMs put them here as closures, and
DualPipeV runs them in its W phase.

WeightGradStore is derived from dualpipe/utils.py in DeepSeek's DualPipe project
(https://github.com/deepseek-ai/DualPipe), licensed under the MIT License. Copyright (c) 2025
DeepSeek. See pithtrain/pipeline/LICENSE for the full license text.
"""

import queue
from typing import Callable, List

__all__ = ["WeightGradStore"]


class WeightGradStore:
    """
    While enabled is True, a backward defers its weight gradients with put; flush closes the current
    group and pop runs the oldest group.
    """

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
