"""Utility classes and functions for DualPipeV pipeline parallelism.

The ``WeightGradStore`` class and the ``run_backward``, ``chunk_tensor``,
``cat_tensor``, ``scatter``, and ``gather`` functions are derived from
``dualpipe/utils.py`` in DeepSeek's DualPipe project
(https://github.com/deepseek-ai/DualPipe), licensed under the MIT License.
Copyright (c) 2025 DeepSeek. See ``pithtrain/dualpipe/LICENSE`` for the
full license text.

``FP8WeightCacheControl`` and the diagnostic utilities (``format_size``,
``print_msg``, ``print_model_size_grad_size_per_device``, etc.) are original
additions.
"""

import queue
from typing import Callable, List

import torch
import torch.distributed as dist
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


def chunk_tensor(x, chunks, dim):
    if x is None:
        return [None for _ in range(chunks)]
    return x.tensor_split(chunks, dim=dim)


def cat_tensor(x, dim):
    if isinstance(x, tuple) or isinstance(x, list):
        if len(x) == 1:
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x, dim=dim)


def scatter(inputs, chunks, dim):
    assert isinstance(inputs, (torch.Tensor, tuple, list))
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    assert all(x is None or isinstance(x, torch.Tensor) for x in inputs)
    inputs = [chunk_tensor(x, chunks, dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
    return microbatches


def gather(micro_outputs, dim):
    assert isinstance(micro_outputs[0], (torch.Tensor, tuple, list))
    if isinstance(micro_outputs[0], torch.Tensor):
        micro_outputs = [(x,) for x in micro_outputs]
    outputs = [x for x in zip(*micro_outputs)]
    outputs = tuple(cat_tensor(x, dim=dim) for x in outputs)
    return outputs


def format_size(bytes_num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes_num)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise ValueError(f"Invalid size: {bytes_num}")


def print_msg(msg: str, rank0_only: bool = False) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank0_only and rank != 0:
        return
    print(f"[rank{rank}] {msg}", flush=True)


def print_model_size_grad_size_per_device(model: nn.Module):
    sizes = {"params": 0, "buffers": 0}
    grad_sizes = 0
    bf16_count = 0
    fp32_count = 0
    bf16_grad_count = 0
    fp32_grad_count = 0
    # Parameters
    for p in model.parameters():
        if p is None:
            continue
        # If this is a DTensor-like object, grab its shard.
        # If not, just use p directly.
        t = getattr(p, "local_tensor", p)
        if torch.is_tensor(t) and not t.is_meta:
            sizes["params"] += t.numel() * t.element_size()
            if t.dtype == torch.bfloat16:
                bf16_count += 1
            elif t.dtype == torch.float32:
                fp32_count += 1

        if p.grad is None:
            continue
        g = p.grad
        g_local = getattr(g, "local_tensor", g)
        if torch.is_tensor(g_local):
            grad_sizes += g_local.numel() * g_local.element_size()
            if g_local.dtype == torch.bfloat16:
                bf16_grad_count += 1
            elif g_local.dtype == torch.float32:
                fp32_grad_count += 1
    # Buffers (e.g., running stats in BatchNorm)
    for b in model.buffers():
        if b is None or b.data is None:
            continue
        sizes["buffers"] += b.numel() * b.element_size()
    total = sizes["params"] + sizes["buffers"]

    print_msg(
        f"model params: {format_size(sizes['params'])}, "
        f"buffers: {format_size(sizes['buffers'])}, "
        f"total: {format_size(total)}. "
        f"gradients: {format_size(grad_sizes)}. "
        f"bf16 params: {bf16_count}, fp32 params: {fp32_count}, bf16 grads: {bf16_grad_count}, fp32 grads: {fp32_grad_count}",
    )


def print_optimizer_state_size_per_device(optimizer: torch.optim.Optimizer):
    # optimizer.state is: {param: {state_name: tensor_or_other}}
    sizes = 0
    for _, state in optimizer.state.items():
        for v in state.values():
            if torch.is_tensor(v):
                t = getattr(v, "local_tensor", v)
                sizes += t.numel() * t.element_size()
            # Some optimizers store nested structures:
            elif isinstance(v, (list, tuple)):
                for t in v:
                    if torch.is_tensor(t):
                        t = getattr(t, "local_tensor", t)
                        sizes += t.numel() * t.element_size()
            elif isinstance(v, dict):
                for t in v.values():
                    if torch.is_tensor(t):
                        t = getattr(t, "local_tensor", t)
                        sizes += t.numel() * t.element_size()
    print_msg(f"optimizer state: {format_size(sizes)}")


def print_cuda_memory_usage(name: str, rank0_only: bool = False) -> None:
    print_msg(
        f"CUDA mem usage {name}: {format_size(torch.cuda.memory_allocated())},"
        f" actual free mem: {format_size(torch.cuda.mem_get_info()[0])}",
        rank0_only,
    )
