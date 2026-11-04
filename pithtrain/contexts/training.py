from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from pithtrain.operators import grouped_linear, linear
    from pithtrain.pipeline import DualPipeV

__all__ = [
    "PARAM_DTYPE",
    "fp8",
    "Linear",
    "GroupedLinear",
    "model",
    "optimizers",
    "schedulers",
    "current_microbatch",
]

PARAM_DTYPE = torch.bfloat16
"""
Parameter compute and activation dtype.

Pipeline parallelism cuts at layer boundaries, so the tensors crossing between stages are layer
inputs and carry this dtype. DualPipeV allocates its receive buffers with it.
"""

fp8: bool
"""
Whether layers are built for FP8 compute, False for BF16.

This and the two class bindings below are read by models and operators that hold no configuration
object, so they live here rather than being passed down.
"""


Linear: type[linear.BF16Linear | linear.FP8Linear]

GroupedLinear: type[grouped_linear.GroupedLinear | grouped_linear.FP8GroupedLinear]

model: DualPipeV

optimizers: tuple[Optimizer, ...]
"""
One per parameter class under Muon.
"""

schedulers: tuple[LRScheduler, ...]
"""
One learning-rate scheduler per optimizer, in the same order.
"""

current_microbatch: int | None = None
"""
Index into the micro-batch list of the step in flight, or None outside a step, for code that cannot
tell the micro-batch from the layer alone: DualPipeV interleaves a rank's two chunks across
micro-batches. It is set before each forward only, which suffices because the backward replays the
saved graph without re-entering module code.
"""

offloaded: bool = False
"""
True between offload_training_state and reload_training_state.

Tensors keep their shapes over zero-byte device storages, so model.step, save_checkpoint,
load_checkpoint and iter_canonical_parameters reject this state.
"""
