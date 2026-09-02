"""Training runtime state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from pithtrain.dualpipe import DualPipeV
    from pithtrain.modules.dataset import ConcatDataset
    from pithtrain.operators import grouped_linear, linear

PARAM_DTYPE = torch.bfloat16
"""
Parameter compute and activation dtype.

Pipeline parallelism cuts at layer boundaries, so the tensors crossing between stages are layer
inputs and carry this dtype. DualPipeV allocates its receive buffers with it.
"""

fp8: bool
Linear: type[nn.Linear | linear.FP8Linear]
GroupedLinear: type[grouped_linear.GroupedLinear | grouped_linear.FP8GroupedLinear]

model: DualPipeV
dataset: ConcatDataset
optimizers: tuple[Optimizer, ...]
schedulers: tuple[LRScheduler, ...]

step: int
