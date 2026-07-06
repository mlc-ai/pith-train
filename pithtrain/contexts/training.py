"""
Training runtime state, populated once at startup.

Import the module and read fields in-line, not the names: a field does not exist until setup
assigns it, so importing it up front fails and reading it early raises AttributeError.

from pithtrain.contexts import training
self.gate_proj = training.linear_cls(hidden_size, intermediate_size, bias=False)
"""

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pithtrain.dualpipe import DualPipeV
from pithtrain.modules.dataset import ConcatDataset
from pithtrain.operators.grouped_linear import GroupedLinear

# Linear backend: BF16 by default; setup_model selects FP8 (DeepGEMM) when enabled.
fp8: bool = False
linear_cls: type = nn.Linear
grouped_linear_cls: type = GroupedLinear

# Populated once at startup by the training setup.
dataset: ConcatDataset
model: DualPipeV
optimizers: tuple[Optimizer, ...]
schedulers: tuple[LRScheduler, ...]
step: int
