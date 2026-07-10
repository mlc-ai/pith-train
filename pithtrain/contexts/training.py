"""
Training runtime state, populated once at startup.

Import the module and read fields in-line, not the names: a field does not exist until setup
assigns it, so importing it up front fails and reading it early raises AttributeError.

from pithtrain.contexts import training
self.gate_proj = training.Linear(hidden_size, intermediate_size, bias=False)
"""

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pithtrain.dualpipe import DualPipeV
from pithtrain.modules.dataset import ConcatDataset
from pithtrain.operators import grouped_linear

fp8: bool = False
Linear: type[nn.Linear] = nn.Linear
GroupedLinear: type[grouped_linear.GroupedLinear | grouped_linear.FP8GroupedLinear] = grouped_linear.GroupedLinear

model: DualPipeV
dataset: ConcatDataset
optimizers: tuple[Optimizer, ...]
schedulers: tuple[LRScheduler, ...]

step: int
