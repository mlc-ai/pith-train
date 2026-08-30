"""Training runtime state."""

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pithtrain.dualpipe import DualPipeV
from pithtrain.modules.dataset import ConcatDataset, WeightedMixtureDataset
from pithtrain.operators import grouped_linear, linear

fp8: bool
Linear: type[nn.Linear | linear.FP8Linear]
GroupedLinear: type[grouped_linear.GroupedLinear | grouped_linear.FP8GroupedLinear]

model: DualPipeV
dataset: ConcatDataset | WeightedMixtureDataset
optimizers: tuple[Optimizer, ...]
schedulers: tuple[LRScheduler, ...]

step: int
