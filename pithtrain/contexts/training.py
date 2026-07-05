"""
Training runtime state, populated once at startup.

Import the module and read fields in-line, not the names. The runtime fields
(``dataset``/``model``/``optimizers``/``schedulers``/``step``) do not exist until setup assigns
them, so reading them early raises AttributeError. The linear backend defaults to BF16 and is
switched to FP8 by ``setup_model``.

    from pithtrain.contexts import training
    self.gate_proj = training.linear_cls(hidden_size, intermediate_size, bias=False)
"""

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from pithtrain.dualpipe import DualPipeV
from pithtrain.layers.group_linear import GroupLinear
from pithtrain.modules.dataset import ConcatDataset

# Linear backend: BF16 by default; setup_model selects FP8 (DeepGEMM) when enabled.
fp8: bool = False
linear_cls: type = nn.Linear
group_linear_cls: type = GroupLinear

# Populated once at startup by the training setup.
dataset: ConcatDataset
model: DualPipeV
optimizers: tuple[Optimizer, ...]
schedulers: tuple[LRScheduler, ...]
step: int
