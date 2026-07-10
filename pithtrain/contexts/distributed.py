"""
Distributed runtime state, populated once at startup.

Import the module and read fields in-line, not the names: a field does not exist until setup
assigns it, so importing it up front fails and reading it early raises AttributeError.

from pithtrain.contexts import distributed
ep_group, ep_size = distributed.ep_group, distributed.ep_size
"""

import torch

rank: int; world_size: int
local_rank: int; local_world_size: int

device_mesh: torch.distributed.DeviceMesh
pp_group: torch.distributed.ProcessGroup
dp_group: torch.distributed.ProcessGroup
cp_group: torch.distributed.ProcessGroup
ep_group: torch.distributed.ProcessGroup

pp_rank: int; pp_size: int
dp_rank: int; dp_size: int
cp_rank: int; cp_size: int
ep_rank: int; ep_size: int
