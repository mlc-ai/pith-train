"""Distributed runtime state."""

import torch

rank: int
"""Global worker rank."""

world_size: int
"""Total number of workers."""

local_rank: int
"""Worker rank within the node."""

local_world_size: int
"""Number of workers on the node."""

device_mesh: torch.distributed.DeviceMesh
"""4D mesh over (PP, DP, CP, EP) axes."""

pp_rank: int
pp_size: int
pp_group: torch.distributed.ProcessGroup

dp_rank: int
dp_size: int
dp_group: torch.distributed.ProcessGroup

cp_rank: int
cp_size: int
cp_group: torch.distributed.ProcessGroup

ep_rank: int
ep_size: int
ep_group: torch.distributed.ProcessGroup
