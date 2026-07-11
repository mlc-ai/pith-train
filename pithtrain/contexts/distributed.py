"""Distributed runtime state."""

import torch

rank: int
world_size: int
local_rank: int
local_world_size: int
device: torch.device

device_mesh: torch.distributed.DeviceMesh
pp_group: torch.distributed.ProcessGroup
dp_group: torch.distributed.ProcessGroup
cp_group: torch.distributed.ProcessGroup
ep_group: torch.distributed.ProcessGroup

pp_rank: int
pp_size: int
dp_rank: int
dp_size: int
cp_rank: int
cp_size: int
ep_rank: int
ep_size: int
