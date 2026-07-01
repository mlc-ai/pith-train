"""
Process-global runtime state, one module per concern.

This state is a per-process singleton, so keeping it here lets others read it in-line
instead of threading it down through every constructor and call. Import the context module
and read its attributes. Fields are populated once at startup and are unset until then.

For example, to read the global rank and expert-parallel group:

from pithtrain.contexts import distributed
rank, ep_group = distributed.rank, distributed.ep_group
"""
