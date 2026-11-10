"""Load the real task loss helpers without importing the GPU training runtime."""

import ast
from pathlib import Path

path = Path(__file__).parents[1] / "pithtrain/tasks/pretrain_lm.py"
names = {"global_target_count", "global_loss_mean"}
nodes = [
    node
    for node in ast.parse(path.read_text(), filename=str(path)).body
    if (isinstance(node, ast.FunctionDef) and node.name in names)
    or (
        isinstance(node, ast.Import)
        and all(alias.name in {"torch", "torch.distributed"} for alias in node.names)
    )
]
namespace = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
global_target_count = namespace["global_target_count"]
global_loss_mean = namespace["global_loss_mean"]
dist = namespace["dist"]
