"""Strict torchrun adapter for the reviewed native DualPipeV numerical test.

This entry operates only in an already allocated GPU environment. It neither
creates cloud resources nor changes the original training fixture or tolerances.
"""

import argparse
import ast
import hashlib
import os
import sys
from pathlib import Path

from tests.acceptance.omni_topology_contract import BY_NAME, MODEL, TEST_SHA256, verify_source


def overlay(source):
    if hashlib.sha256(source.encode()).hexdigest() != TEST_SHA256:
        raise ValueError("Frozen test hash mismatch; refusing to adapt another source")
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    starts = [
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.Assign) and ast.dump(n.targets[0]) == "Name(id='eps', ctx=Store())"
    ]
    stops = [
        i
        for i, n in enumerate(main.body)
        if isinstance(n, ast.Assert)
        and ast.dump(n.test) == "UnaryOp(op=Not(), operand=Name(id='failed', ctx=Load()))"
    ]
    if len(starts) != 1 or len(stops) != 1 or stops[0] <= starts[0]:
        raise ValueError("Expected gradient block not found exactly once")
    replacement = ast.parse(
        "largest_diff, largest_diff_param = __strict_hook(local_modules, local_full_modules, calculate_difference)"
    ).body
    main.body[starts[0] : stops[0] + 1] = replacement
    return ast.fix_missing_locations(tree)


def validate_torchrun_environment(case, environment):
    required = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    if any(not environment.get(key) for key in required):
        raise RuntimeError("An existing single-node torchrun environment is required")
    world, local_world = (int(environment[key]) for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"))
    rank, local_rank = (int(environment[key]) for key in ("RANK", "LOCAL_RANK"))
    if world != case.world or local_world != world:
        raise RuntimeError("torchrun world does not match the single-node fixed case")
    if rank != local_rank or not 0 <= rank < world:
        raise RuntimeError("Invalid single-node torchrun rank")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=BY_NAME, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = verify_source(repo, args.expected_commit)
    case = BY_NAME[args.case]
    validate_torchrun_environment(case, os.environ)
    os.environ.update(case.environment())
    args.out.mkdir(parents=True, exist_ok=True)
    source_path = repo / "tests/test_dualpipev.py"
    tree = overlay(source_path.read_text())
    from tests.acceptance.omni_topology_gradients import runtime_check

    def hook(actual, reference, metric):
        return runtime_check(actual, reference, case, args.out, metric, source)

    sys.argv = [
        str(source_path),
        "--model",
        MODEL,
        "--pp-size",
        str(case.pp),
        "--ep-size",
        str(case.ep),
        "--cp-size",
        str(case.cp),
    ]
    namespace = {"__name__": "__main__", "__file__": str(source_path), "__strict_hook": hook}
    exec(compile(tree, str(source_path), "exec"), namespace)


if __name__ == "__main__":
    main()
