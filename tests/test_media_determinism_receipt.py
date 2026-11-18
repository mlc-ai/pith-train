"""Ensure media receipts observe actual runtime without replacing original gates."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_media_rank_report_reads_actual_policy_without_replacing_assertions():
    path = ROOT / "tests/acceptance/omni_media_resume.py"
    node = next(
        item
        for item in ast.parse(path.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "run_gpu"
    )
    report = next(
        item.value
        for item in node.body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "report" for target in item.targets)
    )
    value = next(item.value for item in report.keywords if item.arg == "runtime_determinism")
    assert isinstance(value, ast.Call) and value.func.id == "runtime_determinism_snapshot"
    phases = [
        call.args[0].value
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_phase"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    ]
    assert phases == [
        "exact complete boundary",
        "exact next media input",
        "exact third objective",
        "exact third state",
        "exact third component updates",
    ]
