"""
The authoritative kernel-shape spec of PithTrain: tools/kernel_cases/kernel_cases.json, generated
from the model configs and the training meshes, and read by tests/spec, benchmarks/kernels and
kernel-design agents. See docs/kernel-validation.md.
"""

import json
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parent / "kernel_cases.json"

__all__ = ["SPEC_PATH", "dumps", "load_spec", "select"]


def dumps(spec: dict) -> str:
    """
    The spec as committed: the header indented, then one case per line (so a diff shows the cases
    that changed).
    """
    header = {key: value for key, value in spec.items() if key != "cases"}
    text = json.dumps(header, indent=1)
    assert text.endswith("}")
    lines = [json.dumps(case, separators=(",", ":")) for case in spec["cases"]]
    return text[:-1] + ',\n "cases": [\n' + ",\n".join(lines) + "\n ]\n}\n"


def load_spec(path: Path = SPEC_PATH) -> dict:
    return json.loads(path.read_text())


TIER_ORDER = {"smoke": 0, "spec": 1, "stress": 2}


def select(
    family: str, tier: str = "spec", model: str | None = None, spec: dict | None = None
) -> list[dict]:
    """
    The cases of a family up to tier: "smoke" gives the smoke cases, "spec" smoke and spec, "stress"
    every case.
    """
    spec = spec or load_spec()
    limit = TIER_ORDER[tier]
    return [
        case
        for case in spec["cases"]
        if case["family"] == family
        and TIER_ORDER[case["tier"]] <= limit
        and (model is None or case["model"] == model)
    ]
