"""
Parametrization of the spec-driven tests from tools/kernel_cases/kernel_cases.json: one pytest param
per case, id = the case id, marked with its tier (smoke / spec / stress), so pytest -m selects the
tier and nothing in a test file restates a shape.
"""

import pytest

from tools.kernel_cases import load_spec, select

SPEC = load_spec()
TIER_MARKS = {"smoke": pytest.mark.smoke, "spec": pytest.mark.spec, "stress": pytest.mark.stress}


def spec_params(family: str, keep=None, marks=(), tier: str = "stress") -> list:
    """
    The family's cases up to tier as pytest params of one argument (the case dict), keeping those
    for which keep(case) is true; marks are added to every param.
    """
    params = []
    for case in select(family, tier, spec=SPEC):
        if keep is not None and not keep(case):
            continue
        params.append(pytest.param(case, id=case["id"], marks=[TIER_MARKS[case["tier"]], *marks]))
    return params
