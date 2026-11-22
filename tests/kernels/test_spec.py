"""
The spec itself, checked in the smoke tier (static: no GPU work):

- README.md is current: its per-model sections (shapes, mesh table, case counts) are the ones
  cases.py generates (`python -m tests.kernels.cases --readme` prints them);
- the mesh rules hold for every model: contexts from 4k, CP > 1 only at context / CP >= 4k and
  micro-batch 1, EP dividing the expert count (EP 1 for dense models), CP 1 only for gpt-oss;
- every examples/pretrain_lm/<model>/config.json has cases, a smoke case among them, in every family
  its architecture launches;
- every name in pithtrain.kernels.__all__ belongs to a family (or is shape-independent or SM100-only)
  and every family's kernels exist;
- every family has its tests/kernels/test_<family>.py (and any test_<family>_<part>.py), each
  parametrized from params("<family>"), and each of the family's kernels is called there or by a
  pithtrain operator those tests import;
- the expert cases reach an empty expert, a group past the SM90 k-grouped split (65536 rows), every
  local expert count E / ep, and both edges of every BF16 tuning bucket the realistic routings reach
  at each EP the fused BF16 path runs; their routing is consistent.
"""

import ast
import re
from pathlib import Path

import pytest

from tests.kernels import cases
from tests.kernels.cases import (
    BF16_FUSED_MAX_GROUPS,
    CONFIG_ROOT,
    FAMILIES,
    MIN_LOCAL,
    README,
    REPO_ROOT,
    SHAPE_INDEPENDENT,
    SM100_ONLY,
    TUNING_BUCKETS,
    families_of,
    load_models,
    meshes_of,
    select,
    tuning_bucket,
)

pytestmark = pytest.mark.smoke
MODELS = {model.name: model for model in load_models()}
KERNELS = REPO_ROOT / "pithtrain" / "kernels"


def test_readme_is_current() -> None:
    have = cases.readme_models_section(README.read_text())
    want = cases.readme_models()
    assert have == want, "tests/kernels/README.md is stale: replace its per-model sections with the output of python -m tests.kernels.cases --readme"  # fmt: skip


@pytest.mark.parametrize("name", sorted(MODELS))
def test_mesh_rules(name: str) -> None:
    model = MODELS[name]
    meshes = meshes_of(model)
    assert min(m.context for m in meshes) == MIN_LOCAL
    for mesh in meshes:
        assert mesh.cp == 1 or (mesh.s_local >= MIN_LOCAL and mesh.mbs == 1), mesh
        assert model.experts is None or model.experts.num_experts % mesh.ep == 0, mesh
        assert model.experts is not None or mesh.ep == 1, mesh
    cps = {m.cp for m in meshes}
    assert cps == ({1, 2, 4, 8, 16} if model.attention.supports_cp else {1}), (name, cps)
    if not model.attention.supports_cp:
        assert {m.context for m in meshes} == {4096, 8192}


@pytest.mark.parametrize("config", sorted(CONFIG_ROOT.glob("*/config.json")), ids=lambda p: p.parent.name)  # fmt: skip
def test_every_model_has_cases(config: Path) -> None:
    name = config.parent.name
    assert name in MODELS, f"{name} has no cases"
    for family in families_of(MODELS[name]):
        found = select(family, "spec", model=name)
        assert found, f"{name}: no {family} cases"
        assert any(case["tier"] == "smoke" for case in found), f"{name}: no smoke {family} case"


def exported_kernels() -> list[str]:
    tree = ast.parse((KERNELS / "__init__.py").read_text())
    return [node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "append"]  # fmt: skip


def test_every_kernel_is_in_a_family() -> None:
    exported = set(exported_kernels())
    in_families = {name for meta in FAMILIES.values() for name in meta["kernels"]}
    assert not in_families - exported, f"family kernels not exported: {sorted(in_families - exported)}"  # fmt: skip
    missing = exported - in_families - set(SHAPE_INDEPENDENT) - set(SM100_ONLY)
    assert not missing, f"pithtrain.kernels names in no family (tests/kernels/cases.py): {sorted(missing)}"  # fmt: skip


def imported_sources(path: Path, seen: set[Path]) -> list[str]:
    """
    The source of every pithtrain module (outside pithtrain.kernels) path imports, following
    pithtrain.operators modules into the operators they import.
    """
    sources = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pithtrain.") and node.module != "pithtrain.kernels":  # fmt: skip
            target = REPO_ROOT / Path(*node.module.split(".")).with_suffix(".py")
            if target.exists() and target not in seen:
                seen.add(target)
                sources.append(target.read_text())
                if node.module.startswith("pithtrain.operators."):
                    sources += imported_sources(target, seen)
    return sources


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_has_a_test(family: str) -> None:
    tests = sorted((REPO_ROOT / "tests" / "kernels").glob(f"test_{family}*.py"))
    assert tests, f"no tests/kernels/test_{family}.py consumes the {family} cases"
    for test in tests:
        assert f'params("{family}"' in test.read_text(), f"{test.name} does not parametrize from params({family!r})"  # fmt: skip
    reached = "\n".join(
        text for test in tests for text in [test.read_text(), *imported_sources(test, set())]
    )
    unreached = [
        name for name in FAMILIES[family]["kernels"] if not re.search(rf"\b{name}\b", reached)
    ]
    assert not unreached, f"{', '.join(t.name for t in tests)} call none of {unreached}, directly or through an operator"  # fmt: skip


def module_constant(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple) and name in [t.id for t in target.elts]:
                    return ast.literal_eval(node.value)[[t.id for t in target.elts].index(name)]
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(name)


def test_tuning_buckets_are_the_tuners() -> None:
    tuning = KERNELS / "tuning.py"
    assert TUNING_BUCKETS == (module_constant(tuning, "MIN_BUCKET"), module_constant(tuning, "MAX_BUCKET"))  # fmt: skip


MOE_MODELS = sorted(name for name, model in MODELS.items() if model.experts is not None)


@pytest.mark.parametrize("name", MOE_MODELS)
def test_moe_cases_reach_the_routes(name: str) -> None:
    experts = MODELS[name].experts
    found = [case["axes"] for case in select("moe_experts", "stress", model=name)]
    assert any(a["empty_groups"] for a in found), "no case with an empty expert"
    split = module_constant(KERNELS / "fp8_gemm.py", "K_GROUPED_MAX_ROWS")
    assert any(a["max_rows"] > split for a in found), f"no group past {split} rows (SM90 k-grouped split)"  # fmt: skip
    groups = {a["G"] for a in found}
    assert groups == {experts.num_experts // ep for ep in cases.allowed_ep(MODELS[name])}, groups
    realistic = [a for a in found if a["routing"] not in ("stress", "boundary")]
    boundary = {(a["G"], a["avg_rows"]) for a in found if a["routing"] == "boundary"}
    for G in sorted({a["G"] for a in realistic if a["G"] <= BF16_FUSED_MAX_GROUPS}):
        for bucket in {tuning_bucket(a["avg_rows"]) for a in realistic if a["G"] == G}:
            assert (G, 2**bucket) in boundary, (G, bucket)
            if bucket < TUNING_BUCKETS[1]:
                assert (G, 2 ** (bucket + 1) - 1) in boundary, (G, bucket)


def test_moe_routing_is_consistent() -> None:
    for case in select("moe_experts", "stress"):
        a = case["axes"]
        sizes = a["group_sizes"]
        assert len(sizes) == a["G"] and sum(sizes) == a["rows"], case["id"]
        assert a["fp8_rows"] == sum(-(-size // 128) * 128 for size in sizes), case["id"]
        if a["routing"] != "boundary":
            assert max(sizes) <= a["T"] * a["ep"], case["id"]
        if a["routing"] not in ("stress", "boundary"):
            assert a["rows"] <= a["T"] * a["top_k"] * 2, case["id"]
