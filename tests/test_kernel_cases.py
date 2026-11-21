"""
CPU-only checks (no torch, no GPU) of the kernel-shape spec, tools/kernel_cases/kernel_cases.json:

- it is current: regenerating it from the model configs and meshes gives the committed bytes, so a
  changed config, mesh, benchmark script or generator fails here until `python -m tools.kernel_cases
  --write` is run and the diff reviewed;
- every examples/pretrain_lm/<model>/config.json has cases in each family its architecture uses,
  and every shipped benchmark / example mesh is the source of cases;
- every name in pithtrain.kernels.__all__ belongs to a family (or is shape-independent), and every
  family's kernels exist;
- every family with cases has a spec-driven test (tests/spec/test_spec_<family>.py parametrized from
  it) unless listed in UNTESTED with the reason;
- the expert cases reach both sides of every route boundary the SM100 pickers switch on, and the
  expert counts the meshes produce (G = 1 .. 256);
- the materialized routing is consistent (counts sum to the rows, none above the EP group's tokens).
"""

import ast
from pathlib import Path

import pytest

from tools.kernel_cases import SPEC_PATH, dumps, load_spec, select
from tools.kernel_cases.cases import FAMILIES, SHAPE_INDEPENDENT, generate
from tools.kernel_cases.meshes import shipped_meshes
from tools.kernel_cases.models import CONFIG_ROOT, REPO_ROOT, load_models

SPEC = load_spec()
MODELS = {model.name: model for model in load_models()}

UNTESTED = {
    "pointwise_layers": "no pithtrain kernel: RMSNorm, RoPE and the router are plain PyTorch inside torch.compile stages; tests/test_dualpipev.py runs them end to end",
}
"""
Families whose cases no spec-driven test consumes yet, with the reason.
"""


def test_spec_is_current() -> None:
    assert SPEC_PATH.read_text() == dumps(generate()), (
        "stale: run python -m tools.kernel_cases --write"
    )


def families_of(name: str) -> set[str]:
    model = MODELS[name]
    families = {"dense_gemm", "weight_quant", "attention", "cross_entropy", "pointwise_layers"}
    if model.experts is not None:
        families.add("moe_experts")
    if model.attention.supports_cp:
        families.add("ring_attention")
    if model.linear_attention is not None:
        families |= {"gated_delta_rule", "causal_conv1d"}
    if model.swiglu_widths:
        families.add("swiglu_dense")
    return families


@pytest.mark.parametrize(
    "config", sorted(CONFIG_ROOT.glob("*/config.json")), ids=lambda p: p.parent.name
)
def test_every_model_has_cases(config: Path) -> None:
    name = config.parent.name
    assert name in SPEC["inputs"]["models"], f"{name} has no spec entry"
    for family in families_of(name):
        cases = select(family, "spec", model=name, spec=SPEC)
        assert cases, f"{name}: no {family} cases"
        assert any(case["tier"] == "smoke" for case in cases), f"{name}: no smoke {family} case"


def test_every_shipped_mesh_has_cases() -> None:
    labels = {label for case in SPEC["cases"] for label in case["meshes"]}
    for mesh in shipped_meshes(MODELS):
        assert mesh.label in labels, f"{mesh.label} produced no case"


def kernel_names() -> list[str]:
    tree = ast.parse((REPO_ROOT / "pithtrain" / "kernels" / "__init__.py").read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "append":
            names.append(node.args[0].value)
    return names


def test_every_kernel_is_in_a_family() -> None:
    exported = set(kernel_names())
    in_families = {name for meta in FAMILIES.values() for name in meta["kernels"]}
    assert not in_families - exported, (
        f"family kernels not exported: {sorted(in_families - exported)}"
    )
    missing = exported - in_families - set(SHAPE_INDEPENDENT)
    assert not missing, (
        f"pithtrain.kernels names in no family (tools/kernel_cases/cases.py): {sorted(missing)}"
    )


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_family_has_a_spec_test(family: str) -> None:
    if not any(case["family"] == family for case in SPEC["cases"]):
        return
    path = REPO_ROOT / "tests" / "spec" / f"test_spec_{family}.py"
    if family in UNTESTED:
        assert not path.exists(), f"{path.name} exists: drop {family} from UNTESTED"
        return
    assert path.exists(), f"no {path.relative_to(REPO_ROOT)} consumes the {family} cases"
    assert f'spec_params("{family}"' in path.read_text(), (
        f"{path.name} does not parametrize from spec_params({family!r})"
    )


def moe(name: str) -> list[dict]:
    return [case["axes"] for case in select("moe_experts", "stress", model=name, spec=SPEC)]


def fp8_avg(axes: dict) -> float:
    return axes["fp8_rows"] / axes["G"]


ROUTE_BOUNDARIES = {
    "fp8_grouped_gemm small-group routes (M < 512 G)": lambda a: fp8_avg(a) < 512,
    "fp8_grouped_gemm large-group routes (M >= 512 G)": lambda a: fp8_avg(a) >= 512,
    "k_grouped_fp8_gemm short groups (M < 2048 G)": lambda a: fp8_avg(a) < 2048,
    "k_grouped_fp8_gemm long groups (M >= 2048 G)": lambda a: fp8_avg(a) >= 2048,
    "bf16 swap-AB (<= 512 rows / expert)": lambda a: a["bf16_fused"] and a["rows"] <= 512 * a["G"],
    "bf16 non-swap (> 512 rows / expert)": lambda a: a["bf16_fused"] and a["rows"] > 512 * a["G"],
    "an empty expert": lambda a: a["empty_groups"] > 0,
    "a group past 65536 rows (SM90 k-grouped split)": lambda a: a["max_rows"] > 65536,
}


@pytest.mark.parametrize(
    "name", [name for name, model in MODELS.items() if model.experts is not None]
)
def test_moe_cases_cross_route_boundaries(name: str) -> None:
    cases = moe(name)
    for route, hits in ROUTE_BOUNDARIES.items():
        assert any(hits(case) for case in cases), f"{name}: no case reaches {route}"
    if MODELS[name].experts.hidden == 2880:
        assert any(case["rows"] >= 1024 * case["G"] for case in cases), (
            "bf16 materialized gate_up (H = I = 2880, >= 1024 rows / expert)"
        )
    groups = {case["G"] for case in cases}
    experts = MODELS[name].experts.num_experts
    assert experts in groups and min(groups) == experts // max(
        ep for ep in (1, 2, 4, 8, 16, 32) if experts % ep == 0
    )


def test_moe_routing_is_consistent() -> None:
    for case in select("moe_experts", "stress", spec=SPEC):
        axes = case["axes"]
        sizes = axes["group_sizes"]
        assert len(sizes) == axes["G"] and sum(sizes) == axes["rows"], case["id"]
        assert max(sizes) <= axes["T"] * axes["ep"], case["id"]
        assert axes["fp8_rows"] == sum(-(-size // 128) * 128 for size in sizes), case["id"]
        if axes["routing"] != "stress":
            assert axes["rows"] <= axes["T"] * axes["top_k"] * 2, case["id"]


def test_attention_ring_steps_cover_every_cp() -> None:
    for name, model in MODELS.items():
        if not model.attention.supports_cp:
            continue
        kinds = {
            (case["axes"]["kind"], case["axes"]["B"])
            for case in select("attention", "spec", model=name, spec=SPEC)
        }
        for kind in ("dense", "ring_diag", "ring_front", "ring_back", "varlen"):
            for B in (1, 2, 4) if kind != "varlen" else (1,):
                assert (kind, B) in kinds, f"{name}: no {kind} attention case at B = {B}"
        cps = {
            case["axes"]["cp"] for case in select("ring_attention", "spec", model=name, spec=SPEC)
        }
        assert cps == {2, 4, 8}, f"{name}: ring CP degrees {sorted(cps)}"
