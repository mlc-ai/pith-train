import ast
import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from tests.acceptance import omni_topology_gradients as strict_gradients
from tests.acceptance.omni_topology_contract import (
    BY_NAME,
    CASES,
    GRAD_EPS,
    MODEL,
    SMALL_PARAMETERS,
    Case,
    canonical,
    expected_parameters,
    verify_source,
)
from tests.acceptance.omni_topology_entry import overlay, validate_torchrun_environment
from tests.acceptance.omni_topology_gradients import calculate_difference, compare_cpu, compare_pair
from tests.acceptance.omni_topology_runner import validate_reports

SOURCE = {"commit": "fixture", "source_hashes": {}}


@pytest.fixture
def package():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def param(values, shape=None, gradient=True):
    grad = torch.tensor(values, dtype=torch.float32)
    return SimpleNamespace(
        shape=tuple(grad.shape) if shape is None else shape, grad=grad if gradient else None
    )


NAMES = (
    "0.layers.0.mlp.gate.weight",
    "0.layers.0.post_attention_layernorm.weight",
    "1.norm.weight",
)


def inputs():
    actual = [(name, param([1e-7, -2e-7])) for name in NAMES]
    reference = copy.deepcopy(actual)
    expected = {name: (2,) for name in NAMES}
    return actual, reference, expected


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_actual",
        "missing_reference",
        "both_missing",
        "duplicate_actual",
        "duplicate_reference",
        "rename_actual",
        "rename_both",
        "shape_actual",
        "shape_reference",
        "shape_both",
        "grad_shape_actual",
        "grad_shape_reference",
        "grad_none_actual",
        "grad_none_reference",
        "extra_actual",
        "extra_reference",
    ],
)
def test_structure_faults_cannot_pass(mutation):
    a, r, expected = inputs()
    if mutation.startswith("missing_"):
        (a if mutation.endswith("actual") else r).pop()
    elif mutation == "both_missing":
        a.pop()
        r.pop()
    elif mutation.startswith("duplicate_"):
        target = a if mutation.endswith("actual") else r
        target[-1] = copy.deepcopy(target[0])
    elif mutation.startswith("rename_"):
        a[-1] = ("1.wrong.weight", a[-1][1])
        if mutation == "rename_both":
            r[-1] = ("1.wrong.weight", r[-1][1])
    elif mutation.startswith("shape_"):
        for target in (
            [a, r] if mutation == "shape_both" else [a if mutation.endswith("actual") else r]
        ):
            target[-1][1].shape = (1, 2)
    elif mutation.startswith("grad_shape_"):
        (a if mutation.endswith("actual") else r)[-1][1].grad = torch.ones(1, 2)
    elif mutation.startswith("grad_none_"):
        (a if mutation.endswith("actual") else r)[-1][1].grad = None
    elif mutation.startswith("extra_"):
        (a if mutation.endswith("actual") else r).append(("1.extra.weight", param([1, 2])))
    report = compare_cpu(a, r, expected)
    assert report["status"] == "FAIL"
    assert report["structural_issues"] and report["checked_count"] == 0


def test_name_mapping_is_complete_and_order_independent():
    a, r, expected = inputs()
    report = compare_cpu(a, list(reversed(r)), expected)
    assert report["status"] == "PASS" and report["checked_count"] == 3
    assert all(row["compared"] for row in report["parameters"])


@pytest.mark.parametrize("name", sorted(SMALL_PARAMETERS))
def test_each_historical_small_parameter_is_compared_and_fails_wrong_gradient(name):
    row = compare_pair("0." + name, torch.tensor([1e-6, -2e-6]), torch.tensor([1e-7, -2e-7]), 1)
    assert row["legacy_would_skip"] == "reference_abs_max_below_1e-5_non_expert"
    assert row["compared"] and row["small_gradient_target"]
    assert row["difference"] == pytest.approx(81 / 101, rel=1e-6)
    assert row["status"] == "FAIL" and row["threshold"] == 1e-2


@pytest.mark.parametrize("side", ["actual", "reference"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_small_gradients_are_failures(side, value):
    a = torch.tensor([1e-8, -1e-8])
    r = a.clone()
    (a if side == "actual" else r)[0] = value
    row = compare_pair(NAMES[0], a, r, 1)
    assert row["status"] == "FAIL" and not row["compared"]
    assert row["difference"] is None and not row[side]["finite"]
    json.dumps(row, allow_nan=False)


def test_zero_and_expert_gradients_recorded_without_skip():
    both_zero = compare_pair(NAMES[0], torch.zeros(2), torch.zeros(2), 1)
    assert both_zero["legacy_would_skip"] == "both_zero"
    assert both_zero["compared"] and both_zero["difference"] == 0 and both_zero["status"] == "PASS"
    one_zero = compare_pair(NAMES[0], torch.zeros(2), torch.ones(2) * 1e-8, 1)
    assert one_zero["status"] == "FAIL" and one_zero["difference"] == 1
    expert = compare_pair(
        "0.layers.0.mlp.experts.down_proj", torch.ones(2) * 1e-8, torch.ones(2) * 1e-8, 1
    )
    assert expert["legacy_would_skip"] is None and expert["compared"]


def test_cp_division_and_threshold_boundary_are_unchanged():
    ref = torch.tensor([1.0, -2.0])
    assert compare_pair(NAMES[0], ref * 2, ref, 2)["difference"] == 0
    assert compare_pair(NAMES[0], ref * 2, ref, 1)["status"] == "FAIL"
    # Inject a metric result to test the exact frozen 'diff > eps' rejection boundary.
    assert compare_pair(NAMES[0], ref, ref, 1, lambda x, y: GRAD_EPS)["status"] == "PASS"
    assert (
        compare_pair(NAMES[0], ref, ref, 1, lambda x, y: math.nextafter(GRAD_EPS, math.inf))[
            "status"
        ]
        == "FAIL"
    )


@pytest.mark.parametrize("size", [1, 37, (1 << 20) - 1, 1 << 20, (1 << 20) + 13])
def test_chunked_metric_matches_independent_full_fp64_expression(size, package):
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(size, generator=generator)
    y = x + 0.03 * torch.randn(size, generator=generator)
    xd, yd = x.double(), y.double()
    expected = ((xd - yd).square().sum() / (xd.square().sum() + yd.square().sum())).item()
    source = ast.parse((package / "tests/test_dualpipev.py").read_text())
    original = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "calculate_difference"
    )
    namespace = {"torch": torch, "math": math}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])),
            "original-metric-only",
            "exec",
        ),
        namespace,
    )
    observed = calculate_difference(x, y)
    assert observed == namespace["calculate_difference"](
        x, y
    )  # Bitwise unchanged frozen implementation.
    # The algebraically equivalent squared-difference expression has a different
    # FP64 reduction order. Use its standard conservative gamma_n forward-error
    # bound, not an arbitrary 2e-14 that failed at million-element BLAS reductions.
    # All five original test inputs remain; the production 1e-2 bound is unchanged.
    n_eps = size * torch.finfo(torch.float64).eps
    assert abs(observed - expected) <= 8 * n_eps / (1 - n_eps)


def test_reviewed_hashes_and_commit_binding(package):
    import subprocess

    head = subprocess.check_output(
        ["git", "-C", str(package), "rev-parse", "HEAD"], text=True
    ).strip()
    assert verify_source(package, head)["commit"] == head
    with pytest.raises(ValueError, match="Commit mismatch"):
        verify_source(package, "0" * 40)
    with pytest.raises(ValueError, match="40-character"):
        verify_source(package, "HEAD")


def test_overlay_only_changes_final_gradient_verification(package):
    source = (package / "tests/test_dualpipev.py").read_text()
    old, new = ast.parse(source), overlay(source)
    old_main = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    new_main = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    index = next(i for i, n in enumerate(new_main.body) if "__strict_hook" in ast.unparse(n))
    assert ast.dump(ast.Module(body=old_main.body[:index], type_ignores=[])) == ast.dump(
        ast.Module(body=new_main.body[:index], type_ignores=[])
    )
    stop = next(
        i
        for i, n in enumerate(old_main.body)
        if isinstance(n, ast.Assert) and ast.unparse(n.test) == "not failed"
    )
    assert ast.dump(ast.Module(body=old_main.body[stop + 1 :], type_ignores=[])) == ast.dump(
        ast.Module(body=new_main.body[index + 1 :], type_ignores=[])
    )
    for before, after in zip(old.body, new.body, strict=True):
        if before is not old_main:
            assert ast.dump(before) == ast.dump(after)
    with pytest.raises(ValueError, match="hash mismatch"):
        overlay(source.replace("label_scale = 10.0", "label_scale = 100.0"))
    compile(new, "cpu-structure-only", "exec")  # Do not execute native imports.


def native_constructor_namespace(package, case, rank):
    """Execute actual pinned constructors on meta, without native CUDA imports/forwards."""
    source = ast.parse((package / "pithtrain/models/qwen3_omni_moe.py").read_text())
    classes = [node for node in source.body if isinstance(node, ast.ClassDef)]
    for node in classes:
        for fn in [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]:
            fn.decorator_list = [
                d
                for d in fn.decorator_list
                if not (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute)
                    and ast.unparse(d.func) == "torch.compile"
                )
            ]
    partition = next(
        n
        for n in ast.parse((package / "pithtrain/pipeline/dualpipev.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "layer_partition"
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            partition,
            *classes,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "training": SimpleNamespace(Linear=torch.nn.Linear, fp8=False),
        "BF16Linear": torch.nn.Linear,
        "distributed": SimpleNamespace(
            pp_size=case.pp, pp_rank=rank // (case.world // case.pp), ep_size=case.ep
        ),
    }
    exec(
        compile(ast.fix_missing_locations(tree), "isolated-native-constructors", "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_all_13_inventory_and_cpu_native_constructor_shapes(package, case):
    config = json.loads((package / MODEL).read_text())
    config["pad_token_id"] = None
    coverage = set()
    for rank in range(case.world):
        namespace = native_constructor_namespace(package, case, rank)
        model = namespace["Qwen3OmniMoeThinkerTextModel"]
        with torch.device("meta"):
            module = torch.nn.Sequential(
                model(SimpleNamespace(**config), 0), model(SimpleNamespace(**config), 1)
            )
        shapes = {name: tuple(p.shape) for name, p in module.named_parameters()}
        assert shapes == expected_parameters(case, rank)
        coverage.update(canonical(name) for name in shapes)
    assert len(coverage) == 47 and SMALL_PARAMETERS <= coverage


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_original_fixture_shapes_packing_and_ragged_shift(package, case):
    main = next(
        n
        for n in ast.parse((package / "tests/test_dualpipev.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    build = next(
        n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "build_chunks"
    )

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def __rmul__(self, other):
            return self

    fake_torch = SimpleNamespace(
        randint=lambda lo, hi, shape: FakeTensor(shape),
        randn=lambda *shape, **kw: FakeTensor(shape),
        tensor=lambda data, **kw: list(data),
        int32="int32",
    )
    namespace = dict(
        torch=fake_torch,
        ragged=case.ragged,
        packed=case.packed,
        sequence_length=128,
        num_global_chunks=20,
        micro_batch_size=1 if case.packed else 3,
        vocab_size=152064,
        label_scale=10.0,
        dtype="bf16",
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[build], type_ignores=[])),
            "fixture-shapes",
            "exec",
        ),
        namespace,
    )
    steps = [namespace["build_chunks"](i) for i in (0, 1)]
    lengths = [[x.shape[1] for x, _, _ in chunks] for chunks in steps]
    assert len(steps[0]) == len(steps[1]) == 20
    if case.ragged:
        assert lengths[0][:4] == [128, 64, 160, 96]
        assert lengths[1][:4] == [160, 96, 128, 64]
    else:
        assert lengths[0] == lengths[1] == [128] * 20
    for chunks in steps:
        for i, (x, labels, cu) in enumerate(chunks):
            assert labels.shape == (*x.shape, 152064)
            if case.packed:
                assert case.cp == 1 and x.shape[0] == 1
                assert cu[0] == 0 and cu[-1] == x.shape[1] and len(cu) == 3 + i % 2
                assert all(b > a for a, b in zip(cu, cu[1:]))
            else:
                assert cu is None
            if case.cp == 2:
                assert x.shape[1] % 4 == 0


def good_reports(case):
    reports = {}
    for rank in range(case.world):
        expected = expected_parameters(case, rank)
        rows = [
            {
                "name": name,
                "canonical": canonical(name),
                "shape": list(shape),
                "cp_divisor": case.cp,
                "status": "PASS",
                "compared": True,
                "difference": 0.0,
                "threshold": GRAD_EPS,
                "actual": {"finite": True},
                "reference": {"finite": True},
            }
            for name, shape in expected.items()
        ]
        reports[rank] = {
            "source": SOURCE,
            "rank": rank,
            "case": case.record(),
            "status": "PASS",
            "structural_issues": [],
            "checked_count": len(expected),
            "expected_count": len(expected),
            "parameters": rows,
            "gradient_threshold": GRAD_EPS,
            "loss_rtol": 1e-3,
            "loss_atol": 1e-3,
            "loss_two_steps_passed_before_hook": True,
        }
    return reports


LOSS_LOG = (
    "[INFO] Loss matches the reference at step 0.\n[INFO] Loss matches the reference at step 1."
)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rank",
        "missing_parameter",
        "duplicate_parameter",
        "wrong_shape",
        "wrong_case",
        "changed_threshold",
        "uncompared",
        "nonfinite",
        "nan_error",
        "above_threshold",
        "nonzero_exit",
        "missing_loss",
        "false_pass_status",
        "wrong_commit",
    ],
)
def test_report_fault_injection_prevents_false_acceptance(fault):
    case = BY_NAME["dense-pp2-ep2"]
    reports = good_reports(case)
    log, exit_code = LOSS_LOG, 0
    report, row = reports[0], reports[0]["parameters"][0]
    if fault == "wrong_commit":
        report["source"] = {"commit": "other"}
    elif fault == "missing_rank":
        reports.pop(3)
    elif fault == "missing_parameter":
        report["parameters"].pop()
    elif fault == "duplicate_parameter":
        report["parameters"][-1] = copy.deepcopy(row)
    elif fault == "wrong_shape":
        row["shape"] = [1]
    elif fault == "wrong_case":
        report["case"]["packed"] = True
    elif fault == "changed_threshold":
        report["gradient_threshold"] = 0.02
    elif fault == "uncompared":
        row["compared"] = False
    elif fault == "nonfinite":
        row["actual"]["finite"] = False
    elif fault == "nan_error":
        row["difference"] = float("nan")
    elif fault == "above_threshold":
        row["difference"] = 0.010001
    elif fault == "nonzero_exit":
        exit_code = 1
    elif fault == "missing_loss":
        log = ""
    elif fault == "false_pass_status":
        report["structural_issues"] = ["missing gradient"]
    assert validate_reports(case, reports, log, exit_code, SOURCE)["status"] == "FAIL"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_complete_report_inventory_can_pass(case):
    assert validate_reports(case, good_reports(case), LOSS_LOG, 0, SOURCE)["status"] == "PASS"


def test_matrix_budget_and_unsupported_packed_cp():
    assert len(CASES) == len(BY_NAME) == 13
    assert sum(case.timeout for case in CASES) == 37 * 60
    with pytest.raises(ValueError, match="packed CP"):
        Case("unsupported", 2, 1, 1, 2, True, True)


@pytest.mark.parametrize(
    "fault", ["none", "local_structure", "peer_structure", "local_nan", "peer_numeric"]
)
def test_runtime_report_and_all_rank_failure_propagation(monkeypatch, tmp_path, fault):
    a, r, expected = inputs()
    if fault == "local_structure":
        a.pop()
    if fault == "local_nan":
        a[0][1].grad[0] = float("nan")
    actual = SimpleNamespace(named_parameters=lambda **kw: iter(a))
    reference = SimpleNamespace(named_parameters=lambda **kw: iter(r))
    monkeypatch.setattr(strict_gradients, "expected_parameters", lambda case, rank: expected)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def gather(destination, row):
        destination[0] = row
        peer = copy.deepcopy(row)
        peer["rank"] = 1
        if "issues" in row and fault == "peer_structure":
            peer["issues"] = ["missing gradient on peer"]
        if "passed" in row and fault == "peer_numeric":
            peer["passed"] = False
        destination[1] = peer

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if fault == "none":
        strict_gradients.runtime_check(
            actual, reference, BY_NAME["dense-dp2"], tmp_path, calculate_difference
        )
    else:
        with pytest.raises(AssertionError, match="Strict"):
            strict_gradients.runtime_check(
                actual, reference, BY_NAME["dense-dp2"], tmp_path, calculate_difference
            )
    report = json.loads((tmp_path / "rank-0.json").read_text())
    assert report["status"] == ("PASS" if fault == "none" else "FAIL")
    assert report["checked_count"] == (0 if "structure" in fault else 3)
    if fault == "none":
        assert all(row["legacy_would_skip"] and row["compared"] for row in report["parameters"])


def test_real_torchrun_environment_requires_exact_world_and_ranks():
    environment = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12345",
    }
    validate_torchrun_environment(BY_NAME["dense-single"], environment)
    assert "SLURM_JOB_ID" not in environment
    for key in environment:
        broken = dict(environment)
        broken.pop(key)
        with pytest.raises(RuntimeError):
            validate_torchrun_environment(BY_NAME["dense-single"], broken)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        with pytest.raises(RuntimeError):
            validate_torchrun_environment(BY_NAME["dense-single"], dict(environment, **{key: "2"}))
