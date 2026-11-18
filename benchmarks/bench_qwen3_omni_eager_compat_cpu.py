"""Bounded CPU microbenchmark of the explicit compatibility helper.

Run this process under an external hard 120-second timeout. The internal work
budget is 100 seconds. Only the SHA-pinned 12 saved fixtures are accepted. Each
case compares compat/HF eager and compat/HF grouped in alternating ABBA/BAAB
blocks, recording forward-no-grad and forward-plus-four-VJP timings separately.
HF grouped has a different arithmetic contract; timing is not an equivalence
claim and cannot be extrapolated to CUDA, pipeline stages or training steps.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

FIXTURE_SHA = "1034d16d66a25ea9053d20b743e38b9e4754a1d12f87854144d8ae1ef23ebcda"
HF_MODEL_SHA = "1c1fc322dfa547c697576e8ad3c88b817aa382abc9db5d966755be3d9fabf7ce"
HF_MOE_SHA = "6f40b9a4220c6d5476a0207c7ef0aa7e188c3d457ad70df1b1b703461239b7ff"
LEAVES = ("inputs", "route_weights", "gate_up_proj", "down_proj")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, result):
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main(args):
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output / "EAGER_COMPAT_CPU_BENCHMARK.json"
    result = dict(
        status="PREFLIGHT",
        started_utc=datetime.now(timezone.utc).isoformat(),
        samples=[],
        summaries=[],
        correctness=[],
        cases_completed=0,
        original_tests_executed=0,
        original_two_FAIL_preserved=True,
        production_default_changed=False,
        new_GPU_jobs=0,
        new_spend_USD=0,
    )
    code = 2
    source_hashes = {}
    try:
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.dont_write_bytecode = True
        import numpy as np
        import torch
        import transformers
        from transformers.integrations import moe
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_model

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.__version__ != "2.13.0+cpu" or transformers.__version__ != "5.17.0":
            raise RuntimeError("Use the existing pinned Torch 2.13.0+cpu / HF 5.17.0 runtime")
        assert not torch.cuda.is_initialized()
        assert sha(args.fixtures) == FIXTURE_SHA
        assert sha(inspect.getfile(hf_model)) == HF_MODEL_SHA
        assert sha(inspect.getfile(moe)) == HF_MOE_SHA
        assert 4 <= args.rounds <= 12 and args.rounds % 2 == 0
        assert 1 <= args.batch <= 5 and 1 <= args.warmup <= 4

        spec = importlib.util.spec_from_file_location(
            "omni_explicit_eager_compat_bench", args.compat_source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compat = module.eager_experts_forward
        source_hashes = {
            str(path.resolve()): sha(path)
            for path in (
                Path(__file__),
                args.compat_source,
                args.fixtures,
                Path(inspect.getfile(hf_model)),
                Path(inspect.getfile(moe)),
            )
        }
        result["source_sha256"] = source_hashes
        result["runtime"] = dict(
            python=sys.version,
            executable=sys.executable,
            torch=torch.__version__,
            transformers=transformers.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
            intra_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            cuda_initialized=torch.cuda.is_initialized(),
        )
        result["method"] = dict(
            rounds=args.rounds,
            warmup_blocks=args.warmup,
            calls_per_sample=args.batch,
            order="Even rounds ABBA; odd rounds BAAB; equal position counts",
            timer="perf_counter_ns, elapsed/batch in ms",
            forward="torch.no_grad forward, includes public argument validation",
            forward_plus_vjp="Fresh graph each call plus autograd.grad of four leaves; no accumulated .grad",
            excluded_from_timing="fixture loading, module/parameter construction, clones and preflight comparisons",
            internal_work_budget_seconds=100,
            required_external_hard_timeout_seconds=120,
            no_full_model_or_GPU_extrapolation=True,
        )
        keys = (*LEAVES, "cotangent", "ids")
        cases = [
            f"s{seed}_k{k}_{dtype}"
            for seed in (82, 83)
            for k in (1, 2, 8)
            for dtype in ("BF16", "FP32")
        ]
        with np.load(args.fixtures, allow_pickle=False) as archive:
            assert set(archive.files) == {f"{case}__{key}" for case in cases for key in keys}
            fixtures = {case: {key: archive[f"{case}__{key}"] for key in keys} for case in cases}

        def budget():
            if time.monotonic() - started >= 100:
                raise TimeoutError(
                    "Internal 100-second work budget reached; partial samples retained"
                )

        def make_arm(name, data, dtype):
            leaves = [
                torch.from_numpy(data[key]).to(dtype).clone().requires_grad_() for key in LEAVES
            ]
            x, weights, gu, down = leaves
            ids = torch.from_numpy(data["ids"])
            cot = torch.from_numpy(data["cotangent"]).to(dtype)
            if name == "compat":

                def forward():
                    return compat(x, weights, ids, gu, down)
            else:
                config = SimpleNamespace(
                    num_experts=gu.shape[0],
                    hidden_size=x.shape[1],
                    moe_intermediate_size=down.shape[-1],
                    hidden_act="silu",
                    _experts_implementation="eager" if name == "hf_eager" else "grouped_mm",
                )
                model = hf_model.Qwen3OmniMoeThinkerTextExperts(config).to(dtype)
                model.gate_up_proj = torch.nn.Parameter(gu)
                model.down_proj = torch.nn.Parameter(down)
                leaves[2:] = [model.gate_up_proj, model.down_proj]

                def forward():
                    return model(x, ids, weights)

            targets = tuple(leaves)

            def with_vjp():
                out = forward()
                return (out, *torch.autograd.grad(out, targets, cot))

            def no_grad():
                with torch.no_grad():
                    return forward()

            return dict(forward_no_grad=no_grad, forward_plus_vjp=with_vjp)

        # Validate all 12 saved cases before collecting any performance sample.
        prepared = {}
        for case in cases:
            budget()
            data = fixtures[case]
            dtype = torch.bfloat16 if case.endswith("BF16") else torch.float32
            prepared[case] = {
                name: make_arm(name, data, dtype) for name in ("compat", "hf_eager", "hf_grouped")
            }
            values = {name: arm["forward_plus_vjp"]() for name, arm in prepared[case].items()}
            comparisons = {}
            for name in ("compat", "hf_grouped"):
                comparisons[name] = {}
                for tensor_name, actual, expected in zip(
                    ("output", *LEAVES), values[name], values["hf_eager"], strict=True
                ):
                    comparisons[name][tensor_name] = dict(
                        exact=torch.equal(actual, expected),
                        differing=int(torch.count_nonzero(actual != expected)),
                        max_abs=float((actual.float() - expected.float()).abs().max()),
                    )
            result["correctness"].append(
                dict(case=case, versus="actual_hf_eager", comparisons=comparisons)
            )
            if not all(item["exact"] for item in comparisons["compat"].values()):
                raise AssertionError(
                    f"Explicit compatibility preflight failed for {case}; no timings collected"
                )
        result["status"] = "TIMING"
        save(receipt_path, result)

        for case in cases:
            for reference in ("hf_eager", "hf_grouped"):
                for workload in ("forward_no_grad", "forward_plus_vjp"):
                    functions = {
                        "A": prepared[case]["compat"][workload],
                        "B": prepared[case][reference][workload],
                    }
                    timed = {"A": [], "B": []}
                    for block in range(-args.warmup, args.rounds):
                        order = "ABBA" if block % 2 == 0 else "BAAB"
                        for position, arm in enumerate(order):
                            budget()
                            call = functions[arm]
                            tick = time.perf_counter_ns()
                            for _ in range(args.batch):
                                call()
                            milliseconds = (time.perf_counter_ns() - tick) / args.batch / 1e6
                            if block >= 0:
                                timed[arm].append(milliseconds)
                                result["samples"].append(
                                    dict(
                                        case=case,
                                        reference=reference,
                                        workload=workload,
                                        round=block,
                                        order=order,
                                        position=position,
                                        arm="compat" if arm == "A" else reference,
                                        milliseconds_per_call=milliseconds,
                                    )
                                )
                    summary = dict(
                        case=case,
                        reference=reference,
                        workload=workload,
                        reference_has_different_arithmetic_contract=reference == "hf_grouped",
                    )
                    for arm, name in (("A", "compat"), ("B", reference)):
                        samples = timed[arm]
                        summary[name] = dict(
                            samples=len(samples),
                            median_ms=statistics.median(samples),
                            min_ms=min(samples),
                            max_ms=max(samples),
                        )
                    summary["median_ratio_compat_over_reference_CPU_fixture_only"] = (
                        summary["compat"]["median_ms"] / summary[reference]["median_ms"]
                    )
                    result["summaries"].append(summary)
            result["cases_completed"] += 1
            save(receipt_path, result)
            print(f"Completed CPU fixture {case}", flush=True)
        assert len(result["summaries"]) == 48
        result["status"] = "COMPLETED_CPU_MICROBENCHMARK"
        code = 0
    except Exception as error:
        result["status"] = "FAILED_OR_INCOMPLETE_CPU_MICROBENCHMARK"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["source_hashes_unchanged"] = {
            path: sha(path) == value for path, value in source_hashes.items()
        }
        save(receipt_path, result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in ("status", "cases_completed", "wall_seconds", "error")
                }
            )
        )
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compat-source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch", type=int, default=3)
    raise SystemExit(main(parser.parse_args()))
