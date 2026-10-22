"""Bounded actual production HF-worker/native GRPO integration acceptance.

One reviewed w4/PP2/EP2 profile. No allocation, download, subprocess orchestration
or substitute rollout/reward implementation. Call produce and restore in separate
torchrun processes with the displayed outer timeouts. Old independent numerical
oracle gates remain separate and unchanged.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.acceptance.omni_posttraining import require, sha, source_identity, write_json

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/omni_rl/local-hf/script.py"
SCHEMA = "production_hf_provider_native_v1"
WORLD, PP, EP, REQUESTS, GROUP, SEQUENCE = 4, 2, 2, 3, 4, 32
CAPS = dict(prepare=120, produce=600, restore=300, kill_grace=10)


def cli_arguments(directory, restore=False):
    directory = Path(directory)
    values = [
        "--model",
        str(directory / "model"),
        "--reference-hf",
        str(directory / "model"),
        "--hf-prefix",
        "",
        "--scope",
        "text",
        "--processor",
        str(directory / "tokenizer"),
        "--prompts",
        str(directory / "prompts.jsonl"),
        "--checkpoint",
        str(directory / "checkpoint"),
        "--requests",
        str(REQUESTS),
        "--updates",
        str(REQUESTS),
        "--group-size",
        str(GROUP),
        "--sequence-length",
        str(SEQUENCE),
        "--new-tokens",
        "2",
        "--worker-timeout",
        "90",
        "--pp",
        str(PP),
        "--ep",
        str(EP),
        "--hf-device",
        "cpu",
        "--seed",
        "1234",
    ]
    if restore:
        values += ["--resume-request", "2", "--no-save"]
    return values


def binding(args):
    require(
        not args.workspace.resolve().is_relative_to(args.output.resolve())
        and not args.output.resolve().is_relative_to(args.workspace.resolve()),
        "Heavy state workspace and summary output must be disjoint",
    )
    return dict(
        schema=SCHEMA,
        source=source_identity(args.expected_commit),
        hardware=args.hardware,
        world=WORLD,
        pp=PP,
        ep=EP,
        cp=1,
        hsdp=1,
        driver_sha256=sha(__file__),
        cli_sha256=sha(SCRIPT),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        caps=CAPS,
        numerical_oracle="separate unchanged omni_posttraining gates",
        released_payload="NOT_RUN",
        fullsize="NOT_RUN",
    )


def prepare(args):
    from safetensors.torch import save_file

    from pithtrain.modules.omni_rollout_data import prepare_prompts
    from pithtrain.tasks.omni_hf_rollout import (
        ExactAnswerReward,
        LocalHFRolloutCfg,
        _canonical,
        _Engine,
        _new_hf,
    )
    from pithtrain.tasks.omni_rl import fingerprint
    from tests.test_omni_hf_rollout import tokenizer

    metadata = binding(args)
    require(not torch.cuda.is_initialized(), "CPU prepare must not initialize CUDA")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    args.workspace.mkdir(parents=True, exist_ok=False)
    raw = json.loads((ROOT / "tests/configs/qwen3_omni_text/tiny.json").read_text())
    raw.update(vocab_size=16)
    torch.manual_seed(42)
    policy, config = _new_hf(raw, "text")
    config.save_pretrained(args.workspace / "model")
    save_file(
        {k: v.detach().contiguous() for k, v in policy.state_dict().items()},
        str(args.workspace / "model/model.safetensors"),
    )
    tokens = tokenizer(args.workspace / "tokenizer")
    rows = [
        dict(id=f"real-prompt-{i}", text="Return the word red", media=[], answer="red")
        for i in range(REQUESTS * (WORLD // PP))
    ]
    manifest = args.workspace / "prompts.jsonl"

    def write_rows():
        manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    write_rows()
    prompts, _ = prepare_prompts(
        manifest, args.workspace / "tokenizer", config, scope="text", max_length=30
    )
    options = LocalHFRolloutCfg(
        compute_dtype="bfloat16", group_size=GROUP, max_sequence_length=SEQUENCE, max_new_tokens=2
    )
    values = _canonical(policy.state_dict(), config, "text")
    engine = _Engine(raw, options, values)
    engine.synchronize(values)
    first_groups = []
    for dp in range(WORLD // PP):
        result = engine.generate(
            prompts.payload(dp),
            torch.Generator().manual_seed(1234 + dp * 100003).get_state(),
            tokens.eos_token_id,
        )
        texts = [
            tokens.decode(
                ids[mask].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for ids, mask in zip(result["sampled_token_ids"], result["token_mask"], strict=True)
        ]
        require(
            len(set(texts)) > 1,
            "Initial real sampler must produce a nonconstant answer group; no retry",
        )
        for index in range(dp, len(rows), WORLD // PP):
            rows[index]["answer"] = texts[0]
        first_groups.append(result)
    # Freeze fixture answers only now, before training; never adapt after update.
    write_rows()
    prompts, _ = prepare_prompts(
        manifest, args.workspace / "tokenizer", config, scope="text", max_length=30
    )
    metadata["first_reward_groups"] = []
    for dp, result in enumerate(first_groups):
        reward = ExactAnswerReward(prompts, tokens, dp_rank=dp, dp_size=WORLD // PP)
        identity = SimpleNamespace(
            contract=SimpleNamespace(reward_config_sha256=fingerprint(reward.config))
        )
        request = SimpleNamespace(
            request_index=0, identity=identity, dp_rank=dp, max_sequence_length=SEQUENCE
        )
        batch = SimpleNamespace(request_index=0, identity=identity, **result)
        scores = reward.score(request, batch)
        require(scores.unique().numel() > 1, "Actual ExactAnswerReward must be nonconstant")
        metadata["first_reward_groups"].append(scores.tolist())
    metadata["files"] = {
        str(p.relative_to(args.workspace)): sha(p)
        for p in sorted(args.workspace.rglob("*"))
        if p.is_file()
    }
    metadata["status"] = "PREPARED_GPU_NOT_RUN"
    metadata["cli"] = cli_arguments(args.workspace)
    write_json(args.output / "prepare.json", metadata)
    require(not torch.cuda.is_initialized(), "CPU prepare initialized CUDA")
    return metadata


def run(args):
    expected = binding(args)
    prepared = json.loads((args.output / "prepare.json").read_text())
    require(
        all(prepared.get(k) == v for k, v in expected.items()), "Prepared source/runtime changed"
    )
    require(
        all(sha(args.workspace / name) == digest for name, digest in prepared["files"].items()),
        "Prepared input changed",
    )
    require(os.environ.get("PITHTRAIN_RUN_OMNI_GPU_TESTS") == "1", "Explicit GPU opt-in required")
    require(
        os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Offline mode required",
    )
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
        "Process-start deterministic CUBLAS required",
    )
    require(int(os.environ.get("WORLD_SIZE", "0")) == WORLD, "Exact four-rank torchrun required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    require(torch.cuda.get_device_name() == args.hardware, "Actual GPU differs")
    phase = args.output / args.action
    # One report per rank, exclusive create: no overwrite or retry claim.
    phase.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("production_hf_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    rank = int(os.environ["RANK"])
    producer = args.workspace / "produce" / f"rank{rank}.pt"
    target = args.workspace / args.action / f"rank{rank}.pt"
    target.parent.mkdir(exist_ok=True)
    require(not target.exists(), "Fresh acceptance output required")
    state, installs, reports = {}, [], []
    original = (
        torch.load(producer, weights_only=True, map_location="cpu")
        if args.action == "restore"
        else None
    )

    def observe(event, trainer, report):
        from tests.acceptance.omni_native_resume import equal
        from tests.acceptance.omni_posttraining import snapshot

        require(
            trainer.backend.__class__.__name__ == "PithTrainPolicyAdapter",
            "Real native adapter required",
        )
        require(
            trainer.rollout.__class__.__name__ == "DistributedHFThinkerRollout",
            "Real production worker required",
        )
        if event == "ready":
            if original is not None:
                equal(
                    snapshot(trainer.backend._checkpoint_state(trainer)),
                    original["boundary"],
                    "restored boundary",
                )
            return
        require(
            trainer.updates == trainer.requests,
            "Every request must make an actual optimizer update",
        )
        require(report["gradient_norm"] > 0, "Actual native gradient norm must be nonzero")
        if trainer.requests == 1:
            require(
                trainer.reward.last_scores.unique().numel() > 1,
                "First actual reward/advantage group must be nonconstant",
            )
            require(
                trainer.reward.last_scores.tolist()
                == prepared["first_reward_groups"][trainer.backend.dp_rank],
                "Production worker initial sampled rewards differ from prepared actual sampler",
            )
        receipt = trainer.rollout.installation_receipt
        require(receipt is not None, "Actual worker install/readback required")
        require(
            receipt["reference_weights_sha256"] == trainer.contract.reference_policy_sha256,
            "Fixed reference changed",
        )
        installs.append(receipt)
        reports.append(report)
        if trainer.requests == 2:
            state["boundary"] = snapshot(trainer.backend._checkpoint_state(trainer))
        if trainer.requests == 3:
            state["after"] = snapshot(trainer.backend._checkpoint_state(trainer))
            if original is not None:
                equal(state["after"], original["after"], "fresh process next update")
                equal(report, original["reports"][-1], "next update report")
                equal(receipt, original["installs"][-1], "restored policy install/readback")

    cli.main(cli_arguments(args.workspace, args.action == "restore"), observer=observe)
    if args.action == "produce":
        require(
            len(installs) == 3 and len({r["installed_weights_sha256"] for r in installs}) == 3,
            "Each updated policy must be independently published and installed",
        )
    else:
        require(len(installs) == 1, "Exactly one fresh-process continuation update required")
    require(binding(args) == expected, "Source/runtime changed during execution")
    with target.open("xb") as stream:
        torch.save(dict(**state, reports=reports, installs=installs), stream)
    result = dict(
        **expected,
        status="PASS",
        phase=args.action,
        rank=rank,
        payload_sha256=sha(target),
        observed_updates=len(reports),
        fresh_restore=args.action == "restore",
        scope="production provider integration; not independent numerical parity",
    )
    write_json(phase / f"rank{rank}.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "produce", "restore"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--workspace",
        required=True,
        type=Path,
        help="Heavy tensors/checkpoints outside summary collection root",
    )
    args = parser.parse_args(argv)
    if args.action == "plan":
        return dict(
            **binding(args), status="PLAN_ONLY_GPU_NOT_RUN", cli=cli_arguments(args.workspace)
        )
    return prepare(args) if args.action == "prepare" else run(args)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
