"""GPU acceptance runner for the Omni data boundary (not an Omni encoder).

Run each arm in a NEW torchrun process. Reports retain full precision metrics,
input hashes and exact checkpoint-state hashes. --legacy also runs on the PR's
base archive. Record rank-local objective sums/counts separately from the training
logger: the base logs DP0, while the feature logs a global DP x CP mean.
"""

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

REPORT_VERSION = 3
OBSERVATION_PROTOCOL = "save_then_runtime_state_v1"


def observation_events(step):
    """Ordered producer boundary, after the completed step/objective observation."""
    return [
        dict(operation=operation, step=step) for operation in ("save_checkpoint", "runtime_state")
    ]


def digest(value):
    """Hash every state entry, including shape/dtype, without storing a second checkpoint."""
    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            value = value.to_local()
        value = value.detach().cpu().contiguous()
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    if isinstance(value, dict):
        return {str(key): digest(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [digest(item) for item in value]
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Prepared fixture root; token_bin uses its tokens/train export",
    )
    parser.add_argument("--data-format", choices=("token_bin", "prepared_bundle"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=1)
    parser.add_argument(
        "--observation-protocol",
        required=True,
        choices=(OBSERVATION_PROTOCOL,),
        help="Explicit versioned checkpoint/state observation, identical for all producers",
    )
    args = parser.parse_args(argv)
    args.data_format = args.data_format or ("token_bin" if args.legacy else "prepared_bundle")
    if args.legacy and (args.data_format != "token_bin" or args.media):
        parser.error("--legacy requires token_bin text on the historical base")
    if args.media and args.data_format != "prepared_bundle":
        parser.error("--media requires prepared_bundle")
    if (args.restore is None) != (args.expected is None):
        parser.error("--restore and --expected must be supplied together")
    if args.restore is not None and args.legacy:
        parser.error("--legacy does not implement checkpoint restore")
    if not 0 < args.checkpoint_step < args.steps:
        parser.error("checkpoint-step must be positive and less than steps")
    if min(args.pp, args.cp, args.ep, args.sequence_length) < 1:
        parser.error("parallelism and sequence length must be positive")
    return args


def configure_data(cfg, args):
    if args.legacy:
        if hasattr(cfg, "data"):
            raise ValueError(
                "--legacy identifies historical logging; use --data-format token_bin for current code"
            )
        cfg.dataset = args.dataset / "tokens/train"
    else:
        cfg.data.dataset = (
            args.dataset / "tokens/train" if args.data_format == "token_bin" else args.dataset
        )
        cfg.data.format = args.data_format
        cfg.data.modalities = ("text", "image", "audio", "video") if args.media else ("text",)


def checkpoint_data(data, args):
    if args.legacy:
        return None
    state = data.checkpoint_state
    if args.data_format == "token_bin":
        assert state is None, "Dense position must follow the training step"
    else:
        assert state is not None, "Prepared data must preserve its committed state"
    return state


def validate_restore_report(expected, args, *, mesh, config, source_sha256):
    assert expected["result"] == "PASSED" and expected["start"] == 0
    assert expected["report_version"] == REPORT_VERSION, (
        "Restore requires the current report schema"
    )
    assert expected["checkpoint_observation"] == dict(
        protocol=args.observation_protocol,
        events=observation_events(args.checkpoint_step),
    ), "Restore requires the complete producer observation boundary"
    assert not expected["legacy"], "Cannot restore against a historical-base report"
    assert expected["data_format"] == args.data_format, "Restore data format differs"
    assert expected["media"] == args.media, "Restore modalities differ"
    assert expected["parallelism"] == dict(pp=args.pp, cp=args.cp, ep=args.ep)
    assert expected["mesh"] == mesh, "Restore rank/mesh differs"
    assert expected["source_sha256"] == source_sha256, "Restore source differs"
    # Output paths differ in a fresh process; all numerical training settings must match.
    numerical_config = lambda cfg: {
        key: value for key, value in cfg.items() if key not in {"model", "save_location"}
    }
    assert numerical_config(expected["config"]) == numerical_config(config), (
        "Restore training configuration differs"
    )
    assert expected["checkpoint_step"] == args.checkpoint_step, "Checkpoint step differs"
    assert expected["steps"] == args.steps and len(expected["batches"]) == args.steps
    logged_steps = list(range(args.steps)) if mesh["rank"] == 0 else []
    assert [row["train/step"] for row in expected["rows"]] == logged_steps
    assert len(expected["loss_statistics"]) == (args.steps if mesh["pp_rank"] == 0 else 0)
    state = expected["checkpoint_state"]
    assert isinstance(state, dict) and set(state) == {
        "weights",
        "optimizers",
        "schedulers",
        "cuda_rng",
        "data",
    }
    assert (state["data"] is None) == (args.data_format == "token_bin"), (
        "Checkpoint data state differs"
    )
    for batch in expected["batches"]:
        assert batch and all(len(micro) == 6 for micro in batch), "Invalid microbatch schema"


def media_kind(microbatch, pp_rank):
    context = microbatch.model_context or {}
    media = microbatch.media_inputs or {}
    fields = (
        ("image_grid_thw", "pixel_values", "image"),
        ("feature_attention_mask", "input_features", "audio"),
        ("video_grid_thw", "pixel_values_videos", "video"),
    )
    assert not {payload for _, payload, _ in fields} & context.keys(), (
        "Shared context contains media payloads"
    )
    expected = {payload for marker, payload, _ in fields if marker in context}
    assert set(media) == (expected if pp_rank == 0 else set()), (
        "Incorrect encoder payload ownership"
    )
    kinds = [kind for marker, _, kind in fields if marker in context]
    return kinds[-1] if kinds else "text"


def main():
    args = parse_args()

    from pithtrain.contexts import distributed, logging, training
    from pithtrain.models.qwen3_moe import Qwen3MoeModel
    from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.logging import setup_logging
    from pithtrain.modules.training import make_adamw_optimizer, make_wsd_scheduler, setup_training
    from pithtrain.pipeline.execution import model_forward
    from pithtrain.tasks import pretrain_lm

    cfg = pretrain_lm.PretrainLMCfg()
    configure_data(cfg, args)
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = args.cp
    cfg.distributed.expert_parallel_size = args.ep
    cfg.distributed.timeout = timedelta(minutes=3)
    t = cfg.training
    t.model = args.output / "model"
    t.optimizer = make_adamw_optimizer
    t.scheduler = partial(make_wsd_scheduler, start_lr=1e-6, warmup_ratio=0.25, decay_ratio=0)
    t.lr, t.max_steps, t.sequence_length = 1e-5, args.steps, args.sequence_length
    t.global_batch_size, t.micro_batch_size = 8, 1
    t.fp8, t.moe_load_balance_coef = False, 0.01
    t.moe_load_balance_type = "global-batch"
    t.save_location = args.output / "checkpoints"
    t.save_interval = None  # Only the explicit common observation boundary may save.
    setup_logging(cfg)
    setup_distributed(cfg)
    if distributed.rank == 0:
        base = (
            Path(pretrain_lm.__file__).parents[2] / "examples/pretrain_lm/qwen3-30b-a3b/config.json"
        )
        config = json.loads(base.read_text())
        config.update(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=max(4, 2 * args.pp),
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            vocab_size=152064,
        )
        t.model.mkdir(parents=True, exist_ok=True)
        (t.model / "config.json").write_text(json.dumps(config))
        args.report.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()

    # A test-only consumer accepts REAL processor outputs. Its scalar injection
    # exercises device transport and context association; it is not a vision/audio
    # encoder and cannot certify Omni semantics. No production capability is changed.
    seen, normal_calls, posemb_calls = Counter(), Counter(), Counter()
    view_checks = []
    media_audit = None

    def audit_method(method):
        def checked(self, *a, **kw):
            from torch.utils._pytree import tree_leaves

            outputs = method(self, *a, **kw)
            for output in tree_leaves(outputs):
                if (
                    not isinstance(output, torch.Tensor)
                    or not output.requires_grad
                    or output._base is None
                ):
                    continue
                record = dict(tensor=output, version=output._version, backward=False)
                view_checks.append(record)

                def before_backward(gradient, record=record):
                    tensor = record.pop("tensor")
                    assert tensor._version == record["version"], (
                        "A decoder view was modified in place"
                    )
                    record["backward"] = True
                    return gradient

                output.register_hook(before_backward)
            return outputs

        return checked

    original_prolog = Qwen3MoeModel.forward_prolog
    original_posemb = Qwen3MoeModel.forward_posemb

    def forward(self, inputs, cu_seqlens=None, model_context=None):
        normal_calls[self.stage_index] += 1
        media_audit.observe_dispatch(self.stage_index, model_context, "normal")
        return model_forward(self, inputs, self.chunk_record, cu_seqlens, model_context)

    def prolog(self, inputs, model_context=None):
        torch.testing.assert_close(inputs, model_context["input_ids"], rtol=0, atol=0)
        hidden = original_prolog(self, inputs)
        for name in ("pixel_values", "input_features", "pixel_values_videos"):
            if name in model_context:
                feature = model_context[name]
                assert feature.device == hidden.device and torch.isfinite(feature).all(), name
                assert feature.numel() > 0, name
                hidden = hidden + feature.float().mean().to(hidden.dtype) * 0.01
        return hidden

    def posemb(self, length, cu_seqlens=None, model_context=None):
        assert model_context["input_ids"].shape[1] == length
        if self.stage_index != 0:
            assert (
                not {"pixel_values", "input_features", "pixel_values_videos"} & model_context.keys()
            )
        for name, value in model_context.items():
            assert value.device == distributed.device, name
            if value.is_floating_point():
                assert value.dtype == torch.float32, (name, value.dtype)
        posemb_calls[self.stage_index] += 1
        return original_posemb(self, length, cu_seqlens)

    with ExitStack() as stack:
        from pithtrain.models.qwen3_moe import Qwen3MoeDecoderLayer

        # Audit every arm identically: registering autograd hooks only on the
        # feature changes the comparison, even when each hook returns its input.
        # Validate the condition in FSDP's view warning instead of suppressing
        # it: every view's hook must run and its version must stay unchanged.
        for method in ("forward_stage1", "forward_stage3", "forward_stage5"):
            stack.enter_context(
                patch.object(
                    Qwen3MoeDecoderLayer,
                    method,
                    audit_method(getattr(Qwen3MoeDecoderLayer, method)),
                )
            )
        if args.media:
            from pithtrain.modules import qwen3_omni_data
            from pithtrain.modules.training_data import OmniPretrainData
            from pithtrain.pipeline import dualpipev
            from pithtrain.tasks import prepare_omni_data
            from tests.omni_media_audit import MediaAudit

            media_audit = MediaAudit(distributed.pp_rank, distributed.pp_size, distributed.pp_group)
            media_audit.install(
                stack, OmniPretrainData, prepare_omni_data, qwen3_omni_data, torch.distributed
            )
            media_audit.install_dispatch(stack, dualpipev)
            stack.enter_context(
                patch.object(
                    Qwen3MoeModel,
                    "input_modalities",
                    {"text", "image", "audio", "video"},
                    create=True,
                )
            )
            for name, method in (
                ("forward", forward),
                ("forward_prolog", prolog),
                ("forward_posemb", posemb),
            ):
                stack.enter_context(patch.object(Qwen3MoeModel, name, method))
        data = pretrain_lm.setup_dataset(cfg)
        data_state = checkpoint_data(data, args)
        setup_training(cfg)

        def runtime_state():
            return digest(
                dict(
                    weights=dict(training.model.named_parameters()),
                    optimizers=[opt.state_dict() for opt in training.optimizers],
                    schedulers=[s.state_dict() for s in training.schedulers],
                    cuda_rng=torch.cuda.get_rng_state(),
                    data=None if data_state is None else data_state.state_dict(),
                )
            )

        mesh = dict(
            rank=distributed.rank,
            pp_rank=distributed.pp_rank,
            dp_rank=distributed.dp_rank,
            cp_rank=distributed.cp_rank,
            pp_size=distributed.pp_size,
            dp_size=distributed.dp_size,
            cp_size=distributed.cp_size,
        )
        config = cfg.training.to_json_dict()
        source_file = Path(pretrain_lm.__file__).resolve()
        source_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()
        start = 0
        expected = None
        if args.restore is not None:
            assert not args.legacy and args.expected is not None
            expected = json.loads((args.expected / f"rank{distributed.rank}.json").read_text())
            validate_restore_report(
                expected, args, mesh=mesh, config=config, source_sha256=source_sha256
            )
            assert digest(dict(training.model.named_parameters())) == expected["initial_state"], (
                "Fresh-process initialization differs"
            )
            load_checkpoint(args.restore, args.checkpoint_step, data_state=data_state)
            assert runtime_state() == expected["checkpoint_state"], (
                "Fresh-process restored state differs"
            )
            start = args.checkpoint_step

        rows, batches, loss_statistics = [], [], []
        original_batch = pretrain_lm.get_global_batch
        original_step = training.model.step
        objective_outputs, local_target_count = None, None

        def model_step(microbatches, objective):
            nonlocal objective_outputs
            objective_outputs = original_step(microbatches, objective)
            return objective_outputs

        def get_batch(*a, **kw):
            nonlocal local_target_count
            result = original_batch(*a, **kw)
            if media_audit is not None:
                media_audit.bind_batches(
                    result, [module.stage_index for module in training.model.module]
                )
            local_target_count = sum(
                int((mb.objective_inputs[0] != -100).sum().item()) for mb in result
            )
            batch = digest(
                [
                    (
                        mb.model_inputs,
                        mb.objective_inputs,
                        mb.cu_seqlens,
                        getattr(mb, "model_context", None),
                        getattr(mb, "media_inputs", None),
                        getattr(mb, "sample_ids", ()),
                    )
                    for mb in result
                ]
            )
            batches.append(batch)
            if expected is not None:
                assert batch == expected["batches"][start + len(batches) - 1], (
                    "Restart changed next inputs/media"
                )
            for mb in result:
                # Historical microbatches have no media fields; all feature
                # modes use the ownership check, including plain text.
                kind = "text" if args.legacy else media_kind(mb, distributed.pp_rank)
                seen[kind] += 1
            return result

        def capture(metrics):
            row = {key: float(value) for key, value in metrics.items() if key.startswith("train/")}
            assert all(torch.isfinite(torch.tensor(v)) for v in row.values()), row
            assert row["train/gradient-norm"] > 0, row
            rows.append(row)

        stack.enter_context(patch.object(training.model, "step", model_step))
        stack.enter_context(patch.object(pretrain_lm, "get_global_batch", get_batch))
        stack.enter_context(patch.object(pretrain_lm, "activate_wandb", lambda _: None))
        stack.enter_context(patch.object(logging, "wandb", object()))
        stack.enter_context(patch.object(pretrain_lm.wandb, "log", capture))
        checkpoint_state = None
        checkpoint_events = []
        initial = digest(dict(training.model.named_parameters()))
        for step in range(start, args.steps):
            pretrain_lm.train_step(cfg, data, step)
            if media_audit is not None:
                media_audit.end_step()
            # Observe the same detached objective outputs in every arm. Do not
            # replace the training logger or change its reduction/gradients.
            # The offline comparator can then form a global token-weighted mean
            # independently, excluding PP copies and validating the raw logs.
            assert objective_outputs is not None and local_target_count is not None
            if distributed.pp_rank == 0:
                assert (
                    len(objective_outputs)
                    == t.global_batch_size // distributed.dp_size // t.micro_batch_size
                )
                loss_statistics.append(
                    dict(
                        loss_sum=float(torch.stack(objective_outputs).double().sum().item()),
                        target_count=local_target_count,
                    )
                )
            else:
                assert not objective_outputs
            if args.restore is None and step + 1 == args.checkpoint_step:
                # Every producer saves and hashes at this same boundary. The
                # historical base predates the data_state keyword; its dense
                # cursor is still derived from the step, just as in current dense.
                if args.legacy:
                    assert data_state is None
                    save_checkpoint(t.save_location, step + 1)
                else:
                    save_checkpoint(t.save_location, step + 1, data_state=data_state)
                checkpoint_events.append(dict(operation="save_checkpoint", step=step + 1))
                checkpoint_state = runtime_state()
                checkpoint_events.append(dict(operation="runtime_state", step=step + 1))
        final = digest(dict(training.model.named_parameters()))
        assert initial != final, "Training did not update weights"
        for parameter in training.model.parameters():
            assert torch.isfinite(parameter.to_local()).all()
        if data_state is not None:
            assert data.consumed_samples == args.steps * t.global_batch_size
        if args.media:
            all_seen = [None] * distributed.world_size
            torch.distributed.all_gather_object(all_seen, dict(seen))
            assert set().union(*(item.keys() for item in all_seen)) == {
                "text",
                "image",
                "audio",
                "video",
            }
            assert normal_calls and posemb_calls
            if args.pp > 1:
                assert sum(posemb_calls.values()) > sum(normal_calls.values()), (
                    "No overlap context calls"
                )
        assert view_checks, "No decoder views were audited"
        assert all(record["backward"] for record in view_checks), (
            "A decoder view lost its backward hook"
        )
        assert checkpoint_events == (
            observation_events(args.checkpoint_step) if expected is None else []
        ), "Incomplete or duplicate checkpoint observation"
        report = dict(
            report_version=REPORT_VERSION,
            checkpoint_observation=dict(
                protocol=args.observation_protocol, events=checkpoint_events
            ),
            data_format=args.data_format,
            parallelism=dict(pp=args.pp, cp=args.cp, ep=args.ep),
            media=args.media,
            checkpoint_step=args.checkpoint_step,
            initial_state=initial,
            audited_view_hooks=len(view_checks),
            source_file=str(source_file),
            source_sha256=source_sha256,
            result="PASSED",
            start=start,
            steps=args.steps,
            rows=rows,
            loss_statistics=loss_statistics,
            legacy=args.legacy,
            mesh=mesh,
            batches=batches,
            checkpoint_state=checkpoint_state,
            exact_restore=expected is not None,
            modalities=dict(seen),
            media_observations=None
            if media_audit is None
            else dict(
                data=media_audit.finish_data(
                    providers=1,
                    validations=1,
                    expected_batches=(args.steps - start)
                    * t.global_batch_size
                    // distributed.dp_size,
                ),
                dispatch=media_audit.finish_dispatch(expected_steps=args.steps - start),
            ),
            normal_calls=dict(normal_calls),
            posemb_calls=dict(posemb_calls),
            config=config,
        )
        (args.report / f"rank{distributed.rank}.json").write_text(json.dumps(report, indent=2))
    torch.distributed.barrier()
    if distributed.rank == 0:
        print(
            json.dumps(
                dict(
                    result="PASSED",
                    report=str(args.report),
                    fresh_process_restore=expected is not None,
                )
            ),
            flush=True,
        )
    # setup_default_process_group owns teardown through its atexit callback.
    # Destroying it here as well makes that callback fail on a clean exit.


if __name__ == "__main__":
    main()
