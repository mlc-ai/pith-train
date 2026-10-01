"""CPU preflights of the actual acceptance runner's modes and restore contract.

The runner executes main with real data providers, CPU autograd/AdamW/scheduler
and serialized CPU state. Model, pipeline, distributed setup, CUDA RNG and the
checkpoint transport are stand-ins. This is not GPU/FSDP/DCP acceptance.
"""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pithtrain.config import SlottedDefault
from pithtrain.modules.microbatch import Microbatch
from pithtrain.modules.training_data import DensePretrainData, OmniPretrainData
from tests import test_omni_training_acceptance as runner
from tests import test_training_data as data_fixtures
from tests.test_data_config import ROOT, task_config
from tests.test_pretrain_entrypoints import HistoricalConfig, stub_module

text_bundle = data_fixtures.text_bundle
text_corpus = data_fixtures.text_corpus


def arguments(root, *extra):
    return [
        "--dataset",
        str(root),
        "--output",
        str(root / "run"),
        "--report",
        str(root / "report"),
        "--observation-protocol",
        "save_then_runtime_state_v1",
        *extra,
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ["--legacy", "--media"],
        ["--legacy", "--data-format", "prepared_bundle"],
        ["--media", "--data-format", "token_bin"],
        ["--restore", "checkpoint"],
        ["--expected", "report"],
        ["--legacy", "--restore", "checkpoint", "--expected", "report"],
        ["--checkpoint-step", "0"],
        ["--steps", "1"],
        ["--cp", "0"],
    ],
)
def test_invalid_modes_stop_before_runtime(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        runner.parse_args(arguments(tmp_path, *extra))
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()
    assert not torch.cuda.is_initialized()


def test_legacy_mode_cannot_mislabel_current_logging(tmp_path):
    args = runner.parse_args(arguments(tmp_path, "--legacy"))
    runner.configure_data(HistoricalConfig(), args)
    with pytest.raises(ValueError, match="historical logging"):
        runner.configure_data(task_config().PretrainLMCfg(), args)


@pytest.mark.parametrize("data_format", ["token_bin", "prepared_bundle"])
def test_required_checkpoint_state_kind(tmp_path, data_format):
    args = runner.parse_args(arguments(tmp_path, "--data-format", data_format))
    with pytest.raises(AssertionError):
        runner.checkpoint_data(
            SimpleNamespace(checkpoint_state=object() if data_format == "token_bin" else None), args
        )


def example_report(data_format="prepared_bundle"):
    return dict(
        report_version=3,
        checkpoint_observation=dict(
            protocol="save_then_runtime_state_v1",
            events=[dict(operation=name, step=1) for name in ("save_checkpoint", "runtime_state")],
        ),
        result="PASSED",
        start=0,
        steps=3,
        legacy=False,
        data_format=data_format,
        parallelism=dict(pp=1, cp=1, ep=1),
        mesh=dict(rank=0, pp_rank=0, dp_rank=0, cp_rank=0, pp_size=1, dp_size=1, cp_size=1),
        config={"lr": 1e-5},
        source_sha256="same-source",
        media=False,
        checkpoint_step=1,
        rows=[{"train/step": n} for n in range(3)],
        loss_statistics=[{"loss_sum": 1.0, "target_count": 2}] * 3,
        batches=[[[[], [], None, None, None, []]] for _ in range(3)],
        checkpoint_state=dict(
            weights={},
            optimizers=[],
            schedulers=[],
            cuda_rng={},
            data=None if data_format == "token_bin" else {"consumed_samples": 8},
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"report_version": 1},
        {"report_version": 2},
        {"checkpoint_observation": None},
        {"checkpoint_observation": {"protocol": "save_then_runtime_state_v1", "events": []}},
        {"legacy": True},
        {"data_format": "token_bin"},
        {"media": True},
        {"parallelism": dict(pp=1, cp=1, ep=2)},
        {"mesh": {"rank": 1}},
        {"config": {"lr": 1e-4}},
        {"source_sha256": "different-source"},
        {"checkpoint_step": 2},
        {"steps": 2},
        {"batches": []},
        {"rows": [{"train/step": 1}]},
        {"loss_statistics": []},
        {"checkpoint_state": None},
        {"batches": [[[[], [], None, None, []]]] * 3},
    ],
)
def test_restore_rejects_wrong_or_incomplete_provenance(tmp_path, change):
    args = runner.parse_args(arguments(tmp_path, "--steps", "3"))
    report = example_report()
    report.update(change)
    with pytest.raises(AssertionError):
        runner.validate_restore_report(
            report,
            args,
            mesh=dict(rank=0, pp_rank=0, dp_rank=0, cp_rank=0, pp_size=1, dp_size=1, cp_size=1),
            config={"lr": 1e-5},
            source_sha256="same-source",
        )


@pytest.mark.parametrize("rank", range(4))
def test_restore_report_matches_rank_zero_logging_and_pp_zero_objectives(tmp_path, rank):
    args = runner.parse_args(arguments(tmp_path, "--steps", "3", "--pp", "2", "--ep", "2"))
    report = example_report()
    mesh = dict(
        rank=rank, pp_rank=rank // 2, dp_rank=rank % 2, cp_rank=0, pp_size=2, dp_size=2, cp_size=1
    )
    report.update(mesh=mesh, parallelism=dict(pp=2, cp=1, ep=2))
    if rank != 0:
        report["rows"] = []
    if mesh["pp_rank"] != 0:
        report["loss_statistics"] = []
    kwargs = dict(mesh=mesh, config={"lr": 1e-5}, source_sha256="same-source")
    runner.validate_restore_report(report, args, **kwargs)
    broken = copy.deepcopy(report)
    broken["rows"] = [] if rank == 0 else [{"train/step": 0}]
    with pytest.raises(AssertionError):
        runner.validate_restore_report(broken, args, **kwargs)
    broken = copy.deepcopy(report)
    broken["loss_statistics"] = [] if mesh["pp_rank"] == 0 else [{"loss_sum": 1.0}]
    with pytest.raises(AssertionError):
        runner.validate_restore_report(broken, args, **kwargs)


@pytest.mark.parametrize(
    "kind,marker,payload",
    [
        ("text", None, None),
        ("image", "image_grid_thw", "pixel_values"),
        ("audio", "feature_attention_mask", "input_features"),
        ("video", "video_grid_thw", "pixel_values_videos"),
    ],
)
def test_media_contract_rejects_lost_leaked_or_extra_payload(kind, marker, payload):
    tokens = torch.ones(1, 2, dtype=torch.long)
    context = {"input_ids": tokens}
    if marker:
        context[marker] = torch.ones(1, 2)
    media = {payload: torch.ones(2, 3)} if payload else None
    mb = Microbatch(
        model_inputs=(tokens,),
        cu_seqlens=None,
        objective_inputs=(tokens,),
        model_context=context,
        media_inputs=media,
    )
    assert runner.media_kind(mb, 0) == kind
    peer = copy.copy(mb)
    peer.media_inputs = None
    assert runner.media_kind(peer, 1) == kind
    if payload:
        with pytest.raises(AssertionError, match="ownership"):
            runner.media_kind(peer, 0)
        with pytest.raises(AssertionError, match="ownership"):
            runner.media_kind(mb, 1)
        leaked = copy.copy(mb)
        leaked.model_context = dict(context, **media)
        with pytest.raises(AssertionError, match="Shared context"):
            runner.media_kind(leaked, 0)
    mb.media_inputs = dict(media or {}, unexpected=torch.ones(1))
    with pytest.raises(AssertionError, match="ownership"):
        runner.media_kind(mb, 0)


class CPUTraining(SimpleNamespace):
    def to_json_dict(self):
        return SlottedDefault._make_json_serializable(vars(self))


def install_cpu_runtime(monkeypatch, data_root, events, fault, *, legacy=False):
    """Fresh model/data per main call; actual runner observes CPU stand-ins."""
    # Import real bundle verification before replacing the logging runtime.
    importlib.import_module("pithtrain.tasks.prepare_omni_data")
    distributed = SimpleNamespace(
        rank=0,
        pp_rank=0,
        dp_rank=0,
        cp_rank=0,
        pp_size=1,
        dp_size=1,
        cp_size=1,
        world_size=1,
        device=torch.device("cpu"),
    )
    context = SimpleNamespace()
    log_context = SimpleNamespace(wandb=None)
    task = stub_module(monkeypatch, "pithtrain.tasks.pretrain_lm")
    task.__file__ = str(ROOT / "pithtrain/tasks/pretrain_lm.py")
    task.PretrainLMCfg = task_config(
        TrainingCfg=lambda: CPUTraining(seed=1234, init_std=0.02),
    ).PretrainLMCfg
    if legacy:

        def historical_config():
            cfg = HistoricalConfig()
            cfg.training = CPUTraining(seed=1234, init_std=0.02)
            return cfg

        task.PretrainLMCfg = historical_config
    task.activate_wandb = lambda _: None
    task.wandb = SimpleNamespace(log=lambda _: None)

    class Decoder:
        def forward_stage1(self, value):
            return value[:]

        forward_stage3 = forward_stage1
        forward_stage5 = forward_stage1

    class Model(torch.nn.Module):
        forward_prolog = lambda self, value: value
        forward_posemb = lambda self, length, cu_seqlens=None: None

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(1))
            self.weight.to_local = lambda: self.weight
            self.layer = Decoder()

        def step(self, microbatches, objective):
            outputs = []
            for mb in microbatches:
                value = self.weight * mb.model_inputs[0].float().mean() * 0.0001
                loss = self.layer.forward_stage1(value + torch.rand(())).square().sum()
                (loss / len(microbatches)).backward()
                outputs.append(loss.detach())
            return outputs

    def setup_training(cfg):
        torch.manual_seed(cfg.training.seed)
        context.model = Model()
        if fault == "initialization" and any(event[0] == "save" for event in events):
            with torch.no_grad():
                context.model.weight.add_(1)
        context.optimizers = [torch.optim.AdamW(context.model.parameters(), lr=cfg.training.lr)]
        context.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(context.optimizers[0], lambda n: 1 / (n + 1))
        ]
        events.append(("model", id(context.model)))

    def setup_dataset(cfg):
        if legacy:
            data = DensePretrainData(cfg.dataset, cfg.training)
        elif cfg.data.format == "token_bin":
            data = DensePretrainData(cfg.data.dataset, cfg.training)
        else:
            data = OmniPretrainData(cfg.data, cfg.training)
        events.append(("data", id(data)))
        return data

    def get_batch(cfg, data, step, device):
        batches = data.get_batch(step, "cpu")
        if fault == "input" and any(event[0] == "load" for event in events):
            batches[0].model_inputs[0][0, 0] += 1
        return batches

    def train_step(cfg, data, step):
        batches = task.get_global_batch(cfg, data, step, torch.device("cpu"))
        losses = context.model.step(batches, None)
        norm = float(torch.nn.utils.clip_grad_norm_(context.model.parameters(), 1))
        context.optimizers[0].step()
        context.optimizers[0].zero_grad(set_to_none=True)
        context.schedulers[0].step()
        data.commit_step(step)
        events.append(("commit", step))
        task.wandb.log(
            {
                "train/step": step,
                "train/gradient-norm": norm,
                "train/cross-entropy-loss": float(torch.stack(losses).mean()),
            }
        )

    def save_current_checkpoint(path, step, *, data_state):
        assert events[-1] == ("commit", step - 1)
        if fault == "save":
            raise RuntimeError("save failed before observation")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                model=context.model.state_dict(),
                optimizer=context.optimizers[0].state_dict(),
                scheduler=context.schedulers[0].state_dict(),
                rng=torch.get_rng_state(),
                data=None if data_state is None else data_state.state_dict(),
            ),
            path / f"{step}.pt",
        )
        events.append(("save", data_state is None))

    def save_legacy_checkpoint(path, step):
        # Exact historical API: supplying data_state would raise TypeError.
        return save_current_checkpoint(path, step, data_state=None)

    def load_checkpoint(path, step, *, data_state):
        state = torch.load(path / f"{step}.pt", weights_only=True)
        assert (state["data"] is None) == (data_state is None)
        context.model.load_state_dict(state["model"])
        context.optimizers[0].load_state_dict(state["optimizer"])
        context.schedulers[0].load_state_dict(state["scheduler"])
        if data_state is not None:
            data_state.load_state_dict(state["data"])
        torch.set_rng_state(state["rng"])
        if fault == "rng":
            torch.rand(())
        events.append(("load", data_state is None))

    task.setup_dataset, task.get_global_batch, task.train_step = (
        setup_dataset,
        get_batch,
        train_step,
    )
    stub_module(
        monkeypatch,
        "pithtrain.contexts",
        distributed=distributed,
        logging=log_context,
        training=context,
    )
    stub_module(
        monkeypatch, "pithtrain.models.qwen3_moe", Qwen3MoeModel=Model, Qwen3MoeDecoderLayer=Decoder
    )
    stub_module(monkeypatch, "pithtrain.modules.distributed", setup_distributed=lambda cfg: None)
    stub_module(monkeypatch, "pithtrain.modules.logging", setup_logging=lambda cfg: None)
    stub_module(
        monkeypatch,
        "pithtrain.modules.training",
        setup_training=setup_training,
        make_adamw_optimizer=lambda: None,
        make_wsd_scheduler=lambda: None,
    )
    stub_module(
        monkeypatch,
        "pithtrain.modules.checkpoint",
        load_checkpoint=load_checkpoint,
        save_checkpoint=save_legacy_checkpoint if legacy else save_current_checkpoint,
    )
    stub_module(monkeypatch, "pithtrain.pipeline.execution", model_forward=lambda *a, **kw: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    return context


@pytest.mark.parametrize("data_format", ["token_bin", "prepared_bundle"])
@pytest.mark.parametrize("fault", [None, "rng", "input", "initialization"])
def test_actual_runner_constructs_and_restores_each_provider(
    monkeypatch,
    tmp_path,
    text_bundle,
    data_format,
    fault,
):
    data_root, _ = text_bundle
    events = []
    with torch.random.fork_rng(devices=[]):
        context = install_cpu_runtime(monkeypatch, data_root, events, fault)
        producer = tmp_path / "producer"
        resumed = tmp_path / "resumed"

        def invoke(output, *extra):
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    str(Path(runner.__file__)),
                    "--dataset",
                    str(data_root),
                    "--output",
                    str(output),
                    "--report",
                    str(output / "report"),
                    "--observation-protocol",
                    "save_then_runtime_state_v1",
                    "--data-format",
                    data_format,
                    "--steps",
                    "3",
                    "--sequence-length",
                    "16",
                    *extra,
                ],
            )
            runner.main()

        invoke(producer)
        final_weights = runner.digest(dict(context.model.named_parameters()))
        first_model = context.model
        if fault:
            message = {
                "rng": "Fresh-process restored state differs",
                "input": "Restart changed next inputs",
                "initialization": "Fresh-process initialization differs",
            }[fault]
            with pytest.raises(AssertionError, match=message):
                invoke(
                    resumed,
                    "--restore",
                    str(producer / "checkpoints"),
                    "--expected",
                    str(producer / "report"),
                )
            assert not (resumed / "report/rank0.json").exists()
        else:
            invoke(
                resumed,
                "--restore",
                str(producer / "checkpoints"),
                "--expected",
                str(producer / "report"),
            )
            assert context.model is not first_model
            assert runner.digest(dict(context.model.named_parameters())) == final_weights
            full = json.loads((producer / "report/rank0.json").read_text())
            replay = json.loads((resumed / "report/rank0.json").read_text())
            assert replay["rows"] == full["rows"][1:]
            assert replay["batches"] == full["batches"][1:]
            assert replay["exact_restore"] and not replay["legacy"]
            assert replay["data_format"] == data_format
        assert ("save", data_format == "token_bin") in events
        if fault == "initialization":
            assert not any(event[0] == "load" for event in events)
        else:
            assert ("load", data_format == "token_bin") in events
        assert not torch.cuda.is_initialized()


def test_historical_producer_must_save_and_hash_at_the_common_boundary(
    monkeypatch, tmp_path, text_bundle
):
    data_root, _ = text_bundle
    events = []
    with torch.random.fork_rng(devices=[]):
        install_cpu_runtime(monkeypatch, data_root, events, None, legacy=True)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(Path(runner.__file__)),
                *arguments(data_root, "--legacy", "--steps", "3", "--sequence-length", "16"),
            ],
        )
        runner.main()
    report = json.loads((data_root / "report/rank0.json").read_text())
    assert [event for event in events if event[0] == "save"] == [("save", True)]
    assert report["checkpoint_state"] is not None
    assert report["checkpoint_state"]["data"] is None
    assert (data_root / "run/checkpoints/1.pt").is_file()


@pytest.mark.parametrize("fault", [None, "save", "state_hash"])
@pytest.mark.parametrize("mode", ["legacy", "token_bin", "prepared_bundle"])
def test_every_producer_has_one_ordered_observation_and_stops_on_failure(
    monkeypatch, tmp_path, text_bundle, mode, fault
):
    data_root, _ = text_bundle
    events = []
    with torch.random.fork_rng(devices=[]):
        install_cpu_runtime(monkeypatch, data_root, events, fault, legacy=mode == "legacy")
        real_digest = runner.digest

        def observed_digest(value):
            if isinstance(value, dict) and set(value) == {
                "weights",
                "optimizers",
                "schedulers",
                "cuda_rng",
                "data",
            }:
                assert events[-1] == ("save", mode != "prepared_bundle")
                events.append(("state_hash", 1))
                if fault == "state_hash":
                    raise RuntimeError("state hash failed")
            return real_digest(value)

        monkeypatch.setattr(runner, "digest", observed_digest)
        extra = ["--legacy"] if mode == "legacy" else ["--data-format", mode]
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(Path(runner.__file__)),
                *arguments(data_root, "--steps", "3", "--sequence-length", "16", *extra),
            ],
        )
        if fault:
            with pytest.raises(RuntimeError, match="failed"):
                runner.main()
            assert not (data_root / "report/rank0.json").exists()
            assert [event for event in events if event[0] == "commit"] == [("commit", 0)]
        else:
            runner.main()
            report = json.loads((data_root / "report/rank0.json").read_text())
            assert report["report_version"] == 3
            assert report["checkpoint_observation"] == {
                "protocol": "save_then_runtime_state_v1",
                "events": [
                    {"operation": "save_checkpoint", "step": 1},
                    {"operation": "runtime_state", "step": 1},
                ],
            }
            assert report["config"]["save_interval"] is None
            assert [event for event in events if event[0] in {"commit", "save", "state_hash"}] == [
                ("commit", 0),
                ("save", mode != "prepared_bundle"),
                ("state_hash", 1),
                ("commit", 1),
                ("commit", 2),
            ]
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "mutation",
    ["wrong_protocol", "missing_save", "missing_hash", "reverse", "duplicate", "wrong_step"],
)
def test_restore_rejects_malformed_observation(tmp_path, mutation):
    report = example_report()
    observation = report["checkpoint_observation"]
    if mutation == "wrong_protocol":
        observation["protocol"] = "feature_only_historical"
    elif mutation == "missing_save":
        observation["events"].pop(0)
    elif mutation == "missing_hash":
        observation["events"].pop()
    elif mutation == "reverse":
        observation["events"].reverse()
    elif mutation == "duplicate":
        observation["events"].append(observation["events"][0])
    else:
        observation["events"][0]["step"] = 2
    args = runner.parse_args(arguments(tmp_path, "--steps", "3"))
    with pytest.raises(AssertionError, match="observation"):
        runner.validate_restore_report(
            report,
            args,
            mesh=report["mesh"],
            config=report["config"],
            source_sha256=report["source_sha256"],
        )


def test_observation_protocol_must_be_explicit(tmp_path):
    argv = arguments(tmp_path)
    at = argv.index("--observation-protocol")
    del argv[at : at + 2]
    with pytest.raises(SystemExit):
        runner.parse_args(argv)
    with pytest.raises(SystemExit):
        runner.parse_args([*argv, "--observation-protocol", "feature_only_historical"])
