"""Complete autograd speech task: independent objectives, transactions and Gloo DP."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from pithtrain.models.qwen3_omni_speech import SpeechTrainingOutput
from pithtrain.tasks.omni_objectives import TargetLoss
from pithtrain.tasks.omni_speech import (
    OmniSpeechCfg,
    _write_verified_checkpoint,
    configure_trainable_components,
    launch,
    load_checkpoint,
)


class Thinker(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.embed = nn.Embedding(11, 4, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(4, 11, dtype=torch.float64)

    def forward(self, tokens):
        hidden = self.dropout(self.embed(tokens))
        return hidden, self.head(hidden)


class ToySpeech(nn.Module):
    """CPU protocol fixture, not an implementation/parity claim for native speech."""

    def __init__(self, dropout=0):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_omni_moe")
        self.thinker = Thinker(dropout)
        self.talker = nn.Sequential(nn.Dropout(dropout), nn.Linear(4, 17, dtype=torch.float64))
        self.code2wav = nn.Sequential(nn.Dropout(dropout), nn.Linear(3, 1, dtype=torch.float64))
        self.register_buffer("forwards", torch.tensor(0))

    def forward(
        self,
        input_ids,
        model_context,
        codes,
        code_lengths,
        *,
        prompt_lengths,
        speaker,
        codebook_mask,
        position_kwargs,
    ):
        self.forwards.add_(1)
        hidden, text = self.thinker(input_ids)
        scores = self.talker(hidden)
        first = torch.cat((codes[..., 0], torch.full((len(codes), 1), 6)), dim=1)
        residual = codes[..., 1:].clone()
        if codebook_mask is not None:
            first[:, :-1].masked_fill_(~codebook_mask[..., 0], -100)
            residual.masked_fill_(~codebook_mask[..., 1:], -100)
        if model_context.get("ignore_codec"):
            first.fill_(-100)
            residual.fill_(-100)
        return SpeechTrainingOutput(
            first_logits=scores[..., :7],
            first_labels=first,
            residual_logits=scores[:, :-1, 7:].reshape(len(codes), 2, 2, 5),
            residual_labels=residual,
            thinker_logits=text,
        )

    def decode_training_codes(self, codes):
        self.forwards.add_(1)
        return self.code2wav(codes.transpose(1, 2).double()).squeeze(-1)


def sample(zero=False, rank=0):
    return dict(
        input_ids=torch.tensor([[1, 2, 3]]) + rank,
        model_context=dict(ignore_codec=zero),
        codes=torch.tensor([[[1, 2, 3], [2, 3, 4]]]),
        code_lengths=torch.tensor([2]),
        prompt_lengths=torch.tensor([1]),
        codebook_mask=torch.tensor([[[True, True, False], [True, rank == 1, True]]]),
        labels=torch.tensor([[-100, 4, 5]]) if not zero else torch.full((1, 3), -100),
        target_waveform=torch.tensor([[0.25, -0.5]], dtype=torch.float64),
        waveform_mask=torch.full((1, 2), not zero),
        sample_ids=(f"sample-{rank}",),
        codec_provenance={"source": "explicit-fixture"},
        position_kwargs={},
    )


class Data:
    def __init__(self, batches):
        self.batches = batches
        self.scanned_batches = self.completed_updates = 0
        self.last_update_scan = -1

    def get_batch(self, scan_index, device):
        assert scan_index == self.scanned_batches and device.type == "cpu"
        self.scanned_batches += 1
        return self.batches[scan_index]

    def commit_update(self, *, scan_index, update_index):
        assert scan_index == self.scanned_batches - 1 and update_index == self.completed_updates
        self.completed_updates += 1
        self.last_update_scan = scan_index

    def state_dict(self):
        return dict(
            fingerprint="toy-v1",
            scanned_batches=self.scanned_batches,
            completed_updates=self.completed_updates,
            last_update_scan=self.last_update_scan,
        )

    def validate_state_dict(self, state):
        if set(state) != set(self.state_dict()) or state["fingerprint"] != "toy-v1":
            raise ValueError("Wrong reader manifest")
        scans, updates, last = (
            state[k] for k in ("scanned_batches", "completed_updates", "last_update_scan")
        )
        if not all(
            type(v) is int for v in (scans, updates, last)
        ) or not 0 <= updates <= scans <= len(self.batches):
            raise ValueError("Invalid reader counters")
        if not -1 <= last < scans or (updates == 0) != (last == -1):
            raise ValueError("Invalid last update scan")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.scanned_batches, self.completed_updates, self.last_update_scan = (
            state[k] for k in ("scanned_batches", "completed_updates", "last_update_scan")
        )


def waveform_objective(output, batch):
    mask = batch["waveform_mask"]
    return TargetLoss(
        torch.where(mask, (output - batch["target_waveform"]).square(), 0).sum(),
        mask.sum(dtype=torch.int64),
    )


def make(mode, *, dropout=0, optimizer_cls=torch.optim.AdamW):
    model = ToySpeech(dropout)
    active = configure_trainable_components(model, mode)
    optimizer = optimizer_cls(active, lr=0.03, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler


def assert_tree(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            assert_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_tree(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("mode", ["codec_only", "text_and_codec", "waveform"])
def test_actual_step_two_local_checkpoint_resume_exact(mode, tmp_path):
    batches = [[sample(True)], [sample()], [sample(True)], [sample(rank=1)]]
    torch.manual_seed(891)
    initial = copy.deepcopy(ToySpeech(0.2).state_dict())

    def build():
        model, optimizer, scheduler = make(mode, dropout=0.2)
        model.load_state_dict(initial)
        return model, optimizer, scheduler, Data(batches)

    options = dict(device=torch.device("cpu"), waveform_objective=waveform_objective)
    cfg = OmniSpeechCfg(mode=mode, waveform_objective_id="cpu-mse-v1", max_updates=2, max_batches=4)
    model, opt, sched, data = build()
    torch.manual_seed(199)
    full = launch(cfg, data, model=model, optimizer=opt, scheduler=sched, **options)
    expected = copy.deepcopy(
        (
            model.state_dict(),
            opt.state_dict(),
            sched.state_dict(),
            data.state_dict(),
            torch.get_rng_state(),
        )
    )
    first, first_opt, first_sched, first_data = build()
    torch.manual_seed(199)
    path = tmp_path / "speech.pt"
    launch(
        OmniSpeechCfg(mode=mode, waveform_objective_id="cpu-mse-v1", max_updates=1, max_batches=2),
        first_data,
        model=first,
        optimizer=first_opt,
        scheduler=first_sched,
        checkpoint_path=path,
        **options,
    )
    resumed, resumed_opt, resumed_sched, resumed_data = build()
    state = launch(
        cfg,
        resumed_data,
        model=resumed,
        optimizer=resumed_opt,
        scheduler=resumed_sched,
        resume_from=path,
        **options,
    )
    assert state.state_dict() == full.state_dict() == dict(scanned_batches=4, completed_updates=2)
    actual = (
        resumed.state_dict(),
        resumed_opt.state_dict(),
        resumed_sched.state_dict(),
        resumed_data.state_dict(),
        torch.get_rng_state(),
    )
    assert_tree(actual, expected)
    assert all(item["step"].item() == 2 for item in resumed_opt.state.values())
    active = {
        "codec_only": {"talker"},
        "text_and_codec": {"thinker", "talker"},
        "waveform": {"code2wav"},
    }[mode]
    for name, value in resumed.state_dict().items():
        if name.split(".")[0] not in active and name != "forwards":
            assert torch.equal(value, initial[name])


@pytest.mark.parametrize("mode", ["codec_only", "text_and_codec", "waveform"])
def test_global_zero_preserves_forward_buffers_modes_rng_optimizer(mode):
    model, optimizer, scheduler = make(mode, dropout=0.4)
    model.eval()
    before = (
        copy.deepcopy(model.state_dict()),
        torch.get_rng_state().clone(),
        scheduler.state_dict(),
    )
    data = Data([[sample(True)]])
    state = launch(
        OmniSpeechCfg(mode=mode, waveform_objective_id="cpu-mse-v1"),
        data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        waveform_objective=waveform_objective,
    )
    assert state.state_dict() == dict(scanned_batches=1, completed_updates=0)
    assert (
        not optimizer.state and not model.training and all(not m.training for m in model.modules())
    )
    assert_tree((model.state_dict(), torch.get_rng_state(), scheduler.state_dict()), before)


@pytest.mark.parametrize(
    "damage",
    [
        "order",
        "shape",
        "dtype",
        "topology",
        "reader",
        "updates",
        "optimizer",
        "scheduler",
        "rng",
        "hyperparameter",
        "missing_adam",
        "rewound_adam",
    ],
)
def test_strict_checkpoint_preflight_rejects_without_live_mutation(damage, tmp_path):
    model, optimizer, scheduler = make("text_and_codec", dropout=0.1)
    data = Data([[sample()], [sample()]])
    path = tmp_path / "good.pt"
    config = OmniSpeechCfg(mode="text_and_codec", max_updates=2, max_batches=2)
    state = launch(
        config,
        data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        checkpoint_path=path,
    )
    payload = torch.load(path, weights_only=True)
    if damage == "order":
        payload["model"] = dict(reversed(list(payload["model"].items())))
    elif damage in ("shape", "dtype"):
        key = "talker.1.weight"
        payload["model"][key] = (
            payload["model"][key][:1] if damage == "shape" else payload["model"][key].float()
        )
    elif damage == "topology":
        payload["manifest"]["topology"]["pp"] = 2
    elif damage == "reader":
        payload["data"]["fingerprint"] = "wrong"
    elif damage == "updates":
        payload["task"]["completed_updates"] = 3
    elif damage == "optimizer":
        payload["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
    elif damage == "scheduler":
        payload["scheduler"]["last_epoch"] = 10
    elif damage == "hyperparameter":
        payload["optimizer"]["param_groups"][0]["lr"] = float("nan")
    elif damage == "missing_adam":
        del payload["optimizer"]["state"][0]
    elif damage == "rewound_adam":
        assert payload["optimizer"]["state"][0]["step"].item() == 2
        payload["optimizer"]["state"][0]["step"].fill_(1)
    else:
        payload["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    bad = tmp_path / "bad.pt"
    _write_verified_checkpoint(bad, payload)  # Valid checksum still requires semantic preflight.
    before = copy.deepcopy(
        (
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            data.state_dict(),
            state.state_dict(),
            torch.get_rng_state(),
        )
    )
    with pytest.raises(ValueError, match="preflight"):
        load_checkpoint(
            bad,
            cfg=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            state=state,
            token_group=None,
            gradient_reduction="sum",
        )
    after = (
        model.state_dict(),
        optimizer.state_dict(),
        scheduler.state_dict(),
        data.state_dict(),
        state.state_dict(),
        torch.get_rng_state(),
    )
    assert_tree(after, before)


def test_optimizer_cannot_include_unused_components_and_waveform_needs_objective():
    model = ToySpeech()
    configure_trainable_components(model, "codec_only")
    bad = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="no frozen"):
        launch(OmniSpeechCfg(), Data([[sample()]]), model=model, optimizer=bad, device="cpu")
    model, optimizer, _ = make("waveform")
    with pytest.raises(ValueError, match="caller-defined"):
        launch(
            OmniSpeechCfg(mode="waveform"),
            Data([[sample()]]),
            model=model,
            optimizer=optimizer,
            device="cpu",
        )
    with pytest.raises(NotImplementedError, match="CP=EP=PP"):
        launch(
            OmniSpeechCfg(cp_size=2),
            Data([[sample()]]),
            model=model,
            optimizer=optimizer,
            device="cpu",
        )


def serial_loss(model, batches, weighting):
    heads, labels, text_scores, text_labels = [[], [], []], [[], [], []], [], []
    for item in batches:
        output = model(
            item["input_ids"],
            item["model_context"],
            item["codes"],
            item["code_lengths"],
            prompt_lengths=item["prompt_lengths"],
            speaker="ethan",
            codebook_mask=item["codebook_mask"],
            position_kwargs={},
        )
        heads[0].append(output.first_logits.flatten(0, 1))
        labels[0].append(output.first_labels.flatten())
        for h in range(2):
            heads[h + 1].append(output.residual_logits[:, :, h].flatten(0, 1))
            labels[h + 1].append(output.residual_labels[:, :, h].flatten())
        text_scores.append(output.thinker_logits.flatten(0, 1))
        text_labels.append(item["labels"].flatten())
    sums, counts = [], []
    for scores, targets in zip(heads, labels, strict=True):
        target = torch.cat(targets)
        sums.append(F.cross_entropy(torch.cat(scores), target, reduction="sum"))
        counts.append(int(target.ne(-100).sum()))
    codec = (
        sum(sums) / sum(counts)
        if weighting == "combined"
        else sum(s / n for s, n in zip(sums, counts, strict=True) if n) / sum(n > 0 for n in counts)
    )
    return codec + F.cross_entropy(torch.cat(text_scores), torch.cat(text_labels))


def _sum_hook(group, bucket):
    future = dist.all_reduce(bucket.buffer(), group=group, async_op=True).get_future()
    return future.then(lambda result: result.value()[0])


def dp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40)
    )
    try:
        for weighting in ("combined", "per_head"):
            for wrapper, reduction in (
                (False, "sum"),
                (False, "mean"),
                (True, "mean"),
                (True, "sum"),
            ):
                torch.manual_seed(71)
                model, optimizer, scheduler = make("text_and_codec", optimizer_cls=torch.optim.SGD)
                oracle = copy.deepcopy(model)
                oracle_opt = torch.optim.SGD(
                    [p for p in oracle.parameters() if p.requires_grad], lr=0.03, weight_decay=0.1
                )
                global_batches = [sample(zero=True), sample(rank=1)]
                expected = serial_loss(oracle, global_batches, weighting)
                expected.backward()
                oracle_opt.step()
                if wrapper:
                    model = DistributedDataParallel(
                        model, process_group=dist.group.WORLD, find_unused_parameters=True
                    )
                    if reduction == "sum":
                        model.register_comm_hook(dist.group.WORLD, _sum_hook)
                # A full ignored scan must close a real DDP reducer cycle and
                # leave gradients/RNG/buffers unchanged before this valid scan.
                local_data = Data([[sample(True)], [global_batches[rank]]])
                state = launch(
                    OmniSpeechCfg(
                        mode="text_and_codec",
                        codec_weighting=weighting,
                        max_grad_norm=None,
                        max_batches=2,
                    ),
                    local_data,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device="cpu",
                    token_group=dist.group.WORLD,
                    gradient_reduction=reduction,
                )
                actual = model.module if wrapper else model
                assert state.completed_updates == 1
                for p, q in zip(actual.parameters(), oracle.parameters(), strict=True):
                    torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)
        # Two individually valid checkpoints from different global updates
        # must be rejected collectively before any rank applies either one.
        from pathlib import Path

        root = Path(rendezvous.removeprefix("file://")).parent
        model, optimizer, scheduler = make("text_and_codec")
        data = Data([[sample(rank=rank)], [sample(rank=rank)]])
        state = launch(
            OmniSpeechCfg(mode="text_and_codec"),
            data,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
            token_group=dist.group.WORLD,
            checkpoint_path=root / "first.pt",
        )
        launch(
            OmniSpeechCfg(mode="text_and_codec", max_updates=2, max_batches=2),
            data,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
            token_group=dist.group.WORLD,
            checkpoint_path=root / "second.pt",
            state=state,
        )
        before = copy.deepcopy(
            (
                model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                data.state_dict(),
                state.state_dict(),
                torch.get_rng_state(),
            )
        )
        with pytest.raises(ValueError, match="ranks disagree"):
            load_checkpoint(
                root / ("first.pt" if rank == 0 else "second.pt"),
                cfg=OmniSpeechCfg(mode="text_and_codec"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data=data,
                state=state,
                token_group=dist.group.WORLD,
                gradient_reduction="sum",
            )
        assert_tree(
            (
                model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                data.state_dict(),
                state.state_dict(),
                torch.get_rng_state(),
            ),
            before,
        )
        # One rank injects NaN into a real backward. All ranks reject before
        # optimizer state, parameter values or scheduler counters can advance.
        model, optimizer, scheduler = make("text_and_codec")
        before = copy.deepcopy(
            (dict(model.named_parameters()), optimizer.state_dict(), scheduler.state_dict())
        )
        hook = model.talker[1].weight.register_hook(
            lambda gradient: gradient * float("nan") if rank == 1 else gradient
        )
        with pytest.raises(FloatingPointError, match="no optimizer"):
            launch(
                OmniSpeechCfg(mode="text_and_codec"),
                Data([[sample(rank=rank)]]),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device="cpu",
                token_group=dist.group.WORLD,
            )
        hook.remove()
        assert_tree(
            (dict(model.named_parameters()), optimizer.state_dict(), scheduler.state_dict()), before
        )
    finally:
        dist.destroy_process_group()


def test_gloo_dp_sum_mean_and_ddp_match_serial_global_head_objectives(tmp_path):
    mp.spawn(dp_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)


def test_apply_failure_rolls_back_model_optimizer_scheduler_data_rng(tmp_path):
    class FailingData(Data):
        fail_once = False

        def load_state_dict(self, state):
            super().load_state_dict(state)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("injected reader apply failure")

    model, optimizer, scheduler = make("text_and_codec")
    data = FailingData([[sample()], [sample()]])
    path = tmp_path / "checkpoint.pt"
    cfg = OmniSpeechCfg(mode="text_and_codec")
    state = launch(
        cfg,
        data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        checkpoint_path=path,
    )
    launch(
        OmniSpeechCfg(mode="text_and_codec", max_updates=2, max_batches=2),
        data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        state=state,
    )
    before = copy.deepcopy(
        (
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            data.state_dict(),
            state.state_dict(),
            torch.get_rng_state(),
        )
    )
    data.fail_once = True
    with pytest.raises(RuntimeError, match="restored"):
        load_checkpoint(
            path,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            state=state,
            token_group=None,
            gradient_reduction="sum",
        )
    assert_tree(
        (
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            data.state_dict(),
            state.state_dict(),
            torch.get_rng_state(),
        ),
        before,
    )


def test_fsdp_rejected_before_accidental_replica_reduction(monkeypatch):
    from pithtrain.tasks import omni_speech as task

    model, optimizer, _ = make("codec_only")
    # Marker substitution exercises only the rejection branch; no sharded or
    # accelerator execution is claimed by this guard test.
    monkeypatch.setattr(task, "FSDPModule", ToySpeech)
    with pytest.raises(NotImplementedError, match="FSDP"):
        task.launch(
            OmniSpeechCfg(), Data([[sample()]]), model=model, optimizer=optimizer, device="cpu"
        )
    with pytest.raises(NotImplementedError, match="FSDP"):
        task.configure_trainable_components(model, "codec_only")


def test_finite_model_payload_edit_is_rejected_by_checksum_before_apply(tmp_path):
    model, optimizer, scheduler = make("text_and_codec")
    data = Data([[sample()]])
    path = tmp_path / "verified.pt"
    cfg = OmniSpeechCfg(mode="text_and_codec")
    state = launch(
        cfg,
        data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        checkpoint_path=path,
    )
    payload = torch.load(path, weights_only=True)
    payload["model"]["talker.1.weight"].add_(0.125)  # Shape and dtype remain valid.
    # Preserve the original checksum footer while modifying a finite payload.
    from pithtrain.tasks.omni_speech import _INTEGRITY_BYTES

    footer = path.read_bytes()[-_INTEGRITY_BYTES:]
    with path.open("wb") as stream:
        torch.save(payload, stream)
        stream.write(footer)
    before = copy.deepcopy(
        (
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            data.state_dict(),
            state.state_dict(),
        )
    )
    with pytest.raises(ValueError, match="preflight") as error:
        load_checkpoint(
            path,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            state=state,
            token_group=None,
            gradient_reduction="sum",
        )
    assert "SHA256/length" in str(error.value.__cause__)
    assert_tree(
        (
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            data.state_dict(),
            state.state_dict(),
        ),
        before,
    )


@pytest.mark.parametrize("clipping", [None, 1.0])
def test_nonfinite_speech_gradients_never_update_parameters_optimizer_scheduler(clipping):
    model, optimizer, scheduler = make("text_and_codec")
    before = copy.deepcopy(
        (dict(model.named_parameters()), optimizer.state_dict(), scheduler.state_dict())
    )
    hook = model.talker[1].weight.register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(FloatingPointError, match="no optimizer"):
        launch(
            OmniSpeechCfg(mode="text_and_codec", max_grad_norm=clipping),
            Data([[sample()]]),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
        )
    hook.remove()
    assert_tree(
        (dict(model.named_parameters()), optimizer.state_dict(), scheduler.state_dict()), before
    )


def test_conditional_adam_presence_and_per_parameter_steps_resume_exactly(tmp_path):
    class ConditionalSpeech(ToySpeech):
        def __init__(self):
            super().__init__()
            self.talker.register_parameter(
                "conditional", nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
            )

        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            if args[1].get("use_conditional"):
                output.first_logits = output.first_logits + self.talker.conditional * torch.arange(
                    7
                )
            return output

    samples = [sample(), sample(), sample()]
    for item in samples[1:]:
        item["model_context"]["use_conditional"] = True
    torch.manual_seed(491)
    initial = copy.deepcopy(ConditionalSpeech().state_dict())

    def make_conditional():
        model = ConditionalSpeech()
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(configure_trainable_components(model, "codec_only"), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        return model, optimizer, scheduler, Data([[item] for item in samples])

    model, optimizer, scheduler, data = make_conditional()
    state = launch(
        OmniSpeechCfg(max_updates=1, max_batches=1),
        data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        checkpoint_path=tmp_path / "step1.pt",
    )
    first = torch.load(tmp_path / "step1.pt", weights_only=True)
    entry = next(item for item in first["optimizer_schema"] if item["name"] == "talker.conditional")
    assert entry["present"] is False and entry["step"] is None
    launch(
        OmniSpeechCfg(max_updates=2, max_batches=2),
        data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        state=state,
        checkpoint_path=tmp_path / "step2.pt",
    )
    second = torch.load(tmp_path / "step2.pt", weights_only=True)
    steps = {item["name"]: item["step"] for item in second["optimizer_schema"]}
    assert steps["talker.conditional"] == 1 and steps["talker.1.weight"] == 2
    launch(
        OmniSpeechCfg(max_updates=3, max_batches=3),
        data,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        state=state,
    )
    expected = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    resumed, resumed_opt, resumed_sched, resumed_data = make_conditional()
    launch(
        OmniSpeechCfg(max_updates=3, max_batches=3),
        resumed_data,
        model=resumed,
        optimizer=resumed_opt,
        scheduler=resumed_sched,
        device="cpu",
        resume_from=tmp_path / "step2.pt",
    )
    assert_tree(
        (resumed.state_dict(), resumed_opt.state_dict(), resumed_sched.state_dict()), expected
    )


def checkpoint_save_preflight_worker(rank, rendezvous):
    from pathlib import Path

    from pithtrain.tasks.omni_speech import save_checkpoint
    from pithtrain.tasks.sft_lm import SFTState

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=25)
    )
    try:
        root = Path(rendezvous.removeprefix("file://")).parent
        for damage in ("gradient", "nonfinite", "reader", "write"):
            torch.manual_seed(61)
            model, optimizer, scheduler = make("codec_only")
            data, state = Data([[sample()]]), SFTState()
            if rank == 1:
                if damage == "gradient":
                    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
                elif damage == "nonfinite":
                    with torch.no_grad():
                        next(model.parameters()).flatten()[0] = float("nan")
                elif damage == "reader":
                    data.scanned_batches = 1
            target = root / (damage + ".pt")
            if damage == "write" and rank == 1:
                target = root / "missing-parent" / "speech.pt"
            before = copy.deepcopy(
                (optimizer.state_dict(), scheduler.state_dict(), state.state_dict())
            )
            with pytest.raises(
                ValueError, match="write failed" if damage == "write" else "save preflight"
            ):
                save_checkpoint(
                    target,
                    cfg=OmniSpeechCfg(),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    data=data,
                    state=state,
                    token_group=dist.group.WORLD,
                    gradient_reduction="sum",
                )
            assert_tree(
                (optimizer.state_dict(), scheduler.state_dict(), state.state_dict()), before
            )
            if damage != "write":
                assert not (root / (damage + ".pt.rank00000")).exists()
                assert not (root / (damage + ".pt.rank00001")).exists()
            # Reaching this collective on both ranks proves rejection did not
            # strand a healthy peer in agreement or let one begin another scan.
            dist.barrier()
        for failure in ("config", "batch", "objective"):
            model, optimizer, scheduler = make("waveform")
            data = Data([[sample()], [sample()]])
            cfg = OmniSpeechCfg(mode="waveform", waveform_objective_id="failure-fixture")
            if failure == "config":
                cfg.max_batches = rank + 1
            if failure == "batch" and rank == 1:

                def bad_reader(*args):
                    raise ValueError("Local reader failed before model execution")

                data.get_batch = bad_reader

            def checked_objective(output, batch):
                item = waveform_objective(output, batch)
                if failure == "objective" and rank == 1:
                    return TargetLoss(item.loss_sum, item.target_count.float())
                return item

            before = copy.deepcopy(
                (dict(model.named_parameters()), optimizer.state_dict(), scheduler.state_dict())
            )
            with pytest.raises(ValueError):
                launch(
                    cfg,
                    data,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device="cpu",
                    token_group=dist.group.WORLD,
                    waveform_objective=checked_objective,
                )
            assert_tree(
                (dict(model.named_parameters()), optimizer.state_dict(), scheduler.state_dict()),
                before,
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_gloo_save_preflight_and_writer_failures_reject_all_ranks(tmp_path):
    mp.spawn(
        checkpoint_save_preflight_worker,
        args=((tmp_path / "save-gloo").as_uri(),),
        nprocs=2,
        join=True,
    )


def sharded_protocol_model(mode, mesh):
    """Real FSDP2 protocol fixture; native architectural setup has separate tests."""
    from torch.distributed.fsdp import (
        MixedPrecisionPolicy,
        fully_shard,
        register_fsdp_forward_method,
    )

    model = ToySpeech().float()
    configure_trainable_components(model, mode)
    for name in ("thinker", "talker", "code2wav", ""):
        module = model if not name else getattr(model, name)
        fully_shard(
            module,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=bool(name),
            ),
        )
        module.set_gradient_divide_factor(1.0)
        module.set_force_sum_reduction_for_comms(True)
    register_fsdp_forward_method(model, "decode_training_codes")
    model._omni_speech_fsdp_plan = dict(
        schema=1,
        mode=mode,
        mesh=mesh,
        group=mesh.get_group(),
        gradient_reduction="sum",
        master_dtype=torch.float32,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        require_backward=True,
        fixed_topology=True,
        thinker_sharding="protocol_fixture_v1",
        owned_modules=("thinker", "talker", "code2wav", ""),
        materialized=True,
        weights_initialized=True,
    )
    return model


def independent_bfloat_loss(model, batches, weighting):
    sums = [[], [], []]
    counts = [0, 0, 0]
    texts, text_count = [], 0
    for batch in batches:
        cast = {name: p.to(torch.bfloat16) for name, p in model.named_parameters()}
        output = torch.func.functional_call(
            model,
            cast,
            (batch["input_ids"], batch["model_context"], batch["codes"], batch["code_lengths"]),
            dict(
                prompt_lengths=batch["prompt_lengths"],
                speaker="ethan",
                codebook_mask=batch["codebook_mask"],
                position_kwargs={},
            ),
        )
        heads = [(output.first_logits, output.first_labels)] + [
            (output.residual_logits[:, :, h], output.residual_labels[:, :, h]) for h in range(2)
        ]
        for h, (logits, labels) in enumerate(heads):
            sums[h].append(
                F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
            )
            counts[h] += int(labels.ne(-100).sum())
        texts.append(
            F.cross_entropy(
                output.thinker_logits.float().flatten(0, 1),
                batch["labels"].flatten(),
                reduction="sum",
            )
        )
        text_count += int(batch["labels"].ne(-100).sum())
    codec = (
        sum(map(sum, sums)) / sum(counts)
        if weighting == "combined"
        else sum(sum(values) / count for values, count in zip(sums, counts, strict=True) if count)
        / sum(count > 0 for count in counts)
    )
    return codec + sum(texts) / text_count


def fsdp_task_worker(rank, rendezvous):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from pithtrain.tasks.omni_speech import clip_sharded_grad_norm

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40)
    )
    # This deliberately small protocol model has no native ChatML/config.
    # Isolate only the parameter-free batch validator; all collective planning,
    # real FSDP communication, objectives, optimizer and checkpoint code execute.
    # Native data admission is covered by test_omni_speech_batch_contract and
    # the actual native speech sharded acceptance driver, not this fixture.
    from unittest.mock import patch

    validation = patch("pithtrain.modules.omni_speech_batch_contract.validate_speech_batch")
    admitted = validation.start()
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        for weighting in ("combined", "per_head"):
            torch.manual_seed(126)
            model = sharded_protocol_model("text_and_codec", mesh)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.03
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
            torch.manual_seed(126)
            oracle = ToySpeech().float()
            configure_trainable_components(oracle, "text_and_codec")
            oracle_opt = torch.optim.AdamW(
                [p for p in oracle.parameters() if p.requires_grad], lr=0.03
            )
            oracle_sched = torch.optim.lr_scheduler.StepLR(oracle_opt, 1, gamma=0.9)
            global_batches = [sample(zero=True), sample(rank=1)]
            for _ in range(2):
                loss = independent_bfloat_loss(oracle, global_batches, weighting)
                loss.backward()
                gradients = [p.grad for p in oracle.parameters() if p.grad is not None]
                norm = sum(
                    torch.linalg.vector_norm(g, dtype=torch.float64).square() for g in gradients
                ).sqrt()
                coefficient = (0.2 / norm.clamp_min(1e-12)).clamp(max=1)
                for grad in gradients:
                    grad.mul_(coefficient.to(grad.dtype))
                oracle_opt.step()
                oracle_opt.zero_grad(set_to_none=True)
                oracle_sched.step()
            data = Data([[sample(True)], [global_batches[rank]], [global_batches[rank]]])
            result = launch(
                OmniSpeechCfg(
                    mode="text_and_codec",
                    codec_weighting=weighting,
                    max_updates=2,
                    max_batches=3,
                    max_grad_norm=0.2,
                ),
                data,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device="cpu",
                token_group=mesh.get_group(),
            )
            assert result.scanned_batches == 3 and result.completed_updates == 2
            assert scheduler.last_epoch == 2
            assert model.forwards.item() == 2  # The ignored scan restored the buffer.
            for (name, actual), (_, expected) in zip(
                model.named_parameters(), oracle.named_parameters(), strict=True
            ):
                assert isinstance(actual, DTensor) and actual.dtype == torch.float32
                torch.testing.assert_close(actual.full_tensor(), expected, rtol=0, atol=0, msg=name)
                assert actual.grad is None
            assert all(state["step"].item() == 2 for state in optimizer.state.values())
        # Norm uses every shard exactly once, including a rank with zero local
        # norm. No full_tensor() is needed to calculate/clip production gradients.
        parameter = next(p for p in model.parameters() if p.requires_grad)
        parameter.grad = torch.zeros_like(parameter)
        parameter.grad.to_local().flatten()[0] = 3 * rank
        expected_sq = torch.tensor(float(9 * rank), dtype=torch.float64)
        dist.all_reduce(expected_sq)
        actual_norm = clip_sharded_grad_norm(model, 0.1, group=mesh.get_group())
        torch.testing.assert_close(actual_norm, expected_sq.sqrt(), rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        before = {name: p.to_local().clone() for name, p in model.named_parameters()}
        before_opt, before_sched = (
            copy.deepcopy(optimizer.state_dict()),
            copy.deepcopy(scheduler.state_dict()),
        )
        original_forward = model.talker.forward

        def inject(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            if rank == 1:
                output.register_hook(lambda gradient: gradient * float("nan"))
            return output

        model.talker.forward = inject
        with pytest.raises(FloatingPointError, match="no optimizer"):
            launch(
                OmniSpeechCfg(mode="text_and_codec"),
                Data([[sample(rank=rank)]]),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device="cpu",
                token_group=mesh.get_group(),
            )
        model.talker.forward = original_forward
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.to_local(), before[name], rtol=0, atol=0)
        assert_tree(optimizer.state_dict(), before_opt)
        assert_tree(scheduler.state_dict(), before_sched)
    finally:
        validation.stop()
        dist.destroy_process_group()
    assert admitted.call_count > 0
    assert all(
        call.args[1].config.model_type == "qwen3_omni_moe" for call in admitted.call_args_list
    )


def test_real_fsdp_task_sum_global_zero_clip_and_nonfinite_rejection(tmp_path):
    mp.spawn(fsdp_task_worker, args=((tmp_path / "fsdp-gloo").as_uri(),), nprocs=2, join=True)


def fsdp_native_task_worker(rank, root_string):
    from pathlib import Path

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from pithtrain.contexts import training
    from pithtrain.tasks.omni_speech import _native_checkpoint
    from pithtrain.tasks.sft_lm import SFTState

    torch.set_num_threads(1)
    root = Path(root_string)
    dist.init_process_group(
        "gloo",
        init_method=(root / "native-fsdp").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    # This deliberately small protocol model has no native ChatML/config.
    # Isolate only the parameter-free batch validator; all collective planning,
    # real FSDP communication, objectives, optimizer and checkpoint code execute.
    # Native data admission is covered by test_omni_speech_batch_contract and
    # the actual native speech sharded acceptance driver, not this fixture.
    from unittest.mock import patch

    validation = patch("pithtrain.modules.omni_speech_batch_contract.validate_speech_batch")
    admitted = validation.start()
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        batches = [[sample(True)], [sample(rank=rank)], [sample(True)], [sample(rank=rank)]]

        def build():
            torch.manual_seed(998)
            model = sharded_protocol_model("text_and_codec", mesh)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.03
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
            return model, optimizer, scheduler, Data(batches), SFTState()

        def local(value):
            if isinstance(value, DTensor):
                return value.to_local().detach().clone()
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, dict):
                return {key: local(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(local(item) for item in value)
            return copy.deepcopy(value)

        def snapshot(parts):
            model, optimizer, scheduler, data, state = parts
            return local(
                (
                    model.state_dict(),
                    optimizer.state_dict(),
                    scheduler.state_dict(),
                    data.state_dict(),
                    state.state_dict(),
                    torch.get_rng_state(),
                )
            )

        def run(parts, updates, scans, **extra):
            model, optimizer, scheduler, data, state = parts
            return launch(
                OmniSpeechCfg(mode="text_and_codec", max_updates=updates, max_batches=scans),
                data,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                device="cpu",
                token_group=mesh.get_group(),
                **extra,
            )

        full = build()
        torch.manual_seed(31 + rank)
        run(full, 2, 4)
        expected = snapshot(full)
        first = build()
        torch.manual_seed(31 + rank)
        native = dict(
            checkpoint_root=root / "shards",
            checkpoint_contract={"fixture": "speech-task-native-v1"},
        )
        run(first, 1, 2, **native)
        assert first[4].state_dict() == dict(scanned_batches=2, completed_updates=1)
        resumed = build()
        run(resumed, 2, 4, resume_scan=2, **native)
        assert_tree(snapshot(resumed), expected)
        assert resumed[2].last_epoch == 2
        assert all(value["step"].item() == 2 for value in resumed[1].state.values())
        zero = build()
        run(zero, 0, 1, resume_scan=1, **native)
        assert zero[4].state_dict() == dict(scanned_batches=1, completed_updates=0)
        assert not zero[1].state and zero[2].last_epoch == 0
        # Same-layout model or FSDP arithmetic-boundary drift must reject
        # every rank before changing model, optimizer, reader or RNG state.
        from pithtrain.modules.distributed_checkpoint import DistributedCheckpointError

        missing = object()
        for owner, key in (
            (vars(zero[0].config), "same_shape_semantics"),
            (zero[0]._omni_speech_fsdp_plan, "thinker_sharding"),
        ):
            previous = owner.get(key, missing)
            if rank == 1:
                owner[key] = "changed"
            before = snapshot(zero)
            with pytest.raises(
                DistributedCheckpointError, match="^Ranks disagree on load contract$"
            ):
                _native_checkpoint(
                    "load",
                    root=root / "shards",
                    scan=2,
                    cfg=OmniSpeechCfg(mode="text_and_codec"),
                    model=zero[0],
                    optimizer=zero[1],
                    scheduler=zero[2],
                    data=zero[3],
                    state=zero[4],
                    contract=native["checkpoint_contract"],
                    token_group=mesh.get_group(),
                )
            assert_tree(snapshot(zero), before)
            if rank == 1:
                if previous is missing:
                    del owner[key]
                else:
                    owner[key] = previous
            dist.barrier()
        # One rank's pending gradient must be rejected before native collectives;
        # temporary global training bindings must be restored on healthy peers.
        prior_model = getattr(training, "model", None)
        if rank == 1:
            parameter = next(p for p in zero[0].parameters() if p.requires_grad)
            parameter.grad = torch.ones_like(parameter)
        before = snapshot(zero)
        with pytest.raises(ValueError, match="wrapper preflight"):
            _native_checkpoint(
                "save",
                root=root / "rejected",
                scan=1,
                cfg=OmniSpeechCfg(mode="text_and_codec"),
                model=zero[0],
                optimizer=zero[1],
                scheduler=zero[2],
                data=zero[3],
                state=zero[4],
                contract=native["checkpoint_contract"],
                token_group=mesh.get_group(),
            )
        assert getattr(training, "model", None) is prior_model
        assert_tree(snapshot(zero), before)
        assert not (root / "rejected").exists()
    finally:
        validation.stop()
        dist.destroy_process_group()
    assert admitted.call_count > 0
    assert all(
        call.args[1].config.model_type == "qwen3_omni_moe" for call in admitted.call_args_list
    )


def test_real_fsdp_task_native_checkpoint_zero_and_step_two_resume_exact(tmp_path):
    mp.spawn(fsdp_native_task_worker, args=(str(tmp_path),), nprocs=2, join=True)


def native_codec_checkpoint_parts():
    """Actual tiny native Code2Wav semantics inside the existing task protocol fixture."""
    from pithtrain.models.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
    from pithtrain.tasks.sft_lm import SFTState
    from tests.test_qwen3_omni_code2wav import tiny_config

    model = ToySpeech()
    model.code2wav = Qwen3OmniMoeCode2Wav(tiny_config())
    model.config.code2wav_config = model.code2wav.config
    active = configure_trainable_components(model, "waveform")
    optimizer = torch.optim.AdamW(active, lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    return model, optimizer, scheduler, Data([[sample()]]), SFTState()


def checkpoint_complete_snapshot(parts):
    from pithtrain.tasks.omni_speech import _rng_state

    model, optimizer, scheduler, data, state = parts
    return copy.deepcopy(
        (
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            data.state_dict(),
            state.state_dict(),
            _rng_state(),
            [(name, module.training) for name, module in model.named_modules()],
            [(name, parameter.grad) for name, parameter in model.named_parameters()],
        )
    )


def checkpoint_call_options(parts):
    model, optimizer, scheduler, data, state = parts
    return dict(
        cfg=OmniSpeechCfg(mode="waveform", waveform_objective_id="native-codec-contract-v1"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        state=state,
        token_group=None,
        gradient_reduction="sum",
    )


@pytest.mark.parametrize(
    "damage",
    ["theta_config", "theta_runtime", "theta_both", "eps_config", "eps_runtime", "eps_both"],
)
def test_native_same_shape_config_and_runtime_drift_rejected_before_apply(damage, tmp_path):
    from pithtrain.tasks.omni_speech import save_checkpoint

    producer = native_codec_checkpoint_parts()
    target = tmp_path / "native-codec.pt"
    save_checkpoint(target, **checkpoint_call_options(producer))
    resumed = native_codec_checkpoint_parts()
    codec = resumed[0].code2wav
    if damage.startswith("theta"):
        if damage.endswith(("config", "both")):
            codec.config.rope_parameters["rope_theta"] *= 2
        if damage.endswith(("runtime", "both")):
            codec.pre_transformer.theta *= 2
    else:
        if damage.endswith(("config", "both")):
            codec.config.rms_norm_eps *= 2
        if damage.endswith(("runtime", "both")):
            for module in codec.modules():
                if hasattr(module, "variance_epsilon"):
                    module.variance_epsilon *= 2
    assert [(n, tuple(v.shape)) for n, v in producer[0].state_dict().items()] == [
        (n, tuple(v.shape)) for n, v in resumed[0].state_dict().items()
    ]
    before = checkpoint_complete_snapshot(resumed)
    with pytest.raises(ValueError, match="preflight") as rejected:
        load_checkpoint(target, **checkpoint_call_options(resumed))
    expected_reason = (
        "contract mismatch"
        if damage.endswith("both")
        else ("Runtime RoPE theta" if damage.startswith("theta") else "Runtime RMS epsilon")
    )
    assert expected_reason in str(rejected.value.__cause__)
    assert_tree(checkpoint_complete_snapshot(resumed), before)


@pytest.mark.parametrize("attribute", ["theta", "variance_epsilon"])
def test_save_rejects_config_shell_disagreeing_with_actual_native_runtime(attribute, tmp_path):
    from pithtrain.tasks.omni_speech import save_checkpoint

    parts = native_codec_checkpoint_parts()
    runtime = (
        parts[0].code2wav.pre_transformer
        if attribute == "theta"
        else parts[0].code2wav.pre_transformer.norm
    )
    setattr(runtime, attribute, getattr(runtime, attribute) * 2)
    before = checkpoint_complete_snapshot(parts)
    target = tmp_path / "inconsistent.pt"
    with pytest.raises(ValueError, match="save preflight") as rejected:
        save_checkpoint(target, **checkpoint_call_options(parts))
    expected_reason = "Runtime RoPE theta" if attribute == "theta" else "Runtime RMS epsilon"
    assert expected_reason in str(rejected.value.__cause__)
    assert not target.exists()
    assert_tree(checkpoint_complete_snapshot(parts), before)


def test_native_codec_contract_same_source_roundtrip_and_canonical_no_addresses(tmp_path):
    from pithtrain.tasks.omni_speech import _model_contract, save_checkpoint

    producer, resumed = native_codec_checkpoint_parts(), native_codec_checkpoint_parts()
    left, right = _model_contract(producer[0]), _model_contract(resumed[0])
    assert left == right
    assert left["identity"]["schema"] == 3
    assert left["identity"]["modules"]["code2wav.pre_transformer"]["numerical_attributes"]["theta"]
    target = tmp_path / "same-source.pt"
    save_checkpoint(target, **checkpoint_call_options(producer))
    load_checkpoint(target, **checkpoint_call_options(resumed))
    assert_tree(checkpoint_complete_snapshot(resumed), checkpoint_complete_snapshot(producer))


def test_model_contract_rejects_detached_composite_config_and_binds_implementation():
    from pithtrain.tasks.omni_speech import _model_contract

    parts = native_codec_checkpoint_parts()
    model = parts[0]
    expected = _model_contract(model)
    assert expected["identity"]["modules"]["code2wav"]["implementation"].endswith(
        ".Qwen3OmniMoeCode2Wav"
    )
    model.config.code2wav_config = copy.deepcopy(model.config.code2wav_config)
    model.config.code2wav_config.rms_norm_eps *= 2
    with pytest.raises(ValueError, match="component config"):
        _model_contract(model)


def test_model_contract_rejects_legacy_checkpoint_without_semantic_identity(tmp_path):
    from pithtrain.tasks.omni_speech import _read_verified_checkpoint, save_checkpoint

    parts = native_codec_checkpoint_parts()
    target = tmp_path / "legacy.pt"
    save_checkpoint(target, **checkpoint_call_options(parts))
    payload = _read_verified_checkpoint(target)
    payload["manifest"].pop("model_contract")
    payload["manifest"]["version"] = 1
    _write_verified_checkpoint(target, payload)
    before = checkpoint_complete_snapshot(parts)
    with pytest.raises(ValueError, match="preflight"):
        load_checkpoint(target, **checkpoint_call_options(parts))
    assert_tree(checkpoint_complete_snapshot(parts), before)


def _speech_schedule(factor):
    return lambda step: 1.0 if step < 2 else factor


@pytest.mark.parametrize("change", ["code", "closure"])
def test_speech_changed_lambda_rejected_without_mutating_state(change, tmp_path):
    from pithtrain.tasks.omni_speech import save_checkpoint

    original, restored = (
        list(native_codec_checkpoint_parts()),
        list(native_codec_checkpoint_parts()),
    )
    original[2] = torch.optim.lr_scheduler.LambdaLR(original[1], _speech_schedule(1.0))
    restored[2] = torch.optim.lr_scheduler.LambdaLR(
        restored[1],
        _speech_schedule(0.1) if change == "closure" else (lambda step: 1.0 if step < 2 else 0.1),
    )
    target = tmp_path / "speech-lambda.pt"
    save_checkpoint(target, **checkpoint_call_options(original))
    before = checkpoint_complete_snapshot(restored)
    with pytest.raises(ValueError, match="preflight"):
        load_checkpoint(target, **checkpoint_call_options(restored))
    assert_tree(before, checkpoint_complete_snapshot(restored))


def test_speech_identical_lambda_resume_second_optimizer_update(tmp_path):
    torch.manual_seed(501)
    initial = copy.deepcopy(ToySpeech(0.2).state_dict())

    def build():
        model, optimizer, _ = make("text_and_codec", dropout=0.2)
        model.load_state_dict(initial)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _speech_schedule(0.5))
        return model, optimizer, scheduler, Data([[sample()], [sample(rank=1)]])

    options = dict(device=torch.device("cpu"))
    cfg = OmniSpeechCfg(mode="text_and_codec", max_updates=2, max_batches=2)
    original, opt, sched, data = build()
    torch.manual_seed(39)
    state = launch(cfg, data, model=original, optimizer=opt, scheduler=sched, **options)
    expected = checkpoint_complete_snapshot((original, opt, sched, data, state))
    first, first_opt, first_sched, first_data = build()
    torch.manual_seed(39)
    path = tmp_path / "speech-resume.pt"
    launch(
        OmniSpeechCfg(mode="text_and_codec", max_updates=1, max_batches=1),
        first_data,
        model=first,
        optimizer=first_opt,
        scheduler=first_sched,
        checkpoint_path=path,
        **options,
    )
    resumed, opt, sched, data = build()
    state = launch(
        cfg, data, model=resumed, optimizer=opt, scheduler=sched, resume_from=path, **options
    )
    assert sched.last_epoch == 2
    assert all(value["step"].item() == 2 for value in opt.state.values())
    assert_tree(expected, checkpoint_complete_snapshot((resumed, opt, sched, data, state)))


def fsdp_waveform_microbatch_worker(rank, rendezvous):
    from unittest.mock import patch

    from torch.distributed.device_mesh import init_device_mesh

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40)
    )
    # Reuse the real FSDP protocol fixture; only native batch admission is mocked.
    validation = patch("pithtrain.modules.omni_speech_batch_contract.validate_speech_batch")
    validation.start()
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        torch.manual_seed(126)
        model = sharded_protocol_model("waveform", mesh)
        torch.manual_seed(126)
        oracle = ToySpeech().float()
        configure_trainable_components(oracle, "waveform")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0,
            foreach=False,
        )
        oracle_opt = torch.optim.AdamW(
            oracle.code2wav.parameters(), lr=1e-4, weight_decay=0, foreach=False
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1.0)

        def batches(scan, worker_rank):
            result = []
            for microbatch in range(5):
                batch = sample(zero=scan in (0, 2), rank=worker_rank)
                batch["target_waveform"] += (microbatch + worker_rank) * 0.037
                if scan not in (0, 2):
                    batch["waveform_mask"][0, 1] = microbatch % 2 == 0
                    if scan == 3 and worker_rank == 0:
                        batch["waveform_mask"].zero_()
                result.append(batch)
            return result

        data = Data([batches(scan, rank) for scan in range(4)])
        expected_gradients = {}
        optimizer_steps = []

        def check_gradients(opt, _args, _kwargs):
            optimizer_steps.append(1)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad.dtype == torch.float32
                    torch.testing.assert_close(
                        parameter.grad.full_tensor(), expected_gradients[name], rtol=0, atol=0
                    )

        hook = optimizer.register_step_pre_hook(check_gradients)
        state = None
        for scan in range(4):
            oracle_opt.zero_grad(set_to_none=True)
            global_batches = batches(scan, 0) + batches(scan, 1)
            count = sum(int(batch["waveform_mask"].sum()) for batch in global_batches)
            if count:
                losses = []
                for batch in global_batches:
                    # Fresh casts give each microbatch its own BF16 compute leaf,
                    # so contributions accumulate into the original FP32 masters.
                    cast = {n: p.to(torch.bfloat16) for n, p in oracle.code2wav.named_parameters()}
                    wave = torch.func.functional_call(
                        oracle.code2wav, cast, (batch["codes"].to(torch.bfloat16),)
                    ).squeeze(-1)
                    losses.append(waveform_objective(wave, batch).loss_sum)
                (sum(losses) / count).backward()
                norm = sum(
                    p.grad.double().square().sum() for p in oracle.code2wav.parameters()
                ).sqrt()
                coefficient = (0.2 / norm.clamp_min(1e-12)).clamp(max=1)
                for name, parameter in oracle.named_parameters():
                    if parameter.requires_grad:
                        parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
                        expected_gradients[name] = parameter.grad.detach().clone()
                oracle_opt.step()
            before = {
                name: parameter.to_local().detach().clone()
                for name, parameter in model.named_parameters()
            }
            before_optimizer = copy.deepcopy(optimizer.state_dict())
            before_rng = torch.get_rng_state().clone()
            before_forwards = model.forwards.clone()
            state = launch(
                OmniSpeechCfg(
                    mode="waveform",
                    max_updates=2,
                    max_batches=scan + 1,
                    max_grad_norm=0.2,
                    waveform_objective_id="cpu-mse-v1",
                ),
                data,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                device="cpu",
                token_group=mesh.get_group(),
                waveform_objective=waveform_objective,
            )
            if not count:
                assert_tree(optimizer.state_dict(), before_optimizer)
                assert_tree(torch.get_rng_state(), before_rng)
                assert_tree(model.forwards, before_forwards)
            for name, parameter in model.named_parameters():
                assert parameter.grad is None
                if not count:
                    assert_tree(parameter.to_local(), before[name])
                torch.testing.assert_close(
                    parameter.full_tensor(), oracle.get_parameter(name), rtol=0, atol=0
                )
            assert state.scanned_batches == scan + 1
            assert state.completed_updates == (scan + 1) // 2
            assert scheduler.last_epoch == state.completed_updates
        assert len(optimizer_steps) == 2 and model.forwards.item() == 10
        hook.remove()
    finally:
        validation.stop()
        dist.destroy_process_group()


def test_real_fsdp_waveform_microbatch_fp32_accumulation(tmp_path):
    mp.spawn(
        fsdp_waveform_microbatch_worker,
        args=((tmp_path / "waveform-fp32-gloo").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_checkpoint_combine_mode_change_rejected_before_any_state_apply(tmp_path):
    from pithtrain.tasks.omni_speech import save_checkpoint

    producer = native_codec_checkpoint_parts()
    resumed = native_codec_checkpoint_parts()
    # model_contract binds the actual module attribute, not a caller config shell.
    producer[0].thinker.deterministic_token_layout = False
    resumed[0].thinker.deterministic_token_layout = True
    target = tmp_path / "combine-mode.pt"
    save_checkpoint(target, **checkpoint_call_options(producer))
    before = checkpoint_complete_snapshot(resumed)
    with pytest.raises(ValueError, match="preflight") as rejected:
        load_checkpoint(target, **checkpoint_call_options(resumed))
    assert "contract mismatch" in str(rejected.value.__cause__)
    assert_tree(checkpoint_complete_snapshot(resumed), before)
