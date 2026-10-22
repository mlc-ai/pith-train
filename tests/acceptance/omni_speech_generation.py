"""Bounded native Thinker -> 16-group Talker -> Code2Wav GPU generation parity.

Uses the source-bound tiny speech_chain fixture and the actual HF 5.17 full
Qwen3OmniMoeForConditionalGeneration.generate entry point. Every media case
requires unforced nonempty speech. EOS/ragged and decoder chunk controls are
reported separately from natural end-to-end generation.
No downloads, released-weight quality, sampling-quality or FSDP generation claim.
"""

import argparse
import copy
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.func import functional_call

from tests.acceptance import omni_speech_chain as chain

SCHEMA = "qwen3_omni_generation_acceptance_v2"
TEXT_TOKENS = 4
CODEC_FRAMES = 8
OUTPUT_NSE = chain.OUTPUT_NSE
MEDIA = chain.MEDIA


class MethodCall(nn.Module):
    """Functional-call dispatch reaches methods without converting RoPE buffers."""

    def __init__(self, model, method):
        super().__init__()
        self.model, self.method = model, method

    def forward(self, *args, **kwargs):
        return self.method(self.model, *args, **kwargs)


def compute(model, method, *args, **kwargs):
    wrapper = MethodCall(model, method)
    parameters = {name: p.to(torch.bfloat16) for name, p in wrapper.named_parameters()}
    return functional_call(
        wrapper, (parameters, dict(wrapper.named_buffers())), args, kwargs, strict=True
    )


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


@contextmanager
def preserve_generation_state(*models):
    from pithtrain.tasks.omni_speech import _rng_state, _set_rng

    rng = _rng_state()
    modes = [(module, module.training) for model in models for module in model.modules()]
    mutable = [
        (module, name, hasattr(module, name), copy.deepcopy(getattr(module, name, None)))
        for module, _ in modes
        for name in ("rope_deltas", "generation_config")
    ]
    before = [state_hash(model) for model in models]
    identities = [[(name, id(p)) for name, p in model.named_parameters()] for model in models]
    proof = {}
    try:
        for model in models:
            model.eval()
        yield proof
    finally:
        for module, name, existed, value in mutable:
            if existed:
                setattr(module, name, value)
            elif hasattr(module, name):
                delattr(module, name)
        for module, mode in modes:
            module.training = mode
        _set_rng(rng)
        proof.update(
            weights_unchanged=before == [state_hash(model) for model in models],
            parameter_identities_unchanged=identities
            == [[(name, id(p)) for name, p in model.named_parameters()] for model in models],
            modes_restored=all(module.training == mode for module, mode in modes),
            rng_restored=True,
        )
        chain.exact(_rng_state(), rng, "generation-rng")
        if not all(proof.values()):
            raise AssertionError("Generation/control changed model state or parameter identities")


def prompt_sample(sample):
    end = sample["prompt_lengths"][0]
    ids = sample["input_ids"][:, :end].clone()
    context = dict(sample["model_context"], input_ids=ids, attention_mask=torch.ones_like(ids))
    context.pop("position_ids", None)
    return dict(
        input_ids=ids,
        model_context=context,
        prompt_lengths=[end],
        position_kwargs=sample["position_kwargs"],
    )


def official_composite(reference):
    """Use the real HF composite with the already paired official components.

    Construct its ordinary class on meta to avoid duplicating stored weights;
    install only the same official submodules, never a project generation loop.
    No reference forward/generate implementation is replaced.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration as HFComposite,
    )

    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        result = HFComposite(copy.deepcopy(reference.config))
    for name in ("thinker", "talker", "code2wav"):
        if type(getattr(result, name)) is not type(getattr(reference, name)):
            raise chain.PreconditionError("Composite reference requires official HF components")
        setattr(result, name, getattr(reference, name))
    if not result.has_talker or state_hash(result) != state_hash(reference):
        raise chain.PreconditionError("Composite reference did not retain the paired full state")
    return result


@contextmanager
def observe_method(owner, name, observer):
    """Observe an original result without replacing its value or its algorithm."""
    original = getattr(owner, name)
    existed, previous = name in owner.__dict__, owner.__dict__.get(name)

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        observer(args, kwargs, result)
        return result

    setattr(owner, name, observed)
    try:
        yield
    finally:
        if existed:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def position_kwargs(batch):
    names = {"audio_seqlens": "audio_feature_lengths", "second_per_grids": "video_second_per_grid"}
    return {names.get(key, key): value for key, value in batch["position_kwargs"].items()}


def hf_codec(model, condition, batch, *, max_frames=CODEC_FRAMES, logits_processor=None):
    options = dict(
        condition,
        max_new_tokens=max_frames + 1,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=model.config.talker_config.codec_eos_token_id,
        pad_token_id=model.config.talker_config.codec_pad_id,
        suppress_tokens=[
            i
            for i in range(
                model.config.talker_config.code_predictor_config.vocab_size,
                model.config.talker_config.text_config.vocab_size,
            )
            if i != model.config.talker_config.codec_eos_token_id
        ],
        output_hidden_states=True,
        return_dict_in_generate=True,
        **position_kwargs(batch),
    )
    if logits_processor is not None:
        options["logits_processor"] = logits_processor
    generated = model.talker.generate(**options)
    frames = [hidden[-1] for hidden in generated.hidden_states if hidden[-1] is not None]
    rows = condition["inputs_embeds"].shape[0]
    codes = (
        torch.stack(frames, 1) if frames else condition["talker_input_ids"].new_empty((rows, 0, 16))
    )
    first = generated.sequences[:, :max_frames]
    eos_mask = first == model.config.talker_config.codec_eos_token_id
    reached = eos_mask.any(-1)
    positions = torch.arange(first.shape[1], device=first.device)[None].expand_as(first)
    lengths = torch.where(eos_mask, positions, codes.shape[1]).min(-1).values
    if int(lengths.max()) > max_frames:
        raise AssertionError("Official generation exceeded the reviewed codec cap")
    valid = torch.arange(codes.shape[1], device=codes.device)[None] < lengths[:, None]
    codes = codes.masked_fill(~valid[..., None], 0)
    waves = []
    for row, length in enumerate(lengths.tolist()):
        waves.append(
            model.code2wav.chunked_decode(
                codes[row : row + 1, :length].transpose(1, 2), chunk_size=300, left_context_size=25
            )[0].float()
            if length
            else condition["inputs_embeds"].new_empty((1, 0), dtype=torch.float32)
        )
    return SimpleNamespace(codes=codes, code_lengths=lengths, reached_eos=reached, waveforms=waves)


def hf_generate(model, batch, *, eos_id=None, text_tokens=TEXT_TOKENS, codec_frames=CODEC_FRAMES):
    """Invoke the unmodified official composite, including its conditioning state machine."""
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeForConditionalGeneration as HFComposite,
    )

    from tests.acceptance.omni_reference_attention import reference_attention_kwargs

    if type(model) is not HFComposite:
        raise TypeError("Generation reference must be the actual official HF composite class")
    eos_id = model.config.im_end_token_id if eos_id is None else eos_id
    capture = {}

    def thinker(args, kwargs, output):
        capture["thinker"] = output

    def talker(args, kwargs, output):
        capture["talker"] = output
        capture["condition"] = {
            key: kwargs[key].detach().clone()
            for key in (
                "inputs_embeds",
                "talker_input_ids",
                "attention_mask",
                "trailing_text_hidden",
                "tts_pad_embed",
            )
        }

    def decoder(args, kwargs, output):
        capture["codes"] = args[0].detach().clone()
        capture["waveform"] = output.detach().clone()

    context = dict(batch["model_context"])
    context.pop("input_ids", None)
    context.pop("position_ids", None)
    options = dict(
        context,
        thinker_max_new_tokens=text_tokens,
        thinker_eos_token_id=eos_id,
        thinker_do_sample=False,
        thinker_pad_token_id=model.config.im_end_token_id,
        thinker_output_logits=True,
        talker_max_new_tokens=codec_frames + 1,
        talker_do_sample=False,
        talker_temperature=1.0,
        talker_pad_token_id=model.config.talker_config.codec_pad_id,
        # Public override for the tiny RVQ vocabulary. The released default
        # suppresses the final1024 tokens and cannot index a41-token fixture.
        talker_suppress_tokens=[
            i
            for i in range(
                model.config.talker_config.code_predictor_config.vocab_size,
                model.config.talker_config.text_config.vocab_size,
            )
            if i != model.config.talker_config.codec_eos_token_id
        ],
        return_audio=True,
    )
    options.update(
        {
            f"thinker_{key}": value
            for key, value in reference_attention_kwargs(model.thinker).items()
        }
    )
    with (
        observe_method(model.thinker, "generate", thinker),
        observe_method(model.talker, "generate", talker),
        observe_method(model.code2wav, "chunked_decode", decoder),
    ):
        sequences, waveform = model.generate(input_ids=batch["input_ids"], **options)
    if set(capture) != {"thinker", "talker", "condition", "codes", "waveform"}:
        raise AssertionError("Official composite did not execute every component")
    codes = capture["codes"].transpose(1, 2)
    if (
        codes.ndim != 3
        or codes.shape[0] != 1
        or codes.shape[-1] != 16
        or codes.shape[1] > codec_frames
    ):
        raise AssertionError("Official composite codec output exceeds the reviewed B1/16-group cap")
    chain.exact(waveform, capture["waveform"].float(), "official-returned-waveform")
    generated = sequences[:, batch["input_ids"].shape[1] :]
    reached = bool((generated == eos_id).any())
    first = capture["talker"].sequences[:, :codec_frames]
    codec_eos = (first == model.config.talker_config.codec_eos_token_id).any(-1)
    speech = SimpleNamespace(
        codes=codes,
        code_lengths=codes.new_tensor([codes.shape[1]]),
        reached_eos=codec_eos,
        waveforms=[waveform[0].float()],
    )
    return SimpleNamespace(
        sequences=sequences,
        text_reached_eos=reached,
        speech=speech,
        logits=[value.float().clone() for value in capture["thinker"].logits],
        condition=capture["condition"],
        reference_entrypoint="transformers.Qwen3OmniMoeForConditionalGeneration.generate",
    )


def native_generate(
    model, batch, *, eos_id=None, text_tokens=TEXT_TOKENS, codec_frames=CODEC_FRAMES
):
    rows, original = [], model.thinker.forward_epilog
    had_override = "forward_epilog" in model.thinker.__dict__
    prior_override = model.thinker.__dict__.get("forward_epilog")

    def observe(*args, **kwargs):
        logits = original(*args, **kwargs)
        rows.append(logits[:, -1].float().clone())
        return logits

    # A read-only observer: returns the exact original tensor and is always removed.
    model.thinker.forward_epilog = observe
    try:
        output = model.generate(
            batch["input_ids"],
            batch["model_context"],
            max_text_tokens=text_tokens,
            max_codec_frames=codec_frames,
            text_eos_token_id=eos_id,
            do_sample=False,
            position_kwargs=batch["position_kwargs"],
        )
    finally:
        if had_override:
            model.thinker.forward_epilog = prior_override
        else:
            del model.thinker.forward_epilog
    return output, rows


def compare_speech(actual, expected):
    if (actual is None) != (expected is None):
        raise AssertionError("Speech presence differs")
    report = dict(
        speech_presence_exact=True,
        code_groups=16,
        code_exact=True,
        code_lengths_exact=True,
        reached_eos_exact=True,
        code_lengths=[],
        reached_eos=[],
        native_codes=[],
        hf_codes=[],
        waveforms=[],
    )
    if actual is None:
        return report
    chain.exact(actual.code_lengths, expected.code_lengths, "code-lengths")
    chain.exact(actual.reached_eos, expected.reached_eos, "codec-eos")
    if actual.codes.shape[-1] != 16 or expected.codes.shape[-1] != 16:
        raise AssertionError("Generation must exercise all sixteen codebooks")
    for row, length in enumerate(actual.code_lengths.tolist()):
        chain.exact(actual.codes[row, :length], expected.codes[row, :length], "all-codebooks")
        if actual.codes[row, length:].ne(0).any():
            raise AssertionError("Finished row retained codec tokens after its EOS")
        left, right = actual.waveforms[row], expected.waveforms[row]
        error = chain.nse(left, right)
        if error >= OUTPUT_NSE or left.shape != right.shape or left.ndim != 2 or left.shape[0] != 1:
            raise AssertionError("Waveform shape/finite/numerical parity failed")
        if (length == 0) != (left.numel() == 0):
            raise AssertionError("Empty codec rows must produce exactly empty waveforms")
        report["waveforms"].append(
            dict(
                row=row,
                codec_frames=length,
                samples=left.shape[-1],
                shape=list(left.shape),
                finite=True,
                nse=error,
                native_sha256=chain.tensor_tree(left)["sha256"],
                hf_sha256=chain.tensor_tree(right)["sha256"],
            )
        )
    report.update(
        code_lengths=actual.code_lengths.tolist(),
        reached_eos=actual.reached_eos.tolist(),
        native_codes=actual.codes.tolist(),
        hf_codes=expected.codes.tolist(),
    )
    return report


def compare_generation(actual, rows, expected, prompt_length):
    chain.exact(actual.sequences, expected.sequences, "generated-text")
    expected_steps = expected.sequences.shape[1] - prompt_length
    if (
        actual.text_reached_eos != expected.text_reached_eos
        or len(expected.logits) != expected_steps
        or len(rows) != expected_steps
    ):
        raise AssertionError("Text EOS or generation/conditioning forward count differs")
    # Both state machines expose one actual decode forward per sampled token.
    errors = [
        chain.nse(left, right)
        for left, right in zip(rows[:expected_steps], expected.logits, strict=True)
    ]
    if not errors or any(error >= OUTPUT_NSE for error in errors):
        raise AssertionError("Native/HF Thinker generation logits differ")
    return dict(
        status="PASS",
        scope="unforced_generation",
        text_exact=True,
        text_eos_exact=True,
        generated_tokens=actual.sequences[:, prompt_length:].tolist(),
        native_generated_tokens=actual.sequences[:, prompt_length:].tolist(),
        hf_generated_tokens=expected.sequences[:, prompt_length:].tolist(),
        text_reached_eos=actual.text_reached_eos,
        thinker_logits_nse=errors,
        native_conditioning_forwards=0,
        **compare_speech(actual.speech, expected.speech),
    )


def ragged_control(native, hf, condition, batch):
    from transformers import LogitsProcessor, LogitsProcessorList

    from pithtrain.models.qwen3_omni_speech import TalkerCondition, generate_speech_from_condition

    eos = native.config.talker_config.codec_eos_token_id
    repeated = {key: value.expand(2, *value.shape[1:]).clone() for key, value in condition.items()}
    # This control is text-only; no media grids are duplicated or invented.
    positions, _ = hf.talker.get_rope_index(
        repeated["talker_input_ids"], attention_mask=repeated["attention_mask"]
    )
    prepared = TalkerCondition(
        repeated["inputs_embeds"],
        repeated["talker_input_ids"],
        repeated["attention_mask"],
        positions,
        repeated["trailing_text_hidden"],
        repeated["tts_pad_embed"],
    )
    original, counter = native.talker.forward, [0]
    had_override = "forward" in native.talker.__dict__
    prior_override = native.talker.__dict__.get("forward")

    def controlled(*args, **kwargs):
        output = original(*args, **kwargs)
        logits = output.logits.clone()
        logits[:, -1] = -torch.inf
        logits[0, -1, eos] = 0
        logits[1, -1, eos if counter[0] == 2 else 3] = 0
        counter[0] += 1
        return type(output)(logits, output.last_hidden_state)

    class ForcedEOS(LogitsProcessor):
        def __call__(self, input_ids, scores):
            result = torch.full_like(scores, -torch.inf)
            result[0, eos] = 0
            result[1, eos if input_ids.shape[1] == 2 else 3] = 0
            return result

    native.talker.forward = controlled
    try:
        actual = compute(
            native,
            lambda model: generate_speech_from_condition(
                model.talker, model.code2wav, prepared, max_frames=4, do_sample=False
            ),
        )
    finally:
        if had_override:
            native.talker.forward = prior_override
        else:
            del native.talker.forward
    expected = compute(
        hf,
        lambda model: hf_codec(
            model,
            repeated,
            batch,
            max_frames=4,
            logits_processor=LogitsProcessorList([ForcedEOS()]),
        ),
    )
    if actual.code_lengths.tolist() != [0, 2] or actual.reached_eos.tolist() != [True, True]:
        raise AssertionError("Controlled individual EOS did not produce ragged [0,2] lengths")
    return dict(
        status="PASS",
        scope="controlled_codec_eos_boundary",
        expected_lengths=[0, 2],
        **compare_speech(actual, expected),
    )


@contextmanager
def hf_position_metadata(hf, config):
    owner = hf.talker.config
    existed = hasattr(owner, "audio_config")
    previous = getattr(owner, "audio_config", None)
    owner.audio_config = copy.deepcopy(config.thinker_config.audio_config)
    try:
        yield
    finally:
        if existed:
            owner.audio_config = previous
        else:
            del owner.audio_config


def multi_turn(batch, config):
    # Earlier user/assistant turns, with the original media-bearing user turn
    # and its exact grids/features retained. These are explicit prompt tokens.
    ids = batch["input_ids"]
    prefix = ids.new_tensor(
        [
            [
                config.im_start_token_id,
                config.user_token_id,
                198,
                40,
                config.im_end_token_id,
                198,
                config.im_start_token_id,
                config.assistant_token_id,
                198,
                41,
                config.im_end_token_id,
                198,
            ]
        ]
    )
    sequence = torch.cat((prefix, ids), 1)
    return dict(
        batch,
        input_ids=sequence,
        model_context=dict(
            batch["model_context"], input_ids=sequence, attention_mask=torch.ones_like(sequence)
        ),
        prompt_lengths=[sequence.shape[1]],
    )


def require_nonempty_case(report, name):
    if not report["waveforms"] or any(w["samples"] <= 0 for w in report["waveforms"]):
        raise AssertionError(f"{name}: every natural media case must produce nonempty speech")


def chunk_boundary(native, hf, speech):
    """Decoder-only301-frame boundary, explicitly not a natural generation claim."""
    length = int(speech.code_lengths[0])
    if length < 1:
        raise AssertionError("Chunk control requires a real unforced codec prefix")
    source = speech.codes[:, :length].transpose(1, 2)
    frames, chunk, context = 301, 300, 25
    codes = source.repeat(1, 1, (frames + length - 1) // length)[..., :frames]
    actual = compute(
        native,
        lambda model: model.code2wav.chunked_decode(
            codes, chunk_size=chunk, left_context_size=context
        ),
    )
    expected = compute(
        hf,
        lambda model: model.code2wav.chunked_decode(
            codes, chunk_size=chunk, left_context_size=context
        ),
    )
    error = chain.nse(actual.float(), expected.float())
    if actual.shape != expected.shape or error >= OUTPUT_NSE or actual.shape[-1] <= 0:
        raise AssertionError("Code2Wav300/301 chunk-boundary parity failed")
    return dict(
        status="PASS",
        scope="decoder_component_chunk_boundary",
        source="repeated_unforced_codec_prefix_not_natural_301_frame_generation",
        source_frames=length,
        code_groups=16,
        frames=frames,
        chunk_size=chunk,
        left_context_size=context,
        chunks=2,
        codes_sha256=chain.tensor_tree(codes)["sha256"],
        shape=list(actual.shape),
        finite=True,
        nse=error,
    )


@torch.no_grad()
def validate(args, config, samples, native, hf):
    native, hf = native.model, official_composite(hf.model)
    cases, boundaries, text_context = {}, {}, None
    with preserve_generation_state(native, hf) as preservation, hf_position_metadata(hf, config):
        for sample in samples:
            batch = prompt_sample(chain.move(sample, args.device))
            actual, rows = compute(native, native_generate, batch)
            expected = compute(hf, hf_generate, batch)
            report = compare_generation(actual, rows, expected, batch["input_ids"].shape[1])
            require_nonempty_case(report, sample["media_kind"])
            report["media_kind"] = sample["media_kind"]
            cases[sample["media_kind"]] = report
            chain.write_json(args.output / f"generation-{sample['media_kind']}.json", report)
            if sample["media_kind"] == "text":
                text_context = batch, expected, actual
        if set(cases) != set(MEDIA):
            raise AssertionError("All five unforced media cases are mandatory")
        batch, previous, actual_text = text_context
        variants = {
            "multiturn": (multi_turn(batch, config), {}),
            "cap": (batch, dict(text_tokens=2, codec_frames=1)),
        }
        for name, (prompt, options) in variants.items():
            actual, rows = compute(native, native_generate, prompt, **options)
            expected = compute(hf, hf_generate, prompt, **options)
            report = compare_generation(actual, rows, expected, prompt["input_ids"].shape[1])
            require_nonempty_case(report, name)
            report.update(
                scope=f"unforced_{name}_generation",
                limits=options or dict(text_tokens=TEXT_TOKENS, codec_frames=CODEC_FRAMES),
            )
            if name == "cap" and (
                actual.text_reached_eos
                or actual.speech.reached_eos.any()
                or actual.sequences.shape[1] != prompt["input_ids"].shape[1] + 2
                or actual.speech.code_lengths.tolist() != [1]
            ):
                raise AssertionError("Natural cap control did not reach both original caps")
            boundaries[name] = report
            chain.write_json(args.output / f"generation-{name}.json", report)
        generated = previous.sequences[0, batch["input_ids"].shape[1] :]
        if generated.numel() < 2 or generated[0] == generated[1]:
            raise AssertionError("Unforced fixture cannot supply a distinct delayed-EOS control")
        eos_id = int(generated[1])
        actual, rows = compute(native, native_generate, batch, eos_id=eos_id)
        expected = compute(hf, hf_generate, batch, eos_id=eos_id)
        boundary = compare_generation(actual, rows, expected, batch["input_ids"].shape[1])
        if (
            not actual.text_reached_eos
            or actual.sequences.shape[1] != batch["input_ids"].shape[1] + 2
        ):
            raise AssertionError("Explicit delayed EOS did not stop after exactly two tokens")
        require_nonempty_case(boundary, "delayed_eos")
        boundary.update(
            scope="explicit_delayed_text_eos_control", eos_id=eos_id, eos_generated_index=1
        )
        boundaries["eos"] = boundary
        chain.write_json(args.output / "generation-eos.json", boundary)
        boundaries["ragged"] = ragged_control(native, hf, previous.condition, batch)
        chain.write_json(args.output / "generation-ragged.json", boundaries["ragged"])
        boundaries["chunk"] = chunk_boundary(native, hf, actual_text.speech)
        chain.write_json(args.output / "generation-chunk.json", boundaries["chunk"])
    return dict(
        status="PASS",
        cases=cases,
        boundaries=boundaries,
        **preservation,
        native_parameter_count=len(list(native.named_parameters())),
        reference="official Transformers5.17 full Qwen3OmniMoeForConditionalGeneration.generate",
        reference_entrypoint="transformers.Qwen3OmniMoeForConditionalGeneration.generate",
        precision="FP32 model leaves/rotary buffers; functional BF16 compute parameters",
        nonempty_waveform_cases=len(cases),
        immediate_text_eos="NATIVE_CPU_CONTROL_ONLY; official_composite_audio_parity_NOT_ESTABLISHED",
        independent_attention_math="NOT_ESTABLISHED; separate original FP64 math gates required",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema=SCHEMA,
        status="NOT_RUN",
        gpu_acceptance="NOT_RUN",
        seed=chain.SEED,
        released_config_revision=chain.REVISION,
        limits=dict(text_tokens=TEXT_TOKENS, codec_frames=CODEC_FRAMES),
        thresholds=dict(output_nse=OUTPUT_NSE),
        full_size="NOT_RUN",
        fsdp_generation="UNSUPPORTED",
    )
    started, phase = time.monotonic(), "source_precondition"
    try:
        result["source"] = chain.source_identity(args.expected_commit)
        result["runtime"] = chain.runtime_precondition(args)
        config, samples, _ = chain.load_fixture(args.prepared, args.expected_commit)
        if (
            config.talker_config.num_code_groups != 16
            or config.code2wav_config.num_quantizers != 16
        ):
            raise chain.PreconditionError("All sixteen codebooks are mandatory")
        result["fixture_sha256"] = chain.sha(args.prepared / "fixture.json")
        phase = "model_setup"
        native, hf, attention = chain.build_pair(config, args.device)
        result["reference_attention"] = attention
        phase = "generation"
        result.update(validate(args, config, samples, native, hf))
        result["gpu_acceptance"] = "PASS_THIS_GATE_ONLY"
    except Exception as error:
        result.update(chain.failure_report(phase, error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        chain.write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
