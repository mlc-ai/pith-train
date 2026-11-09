"""CPU calibration of the audio key-bias structural-zero bound (omni_structural_zero).

Measures on CPU, independently of every GPU result:

1. FP64, with the attention softmax kept in FP64: which audio-encoder parameters have a
   structurally zero gradient, for native and HF. Expected: exactly the AUDIO_KEY_BIAS
   set, far below every other parameter.
2. BF16, the compute dtype of every gate: the residue ratio |g_k| / |g_q| of each layer's
   key and query bias gradients, per implementation, for the whole vector and for each
   half (the local shard of a DP2 FSDP rank). The grid covers the official full-size
   audio configuration (full gate: BF16 parameters, one microbatch) and the tiny audio
   configurations of the staged audio and staged multimodal gates (FP32 master parameters
   with BF16 compute, and BF16 parameters; eight accumulated microbatches), the real
   fixture recordings and random features, several initialization seeds, and two
   objectives (random output probe; cross entropy through a random head). Seed 0 of the
   full-size configuration reproduces the full gate's audio-tower weights bit for bit.
   Two kernel variants are used. "emulated" evaluates BF16 GEMMs and convolutions as FP32
   kernels on BF16 inputs with the output rounded to BF16, the rounding points of GPU BF16
   kernels (FP32 accumulation, BF16 output), and is fast enough for the full-size model;
   "cpu" uses PyTorch's own CPU BF16 kernels, for the tiny configurations.
   Each configuration gets its own recommended bound: MARGIN times its largest ratio,
   rounded up to two significant figures. Ratios whose query gradient is exactly zero
   (one-frame windows) are undefined and skipped.
3. Negative controls: native attention faults that break shift invariance (RoPE on keys,
   key layer norm, bias on the first half of each window's keys) and the ratio they give.

python -m tests.acceptance.omni_key_bias_calibration --official-config CONFIG \
    --dataset BUNDLE --out OUT.json
"""

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tests.acceptance import omni_full_size as text_gate
from tests.acceptance.omni_reference_precision import FP32GradientReference
from tests.acceptance.omni_structural_zero import (
    AUDIO_KEY_BIAS,
    CALIBRATED_FIELDS,
    MARGIN,
    ceil_two_significant,
)

REPO = Path(__file__).resolve().parents[2]
STD = 0.02
VOCAB = 257
FP64_ZERO = 1e-9
FAULTS = ("rope_on_keys", "key_layer_norm", "bias_on_half_the_keys")
MODES = ("bf16", "bf16-master-fp32", "fp64")
KERNELS = ("emulated", "cpu")
STATISTICS = ("key_weight", "query_bias")
STATISTIC = "key_weight"
SHARDS = (1, 2, 4)
# Shard layouts each gate evaluates: the full gate is one unsharded GPU; the staged gates
# shard dim 0 over at most four FSDP ranks.
EVALUATED_SHARDS = dict(official=(1,), staged=SHARDS, multimodal=SHARDS)
BOTH = ("native", "hf")


def encoder(config, implementation):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder as HFAudio,
    )

    from pithtrain.models.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder

    config = copy.deepcopy(config)
    config._attn_implementation = "eager"
    return (HFAudio if implementation == "hf" else Qwen3OmniMoeAudioEncoder)(config)


def initialize(model, seed):
    """The full gate's name-seeded BF16 values: weights N(0, STD), biases 0, norms 1."""
    named = {"audio_tower." + name: value for name, value in model.named_parameters()}
    text_gate.initialize(named, STD, seed)


def prepared(config, implementation, mode, seed):
    """(callable, parameters) of one encoder in a calibration precision mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    model = encoder(config, implementation)
    initialize(model, seed)
    if mode == "fp64":
        model = model.double()
        return model, dict(model.named_parameters())
    model = model.to(torch.bfloat16)
    if mode == "bf16-master-fp32":
        reference = FP32GradientReference(model)
        return reference, reference.master_parameters
    return model, dict(model.named_parameters())


def flatten(batch):
    """HF Thinker layout: valid mel frames of every sample, concatenated."""
    mask = batch.model_inputs["feature_attention_mask"].bool()
    features = batch.model_inputs["input_features"].permute(0, 2, 1)[mask].permute(1, 0)
    return features.float().contiguous(), mask.sum(-1)


def random_features(config, lengths, seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(config.num_mel_bins, sum(lengths), generator=generator)
    return features, torch.tensor(lengths)


def objective(output, kind, seed):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64 if output.dtype == torch.float64 else torch.float32
    output = output.to(dtype)
    if kind == "probe":
        return (output * torch.randn(output.shape, generator=generator, dtype=dtype)).sum()
    head = torch.randn(output.shape[-1], VOCAB, generator=generator, dtype=dtype)
    labels = torch.randint(VOCAB, (output.shape[0],), generator=generator)
    return F.cross_entropy(output @ head / output.shape[-1] ** 0.5, labels, reduction="sum")


@contextlib.contextmanager
def exact_softmax(enabled):
    """Keep FP64 softmax in FP64; both attentions request FP32, which caps FP64 runs."""
    if not enabled:
        yield
        return
    original = torch.nn.functional.softmax

    def softmax(input, dim=None, _stacklevel=3, dtype=None):
        return original(input, dim=dim, dtype=None if input.dtype == torch.float64 else dtype)

    torch.nn.functional.softmax = softmax
    try:
        yield
    finally:
        torch.nn.functional.softmax = original


@contextlib.contextmanager
def emulated_bf16_kernels(enabled):
    """BF16 GEMMs and convolutions as FP32 kernels on BF16 inputs, rounded to BF16 output."""
    if not enabled:
        yield
        return
    linear, conv2d, matmul = F.linear, F.conv2d, torch.matmul

    def bf16(*tensors):
        present = [tensor for tensor in tensors if tensor is not None]
        return bool(present) and all(tensor.dtype == torch.bfloat16 for tensor in present)

    def up(tensor):
        return None if tensor is None else tensor.float()

    def emulated_linear(input, weight, bias=None):
        if not bf16(input, weight, bias):
            return linear(input, weight, bias)
        return linear(up(input), up(weight), up(bias)).to(torch.bfloat16)

    def emulated_conv2d(input, weight, bias=None, *args, **kwargs):
        if not bf16(input, weight, bias):
            return conv2d(input, weight, bias, *args, **kwargs)
        return conv2d(up(input), up(weight), up(bias), *args, **kwargs).to(torch.bfloat16)

    def emulated_matmul(input, other, **kwargs):
        if kwargs or not bf16(input, other):
            return matmul(input, other, **kwargs)
        return matmul(up(input), up(other)).to(torch.bfloat16)

    F.linear, F.conv2d, torch.matmul = emulated_linear, emulated_conv2d, emulated_matmul
    try:
        yield
    finally:
        F.linear, F.conv2d, torch.matmul = linear, conv2d, matmul


def rotate_keys(keys):
    """Standard rotate-half RoPE over each window's positions (a fault for audio keys)."""
    count, _, dim = keys.shape
    inverse = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.arange(count, dtype=torch.float32)[:, None] * inverse[None]
    cos, sin = (
        torch.cat((value, value), -1)[:, None].to(keys.dtype)
        for value in (angles.cos(), angles.sin())
    )
    first, second = keys[..., : dim // 2], keys[..., dim // 2 :]
    return keys * cos + torch.cat((-second, first), -1) * sin


@contextlib.contextmanager
def native_fault(fault):
    """Replace the native audio attention body with a shift-invariance-breaking variant."""
    if fault is None:
        yield
        return
    if fault not in FAULTS:
        raise ValueError(f"Unknown fault: {fault}")
    from pithtrain.models import qwen3_omni_audio

    original = qwen3_omni_audio.Qwen3OmniMoeAudioAttention.forward

    def forward(self, hidden_states, lengths):
        shape = (-1, self.num_heads, self.head_dim)
        q, k, v = (
            projection(hidden_states).reshape(shape)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        bias = self.k_proj.bias.reshape(self.num_heads, self.head_dim)
        outputs = []
        for queries, keys, values in zip(
            q.split(lengths), k.split(lengths), v.split(lengths), strict=True
        ):
            if fault == "rope_on_keys":
                keys = rotate_keys(keys)
            elif fault == "key_layer_norm":
                keys = F.layer_norm(keys, keys.shape[-1:])
            else:
                half = keys.shape[0] // 2
                keys = torch.cat((keys[:half], keys[half:] - bias))
            queries, keys, values = (
                tensor.transpose(0, 1).unsqueeze(0) for tensor in (queries, keys, values)
            )
            scores = torch.matmul(queries, keys.transpose(2, 3)) * self.head_dim**-0.5
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(queries.dtype)
            outputs.append(torch.matmul(probabilities, values).transpose(1, 2).contiguous())
        return self.out_proj(torch.cat(outputs, dim=1).reshape(hidden_states.shape))

    qwen3_omni_audio.Qwen3OmniMoeAudioAttention.forward = forward
    try:
        yield
    finally:
        qwen3_omni_audio.Qwen3OmniMoeAudioAttention.forward = original


def gradient_norms(model, implementation, mode, kernels, microbatches, fault=None):
    """FP64 L2 norms of every parameter gradient after accumulation, whole and per shard.

    parts[n] holds the norms of the n dim-0 shards (FSDP Shard(0) local tensors) for each
    n in SHARDS. model is a prepared() pair; gradients are cleared first, so one pair serves
    many cases.
    """
    call, parameters = model
    if kernels not in KERNELS or (fault is not None and implementation != "native"):
        raise ValueError("Unknown kernels or a fault outside the native implementation")
    for parameter in parameters.values():
        parameter.grad = None
    if mode == "bf16-master-fp32":
        call.normalized = False
    emulate = mode != "fp64" and kernels == "emulated"
    with native_fault(fault), exact_softmax(mode == "fp64"), emulated_bf16_kernels(emulate):
        for features, lengths, kind, objective_seed in microbatches:
            features = features.double() if mode == "fp64" else features
            if implementation == "hf":
                output = call(features, feature_lens=lengths).last_hidden_state
            else:
                output = call(features, lengths).last_hidden_state
            objective(output, kind, objective_seed).backward()
    norms = {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"Missing or nonfinite calibration gradient: {name}")
        rows = gradient.detach().double()
        norms[name] = dict(
            l2=rows.norm().item(),
            parts={str(n): [part.norm().item() for part in rows.chunk(n, dim=0)] for n in SHARDS},
            shape=list(gradient.shape),
        )
    return norms


def layer_ratios(norms):
    """Per layer, the key-bias gradient norm over each candidate scale, per shard layout.

    query_bias: the same layer's q_proj.bias gradient norm. key_weight: the k_proj.weight
    gradient norm over sqrt(input width), the size of the key-gradient rows whose rounded
    sum is the residue. Shard i of n pairs the i-th dim-0 shards of both tensors, as one
    FSDP rank holds them. A ratio is None where its scale is zero: a one-frame window has a
    constant softmax, so its bias gradients are exactly zero; the gates' criterion fails
    closed on such a ratio, and calibration skips it.
    """
    rows = []
    for name, norm in norms.items():
        if AUDIO_KEY_BIAS.match("audio_tower." + name) is None:
            continue
        prefix = name.removesuffix("k_proj.bias")
        query, weight = norms[prefix + "q_proj.bias"], norms[prefix + "k_proj.weight"]
        width = weight["shape"][-1] ** 0.5
        row = dict(
            parameter="audio_tower." + name,
            key_l2=norm["l2"],
            query_l2=query["l2"],
            key_weight_l2=weight["l2"],
        )
        for statistic in STATISTICS:
            row[statistic] = {}
            for n, keys in norm["parts"].items():
                if statistic == "query_bias":
                    scales = query["parts"][n]
                else:
                    scales = [value / width for value in weight["parts"][n]]
                row[statistic][n] = [
                    k / s if s > 0 else None for k, s in zip(keys, scales, strict=True)
                ]
        rows.append(row)
    if not rows:
        raise AssertionError("No audio key-bias parameters found")
    return rows


def defined_ratios(rows, statistic=None, shards=None):
    statistic, shards = statistic or STATISTIC, shards or SHARDS
    return [
        value
        for row in rows
        for n in shards
        for value in row[statistic][str(n)]
        if value is not None
    ]


def structural_scan(norms):
    """Parameters whose FP64 gradient is below FP64_ZERO of the largest gradient norm."""
    largest = max(norm["l2"] for norm in norms.values())
    relative = {"audio_tower." + name: norm["l2"] / largest for name, norm in norms.items()}
    zero = sorted(name for name, value in relative.items() if value < FP64_ZERO)
    expected = sorted(name for name in relative if AUDIO_KEY_BIAS.match(name))
    return dict(
        structurally_zero=zero,
        equals_audio_key_bias_set=zero == expected,
        audio_key_bias_parameters=len(expected),
        largest_zero_relative=max((relative[name] for name in zero), default=None),
        smallest_other_relative=min(v for name, v in relative.items() if name not in zero),
        threshold=FP64_ZERO,
    )


def case_max(rows, statistic=None, shards=None):
    return max(defined_ratios(rows, statistic, shards), default=None)


def case_min(rows, statistic=None, shards=None):
    return min(defined_ratios(rows, statistic, shards), default=None)


def groups(official, tiny, fixture, recordings, quick):
    """(prefix, config, mode, kernels, seed, cases, implementations) per prepared model.

    cases are (label, microbatches). official is the full gate's audio config; tiny maps a
    staged gate's name to its audio config. Every prefix starts with its configuration
    name. One full-size case also runs on PyTorch's CPU BF16 kernels (native only; native
    and HF arithmetic coincide there), to cross-check the emulated kernels.
    """
    inputs = [("fixture", fixture)]
    if not quick:
        inputs += [(f"random-{n}", random_features(official, [n], n)) for n in (700, 3000)]
    objectives = (("probe", 0),) if quick else (("probe", 0), ("probe", 1), ("ce", 0))
    for seed in (0,) if quick else (0, 1):
        cases = [
            (f"{name}/{kind}{index}", [(features, lengths, kind, index)])
            for name, (features, lengths) in inputs
            for kind, index in objectives
        ]
        yield f"official/seed{seed}", official, "bf16", "emulated", seed, cases, BOTH
    if not quick:
        cross = [("fixture/probe0", [(*fixture, "probe", 0)])]
        yield "official/seed0/cpu-kernels", official, "bf16", "cpu", 0, cross, ("native",)
    random_lengths = ([1], [99], [100], [101], [201, 17], [300, 100])
    for prefix, config in tiny.items():
        sources = [("recordings", recordings)] + [
            (f"random-{'+'.join(map(str, values))}", [random_features(config, values, index)])
            for index, values in enumerate(random_lengths[:2] if quick else random_lengths)
        ]
        cases = [
            (f"{name}/{kind}", [(*items[index % len(items)], kind, index) for index in range(8)])
            for name, items in sources
            for kind in ("probe", "ce")
        ]
        for seed in range(2 if quick else 10):
            for mode in ("bf16-master-fp32", "bf16"):
                for kernels in KERNELS:
                    label = f"{prefix}/seed{seed}/{mode}/{kernels}"
                    yield label, config, mode, kernels, seed, cases, BOTH


def summarize(report, names, statistic):
    """Per configuration: largest ratio as its gate evaluates it, bound and fault detection.

    A gate fails when any evaluated ratio exceeds the bound, so a fault is detected in a
    shard layout when its largest ratio there does; every_layer additionally requires its
    smallest ratio to exceed the bound.
    """
    configurations = {}
    for name in names:
        shards = EVALUATED_SHARDS[name]
        cases = [case for case in report["cases"] if case["label"].startswith(name + "/")]
        values = [
            value for case in cases for value in defined_ratios(case["layers"], statistic, shards)
        ]
        largest = max(values)
        bound = ceil_two_significant(MARGIN * largest)
        faults = {}
        for fault in report["faults"]:
            if fault["label"] != name:
                continue
            layouts = {}
            for n in shards:
                high = case_max(fault["layers"], statistic, (n,))
                low = case_min(fault["layers"], statistic, (n,))
                layouts[str(n)] = dict(
                    max_ratio=high, min_ratio=low, detected=high > bound, every_layer=low > bound
                )
            faults[fault["fault"]] = layouts
        detection = [
            entry["max_ratio"] / bound for layouts in faults.values() for entry in layouts.values()
        ]
        scans = [scan for scan in report["fp64"] if scan["label"] == name]
        by_shards = {}
        for n in shards:
            by_shards[str(n)] = max(
                value for case in cases for value in defined_ratios(case["layers"], statistic, (n,))
            )
        by_kernels = {}
        for kernels in KERNELS:
            by_kernels[kernels] = max(
                (
                    value
                    for case in cases
                    if case["kernels"] == kernels
                    for value in defined_ratios(case["layers"], statistic, shards)
                ),
                default=None,
            )
        configurations[name] = dict(
            evaluated_shards=list(shards),
            cases=len(cases),
            ratios=len(values),
            undefined_ratios=sum(
                value is None
                for case in cases
                for row in case["layers"]
                for n in shards
                for value in row[statistic][str(n)]
            ),
            max_bf16_ratio=largest,
            max_by_shards=by_shards,
            median_bf16_ratio=sorted(values)[len(values) // 2],
            max_by_kernels=by_kernels,
            recommended_bound=bound,
            faults=faults,
            weakest_detection_margin=min(detection),
            all_faults_detected=all(
                entry["detected"] for layouts in faults.values() for entry in layouts.values()
            ),
            fp64_zero_sets_match=all(scan["equals_audio_key_bias_set"] for scan in scans),
            fp64_max_ratio=max(scan["max_ratio"][statistic] for scan in scans),
        )
    return configurations


def calibrate(official, tiny, fixture, recordings, quick=False, log=print):
    started = time.monotonic()
    report = dict(
        margin=MARGIN,
        fp64_zero_threshold=FP64_ZERO,
        shards=list(SHARDS),
        evaluated_shards={name: list(value) for name, value in EVALUATED_SHARDS.items()},
        cases=[],
        fp64=[],
        faults=[],
    )
    for prefix, config, mode, kernels, seed, cases, implementations in groups(
        official, tiny, fixture, recordings, quick
    ):
        for implementation in implementations:
            model = prepared(config, implementation, mode, seed)
            for label, batch in cases:
                norms = gradient_norms(model, implementation, mode, kernels, batch)
                rows = layer_ratios(norms)
                report["cases"].append(
                    dict(
                        label=f"{prefix}/{label}",
                        implementation=implementation,
                        mode=mode,
                        kernels=kernels,
                        layers=rows,
                    )
                )
                largest = {statistic: case_max(rows, statistic) for statistic in STATISTICS}
                log(f"{prefix}/{label} {implementation}: max {largest}")
            del model
    checks = [("official", official, "bf16", [(*fixture, "probe", 0)])] + [
        (
            name,
            config,
            "bf16-master-fp32",
            [(*recordings[0], "probe", 0), (*recordings[1], "ce", 1)],
        )
        for name, config in tiny.items()
    ]
    for name, config, mode, batch in checks:
        for implementation in ("native", "hf"):
            norms = gradient_norms(
                prepared(config, implementation, "fp64", 0), implementation, "fp64", "cpu", batch
            )
            scan = structural_scan(norms)
            rows = layer_ratios(norms)
            report["fp64"].append(
                dict(
                    label=name,
                    implementation=implementation,
                    max_ratio={statistic: case_max(rows, statistic) for statistic in STATISTICS},
                    **scan,
                )
            )
            log(f"fp64 {name} {implementation}: zero set ok={scan['equals_audio_key_bias_set']}")
        model = prepared(config, "native", mode, 0)
        for fault in FAULTS:
            rows = layer_ratios(gradient_norms(model, "native", mode, "emulated", batch, fault))
            report["faults"].append(
                dict(label=name, fault=fault, mode=mode, kernels="emulated", layers=rows)
            )
            smallest = {statistic: case_min(rows, statistic) for statistic in STATISTICS}
            log(f"fault {name} {fault}: min ratio {smallest}")
        del model
    names = ("official", *tiny)
    report["statistics"] = {
        statistic: summarize(report, names, statistic) for statistic in STATISTICS
    }
    # The criterion's scale is the statistic whose weakest gate-level detection is largest.
    report["statistic_choice"] = max(
        STATISTICS,
        key=lambda statistic: min(
            entry["weakest_detection_margin"] for entry in report["statistics"][statistic].values()
        ),
    )
    report["seconds"] = time.monotonic() - started
    return report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True)
    status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).splitlines()
    return dict(
        commit=commit.strip(),
        working_tree_changes=status,
        sources={
            path: sha(REPO / path)
            for path in (
                "tests/acceptance/omni_key_bias_calibration.py",
                "tests/acceptance/omni_structural_zero.py",
                "pithtrain/models/qwen3_omni_audio.py",
            )
        },
        torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        python=platform.python_version(),
        machine=platform.machine(),
        processor=platform.processor(),
        threads=torch.get_num_threads(),
        device="cpu",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new file")
    from tests.acceptance import omni_audio_training
    from tests.acceptance.omni_full_media import official_config, select_fixture
    from tests.acceptance.omni_multimodal_training import tiny_components

    full = official_config(args.official_config, REPO)
    batch, fixture = select_fixture(args.dataset, "audio", full)
    _, samples, batches, staged = omni_audio_training.source_inputs(args.dataset)
    tiny = dict(
        staged=staged.audio_config,
        multimodal=tiny_components(copy.deepcopy(full)).audio_config,
    )
    report = calibrate(
        full.audio_config,
        tiny,
        flatten(batch),
        [flatten(item) for item in batches],
        quick=args.quick,
        log=lambda line: print(line, flush=True),
    )
    report.update(
        environment=environment(),
        official_config_sha256=sha(args.official_config),
        fixture=dict(
            sample_id=fixture["sample_id"],
            bundle_checksums_sha256=fixture["bundle_checksums_sha256"],
            input_shapes=fixture["input_shapes"],
        ),
        recordings=[sample["id"] for sample in samples],
        audio_configs={
            name: {field: getattr(config, field) for field in CALIBRATED_FIELDS}
            for name, config in (("official", full.audio_config), *tiny.items())
        },
        quick=args.quick,
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(choice=report["statistic_choice"], **report["statistics"]), indent=2))
    configurations = report["statistics"][STATISTIC].values()
    return 0 if all(entry["fp64_zero_sets_match"] for entry in configurations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
