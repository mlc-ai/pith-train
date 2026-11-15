"""Official-size Thinker text HF parity on the pinned released weights.

This applies omni_full_size's route-conditioned gate unchanged (fixture, phases, metric
and thresholds) to the released Qwen3-Omni Thinker text decoder. Only the weights differ:
the HF reference and the native model are both filled from the pinned public split-HF
shards, read directly rather than through the PithTrain converter, after the metadata
manifest, every shard's full SHA-256 and every header have been verified. Experts are
assembled as HF's own loader assembles them, and both models must hold identical bytes.

The native phase that replays the reference routes decides. The native free-running
phase, which is the original free-running criterion, and the HF self-control under the
other attention contract are recorded diagnostics that never change the verdict.
--plan is CPU only: it checks the pinned metadata and the parameter mapping without
reading any tensor payload.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from tests.acceptance import omni_full_size as gate
from tests.acceptance.omni_reference_attention import (
    add_reference_attention_argument,
    configure_reference_attention,
    reference_contract,
)
from tests.acceptance.omni_topology_contract import verify_source, write_json

# The pinned released metadata manifest, the same file omni_released_continuation pins.
METADATA_SHA256 = "4753ef18b35604e727ba2f3e3501ab8a8c277233dd892b68a9df4e70e421e886"
DOWNLOAD_STATUS = "DOWNLOADED_FROM_PINNED_PUBLIC_REVISION_NOT_YET_CONVERTED"
RELEASED_SHARDS = 15
RELEASED_TENSORS = 28_010
TEXT_RAW_KEYS = 18_867
TEXT_PREFIXES = ("thinker.model.", "thinker.lm_head.")
HASH_WORKERS = 8
STATUS = "PASS_RELEASED_TEXT_ROUTE_CONDITIONED"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def pinned_headers(metadata, manifest_sha256=METADATA_SHA256):
    """Verify the pinned metadata manifest and return every released tensor's header row."""
    metadata = Path(metadata)
    if file_sha256(metadata / "manifest.json") != manifest_sha256:
        raise ValueError("Released metadata manifest differs from the pinned manifest")
    manifest = read_json(metadata / "manifest.json")
    if manifest["model"] != gate.MODEL_ID or manifest["revision"] != gate.REVISION:
        raise ValueError("Released metadata names another model or revision")
    for row in manifest["files"]:
        path = metadata / row["file"]
        if (
            Path(row["file"]).name != row["file"]
            or path.stat().st_size != row["bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise ValueError(f"Released metadata file differs from its manifest: {row['file']}")
    index = read_json(metadata / "model.safetensors.index.json")["weight_map"]
    shards = sorted(set(index.values()))
    listed = {row["file"] for row in manifest["files"]}
    required = {"config.json", "model.safetensors.index.json"}
    if not required.union(f"{name}.header.json" for name in shards) <= listed:
        raise ValueError("Released metadata manifest does not pin every file the gate reads")
    headers = {}
    for name in shards:
        if Path(name).name != name:
            raise ValueError(f"Unexpected shard path: {name}")
        raw = (metadata / f"{name}.header.json").read_bytes()
        for key, item in json.loads(raw).items():
            if key == "__metadata__":
                continue
            if key in headers or index.get(key) != name:
                raise ValueError(f"Released header and index disagree on {key}")
            start, end = item["data_offsets"]
            headers[key] = dict(
                file=name,
                offset=8 + len(raw) + start,
                bytes=end - start,
                dtype=item["dtype"],
                shape=tuple(item["shape"]),
            )
    if headers.keys() != index.keys():
        raise ValueError("Released headers and index list different tensors")
    return headers


def raw_sources(name, shape):
    """Original split-HF keys, in assembly order, and part shape of one full-size parameter.

    HF's loader concatenates each expert's gate_proj and then up_proj into gate_up_proj,
    whose first half Qwen3OmniMoeThinkerTextExperts uses as the gate, and stacks
    down_proj in ascending expert order. Every other parameter is one tensor under thinker.
    """
    if name.endswith(".mlp.experts.gate_up_proj"):
        if len(shape) != 3 or shape[1] % 2:
            raise ValueError(f"Unexpected fused expert shape: {name}")
        prefix = "thinker." + name.removesuffix("gate_up_proj")
        keys = [
            f"{prefix}{e}.{part}_proj.weight" for e in range(shape[0]) for part in ("gate", "up")
        ]
        return keys, (shape[1] // 2, shape[2])
    if name.endswith(".mlp.experts.down_proj"):
        if len(shape) != 3:
            raise ValueError(f"Unexpected fused expert shape: {name}")
        prefix = "thinker." + name.removesuffix("down_proj")
        return [f"{prefix}{e}.down_proj.weight" for e in range(shape[0])], tuple(shape[1:])
    return ["thinker." + name], tuple(shape)


def release_plan(shapes, headers):
    """Map every full-size parameter to its released parts, checked before any model write."""
    plan, selected = {}, set()
    for name in sorted(shapes):
        keys, part = raw_sources(name, tuple(shapes[name]))
        for key in keys:
            item = headers.get(key)
            if item is None:
                raise ValueError(f"Released checkpoint lacks {key}")
            if key in selected:
                raise ValueError(f"Released tensor mapped twice: {key}")
            if (
                item["dtype"] != "BF16"
                or tuple(item["shape"]) != part
                or item["bytes"] != 2 * math.prod(part)
            ):
                raise ValueError(f"Released tensor has the wrong dtype or shape: {key}")
            selected.add(key)
        plan[name] = keys
    unmapped = sorted(
        key for key in headers if key.startswith(TEXT_PREFIXES) and key not in selected
    )
    if unmapped:
        raise ValueError(f"Released text tensors left unmapped: {unmapped[:8]}")
    return plan


def pinned_text_plan(shapes, headers):
    """release_plan, plus the pinned release's shard, tensor and text split-key counts."""
    if (
        len({row["file"] for row in headers.values()}) != RELEASED_SHARDS
        or len(headers) != RELEASED_TENSORS
    ):
        raise ValueError("Released metadata does not list the pinned shards and tensors")
    plan = release_plan(shapes, headers)
    if sum(len(keys) for keys in plan.values()) != TEXT_RAW_KEYS:
        raise ValueError("Released text split keys differ from the pinned inventory")
    return plan


def verify_download(metadata, source, receipt_sha256, headers):
    """Bind the download directory to the pinned metadata, rehashing every shard in full.

    The same checks as omni_released_continuation's prepare: config and index bytes, the
    download receipt, each shard's size, full SHA-256 and header bytes.
    """
    metadata, source = Path(metadata), Path(source)
    for name in ("config.json", "model.safetensors.index.json"):
        if (source / name).read_bytes() != (metadata / name).read_bytes():
            raise ValueError(f"Downloaded {name} differs from the pinned metadata")
    if file_sha256(source / "download.json") != receipt_sha256:
        raise ValueError("Download receipt differs from the expected receipt")
    download = read_json(source / "download.json")
    if (
        download["status"] != DOWNLOAD_STATUS
        or download["model"] != gate.MODEL_ID
        or download["revision"] != gate.REVISION
        or download["application_retries"] != 0
        or download["private_uploads"] != 0
    ):
        raise ValueError("Download receipt does not describe the pinned public revision")
    files = {row["name"]: row for row in download["files"]}
    shards = sorted({row["file"] for row in headers.values()})
    if len(files) != len(download["files"]) or sorted(files) != shards:
        raise ValueError("Download receipt and released headers list different shards")
    for name in shards:
        path = source / name
        if path.is_symlink() or path.stat().st_size != files[name]["bytes"]:
            raise ValueError(f"Downloaded shard has the wrong type or size: {name}")
    with ThreadPoolExecutor(HASH_WORKERS) as pool:
        digests = list(pool.map(file_sha256, [source / name for name in shards]))
    rows = []
    for name, digest in zip(shards, digests, strict=True):
        if digest != files[name]["observed_sha256"]:
            raise ValueError(f"Downloaded shard SHA-256 differs from its receipt: {name}")
        header = (metadata / f"{name}.header.json").read_bytes()
        with (source / name).open("rb") as stream:
            prefix = stream.read(8 + len(header))
        if prefix != len(header).to_bytes(8, "little") + header:
            raise ValueError(f"Downloaded shard header differs from the pinned header: {name}")
        status = (source / name).stat()
        rows.append(
            dict(
                name=name,
                bytes=status.st_size,
                sha256=digest,
                inode=status.st_ino,
                mtime_ns=status.st_mtime_ns,
            )
        )
    return dict(
        metadata_manifest_sha256=file_sha256(metadata / "manifest.json"),
        download_receipt_sha256=receipt_sha256,
        shards=rows,
        tensors=len(headers),
    )


def unchanged(identity, source):
    """True when no verified shard was replaced or modified after it was hashed."""
    for row in identity["shards"]:
        status = (Path(source) / row["name"]).stat()
        if (status.st_size, status.st_ino, status.st_mtime_ns) != (
            row["bytes"],
            row["inode"],
            row["mtime_ns"],
        ):
            return False
    return True


def tensor_sha256(value):
    """SHA-256 of a tensor's bytes, streamed through host memory in chunks."""
    import torch

    flat, digest = value.detach().reshape(-1), hashlib.sha256()
    for start in range(0, flat.numel(), gate.INIT_CHUNK):
        chunk = flat[start : start + gate.INIT_CHUNK].cpu().contiguous()
        digest.update(chunk.view(torch.uint8).numpy())
    return digest.hexdigest()


def read_exact(stream, offset, target):
    stream.seek(offset)
    filled = 0
    while filled < target.nbytes:
        count = stream.readinto(target[filled:])
        if not count:
            raise AssertionError("Released shard ended before a tensor's last byte")
        filled += count


def load_released(parameters, plan, headers, source, parts=False):
    """Copy the pinned BF16 bytes into every parameter; return receipts of the loaded bytes.

    Each receipt's sha256 is read back from the parameter after the copy. With parts, it
    also lists the source keys in assembly order and the SHA-256 of each source tensor.
    """
    import torch

    receipts = {}
    with ExitStack() as stack, torch.no_grad():
        streams = {}
        for name in sorted(plan):
            parameter, keys = parameters[name], plan[name]
            if parameter.dtype != torch.bfloat16:
                raise AssertionError(f"Released parameter is not BF16: {name}")
            value = torch.empty(tuple(parameter.shape), dtype=torch.bfloat16)
            rows = []
            for slot, key in zip(value.view(len(keys), -1), keys, strict=True):
                item = headers[key]
                if item["file"] not in streams:
                    path = Path(source) / item["file"]
                    streams[item["file"]] = stack.enter_context(path.open("rb", buffering=0))
                target = memoryview(slot.view(torch.uint8).numpy())
                if target.nbytes != item["bytes"]:
                    raise AssertionError(f"Released tensor size differs from its slot: {key}")
                read_exact(streams[item["file"]], item["offset"], target)
                if parts:
                    rows.append(
                        dict(
                            key=key,
                            shape=list(item["shape"]),
                            sha256=hashlib.sha256(target).hexdigest(),
                        )
                    )
            parameter.copy_(value)
            receipts[name] = dict(sha256=tensor_sha256(parameter), raw_keys=len(keys))
            if parts:
                receipts[name]["parts"] = rows
    return receipts


def free_running_summary(result, loss, reference_loss):
    """The original free-running criterion's outcome; a diagnostic, never the verdict."""
    import torch

    rows = result["gradient_parameters"]
    summary = dict(
        verdict_effect="NONE",
        logits_error=result["logits_error"],
        logits_threshold=gate.LOGIT_THRESHOLD,
        loss_close=bool(
            torch.isclose(loss, reference_loss, rtol=gate.LOSS_RTOL, atol=gate.LOSS_ATOL)
        ),
        gradient_pass=sum(row["status"] == "PASS" for row in rows),
        gradient_fail=sum(row["status"] != "PASS" for row in rows),
        route_flips=result["routes"]["flipped_token_layers"],
        token_layers=result["routes"]["token_layers"],
    )
    passed = (
        summary["logits_error"] < gate.LOGIT_THRESHOLD
        and summary["loss_close"]
        and summary["gradient_fail"] == 0
    )
    summary["original_criterion"] = "PASS" if passed else "FAIL"
    return summary


def self_control_agreement(control, native):
    """Whether native free-running reproduces HF under the other attention contract."""
    if control.get("status") == "ERROR":
        return dict(available=False, verdict_effect="NONE")

    def failures(result):
        return sorted(
            row["name"] for row in result["gradient_parameters"] if row["status"] != "PASS"
        )

    return dict(
        available=True,
        verdict_effect="NONE",
        cross_entropy_equal=control["cross_entropy"] == native["cross_entropy"],
        logits_error_equal=control["logits_error"] == native["logits_error"],
        route_flips_equal=(
            control["routes"]["flipped_tokens_per_layer"]
            == native["routes"]["flipped_tokens_per_layer"]
        ),
        failing_gradients_equal=failures(control) == failures(native),
    )


def resource_estimate():
    estimate = gate.resource_estimate()
    estimate.update(
        weight_source="released",
        released_shards_rehashed=RELEASED_SHARDS,
        suggested_outer_timeout_seconds={"route-conditioned": 1800},
        does_not_cover=[item for item in estimate["does_not_cover"] if item != "released weights"],
    )
    return estimate


def execute(args, config_data, shapes, headers, plan, source, repo):
    import torch
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTextConfig,
    )

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "source": source,
        "model": gate.MODEL_ID,
        "revision": gate.REVISION,
        "official_config_sha256": gate.OFFICIAL_CONFIG_SHA256,
        "weight_source": "released",
        "mode": "route-conditioned",
        "verdict_phase": "native-route-replay",
        "reference_attention": {"contract": reference_contract(args)},
        "status": "RUNNING",
        "resource_estimate": resource_estimate(),
        "dtype": "bfloat16",
        "gradients": "autograd BF16; no FP32 main_grad installation",
        "released_identity": "NOT_RUN",
        "dependencies": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "phases": [],
    }

    def save():
        write_json(args.out / "summary.json", report)

    save()
    started = time.monotonic()
    try:
        if (
            torch.__version__ != "2.13.0+cu130"
            or importlib.metadata.version("transformers") != "5.17.0"
        ):
            raise RuntimeError(
                "This gate requires the reviewed torch2.13.0+cu130 / Transformers5.17.0"
            )
        snapshot = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
        (args.out / "gpu.csv").write_text(snapshot)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("The full-size gate requires exactly one visible dedicated H200")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        required = gate.resource_estimate()["bf16_weights_and_gradients_bytes"] + 12 * 2**30
        if "H200" not in properties.name or free < required:
            raise RuntimeError(f"Dedicated H200 has insufficient free memory: {free} < {required}")
        report["hardware"] = {"name": properties.name, "free_bytes": free, "total_bytes": total}
        identity = verify_download(
            args.metadata, args.source, args.download_receipt_sha256, headers
        )
        report["released_identity"] = identity
        save()
        gate.configure_native(device)
        config = Qwen3OmniMoeTextConfig(**config_data)
        generator = torch.Generator(device="cpu").manual_seed(7)
        tokens = torch.randint(
            0, config.vocab_size, (1, gate.SEQUENCE_LENGTH + 1), generator=generator
        )
        inputs, labels = (
            tokens[:, :-1].contiguous().to(device),
            tokens[:, 1:].contiguous().to(device),
        )
        model = gate.build_model(config, False, device)
        report["reference_attention"] = configure_reference_attention(
            model, config, reference_contract(args)
        )
        report["reference_experts_implementation"] = model["model"].config._experts_implementation
        save()
        parameters = gate.parameter_map(model, False)
        gate.check_inventory(parameters, shapes)
        reference_map = load_released(parameters, plan, headers, args.source, parts=True)
        write_json(args.out / "released-parameter-map.json", reference_map)
        result, reference_logits, reference_loss, reference_routes = gate.phase(
            model, False, inputs, labels, shapes
        )
        result["role"] = "reference"
        report["phases"].append(result)
        save()
        if result["status"] != "PASS":
            raise AssertionError("HF full-size forward/backward failed; native phase not started")
        reference_gradients = {name: p.grad.detach().cpu() for name, p in parameters.items()}
        report["phases"].append(
            gate.self_control(
                model,
                config,
                parameters,
                inputs,
                labels,
                shapes,
                (reference_gradients, reference_logits, reference_routes),
                reference_contract(args),
            )
        )
        save()
        del parameters, model
        gc.collect()
        torch.cuda.empty_cache()
        model = gate.build_model(config, True, device)
        parameters = gate.parameter_map(model, True)
        gate.check_inventory(parameters, shapes)
        native_map = load_released(parameters, plan, headers, args.source)
        if {name: row["sha256"] for name, row in native_map.items()} != {
            name: row["sha256"] for name, row in reference_map.items()
        }:
            raise AssertionError("Native and HF released weights differ")
        report["weight_fingerprints"] = native_map
        save()
        result, logits, loss, routes = gate.phase(
            model, True, inputs, labels, shapes, reference_gradients
        )
        result["role"] = "native-free-running"
        gate.compare_phase(result, logits, routes, reference_logits, reference_routes)
        report["phases"].append(result)
        report["original_free_running_criterion"] = free_running_summary(
            result, loss, reference_loss
        )
        report["self_control_agreement"] = self_control_agreement(report["phases"][1], result)
        save()
        for parameter in parameters.values():
            parameter.grad = None
        torch.cuda.empty_cache()
        result, logits, loss, routes = gate.phase(
            model, True, inputs, labels, shapes, reference_gradients, replay=reference_routes
        )
        result["role"] = "native-route-replay"
        gate.compare_phase(result, logits, routes, reference_logits, reference_routes)
        report["phases"].append(result)
        report["parity"] = {
            "route_policy": "replayed-reference-top-k",
            "logits_error": gate.metric(logits, reference_logits),
            "logits_threshold": gate.LOGIT_THRESHOLD,
            "loss_rtol": gate.LOSS_RTOL,
            "loss_atol": gate.LOSS_ATOL,
            "loss_close": bool(
                torch.isclose(loss, reference_loss, rtol=gate.LOSS_RTOL, atol=gate.LOSS_ATOL)
            ),
            "all_initial_weights_identical": True,
            # The replay must actually have produced the reference routes.
            "replayed_route_flips": result["routes"]["flipped_token_layers"],
        }
        save()
        if (
            report["parity"]["logits_error"] >= gate.LOGIT_THRESHOLD
            or not report["parity"]["loss_close"]
            or report["parity"]["replayed_route_flips"] != 0
        ):
            raise AssertionError("Full-size logits, cross entropy or route parity failed")
        if result["status"] != "PASS":
            raise AssertionError("Full-size native gradient verification failed")
        if not unchanged(identity, args.source):
            raise AssertionError("A released shard changed after it was verified")
        report["source_after"] = verify_source(repo, args.expected_commit)
        report["status"] = STATUS
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
    finally:
        if "source_after" not in report:
            try:
                report["source_after"] = verify_source(repo, args.expected_commit)
            except Exception as error:
                report["source_after"] = {"error": repr(error)}
        report["elapsed_seconds"] = time.monotonic() - started
        save()
    return 0 if report["status"] == STATUS else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_attention_argument(parser)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--download-receipt-sha256")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and not (args.source and args.download_receipt_sha256 and args.out):
        parser.error("--execute requires --source, --download-receipt-sha256 and --out")
    repo = Path(__file__).resolve().parents[2]
    source = verify_source(repo, args.expected_commit)
    config, shapes = gate.load_config(args.metadata / "config.json", repo)
    headers = pinned_headers(args.metadata)
    plan = pinned_text_plan(shapes, headers)
    if args.plan:
        print(
            json.dumps(
                {
                    "source": source,
                    "model": gate.MODEL_ID,
                    "revision": gate.REVISION,
                    "weight_source": "released",
                    "mode": "route-conditioned",
                    "status": "PASS_RELEASED_TEXT_METADATA_AND_MAPPING_ONLY",
                    "parameter_keys": len(plan),
                    "raw_keys": sum(len(keys) for keys in plan.values()),
                    "released_tensors": len(headers),
                    "gpu_execution": "NOT_RUN",
                    "estimate": resource_estimate(),
                },
                indent=2,
            )
        )
        return 0
    return execute(args, config, shapes, headers, plan, source, repo)


if __name__ == "__main__":
    raise SystemExit(main())
