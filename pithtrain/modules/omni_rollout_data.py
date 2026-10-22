"""Local, content-bound prompts for the Qwen3-Omni HF rollout provider."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from pithtrain.modules.resume_semantics import implementation_contract
from pithtrain.tasks.omni_rl import _cpu_tree, _weights_digest, fingerprint


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def processor_identity(path, *, max_bytes=64 << 20):
    """Bind local tokenizer/processor assets; never inspect credentials or hub caches."""
    path = Path(path).resolve(strict=True)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "tokenizer.model",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "config.json",
    )
    files = [path / name for name in names if (path / name).is_file()]
    if not files or sum(p.stat().st_size for p in files) > max_bytes:
        raise ValueError("Missing or oversized local processor assets")
    if any(p.is_symlink() for p in files):
        raise ValueError("Processor assets must be ordinary local files")
    return {p.name: file_sha(p) for p in files}


@dataclass(frozen=True)
class Prompt:
    sample_id: str
    inputs: dict[str, torch.Tensor]
    answer: str


class PromptSet:
    """One real prompt per request, complete local reward groups, no implicit cycling.

    The source reader checks raw file hashes again before each request and resume.
    Tensor hashes bind the actual processor output, including every media feature.
    """

    def __init__(self, prompts, *, provenance, files=(), max_tensor_bytes=256 << 20):
        self.prompts = tuple(prompts)
        self.files = tuple((str(Path(p).resolve()), digest) for p, digest in files)
        if not self.prompts or len({p.sample_id for p in self.prompts}) != len(self.prompts):
            raise ValueError("Nonempty prompts with unique IDs required")
        size = 0
        records = []
        for prompt in self.prompts:
            if (
                not isinstance(prompt.sample_id, str)
                or not prompt.sample_id
                or not isinstance(prompt.answer, str)
            ):
                raise ValueError("Each prompt needs an ID and explicit answer string")
            ids = prompt.inputs.get("input_ids")
            if (
                not isinstance(ids, torch.Tensor)
                or ids.dtype != torch.long
                or ids.ndim != 2
                or ids.shape[0] != 1
                or ids.numel() < 1
            ):
                raise ValueError("Each prepared prompt requires one nonempty int64 token row")
            for value in prompt.inputs.values():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.requires_grad
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("Prompt inputs must be finite detached CPU tensors")
                size += value.numel() * value.element_size()
            if size > max_tensor_bytes:
                raise ValueError("Prepared prompt tensor byte limit exceeded")
            mask = prompt.inputs.get("attention_mask")
            if mask is not None and (mask.shape != ids.shape or not (mask == 1).all()):
                raise ValueError("Prepared prompts must be unpadded")
            records.append(
                dict(
                    id=prompt.sample_id,
                    tensors=_weights_digest(prompt.inputs),
                    answer=prompt.answer,
                )
            )
        self.identity = dict(
            schema=1,
            provenance=provenance,
            records=records,
            tensor_bytes=size,
            implementation=implementation_contract(type(self)),
        )
        self.sha256 = fingerprint(self.identity)

    def validate(self):
        for path, digest in self.files:
            if not Path(path).is_file() or file_sha(path) != digest:
                raise ValueError("Prompt, media or processor source changed")
        for prompt, expected in zip(self.prompts, self.identity["records"], strict=True):
            if (
                _weights_digest(prompt.inputs) != expected["tensors"]
                or prompt.answer != expected["answer"]
            ):
                raise ValueError("Prepared prompt tensors/answer changed")

    def payload(self, index):
        self.validate()
        if not 0 <= index < len(self.prompts):
            raise ValueError("Prompt data exhausted; implicit repeat/cycling is disabled")
        return _cpu_tree(self.prompts[index].inputs)


def prepare_prompts(
    manifest,
    processor_path,
    config,
    *,
    scope,
    max_length,
    max_records=1024,
    max_source_bytes=256 << 20,
    media_options=None,
):
    """Read {id,text,media,answer} JSONL using only an explicitly local processor.

    Document-prefix format is intentionally explicit: EOS boundary, media in
    listed order, then the supplied prompt text. No chat template is guessed.
    The existing Omni collator removes the appended target EOS from its inputs.
    """
    from transformers import AutoTokenizer

    manifest = Path(manifest).resolve(strict=True)
    if manifest.stat().st_size > max_source_bytes:
        raise ValueError("Prompt manifest byte limit exceeded")
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 < len(rows) <= max_records:
        raise ValueError("Prompt record limit exceeded")
    required = {"id", "text", "media", "answer"}
    if any(
        set(row) != required
        or not isinstance(row["answer"], str)
        or not isinstance(row["text"], str)
        or not row["text"].strip()
        or not isinstance(row["media"], list)
        for row in rows
    ):
        raise ValueError("Expected exact {id,text,media,answer} prompt schema")
    assets = processor_identity(processor_path)
    files = [(manifest, file_sha(manifest))]
    files.extend((Path(processor_path) / name, digest) for name, digest in assets.items())
    source_bytes = manifest.stat().st_size
    for row in rows:
        for media in row["media"]:
            if (
                not isinstance(media, dict)
                or set(media) != {"type", "path"}
                or media["type"] not in {"image", "audio", "video"}
                or not isinstance(media["path"], str)
            ):
                raise ValueError("Invalid local media descriptor")
            path = (manifest.parent / media["path"]).resolve(strict=True)
            if not path.is_relative_to(manifest.parent) or not path.is_file():
                raise ValueError("Media must be an existing file inside the prompt directory")
            source_bytes += path.stat().st_size
            if source_bytes > max_source_bytes:
                raise ValueError("Prompt/media source byte limit exceeded")
            files.append((path, file_sha(path)))
    prompts = []
    if scope == "text":
        if any(row["media"] for row in rows):
            raise ValueError("Text-only policy cannot silently drop prompt media")
        tokenizer = AutoTokenizer.from_pretrained(
            processor_path, local_files_only=True, trust_remote_code=False
        )
        if tokenizer.eos_token_id is None:
            raise ValueError("Document-prefix prompts require a tokenizer EOS boundary")
        for row in rows:
            ids = [tokenizer.eos_token_id] + tokenizer.encode(row["text"], add_special_tokens=False)
            if not 1 < len(ids) <= max_length:
                raise ValueError("Prompt token length outside bounds; truncation is disabled")
            prompts.append(
                Prompt(
                    row["id"], {"input_ids": torch.tensor([ids], dtype=torch.long)}, row["answer"]
                )
            )
    elif scope == "thinker":
        from transformers import Qwen3OmniMoeProcessor

        from pithtrain.modules.qwen3_omni_data import Qwen3OmniCollator, Qwen3OmniDataset

        processor = Qwen3OmniMoeProcessor.from_pretrained(processor_path, local_files_only=True)
        tokenizer = processor.tokenizer
        collator = Qwen3OmniCollator(
            processor, config, max_length=max_length, **(media_options or {})
        )
        dataset = Qwen3OmniDataset(manifest)
        for index, row in enumerate(rows):
            inputs = collator([dataset[index]]).model_inputs
            prompts.append(Prompt(row["id"], _cpu_tree(inputs), row["answer"]))
    else:
        raise ValueError("Only explicit text or complete Thinker rollout is supported")
    identity = dict(
        format="omni-document-prefix-v1",
        manifest=file_sha(manifest),
        assets=assets,
        media=[digest for _, digest in files[len(assets) + 1 :]],
        media_options=media_options or {},
        model_config=config.to_dict(),
        max_length=max_length,
        implementation=implementation_contract(prepare_prompts),
    )
    result = PromptSet(prompts, provenance=identity, files=files)
    result.validate()
    return result, tokenizer
