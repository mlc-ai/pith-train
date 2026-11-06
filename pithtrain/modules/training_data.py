"""A shared pretraining data boundary; optional media imports remain lazy."""

import math
from pathlib import Path
from typing import Protocol

import torch
import torch.distributed as dist

from pithtrain.modules.data_config import DataCfg, data_fingerprint, resolve_bundle_modalities
from pithtrain.modules.dataset import ConcatDataset, MemmapDataset
from pithtrain.modules.microbatch import Microbatch
from pithtrain.operators.cp_sequence import zigzag_spans

# Qwen processor payloads belong to the encoder; masks and grids remain available
# to decoder position construction. Never broadcast these large feature tensors.
_MEDIA_INPUTS = frozenset({"pixel_values", "pixel_values_videos", "input_features"})


class PretrainData(Protocol):
    """Read a deterministic global batch by training step.

    With the same data/configuration, retries and checkpoint resumes select the
    same samples. Readers may cache prefetched batches, but have no durable cursor.
    """

    data_fingerprint: str
    """Static data/configuration identity for checkpoint validation; never a cursor."""

    def get_batch(self, step: int, device: torch.device) -> list[Microbatch]: ...

    def validate_model(self, model_class, model_config) -> None: ...


class TokenBinData:
    """The existing shuffled token stream, with its DP ordering and zigzag CP reads.

    The training step determines the position, so legacy checkpoints need no
    additional data state. PP and EP ranks never select different samples.
    """

    def __init__(self, root, training_cfg, *, dp_rank=0, dp_size=1, cp_rank=0, cp_size=1):
        self.training_cfg = training_cfg
        self.dp_rank, self.dp_size = dp_rank, dp_size
        self.cp_rank, self.cp_size = cp_rank, cp_size
        root = Path(root).resolve()
        files = sorted(root.rglob("*.bin"))
        if not files:
            raise ValueError(f"No token shards under {root}")
        # Raw token shards have no content manifest. Record their location/inventory
        # without scanning a potentially multi-terabyte corpus on every rank.
        inventory = []
        for file in files:
            stat = file.stat()
            inventory.append((file.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
        self.data_fingerprint = data_fingerprint(
            dict(format="token_bin", root=str(root), files=inventory), training_cfg
        )
        memmaps = [MemmapDataset(file, training_cfg.sequence_length) for file in files]
        self.corpus = ConcatDataset(memmaps, training_cfg.seed)
        required = training_cfg.max_steps * training_cfg.global_batch_size
        assert len(self.corpus) >= required, (
            f"corpus has {len(self.corpus)} samples, run needs {required}"
        )

    def validate_model(self, model_class, model_config) -> None:
        """Token shards carry no tokenizer/vocabulary metadata to compare with the model.

        All supported language models accept text. As with tokenize_corpus, the
        caller must use shards produced with the model's tokenizer.
        """

    def get_batch(self, step: int, device: torch.device) -> list[Microbatch]:
        # short-hands
        micro_batch_size = self.training_cfg.micro_batch_size
        global_batch_size = self.training_cfg.global_batch_size
        dp_size = self.dp_size
        dp_rank = self.dp_rank
        sequence_length = self.training_cfg.sequence_length

        # arithmetic for dataset indices
        effective_batch_size = micro_batch_size * dp_size
        local_batch_size = global_batch_size // dp_size
        start0 = step * global_batch_size + dp_rank * micro_batch_size

        # CP1 spans cover the full sequence and can have unequal lengths.
        front, back = zigzag_spans(self.cp_rank, self.cp_size, sequence_length)
        front_len, back_len = len(front), len(back)
        local_seq_len = front_len + back_len

        # single allocation on host, then one HtoD transfer per tensor
        local_tokens = torch.empty((local_batch_size, local_seq_len), dtype=torch.long)
        local_labels = torch.empty((local_batch_size, local_seq_len), dtype=torch.long)

        # fill in one pass: k iterates over our rank-local batch rows. Each sample
        # is two memmap reads (front block + back block) followed by an in-place
        # concat into the pre-allocated host buffer.
        for k in range(local_batch_size):
            acc, off = divmod(k, micro_batch_size)
            index = start0 + acc * effective_batch_size + off
            tokens_a, labels_a = self.corpus.get_chunk(index, front.start, front_len)
            tokens_b, labels_b = self.corpus.get_chunk(index, back.start, back_len)
            local_tokens[k, :front_len] = tokens_a
            local_tokens[k, front_len:] = tokens_b
            local_labels[k, :front_len] = labels_a
            local_labels[k, front_len:] = labels_b

        local_tokens = local_tokens.to(device, non_blocking=True)
        local_labels = local_labels.to(device, non_blocking=True)

        # Rows are already micro-batch major, so a plain split reproduces the partitioning the pipeline
        # applies itself: rows [i * mbs, (i + 1) * mbs) belong to micro-batch i.
        return [
            Microbatch(
                model_inputs=(local_tokens[i : i + micro_batch_size],),
                cu_seqlens=None,
                objective_inputs=(local_labels[i : i + micro_batch_size],),
            )
            for i in range(0, local_batch_size, micro_batch_size)
        ]


class PreparedBundleData:
    """Step-indexed reader for a verified prepared bundle, including text-only bundles.

    Text uses the existing dense .bin loader and its CP layout. Media uses one
    sample per microbatch so feature tensors never need ambiguous batch slicing.
    Only PP rank 0 reads/processes media; its peers receive tokens, labels and
    position/layout tensors. Large encoder payloads never cross the PP group.
    The media sampler maps seed/epoch/global sample position to records. A cached
    iterator only avoids repeating setup during consecutive steps; seeking or
    retrying a step rebuilds it from step * global_batch_size.
    """

    def __init__(
        self,
        cfg: DataCfg,
        training_cfg,
        *,
        dp_rank=0,
        dp_size=1,
        cp_rank=0,
        cp_size=1,
        pp_rank=0,
        pp_size=1,
        pp_group=None,
    ):
        cfg.validate()
        if cfg.format != "prepared_bundle":
            raise ValueError("PreparedBundleData requires prepared_bundle format")
        # Keep legacy text training independent of the optional omni-data extra.
        from pithtrain.modules.omni_bundle import read_verified_bundle

        self.root, self.cfg = Path(cfg.dataset).resolve(), cfg
        self.is_text = list(cfg.modalities) == ["text"]
        self.pp_rank, self.pp_size, self.pp_group = pp_rank, pp_size, pp_group
        if (
            type(pp_size) is not int
            or pp_size < 1
            or type(pp_rank) is not int
            or not 0 <= pp_rank < pp_size
        ):
            raise ValueError("Invalid pipeline-parallel rank/size")
        if not self.is_text and pp_size > 1:
            if pp_group is None or not dist.is_initialized():
                raise ValueError("Media PP requires an initialized pipeline process group")
            if dist.get_world_size(pp_group) != pp_size or dist.get_rank(pp_group) != pp_rank:
                raise ValueError("Pipeline process group does not match the data rank/size")

        self.bundle, bundle_sha256 = self._share_metadata(lambda: read_verified_bundle(self.root))
        self.recipe = self.bundle["recipe"]
        self.modalities, self.stage = resolve_bundle_modalities(self.recipe, cfg.modalities)
        if set(self.modalities) - set(self.bundle["manifests"]["train"]):
            raise ValueError("The selected modalities have not been prepared")
        self.global_batch_size = training_cfg.global_batch_size
        self.micro_batch_size = training_cfg.micro_batch_size
        self.sequence_length = training_cfg.sequence_length
        self.seed = training_cfg.seed
        self.dp_rank, self.dp_size = dp_rank, dp_size
        if self.global_batch_size <= 0 or self.micro_batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("Training sequence/batch sizes must be positive")
        if dp_size < 1 or not 0 <= dp_rank < dp_size:
            raise ValueError("Invalid data-parallel rank/size")
        if self.global_batch_size % (dp_size * self.micro_batch_size):
            raise ValueError("Global batch must divide into whole data-rank microbatches")
        if type(cfg.num_workers) is not int or cfg.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        self.weights = (
            cfg.sampling_weights
            if cfg.sampling_weights is not None
            else {kind: self.recipe["sampling_weights"][kind] for kind in self.modalities}
        )
        if set(self.weights) != set(self.modalities) or any(
            not math.isfinite(value) or value <= 0 for value in self.weights.values()
        ):
            raise ValueError("Supply positive weights for exactly the enabled modalities")
        if self.is_text:
            if cfg.sampling_weights is not None or cfg.epoch_samples is not None:
                raise ValueError(
                    "Dense text follows the existing corpus shuffle, without mixture/epoch overrides"
                )
        elif self.micro_batch_size != 1 or cp_size != 1:
            raise ValueError("Omni media currently requires micro_batch_size=1 and CP=1")
        samples = sum(
            self.bundle["statistics"]["train"][kind]["samples"] for kind in self.modalities
        )
        self.epoch_samples = (
            cfg.epoch_samples
            if cfg.epoch_samples is not None
            else math.ceil(samples / self.global_batch_size) * self.global_batch_size
        )
        if (
            type(self.epoch_samples) is not int
            or self.epoch_samples <= 0
            or self.epoch_samples % self.global_batch_size
        ):
            raise ValueError("epoch_samples must be a positive multiple of global_batch_size")
        self._iterator, self._next_step = None, None
        self._processor = self._thinker_config = None
        self.batch_cfg = dict(self.recipe["batch"], max_length=self.sequence_length)
        identity = dict(
            bundle_sha256=bundle_sha256,
            stage=self.stage,
            weights=self.weights,
            epoch_samples=self.epoch_samples,
        )
        if self.stage is None:
            identity["modalities"] = self.modalities
        self.data_fingerprint = data_fingerprint(identity, training_cfg)
        self._dense = (
            TokenBinData(
                self.root / "tokens/train",
                training_cfg,
                dp_rank=dp_rank,
                dp_size=dp_size,
                cp_rank=cp_rank,
                cp_size=cp_size,
            )
            if self.is_text
            else None
        )

    def get_batch(self, step: int, device: torch.device) -> list[Microbatch]:
        if self._dense is not None:
            return self._dense.get_batch(step, device)
        return self._media_microbatches(step, device)

    def validate_model(self, model_class, model_config):
        text_config = getattr(
            getattr(model_config, "thinker_config", model_config), "text_config", model_config
        )
        # The pinned processor config defines the embedding vocabulary, not tokenizer.vocab_size.
        from pithtrain.modules.omni_bundle import processor_for

        def prepare():
            self._processor, self._thinker_config = processor_for(self.recipe, True)
            return self._thinker_config.text_config.vocab_size

        required_vocab = self._share_metadata(prepare)
        if text_config.vocab_size < required_vocab:
            raise ValueError("The model vocabulary cannot consume the prepared Omni token IDs")
        supported = set(getattr(model_class, "input_modalities", {"text"}))
        if not set(self.modalities) <= supported:
            raise ValueError(
                f"{model_class.__name__} does not implement the requested Omni modalities: {self.modalities}"
            )

    def _share_metadata(self, read):
        """Read on PP rank 0; share only small metadata or an explicit read failure."""
        if self.is_text or self.pp_size == 1:
            return read()
        message = [None]
        failure = None
        if self.pp_rank == 0:
            try:
                message[0] = (read(), None)
            except Exception as error:
                failure = error
                message[0] = (None, f"{type(error).__name__}: {error}")
        source = dist.get_global_rank(self.pp_group, 0)
        dist.broadcast_object_list(message, src=source, group=self.pp_group)
        result, error = message[0]
        if error is not None:
            raise RuntimeError(f"Pipeline media reader failed: {error}") from failure
        return result

    def _make_iterator(self, step):
        from pithtrain.modules.omni_bundle import processor_for
        from pithtrain.modules.qwen3_omni_data import create_omni_dataloader

        if self._processor is None:
            self._processor, self._thinker_config = processor_for(self.recipe, True)
        position = step * self.global_batch_size
        loader = create_omni_dataloader(
            self.root,
            self._processor,
            self._thinker_config,
            modalities=self.modalities,
            split="train",
            batch_size=1,
            num_samples=self.epoch_samples,
            weights=self.weights,
            epoch=position // self.epoch_samples,
            start_sample=position % self.epoch_samples,
            rank=self.dp_rank,
            world_size=self.dp_size,
            num_workers=self.cfg.num_workers,
            seed=self.seed,
            batch_cfg=self.batch_cfg,
        )
        with torch.device("cpu"):
            self._iterator = iter(loader)

    def _media_microbatches(self, step, device):
        if step != self._next_step:
            self._iterator = None
        # A partial read or processing failure must also rebuild on retry.
        self._next_step = None
        batches = [
            self._next_media_microbatch(step, device)
            for _ in range(self.global_batch_size // self.dp_size)
        ]
        self._next_step = step + 1
        # Epochs contain whole global steps. Reset only after yielding the final step.
        if (step + 1) * self.global_batch_size % self.epoch_samples == 0:
            self._iterator = None
        return batches

    def _next_media_microbatch(self, step, device):
        shared, media_inputs = {}, {}

        def read():
            if self._iterator is None:
                self._make_iterator(step)
            with torch.device("cpu"):
                batch = next(self._iterator)
            for name, value in batch.model_inputs.items():
                destination = media_inputs if name in _MEDIA_INPUTS else shared
                destination[name] = value.to(device, non_blocking=True).contiguous()
            shared["labels"] = batch.labels.to(device, non_blocking=True).contiguous()
            return batch.sample_ids, [
                (name, tuple(value.shape), value.dtype) for name, value in shared.items()
            ]

        sample_ids, specs = self._share_metadata(read)
        if self.pp_size > 1:
            source = dist.get_global_rank(self.pp_group, 0)
            for name, shape, dtype in specs:
                if self.pp_rank != 0:
                    shared[name] = torch.empty(shape, dtype=dtype, device=device)
                dist.broadcast(shared[name], src=source, group=self.pp_group)
        labels = shared.pop("labels")
        return Microbatch(
            model_inputs=(shared["input_ids"],),
            cu_seqlens=None,
            objective_inputs=(labels,),
            model_context=shared,
            media_inputs=media_inputs or None,
            sample_ids=sample_ids,
        )
