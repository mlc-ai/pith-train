"""Prepared Omni data at the training-task boundary; optional media imports are lazy."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from pithtrain.config import SlottedDefault
from pithtrain.modules.microbatch import Microbatch


@dataclass(init=False, slots=True)
class OmniDataCfg(SlottedDefault):
    stage: str = "text"
    sampling_weights: dict[str, float] | None = None
    epoch_samples: int | None = None
    num_workers: int = 0


def global_target_count(microbatches, group=None):
    """Count labels over one pipeline stage's DP x CP group, without PP duplication."""
    count = sum((mb.objective_inputs[0] != -100).sum() for mb in microbatches)
    if dist.is_initialized():
        dist.all_reduce(count, group=group)
    if count.item() == 0:
        raise ValueError("The global batch has no valid next-token targets")
    return count.to(dtype=torch.float32)


def global_loss_mean(local_loss_sum, target_count, group=None):
    """Reduce loss sums, then divide once; ranks may have unequal or zero targets."""
    total = local_loss_sum.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(total, group=group)
    return total / target_count


class OmniPretrainData:
    """Checkpointable data position shared by all ranks.

    Text uses the existing dense .bin loader and its CP layout. Media uses one
    sample per microbatch so feature tensors never need ambiguous batch slicing.
    Only commit_step advances the durable cursor; prefetch does not count.
    """

    def __init__(self, root, cfg, training_cfg, *, dp_rank=0, dp_size=1, cp_size=1):
        # Keep legacy text training independent of the optional omni-data extra.
        from pithtrain.tasks.prepare_omni_data import verify_bundle

        self.root, self.cfg = Path(root).resolve(), cfg
        self.bundle = verify_bundle(self.root)
        self.recipe = self.bundle["recipe"]
        self.stage = cfg.stage
        if self.stage not in self.recipe["stages"]:
            raise ValueError(f"Unknown Omni data stage: {self.stage}")
        self.modalities = self.recipe["stages"][self.stage]
        if set(self.modalities) - set(self.bundle["manifests"]["train"]):
            raise ValueError("This training stage has not been prepared")
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
        self.is_text = self.modalities == ["text"]
        self.dense = None
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
        self.consumed_samples, self.pending_step = 0, None
        self._iterator = None
        self._processor = self._thinker_config = None
        self.batch_cfg = dict(self.recipe["batch"], max_length=self.sequence_length)
        identity = dict(
            version=1,
            bundle_sha256=hashlib.sha256((self.root / "bundle.json").read_bytes()).hexdigest(),
            stage=self.stage,
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            sequence_length=self.sequence_length,
            seed=self.seed,
            weights=self.weights,
            epoch_samples=self.epoch_samples,
        )
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def validate_model(self, model_class, model_config):
        text_config = getattr(
            getattr(model_config, "thinker_config", model_config), "text_config", model_config
        )
        # The pinned processor config defines the embedding vocabulary, not tokenizer.vocab_size.
        from pithtrain.tasks.prepare_omni_data import processor_for

        self._processor, self._thinker_config = processor_for(self.recipe, True)
        if text_config.vocab_size < self._thinker_config.text_config.vocab_size:
            raise ValueError("The model vocabulary cannot consume the prepared Omni token IDs")
        supported = set(getattr(model_class, "input_modalities", {"text"}))
        if not set(self.modalities) <= supported:
            raise ValueError(
                f"{model_class.__name__} does not implement the requested Omni modalities: {self.modalities}"
            )

    def state_dict(self):
        if self.pending_step is not None:
            raise RuntimeError("Only checkpoint data after the optimizer step has completed")
        return dict(version=1, fingerprint=self.fingerprint, consumed_samples=self.consumed_samples)

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("fingerprint") != self.fingerprint:
            raise ValueError(
                "Checkpoint data recipe/stage/sampling/training layout differs from this run"
            )
        consumed = state.get("consumed_samples")
        if type(consumed) is not int or consumed < 0 or consumed % self.global_batch_size:
            raise ValueError("Invalid checkpoint data consumption position")
        self.consumed_samples, self.pending_step, self._iterator = consumed, None, None

    def begin_step(self, step):
        if self.pending_step is not None or self.consumed_samples != step * self.global_batch_size:
            raise ValueError("Training step and committed data position disagree")
        self.pending_step = step

    def commit_step(self, step):
        if self.pending_step != step:
            raise ValueError("Cannot commit a data step that was not read")
        self.consumed_samples += self.global_batch_size
        self.pending_step = None

    def _make_iterator(self):
        from pithtrain.modules.qwen3_omni_data import create_omni_dataloader
        from pithtrain.tasks.prepare_omni_data import processor_for

        if self._processor is None:
            self._processor, self._thinker_config = processor_for(self.recipe, True)
        loader = create_omni_dataloader(
            self.root,
            self._processor,
            self._thinker_config,
            stage=self.stage,
            split="train",
            batch_size=1,
            num_samples=self.epoch_samples,
            weights=self.weights,
            epoch=self.consumed_samples // self.epoch_samples,
            start_sample=self.consumed_samples % self.epoch_samples,
            rank=self.dp_rank,
            world_size=self.dp_size,
            num_workers=self.cfg.num_workers,
            seed=self.seed,
            batch_cfg=self.batch_cfg,
        )
        with torch.device("cpu"):
            self._iterator = iter(loader)

    def media_microbatches(self, device):
        if self.is_text or self.pending_step is None:
            raise RuntimeError("Media batches must be requested inside a media training step")
        if self._iterator is None:
            self._make_iterator()
        batches = []
        for _ in range(self.global_batch_size // self.dp_size):
            with torch.device("cpu"):
                batch = next(self._iterator)
            inputs = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.model_inputs.items()
            }
            labels = batch.labels.to(device, non_blocking=True)
            # With batch_size=1 there is no batch padding or media-feature slicing.
            batches.append(
                Microbatch(
                    model_inputs=(inputs["input_ids"],),
                    cu_seqlens=None,
                    objective_inputs=(labels,),
                    model_context=inputs,
                    sample_ids=batch.sample_ids,
                )
            )
        # Epochs contain whole global steps. Reset only after yielding the final step.
        if (self.consumed_samples + self.global_batch_size) % self.epoch_samples == 0:
            self._iterator = None
        return batches
