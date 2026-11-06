"""Training data configuration shared by token and prepared-bundle readers."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pithtrain.config import SlottedDefault


def validate_modalities(modalities):
    if (
        not isinstance(modalities, (tuple, list))
        or not modalities
        or any(not isinstance(kind, str) for kind in modalities)
        or len(set(modalities)) != len(modalities)
        or set(modalities) - {"text", "image", "audio", "video"}
    ):
        raise ValueError(
            "modalities must be a nonempty, unique selection of text/image/audio/video"
        )


def resolve_bundle_modalities(recipe, modalities):
    """Retain a matching recipe preset's ordering and checkpoint identity.

    An explicit subset need not have a named preset. Give it stable ordering and
    fingerprint its modalities instead of borrowing another stage's identity.
    """
    validate_modalities(modalities)
    matches = [stage for stage, kinds in recipe["stages"].items() if set(kinds) == set(modalities)]
    if len(matches) > 1:
        raise ValueError("Selected modalities match multiple recipe stages")
    if matches:
        stage = matches[0]
        return list(recipe["stages"][stage]), stage
    return sorted(modalities), None


@dataclass(init=False, slots=True)
class DataCfg(SlottedDefault):
    """Source location, storage format and enabled data modalities.

    token_bin reads the existing dense token corpus. prepared_bundle verifies a
    bundle, using its dense text export or its selected media manifests.
    """

    dataset: Path
    format: Literal["token_bin", "prepared_bundle"] = "token_bin"
    modalities: tuple[str, ...] = ("text",)
    sampling_weights: dict[str, float] | None = None
    epoch_samples: int | None = None
    num_workers: int = 0

    def validate(self):
        if not isinstance(getattr(self, "dataset", None), (str, Path)) or not str(self.dataset):
            raise ValueError("data.dataset must name a dataset directory")
        if not isinstance(self.format, str) or self.format not in {"token_bin", "prepared_bundle"}:
            raise ValueError("data.format must be token_bin or prepared_bundle")
        validate_modalities(self.modalities)
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.format == "token_bin" and tuple(self.modalities) != ("text",):
            raise ValueError("token_bin supports only the text modality")
        if tuple(self.modalities) == ("text",):
            if self.sampling_weights is not None or self.epoch_samples is not None:
                raise ValueError(
                    "Dense text follows the existing corpus shuffle, without mixture/epoch overrides"
                )
        if self.sampling_weights is not None and (
            not isinstance(self.sampling_weights, dict)
            or set(self.sampling_weights) != set(self.modalities)
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in self.sampling_weights.values()
            )
        ):
            raise ValueError("Supply positive weights for exactly the enabled modalities")
        if self.epoch_samples is not None and (
            type(self.epoch_samples) is not int or self.epoch_samples <= 0
        ):
            raise ValueError("epoch_samples must be a positive integer")
