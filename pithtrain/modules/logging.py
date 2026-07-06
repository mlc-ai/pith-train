"""PithTrain logging module."""

import os
import sys
from dataclasses import asdict, dataclass
from logging import INFO, Formatter, Logger, StreamHandler
from typing import Optional

import wandb

from pithtrain.config import SlottedDefault
from pithtrain.contexts import logging


class StdoutLogger(Logger):
    """Logger that prints to standard output."""

    def __init__(self, name: str, level: int = 0):
        super().__init__(name, level)
        handler = StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = Formatter(fmt, datefmt)
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def info(self, *args, rank: int = 0, **kwargs):
        """
        Log an info message only if the current process rank matches the specified rank.
        This is useful in distributed settings to avoid duplicate logs.

        Parameters
        ----------
        rank : int
            The rank of the process that should log the message. Defaults to 0. If "RANK"
            is not in the environment, all ranks will log. If rank is negative, all ranks
            will log.
        """
        if "RANK" not in os.environ or rank < 0 or int(os.environ["RANK"]) == rank:
            super().info(*args, **kwargs)


@dataclass(init=False, slots=True)
class LoggingWandbCfg(SlottedDefault):
    """Configuration for logging with Weights & Biases."""

    entity: str
    """The username or team name the runs are logged to."""

    project: str
    """The name of the project under which this run will be logged."""

    name: str
    """A short display name for this run, which appears in the UI to help you identify it."""

    group: Optional[str] = None
    """A group name to organize related runs together."""


@dataclass(init=False, slots=True)
class LoggingCfg(SlottedDefault):
    """Configuration for logging."""

    wandb: Optional[LoggingWandbCfg] = None
    """Configuration for logging with Weights & Biases."""


def setup_stdout() -> None:
    """Setup the stdout logger."""
    logging.stdout = StdoutLogger("pithtrain", INFO)


def setup_wandb(cfg: LoggingCfg) -> None:
    """Setup the WandB run."""
    if logging.wandb is not None:
        return
    if cfg.wandb is None:
        return
    if os.environ.get("RANK", "0") != "0":
        return
    kwargs = asdict(cfg.wandb)
    kwargs["resume"] = "allow"
    kwargs["dir"] = os.environ.get("WANDB_DIR", "/tmp/wandb")
    logging.wandb = wandb.init(**kwargs)
    # Define the metrics for monitoring.
    logging.wandb.define_metric("train/step", hidden=True)
    logging.wandb.define_metric("train/cross-entropy-loss", step_metric="train/step")
    logging.wandb.define_metric("train/load-balance-loss", step_metric="train/step")
    logging.wandb.define_metric("train/learning-rate", step_metric="train/step")
    logging.wandb.define_metric("train/gradient-norm", step_metric="train/step")
    logging.wandb.define_metric("infra/tokens-per-second", step_metric="train/step")
    logging.wandb.define_metric("infra/peak-gpu-memory", step_metric="train/step")


def activate_wandb(cfg: object) -> None:
    """
    Lazily initialize the WandB run and upload config.

    Intended to be called from the training loop right before the first
    ``wandb.log()``, so that a run is only created after a training step fully
    succeeds.  The double-init guard in ``setup_wandb`` makes repeated calls a
    no-op.
    """
    assert hasattr(cfg, "logging") and isinstance(cfg.logging, LoggingCfg)
    setup_wandb(cfg.logging)
    if logging.wandb is not None:
        config = {}
        for section in ("distributed", "training"):
            if hasattr(cfg, section):
                config[section] = getattr(cfg, section).to_json_dict()
        logging.wandb.config.update(config)


def setup_logging(cfg: object) -> None:
    """Initialize the logging runtime: the stdout logger."""
    assert hasattr(cfg, "logging") and isinstance(cfg.logging, LoggingCfg)
    setup_stdout()
