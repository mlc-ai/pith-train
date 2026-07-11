"""Logging runtime state."""

from logging import Logger
from typing import Optional

from wandb.sdk.wandb_run import Run as WandbRun

stdout: Logger
wandb: Optional[WandbRun] = None
