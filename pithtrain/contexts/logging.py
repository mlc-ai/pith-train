"""
Logging runtime state, populated once at startup.

Import the module and read fields in-line, not the names: a field does not exist until setup
assigns it, so importing it up front fails and reading it early raises AttributeError.

from pithtrain.contexts import logging
logging.stdout.info("...")
"""

from logging import Logger
from typing import Optional

from wandb.sdk.wandb_run import Run as WandbRun

stdout: Logger
wandb: Optional[WandbRun] = None
