"""Fast-fail shutdown when any rank raises.

Default torch.distributed shutdown can hang indefinitely: atexit
destroy_process_group drains in-flight NCCL work that peers will never
satisfy, so the failing rank hangs in atexit, torchrun never sees the
death, and peers spin until something external kills the job.

_abort_process_group has the same drain-deadlock in NCCL 2.28 (despite
docs claiming non-blocking). The only escape is os._exit(1) from a
sys.excepthook -- the kernel reaps FDs/sockets/CUDA IPC, peers' NCCL
ops fail in milliseconds, and torchrun SIGTERMs the rest.
"""

import os
import sys
import threading

import torch  # noqa: F401
from torch.distributed.elastic.multiprocessing.errors import record  # noqa: F401


def set_env_defaults() -> None:
    """NCCL env defaults that bound failure detection. Must run before init."""
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")


def set_heartbeat_timeout(seconds: int) -> None:
    """Set the NCCL watchdog heartbeat timeout. Must run before init_process_group."""
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = str(seconds)


def install_failfast_excepthook() -> None:
    """On uncaught exception: print, flush, os._exit(1) -- no abort/destroy."""
    prev = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        try:
            prev(exc_type, exc_value, exc_tb)
        except Exception:
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # No abort/destroy: both block on the drain we're escaping.
        os._exit(1)

    sys.excepthook = hook
    # Background threads (e.g. logging workers) would otherwise just print and
    # let the main thread continue into the next collective and hang.
    threading.excepthook = lambda args: hook(args.exc_type, args.exc_value, args.exc_traceback)


set_env_defaults()
