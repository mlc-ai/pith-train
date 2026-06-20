"""PithTrain base classes."""

from dataclasses import MISSING, asdict, fields
from datetime import timedelta
from functools import partial
from pathlib import Path


class SlottedDefault:
    """
    Base class for slotted dataclasses with automatic default initialization.

    Subclasses should be decorated with ``@dataclass(init=False, slots=True)``.
    Calling ``__init__()`` auto-applies every field that declares a default
    value, while leaving required fields (no default) unset.
    """

    __slots__ = ()

    def __init__(self):
        for f in fields(self):
            if f.default is not MISSING:
                setattr(self, f.name, f.default)
            elif f.default_factory is not MISSING:
                setattr(self, f.name, f.default_factory())

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable dict representation of this dataclass."""
        return self._make_json_serializable(asdict(self))

    @staticmethod
    def _make_json_serializable(obj):
        """Recursively convert non-serializable types into JSON primitives."""
        if isinstance(obj, dict):
            return {k: SlottedDefault._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif isinstance(obj, partial):
            # e.g. partial(make_wsd_scheduler, decay_steps=512) -> the call form.
            name = getattr(obj.func, "__name__", repr(obj.func))
            kw = ", ".join(f"{k}={v!r}" for k, v in obj.keywords.items())
            return f"{name}({kw})" if kw else name
        elif callable(obj):
            # Optimizer/scheduler ``make_*`` builders log by name.
            return getattr(obj, "__name__", repr(obj))
        return obj
