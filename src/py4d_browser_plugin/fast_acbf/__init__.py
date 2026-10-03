"""py4D-browser plugin for fast-acbf reconstructions."""

from __future__ import annotations

__version__ = "0.6.0"  # 2026.10.03

# IMPORTANT: `FastAcbfPlugin` MUST be eagerly imported into this module's
# namespace. py4D-browser discovers plugins via `inspect.getmembers(module,
# inspect.isclass)` (see py4D_browser/plugins.py), which only iterates over
# names present in `dir(module)`. A lazy `__getattr__`-based loader would
# silently break plugin discovery — `__all__` is not consulted by `dir()`.
try:
    from .plugin import FastAcbfPlugin
except ModuleNotFoundError as exc:
    if exc.name != "PyQt5":
        raise
    __all__ = []
else:
    __all__ = ["FastAcbfPlugin"]
