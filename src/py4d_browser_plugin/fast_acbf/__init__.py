"""py4D-browser plugin for fast-acbf reconstructions."""

from __future__ import annotations

__version__ = "0.1.1" # 2026.05.11

try:
    from .plugin import FastAcbfPlugin
except ModuleNotFoundError as exc:
    if exc.name != "PyQt5":
        raise
    __all__ = []
else:
    __all__ = ["FastAcbfPlugin"]
