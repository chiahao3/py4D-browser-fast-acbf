"""Real py4D-browser Live View integration for fast-acbf.

Also hosts the headless live-acquisition engine (``engine.py``,
``metadata.py``, ``solver.py``) that ``LiveViewWorker`` builds on top of.
"""

# engine, metadata, output, and solver are pure Python — safe to import
# without PyQt5 (see tests/test_live_engine.py's headless-import test).
from .output import (
    LIVE_OUTPUT_NONE,
    LiveViewOutputs,
    compute_live_view_outputs,
    live_output_title,
    normalize_live_output,
)

# Qt-dependent modules guarded so live_view.engine/.metadata/.solver stay
# headless-importable.
try:
    from .dock import LiveViewDock
    from .session import LiveViewSession, stop_live_view
    from .worker import LiveViewWorker
except ImportError:
    pass

__all__ = [
    "LIVE_OUTPUT_NONE",
    "LiveViewDock",
    "LiveViewOutputs",
    "LiveViewSession",
    "LiveViewWorker",
    "compute_live_view_outputs",
    "live_output_title",
    "normalize_live_output",
    "stop_live_view",
]
