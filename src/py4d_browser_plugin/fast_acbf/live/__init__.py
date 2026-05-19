"""Live-acquisition subpackage for the fast-acbf plugin."""

# engine and metadata are pure Python — safe to import without PyQt5
from .engine import FrameMetrics, LiveSolverEngine

# Qt-dependent modules guarded so live.engine stays headless-importable
try:
    from .controller import (
        DEFAULT_GUI_FRAME_INTERVAL_MS,
        LiveSession,
        create_live_session,
        stop_live,
    )
    from .worker import LiveSolverWorker
except ImportError:
    pass

__all__ = [
    "DEFAULT_GUI_FRAME_INTERVAL_MS",
    "FrameMetrics",
    "LiveSession",
    "LiveSolverEngine",
    "LiveSolverWorker",
    "create_live_session",
    "stop_live",
]
