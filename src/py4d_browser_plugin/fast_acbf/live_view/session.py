"""Qt lifecycle wrapper for real py4D-browser Live View."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from ..config import FastAcbfConfig
from .worker import LiveViewWorker


class LiveViewSession(QObject):
    frame_ready = pyqtSignal(object)
    started_ready = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, *, parent=None) -> None:
        super().__init__(parent=parent)
        self.worker = LiveViewWorker(parent=parent)
        self.worker.frame_ready.connect(self.frame_ready)
        self.worker.started_ready.connect(self.started_ready)
        self.worker.error.connect(self.error)
        self.worker.finished.connect(self.finished)

    def start(self) -> None:
        if not self.worker.isRunning():
            self.worker.start()

    def submit(self, datacube_data: np.ndarray, config: FastAcbfConfig) -> None:
        self.worker.submit(datacube_data, config)

    def set_auto_refinement(self, *, focus: bool, aberrations: bool) -> None:
        self.worker.set_auto_refinement(focus=focus, aberrations=aberrations)

    def stop(self, timeout_ms: int = 2000) -> None:
        self.worker.stop()
        if self.worker.isRunning():
            self.worker.wait(timeout_ms)


def stop_live_view(session: LiveViewSession | None, timeout_ms: int = 2000) -> None:
    if session is not None:
        session.stop(timeout_ms=timeout_ms)
