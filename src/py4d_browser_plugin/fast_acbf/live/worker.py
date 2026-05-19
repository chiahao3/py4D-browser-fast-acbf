"""Qt wrapper around :class:`live.engine.LiveSolverEngine`.

The non-Qt engine lives in :mod:`live.engine` so it can be imported
without PyQt5 installed. ``LiveSolverEngine`` and ``FrameMetrics`` are
re-exported from this module for backwards compatibility with existing
callers; new code should import them from :mod:`live.engine` directly.
"""

from __future__ import annotations

import queue
import traceback
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ..config import FastAcbfConfig
from .engine import FrameMetrics, LiveSolverEngine, _cuda_sync

__all__ = ["FrameMetrics", "LiveSolverEngine", "LiveSolverWorker", "_cuda_sync"]


class LiveSolverWorker(QThread):
    """QThread wrapping ``LiveSolverEngine`` with a drop-oldest job queue."""

    frame_ready = pyqtSignal(np.ndarray, dict)
    started_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    _SENTINEL = object()

    def __init__(
        self,
        *,
        cfg: FastAcbfConfig,
        initial_dataset: np.ndarray,
        initial_metadata: dict[str, Any] | None = None,
        pinned_source_tensor=None,
        drift_per_frame: tuple[float, float] = (0.0, 0.0),
        display_noise_sigma_pct: float = 0.0,
        noise_seed: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.cfg = cfg.copy()
        self._initial_dataset = initial_dataset
        self._initial_metadata = initial_metadata
        self._pinned_source_tensor = pinned_source_tensor
        self._drift_per_frame = drift_per_frame
        self._display_noise_sigma_pct = display_noise_sigma_pct
        self._noise_seed = noise_seed
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = False
        self.engine: LiveSolverEngine | None = None

    def submit(self, dataset: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Enqueue a frame, dropping the queued frame if the slot is full."""
        try:
            self._queue.put_nowait((dataset, metadata))
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait((dataset, metadata))

    def stop(self) -> None:
        self._stop = True
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            # The queue was refilled concurrently; the timeout-free worker loop
            # will still observe _stop after the current frame.
            pass

    def run(self) -> None:
        try:
            self.engine = LiveSolverEngine(
                self.cfg,
                self._initial_dataset,
                self._initial_metadata,
                pinned_source_tensor=self._pinned_source_tensor,
                drift_per_frame=self._drift_per_frame,
                display_noise_sigma_pct=self._display_noise_sigma_pct,
                noise_seed=self._noise_seed,
            )
            self.started_ready.emit(self.engine.device)
            while not self._stop:
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                dataset, metadata = item
                image, metrics = self.engine.process_one(dataset, metadata, profile=True)
                self.frame_ready.emit(
                    image,
                    {
                        "latency_s": metrics.latency_s,
                        "fps": metrics.fps,
                        "device": metrics.device,
                        "mask_path": metrics.mask_path,
                        "bf_pixels": metrics.bf_pixels,
                        "mode": metrics.mode,
                        "max_alpha_mrad": metrics.max_alpha_mrad,
                        "display_noise_sigma_pct": metrics.display_noise_sigma_pct,
                        "stage_times": metrics.stage_times or {},
                    },
                )
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.engine = None
