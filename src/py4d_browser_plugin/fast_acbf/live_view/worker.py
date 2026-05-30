"""Worker thread for real py4D-browser Live View datacube updates."""

from __future__ import annotations

import queue
import time
import traceback
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ..config import FastAcbfConfig
from ..live.solver import LiveBFSolver
from ..utils import apply_config_to_solver, build_solver, choose_device
from .output import compute_live_view_outputs


def live_state_signature(config: FastAcbfConfig) -> tuple:
    return (
        round(float(config.rotation_deg), 9),
        bool(config.flipud),
        bool(config.fliplr),
        bool(config.transpose),
        tuple(sorted((str(k), round(float(v), 9)) for k, v in config.aberrations.items())),
    )


class LiveViewWorker(QThread):
    """Own one live solver and process the latest submitted datacube only."""

    frame_ready = pyqtSignal(object)
    started_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    _SENTINEL = object()

    def __init__(self, *, parent=None) -> None:
        super().__init__(parent=parent)
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = False
        self._solver: LiveBFSolver | None = None
        self._signature: tuple | None = None
        self._state_signature: tuple | None = None
        self._runtime_device: str | None = None
        self.rebuild_count = 0

    def submit(self, dataset: np.ndarray, config: FastAcbfConfig) -> None:
        item = (dataset, config.copy())
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(item)

    def stop(self) -> None:
        self._stop = True
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

    def _ensure_solver(
        self, dataset: np.ndarray, config: FastAcbfConfig
    ) -> tuple[LiveBFSolver, bool]:
        runtime_device = choose_device(config.device)
        data = np.ascontiguousarray(np.asarray(dataset, dtype=np.float32))
        signature_cfg = config.copy()
        signature_cfg.device = runtime_device
        signature = signature_cfg.solver_signature(data)
        rebuilt = False
        if self._solver is None or self._signature != signature:
            self._solver = LiveBFSolver(build_solver(config, data, runtime_device))
            self._signature = signature
            self._state_signature = live_state_signature(config)
            self._runtime_device = runtime_device
            self.rebuild_count += 1
            rebuilt = True
        else:
            self._solver.update_dataset(data)
            state_signature = live_state_signature(config)
            if self._state_signature != state_signature:
                apply_config_to_solver(self._solver, config)
                self._state_signature = state_signature
        return self._solver, rebuilt

    def _process_one(self, dataset: np.ndarray, config: FastAcbfConfig) -> dict[str, Any]:
        config.validate_upscale_settings()
        t0 = time.perf_counter()
        solver, rebuilt = self._ensure_solver(dataset, config)
        outputs = compute_live_view_outputs(solver, config)
        t1 = time.perf_counter()
        latency = t1 - t0
        metrics = {
            "latency_s": latency,
            "fps": (1.0 / latency) if latency > 0 else float("inf"),
            "device": self._runtime_device or "?",
            "c10_angstrom": float(config.aberrations.get("C10", 0.0)),
            "max_alpha_mrad": float(config.max_alpha_mrad),
            "rebuilt_solver": rebuilt,
        }
        return {
            "outputs": outputs.images,
            "routes": outputs.routes,
            "config": config.copy(),
            "metrics": metrics,
            "reset": rebuilt,
        }

    def run(self) -> None:
        self.started_ready.emit("idle")
        try:
            while not self._stop:
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                dataset, config = item
                self.frame_ready.emit(self._process_one(dataset, config))
        except Exception:
            self.error.emit(traceback.format_exc())
