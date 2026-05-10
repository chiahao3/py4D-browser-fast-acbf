"""Long-lived live-acquisition worker for fast-acbf.

``LiveSolverEngine`` is the non-Qt core: it owns one ``BFSolver`` plus a
``MetadataAdapter`` for its lifetime and exposes ``process_one(dataset,
metadata)`` for headless callers (benchmark scripts, integration tests).

``LiveSolverWorker(QThread)`` wraps the engine with a drop-oldest single-slot
queue and PyQt signals for GUI use.
"""

from __future__ import annotations

import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .config import FastAcbfConfig
from .metadata import MetadataAdapter
from .worker import choose_device, tensor_to_numpy


def _build_solver(cfg: FastAcbfConfig, data: np.ndarray, runtime_device: str):
    from fast_acbf.solver import BFSolver

    return BFSolver(
        dataset=data,
        max_alpha=float(cfg.max_alpha_mrad),
        scan_step_size=float(cfg.scan_step_angstrom),
        dk=float(cfg.dk_inv_angstrom),
        wavelength=float(cfg.wavelength_angstrom),
        max_order=int(cfg.max_order),
        aberrations=cfg.aberration_dict(),
        device=runtime_device,
        coord_transform=cfg.coord_transform(),
        eps=float(cfg.eps),
        cache_mode=str(cfg.cache_mode),
    )


@dataclass
class FrameMetrics:
    latency_s: float
    fps: float
    stage_times: dict[str, float] | None = None


def _cuda_sync(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


class LiveSolverEngine:
    """Holds one BFSolver + MetadataAdapter and reconstructs frames in place."""

    def __init__(
        self,
        cfg: FastAcbfConfig,
        initial_dataset: np.ndarray,
        initial_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg.copy()
        self.runtime_device = choose_device(self.cfg.device)
        data = np.ascontiguousarray(np.asarray(initial_dataset, dtype=np.float32))
        self.solver = _build_solver(self.cfg, data, self.runtime_device)
        self.adapter = MetadataAdapter()
        if initial_metadata is not None:
            self.adapter.diff(initial_metadata)
        self._last_frame_t: float | None = None

    @property
    def device(self) -> str:
        return self.runtime_device

    def process_one(
        self,
        dataset: np.ndarray,
        metadata: dict[str, Any] | None = None,
        *,
        profile: bool = False,
    ) -> tuple[np.ndarray, FrameMetrics]:
        device = self.runtime_device
        t0 = time.perf_counter()
        data = np.ascontiguousarray(np.asarray(dataset, dtype=np.float32))
        delta = self.adapter.diff(metadata or {})
        t_prep = time.perf_counter()
        if delta:
            self.solver.apply_metadata(delta, dataset=data)
        else:
            self.solver.update_dataset(data)
        if profile:
            _cuda_sync(device)
        t_apply = time.perf_counter()
        recon_kwargs = self.cfg.reconstruct_kwargs()
        image = self.solver.get_reconstructed_image(
            mode=self.cfg.mode,
            frame=self.cfg.output_frame,
            **recon_kwargs,
        )
        if profile:
            _cuda_sync(device)
        t_recon = time.perf_counter()
        image_np = tensor_to_numpy(image)
        t1 = time.perf_counter()
        latency = t1 - t0
        if self._last_frame_t is None:
            fps = (1.0 / latency) if latency > 0 else float("inf")
        else:
            dt = t1 - self._last_frame_t
            fps = (1.0 / dt) if dt > 0 else float("inf")
        self._last_frame_t = t1
        stage_times: dict[str, float] | None = None
        if profile:
            stage_times = {
                "prep": t_prep - t0,
                "apply_metadata_or_update_dataset": t_apply - t_prep,
                "get_reconstructed_image": t_recon - t_apply,
                "tensor_to_numpy": t1 - t_recon,
            }
        return image_np, FrameMetrics(latency_s=latency, fps=fps, stage_times=stage_times)


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
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.cfg = cfg.copy()
        self._initial_dataset = initial_dataset
        self._initial_metadata = initial_metadata
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
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            self.engine = LiveSolverEngine(
                self.cfg, self._initial_dataset, self._initial_metadata
            )
            self.started_ready.emit(self.engine.device)
            while not self._stop:
                item = self._queue.get()
                if item is self._SENTINEL:
                    break
                dataset, metadata = item
                image, metrics = self.engine.process_one(dataset, metadata)
                self.frame_ready.emit(
                    image,
                    {"latency_s": metrics.latency_s, "fps": metrics.fps},
                )
        except Exception:
            self.error.emit(traceback.format_exc())
