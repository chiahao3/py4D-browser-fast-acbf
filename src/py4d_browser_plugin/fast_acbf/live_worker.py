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
    device: str
    mask_path: str
    bf_pixels: int | None = None
    mode: str | None = None
    max_alpha_mrad: float | None = None
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
        pinned_source_tensor=None,
        drift_per_frame: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.cfg = cfg.copy()
        self.runtime_device = choose_device(self.cfg.device)
        data = np.ascontiguousarray(np.asarray(initial_dataset, dtype=np.float32))
        self.solver = _build_solver(self.cfg, data, self.runtime_device)
        self.adapter = MetadataAdapter()
        if initial_metadata is not None:
            self.adapter.diff(initial_metadata)
        self._last_frame_t: float | None = None
        self._pinned_source_tensor = pinned_source_tensor
        self._pinned_source_np = None
        if pinned_source_tensor is not None:
            self._pinned_source_np = pinned_source_tensor.numpy()
            self.solver._dataset_pinned_buffer_4d = pinned_source_tensor
        self._drift_y_per_frame = float(drift_per_frame[0])
        self._drift_x_per_frame = float(drift_per_frame[1])
        self._frame_index = 0

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
        if self._can_use_pinned_source(data, delta):
            if delta:
                self.solver.apply_metadata(delta)
            self._update_from_pinned_source()
        elif delta:
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
        image_np = self._apply_scan_drift(image_np)
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
        bf_mask = getattr(self.solver, "_bf_mask_bool", None)
        bf_pixels = int(np.count_nonzero(bf_mask)) if bf_mask is not None else None
        device_staging = getattr(self.solver, "_dataset_device_staging_4d", None)
        if str(device).startswith("cuda"):
            if self._pinned_source_tensor is not None and device_staging is not None:
                mask_path = "cuda/pinned-source"
            else:
                mask_path = "cuda/device-mask" if device_staging is not None else "cuda/uninitialized"
        else:
            mask_path = "host-mask"
        return image_np, FrameMetrics(
            latency_s=latency,
            fps=fps,
            device=device,
            mask_path=mask_path,
            bf_pixels=bf_pixels,
            mode=self.cfg.mode,
            max_alpha_mrad=float(self.cfg.max_alpha_mrad),
            stage_times=stage_times,
        )

    def _can_use_pinned_source(self, data: np.ndarray, delta: dict[str, Any]) -> bool:
        if self._pinned_source_tensor is None or not str(self.runtime_device).startswith("cuda"):
            return False
        if self._pinned_source_np is None or data is not self._pinned_source_np:
            return False
        heavy_keys = {"wavelength", "max_alpha", "dk", "scan_shape"}
        return not any(key in delta for key in heavy_keys)

    def _update_from_pinned_source(self) -> None:
        import torch
        from fast_acbf.pipeline import build_image_fft

        solver = self.solver
        expected_shape = tuple(self._pinned_source_tensor.shape)
        device_buffer = getattr(solver, "_dataset_device_staging_4d", None)
        if device_buffer is None or tuple(device_buffer.shape) != expected_shape:
            device_buffer = torch.empty(expected_shape, dtype=torch.float32, device=self.runtime_device)
            solver._dataset_device_staging_4d = device_buffer
        device_buffer.copy_(self._pinned_source_tensor, non_blocking=True)
        solver.vbf_images = device_buffer[:, :, solver._bf_mask_bool_d].permute(2, 0, 1).contiguous()
        solver.dataset = self._pinned_source_np
        solver._image_fft = build_image_fft(solver.vbf_images)

    def _apply_scan_drift(self, image: np.ndarray) -> np.ndarray:
        dy = int(round(self._frame_index * self._drift_y_per_frame))
        dx = int(round(self._frame_index * self._drift_x_per_frame))
        self._frame_index += 1
        if dy == 0 and dx == 0:
            return image
        return np.roll(image, shift=(dy, dx), axis=(0, 1))


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
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.cfg = cfg.copy()
        self._initial_dataset = initial_dataset
        self._initial_metadata = initial_metadata
        self._pinned_source_tensor = pinned_source_tensor
        self._drift_per_frame = drift_per_frame
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
                    {
                        "latency_s": metrics.latency_s,
                        "fps": metrics.fps,
                        "device": metrics.device,
                        "mask_path": metrics.mask_path,
                        "bf_pixels": metrics.bf_pixels,
                        "mode": metrics.mode,
                        "max_alpha_mrad": metrics.max_alpha_mrad,
                    },
                )
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            self.engine = None
