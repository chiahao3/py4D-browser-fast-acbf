"""Worker thread for real py4D-browser Live View datacube updates."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ..config import FastAcbfConfig
from ..utils import apply_config_to_solver, build_solver, choose_device, sync_config_from_solver
from .output import compute_live_view_outputs
from .solver import LiveBFSolver


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
        self._auto_lock = threading.Lock()
        self._auto_focus_enabled = False
        self._auto_aberrations_enabled = False
        self._auto_focus_due = False
        self._auto_aberrations_due = False
        self._auto_focus_last_done_s: float | None = None
        self._auto_aberrations_last_done_s: float | None = None
        self.rebuild_count = 0

    def set_auto_refinement(self, *, focus: bool, aberrations: bool) -> None:
        with self._auto_lock:
            focus = bool(focus)
            aberrations = bool(aberrations)
            if focus and not self._auto_focus_enabled:
                self._auto_focus_due = True
            if aberrations and not self._auto_aberrations_enabled:
                self._auto_aberrations_due = True
            self._auto_focus_enabled = focus
            self._auto_aberrations_enabled = aberrations
            if not focus:
                self._auto_focus_due = False
            if not aberrations:
                self._auto_aberrations_due = False

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
        config, auto_metrics = self._run_due_auto_refinements(solver, config)
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
        metrics.update(auto_metrics)
        return {
            "outputs": outputs.images,
            "routes": outputs.routes,
            "config": config.copy(),
            "metrics": metrics,
            "reset": rebuilt,
        }

    def _run_due_auto_refinements(
        self, solver: LiveBFSolver, config: FastAcbfConfig
    ) -> tuple[FastAcbfConfig, dict[str, Any]]:
        cfg = config.copy()
        metrics: dict[str, Any] = {
            "auto_focus_enabled": False,
            "auto_aberrations_enabled": False,
            "auto_refinement_messages": [],
        }
        now = time.monotonic()
        with self._auto_lock:
            focus_due = self._auto_focus_enabled and (
                self._auto_focus_due
                or self._auto_focus_last_done_s is None
                or now - self._auto_focus_last_done_s >= float(cfg.live_auto_focus_interval_s)
            )
            aberrations_due = self._auto_aberrations_enabled and (
                self._auto_aberrations_due
                or self._auto_aberrations_last_done_s is None
                or now - self._auto_aberrations_last_done_s
                >= float(cfg.live_auto_aberrations_interval_s)
            )
            metrics["auto_focus_enabled"] = self._auto_focus_enabled
            metrics["auto_aberrations_enabled"] = self._auto_aberrations_enabled
            self._auto_focus_due = False
            self._auto_aberrations_due = False

        if focus_due:
            cfg = self._run_auto_focus(solver, cfg, metrics)
        if aberrations_due:
            cfg = self._run_auto_aberrations(solver, cfg, metrics)
        if focus_due or aberrations_due:
            self._state_signature = live_state_signature(cfg)
        return cfg, metrics

    def _run_auto_focus(
        self, solver: LiveBFSolver, config: FastAcbfConfig, metrics: dict[str, Any]
    ) -> FastAcbfConfig:
        messages = metrics["auto_refinement_messages"]
        messages.append("Auto Focus running...")
        t0 = time.perf_counter()
        try:
            solver.refine_defocus(
                search_range=config.defocus_search_range(),
                num_points=int(config.defocus_points),
                metric=config.metric,
                plot_search=False,
                mode=config.refinement_mode,
                search_halfwidth=config.defocus_search_halfwidth_angstrom,
                defocus_range_tolerance_factor=float(config.defocus_range_tolerance_factor),
                **config.reconstruct_kwargs(),
            )
        except Exception as exc:
            with self._auto_lock:
                self._auto_focus_enabled = False
                self._auto_focus_due = False
            metrics["auto_focus_enabled"] = False
            metrics["auto_focus_error"] = str(exc)
            messages.append(f"Auto Focus failed: {exc}")
            return config
        elapsed = time.perf_counter() - t0
        with self._auto_lock:
            self._auto_focus_last_done_s = time.monotonic()
        cfg = sync_config_from_solver(config, solver)
        metrics["auto_focus_duration_s"] = elapsed
        metrics["auto_focus_updated"] = True
        messages.append("Auto Focus updated")
        return cfg

    def _run_auto_aberrations(
        self, solver: LiveBFSolver, config: FastAcbfConfig, metrics: dict[str, Any]
    ) -> FastAcbfConfig:
        messages = metrics["auto_refinement_messages"]
        messages.append("Auto Aberrations running...")
        t0 = time.perf_counter()
        try:
            solver.refine_aberrations(
                lr=float(config.aberration_lr),
                iters=int(config.aberration_iters),
                metric=config.metric,
                mode=config.refinement_mode,
                **config.reconstruct_kwargs(),
            )
        except Exception as exc:
            with self._auto_lock:
                self._auto_aberrations_enabled = False
                self._auto_aberrations_due = False
            metrics["auto_aberrations_enabled"] = False
            metrics["auto_aberrations_error"] = str(exc)
            messages.append(f"Auto Aberrations failed: {exc}")
            return config
        elapsed = time.perf_counter() - t0
        with self._auto_lock:
            self._auto_aberrations_last_done_s = time.monotonic()
        cfg = sync_config_from_solver(config, solver)
        metrics["auto_aberrations_duration_s"] = elapsed
        metrics["auto_aberrations_updated"] = True
        messages.append("Auto Aberrations updated")
        return cfg

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
