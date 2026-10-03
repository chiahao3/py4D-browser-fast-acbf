"""Background execution for fast-acbf plugin jobs."""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .config import FastAcbfConfig
from .solver_job import SolverJob
from .utils import (
    apply_config_to_solver,
    build_solver,
    choose_device,
    sync_config_from_solver,
    tensor_to_complex,
    tensor_to_numpy,
)


def optics_diagnostics(solver, frame: str, upscale: float) -> dict:
    """Probe (complex), aberration surface and vBF shift vectors for the dashboard.

    - ``probe_complex``: real-space probe (``get_probe``), pixel ``probe_pixel_size`` Å
      (``1 / (N_probe * dk)``; upscaling refines it)
    - ``chi``: aberration surface in rad on the raw detector grid, NaN outside the
      bright-field disk; pixel ``dk`` Å⁻¹
    - ``shifts_yx``: image shift of each bright-field pixel's vBF image in Å, ``(Nb, 2)``
      as (y, x) in ``frame``; ``bf_rows`` / ``bf_cols``: that pixel on the detector grid
    """
    import torch

    with torch.no_grad():
        probe = tensor_to_complex(solver.get_probe(frame=frame, upscale=upscale))
        mask = tensor_to_numpy(solver.bf_mask) > 0
        chi = tensor_to_numpy(solver.get_chi_surface(frame=frame))
        shifts = tensor_to_numpy(solver.get_yx_shifts_ang(frame=frame))
    chi = np.where(mask, chi, np.nan).astype(np.float32)
    rows, cols = np.nonzero(mask)
    dk = float(solver.dk)
    return {"probe_complex": probe, "probe_pixel_size": 1.0 / (probe.shape[-1] * dk),
            "chi": chi, "chi_pixel_size": dk, "shifts_yx": shifts, "bf_rows": rows,
            "bf_cols": cols}


def depth_stack(solver, cfg: FastAcbfConfig, n_slices: int, step: float) -> dict:
    """Reconstructions, probes, χ and vBF shifts at ``n_slices`` C10 values ``step`` Å
    apart around the current C10 (the solver's C10 is restored afterwards).

    ``chi_stack`` ``(n, Ky, Kx)`` and ``shifts_stack`` ``(n, Nb, 2)`` are laid out like
    :func:`optics_diagnostics`' ``chi`` and ``shifts_yx``."""
    import torch

    stack = tensor_to_numpy(solver.get_defocus_stack(
        mode=cfg.mode, frame=cfg.output_frame, n_layers=int(n_slices),
        slice_thickness=float(step), **cfg.reconstruct_kwargs()))
    axis = tensor_to_numpy(solver.last_c10_stack_axis).astype(np.float64)
    c10 = float(solver.ab_state.get_physical("C_1_0"))
    probes, chis, shifts = [], [], []
    try:
        for value in axis:
            with torch.no_grad():
                solver.ab_state.set_physical("C_1_0", float(value))
            optics = optics_diagnostics(solver, cfg.output_frame, cfg.upscale)
            probes.append(optics["probe_complex"])
            chis.append(optics["chi"])
            shifts.append(optics["shifts_yx"])
    finally:
        with torch.no_grad():
            solver.ab_state.set_physical("C_1_0", c10)
    return {"stack": stack, "stack_c10": axis, "probe_stack": np.stack(probes),
            "chi_stack": np.stack(chis), "shifts_stack": np.stack(shifts),
            "stack_step": float(step)}


def evaluate_metric(image: np.ndarray, metric: str) -> float:
    import torch
    from fast_acbf.optimization.metrics import QualityMetrics

    with torch.no_grad():
        score = QualityMetrics.evaluate(torch.as_tensor(image), metric=metric)
    return float(score.detach().cpu().item())


class FastAcbfJobState:
    solver: Any = None
    signature: tuple | None = None


class FastAcbfRunner(QThread):
    message = pyqtSignal(str)
    finished_result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        job: SolverJob,
        data,
        config: FastAcbfConfig,
        state: FastAcbfJobState,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.job = job
        self.data = data
        self.config = config.copy()
        self.state = state

    @property
    def command(self) -> str:
        return self.job.command

    def _get_solver(self):
        cfg = self.config
        runtime_device = choose_device(cfg.device)
        signature_cfg = cfg.copy()
        signature_cfg.device = runtime_device
        data = np.ascontiguousarray(np.asarray(self.data, dtype=np.float32))
        signature = signature_cfg.solver_signature(self.data)

        solver = self.state.solver
        if solver is None or self.state.signature != signature:
            self.message.emit(f"Building fast-acbf solver on {runtime_device}...")
            solver = build_solver(cfg, data, runtime_device)
            self.state.solver = solver
            self.state.signature = signature
        else:
            self.message.emit("Reusing cached fast-acbf solver...")
            apply_config_to_solver(solver, cfg)
        return solver

    def _reconstruct(self, solver, mode: str) -> np.ndarray:
        cfg = self.config
        self.message.emit(f"Reconstructing {mode}...")
        kwargs = cfg.reconstruct_kwargs()
        if mode.lower() == "tcbf":
            image = solver.get_tcBF(frame=cfg.output_frame, **kwargs)
        else:
            image = solver.get_acBF(frame=cfg.output_frame, **kwargs)
        return tensor_to_numpy(image)

    def run(self) -> None:
        try:
            self.config.validate_upscale_settings()
            solver = self._get_solver()
            cfg = self.config
            display_mode = cfg.mode
            output_frame = cfg.output_frame

            self.job.execute(solver, cfg, self.message.emit)

            image = self._reconstruct(solver, display_mode)
            extras = optics_diagnostics(solver, output_frame, cfg.upscale)
            probe = np.abs(extras["probe_complex"])
            if self.job.command == "depth_stack":
                n, step = int(self.job.n_slices), float(self.job.step)
                self.message.emit(f"Reconstructing a depth stack: {n} slices, {step:g} Å apart...")
                extras.update(depth_stack(solver, cfg, n, step))
            metric_value = evaluate_metric(image, cfg.metric)
            updated_config = sync_config_from_solver(cfg, solver)

            if self.job.command in ("refine_defocus", "simple_menu_reconstruct"):
                self.message.emit(f"Optimal C10 found at {solver.ab_state.get_physical('C_1_0'):.5g} Å")
            elif self.job.command == "depth_stack":
                self.message.emit(f"Depth stack ready ({len(extras['stack'])} slices)")
            elif self.job.command == "manual":
                self.message.emit(f"Reconstructed at C10 = {solver.ab_state.get_physical('C_1_0'):.5g} Å")
            else:
                self.message.emit("Done")

            self.finished_result.emit(
                {
                    "solver": solver,
                    "signature": self.state.signature,
                    "config": updated_config,
                    "image": image,
                    "probe": probe,
                    "mode": display_mode,
                    "device": solver.device,
                    "metric_value": metric_value,
                    "metric_text": f"{metric_value:.5g}",
                    "command": self.command,
                    **extras,
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
