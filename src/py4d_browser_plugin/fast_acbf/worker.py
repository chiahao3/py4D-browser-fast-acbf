"""Background execution for fast-acbf plugin jobs."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .config import FastAcbfConfig
from .utils import (
    apply_config_to_solver,
    build_solver,
    choose_device,
    sync_config_from_solver,
    tensor_to_numpy,
)


def evaluate_metric(image: np.ndarray, metric: str) -> float:
    import torch
    from fast_acbf.optimization.metrics import QualityMetrics

    with torch.no_grad():
        score = QualityMetrics.evaluate(torch.as_tensor(image), metric=metric)
    return float(score.detach().cpu().item())


@dataclass
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
        command: str,
        data,
        config: FastAcbfConfig,
        state: FastAcbfJobState,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.command = command
        self.data = data
        self.config = config.copy()
        self.state = state

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
            solver.update_dataset(data)
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
            solver = self._get_solver()
            cfg = self.config
            display_mode = cfg.mode
            output_frame = cfg.output_frame
            reconstruct_kwargs = cfg.reconstruct_kwargs()

            if self.command == "refine_defocus":
                self.message.emit("Refining defocus...")
                solver.refine_defocus(
                    num_points=int(cfg.defocus_points),
                    metric=cfg.metric,
                    plot_search=False,
                    mode=cfg.refinement_mode,
                    **reconstruct_kwargs,
                )
            elif self.command == "refine_flips":
                self.message.emit("Refining flips...")
                solver.refine_flips(
                    metric=cfg.metric,
                    plot_search=False,
                    mode=cfg.refinement_mode,
                    **reconstruct_kwargs,
                )
            elif self.command == "refine_scan_rotation":
                self.message.emit("Refining scan rotation...")
                solver.refine_scan_rotation(
                    num_points=int(cfg.rotation_points),
                    metric=cfg.metric,
                    plot_search=False,
                    mode=cfg.refinement_mode,
                    **reconstruct_kwargs,
                )
            elif self.command == "refine_orientation":
                self.message.emit("Refining flips and scan rotation...")
                solver.refine_flips(
                    metric=cfg.metric,
                    plot_search=False,
                    mode=cfg.refinement_mode,
                    **reconstruct_kwargs,
                )
                solver.refine_scan_rotation(
                    num_points=int(cfg.rotation_points),
                    metric=cfg.metric,
                    plot_search=False,
                    mode=cfg.refinement_mode,
                    **reconstruct_kwargs,
                )
            elif self.command == "refine_aberrations":
                self.message.emit("Refining aberrations...")
                solver.refine_aberrations(
                    lr=float(cfg.aberration_lr),
                    iters=int(cfg.aberration_iters),
                    metric=cfg.metric,
                    mode=cfg.refinement_mode,
                    **reconstruct_kwargs,
                )
            elif self.command == "auto_tune":
                self.message.emit("Refining all fast-acbf parameters...")
                solver.refine_all_params(
                    metric=cfg.metric,
                    mode=cfg.refinement_mode,
                    rotation_num_points=int(cfg.rotation_points),
                    defocus_num_points=int(cfg.defocus_points),
                    aberration_lr=float(cfg.aberration_lr),
                    aberration_iters=int(cfg.aberration_iters),
                    **reconstruct_kwargs,
                )

            image = self._reconstruct(solver, display_mode)
            probe = tensor_to_numpy(solver.get_probe(frame=output_frame).abs())
            metric_value = evaluate_metric(image, cfg.metric)
            updated_config = sync_config_from_solver(cfg, solver)

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
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
