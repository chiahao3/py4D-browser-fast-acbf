"""Background execution for fast-acbf plugin jobs."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .config import FastAcbfConfig, LABEL_TO_STATE_KEY


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
    if mps is not None and mps():
        return "mps"
    return "cpu"


def tensor_to_numpy(value) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    return np.asarray(value, dtype=np.float32)


def evaluate_metric(image: np.ndarray, metric: str) -> float:
    import torch
    from fast_acbf.optimization.metrics import QualityMetrics

    with torch.no_grad():
        score = QualityMetrics.evaluate(torch.as_tensor(image), metric=metric)
    return float(score.detach().cpu().item())


def apply_config_to_solver(solver, config: FastAcbfConfig) -> None:
    solver.apply_metadata(
        {
            "flipud": bool(config.flipud),
            "fliplr": bool(config.fliplr),
            "transpose": bool(config.transpose),
            "rotation_deg": float(config.rotation_deg),
        }
    )
    for label, value in config.aberrations.items():
        state_key = LABEL_TO_STATE_KEY.get(label)
        if state_key is None:
            continue
        if state_key in solver.ab_state.coeffs:
            solver.ab_state.set_physical(state_key, float(value))
    solver.clear_basis_cache()


def sync_config_from_solver(config: FastAcbfConfig, solver) -> FastAcbfConfig:
    cfg = config.copy()
    cfg.rotation_deg = float(solver.rotation_deg)
    cfg.flipud = bool(solver.coord_transform.get("flipud", False))
    cfg.fliplr = bool(solver.coord_transform.get("fliplr", False))
    cfg.transpose = bool(solver.coord_transform.get("transpose", False))
    for label, state_key in LABEL_TO_STATE_KEY.items():
        if state_key in solver.ab_state.coeffs:
            cfg.aberrations[label] = float(solver.ab_state.get_physical(state_key))
    return cfg


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
        from fast_acbf.solver import BFSolver

        cfg = self.config
        runtime_device = choose_device(cfg.device)
        signature_cfg = cfg.copy()
        signature_cfg.device = runtime_device
        data = np.ascontiguousarray(np.asarray(self.data, dtype=np.float32))
        signature = signature_cfg.solver_signature(self.data)

        solver = self.state.solver
        if solver is None or self.state.signature != signature:
            self.message.emit(f"Building fast-acbf solver on {runtime_device}...")
            solver = BFSolver(
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
            probe = tensor_to_numpy(solver.get_probe(frame="detector").abs())
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
