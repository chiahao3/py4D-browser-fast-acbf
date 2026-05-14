"""Shared helpers for constructing and synchronizing fast-acbf solvers."""

from __future__ import annotations

from typing import Any

import numpy as np

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


def tensor_to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    return np.asarray(value, dtype=np.float32)


def build_solver(config: FastAcbfConfig, data: np.ndarray, runtime_device: str):
    from fast_acbf.solver import BFSolver

    return BFSolver(
        dataset=data,
        max_alpha=float(config.max_alpha_mrad),
        scan_step_size=float(config.scan_step_angstrom),
        dk=float(config.dk_inv_angstrom),
        wavelength=float(config.wavelength_angstrom),
        max_order=int(config.max_order),
        aberrations=config.aberration_dict(),
        device=runtime_device,
        coord_transform=config.coord_transform(),
        eps=float(config.eps),
        cache_mode=str(config.cache_mode),
    )


def apply_config_to_solver(solver: Any, config: FastAcbfConfig) -> None:
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


def sync_config_from_solver(config: FastAcbfConfig, solver: Any) -> FastAcbfConfig:
    cfg = config.copy()
    cfg.rotation_deg = float(solver.rotation_deg)
    cfg.flipud = bool(solver.coord_transform.get("flipud", False))
    cfg.fliplr = bool(solver.coord_transform.get("fliplr", False))
    cfg.transpose = bool(solver.coord_transform.get("transpose", False))
    for label, state_key in LABEL_TO_STATE_KEY.items():
        if state_key in solver.ab_state.coeffs:
            cfg.aberrations[label] = float(solver.ab_state.get_physical(state_key))
    return cfg
