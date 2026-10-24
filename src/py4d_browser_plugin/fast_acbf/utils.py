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


def tensor_to_complex(value: Any) -> np.ndarray:
    """Like :func:`tensor_to_numpy` but keeps complex values (complex64)."""
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    return np.asarray(value).astype(np.complex64)


def to_output_grid(a: np.ndarray, coord_transform: dict | None) -> np.ndarray:
    """A k-space (or probe) array of fast-acbf, on the *raw* detector grid, laid out on the
    output frame's own grid: PtyRAD's ``meas_flipT`` (flipud → fliplr → transpose of the
    last two axes), mirrored about the k = 0 pixel (index ``N // 2``; for even ``N``
    ``np.flip`` plus a one-pixel roll).

    fast-acbf applies the flags to the k value of each raw pixel, so ``get_chi_surface`` /
    ``get_probe`` are χ of the frame's coefficients at the flipped k, still on the raw
    grid. After this, pixel k holds χ at k: in the detector frame PtyRAD's χ and probe
    arrays, in the scan frame the scan-frame coefficients' (the detector frame's rotated
    by the scan rotation, like the image). Mirroring about k = 0 in k space is the same
    mirror about the centre pixel in real space, so this serves the real-space probe too."""
    ct = coord_transform or {}
    a = np.asarray(a)
    if ct.get("flipud"):
        a = np.roll(np.flip(a, -2), 1 - a.shape[-2] % 2, axis=-2)
    if ct.get("fliplr"):
        a = np.roll(np.flip(a, -1), 1 - a.shape[-1] % 2, axis=-1)
    if ct.get("transpose"):
        a = np.swapaxes(a, -2, -1)
    return np.ascontiguousarray(a)


def build_solver(config: FastAcbfConfig, data: np.ndarray, runtime_device: str):
    from fast_acbf.solver import BFSolver

    config.validate_upscale_settings()
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
        normalize=True,
        pipeline=str(config.pipeline),
        basis_mode=str(config.basis_mode),
        pad_width=config.normalized_pad_width(),
        upscale=float(config.upscale),
        upscale_method=str(config.upscale_method),
    )


def apply_config_to_solver(solver: Any, config: FastAcbfConfig) -> None:
    solver.set_flips(bool(config.flipud), bool(config.fliplr), bool(config.transpose))
    solver.set_rotation_deg(float(config.rotation_deg))
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
