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


def normalize_dataset_by_pacbed_max(data: np.ndarray) -> np.ndarray:
    """TEMPORARY plugin-side dataset normalization (in place).

    Divides a 4D (Ny, Nx, Ky, Kx) float array by the maximum of its
    scan-averaged diffraction pattern (PACBED), so the BF disk peaks near 1.

    Why this lives here, not in the solver:
        BFSolver does not yet apply intensity normalization at init. Without
        it, reconstruction sensitivity (gradients, refinement step sizes,
        metrics) depends on absolute counts, which makes results dataset-
        dependent in a way we want to remove. This helper is the
        plugin-side workaround until normalization is absorbed into
        ``BFSolver.__init__`` as an opt-in flag — at which point this
        function and its callers should be deleted.

    Contract:
        Mutates ``data`` in place. Callers MUST pass a private copy; never
        hand in ``parent.datacube.data`` directly (use
        :func:`prepare_solver_dataset` instead, which guarantees a copy).
    """
    mean_dp = data.mean(axis=(0, 1))
    scale = float(mean_dp.max())
    if scale > 0:
        data /= scale
    return data


def prepare_solver_dataset(data: np.ndarray) -> np.ndarray:
    """Return a private, normalized float32 C-contiguous copy of ``data``.

    Single entry point for any plugin path that hands a dataset to the
    solver (initial build OR cache-hit ``update_dataset``). Going through
    one helper keeps the build path and the update path on the same
    intensity scale, which avoids the "histogram jumps between raw and
    normalized" bug we hit when only the build path was normalized.

    Cost (typical 1 GB datacube):
        ~50–100 ms total — one ``np.array(copy=True)`` (memcpy bounded by
        RAM bandwidth, ~30–60 ms) plus the PACBED reduce-and-divide
        (~10–30 ms). Peak RAM transiently +1× the dataset size while the
        copy is alive. Negligible against the reconstruction itself, but
        worth knowing this runs on every interactive reconstruction click
        because we do NOT cache the prepared array (see commit history /
        discussion 2026-05-15 for why caching was deferred to the
        solver-side refactor).

    Status: TEMPORARY. Remove together with
    :func:`normalize_dataset_by_pacbed_max` when ``BFSolver`` learns to
    normalize at init.
    """
    out = np.array(data, dtype=np.float32, order="C", copy=True)
    normalize_dataset_by_pacbed_max(out)
    return out


def build_solver(config: FastAcbfConfig, data: np.ndarray, runtime_device: str):
    from fast_acbf.solver import BFSolver

    # TEMPORARY: normalize on the way in. The copy is also the safety
    # barrier that lets us run the in-place divide without ever touching
    # the host's ``parent.datacube.data``. Remove once BFSolver normalizes
    # at init; pass ``data`` straight through then.
    data = prepare_solver_dataset(data)
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
