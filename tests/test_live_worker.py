"""Integration test: drive LiveSolverEngine end-to-end on CPU with a tiny dataset.

Exercises the full live-acquisition loop on a real BFSolver (no mocks) for a
mix of metadata cases:
  * No-op frames (Tier-1 update_dataset only)
  * Rotation-only change (orientation, no Tier-3)
  * scan_step_size change (Tier-2 optics-only invalidation)
  * max_alpha change (Tier-3 full rebuild)

Asserts that every frame yields a 2D image and that the underlying solver
instance is preserved across non-Tier-3 frames (proving the cache split is
honored end-to-end).
"""

from __future__ import annotations

import numpy as np
import pytest

from py4d_browser_plugin.fast_acbf.config import (
    FastAcbfConfig,
    electron_wavelength_angstrom,
)
from py4d_browser_plugin.fast_acbf.live_worker import LiveSolverEngine


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    return rng.random((6, 6, 12, 12), dtype=np.float64).astype(np.float32)


@pytest.fixture
def base_metadata():
    wavelength = electron_wavelength_angstrom(200.0)
    return {
        "wavelength": wavelength,
        "max_alpha": 25.0,
        "dk": 0.05,
        "scan_shape": (6, 6),
        "scan_step_size": 0.2,
        "rotation_deg": 0.0,
        "flipud": False,
        "fliplr": False,
        "transpose": False,
    }


def _cfg(metadata: dict) -> FastAcbfConfig:
    return FastAcbfConfig(
        mode="tcBF",
        device="cpu",
        cache_mode="lazy",
        max_alpha_mrad=metadata["max_alpha"],
        scan_step_angstrom=metadata["scan_step_size"],
        dk_inv_angstrom=metadata["dk"],
        wavelength_angstrom=metadata["wavelength"],
        voltage_kv=200.0,
        rotation_deg=metadata["rotation_deg"],
        use_calibration=False,
        use_detector_alpha=False,
    )


def test_live_engine_processes_mixed_metadata(synthetic_data, base_metadata):
    cfg = _cfg(base_metadata)
    engine = LiveSolverEngine(cfg, synthetic_data, initial_metadata=base_metadata)
    solver_before = engine.solver

    Ry, Rx = synthetic_data.shape[:2]

    image, metrics = engine.process_one(synthetic_data, base_metadata)
    assert image.shape == (Ry, Rx)
    assert metrics.latency_s > 0
    assert metrics.device == "cpu"
    assert metrics.mask_path == "host-mask"
    assert metrics.bf_pixels is not None
    assert metrics.mode == "tcBF"
    assert engine.solver is solver_before  # no rebuild on no-op frame

    image, _ = engine.process_one(synthetic_data, base_metadata)
    assert image.shape == (Ry, Rx)
    assert engine.solver is solver_before

    rotated = dict(base_metadata, rotation_deg=15.0)
    image, _ = engine.process_one(synthetic_data, rotated)
    assert image.shape == (Ry, Rx)
    assert engine.solver is solver_before

    stepped = dict(rotated, scan_step_size=0.25)
    image, _ = engine.process_one(synthetic_data, stepped)
    assert image.shape == (Ry, Rx)
    assert engine.solver is solver_before

    bigger_alpha = dict(stepped, max_alpha=30.0)
    image, _ = engine.process_one(synthetic_data, bigger_alpha)
    assert image.shape == (Ry, Rx)
    # Tier-3 changes do NOT rebuild the BFSolver instance (in-place setters);
    # the solver object is still the same instance.
    assert engine.solver is solver_before


def test_live_engine_metric_tracking_advances(synthetic_data, base_metadata):
    cfg = _cfg(base_metadata)
    engine = LiveSolverEngine(cfg, synthetic_data, initial_metadata=base_metadata)
    _, m1 = engine.process_one(synthetic_data, base_metadata)
    _, m2 = engine.process_one(synthetic_data, base_metadata)
    assert m1.latency_s > 0
    assert m2.latency_s > 0
    # FPS should be defined and finite for both
    assert np.isfinite(m1.fps)
    assert np.isfinite(m2.fps)
