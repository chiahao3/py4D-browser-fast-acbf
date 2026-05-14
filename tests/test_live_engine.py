"""Tests for py4d_browser_plugin.fast_acbf.live_engine.

The engine is the non-Qt core of the live-acquisition path. These tests
mirror the integration coverage in ``test_live_worker.py`` but assert the
public surface lives at ``live_engine`` and that the module is importable
without PyQt5 at module load time.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from py4d_browser_plugin.fast_acbf.calibration import electron_wavelength_angstrom
from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.live_engine import (
    FrameMetrics,
    LiveSolverEngine,
    _cuda_sync,
)


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


def test_live_engine_processes_one_frame(synthetic_data, base_metadata):
    cfg = _cfg(base_metadata)
    engine = LiveSolverEngine(cfg, synthetic_data, initial_metadata=base_metadata)
    image, metrics = engine.process_one(synthetic_data, base_metadata)
    assert image.shape == synthetic_data.shape[:2]
    assert isinstance(metrics, FrameMetrics)
    assert metrics.device == "cpu"
    assert metrics.mode == "tcBF"


def test_cuda_sync_is_noop_on_cpu():
    # Should not raise even if torch.cuda is not available.
    _cuda_sync("cpu")


def test_live_engine_imports_without_pyqt5_installed():
    """Engine module must import in an environment where PyQt5 is absent.

    The plugin's ``__init__.py`` already guards its eager import of
    ``plugin.FastAcbfPlugin`` against a missing PyQt5, so a headless
    environment can ``import py4d_browser_plugin.fast_acbf.live_engine``
    even without PyQt5 installed. Simulate that by installing a
    ``meta_path`` finder that blocks PyQt5 before the first import.
    """
    code = textwrap.dedent(
        """
        import sys

        class _BlockPyQt5:
            def find_spec(self, name, path=None, target=None):
                if name == "PyQt5" or name.startswith("PyQt5."):
                    raise ModuleNotFoundError(
                        f"No module named {name!r}", name=name
                    )
                return None

        sys.meta_path.insert(0, _BlockPyQt5())

        from py4d_browser_plugin.fast_acbf.live_engine import (
            FrameMetrics, LiveSolverEngine,
        )

        assert LiveSolverEngine is not None
        assert FrameMetrics is not None

        pyqt = sorted(m for m in sys.modules if m.startswith("PyQt"))
        if pyqt:
            print("LEAKED:" + ",".join(pyqt))
            sys.exit(1)
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"live_engine could not import with PyQt5 absent. "
        f"stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
    assert "clean" in result.stdout


def test_live_worker_still_re_exports_engine_for_backwards_compat():
    """live_worker.LiveSolverEngine should still resolve to the same class.

    External callers (scripts, downstream code) may still import from the
    old location; we re-export to avoid breaking them while signalling
    that live_engine is the canonical home.
    """
    from py4d_browser_plugin.fast_acbf import live_engine, live_worker

    assert live_worker.LiveSolverEngine is live_engine.LiveSolverEngine
    assert live_worker.FrameMetrics is live_engine.FrameMetrics
