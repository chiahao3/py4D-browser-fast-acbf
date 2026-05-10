import os
import sys
import time

import numpy as np
from PyQt5.QtWidgets import QApplication

from py4d_browser_plugin.fast_acbf.config import (
    FastAcbfConfig,
    electron_wavelength_angstrom,
)
from py4d_browser_plugin.fast_acbf.live_controller import (
    create_live_session,
    jitter_from_options,
    metadata_from_config,
    stop_live,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    return _APP


def _wait_until(predicate, timeout_s=10.0):
    app = _app()
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _config() -> FastAcbfConfig:
    wavelength = electron_wavelength_angstrom(200.0)
    return FastAcbfConfig(
        mode="tcBF",
        device="cpu",
        cache_mode="lazy",
        max_alpha_mrad=25.0,
        scan_step_angstrom=0.2,
        dk_inv_angstrom=0.05,
        wavelength_angstrom=wavelength,
        voltage_kv=200.0,
        rotation_deg=3.0,
        use_calibration=False,
        use_detector_alpha=False,
    )


def test_live_metadata_and_jitter_helpers():
    data = np.zeros((6, 7, 12, 12), dtype=np.float32)
    cfg = _config()

    metadata = metadata_from_config(cfg, data)
    assert metadata["scan_shape"] == (6, 7)
    assert metadata["scan_step_size"] == 0.2
    assert metadata["rotation_deg"] == 3.0
    assert metadata["flipud"] is False

    jitter = jitter_from_options(
        {"jitter_rotation_deg": 0.5, "jitter_scan_step_angstrom": 0.01}
    )
    assert jitter == {"rotation_deg": 0.5, "scan_step_size": 0.01}


def test_live_session_processes_finite_mock_stream():
    _app()
    rng = np.random.default_rng(0)
    data = rng.random((6, 6, 12, 12), dtype=np.float32)
    frames = []
    errors = []
    finished = []
    session = create_live_session(
        config=_config(),
        datacube_data=data,
        options={"n_frames": 3, "jitter_rotation_deg": 0.1},
        frame_interval_ms=100,
    )
    session.worker.frame_ready.connect(lambda image, metrics: frames.append((image, metrics)))
    session.error.connect(errors.append)
    session.finished.connect(lambda: finished.append(True))

    session.start()
    try:
        assert _wait_until(lambda: finished or errors, timeout_s=10.0)
        assert errors == []
        assert len(frames) == 3
        assert all(image.shape == data.shape[:2] for image, _metrics in frames)
        assert all(metrics["latency_s"] > 0 for _image, metrics in frames)
    finally:
        stop_live(session)
