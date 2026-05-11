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
    DEFAULT_GUI_FRAME_INTERVAL_MS,
    create_live_session,
    cyclic_sweep_from_options,
    metadata_from_config,
    linear_sweep_from_options,
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


def test_live_metadata_and_sweep_helpers():
    data = np.zeros((6, 7, 12, 12), dtype=np.float32)
    cfg = _config()

    metadata = metadata_from_config(cfg, data)
    assert metadata["scan_shape"] == (6, 7)
    assert metadata["scan_step_size"] == 0.2
    assert metadata["rotation_deg"] == 3.0
    assert metadata["flipud"] is False
    assert metadata["defocus_angstrom"] == 0.0

    linear = linear_sweep_from_options({"rotation_sweep_deg_per_frame": 0.5})
    assert linear == {"rotation_deg": 0.5}

    cyclic = cyclic_sweep_from_options(
        {"defocus_sweep_angstrom": 150.0, "defocus_sweep_period_frames": 80}
    )
    assert cyclic == {"defocus_angstrom": (150.0, 80.0)}


def test_live_session_defaults_to_paced_gui_stream():
    _app()
    data = np.zeros((6, 6, 12, 12), dtype=np.float32)
    session = create_live_session(
        config=_config(),
        datacube_data=data,
        options={"n_frames": 1},
    )
    try:
        assert session.producer.frame_interval_ms == DEFAULT_GUI_FRAME_INTERVAL_MS
    finally:
        stop_live(session)


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
        options={"n_frames": 3, "rotation_sweep_deg_per_frame": 0.1},
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
