import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QAction, QActionGroup, QApplication, QMainWindow, QMenu

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.live_view.dock import LiveViewDock
from py4d_browser_plugin.fast_acbf.live_view.output import compute_live_view_outputs
from py4d_browser_plugin.fast_acbf.live_view.worker import LiveViewWorker
from py4d_browser_plugin.fast_acbf.plugin import FastAcbfPlugin


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    return _APP


class _WritableCal:
    def __init__(self, r_size=1.0, q_size=0.01, voltage=300.0):
        self.r_size = r_size
        self.r_units = "A"
        self.q_size = q_size
        self.q_units = "A^-1"
        self.values = {"voltage": voltage}

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def get_R_pixel_size(self):
        return self.r_size

    def set_R_pixel_size(self, value):
        self.r_size = value

    def get_R_pixel_units(self):
        return self.r_units

    def set_R_pixel_units(self, value):
        self.r_units = value

    def get_Q_pixel_size(self):
        return self.q_size

    def set_Q_pixel_size(self, value):
        self.q_size = value

    def get_Q_pixel_units(self):
        return self.q_units

    def set_Q_pixel_units(self, value):
        self.q_units = value


class _SignalParent(QMainWindow):
    signal_datacube_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.datacube = SimpleNamespace(
            data=np.ones((2, 2, 4, 4), dtype=np.float32),
            calibration=_WritableCal(),
        )
        self.registered = None
        self.restored = 0
        self.virtual_images = []
        self.result_images = []
        self.result_scaling_group = QActionGroup(self)
        self.result_scaling_group.setExclusive(True)
        self.result_scale_linear_action = QAction("Linear", self)
        self.result_scale_linear_action.setCheckable(True)
        self.result_scale_log_action = QAction("Log", self)
        self.result_scale_log_action.setCheckable(True)
        self.result_scale_log_action.setChecked(True)
        self.result_scaling_group.addAction(self.result_scale_linear_action)
        self.result_scaling_group.addAction(self.result_scale_log_action)

    def register_result_callback(self, title, cleanup, **callbacks):
        self.registered = {"title": title, "cleanup": cleanup, "callbacks": callbacks}

    def set_internal_result_callback(self):
        self.restored += 1

    def set_virtual_image(self, image, reset=False):
        self.virtual_images.append((np.asarray(image), reset))

    def set_result_image(self, image, reset=False, pixel_size=1.0, pixel_units="", title=""):
        self.result_images.append((np.asarray(image), reset, pixel_size, pixel_units, title))


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def abs(self):
        return _FakeTensor(np.abs(self.array))

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)


class _FakeSolver:
    def __init__(self):
        self.calls = []

    def get_reconstructed_image(self, **kwargs):
        self.calls.append(("recon", kwargs["mode"]))
        fill = 1.0 if kwargs["mode"] == "tcBF" else 2.0
        return _FakeTensor(np.full((2, 2), fill, dtype=np.float32))

    def get_probe(self, **kwargs):
        self.calls.append(("probe", kwargs["frame"]))
        return _FakeTensor(np.array([[1 + 1j, 0], [0, -2j]], dtype=np.complex64))

    def get_chi_surface(self, **kwargs):
        self.calls.append(("chi", kwargs["frame"]))
        return _FakeTensor(np.array([[0.0, np.pi], [3 * np.pi, -3 * np.pi]], dtype=np.float32))


def test_live_view_output_computation_deduplicates_and_wraps_chi():
    cfg = FastAcbfConfig(live_virtual_output="chi", live_result_output="chi")
    solver = _FakeSolver()

    outputs = compute_live_view_outputs(solver, cfg)

    assert outputs.routes == {"virtual": "chi", "result": "chi"}
    assert list(outputs.images) == ["chi"]
    assert len([call for call in solver.calls if call[0] == "chi"]) == 1
    assert np.all(outputs.images["chi"] <= np.pi)
    assert np.all(outputs.images["chi"] >= -np.pi)

    cfg.live_virtual_output = "probe"
    cfg.live_result_output = "tcBF"
    outputs = compute_live_view_outputs(solver, cfg)
    assert outputs.images["probe"].dtype == np.float32
    assert outputs.images["tcBF"].shape == (2, 2)


def test_live_view_worker_drops_stale_queued_frames():
    worker = LiveViewWorker()
    first = np.zeros((1, 1, 2, 2), dtype=np.float32)
    second = np.ones((1, 1, 2, 2), dtype=np.float32)

    worker.submit(first, FastAcbfConfig())
    worker.submit(second, FastAcbfConfig())

    dataset, _config = worker._queue.get_nowait()
    assert np.array_equal(dataset, second)


def test_live_view_worker_rebuild_policy_ignores_output_only_changes(monkeypatch):
    import py4d_browser_plugin.fast_acbf.live_view.worker as worker_module

    class _Wrapped:
        def __init__(self, solver):
            self.solver = solver
            self.datasets = []

        def update_dataset(self, data):
            self.datasets.append(np.asarray(data).shape)

        def get_reconstructed_image(self, **kwargs):
            return _FakeTensor(np.ones((2, 2), dtype=np.float32))

    monkeypatch.setattr(worker_module, "choose_device", lambda _device: "cpu")
    monkeypatch.setattr(worker_module, "build_solver", lambda cfg, data, dev: object())
    monkeypatch.setattr(worker_module, "LiveBFSolver", _Wrapped)
    monkeypatch.setattr(
        worker_module,
        "compute_live_view_outputs",
        lambda solver, cfg: SimpleNamespace(
            images={"tcBF": np.ones((2, 2), dtype=np.float32)},
            routes={"virtual": "None", "result": "tcBF"},
        ),
    )

    worker = LiveViewWorker()
    cfg = FastAcbfConfig(device="cpu")
    data = np.ones((2, 2, 4, 4), dtype=np.float32)

    first = worker._process_one(data, cfg)
    cfg.live_result_output = "probe"
    second = worker._process_one(data, cfg)
    third = worker._process_one(np.ones((3, 2, 4, 4), dtype=np.float32), cfg)

    assert first["reset"] is True
    assert second["reset"] is False
    assert third["reset"] is True
    assert worker.rebuild_count == 2


def test_live_view_plugin_registers_callback_and_routes_payload(monkeypatch):
    _app()
    parent = _SignalParent()
    plugin = FastAcbfPlugin(parent, QMenu(parent))

    class _Session:
        def __init__(self, parent=None):
            self.frame_ready = SimpleNamespace(connect=lambda cb: setattr(self, "frame_cb", cb))
            self.started_ready = SimpleNamespace(connect=lambda cb: setattr(self, "started_cb", cb))
            self.error = SimpleNamespace(connect=lambda cb: setattr(self, "error_cb", cb))
            self.finished = SimpleNamespace(connect=lambda cb: setattr(self, "finished_cb", cb))
            self.submissions = []
            self.started = False

        def start(self):
            self.started = True

        def submit(self, data, config):
            self.submissions.append((data, config.copy()))

        def stop(self, timeout_ms=2000):
            self.stopped = True

    monkeypatch.setattr("py4d_browser_plugin.fast_acbf.plugin.LiveViewSession", _Session)
    plugin.live_view_action.setChecked(True)

    assert plugin.live_view_dock is not None
    assert plugin.live_view_session is None
    plugin.live_view_dock.start_btn.click()

    assert parent.registered["title"] == "fast-acbf Live View"
    assert "callback_datacube_changed" in parent.registered["callbacks"]
    assert plugin.live_view_session.started is True
    assert len(plugin.live_view_session.submissions) == 1

    parent.datacube.data = np.zeros((2, 2, 4, 4), dtype=np.float32)
    parent.registered["callbacks"]["callback_datacube_changed"]()
    assert len(plugin.live_view_session.submissions) == 2

    plugin._live_view_frame_ready(
        {
            "config": FastAcbfConfig(live_virtual_output="tcBF", live_result_output="tcBF"),
            "routes": {"virtual": "tcBF", "result": "tcBF"},
            "outputs": {"tcBF": np.ones((2, 2), dtype=np.float32)},
            "metrics": {
                "device": "cpu",
                "fps": 4.0,
                "latency_s": 0.25,
                "c10_angstrom": 12.0,
                "max_alpha_mrad": 31.0,
            },
            "reset": True,
        }
    )

    assert parent.virtual_images[-1][1] is True
    assert parent.result_images[-1][4] == "fast-acbf Live View tcBF"
    assert parent.result_scale_linear_action.isChecked() is True
    assert "C10(-df): 12 Ang" in plugin.live_view_dock.c10_label.text()
    assert plugin.live_view_dock.alpha_label.text() == "max alpha: 31 mrad"

    before = len(parent.virtual_images)
    parent.registered["callbacks"]["callback_datacube_changed"]()
    assert len(parent.virtual_images) == before + 1
    assert parent.virtual_images[-1][1] is False

    plugin.live_view_dock.stop_btn.click()
    assert plugin.live_view_session is None
    assert plugin.live_view_dock is not None
    assert parent.restored == 1
    assert plugin.live_view_dock.start_btn.isEnabled() is True

    plugin.live_view_action.setChecked(False)
    assert plugin.live_view_dock is None


def test_live_view_dock_configuration_button_opens_plugin_config(monkeypatch):
    _app()
    parent = _SignalParent()
    plugin = FastAcbfPlugin(parent, QMenu(parent))
    opened = []
    monkeypatch.setattr(plugin, "launch_config", lambda: opened.append("config"))

    plugin.live_view_action.setChecked(True)
    plugin.live_view_dock.configure_btn.click()

    assert opened == ["config"]
    plugin.live_view_action.setChecked(False)


def test_live_view_uses_accepted_config_without_re_resolving(monkeypatch):
    _app()
    parent = _SignalParent()
    plugin = FastAcbfPlugin(parent, QMenu(parent))
    plugin.live_view_session = SimpleNamespace(submit=lambda data, config: submissions.append(config.copy()))
    plugin.live_view_dock = SimpleNamespace(
        set_config=lambda config: None,
        set_status=lambda message: None,
    )
    submissions = []
    accepted = FastAcbfConfig(
        use_calibration=False,
        scan_step_angstrom=7.0,
        dk_inv_angstrom=0.25,
        voltage_kv=80.0,
        wavelength_angstrom=0.0418,
    )

    plugin.config = accepted.copy()
    plugin._update_live_view_config(config=accepted)

    assert submissions[-1].scan_step_angstrom == 7.0
    assert submissions[-1].dk_inv_angstrom == 0.25
    assert submissions[-1].voltage_kv == 80.0
    assert submissions[-1].wavelength_angstrom == 0.0418


def test_live_view_config_accept_syncs_py4d_calibration_and_resubmits(monkeypatch):
    _app()
    parent = _SignalParent()
    plugin = FastAcbfPlugin(parent, QMenu(parent))
    plugin.live_view_session = SimpleNamespace(submit=lambda data, config: submissions.append(config.copy()))
    plugin.live_view_dock = SimpleNamespace(
        set_config=lambda config: None,
        set_status=lambda message: None,
    )
    submissions = []
    accepted = FastAcbfConfig(
        use_calibration=True,
        use_detector_alpha=False,
        scan_step_angstrom=7.0,
        dk_inv_angstrom=0.25,
        voltage_kv=80.0,
        wavelength_angstrom=0.0418,
    )
    seen_configs = []

    class _Dialog:
        Accepted = 1

        def __init__(self, config, parent=None):
            seen_configs.append(config.copy())
            self.config = accepted.copy()
            self.request_calibration = SimpleNamespace(connect=lambda callback: None)

        def exec_(self):
            return self.Accepted

    monkeypatch.setattr("py4d_browser_plugin.fast_acbf.plugin.ConfigurationDialog", _Dialog)

    plugin.launch_config()

    cal = parent.datacube.calibration
    assert cal.get_R_pixel_size() == 7.0
    assert cal.get_R_pixel_units() == "A"
    assert cal.get_Q_pixel_size() == 0.25
    assert cal.get_Q_pixel_units() == "A^-1"
    assert cal["voltage"] == 80.0
    assert plugin.config.scan_step_angstrom == 7.0
    assert plugin.config.dk_inv_angstrom == 0.25
    assert plugin.config.voltage_kv == 80.0
    assert submissions[-1].scan_step_angstrom == 7.0
    assert submissions[-1].dk_inv_angstrom == 0.25

    plugin.launch_config()

    assert seen_configs[-1].scan_step_angstrom == 7.0
    assert seen_configs[-1].dk_inv_angstrom == 0.25
    assert seen_configs[-1].voltage_kv == 80.0


def test_live_view_refresh_calibration_resubmits_active_session(monkeypatch):
    _app()
    parent = _SignalParent()
    plugin = FastAcbfPlugin(parent, QMenu(parent))
    plugin.live_view_session = SimpleNamespace(submit=lambda data, config: submissions.append(config.copy()))
    plugin.live_view_dock = SimpleNamespace(
        set_config=lambda config: None,
        set_status=lambda message: None,
    )
    submissions = []

    plugin._refresh_calibration_display()

    assert submissions


def test_live_view_dock_renders_c10_label():
    _app()
    dock = LiveViewDock(FastAcbfConfig(aberrations={"C10": -25.0}, max_alpha_mrad=42.0))
    assert dock.allowedAreas() & Qt.TopDockWidgetArea
    assert dock.c10_label.text() == "C10(-df): -25 Ang"
    assert dock.alpha_label.text() == "max alpha: 42 mrad"
    assert dock.start_btn.isEnabled() is True
    assert dock.stop_btn.isEnabled() is False
    dock.close()
