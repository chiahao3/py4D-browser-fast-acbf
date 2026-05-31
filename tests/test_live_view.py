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


class _FakeAbState:
    def __init__(self):
        self.coeffs = {"C_1_0": object(), "C_1_2_a": object()}
        self.values = {"C_1_0": 0.0, "C_1_2_a": 0.0}

    def set_physical(self, key, value):
        self.values[key] = float(value)

    def get_physical(self, key):
        return self.values[key]


class _FakeRefiningSolver:
    device = "cpu"

    def __init__(self, *, fail_focus=False, fail_aberrations=False):
        self.ab_state = _FakeAbState()
        self.rotation_deg = 0.0
        self.coord_transform = {}
        self.calls = []
        self.datasets = []
        self.fail_focus = fail_focus
        self.fail_aberrations = fail_aberrations

    def update_dataset(self, data):
        self.datasets.append(np.asarray(data).shape)

    def set_flips(self, flipud, fliplr, transpose):
        self.coord_transform.update(
            {"flipud": bool(flipud), "fliplr": bool(fliplr), "transpose": bool(transpose)}
        )

    def set_rotation_deg(self, value):
        self.rotation_deg = float(value)

    def clear_basis_cache(self):
        self.calls.append(("clear_basis_cache", None))

    def refine_defocus(self, **kwargs):
        self.calls.append(("focus", kwargs))
        if self.fail_focus:
            raise RuntimeError("focus boom")
        self.ab_state.set_physical("C_1_0", 11.0)

    def refine_aberrations(self, **kwargs):
        self.calls.append(("aberrations", kwargs))
        if self.fail_aberrations:
            raise RuntimeError("aberrations boom")
        self.ab_state.set_physical("C_1_0", 12.0)
        self.ab_state.set_physical("C_1_2_a", 3.0)

    def get_reconstructed_image(self, **kwargs):
        self.calls.append(("recon", kwargs["mode"]))
        return _FakeTensor(np.ones((2, 2), dtype=np.float32))


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


def test_live_view_worker_auto_refinement_runs_focus_then_aberrations(monkeypatch):
    import py4d_browser_plugin.fast_acbf.live_view.worker as worker_module

    solver = _FakeRefiningSolver()
    monkeypatch.setattr(worker_module, "choose_device", lambda _device: "cpu")
    monkeypatch.setattr(worker_module, "build_solver", lambda cfg, data, dev: solver)
    monkeypatch.setattr(worker_module, "LiveBFSolver", lambda wrapped: wrapped)
    monkeypatch.setattr(
        worker_module,
        "compute_live_view_outputs",
        lambda solver, cfg: SimpleNamespace(
            images={"tcBF": np.ones((2, 2), dtype=np.float32)},
            routes={"virtual": "None", "result": "tcBF"},
        ),
    )

    worker = LiveViewWorker()
    worker.set_auto_refinement(focus=True, aberrations=True)
    cfg = FastAcbfConfig(
        device="cpu",
        defocus_points=9,
        defocus_search_halfwidth_angstrom=15.0,
        defocus_range_tolerance_factor=10.0,
        aberration_lr=0.5,
        aberration_iters=3,
    )
    result = worker._process_one(np.ones((2, 2, 4, 4), dtype=np.float32), cfg)

    assert [call[0] for call in solver.calls if call[0] in {"focus", "aberrations"}] == [
        "focus",
        "aberrations",
    ]
    focus_kwargs = solver.calls[0][1]
    assert focus_kwargs["num_points"] == 9
    assert focus_kwargs["search_halfwidth"] == 15.0
    assert focus_kwargs["defocus_range_tolerance_factor"] == 10.0
    aberration_kwargs = solver.calls[1][1]
    assert aberration_kwargs["lr"] == 0.5
    assert aberration_kwargs["iters"] == 3
    assert result["config"].aberrations["C10"] == 12.0
    assert result["config"].aberrations["C12a"] == 3.0
    assert result["metrics"]["auto_focus_updated"] is True
    assert result["metrics"]["auto_aberrations_updated"] is True


def test_live_view_worker_auto_refinement_failure_disables_only_failing_mode(monkeypatch):
    import py4d_browser_plugin.fast_acbf.live_view.worker as worker_module

    solver = _FakeRefiningSolver(fail_focus=True)
    monkeypatch.setattr(worker_module, "choose_device", lambda _device: "cpu")
    monkeypatch.setattr(worker_module, "build_solver", lambda cfg, data, dev: solver)
    monkeypatch.setattr(worker_module, "LiveBFSolver", lambda wrapped: wrapped)
    monkeypatch.setattr(
        worker_module,
        "compute_live_view_outputs",
        lambda solver, cfg: SimpleNamespace(
            images={"tcBF": np.ones((2, 2), dtype=np.float32)},
            routes={"virtual": "None", "result": "tcBF"},
        ),
    )

    worker = LiveViewWorker()
    worker.set_auto_refinement(focus=True, aberrations=True)
    result = worker._process_one(
        np.ones((2, 2, 4, 4), dtype=np.float32),
        FastAcbfConfig(device="cpu"),
    )

    assert "auto_focus_error" in result["metrics"]
    assert result["metrics"]["auto_focus_enabled"] is False
    assert result["metrics"]["auto_aberrations_updated"] is True
    assert result["config"].aberrations["C10"] == 12.0


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
            self.auto_refinement = []
            self.started = False

        def start(self):
            self.started = True

        def submit(self, data, config):
            self.submissions.append((data, config.copy()))

        def set_auto_refinement(self, *, focus, aberrations):
            self.auto_refinement.append((focus, aberrations))

        def stop(self, timeout_ms=2000):
            self.stopped = True

    monkeypatch.setattr("py4d_browser_plugin.fast_acbf.plugin.LiveViewSession", _Session)
    plugin.live_view_action.setChecked(True)

    assert plugin.live_view_dock is not None
    assert plugin.live_view_session is None
    plugin.live_view_dock.auto_focus_cb.setChecked(True)
    plugin.live_view_dock.start_btn.click()

    assert parent.registered["title"] == "fast-acbf Live View"
    assert "callback_datacube_changed" in parent.registered["callbacks"]
    assert plugin.live_view_session.started is True
    assert len(plugin.live_view_session.submissions) == 1
    assert plugin.live_view_session.auto_refinement[-1] == (True, False)

    plugin.live_view_dock.auto_aberrations_cb.setChecked(True)
    assert plugin.live_view_session.auto_refinement[-1] == (True, True)

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
                "auto_refinement_messages": ["Auto Focus updated"],
            },
            "reset": True,
        }
    )

    assert parent.virtual_images[-1][1] is True
    assert parent.result_images[-1][4] == "fast-acbf Live View tcBF"
    assert parent.result_scale_linear_action.isChecked() is True
    assert "C10(-df): 12 Ang" in plugin.live_view_dock.c10_label.text()
    assert plugin.live_view_dock.alpha_label.text() == "max alpha: 31 mrad"
    assert plugin.live_view_dock.status_label.text() == "Auto Focus updated"

    plugin._live_view_frame_ready(
        {
            "config": FastAcbfConfig(live_virtual_output="tcBF", live_result_output="tcBF"),
            "routes": {"virtual": "tcBF", "result": "tcBF"},
            "outputs": {"tcBF": np.ones((2, 2), dtype=np.float32)},
            "metrics": {
                "auto_refinement_messages": ["Auto Focus failed: focus boom"],
                "auto_focus_error": "focus boom",
            },
            "reset": False,
        }
    )
    assert plugin.live_view_dock.auto_refinement_state() == (False, True)

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
    assert dock.auto_refinement_state() == (False, False)
    changes = []
    dock.auto_refinement_changed.connect(lambda focus, aberrations: changes.append((focus, aberrations)))
    dock.auto_focus_cb.setChecked(True)
    assert changes[-1] == (True, False)
    dock.set_auto_refinement_state(focus=False, aberrations=True)
    assert dock.auto_refinement_state() == (False, True)
    dock.close()
